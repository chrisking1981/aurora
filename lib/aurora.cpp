#include <aurora/aurora.h>

#ifdef AURORA_ENABLE_GX
#include "gfx/common.hpp"
#include "gfx/texture.hpp"
#include "gx/fifo.hpp"
#include "imgui.hpp"
#include "webgpu/gpu.hpp"
#include <webgpu/webgpu_cpp.h>
#endif

#ifdef AURORA_ENABLE_RMLUI
#include "rmlui.hpp"
#endif

#include "input.hpp"
#include "internal.hpp"
#include "window.hpp"

#include <SDL3/SDL_filesystem.h>
#include <magic_enum.hpp>

#include "tracy/Tracy.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#ifdef AURORA_ENABLE_GX
extern "C" const aurora::gfx::TextureRef* AuroraGetLastCopyDispTextureForReadback(void);
#endif

namespace aurora {
AuroraConfig g_config;
uint32_t g_sdlCustomEventsStart;
char g_gameName[4];

namespace {
Module Log("aurora");

#ifdef AURORA_ENABLE_GX
uint32_t capture_align_to(uint32_t value, uint32_t alignment) {
  return (value + alignment - 1u) & ~(alignment - 1u);
}

uint32_t capture_crc32_update(uint32_t crc, const uint8_t* data, size_t size) {
  crc = ~crc;
  for (size_t i = 0; i < size; i++) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; bit++) {
      crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
  }
  return ~crc;
}

uint32_t capture_adler32(const std::vector<uint8_t>& data) {
  uint32_t a = 1;
  uint32_t b = 0;
  for (uint8_t value : data) {
    a = (a + value) % 65521u;
    b = (b + a) % 65521u;
  }
  return (b << 16) | a;
}

void capture_append_u32(std::vector<uint8_t>& out, uint32_t value) {
  out.push_back(uint8_t(value >> 24));
  out.push_back(uint8_t(value >> 16));
  out.push_back(uint8_t(value >> 8));
  out.push_back(uint8_t(value));
}

void capture_append_chunk(std::vector<uint8_t>& out, const char type[4], const std::vector<uint8_t>& payload) {
  capture_append_u32(out, uint32_t(payload.size()));
  const size_t type_pos = out.size();
  out.insert(out.end(), type, type + 4);
  out.insert(out.end(), payload.begin(), payload.end());
  capture_append_u32(out, capture_crc32_update(0, out.data() + type_pos, 4 + payload.size()));
}

bool capture_write_png_rgba(const std::filesystem::path& path, int w, int h, const std::vector<uint8_t>& rgba) {
  if (w <= 0 || h <= 0 || rgba.size() < size_t(w) * size_t(h) * 4) {
    return false;
  }
  std::filesystem::create_directories(path.parent_path());

  std::vector<uint8_t> filtered;
  filtered.reserve(size_t(h) * (size_t(w) * 4 + 1));
  for (int y = h - 1; y >= 0; --y) {
    filtered.push_back(0);
    const uint8_t* row = rgba.data() + size_t(y) * size_t(w) * 4;
    filtered.insert(filtered.end(), row, row + size_t(w) * 4);
  }

  std::vector<uint8_t> zlib;
  zlib.push_back(0x78);
  zlib.push_back(0x01);
  size_t pos = 0;
  while (pos < filtered.size()) {
    const uint16_t block = uint16_t(std::min<size_t>(65535, filtered.size() - pos));
    const bool final = pos + block == filtered.size();
    zlib.push_back(final ? 1 : 0);
    zlib.push_back(uint8_t(block));
    zlib.push_back(uint8_t(block >> 8));
    const uint16_t nlen = uint16_t(~block);
    zlib.push_back(uint8_t(nlen));
    zlib.push_back(uint8_t(nlen >> 8));
    zlib.insert(zlib.end(), filtered.begin() + pos, filtered.begin() + pos + block);
    pos += block;
  }
  capture_append_u32(zlib, capture_adler32(filtered));

  std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
  std::vector<uint8_t> ihdr;
  capture_append_u32(ihdr, uint32_t(w));
  capture_append_u32(ihdr, uint32_t(h));
  ihdr.push_back(8);
  ihdr.push_back(6);
  ihdr.push_back(0);
  ihdr.push_back(0);
  ihdr.push_back(0);
  capture_append_chunk(png, "IHDR", ihdr);
  capture_append_chunk(png, "IDAT", zlib);
  capture_append_chunk(png, "IEND", {});

  std::ofstream f(path, std::ios::binary);
  if (!f) {
    return false;
  }
  f.write(reinterpret_cast<const char*>(png.data()), std::streamsize(png.size()));
  return f.good();
}

bool capture_texture_png(const char* path, const wgpu::Texture& texture, wgpu::TextureFormat format,
                         const wgpu::Extent3D& size) {
  if (!path || !*path || size.width == 0 || size.height == 0 || !texture) {
    return false;
  }

  const uint32_t bytesPerPixel = 4;
  const uint32_t bytesPerRow = capture_align_to(size.width * bytesPerPixel, 256);
  const uint64_t bufferSize = uint64_t(bytesPerRow) * size.height;
  wgpu::BufferDescriptor bufferDesc{
      .label = "Aurora PNG texture capture",
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = bufferSize,
  };
  auto readback = aurora::webgpu::g_device.CreateBuffer(&bufferDesc);

  const wgpu::TexelCopyTextureInfo src{
      .texture = texture,
      .mipLevel = 0,
      .origin = {0, 0, 0},
      .aspect = wgpu::TextureAspect::All,
  };
  const wgpu::TexelCopyBufferInfo dst{
      .layout = {.offset = 0, .bytesPerRow = bytesPerRow, .rowsPerImage = size.height},
      .buffer = readback,
  };
  const wgpu::Extent3D copySize{.width = size.width, .height = size.height, .depthOrArrayLayers = 1};
  const wgpu::CommandEncoderDescriptor encDesc{.label = "Aurora PNG texture capture encoder"};
  auto encoder = aurora::webgpu::g_device.CreateCommandEncoder(&encDesc);
  encoder.CopyTextureToBuffer(&src, &dst, &copySize);
  const wgpu::CommandBufferDescriptor cmdDesc{.label = "Aurora PNG texture capture command"};
  auto cmd = encoder.Finish(&cmdDesc);
  aurora::webgpu::g_queue.Submit(1, &cmd);

  bool done = false;
  bool ok = false;
  readback.MapAsync(wgpu::MapMode::Read, 0, bufferSize, wgpu::CallbackMode::AllowSpontaneous,
                    [&](wgpu::MapAsyncStatus status, wgpu::StringView) {
                      ok = status == wgpu::MapAsyncStatus::Success;
                      done = true;
                    });
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!done && std::chrono::steady_clock::now() < deadline) {
    aurora::webgpu::g_instance.ProcessEvents();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (!done || !ok) {
    return false;
  }

  const auto* mapped = static_cast<const uint8_t*>(readback.GetConstMappedRange(0, bufferSize));
  std::vector<uint8_t> rgba(size_t(size.width) * size.height * 4);
  const bool bgra = format == wgpu::TextureFormat::BGRA8Unorm || format == wgpu::TextureFormat::BGRA8UnormSrgb;
  for (uint32_t y = 0; y < size.height; ++y) {
    const uint8_t* srcRow = mapped + size_t(y) * bytesPerRow;
    uint8_t* dstRow = rgba.data() + size_t(y) * size.width * 4;
    for (uint32_t x = 0; x < size.width; ++x) {
      const uint8_t* p = srcRow + size_t(x) * 4;
      uint8_t* q = dstRow + size_t(x) * 4;
      q[0] = bgra ? p[2] : p[0];
      q[1] = p[1];
      q[2] = bgra ? p[0] : p[2];
      q[3] = p[3];
    }
  }
  readback.Unmap();
  return capture_write_png_rgba(path, int(size.width), int(size.height), rgba);
}
#endif

#ifdef AURORA_ENABLE_GX
// GPU
using webgpu::g_device;
using webgpu::g_queue;
using webgpu::g_surface;
#endif

#ifdef AURORA_ENABLE_GX
constexpr std::array PreferredBackendOrder{
#ifdef ENABLE_BACKEND_WEBGPU
    BACKEND_WEBGPU,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D12
    BACKEND_D3D12,
#endif
#ifdef DAWN_ENABLE_BACKEND_METAL
    BACKEND_METAL,
#endif
#ifdef DAWN_ENABLE_BACKEND_VULKAN
    BACKEND_VULKAN,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D11
    BACKEND_D3D11,
#endif
// #ifdef DAWN_ENABLE_BACKEND_DESKTOP_GL
//     BACKEND_OPENGL,
// #endif
// #ifdef DAWN_ENABLE_BACKEND_OPENGLES
//     BACKEND_OPENGLES,
// #endif
#ifdef DAWN_ENABLE_BACKEND_NULL
    BACKEND_NULL,
#endif
};
#else
constexpr std::array<AuroraBackend, 0> PreferredBackendOrder{};
#endif

bool g_initialFrame = false;

AuroraInfo initialize(int argc, char* argv[], const AuroraConfig& config) noexcept {
  g_config = config;
  Log.info("Aurora initializing");
  if (g_config.appName == nullptr) {
    g_config.appName = "Aurora";
  } else {
    g_config.appName = strdup(g_config.appName);
  }
  if (g_config.configPath == nullptr) {
    g_config.configPath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.configPath = strdup(g_config.configPath);
  }
  if (g_config.msaa == 0) {
    g_config.msaa = 1;
  }
  if (g_config.maxTextureAnisotropy == 0) {
    g_config.maxTextureAnisotropy = 16;
  }
  ASSERT(window::initialize(), "Error initializing window");

  g_sdlCustomEventsStart = SDL_RegisterEvents(2);
  ASSERT(g_sdlCustomEventsStart, "Failed to allocate user events: {}", SDL_GetError());
  ASSERT(window::initialize_event_watch(), "Error initializing SDL event watch");

#ifdef AURORA_ENABLE_GX
  /* Attempt to create a window using the calling application's desired backend */
  AuroraBackend selectedBackend = config.desiredBackend;
  bool windowCreated = false;
  if (selectedBackend != BACKEND_AUTO && window::create_window(selectedBackend)) {
    if (webgpu::initialize(selectedBackend)) {
      windowCreated = true;
    } else {
      window::destroy_window();
    }
  }

  if (!windowCreated) {
    for (const auto backendType : PreferredBackendOrder) {
      selectedBackend = backendType;
      if (!window::create_window(selectedBackend)) {
        continue;
      }
      if (webgpu::initialize(selectedBackend)) {
        windowCreated = true;
        break;
      } else {
        window::destroy_window();
      }
    }
  }

  ASSERT(windowCreated, "Error creating window: {}", SDL_GetError());

  // Initialize SDL_Renderer for ImGui when we can't use a Dawn backend
  if (webgpu::g_backendType == wgpu::BackendType::Null) {
    ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
  }
#else
  AuroraBackend selectedBackend = BACKEND_NULL;
  ASSERT(window::create_window(BACKEND_NULL), "Error creating window: {}", SDL_GetError());
  ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
#endif

  window::show_window();

#ifdef AURORA_ENABLE_GX
  gfx::initialize();

  imgui::create_context();
#endif
  const auto size = window::get_window_size();
  Log.info("Using framebuffer size {}x{} scale {}", size.fb_width, size.fb_height, size.scale);
#ifdef AURORA_ENABLE_GX
  if (g_config.imGuiInitCallback != nullptr) {
    g_config.imGuiInitCallback(&size);
  }
  imgui::initialize();
#endif

#ifdef AURORA_ENABLE_RMLUI
  rmlui::initialize(size);
#endif

  g_initialFrame = true;
  g_config.desiredBackend = selectedBackend;
  return {
      .backend = selectedBackend,
      .configPath = g_config.configPath,
      .window = window::get_sdl_window(),
      .windowSize = size,
  };
}

#ifdef AURORA_ENABLE_GX
wgpu::TextureView g_currentView;
#endif

void shutdown() noexcept {
#ifdef AURORA_ENABLE_RMLUI
  rmlui::shutdown();
#endif
#ifdef AURORA_ENABLE_GX
  g_currentView = {};
  imgui::shutdown();
  gfx::shutdown();
  webgpu::shutdown();
#endif
  input::shutdown();
  window::shutdown();
}

const AuroraEvent* update() noexcept {
  ZoneScoped;
  if (g_initialFrame) {
    g_initialFrame = false;
    input::initialize();
  }
  return window::poll_events();
}

bool begin_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
  {
    window::SurfaceLock surfaceLock;
    if (!window::is_presentable()) {
      webgpu::release_surface();
      return false;
    }
    if (window::is_paused()) {
      return false;
    }
    if (!g_surface) {
      webgpu::refresh_surface(true);
      if (!g_surface) {
        return false;
      }
    }
    wgpu::SurfaceTexture surfaceTexture;
    g_surface.GetCurrentTexture(&surfaceTexture);
    switch (surfaceTexture.status) {
    case wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal:
      g_currentView = surfaceTexture.texture.CreateView();
      break;
    case wgpu::SurfaceGetCurrentTextureStatus::Timeout:
      Log.warn("Surface texture acquisition timed out");
      return false;
    case wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal:
    case wgpu::SurfaceGetCurrentTextureStatus::Outdated:
      Log.info("Surface texture is {}, reconfiguring swapchain", magic_enum::enum_name(surfaceTexture.status));
      webgpu::refresh_surface(false);
      return false;
    case wgpu::SurfaceGetCurrentTextureStatus::Lost:
      Log.warn("Surface texture is {}, releasing surface", magic_enum::enum_name(surfaceTexture.status));
      webgpu::release_surface();
    case wgpu::SurfaceGetCurrentTextureStatus::Error:
      Log.warn("Surface texture is {}, dropping surface", magic_enum::enum_name(surfaceTexture.status));
      g_surface = {};
      return false;
    default:
      Log.error("Failed to get surface texture: {}", magic_enum::enum_name(surfaceTexture.status));
      return false;
    }
  }

  imgui::new_frame(window::get_window_size());
  if (!gfx::begin_frame()) {
    g_currentView = {};
    return false;
  }
#endif
  return true;
}

void end_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
  gx::fifo::drain();
  const auto encoderDescriptor = wgpu::CommandEncoderDescriptor{
      .label = "Redraw encoder",
  };
  auto encoder = g_device.CreateCommandEncoder(&encoderDescriptor);
  gfx::end_frame(encoder);
  gfx::render(encoder);
  {
    window::SurfaceLock surfaceLock;
    if (window::is_presentable() && g_surface && g_currentView) {
      const auto& presentSource = webgpu::present_source();
      auto viewport = webgpu::calculate_present_viewport(webgpu::g_graphicsConfig.surfaceConfiguration.width,
                                                         webgpu::g_graphicsConfig.surfaceConfiguration.height,
                                                         presentSource.size.width, presentSource.size.height);
      wgpu::BindGroup presentBindGroup = webgpu::g_CopyBindGroup;
    #if AURORA_ENABLE_RMLUI
      if (rmlui::is_initialized()) {
        const auto rmlOutput = rmlui::render(encoder, viewport);
        if (rmlOutput.texture != nullptr) {
          presentBindGroup = rmlOutput.copyBindGroup;
        }
      }
    #endif
      {
        const std::array attachments{
            wgpu::RenderPassColorAttachment{
                .view = g_currentView,
                .loadOp = wgpu::LoadOp::Clear,
                .storeOp = wgpu::StoreOp::Store,
            },
        };
        const wgpu::RenderPassDescriptor renderPassDescriptor{
            .label = "EFB copy render pass",
            .colorAttachmentCount = attachments.size(),
            .colorAttachments = attachments.data(),
        };
        const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
        // Copy EFB -> XFB (swapchain)
        pass.SetPipeline(webgpu::g_CopyPipeline);
        pass.SetBindGroup(0, presentBindGroup, 0, nullptr);
        pass.SetViewport(viewport.left, viewport.top, viewport.width, viewport.height, viewport.znear, viewport.zfar);

        pass.Draw(3);
        pass.End();
      }
      {
        const std::array attachments{
            wgpu::RenderPassColorAttachment{
                .view = g_currentView,
                .loadOp = wgpu::LoadOp::Load,
                .storeOp = wgpu::StoreOp::Store,
            },
        };
        const wgpu::RenderPassDescriptor renderPassDescriptor{
            .label = "ImGui render pass",
            .colorAttachmentCount = attachments.size(),
            .colorAttachments = attachments.data(),
        };
        const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
        pass.SetViewport(0.f, 0.f, static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.width),
                         static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.height), 0.f, 1.f);
        imgui::render(pass);
        pass.End();
      }
    } else {
      Log.info("Skipping present; window not presentable");
      webgpu::release_surface();
    }
    const wgpu::CommandBufferDescriptor cmdBufDescriptor{.label = "Redraw command buffer"};
    const auto buffer = encoder.Finish(&cmdBufDescriptor);
    g_queue.Submit(1, &buffer);
    gfx::after_submit();
    if (window::is_presentable() && g_surface) {
      auto presentStatus = g_surface.Present();
      if (presentStatus != wgpu::Status::Success) {
        Log.warn("Surface present failed: {}", static_cast<int>(presentStatus));
        webgpu::release_surface();
      }
      webgpu::g_CopyBindGroup = webgpu::create_copy_bind_group(webgpu::present_source());
    } else if (g_surface) {
      webgpu::release_surface();
    }
    g_currentView = {};
  }

  TracyPlotConfig("aurora: lastVertSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastUniformSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastIndexSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastStorageSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastTextureUploadSize", tracy::PlotFormatType::Memory, false, true, 0);

  TracyPlot("aurora: queuedPipelines", static_cast<int64_t>(gfx::g_stats.queuedPipelines));
  TracyPlot("aurora: createdPipelines", static_cast<int64_t>(gfx::g_stats.createdPipelines));
  TracyPlot("aurora: drawCallCount", static_cast<int64_t>(gfx::g_stats.drawCallCount));
  TracyPlot("aurora: mergedDrawCallCount", static_cast<int64_t>(gfx::g_stats.mergedDrawCallCount));
  TracyPlot("aurora: lastVertSize", static_cast<int64_t>(gfx::g_stats.lastVertSize));
  TracyPlot("aurora: lastUniformSize", static_cast<int64_t>(gfx::g_stats.lastUniformSize));
  TracyPlot("aurora: lastIndexSize", static_cast<int64_t>(gfx::g_stats.lastIndexSize));
  TracyPlot("aurora: lastStorageSize", static_cast<int64_t>(gfx::g_stats.lastStorageSize));
  TracyPlot("aurora: lastTextureUploadSize", static_cast<int64_t>(gfx::g_stats.lastTextureUploadSize));

#endif
}
} // namespace
} // namespace aurora

// C API bindings
AuroraInfo aurora_initialize(int argc, char* argv[], const AuroraConfig* config) {
  return aurora::initialize(argc, argv, *config);
}
void aurora_shutdown() { aurora::shutdown(); }
const AuroraEvent* aurora_update() { return aurora::update(); }
bool aurora_begin_frame() { return aurora::begin_frame(); }
void aurora_end_frame() { aurora::end_frame(); }
bool aurora_capture_present_png(const char* path) {
#ifdef AURORA_ENABLE_GX
  const auto& source = aurora::webgpu::present_source();
  return aurora::capture_texture_png(path, source.texture, source.format, source.size);
#else
  (void)path;
  return false;
#endif
}
bool aurora_capture_efb_png(const char* path) {
#ifdef AURORA_ENABLE_GX
  return aurora::capture_texture_png(path, aurora::webgpu::g_frameBuffer.texture, aurora::webgpu::g_frameBuffer.format,
                                     aurora::webgpu::g_frameBuffer.size);
#else
  (void)path;
  return false;
#endif
}
bool aurora_capture_xfb_png(const char* path) {
#ifdef AURORA_ENABLE_GX
  const auto* xfb = AuroraGetLastCopyDispTextureForReadback();
  return xfb && aurora::capture_texture_png(path, xfb->texture, xfb->format, xfb->size);
#else
  (void)path;
  return false;
#endif
}
AuroraBackend aurora_get_backend() { return aurora::g_config.desiredBackend; }
const AuroraBackend* aurora_get_available_backends(size_t* count) {
  if (count != nullptr) {
    *count = aurora::PreferredBackendOrder.size();
  }
  return aurora::PreferredBackendOrder.data();
}
void aurora_set_log_level(AuroraLogLevel level) { aurora::g_config.logLevel = level; }
void aurora_set_pause_on_focus_lost(bool value) { aurora::g_config.pauseOnFocusLost = value; }
void aurora_set_background_input(bool value) {
  aurora::g_config.allowJoystickBackgroundEvents = value;
  aurora::window::set_background_input(value);
}
