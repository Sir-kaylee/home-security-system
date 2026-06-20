#include <IOT_security_model_inferencing.h>//custom library showing friwndlies and hostiles


/*******************************************************************************************
 * ESP32-CAM Smart Security Stream + Edge Impulse Detection
 * Features:
 *  - MJPEG stream + HTML UI with resize slider
 *  - Inference runs in background, non-blocking
 *  - Bounding boxes + confidence overlay (drawn in browser)
 *  - Efficient RAM usage (QQVGA, PSRAM buffering)
 *******************************************************************************************/
/* ESP32-CAM Smart Stream + Detection
 * ----------------------------------
 * Optimized for RAM safety (PSRAM buffers)
 *  Non-blocking stream + background inference
 *  Bounding boxes, labels, and confidence %
 *  No accumulation — always current frame
 *  Fixed Queue assert error
 */

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include "edge-impulse-sdk/dsp/image/image.hpp"

// ===================
// Camera Model
// ===================
#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

// ===================
// WiFi
// ===================
const char *ssid = "";
const char *password = "";

// ===================
// Globals
// ===================
SemaphoreHandle_t cameraMux = NULL;
SemaphoreHandle_t sseMutex = NULL;
static uint8_t *rgb_src_buf = NULL;
static uint8_t *resized_buf = NULL;
static uint8_t *jpg_copy_buf = NULL;
static size_t jpg_copy_size = 0;

static bool debug_nn = false;
static char latest_msg[512];
static uint32_t latest_seq = 0;

// ===================
// Sizes
// ===================
const size_t SRC_RGB_SIZE = 160 * 120 * 3;
const size_t MODEL_RGB_SIZE = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT * 3;

// ===================
// Forward Decls
// ===================
void startCameraServer();
void run_inference(void *parameter);
static esp_err_t mjpeg_stream_handler(httpd_req_t *req);
static esp_err_t root_handler(httpd_req_t *req);
static esp_err_t events_handler(httpd_req_t *req);
static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr);

// ===================
// SSE helper
// ===================
static void publish_latest_json(const char *json) {
  if (!sseMutex) return;
  if (xSemaphoreTake(sseMutex, pdMS_TO_TICKS(100))) {
    strncpy(latest_msg, json, sizeof(latest_msg) - 1);
    latest_msg[sizeof(latest_msg) - 1] = '\0';
    latest_seq++;
    xSemaphoreGive(sseMutex);
  }
}

// ===================
// Setup
// ===================
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println("ESP32-CAM Stream + Detection starting...");

  // --- Camera Config ---
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QQVGA; // 160x120
  config.jpeg_quality = 12;
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera init failed!");
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);

  // --- WiFi ---
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");

  // --- Mutexes + Buffers ---
  cameraMux = xSemaphoreCreateMutex();
  sseMutex = xSemaphoreCreateMutex();
  rgb_src_buf = (uint8_t *)heap_caps_malloc(SRC_RGB_SIZE, MALLOC_CAP_SPIRAM);
  resized_buf = (uint8_t *)heap_caps_malloc(MODEL_RGB_SIZE, MALLOC_CAP_8BIT);

  if (!rgb_src_buf || !resized_buf || !cameraMux || !sseMutex) {
    Serial.println("Buffer or mutex allocation failed!");
    return;
  }

  strcpy(latest_msg, "{\"clear\":true}");
  latest_seq = 1;

  startCameraServer();

  // Run inference on core 1 (low priority)
  xTaskCreatePinnedToCore(run_inference, "InferenceTask", 12288, NULL, 0, NULL, 1);

  Serial.print("Camera Stream Ready! http://");
  Serial.println(WiFi.localIP());
}

void loop() {
  if (ESP.getFreeHeap() < 45000) {
    Serial.println("Low memory, restarting...");//watchdog to prevent full system hangs after many resizes
    ESP.restart();
  }
  delay(10000);
}


// ===================
// Inference
// ===================
static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr) {
  if (offset + length > EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT) return -1;
  size_t pix = offset * 3;
  for (size_t i = 0; i < length; i++) {
    uint8_t r = resized_buf[pix + 0];
    uint8_t g = resized_buf[pix + 1];
    uint8_t b = resized_buf[pix + 2];
    out_ptr[i] = (r << 16) | (g << 8) | b;
    pix += 3;
  }
  return 0;
}

void run_inference(void *parameter) {
  static bool skip = false;  // declare once, at the top

  for (;;) {
    if (xSemaphoreTake(cameraMux, pdMS_TO_TICKS(500)) == pdFALSE) {
      vTaskDelay(100 / portTICK_PERIOD_MS);
      continue;
    }

    skip = !skip;
    if (skip) { 
      xSemaphoreGive(cameraMux);  //release the lock on the semaphore
      vTaskDelay(100 / portTICK_PERIOD_MS); 
      continue; 
    }


    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      xSemaphoreGive(cameraMux);
      vTaskDelay(100 / portTICK_PERIOD_MS);
      continue;
    }

    if (fb->len > jpg_copy_size) {
      if (jpg_copy_buf) {
        heap_caps_free(jpg_copy_buf);
        jpg_copy_buf = NULL;
      }
      jpg_copy_buf = (uint8_t *)heap_caps_malloc(fb->len, MALLOC_CAP_SPIRAM);
      jpg_copy_size = fb->len;
    }

    memcpy(jpg_copy_buf, fb->buf, fb->len);
    size_t local_len = fb->len;
    esp_camera_fb_return(fb);
    xSemaphoreGive(cameraMux);

    bool ok = fmt2rgb888(jpg_copy_buf, local_len, PIXFORMAT_JPEG, rgb_src_buf);
    if (!ok) continue;

    ei::image::processing::crop_and_interpolate_rgb888(
        rgb_src_buf, 160, 120, resized_buf,
        EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT);

    ei::signal_t signal;
    signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
    signal.get_data = &ei_camera_get_data;

    ei_impulse_result_t result = {0};
    EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) continue;

    char json[512];
    bool found = false;

#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    strcpy(json, "{\"detections\":[");
    bool first = true;
    for (size_t i = 0; i < result.bounding_boxes_count; i++) {
      auto bb = result.bounding_boxes[i];
      if (bb.value == 0) continue;
      found = true;
      if (!first) strcat(json, ",");
      char buf[96];
      snprintf(buf, sizeof(buf),
               "{\"label\":\"%s\",\"value\":%.2f,\"x\":%u,\"y\":%u,\"w\":%u,\"h\":%u}",
               bb.label, bb.value, bb.x, bb.y, bb.width, bb.height);
      strcat(json, buf);
      first = false;
    }
    strcat(json, "]}");
#else
    strcpy(json, "{\"detections\":[]}");
#endif

    if (!found) strcpy(json, "{\"clear\":true}");
    publish_latest_json(json);
    vTaskDelay(300 / portTICK_PERIOD_MS);
  }
}

// ===================
// Web server setup
// ===================
void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  httpd_handle_t server = NULL;

  if (httpd_start(&server, &config) == ESP_OK) {
    httpd_uri_t root_uri = {"/", HTTP_GET, root_handler, NULL};
    httpd_uri_t stream_uri = {"/stream", HTTP_GET, mjpeg_stream_handler, NULL};
    httpd_uri_t events_uri = {"/events", HTTP_GET, events_handler, NULL};
    httpd_register_uri_handler(server, &root_uri);
    httpd_register_uri_handler(server, &stream_uri);
    httpd_register_uri_handler(server, &events_uri);
  }
}

// ===================
// MJPEG Stream
// ===================
static esp_err_t mjpeg_stream_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "multipart/x-mixed-replace; boundary=frame");
  while (true) {
    if (xSemaphoreTake(cameraMux, pdMS_TO_TICKS(1000)) == pdFALSE) continue;
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      xSemaphoreGive(cameraMux);
      continue;
    }

    char buf[64];
    size_t hlen = snprintf(buf, sizeof(buf),
                           "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
    if (httpd_resp_send_chunk(req, buf, hlen) != ESP_OK ||
        httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len) != ESP_OK ||
        httpd_resp_send_chunk(req, "\r\n", 2) != ESP_OK) {
      esp_camera_fb_return(fb);
      xSemaphoreGive(cameraMux);
      break;
    }
    esp_camera_fb_return(fb);
    xSemaphoreGive(cameraMux);
    vTaskDelay(80 / portTICK_PERIOD_MS);//from initial 40 ms
  }
  return ESP_OK;
}

// ===================
// Root HTML
// ===================
static const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html><html><head>
<meta charset="utf-8"><title>Smart Security</title>
<style>
body{font-family:Arial;text-align:center;background:#111;color:#eee;}
#video-container{position:relative;display:inline-block;margin-top:10px;}
#stream{border:2px solid #333;}
#overlay{position:absolute;top:0;left:0;pointer-events:none;}
#logs{max-width:720px;margin:10px auto;text-align:left;background:#0b0b0b;
padding:8px;border-radius:6px;border:1px solid #222;height:120px;overflow:auto;
font-size:13px;color:#dcdcdc;}
</style></head><body>
<h2>Smart Security System</h2>
<label>Resize Stream:</label>
<input type="range" id="slider" min="160" max="640" value="320" step="10">
<div id="video-container">
<img id="stream" src="/stream" width="320"><canvas id="overlay"></canvas>
</div><div id="logs"></div>
<script>
const img = document.getElementById('stream'),
      canvas = document.getElementById('overlay'),
      ctx = canvas.getContext('2d'),
      logs = document.getElementById('logs'),
      slider = document.getElementById('slider');

function resize() {
  canvas.width = img.width;
  canvas.height = img.height;
}

// Resize handling
img.onload = resize;

//SAFER: debounce stream reconnect to avoid multiple open connections
let resizeTimer = null;
slider.oninput = () => {
  img.width = slider.value;
  resize();

  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    // Add random query param to force new connection, close old cleanly
    img.src = `/stream?rand=${Date.now()}`;
  }, 500);
};

// SSE event listener
const evt = new EventSource('/events');
evt.onmessage = e => {
  try {
    const data = JSON.parse(e.data);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    logs.textContent = "";

    if (data.clear) {
      logs.textContent = "No objects";
      return;
    }

    if (data.detections) {
      data.detections.forEach(det => {
        const sx = det.x * (canvas.width / 160),
              sy = det.y * (canvas.height / 120),
              sw = det.w * (canvas.width / 160),
              sh = det.h * (canvas.height / 120),
              color = det.label === "safe" ? "lime" : "red";

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(sx, sy, sw, sh);
        ctx.font = "14px Arial";
        ctx.fillStyle = color;
        const label = `${det.label} ${(det.value * 100).toFixed(1)}%`;
        ctx.fillText(label, sx + 2, sy > 10 ? sy - 4 : sy + 14);
        logs.textContent += `${label}\n`;
      });
    }
  } catch (ex) {
    console.warn("SSE parse error", ex);
  }
};

// Optional: log SSE errors to console
evt.onerror = () => {
  console.warn("SSE connection lost — retrying...");
};
</script>
</body></html>
)rawliteral";

static esp_err_t root_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, index_html, HTTPD_RESP_USE_STRLEN);
}

// ===================
// SSE Events
// ===================
static esp_err_t events_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/event-stream");
  httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
  httpd_resp_set_hdr(req, "Connection", "keep-alive");

  char buf[512];
  uint32_t last_sent_seq = 0;

  while (true) {
    if (xSemaphoreTake(sseMutex, pdMS_TO_TICKS(200))) {
      if (latest_seq != last_sent_seq) {
        strncpy(buf, latest_msg, sizeof(buf) - 1);
        last_sent_seq = latest_seq;
        xSemaphoreGive(sseMutex);
        if (httpd_resp_send_chunk(req, "data: ", 6) != ESP_OK) return ESP_FAIL;
        if (httpd_resp_send_chunk(req, buf, strlen(buf)) != ESP_OK) return ESP_FAIL;
        if (httpd_resp_send_chunk(req, "\n\n", 2) != ESP_OK) return ESP_FAIL;
      } else xSemaphoreGive(sseMutex);
    }
    vTaskDelay(100 / portTICK_PERIOD_MS);
  }
  return ESP_OK;
}
