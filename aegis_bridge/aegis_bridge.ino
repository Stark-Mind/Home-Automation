#include <WiFi.h>
#include <HTTPClient.h>
#include <WebServer.h>
#include <ArduinoJson.h>

// 1. Hardware Pin Definitions
#define MAIN_LIGHT_PIN 23  // Red LED
#define LEFT_LIGHT_PIN 22  // Green LED

// 2. Network Configuration
const char* ssid = "DESKTOP-46EK6B5 2554";
const char* password = "6jY1979/";
const char* bridgeBaseUrl = "http://192.168.137.1:5000";

// 3. Telemetry State Tracking
bool mainLightState = false;
bool leftLightState = false;

// 4. Local HTTP Server for Instant Voice Push Commands
WebServer espServer(80);
unsigned long lastHeartbeat = 0;
const unsigned long HEARTBEAT_INTERVAL = 30000; // 30 seconds

void registerWithBridge() {
    if (WiFi.status() != WL_CONNECTED) return;

    HTTPClient http;
    http.begin(String(bridgeBaseUrl) + "/register");
    http.setTimeout(3000);
    http.addHeader("Content-Type", "application/json");

    StaticJsonDocument<256> doc;
    doc["ip"] = WiFi.localIP().toString();
    JsonObject telemetry = doc.createNestedObject("telemetry");
    telemetry["main_light"] = mainLightState;
    telemetry["left_light"] = leftLightState;

    String payload;
    serializeJson(doc, payload);
    http.POST(payload);
    http.end();
}

void handlePushExecute() {
    if (!espServer.hasArg("plain")) {
        espServer.send(400, "application/json", "{\"error\":\"missing body\"}");
        return;
    }

    String body = espServer.arg("plain");
    Serial.println("\n[VOICE PUSH]: " + body);

    StaticJsonDocument<256> commandDoc;
    DeserializationError error = deserializeJson(commandDoc, body);

    if (!error) {
        String targetDevice = commandDoc["device"];
        String targetState = commandDoc["state"];
        executeHardwareAction(targetDevice, targetState);

        // Return ground-truth GPIO states back to Python bridge
        StaticJsonDocument<128> respDoc;
        respDoc["main_light"] = mainLightState;
        respDoc["left_light"] = leftLightState;
        String respJson;
        serializeJson(respDoc, respJson);
        espServer.send(200, "application/json", respJson);
    } else {
        espServer.send(400, "application/json", "{\"error\":\"invalid json\"}");
    }
}

void setup() {
    Serial.begin(115200);
    delay(100);

    // Initialize Motor Cortex Pins
    pinMode(MAIN_LIGHT_PIN, OUTPUT);
    pinMode(LEFT_LIGHT_PIN, OUTPUT);
    digitalWrite(MAIN_LIGHT_PIN, LOW);
    digitalWrite(LEFT_LIGHT_PIN, LOW);

    // Connect to Wi-Fi
    Serial.print("\nConnecting to Wi-Fi");
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    Serial.println("\nWi-Fi Connected! IP: " + WiFi.localIP().toString());

    // Start local listener for voice commands pushed from PC
    espServer.on("/execute", HTTP_POST, handlePushExecute);
    espServer.begin();

    // Register IP and initial states with Python bridge
    registerWithBridge();
    lastHeartbeat = millis();

    Serial.println("A.E.G.I.S. Dual-Mode Cortex Ready (Voice Push + Serial Text).");
}

void loop() {
    // 1. Handle incoming voice commands pushed from PC
    espServer.handleClient();

    // 2. Keep ESP32 IP registered with Python bridge
    if (millis() - lastHeartbeat >= HEARTBEAT_INTERVAL) {
        registerWithBridge();
        lastHeartbeat = millis();
    }

    // 3. Preserve Serial Monitor text input
    if (Serial.available() > 0) {
        String userInput = Serial.readStringUntil('\n');
        userInput.trim();

        if (userInput.length() > 0) {
            Serial.println("\n[USER]: " + userInput);
            sendToAegis(userInput);
        }
    }
}

void sendToAegis(String userInput) {
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(String(bridgeBaseUrl) + "/aegis");
        http.setTimeout(15000);
        http.addHeader("Content-Type", "application/json");

        StaticJsonDocument<256> doc;
        JsonObject telemetry = doc.createNestedObject("telemetry");
        telemetry["main_light"] = mainLightState;
        telemetry["left_light"] = leftLightState;
        doc["input"] = userInput;

        String requestBody;
        serializeJson(doc, requestBody);

        int httpResponseCode = http.POST(requestBody);

        if (httpResponseCode > 0) {
            String response = http.getString();
            Serial.println("[A.E.G.I.S. RAW]: " + response);

            StaticJsonDocument<512> responseDoc;
            DeserializationError error = deserializeJson(responseDoc, response);

            if (!error) {
                String rawCommand = responseDoc["command"];
                StaticJsonDocument<256> commandDoc;
                deserializeJson(commandDoc, rawCommand);

                String targetDevice = commandDoc["device"];
                String targetState = commandDoc["state"];
                executeHardwareAction(targetDevice, targetState);
            } else {
                Serial.println("Failed to parse A.E.G.I.S. wrapper JSON.");
            }
        } else {
            Serial.print("Error reaching PC. HTTP Code: ");
            Serial.println(httpResponseCode);
        }
        http.end();
    } else {
        Serial.println("Wi-Fi disconnected. Cannot reach A.E.G.I.S.");
    }
}

void executeHardwareAction(String device, String state) {
    bool turnOn = (state == "on");

    if (device == "main_light") {
        mainLightState = turnOn;
        digitalWrite(MAIN_LIGHT_PIN, turnOn ? HIGH : LOW);
        Serial.println(turnOn ? "[ACTION]: Main Light (Red LED) -> ON" : "[ACTION]: Main Light (Red LED) -> OFF");
    }
    else if (device == "left_light") {
        leftLightState = turnOn;
        digitalWrite(LEFT_LIGHT_PIN, turnOn ? HIGH : LOW);
        Serial.println(turnOn ? "[ACTION]: Left Light (Green LED) -> ON" : "[ACTION]: Left Light (Green LED) -> OFF");
    }
    else if (device == "both") {
        mainLightState = turnOn;
        leftLightState = turnOn;
        digitalWrite(MAIN_LIGHT_PIN, turnOn ? HIGH : LOW);
        digitalWrite(LEFT_LIGHT_PIN, turnOn ? HIGH : LOW);
        Serial.println(turnOn ? "[ACTION]: Both Lights -> ON" : "[ACTION]: Both Lights -> OFF");
    }
    else {
        Serial.println("[ACTION]: None required or invalid device.");
    }

    Serial.println("Awaiting next command...");
}
