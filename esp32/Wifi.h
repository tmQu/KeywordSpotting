#include <WiFi.h>
#include <WiFiUdp.h>


#define SSID "Linksys 2.4"
#define PASSWORD "@tang01234"

#define I2S_SERVER_URL "http://192.168.99.135:5003/i2s_samples"
#define SERVER_URL "192.168.99.135"
const int udpServerPort = 3333;
extern WiFiUDP udp;


void sendDataUDP(uint8_t *bytes, size_t count);
