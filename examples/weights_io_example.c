/*
 * Zant Weights I/O Example
 * 
 * This example demonstrates how to use the new weights I/O layer
 * with custom C callbacks for reading model weights from various sources.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "../src/zant/weights_io.h"

// Mock flash memory simulation
static uint8_t mock_flash_storage[1024 * 1024]; // 1MB simulated flash
static bool flash_initialized = false;

// Initialize mock flash with some test data
void init_mock_flash() {
    if (flash_initialized) return;
    
    // Fill with pattern for testing
    for (size_t i = 0; i < sizeof(mock_flash_storage); i++) {
        mock_flash_storage[i] = (uint8_t)(i % 256);
    }
    flash_initialized = true;
    printf("Mock flash initialized with %zu bytes\n", sizeof(mock_flash_storage));
}

// Custom callback for reading from simulated flash
int spi_flash_read_callback(size_t offset, uint8_t* buffer, size_t size) {
    init_mock_flash();
    
    printf("Reading %zu bytes from flash offset 0x%zx\n", size, offset);
    
    // Bounds checking
    if (offset + size > sizeof(mock_flash_storage)) {
        printf("Error: Read request exceeds flash bounds\n");
        return -1; // Error
    }
    
    // Simulate some flash read latency
    // In real implementation, this would be SPI transfer
    memcpy(buffer, &mock_flash_storage[offset], size);
    
    return 0; // Success
}

// Alternative callback for reading from external EEPROM
int i2c_eeprom_read_callback(size_t offset, uint8_t* buffer, size_t size) {
    printf("Reading %zu bytes from EEPROM offset 0x%zx\n", size, offset);
    
    // Simulate I2C EEPROM read
    // Fill with different pattern for testing
    for (size_t i = 0; i < size; i++) {
        buffer[i] = (uint8_t)((offset + i) & 0xFF) ^ 0xAA;
    }
    
    return 0; // Success
}

// Direct memory access example (traditional XIP mode)
void test_direct_access() {
    printf("\n=== Testing Direct Memory Access ===\n");
    
    // Set up direct access to mock flash
    zant_set_weights_base_address(mock_flash_storage);
    
    // Get info about current configuration
    zant_weights_io_info_t info = zant_get_weights_io_info();
    printf("Direct access configured:\n");
    printf("  Has callback: %s\n", info.has_callback ? "yes" : "no");
    printf("  Has base address: %s\n", info.has_base_address ? "yes" : "no");
    printf("  Base address: 0x%lx\n", info.base_address);
}

// Callback-based access example
void test_callback_access() {
    printf("\n=== Testing Callback-based Access ===\n");
    
    // Register SPI flash callback
    zant_register_weight_callback(spi_flash_read_callback);
    
    // Get info about current configuration
    zant_weights_io_info_t info = zant_get_weights_io_info();
    printf("Callback access configured:\n");
    printf("  Has callback: %s\n", info.has_callback ? "yes" : "no");
    printf("  Has base address: %s\n", info.has_base_address ? "yes" : "no");
    printf("  Callback ptr: 0x%lx\n", info.callback_ptr);
    
    // Test reading some data
    uint8_t test_buffer[32];
    int result = spi_flash_read_callback(0x100, test_buffer, sizeof(test_buffer));
    if (result == 0) {
        printf("Successfully read data: ");
        for (int i = 0; i < 16; i++) {
            printf("%02x ", test_buffer[i]);
        }
        printf("...\n");
    }
}

// Alternative storage callback example
void test_eeprom_access() {
    printf("\n=== Testing EEPROM Callback Access ===\n");
    
    // Switch to EEPROM callback
    zant_register_weight_callback(i2c_eeprom_read_callback);
    
    printf("Switched to EEPROM callback\n");
    
    // Test reading some data
    uint8_t test_buffer[16];
    int result = i2c_eeprom_read_callback(0x200, test_buffer, sizeof(test_buffer));
    if (result == 0) {
        printf("Successfully read EEPROM data: ");
        for (int i = 0; i < sizeof(test_buffer); i++) {
            printf("%02x ", test_buffer[i]);
        }
        printf("\n");
    }
}

// Clear configuration example
void test_clear_config() {
    printf("\n=== Testing Configuration Reset ===\n");
    
    // Clear all configuration
    zant_register_weight_callback(NULL);
    zant_set_weights_base_address(NULL);
    
    zant_weights_io_info_t info = zant_get_weights_io_info();
    printf("Configuration cleared:\n");
    printf("  Has callback: %s\n", info.has_callback ? "yes" : "no");
    printf("  Has base address: %s\n", info.has_base_address ? "yes" : "no");
}

int main() {
    printf("Zant Weights I/O System Example\n");
    printf("===============================\n");
    
    // Initialize the weights I/O system
    zant_init_weights_io();
    
    // Initialize mock data
    init_mock_flash();
    
    // Test different access methods
    test_direct_access();
    test_callback_access();
    test_eeprom_access();
    test_clear_config();
    
    printf("\n=== Real-world Usage Examples ===\n");
    printf("For Arduino/embedded use:\n");
    printf("1. Register callback in setup():\n");
    printf("   zant_register_weight_callback(my_flash_read);\n");
    printf("2. Implement custom reading function:\n");
    printf("   int my_flash_read(size_t offset, uint8_t* buffer, size_t size) {\n");
    printf("     return spi_flash_read(WEIGHTS_BASE + offset, buffer, size);\n");
    printf("   }\n");
    printf("3. Generated inference code automatically uses your callback!\n");
    
    return 0;
} 