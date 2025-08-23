#ifndef ZANT_WEIGHTS_IO_H
#define ZANT_WEIGHTS_IO_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Function pointer type for C weight reading callback
/// Parameters:
/// - offset: Byte offset from the start of weights region
/// - buffer: Destination buffer to read data into
/// - size: Number of bytes to read
/// Returns: 0 on success, non-zero on error
typedef int (*zant_weight_read_callback_t)(size_t offset, uint8_t* buffer, size_t size);

/// Information structure about weights I/O configuration
typedef struct {
    bool has_callback;
    bool has_base_address;
    uintptr_t callback_ptr;
    uintptr_t base_address;
} zant_weights_io_info_t;

/// Register a C callback function for reading weights
/// If callback is NULL, clears the current callback
void zant_register_weight_callback(zant_weight_read_callback_t callback);

/// Set the base address for direct weight access (fallback mode)
/// This should point to the start of the weights region in memory
void zant_set_weights_base_address(const uint8_t* base_address);

/// Get information about the current weight I/O configuration
zant_weights_io_info_t zant_get_weights_io_info(void);

/// Initialize the weights I/O system
/// This should be called during system initialization
void zant_init_weights_io(void);

// Example usage for custom flash reading:
//
// int my_flash_read(size_t offset, uint8_t* buffer, size_t size) {
//     // Read from SPI flash, I2C EEPROM, etc.
//     return spi_flash_read(WEIGHTS_FLASH_BASE + offset, buffer, size);
// }
//
// void setup() {
//     zant_register_weight_callback(my_flash_read);
//     // Now inference will use your custom reading function
// }

#ifdef __cplusplus
}
#endif

#endif // ZANT_WEIGHTS_IO_H 