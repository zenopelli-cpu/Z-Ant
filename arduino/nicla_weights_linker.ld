/*
 * Additional linker script sections for Zant weights on Nicla Vision
 * Include this in your main linker script or add to platform.txt
 */

SECTIONS
{
    /* Section for storing AI model weights in flash */
    .zant_weights : ALIGN(4)
    {
        _weights_start = .;
        KEEP(*(.zant_weights*))
        . = ALIGN(4);
        _weights_end = .;
    } > FLASH

    /* Metadata for weights I/O system */
    .zant_weight_metadata : ALIGN(4)
    {
        _weight_metadata_start = .;
        KEEP(*(.zant_weight_metadata*))
        . = ALIGN(4);
        _weight_metadata_end = .;
    } > FLASH
} 