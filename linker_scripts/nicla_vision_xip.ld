/*
 * Linker script for Arduino Nicla Vision with XIP support
 * STM32H747XI (Cortex-M7 @ 480MHz + Cortex-M4 @ 240MHz)
 * 
 * Flash: 8MB total (0x08000000 - 0x087FFFFF) - using external flash
 * RAM:   1MB (0x20000000 - 0x200FFFFF) 
 * 
 * This script reserves a large 5MB flash section for AI model weights
 * that can be accessed directly without copying to RAM (XIP mode).
 * Perfect for large models like ResNet, EfficientNet, etc.
 */

MEMORY
{
  /* Main flash for code and data */
  FLASH (rx)      : ORIGIN = 0x08000000, LENGTH = 512K   /* 512KB for code */
  
  /* Dedicated flash section for AI weights (XIP) */
  FLASH_WEIGHTS (r) : ORIGIN = 0x08080000, LENGTH = 5120K  /* 5MB for weights */
  
  /* RAM sections */
  RAM (xrw)       : ORIGIN = 0x20000000, LENGTH = 512K    /* Main RAM */
  RAM_D2 (xrw)    : ORIGIN = 0x30000000, LENGTH = 288K    /* AHB SRAM */
  RAM_D3 (xrw)    : ORIGIN = 0x38000000, LENGTH = 64K     /* AHB SRAM */
  ITCMRAM (xrw)   : ORIGIN = 0x00000000, LENGTH = 64K     /* ITCM RAM */
  DTCMRAM (xrw)   : ORIGIN = 0x20000000, LENGTH = 128K    /* DTCM RAM */
}

SECTIONS
{
  /* Standard sections */
  .text :
  {
    KEEP(*(.isr_vector))
    *(.text*)
    *(.rodata*)
    . = ALIGN(4);
    _etext = .;
  } > FLASH

  /* AI model weights section - stays in flash for XIP access */
  .flash_weights :
  {
    . = ALIGN(4);
    __flash_weights_start__ = .;
    *(.flash_weights)
    . = ALIGN(4);
    __flash_weights_end__ = .;
  } > FLASH_WEIGHTS

  /* Data section initialization */
  .data : 
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } > RAM AT> FLASH

  _sidata = LOADADDR(.data);

  /* BSS section */
  .bss :
  {
    . = ALIGN(4);
    _sbss = .;
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
  } > RAM

  /* Stack and heap */
  .heap :
  {
    . = ALIGN(8);
    PROVIDE(__heap_start__ = .);
    . = . + 64K;  /* 64KB heap */
    PROVIDE(__heap_end__ = .);
  } > RAM

  .stack :
  {
    . = ALIGN(8);
    . = . + 16K;  /* 16KB stack */
    PROVIDE(__stack_top__ = .);
  } > RAM
}

/* Provide symbols for flash weights access */
PROVIDE(__flash_weights_size__ = __flash_weights_end__ - __flash_weights_start__);

/* Standard ARM symbols */
PROVIDE(_stack = ORIGIN(RAM) + LENGTH(RAM));
PROVIDE(_minimum_stack_size = 8K);
PROVIDE(_minimum_heap_size = 16K); 