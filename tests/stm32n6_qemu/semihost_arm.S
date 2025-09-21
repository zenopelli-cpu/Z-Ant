.syntax unified
.thumb
.text
.global semihost_call
.type semihost_call, %function

// long semihost_call(unsigned int op, void *param)
// r0 = op, r1 = param per AAPCS. Perform BKPT 0xAB; QEMU handles semihosting.
semihost_call:
    bkpt 0xAB
    bx lr
