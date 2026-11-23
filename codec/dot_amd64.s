//go:build amd64 && !purego

#include "textflag.h"

// func dotI16Kernel(a, b *int16, n int) int32
TEXT ·dotI16Kernel(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), SI
	MOVQ b+8(FP), DI
	MOVQ n+16(FP), CX
	PXOR X0, X0
	MOVQ CX, R8
	SHRQ $3, R8
	JZ   hsum
loop:
	MOVOU (SI), X1
	MOVOU (DI), X2
	PMADDWL X2, X1
	PADDD X1, X0
	ADDQ $16, SI
	ADDQ $16, DI
	DECQ R8
	JNZ  loop
hsum:
	PSHUFD $0x4E, X0, X1
	PADDD X1, X0
	PSHUFD $0xB1, X0, X1
	PADDD X1, X0
	MOVD X0, AX
	MOVL AX, ret+24(FP)
	RET
