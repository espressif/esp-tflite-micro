// Copyright 2021-2022 Espressif Systems (Shanghai) PTE LTD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

    .text
    .literal_position
    .literal  .nudge_val, 1073741824

    # Program Unit: esp_nn_conv_s8_mult8_1x1_esp32s3
    #
    # Requirements:
    #   - in_channels must be a multiple of 8
    #   - filter_data must be 8-byte aligned (ee.vld.l.64.ip ignores lower 3 address bits)
    #   - input_data must be 8-byte aligned (ee.vld.l/h.64.xp same alignment requirement)
    #   - buffer (scratch) must be 16-byte aligned
    #
    # If filter is not aligned, use esp_nn_conv_s8_1x1() (C+inline asm) as fallback.
    #
    .type   esp_nn_conv_s8_mult8_1x1_esp32s3, @function
    .align   4
    .global esp_nn_conv_s8_mult8_1x1_esp32s3

esp_nn_conv_s8_mult8_1x1_esp32s3:  # 0xdbc
    # scratch_buf = 0   // to store qacc regs need 36 bytes
    # gra_spill_temp_164 = 36, channel itr, (in_channels - 1) >> 3
    # gra_spill_temp_165 = 40, i_out
    # gra_spill_temp_166 = 44, in_channels
    # gra_spill_temp_167 = 48, in_channels/8 - 1
    # gra_spill_temp_168 = 52, in_channels-7
    # gra_spill_temp_169 = 56, input
    # gra_spill_temp_170 = 60, filter_data
    # gra_spill_temp_171 = 64, input_offset
    # gra_spill_temp_172 = 68, input_ptr
    # gra_spill_temp_173 = 72, bias
    # gra_spill_temp_174 = 76, in_channels*8
    # gra_spill_temp_175 = 80, size-7
    # gra_spill_temp_176 = 84, size

 // registers:
 // a2: int8_t *input_data
 // a3: uint16_t input_wd
 // a4: uint16_t input_ht
 // a5: uint16_t in_channels
 // a6: int32_t input_offset
 // a7: int16_t *filter_data

 // on stack:
 // int32_t *bias           // 160
 // int8_t *out_data        // 164
 // uint16_t out_wd         // 168
 // uint16_t out_ht         // 172
 // uint16_t out_channels   // 176
 // int32_t out_offset      // 180
 // int32_t *out_shift      // 184
 // int32_t *out_mult       // 188
 // int32_t activation_min  // 192
 // int32_t activation_max  // 196
 // void *buffer // tmp buf // 200
 // const int32_t *row_addend // 204, NULL or 16-byte aligned

    entry   a1,160                      #
    s32i    a5,a1,44                    # [0]  gra_spill_temp_166, in_channels
    s32i    a6,a1,64                    # [2]  id:619 input_offset+0x0
    s32i    a7,a1,60                    # [1]  gra_spill_temp_170, filter_data
    mul16u  a8,a3,a4                    # [3]  size = input_wd * input_ht;
    s32i    a2,a1,56                    # [0]  gra_spill_temp_169, input
    l32i    a4,a1,164                   # [1]  id:624 out_data+0x0
    mov.n   a3,a1                       # [52]  scratch_buf

    s32i    a8,a1,84                    # [4]  gra_spill_temp_176, size
    blti    a8,8,.prepare_leftover      # [5] // process remaining lines one by one
    addi    a9,a8,-7                    # [32]
    s32i    a9,a1,80                    # [33]  gra_spill_temp_175, size-7

    s32i    a2,a1,68                    # [2]  gra_spill_temp_172 , input_ptr
    srai    a15,a5,3                    # [7] `in_ch/8` loop_cnt
    movi.n  a11,0                       # [10]
    s32i    a11,a1,40                   # [11]  gra_spill_temp_165
    addi    a15,a15,-1                  # [17]  `in_ch/8` loop_cnt - 1
    s32i    a15,a1,48                   # [18]  gra_spill_temp_167
    slli    a9,a5,3                     # [19]  in_channels*8
    s32i    a9,a1,76                    # [20]  gra_spill_temp_174
    addi    a15,a5,-7                   # [31]
    s32i    a15,a1,52                   # [34]  gra_spill_temp_168

.outer_loop: // for (; i_out < size - 7; i_out += 8) {

    l32i    a10,a1,200                  # [1]  gra_spill_temp_165, buffer
    l32i.n  a11,a1,44                   # [1]  gra_spill_temp_166, input_channels
    l32i.n  a8,a1,68                    # [2]  gra_spill_temp_172, input_ptr
    srai    a9,a11,3                    # [7] `in_ch/8` loop_cnt for transpose loop

    ee.zero.q   q7                      # [0]
    addi        a12,a1,64               # [6]
    ee.vldbc.16 q5,a12                  # [0*II+16]  id:638 input_offset

    // load and transose 8 lines of input 8xchannels,
    // add input offset and store 16 bit data to tmp buffer
    loopgtz a9,.transpose_loop_end  # [10]
    mov.n                   a9,a8
    ee.vld.l.64.xp          q0,a9,a11
    ee.vld.l.64.xp          q1,a9,a11
    ee.vld.h.64.xp          q0,a9,a11
    ee.vld.h.64.xp          q1,a9,a11
    ee.vld.l.64.xp          q2,a9,a11
    ee.vzip.8               q0,q1
    ee.vld.l.64.xp          q3,a9,a11
    ee.vld.h.64.xp          q2,a9,a11
    ee.vld.h.64.ip          q3,a9,0
    ee.vzip.16              q0,q1
    ee.vzip.8               q2,q3
    ee.vzip.16              q2,q3
    ee.vzip.32              q0,q2
    ee.vcmp.lt.s8           q4,q2,q7
    ee.vzip.8               q2,q4
    ee.vcmp.lt.s8           q6,q0,q7
    ee.vzip.8               q0,q6
    ee.vadds.s16            q0,q0,q5
    ee.vadds.s16.st.incp    q0,a10,q6,q6,q5
    ee.vadds.s16.st.incp    q6,a10,q2,q2,q5
    ee.vadds.s16.st.incp    q2,a10,q4,q4,q5
    ee.vst.128.ip           q4,a10,16
    ee.vzip.32              q1,q3
    ee.vcmp.lt.s8           q4,q3,q7
    ee.vzip.8               q3,q4
    ee.vcmp.lt.s8           q6,q1,q7
    ee.vzip.8               q1,q6
    ee.vadds.s16            q1,q1,q5
    ee.vadds.s16.st.incp    q1,a10,q6,q6,q5
    ee.vadds.s16.st.incp    q6,a10,q3,q3,q5
    ee.vadds.s16.st.incp    q3,a10,q4,q4,q5
    ee.vst.128.ip           q4,a10,16
    addi.n                  a8,a8,8
.transpose_loop_end:    # 0xeeb

 # 468          uint32_t bias_ptr = (uint32_t) bias;
 # 469          uint32_t filter_ptr = (uint32_t) (filter_data);
 # 470          const int32_t *out_mult_ptr = out_mult;
 # 471          const int32_t *out_shift_ptr = out_shift;
    l32i    a6,a1,184                   # [0]  out_shift
    l32i    a2,a1,188                   # [1]  out_mult
    l32i    a5,a1,60                    # [2]  gra_spill_temp_170, filter
    l32i    a9,a1,160                   # [3]  gra_spill_temp_170, bias
 # 472          for (int32_t out_ch_idx = 0; out_ch_idx < out_channels; out_ch_idx++) {
    l16ui   a8,a1,176                   # [5]  id:620 out_channels+0x0
    s32i    a9,a1,72                    # [5]  gra_spill_temp_173
    blti    a8,1,.outer_ch_loop_end

    movi.n  a7,0

.out_ch_loop:   # 0xf3e
    l32i    a8,a1,200                   # [4]  gra_spill_temp_165, buffer_ptr
    ee.zero.qacc                        # [3]
    ee.zero.q                       q5  #
    l32i    a10,a1,52                   # [1]  gra_spill_temp_168, in_channels-7
    l32i    a9,a1,48                    # [1]  gra_spill_temp_167, in_channels/8 - 1
    # USAR-based filter load for unaligned access
    ee.ld.128.usar.ip               q7,a5,16
    ee.ld.128.usar.ip               q6,a5,0
    addi                            a5,a5,-8     # net advance = 8
    ee.src.q                        q7,q7,q6
    ee.vld.128.ip                   q0,a8,16
    ee.vld.128.ip                   q1,a8,16
    ee.vcmp.lt.s8                   q6,q7,q5
    ee.vzip.8                       q7,q6

    ee.vsmulas.s16.qacc.ld.incp     q2,a8,q0,q7,0
    ee.vsmulas.s16.qacc.ld.incp     q3,a8,q1,q7,1
    ee.vsmulas.s16.qacc.ld.incp     q0,a8,q2,q7,2
    ee.vsmulas.s16.qacc.ld.incp     q1,a8,q3,q7,3
    ee.vsmulas.s16.qacc.ld.incp     q2,a8,q0,q7,4
    ee.vsmulas.s16.qacc.ld.incp     q3,a8,q1,q7,5
    blti    a10,8,.inner_loop_end           # [16]

    loopgtz a9,.inner_loop_end  # [3]

    ee.vsmulas.s16.qacc.ld.incp q0,a8,q2,q7,6   # [0*II+0]  id:657
    ee.vsmulas.s16.qacc.ld.incp q1,a8,q3,q7,7   # [0*II+1]  id:658
    # USAR-based filter load for unaligned access
    ee.ld.128.usar.ip           q7,a5,16
    ee.ld.128.usar.ip           q6,a5,0
    addi                        a5,a5,-8
    ee.src.q                    q7,q7,q6
    ee.vcmp.lt.s8               q6,q7,q5
    ee.vzip.8                   q7,q6
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q7,0   # [0*II+4]  id:660
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q7,1   # [0*II+5]  id:661
    ee.vsmulas.s16.qacc.ld.incp q0,a8,q2,q7,2   # [0*II+6]  id:662
    ee.vsmulas.s16.qacc.ld.incp q1,a8,q3,q7,3   # [0*II+7]  id:663
    ee.vsmulas.s16.qacc.ld.incp q2,a8,q0,q7,4   # [0*II+8]  id:664
    ee.vsmulas.s16.qacc.ld.incp q3,a8,q1,q7,5   # [0*II+9]  id:665
.inner_loop_end:    # 0xfaf

    ee.vsmulas.s16.qacc q2,q7,6     # [2]
    ee.vsmulas.s16.qacc q3,q7,7     # [3]

 # store qacc registers and re-arrange data for low 16 bits

    ee.st.qacc_l.l.128.ip   a3,16       # [5]  id:668
    ee.st.qacc_l.h.32.ip    a3,-16        # [6]  id:669
    l32i.n     a10, a1, 0
    l32i.n     a11, a1, 5
    l32i.n     a12, a1, 10
    l32i.n     a13, a1, 15
    ee.movi.32.q    q0, a10, 0
    ee.movi.32.q    q0, a11, 1
    ee.movi.32.q    q0, a12, 2
    ee.movi.32.q    q0, a13, 3

    ee.st.qacc_h.l.128.ip   a3,16       # [5]  id:668
    ee.st.qacc_h.h.32.ip    a3,-16        # [6]  id:669
    l32i.n     a10, a1, 0
    l32i.n     a11, a1, 5
    l32i.n     a12, a1, 10
    l32i.n     a13, a1, 15
    ee.movi.32.q    q4, a10, 0
    ee.movi.32.q    q4, a11, 1
    ee.movi.32.q    q4, a12, 2
    ee.movi.32.q    q4, a13, 3

    l32i                a9,a1,160       # [17]  gra_spill_temp_170, bias
    l32i                a10,a1,72       # [0]  gra_spill_temp_173, bias_ptr

 # add bias
    beqz.n          a9,.no_bias
    ee.vldbc.32.ip  q6,a10,4
    s32i            a10,a1,72           # [3]  gra_spill_temp_173, bias_ptr
    ee.vadds.s32    q0,q0,q6            # [4]
    ee.vadds.s32    q4,q4,q6            # [5]
.no_bias:   # 0x102e

    # Same eight row terms for every output channel. q0 is positions 0..3
    # and q4 is positions 4..7, both still pre-requantize.
    l32i    a11,a1,204
    beqz    a11,.no_row_addend
    l32i    a10,a1,40                   # i_out, a multiple of 8
    addx4   a11,a10,a11
    ee.vld.128.ip   q6,a11,16           # row_addend[0..3]
    ee.vld.128.ip   q5,a11,16           # row_addend[4..7]
    ee.vadds.s32    q0,q0,q6
    ee.vadds.s32    q4,q4,q5
.no_row_addend:

    l32i.n  a11,a6,0                    # [1]  id:696
    l32i.n  a10,a2,0                    # [3]  id:695
    .global esp_nn_multiply_by_quantized_mult_asm_esp32s3
    call8   esp_nn_multiply_by_quantized_mult_asm_esp32s3   # [4]  esp_nn_multiply_by_quantized_mult_asm_esp32s3

    l32i.n  a10,a2,0                    # [0]  id:697, mult
    l32i.n  a11,a6,0                    # [2]  id:698, shift
    mv.qr   q5,q0
    mv.qr   q0,q4
    call8   esp_nn_multiply_by_quantized_mult_asm_esp32s3   # [5]  esp_nn_multiply_by_quantized_mult_asm_esp32s3

    addi.n  a6,a6,4                     # out_shift_ptr++
    addi.n  a2,a2,4                     # out_mult_ptr++
    addi    a9,a1,180                   # [7]
    addi    a10,a1,192                  # [5]
    addi    a8,a1,196                   # [6]

# load broadcast, activation and out_offset
    ee.vldbc.32     q4,a9               # [14]  id:699 out_offset
    ee.vldbc.32     q2,a10              # [11]  id:700 activation_min
    ee.vldbc.32     q3,a8               # [12]  id:701 activation_max

# add offset
    ee.vadds.s32    q1,q0,q4            # [17]
    ee.vadds.s32    q0,q5,q4            # [22]

 # activation
    ee.vmin.s32     q1,q1,q3            # [19]
    ee.vmax.s32     q1,q1,q2            # [21]
    ee.vmin.s32     q0,q0,q3            # [23]
    ee.vmax.s32     q0,q0,q2            # [24]

    l16ui           a9,a1,176           # [33]  out_channels

# unzip and store
    ee.vunzip.16    q0,q1               # [25]
    ee.vst.128.ip   q0,a3,0             # [26]  id:702, scratch_buf

 # a4 = out_data, out_channels = a1+176

    l8ui    a14,a1,0                    # [27]
    l8ui    a11,a1,2                    # [30]  scratch_buf+2
    add     a10,a4,a9
    s8i     a14,a4,0                    # [28], out_data
    s8i     a11,a10,0                   # [31], out_data + out_channels

    l8ui    a14,a1,4                    # [32]  scratch_buf+4
    l8ui    a11,a1,6                    # [37]  scratch_buf+6
    add     a12,a10,a9
    add     a10,a12,a9
    s8i     a14,a12,0                   # [28]
    s8i     a11,a10,0                   # [31]

    l8ui    a14,a1,8                    # [41]  scratch_buf+8
    l8ui    a11,a1,10                   # [47]  scratch_buf+10
    add     a12,a10,a9
    add     a10,a12,a9
    s8i     a14,a12,0                   # [28]
    s8i     a11,a10,0                   # [31]

    l8ui    a14,a1,12                   # [51]  scratch_buf+12
    l8ui    a11,a1,14                   # [55]  scratch_buf+14
    add     a12,a10,a9
    add     a10,a12,a9
    s8i     a14,a12,0                   # [28]
    s8i     a11,a10,0                   # [31]

    addi.n  a4,a4,1                     # [29] out_data++;
    addi.n  a7,a7,1
    bne     a7,a9,.out_ch_loop

.outer_ch_loop_end:

    subx8   a11,a9,a9                   # (7 * out_channels);
    l32i    a10,a1,76                   # [1]  gra_spill_temp_174, in_channels * 8
    l32i    a15,a1,40                   # [4]  gra_spill_temp_165
    l32i    a9,a1,68                    # [2]  gra_spill_temp_172
    l32i    a8,a1,80                    # [0]  gra_spill_temp_175, size-7
    add.n   a4,a4,a11                   # [5] out_data += (7 * out_channels);
    addi.n  a15,a15,8
    s32i    a15,a1,40                   # [7]  gra_spill_temp_165
    add.n   a9,a9,a10                   # [8]
    s32i    a9,a1,68                    # [9]  gra_spill_temp_172
    blt     a15,a8,.outer_loop          # [10]

 # check if leftover
    l32i    a15,a1,40
    l32i    a13,a1,84                   # [1]  gra_spill_temp_176, size
    l32i    a8,a1,44                    # [0]  gra_spill_temp_166, in_channels
    bge     a15, a13, .return_function  # no leftover

// This block below processes one input channel line at a time.
.process_leftover:
    l32i    a15,a1,40                   # [1]  gra_spill_temp_165, i_out
    l32i    a14,a1,56                   # [2]  gra_spill_temp_169, input
    mull    a15,a15,a8                  # [3] in_channels * i_out
    addi.n  a8,a8,-1                    # [4] in_channels - 1
    add.n   a14,a14,a15                 # [5] input_ptr = in_channels * i_out + input
    srai    a8,a8,3                     # [6] iterations, (in_channels - 1) >> 3
    s32i    a8,a1,36                    # [7]  gra_spill_temp_164, iterations
    s32i    a14,a1,68                   # [8]  gra_spill_temp_172, in_channels * i_out + input
    addi            a12,a1,64
    ee.vldbc.16     q4,a12              # [8]  id:716 input_offset

.leftover_outer_loop:

    l32i    a15,a1,184                  # [0]  out_shift
    l32i    a2,a1,188                   # [1]  out_mult
    l32i    a8,a1,60                    # [3]  gra_spill_temp_170, filter_data
    l32i    a5,a1,160                   # [0]  gra_spill_temp_170, bias
    movi.n  a11,0                       # [2]

.leftover_out_ch_loop:

    ee.zero.qacc                            # [0]
    ee.zero.q       q3                      # [1]
    l32i.n          a9,a1,68                # [4]  gra_spill_temp_172, input_ptr
    l32i            a10,a1,36               # [1]  gra_spill_temp_164, iterations, (in_channels - 1) >> 3
    ee.vld.l.64.ip          q0,a9,8         # [7]  id:717, input
    # USAR-based filter load for unaligned access
    ee.ld.128.usar.ip       q1,a8,16
    ee.ld.128.usar.ip       q7,a8,0
    addi                    a8,a8,-8
    ee.src.q                q1,q1,q7
    ee.vcmp.lt.s8           q6,q0,q3
    ee.vcmp.lt.s8           q7,q1,q3
    ee.vzip.8               q0,q6
    ee.vzip.8               q1,q7
    ee.vadds.s16            q0,q0,q4  # [11]  id:718, add offset

    loopgtz a10,.leftover_inner_loop_end        # [3]

    ee.vmulas.s16.qacc          q0,q1  # mula(q0,q1)
    ee.vld.l.64.ip              q0,a9,8         # load 8 input values
    # USAR-based filter load for unaligned access
    ee.ld.128.usar.ip           q1,a8,16
    ee.ld.128.usar.ip           q7,a8,0
    addi                        a8,a8,-8
    ee.src.q                    q1,q1,q7
    ee.vcmp.lt.s8               q2,q0,q3        # sign
    ee.vcmp.lt.s8               q7,q1,q3
    ee.vzip.8                   q0,q2           # 16 bit input
    ee.vzip.8                   q1,q7           # 16 bit filter
    ee.vadds.s16                q0,q0,q4        # add offset
.leftover_inner_loop_end:   # 0x1262

# re-arrange data from qacc in 32 bit q registers
    ee.vmulas.s16.qacc      q0,q1       # [3]
    ee.st.qacc_l.l.128.ip   a3,16       # [5]  id:722
    ee.st.qacc_l.h.32.ip    a3,0        # [6]  id:723
    l8ui    a10,a1,5                    # [11]  scratch_buf+5
    l8ui    a12,a1,6                    # [10]  scratch_buf+6
    l16ui   a14,a1,10                   # [8]  scratch_buf+10
    l8ui    a9,a1,15                    # [7]  scratch_buf+15
    l8ui    a13,a1,16                   # [9]  scratch_buf+16
    s8i     a10,a1,2                    # [12]  scratch_buf+2
    s8i     a12,a1,3                    # [13]  scratch_buf+3
    s16i    a14,a1,4                    # [15]  scratch_buf+4
    s8i     a9,a1,6                     # [16]  scratch_buf+6
    s8i     a13,a1,7                    # [14]  scratch_buf+7

    ee.st.qacc_h.l.128.ip   a3,16       # [17]  id:724
    ee.st.qacc_h.h.32.ip    a3,-32      # [18]  id:725
    l16ui   a13,a1,16                   # [30]  scratch_buf+16
    l8ui    a14,a1,21                   # [23]  scratch_buf+21
    l8ui    a9,a1,22                    # [22]  scratch_buf+22
    l16ui   a10,a1,26                   # [21]  scratch_buf+26
    s16i    a13,a1,8                    # [31]  scratch_buf+8
    l8ui    a12,a1,31                   # [20]  scratch_buf+31
    l8ui    a13,a1,32                   # [19]  scratch_buf+32
    s8i     a14,a1,10                   # [24]  scratch_buf+10
    s8i     a9,a1,11                    # [25]  scratch_buf+11
    s16i    a10,a1,12                   # [26]  scratch_buf+12
    s8i     a12,a1,14                   # [27]  scratch_buf+14
    s8i     a13,a1,15                   # [28]  scratch_buf+15
    movi.n  a12,16

# get data now
    ee.vld.128.ip       q0,a3,0
    ee.srcmb.s16.qacc   q1,a12,0
    ee.vzip.16          q0,q1

    ee.vadds.s32    q0,q0,q1
    ee.movi.32.a    q0,a10,3
    ee.movi.32.a    q0,a9,2
    ee.movi.32.a    q0,a14,0
    add             a9,a9,a10
    ee.movi.32.a    q0,a10,1
    add             a14,a14,a10
    add             a14,a14,a9

# a14 contains conv_out
    l32i    a10,a1,204
    beqz    a10,.leftover_no_row_addend
    l32i    a12,a1,40                   # i_out
    addx4   a10,a12,a10
    l32i    a10,a10,0
    add     a14,a14,a10
.leftover_no_row_addend:
    l32i    a9,a1,160                   # [43]  gra_spill_temp_170, bias ptr
    l32i.n  a6,a15,0                    # [44]  id:730, shift
    beqz.n  a9,.leftover_multiply_by_quant_mult             # [45]

# load and add bias
    l32i.n  a9,a5,0
    add.n   a14,a14,a9

.leftover_multiply_by_quant_mult:   # 0x12e7
    l32i.n  a9,a2,0                 # [0]  id:729, mult
    movi.n  a10,0                   # [1]
    max     a10,a6,a10              # [2]  left_shift
    ssl     a10                     # [3]
    sll     a14,a14                 # [4] (value << left_shift)

    sub     a7,a10,a6               # right_shift

    l32r    a13,.nudge_val
    mulsh   a12,a9,a14
    mull    a14,a9,a14
    ssai    31

    addi.n  a2,a2,4                 # [0] mult
    addi.n  a15,a15,4               # [1] shift
    addi.n  a5,a5,4                 # [2] bias
    addi.n  a11,a11,1               # [3]

    add     a13,a14,a13             # low part
    saltu   a14,a13,a14
    add     a9,a12,a14              # high part
    src     a12,a9,a13

    blti    a7,1,.leftover_skip_div_by2

    addi.n  a14,a7,-1
    ssl     a14
    movi.n  a10,1
    sll     a10,a10                     # 1 << (exponent - 1)
    extui   a14,a12,31,1
    ssr     a7
    sub     a10,a10,a14                 # 1 << (exponent - 1) - (val < 0)
    add     a12,a12,a10                 # val += to_add
    sra     a12,a12

.leftover_skip_div_by2:
    l32i    a10,a1,180                  # [26]  id:733 out_offset+0x0
    l32i    a9,a1,192                   # [29]  id:732 activation_min+0x0
    l16ui   a13,a1,176                  # [5]  id:620 out_channels+0x0
    l32i    a14,a1,196                  # [31]  id:731 activation_max+0x0

// add offset, apply activation and store
    add.n   a10,a10,a12
    max     a9,a9,a10
    min     a14,a14,a9
    s8i     a14,a4,0
    addi.n  a4,a4,1

    bne     a11,a13,.leftover_out_ch_loop

    l32i    a15,a1,44                   # [0]  gra_spill_temp_166, in_channels
    l32i    a14,a1,68                   # [1]  gra_spill_temp_172, input_ptr
    l32i    a13,a1,40                   # [2]  gra_spill_temp_165, i_out
    l32i    a12,a1,84                   # [3]  gra_spill_temp_176, size
    addi.n  a13,a13,1                   # [4]
    s32i    a13,a1,40                   # [5]  gra_spill_temp_165, i_out
    add     a14,a14,a15                 # [7]  input_ptr += in_channels
    s32i    a14,a1,68                   # [8]  gra_spill_temp_172, input_ptr
    blt     a13,a12,.leftover_outer_loop

.return_function:
    retw.n              # [9]

.prepare_leftover:
    l32i    a8,a1,44                    # [0]  gra_spill_temp_166, in_channels
    movi.n  a15,0
    s32i    a15,a1,40                   # [7]  gra_spill_temp_165, i_out
    j   .process_leftover

    .size   esp_nn_conv_s8_mult8_1x1_esp32s3, . - esp_nn_conv_s8_mult8_1x1_esp32s3
