
/*

SPDX-FileCopyrightText: 2023 Espressif Systems (Shanghai) CO LTD
SPDX-License-Identifier: Apache-2.0

*/

#include "main_functions.h"

extern "C" void app_main(void) {
  setup();
  while (true) {
    loop();
  }
}
