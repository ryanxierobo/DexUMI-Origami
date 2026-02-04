/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "adc.h"
#include "dma.h"
#include "iwdg.h"
#include "spi.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "ads1256.h"
#include "delay.h"
#include "fsr.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define BUILD_FOR_XHAND
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
#ifdef BUILD_FOR_XHAND
__IO uint32_t data[16] = {0};
#else
__IO uint32_t data[10] = {0};
#endif

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  uint8_t i = 0;
  uint16_t checksum = 0;
  // uint8_t device = 1;

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_SPI1_Init();
  MX_USART1_UART_Init();
  MX_IWDG_Init();
  MX_ADC_Init();
  /* USER CODE BEGIN 2 */
  dbh_ADS1256_Init(0); // Initialize the ADS1256
  dbh_ADS1256_Init(1); // Initialize the ADS1256

  HAL_GPIO_WritePin(LED1_GPIO_Port, LED1_Pin, GPIO_PIN_SET); // Turn on the LED1

  //ADC initialization
	// HAL_ADCEx_Calibration_Start(&hadc);
	// HAL_ADC_Start_DMA(&hadc, (uint32_t*)&_u16ADC_Value, 50*5);
	// HAL_ADC_Start(&hadc);

  HAL_GPIO_WritePin(LED2_GPIO_Port, LED2_Pin, GPIO_PIN_SET); // Turn on the LED2

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    // Read the data from the ADS1256
    data[0] = 0x55AA;
    #ifdef BUILD_FOR_XHAND
    for (i = 1; i <= 14; i++)
    {
      // i = 0 - 7, for the first ADS1256
      // i = 8 - 15, for the second ADS1256
      if (i < 13)
      {
        dbh_ADS1256_SelectChannel((i-1)%8, (i-1)/8);
        data[i] = dbh_ADS1256_ReadData((i-1)/8);
      }
      else
      {
        dbh_ADS1256_SelectChannel((i+1)%8, (i+1)/8);
        data[i] = dbh_ADS1256_ReadData((i+1)/8);
      }
      // voltage[i] = (float)data[i] * 5.0 / 0x7FFFFF;
      // voltage[i] = data[i] * 0.000000596;
    }
    #else
    for (i = 1; i <= 8; i++)
    {
      if (i < 7)
      {
        dbh_ADS1256_SelectChannel((i-1)%8, (i-1)/8);
        data[i] = dbh_ADS1256_ReadData((i-1)/8);
      }
      else
      {
        dbh_ADS1256_SelectChannel((i+7)%8, (i+7)/8);
        data[i] = dbh_ADS1256_ReadData((i+7)/8);
      }
      // voltage[i] = (float)data[i] * 5.0 / 0x7FFFFF;
      // voltage[i] = data[i] * 0.000000596;
    }
    #endif

    // Get the FSR data
    // dbh_FSR_GetADCValue();
    // data[13] = (fsr[1] << 16) | fsr[0];

    // Calculate the checksum
    checksum = 0;
    #ifdef BUILD_FOR_XHAND
    for (i = 1; i <= 14; i++)
    {
      checksum += data[i];
    }
    data[15] = (checksum << 16) | dbh_GetTimestamp();

    HAL_UART_Transmit(&huart1, (uint8_t *)data, 64, 1000); // Send the data over UART
    #else
    for (i = 1; i <= 8; i++)
    {
      checksum += data[i];
    }
    data[9] = (checksum << 16) | dbh_GetTimestamp();
    HAL_UART_Transmit(&huart1, (uint8_t *)data, 40, 1000); // Send the data over UART
    #endif
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_LSI|RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.LSIState = RCC_LSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL6;
  RCC_OscInitStruct.PLL.PREDIV = RCC_PREDIV_DIV1;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_USART1;
  PeriphClkInit.Usart1ClockSelection = RCC_USART1CLKSOURCE_PCLK1;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
