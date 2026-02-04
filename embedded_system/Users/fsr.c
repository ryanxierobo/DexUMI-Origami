/**
  ******************************************************************************
  * @file    fsr.c
  * @brief   This file contains the functions to get the ADC value of the fsr voltage.
  * @author  doublehan07
  * @version V1.0
  * @date    2024-12-14
  ******************************************************************************
  */

 #include "fsr.h"

__IO uint16_t _u16ADC_Value[50*5]; // Array to store ADC values for 5 FSRs, each sampled 50 times, data is from ADC DMA
__IO uint16_t fsr[5] = {0, 0, 0, 0, 0}; // Array to store the average ADC values for 5 FSRs

/**
 * @brief  Get the average ADC value for each FSR
 * @note   This function calculates the average ADC value for each of the 5 FSRs
 *         by summing the values from the _u16ADC_Value array and dividing by 50.
 * @retval None
 */
void dbh_FSR_GetADCValue(void)
{
	uint32_t temp[5] = {0, 0, 0, 0, 0};

	uint8_t i = 0;

	// Sum the ADC values for each FSR
	for(i = 0; i < 50*5; i++)
	{
		temp[i%5] += _u16ADC_Value[i];
	}

	// Calculate the average ADC value for each FSR
	for(i = 0; i < 5; i++)
	{
		fsr[i] = temp[i] / 50;
	}
}
