/**
  ******************************************************************************
  * @file    delay.c
  * @brief   This file contains the functions to create a delay using SysTick
  * @author  doublehan07
  * @version V1.0
  * @date    2023-04-29
  ******************************************************************************
  */

#include "delay.h"

static __IO uint32_t s_tick; //s_tick is set for the user's delay function.
static __IO uint16_t current_timestamp = 0; // current_timestamp is used to store the timestamp in milliseconds.

/**
 * @brief  Decrement the tick count
 * @note   This function should be called every 1 ms in the SysTick interrupt function.
 * @retval None
 */
void dbh_DecTick(void)
{
	if (s_tick != 0)
	{
		s_tick--;
	}
}

/**
 * @brief  Get the current tick count
 * @retval The current tick count
 */
uint32_t GetTick(void)
{
	return s_tick;
}

/**
 * @brief  Set the tick count
 * @param  set_tick: The value to set the tick count to
 * @retval None
 */
void SetTick(uint32_t set_tick)
{
	s_tick = set_tick;
}

/**
 * @brief  Delay for a specified number of milliseconds
 * @param  delay_time_ms: The delay time in milliseconds
 * @retval None
 *
 * This function creates a delay by using a loop that waits until the tick count reaches zero.
 * The tick count is decremented every millisecond in the SysTick interrupt function.
 */
void dbh_DelayMS(uint32_t delay_time_ms)
{
	SetTick(delay_time_ms);
	while (GetTick() != 0);
}

/**
 * @brief  Increment the timestamp in milliseconds
 * @note   This function should be called every 1 ms in the SysTick interrupt function.
 * @retval None
 *
 * This function increments the current timestamp and wraps it around if it exceeds 65535.
 */
void dbh_IncTimestampInMS(void)
{
    current_timestamp++;
    if (current_timestamp > 65535)
    {
        current_timestamp = 0;
    }
}

/**
 * @brief  Get the current timestamp in milliseconds
 * @retval The current timestamp
 *
 * This function returns the current timestamp, which is incremented every millisecond.
 */
uint16_t dbh_GetTimestamp(void)
{
    return current_timestamp;
}
