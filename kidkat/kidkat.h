#ifndef __KIDKAT_H__
#define __KIDKAT_H__


#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "esp_partition.h"

#include "freertos/queue.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_event_loop.h"
#include "driver/i2s.h"
#include "driver/gpio.h"
//----- http req -------//
#include "esp_http_client.h"
#include "driver.h"
#include "device.h"
#include "Network.h"
#include "UnrolledNetwork.h"

#include "mpu6050.hpp"

#define SMOOTHING 128

enum InputSource
{
	MICROPHONE,
	ACC,
	GYRO,
	COMPASS,
	BARO,
	ADC1_CH0,
	ADC1_CH6,
	ADC1_CH7,
	ADC1_CH4,
	ADC1_CH5
};
class KidKat : public Device {
	private:		
		char key;
		Network::Ptr network;
		int num_input;
		int num_output;
		int delay_sample_rate;
		InputSource input_source;
		float center_gain = 0;
		uint16_t moving[SMOOTHING];
    	
    	Neuron::Values net_normal_input;
    	Neuron::Values net_negative_input;

    	MPU6050 *mpu;
		
		Value read_data_and_delay();

		Value max_acc = -999999.0;
    	Value min_acc = 999999.0;
	public:
		// overrideS
		void init(void);
		void process(Driver *drv);
		int prop_count(void);
		bool prop_name(int index, char *name);
		bool prop_unit(int index, char *unit);
		bool prop_attr(int index, char *attr);
		bool prop_read(int index, char *value);
		bool prop_write(int index, char *value);
		// constructor
		KidKat();
		void build_deep_network(char * name,int inputNum,const std::vector<int> &hiddenLayersSizes,int outputSize,InputSource source,int sample_rate);
		void build_lstm(char *name,int inputNum,int numberOfCell,int outputNum,InputSource source,int rate);

		void train_network(int32_t num_epoch,Value acceptable_error);		
		void train_network_until_error(Value acceptable_error);		
		void train_network_n_epoch(uint32_t num_epoch);

		Value classify_input();
		void classify_input_until_error(Value error_threshold);
		void classify_input_by_estimate_acc(uint16_t time);

};

#endif
