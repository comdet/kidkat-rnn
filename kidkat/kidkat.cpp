#include <stdio.h>
#include <string.h>
#include <string>
#include <float.h>
#include <math.h>
#include <cmath>
#include "esp_system.h"
#include "kidbright32.h"
#include "sound.h"

#include "kidkat.h"
#include "Common.h"
#include "Network.h"

#include <driver/adc.h>
#include "mpu6050.hpp"
#include <chrono>

#define __DEBUG__

#ifdef __DEBUG__
#include "esp_log.h"
static const char *TAG = "KidKat";
#endif

#define READ_OUT_SEC 2
#define SAMPLE_GAIN_SEC 2
#define BLANK_INIT_WAV 2000 //<<<<<<<< must remove
//===================//
//uint32_t esp_random()
long random_t(long howsmall, long howbig) {
    if(howsmall >= howbig) {
        return howsmall;
    }
    long diff = howbig - howsmall;
    return (esp_random() % diff) + howsmall;
}

static void gen_noise_normal(Neuron::Values &input,int factor=100)
{
    for(int i=0;i<input.size();i++){
        input[i] = input[i] + ((((esp_random() % 1000)/1000.0) * (float)factor) - (factor/2));
    }
}

static void gen_noise(Neuron::Values &input)
{
    float factor = random_t(110,200)/100.0;
    for(int i=0;i<input.size();i++){        
        input[i] = input[i] +  ((((esp_random() % 1000)/1000.0) * (float)factor) - (factor/2));
    }
}

static void gen_section_noise(Neuron::Values &input)
{
    int square_sec = random_t(input.size()/6,input.size()/2);
    int st = random_t(0,square_sec);
    float sq_factor =  random_t(120,200)/100.0;
    for(int i=0;i<input.size();i++){
        if(i>st && i < (st+square_sec)){
            input[i] = input[i] * sq_factor;    
        }  
    }
}

static void gen_squre(Neuron::Values &input)
{
    int square_sec = random_t(input.size()/6,input.size()/2);
    int st = random_t(0,square_sec);
    float factor =  random_t(120,200)/100.0;
    for(int i=0;i<input.size();i++){
        if(i>st && i < (st+square_sec)){
            input[i] = input[i] + ((((esp_random() % 1000)/1000.0) * (float)factor) - (factor/2));    
        }
    }
}

static Value crossEntropyErrorCost(const Neuron::Values &targets, const Neuron::Values &outputs)
{
    Value sumError = 0.0;
    for (size_t i = 0; i < outputs.size(); ++i){
        sumError += log(outputs[i]) * targets[i];
    }
    return sumError / (float)outputs.size();
}

static Value meanSquaredErrorCost(const Neuron::Values &targets, const Neuron::Values &outputs)
{
    Value cost = 0.0;
    for (size_t i = 0; i < outputs.size(); ++i){
        cost += (pow(targets[i] - outputs[i], 2.0));
    }
    return cost / outputs.size();
}

KidKat::KidKat() { }

//======== override =========//
void KidKat::init(void) {    
    //=======init i2s========//
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC1_CHANNEL_3,ADC_ATTEN_MAX);    
    printf("Configured microphone.\n");
    //=======================//
    mpu = new MPU6050(GPIO_NUM_5, GPIO_NUM_4, I2C_NUM_1);
    if(!mpu->init()) {
        printf("MPU6050 Init Fail!\n");
    }

    this->initialized = true;
    this->error = false;
    //esp_log_level_set("*", ESP_LOG_VERBOSE);   

}

int KidKat::prop_count(void) {
    // not supported
    return 0;
}

bool KidKat::prop_name(int index, char *name) {
    // not supported
    return false;
}

bool KidKat::prop_unit(int index, char *unit) {
    // not supported
    return false;
}

bool KidKat::prop_attr(int index, char *attr) {
    // not supported
    return false;
}

bool KidKat::prop_read(int index, char *value) {
    // not supported
    return false;
}

bool KidKat::prop_write(int index, char *value) {
    // not supported
    return false;
}
void KidKat::process(Driver *drv) {
    
}

void KidKat::build_deep_network(char * name,int inputNum,const std::vector<int> &hiddenLayersSizes,int outputSize,InputSource source,int sample_rate)
{
    /*while(true){
        float ax = -mpu->getAccX();
        float ay = -mpu->getAccY();
        float az = -mpu->getAccZ();
        float mag = sqrt(ax*ax + ay*ay + az*az);
        printf("%f\n", mag);
        vTaskDelay(20 / portTICK_RATE_MS);
    }*/
    this->num_input = inputNum;
    this->num_output = outputSize;
    Neuron::Values input_normal(inputNum);
    Neuron::Values input_neg(inputNum);
    this->net_normal_input = input_normal;
    this->net_negative_input = input_neg;    
    this->input_source = source;
    this->delay_sample_rate = (int)(1000.0/sample_rate);
    network = Network::Prefabs::feedForward(name, inputNum, hiddenLayersSizes, outputSize);
    printf("Build Deep Network success!\n");
}

void KidKat::build_lstm(char *name,int inputNum,int numberOfCell,int outputNum,InputSource source,int sample_rate)
{
    this->num_input = inputNum;
    this->num_output = outputNum;
    this->input_source = source;
    Neuron::Values input_normal(inputNum);
    Neuron::Values input_neg(inputNum);
    this->net_normal_input = input_normal;
    this->net_negative_input = input_neg;  
    this->delay_sample_rate = (int)(1000.0/sample_rate);
    network = Network::Prefabs::longShortTermMemory(name, inputNum, {numberOfCell}, outputNum);
}
Value KidKat::read_data_and_delay()
{
    Value result = 0.0;
    if(this->input_source == MICROPHONE){
        result = adc1_get_raw(ADC1_CHANNEL_3);
    }else if(this->input_source == ACC){
        float ax = -mpu->getAccX();
        float ay = -mpu->getAccY();
        float az = -mpu->getAccZ();
        result = sqrt(ax*ax + ay*ay + az*az);
        //printf("%f\n", result);
    }else if(this->input_source == GYRO){
        float gx = mpu->getGyroX();
        float gy = mpu->getGyroY();
        float gz = mpu->getGyroZ();
        result = sqrt(gx*gx + gy*gy + gz*gz);
    }else if(this->input_source == COMPASS){
        result = 0;
    }else if(this->input_source == BARO){
        result = 0;
    }else if(this->input_source == ADC1_CH0){
        result = adc1_get_raw(ADC1_CHANNEL_0);
    }else if(this->input_source == ADC1_CH6){
        result = adc1_get_raw(ADC1_CHANNEL_6);
    }else if(this->input_source == ADC1_CH7){
        result = adc1_get_raw(ADC1_CHANNEL_7);
    }else if(this->input_source == ADC1_CH4){
        result = adc1_get_raw(ADC1_CHANNEL_4);
    }else if(this->input_source == ADC1_CH5){
        result = adc1_get_raw(ADC1_CHANNEL_5);
    }
    vTaskDelay(this->delay_sample_rate / portTICK_RATE_MS); 
    return result;
}
void KidKat::train_network(int32_t num_epoch,Value acceptable_error)
{
    static const Value kTrainingRate = 0.01f;
    //Neuron::Values
    printf("init free heap = %d\n", esp_get_free_heap_size());
    //dec    
    //read out for N SEC
    for(int i=0;i<(READ_OUT_SEC*1000)/this->delay_sample_rate;i++)
    {
        read_data_and_delay();
    }
    //get sample gain
    uint max_loop = (SAMPLE_GAIN_SEC*1000)/this->delay_sample_rate;    
    for(uint i=0;i<max_loop;i++)
    {
        center_gain += read_data_and_delay() / (float)max_loop;        
    }
    printf("center_gain = %f\n", center_gain);
    //fill smoothing
    for(int i=0;i<SMOOTHING;i++){
        moving[i] = read_data_and_delay();
    }

    uint32_t epoch = 0;  
    while(1)
    {
        for(int i=0;i<this->num_input;i++){
            float avgRaw = 0;
            for(int i=0;i<SMOOTHING-1;i++){            
                avgRaw += moving[i] / (float)SMOOTHING;
                moving[i] = moving[i+1];
            }
            moving[SMOOTHING-1] = read_data_and_delay();
            avgRaw += moving[SMOOTHING-1]/(float)SMOOTHING;
            avgRaw = avgRaw-center_gain;
            net_normal_input[i] = avgRaw;
            net_negative_input[i] = avgRaw;
        }
        //random noise
        int r = random_t(0,4);
        //if(r == 0){
        //    gen_noise(net_negative_input);
        //    for(int i=0;i<net_negative_input.size();i++){
        //        printf("%f\n", net_negative_input[i]);
        //    }
        //}else if(r == 1){
        //    gen_section_noise(net_negative_input);
        //}else if(r == 2){
        //    gen_squre(net_negative_input);
        //}
        gen_noise_normal(net_negative_input,(esp_random()%1000)+50);
        
        /*for(int i=0;i<net_normal_input.size();i++){
            printf("%f\n",net_normal_input[i]);
        }*/
        //feed 
        const auto result1 = network->feed(net_normal_input);
        network->train(kTrainingRate, {1.0,0.0}); //normal data
        const auto result2 = network->feed(net_negative_input);
        network->train(kTrainingRate, {0.0,1.0}); //error data        
        //printf("%d > [%f],[%f],[%f],[%f]\n",epoch, result1[0],result1[1],result2[0],result2[1]);        
        //Value error = (1.0f - result1[0])+(result1[1])+(result2[0])+(1.0f-result2[1]-1)/4.0f;
        Value error = crossEntropyErrorCost({1.0,0.0},result1);
        error += crossEntropyErrorCost({0.0,1.0},result2);
        error = -1.0 * error / 2.0;
        epoch++;
        if(num_epoch > 0 && epoch > num_epoch){
            printf("Train finished at %d epoch\n",epoch);
            break;
        }
        if(acceptable_error < 1 && error < acceptable_error){
            printf("Train finished at %f error at %d epoch\n", error, epoch);
            break;
        }
        if(epoch % 100 == 0){
            printf("[%d]\t Error = %f\n",epoch, error);
        }
    }
}
void KidKat::train_network_until_error(Value acceptable_error)
{
    //check heap before user    
    printf("init free heap = %d\n", esp_get_free_heap_size());
    train_network(-1,acceptable_error);
    printf("finished free heap = %d\n", esp_get_free_heap_size());
}

void KidKat::train_network_n_epoch(uint32_t num_epoch)
{
    printf("init free heap = %d\n", esp_get_free_heap_size());
    train_network(num_epoch,1);
    printf("finished free heap = %d\n", esp_get_free_heap_size());    
}

Value KidKat::classify_input()
{    
    //dec    
    Neuron::Values net_normal_input(this->num_input);
    for(int i=0;i<this->num_input;i++){
        float avgRaw = 0;
        for(int i=0;i<SMOOTHING-1;i++){            
            avgRaw += moving[i] / (float)SMOOTHING;
            moving[i] = moving[i+1];
        }
        moving[SMOOTHING-1] = read_data_and_delay();
        avgRaw += moving[SMOOTHING-1]/(float)SMOOTHING;
        avgRaw = avgRaw-center_gain;
        net_normal_input[i] = avgRaw;
        net_negative_input[i] = avgRaw;
    }
    const auto result = network->feed(net_normal_input);
    Value error = crossEntropyErrorCost({1.0,0.0},result);
    error = -1.0 * error / 2.0;
    return error;
}

void KidKat::classify_input_until_error(Value error_threshold)
{
    for(int i=0;i<(READ_OUT_SEC*1000)/this->delay_sample_rate;i++)
    {
        read_data_and_delay();
    }
    while(1){
        Value error = (1.0-classify_input())*100.0;
        printf("Confidential = %f\n", error);
        if(error < error_threshold){
            return;
        }
    }
    return;
}

void KidKat::classify_input_by_estimate_acc(uint16_t time)
{
    for(int i=0;i<(READ_OUT_SEC*1000)/this->delay_sample_rate;i++)
    {
        read_data_and_delay();
    }
    if(max_acc < -999 && min_acc > 999){
        //std::chrono::high_resolution_clock::time_point startTime    
        time = time * (1000.0 / this->delay_sample_rate);    
        for(int i = 0; i < time; i++){
            Value confi = (1.0-classify_input())*100.0;
            if(confi > max_acc){
                max_acc = confi;
                printf("Max Confidential = %f\n", confi);
            }
            if(confi < min_acc){
                min_acc = confi;
                printf("Min Confidential = %f\n", confi);
            }
        }
    }
    while(1){        
        Value confidential = (1.0-classify_input())*100.0;
        printf("Confidential = %f\n", confidential);
        if(confidential > max_acc || confidential < min_acc){
            return;
        }
    }
}