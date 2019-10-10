var KID_KAT_HUE = 75;

Blockly.Blocks["kidkat.build_deep_network"] = {
	init: function() {
		this.appendDummyInput()
			.appendField("สร้างโมเดล Neural Network");

		this.appendDummyInput()
			.appendField("ชื่อ")
			.appendField(new Blockly.FieldTextInput('mynetwork'), 'NETWORK_NAME');

		this.appendDummyInput()
			.appendField("จำนวนข้อมูลน้ำเข้า")
			.appendField(new Blockly.FieldTextInput('32'), 'NN_INPUT');

		this.appendDummyInput("รับข้อมูลจาก")
			.appendField("รับข้อมูลจาก")
			.appendField(new Blockly.FieldDropdown([
				["Microphone", "MICROPHONE"],
				["Accelerometer", "ACC"],
				["Gyroscope", "GYRO"],				
				["ADC1_CH0", "ADC1_CH0"],				
				["ADC1_CH6", "ADC1_CH6"],
				["ADC1_CH7", "ADC1_CH7"],
				["ADC1_CH4", "ADC1_CH4"],
				["ADC1_CH5", "ADC1_CH5"]
			]), 'input_source_nn');

		this.appendDummyInput()
			.appendField("ความถี่การดึงข้อมูล")
			.appendField(new Blockly.FieldNumber(200,1,1000),'nn_sample_rate')
			.appendField("ครั้งต่อวินาที");

		this.appendValueInput("HIDDEN_NUM")
			.setCheck("Number")
			.appendField("จำนวนชั้นซ่อน");

		this.setInputsInline(false);
		this.setPreviousStatement(true, null);
		this.setNextStatement(true, null);
		this.setColour(KID_KAT_HUE);
		this.setTooltip("");
		this.setHelpUrl("");
	}
};

Blockly.Blocks["kidkat.build_lstm"] = {
	init: function() {		
		this.appendDummyInput()
			.appendField("สร้างโมเดล LSTM ชื่อ")
			.appendField(new Blockly.FieldTextInput('mynetwork'), 'NETWORK_NAME');

		this.appendDummyInput()
			.appendField("จำนวนข้อมูลนำเข้า")
			.appendField(new Blockly.FieldTextInput('8'), 'NN_INPUT');

		this.appendDummyInput()
			.appendField("จำนวน Cell ")
			.appendField(new Blockly.FieldTextInput('10'), 'LSTM_CELL');

		this.appendDummyInput()
			.appendField("รับข้อมูลจาก")
			.appendField(new Blockly.FieldDropdown([
				["Microphone", "MICROPHONE"],
				["Accelerometer", "ACC"],
				["Gyroscope", "GYRO"],				
				["ADC1_CH0", "ADC1_CH0"],				
				["ADC1_CH6", "ADC1_CH6"],
				["ADC1_CH7", "ADC1_CH7"],
				["ADC1_CH4", "ADC1_CH4"],
				["ADC1_CH5", "ADC1_CH5"]
			]), 'input_source_lstm');

		this.setInputsInline(false);
		this.setPreviousStatement(true, null);
		this.setNextStatement(true, null);
		this.setColour(KID_KAT_HUE);
		this.setTooltip("");
		this.setHelpUrl("");
	}
};

Blockly.Blocks["kidkat.train_network_n_epoch"] = {
	init : function() {
		this.appendDummyInput()
			.appendField("สอนโมเดล จำนวน")
			.appendField(new Blockly.FieldNumber(1000,1,1000000),'train_epoch')
			.appendField("รอบ");

		this.setInputsInline(true);
		this.setPreviousStatement(true, null);
		this.setNextStatement(true, null);
		this.setColour(KID_KAT_HUE);
		this.setTooltip("");
		this.setHelpUrl("");
	}
};

Blockly.Blocks["kidkat.train_network_until_error"] = {
	init : function(){
		this.appendDummyInput()
			.appendField("สอนโมเดลจนกว่าค่าผิดพลาดต่ำกว่า")
			.appendField(new Blockly.FieldNumber(0.01,0.00001,1.0),"acceptable_error");

		this.setInputsInline(true);
		this.setPreviousStatement(true, null);
		this.setNextStatement(true, null);
		this.setColour(KID_KAT_HUE);
		this.setTooltip("");
		this.setHelpUrl("");
	}
};

Blockly.Blocks["kidkat.classify_input"] = {
	init : function(){
		this.appendDummyInput()
			.appendField("จำแนกข้อมูล");

		this.setInputsInline(true);
		this.setOutput(true,"Number");
		this.setPreviousStatement(false, null);
		this.setNextStatement(false, null);
		this.setColour(KID_KAT_HUE);
		this.setTooltip("");
		this.setHelpUrl("");
	}
};

Blockly.Blocks["kidkat.classify_input_by_estimate_acc"] = {
	init : function(){
		this.appendDummyInput()
			.appendField("จำแนกข้อมูลและรอจนความแม่นยำนอกเหนือจาก")
			.appendField(new Blockly.FieldNumber(5,1,9999),"measuring_acc_time")
			.appendField("วินาทีแรก");

		this.setInputsInline(true);
		this.setPreviousStatement(true, null);
		this.setNextStatement(true, null);
		this.setColour(KID_KAT_HUE);
		this.setTooltip("");
		this.setHelpUrl("");
	}
};

Blockly.Blocks["kidkat.classify_input_until_error"] = {
	init : function(){
		this.appendDummyInput()
			.appendField("จำแนกข้อมูลและรอจนความแม่นยำต่ำกว่า")
			.appendField(new Blockly.FieldNumber(90.00,0.001,99.999),"accuracy_threshold");
		this.setInputsInline(true);
		this.setPreviousStatement(true, null);
		this.setNextStatement(true, null);
		this.setColour(KID_KAT_HUE);
		this.setTooltip("");
		this.setHelpUrl("");
	}
};

Blockly.Blocks["kidkat.hidden_num"] = {
	init : function(){
		this.appendDummyInput()
			.appendField("Hidden Num : ")
			.appendField(new Blockly.FieldNumber(8,2,1024,0),'hidden_num');

		this.appendValueInput("hidden_next")
			.setCheck("Number");

		this.setOutput(true, 'Number');
		this.setInputsInline(false);
		this.setPreviousStatement(false);
		this.setNextStatement(false);
		this.setColour(KID_KAT_HUE);
		this.setTooltip("ไม่มี");
		this.setHelpUrl("ไม่มี");
	}
};