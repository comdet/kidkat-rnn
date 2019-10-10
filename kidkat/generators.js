Blockly.JavaScript["kidkat.build_deep_network"] = function(block){
	var netname = block.getFieldValue("NETWORK_NAME");
	var input_num = block.getFieldValue("NN_INPUT");
	var input_source_nn = block.getFieldValue("input_source_nn");
	var nn_sample_rate = block.getFieldValue("nn_sample_rate");	
	var hidden_num = Blockly.JavaScript.valueToCode(block, 'HIDDEN_NUM', Blockly.JavaScript.ORDER_ATOMIC).substring(1);
	var text = 'DEV_IO.KidKat().build_deep_network((char *)"'+netname+'",'+input_num+',{'+hidden_num+'},2,'+input_source_nn+','+nn_sample_rate+');\n';
	return text;
};

Blockly.JavaScript["kidkat.build_lstm"] = function(block){
	var netname = block.getFieldValue("NETWORK_NAME");
	var input_num = block.getFieldValue("NN_INPUT");
	var lstm_cell = block.getFieldValue("LSTM_CELL");
	var input_source_lstm = block.getFieldValue("input_source_lstm");
	var lstm_sample_rate = block.getFieldValue("lstm_sample_rate");

	return ['DEV_IO.KidKat().build_lstm((char *)"'+netname+'",'+input_num+','+lstm_cell+',2,'+input_source_lstm+','+lstm_sample_rate+');\n',Blockly.JavaScript.ORDER_ATOMIC];
};

Blockly.JavaScript["kidkat.train_network_n_epoch"] = function(block){
	var num_epoch = block.getFieldValue('train_epoch');
	return 'DEV_IO.KidKat().train_network_n_epoch('+num_epoch+');\n';
}

Blockly.JavaScript["kidkat.train_network_until_error"] = function(block){
	var acceptable_error = block.getFieldValue("acceptable_error");	
	return 'DEV_IO.KidKat().train_network_until_error('+acceptable_error+');\n';
}

Blockly.JavaScript["kidkat.classify_input"] = function(block){	
	return 'DEV_IO.KidKat().classify_input();\n';
}

Blockly.JavaScript["kidkat.classify_input_by_estimate_acc"] = function(block)
{
	var measuring_acc_time = block.getFieldValue("measuring_acc_time");
	return 'DEV_IO.KidKat().classify_input_by_estimate_acc('+measuring_acc_time+');\n';
}

Blockly.JavaScript["kidkat.classify_input_until_error"] = function(block){
	var error_threshold = block.getFieldValue("accuracy_threshold");
	return 'DEV_IO.KidKat().classify_input_until_error('+error_threshold+');';
}

Blockly.JavaScript["kidkat.hidden_num"] = function(block){
	var num = ","+block.getFieldValue("hidden_num")+Blockly.JavaScript.valueToCode(block, 'hidden_next', Blockly.JavaScript.ORDER_ATOMIC);//Blockly.JavaScript.valueToCode(block, 'hidden_num', Blockly.JavaScript.ORDER_ATOMIC);
	return [num,Blockly.JavaScript.ORDER_ATOMIC];	
};
/*
Blockly.JavaScript["kidkat.test_xor"] = function(block){
	var num = block.getFieldValue("num_epoch");
	return ['DEV_IO.KidKat().test_xor('+num+');\n',Blockly.JavaScript.ORDER_ATOMIC];
};

Blockly.JavaScript["kidkat.test_read"] = function(block){
	return ['DEV_IO.KidKat().test_read();\n',Blockly.JavaScript.ORDER_ATOMIC];
};*/