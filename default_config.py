class Config:
	lr = 1e-2
	cuda = True
	#batch_size = 360
	#num_workers = 64
	batch_size = 128
	num_workers = 0
	print_step = 128
	log_file = "./logs/first_logs/"
	model_saved = "models_saved/ctc_loss.pth"
	# model_resume = model_saved
	model_resume = None
	local_rank = 0  # only for intelliense, useless otherwise
	multi_gpu = False
