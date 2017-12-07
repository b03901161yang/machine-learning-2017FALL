wget 'https://www.dropbox.com/s/eqabuyoc0hknxw9/model_cnn.h5?dl=1' -O 'model_cnn.h5'
wget 'https://www.dropbox.com/s/hvso323vqr42v3o/model_cnn2.h5?dl=1' -O 'model_cnn2.h5' 
wget 'https://www.dropbox.com/s/hf57xk2iusvu52b/model_8179.h5?dl=1' -O 'model_8179.h5' 
wget 'https://www.dropbox.com/s/7db271d6hx23v0g/model_816.h5?dl=1' -O 'model_816.h5' 
wget 'https://www.dropbox.com/s/l6f8e4aei55sffx/model_gru0.h5?dl=1' -O 'model_gru0.h5'
wget 'https://www.dropbox.com/s/nh88jjfjm6k36rr/model_gru.h5?dl=1' -O 'model_gru.h5' 
wget 'https://www.dropbox.com/s/cw48o8bi7gsks6m/model_gru2.h5?dl=1' -O 'model_gru2.h5' 

python3 hw4_test.py $1 $2
