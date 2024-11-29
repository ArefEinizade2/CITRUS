import numpy as np
import matplotlib.pyplot as plt

SNR_list = [np.inf, 20, 10, 0, -10]
p_list = range(5)

a = np.load('Stability_Results_NEW.npy')

Err_Mat_CGPGNN = a


fig = plt.figure()
for idx_SNR, SNR_2 in enumerate(SNR_list):
    Err_Mat_CGPGNN_temp = np.squeeze(np.flip(Err_Mat_CGPGNN[:, idx_SNR, :], 0))
    Mean_error_CGPGNN = np.mean(Err_Mat_CGPGNN_temp, axis=1)
    Mean_error_CGPGNN_var = np.var(Err_Mat_CGPGNN_temp, axis=1)*2
    plt.plot(p_list, Mean_error_CGPGNN, label='SNR_2='+str(SNR_2))
    plt.fill_between(p_list, Mean_error_CGPGNN - Mean_error_CGPGNN_var, 
                     Mean_error_CGPGNN + Mean_error_CGPGNN_var, alpha=0.2)
    plt.xlabel('SNR_1')
    plt.ylabel('prediction error')    
# plt.legend(['SNR_2=inf', 'SNR_2=20', 'SNR_2=10', 'SNR_2=0', 'SNR_2=-10'])  
plt.legend()  
plt.grid(True)
# plt.xticks(range(5), ['inf', '20', '10', '0', '-10'])
plt.xticks(range(5), ['-10', '0', '10', '20', 'inf'])
    
# plt.savefig('Stability_Results.png')
# fig.savefig('Stability_Results.pdf')
# fig.savefig('Stability_Results.eps')
# fig.savefig('Stability_Results.svg')
plt.show() 


