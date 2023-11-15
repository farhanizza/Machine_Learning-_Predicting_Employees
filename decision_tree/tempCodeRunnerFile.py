TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]

print("\n")

print("True Positive (TP) Jumlah karyawan yang benar-benar mengundurkan diri dan diprediksi mengundurkan diri: ", TP)
print("True Negative (TN) Jumlah karyawan yang benar-benar tidak mengundurkan diri dan diprediksi tidak mengundurkan diri: ", TN)
print("False Positive (FP) Jumlah karyawan yang tidak mengundurkan diri tetapi diprediksi mengundurkan diri: ", FP)
print("False Negative (FN) Jumlah karyawan yang mengundurkan diri tetapi diprediksi tidak mengundurkan diri: ", FN)