# Domain-adaptation-DOA-prediction
A Deep Learning Framework for Anesthesia Depth Prediction from Drug Infusion History
Abstract:
In target-controlled infusion (TCI) of propofol and remifentanil intravenous anesthesia, accurate prediction of the depth of anesthesia (DOA) is very challenging. Patients with different physiological characteristics have inconsistent pharmacodynamic responses during different stages of anesthesia. For example, in TCI, older adults transition smoothly from the induction period to the maintenance period, while younger adults are more prone to anesthetic awareness, resulting in different DOA data distributions among patients. To address these problems, a deep learning framework that incorporates domain adaptation and knowledge distillation and uses propofol and remifentanil doses at historical moments to continuously predict the bispectral index (BIS) is proposed in this paper.
Specifically, a modified adaptive recurrent neural network (AdaRNN) is adopted to address data distribution differences among patients. Moreover, a knowledge distillation pipeline is developed to train the prediction network by enabling it to learn intermediate feature representations of the teacher network. The experimental results show that our method exhibits better performance than existing approaches during all anesthetic phases in TCI of propofol and remifentanil intravenous anesthesia.

## Training
````
python teacher_student.py
````
## predicting
````
python predict_distill.py
````
