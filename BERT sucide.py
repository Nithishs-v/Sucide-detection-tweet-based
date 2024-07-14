#!/usr/bin/env python
# coding: utf-8

# In[20]:


pip install tensorflow pillow


# In[21]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import transformers
from transformers import TFAutoModel, AutoTokenizer


# In[22]:


raw = pd.read_csv('Suicide_Detection.csv')


# In[23]:


df = raw.sample(10000 , random_state=29)
df= df.drop('Unnamed: 0',axis=1)


# In[24]:


raw.head()


# In[25]:


df['class'].replace({'suicide' : 1 , 'non-suicide' : 0}, inplace=True)


# In[26]:


df


# In[27]:


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# In[28]:


from transformers import BertTokenizer
import tensorflow as tf


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


encoded = tokenizer(df['text'].tolist(), truncation=True, padding=True, return_tensors='tf')


labels = tf.convert_to_tensor(df['class'].tolist())


encoded_dataset = tf.data.Dataset.from_tensor_slices((dict(encoded), labels))


# In[29]:


encoded_dataset


# In[30]:


BATCH_SIZE = 2

def order(inp, label):
  
    return {
        'input_ids': inp['input_ids'],
        'attention_mask': inp['attention_mask'],
        'token_type_ids': inp['token_type_ids']
    }, label



train_size = int(0.8 * len(encoded_dataset))
val_size = int(0.1 * len(encoded_dataset))

train_dataset = encoded_dataset.take(train_size).map(order)
val_dataset = encoded_dataset.skip(train_size).take(val_size).map(order)
test_dataset = encoded_dataset.skip(train_size + val_size).map(order)


train_dataset = train_dataset.map(order)
val_dataset = val_dataset.map(order)
test_dataset = test_dataset.map(order)


train_dataset = train_dataset.batch(BATCH_SIZE).shuffle(1000)
val_dataset = val_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


# In[31]:


text, label = next(iter(train_dataset)) 
print(text, '\n\n', label)


# In[32]:


bert_model = TFAutoModel.from_pretrained("bert-base-uncased")


# In[33]:


class BERTForClassification(tf.keras.Model):
    
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        x = self.bert(inputs)[1]
        return self.fc(x)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'bert_model': self.bert,
            'num_classes': self.fc.units,
        })
        return config


# In[37]:


model = BERTForClassification(bert_model)


# In[38]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)


# In[39]:


history = model.fit(
    train_dataset,
    epochs=5,
    validation_data=val_dataset
)


# In[40]:


model.evaluate(test_dataset)


# In[41]:


y_true = []
for batch in test_dataset:
    x, y = batch
    y_true.extend(y.numpy())

y_pred_prob = model.predict(test_dataset)
y_pred = np.round(y_pred_prob).astype(int)\

cm = confusion_matrix(y_true, y_pred)
print(cm)


# In[42]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Extract true labels from the test dataset
y_true = []
for batch in test_dataset:
    x, y = batch
    y_true.extend(y.numpy())

# Predict probabilities on the test dataset
y_pred_prob = model.predict(test_dataset)

# Convert predicted probabilities to binary class labels
y_pred = np.round(y_pred_prob).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Print confusion matrix
print(cm)

# Plot confusion matrix as heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix Heatmap')
plt.show()


# In[43]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_dataset)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Extract true labels from the test dataset
y_true = []
for batch in test_dataset:
    x, y = batch
    y_true.extend(y.numpy())

# Predict probabilities on the test dataset
y_pred_prob = model.predict(test_dataset)

# Convert predicted probabilities to binary class labels
y_pred = np.round(y_pred_prob).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Compute F1 score and accuracy
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# Print confusion matrix, F1 score, and accuracy
print(f"Confusion Matrix:\n{cm}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")

# Plot confusion matrix as heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix Heatmap')
plt.show()


# In[44]:


from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


# In[45]:


f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
print(f"Confusion Matrix:\n{cm}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")


# In[46]:


plt.subplot(1, 2, 1)
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




