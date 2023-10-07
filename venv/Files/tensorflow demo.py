import tensorflow as tf
print(f'Your tensorflow version {tf.__version__}')

# create a constant tensor A
A = tf.constant([[4,3], [6,1]])
print('A:\n', A)

# create a variable tensor V
V = tf.Variable([[3,1], [5,2]])
print('V:\n', V)

# create a constant tensor B
B = tf.constant([[7,8], [4,5]])
print('B:\n', B)

# concatenate columns shape  ~ (2,2,2)
AB_concatenated = tf.concat(values=[A, B], axis=1)
print('AB_concatenated:\n', AB_concatenated)

# concatenate rows
AB_concatenated1 = tf.concat(values=[A, B], axis=0)
print('AB_concatenated1:\n', AB_concatenated1)

# create a 3,4 matrix with zeros
tensor = tf.zeros(shape=[3,4], dtype=tf.float32)
print('tensor:\n', tensor)

# create a 3,4 matrix with ones
tensor2 = tf.ones(shape=[3,4], dtype=tf.float32)
print('tensor2:\n', tensor2)

# create a 3,4 random uniform matrix
tensor3 = tf.random.uniform(shape=[3,4], dtype=tf.float32)
print('tensor3:\n', tensor3)

# reshaped tensor
reshaped_tensor = tf.reshape(tensor=tensor, shape=[4,3])
print('reshaped_tensor:\n', reshaped_tensor)

# typecast a tensor
tensor = tf.constant([[4.6, 4.2], [7.5,3.6], [2.7,9.4], [6.7,8.3]], dtype=tf.float32)
tensor_as_int = tf.cast(tensor, tf.int32)
print('tensor:\n', tensor)
print('tensor_as_int:\n', tensor_as_int)

#transpose a tensor
a = tf.transpose(tensor_as_int)
print('a:\n', a)

# matrix multiplication of tensors
A = tf.constant([[5,8], [3,9]])
v = tf.constant([[4], [2]])
Av = tf.matmul(A, v)
print('Av:\n', Av)

# element-wise multiplication
A_v = tf.multiply(A, v)
print('A_v:\n', A_v)

# creating an identity matrix
A = tf.constant([[4,9],[5,6],[1,8]])
rows, columns = A.shape
print(f'rows: {rows} columns: {columns}')
A_identity = tf.eye(num_rows=rows, num_columns=columns, dtype=tf.int32)
print('A_identity:\n', A_identity)