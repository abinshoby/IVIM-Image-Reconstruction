ȸ;
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
n
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle��element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements#
handle��element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.6.02v2.6.0-rc2-32-g919f693420e8��9
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:d*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
4bidirectional_11/forward_lstm_11/lstm_cell_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*E
shared_name64bidirectional_11/forward_lstm_11/lstm_cell_34/kernel
�
Hbidirectional_11/forward_lstm_11/lstm_cell_34/kernel/Read/ReadVariableOpReadVariableOp4bidirectional_11/forward_lstm_11/lstm_cell_34/kernel*
_output_shapes
:	�*
dtype0
�
>bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*O
shared_name@>bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel
�
Rbidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/Read/ReadVariableOpReadVariableOp>bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel*
_output_shapes
:	2�*
dtype0
�
2bidirectional_11/forward_lstm_11/lstm_cell_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*C
shared_name42bidirectional_11/forward_lstm_11/lstm_cell_34/bias
�
Fbidirectional_11/forward_lstm_11/lstm_cell_34/bias/Read/ReadVariableOpReadVariableOp2bidirectional_11/forward_lstm_11/lstm_cell_34/bias*
_output_shapes	
:�*
dtype0
�
5bidirectional_11/backward_lstm_11/lstm_cell_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*F
shared_name75bidirectional_11/backward_lstm_11/lstm_cell_35/kernel
�
Ibidirectional_11/backward_lstm_11/lstm_cell_35/kernel/Read/ReadVariableOpReadVariableOp5bidirectional_11/backward_lstm_11/lstm_cell_35/kernel*
_output_shapes
:	�*
dtype0
�
?bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*P
shared_nameA?bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel
�
Sbidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/Read/ReadVariableOpReadVariableOp?bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel*
_output_shapes
:	2�*
dtype0
�
3bidirectional_11/backward_lstm_11/lstm_cell_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*D
shared_name53bidirectional_11/backward_lstm_11/lstm_cell_35/bias
�
Gbidirectional_11/backward_lstm_11/lstm_cell_35/bias/Read/ReadVariableOpReadVariableOp3bidirectional_11/backward_lstm_11/lstm_cell_35/bias*
_output_shapes	
:�*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_11/kernel/m
�
*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m*
_output_shapes

:d*
dtype0
�
Adam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_11/bias/m
y
(Adam/dense_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/m*
_output_shapes
:*
dtype0
�
;Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*L
shared_name=;Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/m
�
OAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/m/Read/ReadVariableOpReadVariableOp;Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/m*
_output_shapes
:	�*
dtype0
�
EAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*V
shared_nameGEAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/m
�
YAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpEAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/m*
_output_shapes
:	2�*
dtype0
�
9Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*J
shared_name;9Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/m
�
MAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/m/Read/ReadVariableOpReadVariableOp9Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/m*
_output_shapes	
:�*
dtype0
�
<Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*M
shared_name><Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/m
�
PAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/m/Read/ReadVariableOpReadVariableOp<Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/m*
_output_shapes
:	�*
dtype0
�
FAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*W
shared_nameHFAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/m
�
ZAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/m*
_output_shapes
:	2�*
dtype0
�
:Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*K
shared_name<:Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/m
�
NAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/m/Read/ReadVariableOpReadVariableOp:Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_11/kernel/v
�
*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v*
_output_shapes

:d*
dtype0
�
Adam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_11/bias/v
y
(Adam/dense_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/v*
_output_shapes
:*
dtype0
�
;Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*L
shared_name=;Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/v
�
OAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/v/Read/ReadVariableOpReadVariableOp;Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/v*
_output_shapes
:	�*
dtype0
�
EAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*V
shared_nameGEAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/v
�
YAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpEAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/v*
_output_shapes
:	2�*
dtype0
�
9Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*J
shared_name;9Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/v
�
MAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/v/Read/ReadVariableOpReadVariableOp9Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/v*
_output_shapes	
:�*
dtype0
�
<Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*M
shared_name><Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/v
�
PAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/v/Read/ReadVariableOpReadVariableOp<Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/v*
_output_shapes
:	�*
dtype0
�
FAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*W
shared_nameHFAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/v
�
ZAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/v*
_output_shapes
:	2�*
dtype0
�
:Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*K
shared_name<:Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/v
�
NAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/v/Read/ReadVariableOpReadVariableOp:Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_11/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d**
shared_nameAdam/dense_11/kernel/vhat
�
-Adam/dense_11/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/vhat*
_output_shapes

:d*
dtype0
�
Adam/dense_11/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/dense_11/bias/vhat

+Adam/dense_11/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/vhat*
_output_shapes
:*
dtype0
�
>Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*O
shared_name@>Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/vhat
�
RAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/vhat/Read/ReadVariableOpReadVariableOp>Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/vhat*
_output_shapes
:	�*
dtype0
�
HAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*Y
shared_nameJHAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/vhat
�
\Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpHAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/vhat*
_output_shapes
:	2�*
dtype0
�
<Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*M
shared_name><Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/vhat
�
PAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/vhat/Read/ReadVariableOpReadVariableOp<Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/vhat*
_output_shapes	
:�*
dtype0
�
?Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*P
shared_nameA?Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/vhat
�
SAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/vhat/Read/ReadVariableOpReadVariableOp?Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/vhat*
_output_shapes
:	�*
dtype0
�
IAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*Z
shared_nameKIAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/vhat
�
]Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpIAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/vhat*
_output_shapes
:	2�*
dtype0
�
=Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*N
shared_name?=Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/vhat
�
QAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/vhat/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/vhat*
_output_shapes	
:�*
dtype0

NoOpNoOp
�@
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�@
value�?B�? B�?
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
y
	forward_layer

backward_layer
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
�
iter

beta_1

beta_2
	decay
learning_ratem`mambmcmdmemfmgvhvivjvkvlvmvnvo
vhatp
vhatq
vhatr
vhats
vhatt
vhatu
vhatv
vhatw
 
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
�
 layer_metrics
!non_trainable_variables
regularization_losses
	variables
"metrics
trainable_variables
#layer_regularization_losses

$layers
 
l
%cell
&
state_spec
'regularization_losses
(	variables
)trainable_variables
*	keras_api
l
+cell
,
state_spec
-regularization_losses
.	variables
/trainable_variables
0	keras_api
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
�
1layer_metrics
2non_trainable_variables
regularization_losses
	variables
3metrics
trainable_variables
4layer_regularization_losses

5layers
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_11/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
6layer_metrics
7non_trainable_variables
regularization_losses
	variables
8metrics
trainable_variables
9layer_regularization_losses

:layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE4bidirectional_11/forward_lstm_11/lstm_cell_34/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE>bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2bidirectional_11/forward_lstm_11/lstm_cell_34/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5bidirectional_11/backward_lstm_11/lstm_cell_35/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE?bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3bidirectional_11/backward_lstm_11/lstm_cell_35/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 

;0
 

0
1
�
<
state_size

kernel
recurrent_kernel
bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
 
 

0
1
2

0
1
2
�
Alayer_metrics
Bnon_trainable_variables

Cstates
'regularization_losses
(	variables
Dmetrics
)trainable_variables
Elayer_regularization_losses

Flayers
�
G
state_size

kernel
recurrent_kernel
bias
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
 
 

0
1
2

0
1
2
�
Llayer_metrics
Mnon_trainable_variables

Nstates
-regularization_losses
.	variables
Ometrics
/trainable_variables
Player_regularization_losses

Qlayers
 
 
 
 

	0

1
 
 
 
 
 
4
	Rtotal
	Scount
T	variables
U	keras_api
 
 

0
1
2

0
1
2
�
Vlayer_metrics
Wnon_trainable_variables
=regularization_losses
>	variables
Xmetrics
?trainable_variables
Ylayer_regularization_losses

Zlayers
 
 
 
 
 

%0
 
 

0
1
2

0
1
2
�
[layer_metrics
\non_trainable_variables
Hregularization_losses
I	variables
]metrics
Jtrainable_variables
^layer_regularization_losses

_layers
 
 
 
 
 

+0
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

R0
S1

T	variables
 
 
 
 
 
 
 
 
 
 
~|
VARIABLE_VALUEAdam/dense_11/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE;Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEEAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE9Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE<Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEFAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE:Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE;Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEEAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE9Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE<Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEFAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE:Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_11/kernel/vhatUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/dense_11/bias/vhatSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE>Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/vhatEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEHAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/vhatEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE<Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/vhatEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE?Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/vhatEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEIAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/vhatEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE=Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/vhatEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
y
serving_default_args_0Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
s
serving_default_args_0_1Placeholder*#
_output_shapes
:���������*
dtype0	*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_0_14bidirectional_11/forward_lstm_11/lstm_cell_34/kernel>bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel2bidirectional_11/forward_lstm_11/lstm_cell_34/bias5bidirectional_11/backward_lstm_11/lstm_cell_35/kernel?bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel3bidirectional_11/backward_lstm_11/lstm_cell_35/biasdense_11/kerneldense_11/bias*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_1703300
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpHbidirectional_11/forward_lstm_11/lstm_cell_34/kernel/Read/ReadVariableOpRbidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/Read/ReadVariableOpFbidirectional_11/forward_lstm_11/lstm_cell_34/bias/Read/ReadVariableOpIbidirectional_11/backward_lstm_11/lstm_cell_35/kernel/Read/ReadVariableOpSbidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/Read/ReadVariableOpGbidirectional_11/backward_lstm_11/lstm_cell_35/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp(Adam/dense_11/bias/m/Read/ReadVariableOpOAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/m/Read/ReadVariableOpYAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/m/Read/ReadVariableOpMAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/m/Read/ReadVariableOpPAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/m/Read/ReadVariableOpZAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/m/Read/ReadVariableOpNAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/m/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp(Adam/dense_11/bias/v/Read/ReadVariableOpOAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/v/Read/ReadVariableOpYAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/v/Read/ReadVariableOpMAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/v/Read/ReadVariableOpPAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/v/Read/ReadVariableOpZAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/v/Read/ReadVariableOpNAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/v/Read/ReadVariableOp-Adam/dense_11/kernel/vhat/Read/ReadVariableOp+Adam/dense_11/bias/vhat/Read/ReadVariableOpRAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/vhat/Read/ReadVariableOp\Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/vhat/Read/ReadVariableOpPAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/vhat/Read/ReadVariableOpSAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/vhat/Read/ReadVariableOp]Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/vhat/Read/ReadVariableOpQAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/vhat/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_1706351
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_11/kerneldense_11/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate4bidirectional_11/forward_lstm_11/lstm_cell_34/kernel>bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel2bidirectional_11/forward_lstm_11/lstm_cell_34/bias5bidirectional_11/backward_lstm_11/lstm_cell_35/kernel?bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel3bidirectional_11/backward_lstm_11/lstm_cell_35/biastotalcountAdam/dense_11/kernel/mAdam/dense_11/bias/m;Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/mEAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/m9Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/m<Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/mFAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/m:Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/mAdam/dense_11/kernel/vAdam/dense_11/bias/v;Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/vEAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/v9Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/v<Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/vFAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/v:Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/vAdam/dense_11/kernel/vhatAdam/dense_11/bias/vhat>Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/vhatHAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/vhat<Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/vhat?Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/vhatIAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/vhat=Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/vhat*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_1706478��8
�b
�
"forward_lstm_11_while_body_1704056<
8forward_lstm_11_while_forward_lstm_11_while_loop_counterB
>forward_lstm_11_while_forward_lstm_11_while_maximum_iterations%
!forward_lstm_11_while_placeholder'
#forward_lstm_11_while_placeholder_1'
#forward_lstm_11_while_placeholder_2'
#forward_lstm_11_while_placeholder_3'
#forward_lstm_11_while_placeholder_4;
7forward_lstm_11_while_forward_lstm_11_strided_slice_1_0w
sforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_08
4forward_lstm_11_while_greater_forward_lstm_11_cast_0V
Cforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0:	�X
Eforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0:	2�S
Dforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0:	�"
forward_lstm_11_while_identity$
 forward_lstm_11_while_identity_1$
 forward_lstm_11_while_identity_2$
 forward_lstm_11_while_identity_3$
 forward_lstm_11_while_identity_4$
 forward_lstm_11_while_identity_5$
 forward_lstm_11_while_identity_69
5forward_lstm_11_while_forward_lstm_11_strided_slice_1u
qforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor6
2forward_lstm_11_while_greater_forward_lstm_11_castT
Aforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource:	�V
Cforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource:	2�Q
Bforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource:	���9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp�8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp�:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp�
Gforward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2I
Gforward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape�
9forward_lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_11_while_placeholderPforward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02;
9forward_lstm_11/while/TensorArrayV2Read/TensorListGetItem�
forward_lstm_11/while/GreaterGreater4forward_lstm_11_while_greater_forward_lstm_11_cast_0!forward_lstm_11_while_placeholder*
T0*#
_output_shapes
:���������2
forward_lstm_11/while/Greater�
8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOpCforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02:
8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp�
)forward_lstm_11/while/lstm_cell_34/MatMulMatMul@forward_lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:0@forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)forward_lstm_11/while/lstm_cell_34/MatMul�
:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOpEforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02<
:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp�
+forward_lstm_11/while/lstm_cell_34/MatMul_1MatMul#forward_lstm_11_while_placeholder_3Bforward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+forward_lstm_11/while/lstm_cell_34/MatMul_1�
&forward_lstm_11/while/lstm_cell_34/addAddV23forward_lstm_11/while/lstm_cell_34/MatMul:product:05forward_lstm_11/while/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&forward_lstm_11/while/lstm_cell_34/add�
9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOpDforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02;
9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp�
*forward_lstm_11/while/lstm_cell_34/BiasAddBiasAdd*forward_lstm_11/while/lstm_cell_34/add:z:0Aforward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*forward_lstm_11/while/lstm_cell_34/BiasAdd�
2forward_lstm_11/while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2forward_lstm_11/while/lstm_cell_34/split/split_dim�
(forward_lstm_11/while/lstm_cell_34/splitSplit;forward_lstm_11/while/lstm_cell_34/split/split_dim:output:03forward_lstm_11/while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2*
(forward_lstm_11/while/lstm_cell_34/split�
*forward_lstm_11/while/lstm_cell_34/SigmoidSigmoid1forward_lstm_11/while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22,
*forward_lstm_11/while/lstm_cell_34/Sigmoid�
,forward_lstm_11/while/lstm_cell_34/Sigmoid_1Sigmoid1forward_lstm_11/while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22.
,forward_lstm_11/while/lstm_cell_34/Sigmoid_1�
&forward_lstm_11/while/lstm_cell_34/mulMul0forward_lstm_11/while/lstm_cell_34/Sigmoid_1:y:0#forward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22(
&forward_lstm_11/while/lstm_cell_34/mul�
'forward_lstm_11/while/lstm_cell_34/ReluRelu1forward_lstm_11/while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22)
'forward_lstm_11/while/lstm_cell_34/Relu�
(forward_lstm_11/while/lstm_cell_34/mul_1Mul.forward_lstm_11/while/lstm_cell_34/Sigmoid:y:05forward_lstm_11/while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_11/while/lstm_cell_34/mul_1�
(forward_lstm_11/while/lstm_cell_34/add_1AddV2*forward_lstm_11/while/lstm_cell_34/mul:z:0,forward_lstm_11/while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_11/while/lstm_cell_34/add_1�
,forward_lstm_11/while/lstm_cell_34/Sigmoid_2Sigmoid1forward_lstm_11/while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22.
,forward_lstm_11/while/lstm_cell_34/Sigmoid_2�
)forward_lstm_11/while/lstm_cell_34/Relu_1Relu,forward_lstm_11/while/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22+
)forward_lstm_11/while/lstm_cell_34/Relu_1�
(forward_lstm_11/while/lstm_cell_34/mul_2Mul0forward_lstm_11/while/lstm_cell_34/Sigmoid_2:y:07forward_lstm_11/while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_11/while/lstm_cell_34/mul_2�
forward_lstm_11/while/SelectSelect!forward_lstm_11/while/Greater:z:0,forward_lstm_11/while/lstm_cell_34/mul_2:z:0#forward_lstm_11_while_placeholder_2*
T0*'
_output_shapes
:���������22
forward_lstm_11/while/Select�
forward_lstm_11/while/Select_1Select!forward_lstm_11/while/Greater:z:0,forward_lstm_11/while/lstm_cell_34/mul_2:z:0#forward_lstm_11_while_placeholder_3*
T0*'
_output_shapes
:���������22 
forward_lstm_11/while/Select_1�
forward_lstm_11/while/Select_2Select!forward_lstm_11/while/Greater:z:0,forward_lstm_11/while/lstm_cell_34/add_1:z:0#forward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22 
forward_lstm_11/while/Select_2�
:forward_lstm_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_11_while_placeholder_1!forward_lstm_11_while_placeholder%forward_lstm_11/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_11/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_11/while/add/y�
forward_lstm_11/while/addAddV2!forward_lstm_11_while_placeholder$forward_lstm_11/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/while/add�
forward_lstm_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_11/while/add_1/y�
forward_lstm_11/while/add_1AddV28forward_lstm_11_while_forward_lstm_11_while_loop_counter&forward_lstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/while/add_1�
forward_lstm_11/while/IdentityIdentityforward_lstm_11/while/add_1:z:0^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_11/while/Identity�
 forward_lstm_11/while/Identity_1Identity>forward_lstm_11_while_forward_lstm_11_while_maximum_iterations^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_11/while/Identity_1�
 forward_lstm_11/while/Identity_2Identityforward_lstm_11/while/add:z:0^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_11/while/Identity_2�
 forward_lstm_11/while/Identity_3IdentityJforward_lstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_11/while/Identity_3�
 forward_lstm_11/while/Identity_4Identity%forward_lstm_11/while/Select:output:0^forward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/while/Identity_4�
 forward_lstm_11/while/Identity_5Identity'forward_lstm_11/while/Select_1:output:0^forward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/while/Identity_5�
 forward_lstm_11/while/Identity_6Identity'forward_lstm_11/while/Select_2:output:0^forward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/while/Identity_6�
forward_lstm_11/while/NoOpNoOp:^forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp9^forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp;^forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_11/while/NoOp"p
5forward_lstm_11_while_forward_lstm_11_strided_slice_17forward_lstm_11_while_forward_lstm_11_strided_slice_1_0"j
2forward_lstm_11_while_greater_forward_lstm_11_cast4forward_lstm_11_while_greater_forward_lstm_11_cast_0"I
forward_lstm_11_while_identity'forward_lstm_11/while/Identity:output:0"M
 forward_lstm_11_while_identity_1)forward_lstm_11/while/Identity_1:output:0"M
 forward_lstm_11_while_identity_2)forward_lstm_11/while/Identity_2:output:0"M
 forward_lstm_11_while_identity_3)forward_lstm_11/while/Identity_3:output:0"M
 forward_lstm_11_while_identity_4)forward_lstm_11/while/Identity_4:output:0"M
 forward_lstm_11_while_identity_5)forward_lstm_11/while/Identity_5:output:0"M
 forward_lstm_11_while_identity_6)forward_lstm_11/while/Identity_6:output:0"�
Bforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resourceDforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0"�
Cforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resourceEforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0"�
Aforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resourceCforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0"�
qforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensorsforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2v
9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp2t
8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp2x
:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:���������
�W
�
#backward_lstm_11_while_body_1703888>
:backward_lstm_11_while_backward_lstm_11_while_loop_counterD
@backward_lstm_11_while_backward_lstm_11_while_maximum_iterations&
"backward_lstm_11_while_placeholder(
$backward_lstm_11_while_placeholder_1(
$backward_lstm_11_while_placeholder_2(
$backward_lstm_11_while_placeholder_3=
9backward_lstm_11_while_backward_lstm_11_strided_slice_1_0y
ubackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0W
Dbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0:	�Y
Fbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0:	2�T
Ebackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0:	�#
backward_lstm_11_while_identity%
!backward_lstm_11_while_identity_1%
!backward_lstm_11_while_identity_2%
!backward_lstm_11_while_identity_3%
!backward_lstm_11_while_identity_4%
!backward_lstm_11_while_identity_5;
7backward_lstm_11_while_backward_lstm_11_strided_slice_1w
sbackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensorU
Bbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource:	�W
Dbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource:	2�R
Cbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource:	���:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp�9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp�;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp�
Hbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2J
Hbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape�
:backward_lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_11_while_placeholderQbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02<
:backward_lstm_11/while/TensorArrayV2Read/TensorListGetItem�
9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOpDbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02;
9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp�
*backward_lstm_11/while/lstm_cell_35/MatMulMatMulAbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:0Abackward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*backward_lstm_11/while/lstm_cell_35/MatMul�
;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOpFbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02=
;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp�
,backward_lstm_11/while/lstm_cell_35/MatMul_1MatMul$backward_lstm_11_while_placeholder_2Cbackward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2.
,backward_lstm_11/while/lstm_cell_35/MatMul_1�
'backward_lstm_11/while/lstm_cell_35/addAddV24backward_lstm_11/while/lstm_cell_35/MatMul:product:06backward_lstm_11/while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2)
'backward_lstm_11/while/lstm_cell_35/add�
:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOpEbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02<
:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp�
+backward_lstm_11/while/lstm_cell_35/BiasAddBiasAdd+backward_lstm_11/while/lstm_cell_35/add:z:0Bbackward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+backward_lstm_11/while/lstm_cell_35/BiasAdd�
3backward_lstm_11/while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3backward_lstm_11/while/lstm_cell_35/split/split_dim�
)backward_lstm_11/while/lstm_cell_35/splitSplit<backward_lstm_11/while/lstm_cell_35/split/split_dim:output:04backward_lstm_11/while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2+
)backward_lstm_11/while/lstm_cell_35/split�
+backward_lstm_11/while/lstm_cell_35/SigmoidSigmoid2backward_lstm_11/while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22-
+backward_lstm_11/while/lstm_cell_35/Sigmoid�
-backward_lstm_11/while/lstm_cell_35/Sigmoid_1Sigmoid2backward_lstm_11/while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22/
-backward_lstm_11/while/lstm_cell_35/Sigmoid_1�
'backward_lstm_11/while/lstm_cell_35/mulMul1backward_lstm_11/while/lstm_cell_35/Sigmoid_1:y:0$backward_lstm_11_while_placeholder_3*
T0*'
_output_shapes
:���������22)
'backward_lstm_11/while/lstm_cell_35/mul�
(backward_lstm_11/while/lstm_cell_35/ReluRelu2backward_lstm_11/while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22*
(backward_lstm_11/while/lstm_cell_35/Relu�
)backward_lstm_11/while/lstm_cell_35/mul_1Mul/backward_lstm_11/while/lstm_cell_35/Sigmoid:y:06backward_lstm_11/while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_11/while/lstm_cell_35/mul_1�
)backward_lstm_11/while/lstm_cell_35/add_1AddV2+backward_lstm_11/while/lstm_cell_35/mul:z:0-backward_lstm_11/while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_11/while/lstm_cell_35/add_1�
-backward_lstm_11/while/lstm_cell_35/Sigmoid_2Sigmoid2backward_lstm_11/while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22/
-backward_lstm_11/while/lstm_cell_35/Sigmoid_2�
*backward_lstm_11/while/lstm_cell_35/Relu_1Relu-backward_lstm_11/while/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22,
*backward_lstm_11/while/lstm_cell_35/Relu_1�
)backward_lstm_11/while/lstm_cell_35/mul_2Mul1backward_lstm_11/while/lstm_cell_35/Sigmoid_2:y:08backward_lstm_11/while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_11/while/lstm_cell_35/mul_2�
;backward_lstm_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_11_while_placeholder_1"backward_lstm_11_while_placeholder-backward_lstm_11/while/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_11/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_11/while/add/y�
backward_lstm_11/while/addAddV2"backward_lstm_11_while_placeholder%backward_lstm_11/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/while/add�
backward_lstm_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_11/while/add_1/y�
backward_lstm_11/while/add_1AddV2:backward_lstm_11_while_backward_lstm_11_while_loop_counter'backward_lstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/while/add_1�
backward_lstm_11/while/IdentityIdentity backward_lstm_11/while/add_1:z:0^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_11/while/Identity�
!backward_lstm_11/while/Identity_1Identity@backward_lstm_11_while_backward_lstm_11_while_maximum_iterations^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_11/while/Identity_1�
!backward_lstm_11/while/Identity_2Identitybackward_lstm_11/while/add:z:0^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_11/while/Identity_2�
!backward_lstm_11/while/Identity_3IdentityKbackward_lstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_11/while/Identity_3�
!backward_lstm_11/while/Identity_4Identity-backward_lstm_11/while/lstm_cell_35/mul_2:z:0^backward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/while/Identity_4�
!backward_lstm_11/while/Identity_5Identity-backward_lstm_11/while/lstm_cell_35/add_1:z:0^backward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/while/Identity_5�
backward_lstm_11/while/NoOpNoOp;^backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp:^backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp<^backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_11/while/NoOp"t
7backward_lstm_11_while_backward_lstm_11_strided_slice_19backward_lstm_11_while_backward_lstm_11_strided_slice_1_0"K
backward_lstm_11_while_identity(backward_lstm_11/while/Identity:output:0"O
!backward_lstm_11_while_identity_1*backward_lstm_11/while/Identity_1:output:0"O
!backward_lstm_11_while_identity_2*backward_lstm_11/while/Identity_2:output:0"O
!backward_lstm_11_while_identity_3*backward_lstm_11/while/Identity_3:output:0"O
!backward_lstm_11_while_identity_4*backward_lstm_11/while/Identity_4:output:0"O
!backward_lstm_11_while_identity_5*backward_lstm_11/while/Identity_5:output:0"�
Cbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resourceEbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0"�
Dbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resourceFbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0"�
Bbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resourceDbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0"�
sbackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensorubackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2x
:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp2v
9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp2z
;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_1700964
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1700964___redundant_placeholder05
1while_while_cond_1700964___redundant_placeholder15
1while_while_cond_1700964___redundant_placeholder25
1while_while_cond_1700964___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_1701744
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1701744___redundant_placeholder05
1while_while_cond_1701744___redundant_placeholder15
1while_while_cond_1701744___redundant_placeholder25
1while_while_cond_1701744___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
�b
�
"forward_lstm_11_while_body_1702844<
8forward_lstm_11_while_forward_lstm_11_while_loop_counterB
>forward_lstm_11_while_forward_lstm_11_while_maximum_iterations%
!forward_lstm_11_while_placeholder'
#forward_lstm_11_while_placeholder_1'
#forward_lstm_11_while_placeholder_2'
#forward_lstm_11_while_placeholder_3'
#forward_lstm_11_while_placeholder_4;
7forward_lstm_11_while_forward_lstm_11_strided_slice_1_0w
sforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_08
4forward_lstm_11_while_greater_forward_lstm_11_cast_0V
Cforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0:	�X
Eforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0:	2�S
Dforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0:	�"
forward_lstm_11_while_identity$
 forward_lstm_11_while_identity_1$
 forward_lstm_11_while_identity_2$
 forward_lstm_11_while_identity_3$
 forward_lstm_11_while_identity_4$
 forward_lstm_11_while_identity_5$
 forward_lstm_11_while_identity_69
5forward_lstm_11_while_forward_lstm_11_strided_slice_1u
qforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor6
2forward_lstm_11_while_greater_forward_lstm_11_castT
Aforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource:	�V
Cforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource:	2�Q
Bforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource:	���9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp�8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp�:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp�
Gforward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2I
Gforward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape�
9forward_lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_11_while_placeholderPforward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02;
9forward_lstm_11/while/TensorArrayV2Read/TensorListGetItem�
forward_lstm_11/while/GreaterGreater4forward_lstm_11_while_greater_forward_lstm_11_cast_0!forward_lstm_11_while_placeholder*
T0*#
_output_shapes
:���������2
forward_lstm_11/while/Greater�
8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOpCforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02:
8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp�
)forward_lstm_11/while/lstm_cell_34/MatMulMatMul@forward_lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:0@forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)forward_lstm_11/while/lstm_cell_34/MatMul�
:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOpEforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02<
:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp�
+forward_lstm_11/while/lstm_cell_34/MatMul_1MatMul#forward_lstm_11_while_placeholder_3Bforward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+forward_lstm_11/while/lstm_cell_34/MatMul_1�
&forward_lstm_11/while/lstm_cell_34/addAddV23forward_lstm_11/while/lstm_cell_34/MatMul:product:05forward_lstm_11/while/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&forward_lstm_11/while/lstm_cell_34/add�
9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOpDforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02;
9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp�
*forward_lstm_11/while/lstm_cell_34/BiasAddBiasAdd*forward_lstm_11/while/lstm_cell_34/add:z:0Aforward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*forward_lstm_11/while/lstm_cell_34/BiasAdd�
2forward_lstm_11/while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2forward_lstm_11/while/lstm_cell_34/split/split_dim�
(forward_lstm_11/while/lstm_cell_34/splitSplit;forward_lstm_11/while/lstm_cell_34/split/split_dim:output:03forward_lstm_11/while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2*
(forward_lstm_11/while/lstm_cell_34/split�
*forward_lstm_11/while/lstm_cell_34/SigmoidSigmoid1forward_lstm_11/while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22,
*forward_lstm_11/while/lstm_cell_34/Sigmoid�
,forward_lstm_11/while/lstm_cell_34/Sigmoid_1Sigmoid1forward_lstm_11/while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22.
,forward_lstm_11/while/lstm_cell_34/Sigmoid_1�
&forward_lstm_11/while/lstm_cell_34/mulMul0forward_lstm_11/while/lstm_cell_34/Sigmoid_1:y:0#forward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22(
&forward_lstm_11/while/lstm_cell_34/mul�
'forward_lstm_11/while/lstm_cell_34/ReluRelu1forward_lstm_11/while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22)
'forward_lstm_11/while/lstm_cell_34/Relu�
(forward_lstm_11/while/lstm_cell_34/mul_1Mul.forward_lstm_11/while/lstm_cell_34/Sigmoid:y:05forward_lstm_11/while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_11/while/lstm_cell_34/mul_1�
(forward_lstm_11/while/lstm_cell_34/add_1AddV2*forward_lstm_11/while/lstm_cell_34/mul:z:0,forward_lstm_11/while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_11/while/lstm_cell_34/add_1�
,forward_lstm_11/while/lstm_cell_34/Sigmoid_2Sigmoid1forward_lstm_11/while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22.
,forward_lstm_11/while/lstm_cell_34/Sigmoid_2�
)forward_lstm_11/while/lstm_cell_34/Relu_1Relu,forward_lstm_11/while/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22+
)forward_lstm_11/while/lstm_cell_34/Relu_1�
(forward_lstm_11/while/lstm_cell_34/mul_2Mul0forward_lstm_11/while/lstm_cell_34/Sigmoid_2:y:07forward_lstm_11/while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_11/while/lstm_cell_34/mul_2�
forward_lstm_11/while/SelectSelect!forward_lstm_11/while/Greater:z:0,forward_lstm_11/while/lstm_cell_34/mul_2:z:0#forward_lstm_11_while_placeholder_2*
T0*'
_output_shapes
:���������22
forward_lstm_11/while/Select�
forward_lstm_11/while/Select_1Select!forward_lstm_11/while/Greater:z:0,forward_lstm_11/while/lstm_cell_34/mul_2:z:0#forward_lstm_11_while_placeholder_3*
T0*'
_output_shapes
:���������22 
forward_lstm_11/while/Select_1�
forward_lstm_11/while/Select_2Select!forward_lstm_11/while/Greater:z:0,forward_lstm_11/while/lstm_cell_34/add_1:z:0#forward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22 
forward_lstm_11/while/Select_2�
:forward_lstm_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_11_while_placeholder_1!forward_lstm_11_while_placeholder%forward_lstm_11/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_11/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_11/while/add/y�
forward_lstm_11/while/addAddV2!forward_lstm_11_while_placeholder$forward_lstm_11/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/while/add�
forward_lstm_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_11/while/add_1/y�
forward_lstm_11/while/add_1AddV28forward_lstm_11_while_forward_lstm_11_while_loop_counter&forward_lstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/while/add_1�
forward_lstm_11/while/IdentityIdentityforward_lstm_11/while/add_1:z:0^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_11/while/Identity�
 forward_lstm_11/while/Identity_1Identity>forward_lstm_11_while_forward_lstm_11_while_maximum_iterations^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_11/while/Identity_1�
 forward_lstm_11/while/Identity_2Identityforward_lstm_11/while/add:z:0^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_11/while/Identity_2�
 forward_lstm_11/while/Identity_3IdentityJforward_lstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_11/while/Identity_3�
 forward_lstm_11/while/Identity_4Identity%forward_lstm_11/while/Select:output:0^forward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/while/Identity_4�
 forward_lstm_11/while/Identity_5Identity'forward_lstm_11/while/Select_1:output:0^forward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/while/Identity_5�
 forward_lstm_11/while/Identity_6Identity'forward_lstm_11/while/Select_2:output:0^forward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/while/Identity_6�
forward_lstm_11/while/NoOpNoOp:^forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp9^forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp;^forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_11/while/NoOp"p
5forward_lstm_11_while_forward_lstm_11_strided_slice_17forward_lstm_11_while_forward_lstm_11_strided_slice_1_0"j
2forward_lstm_11_while_greater_forward_lstm_11_cast4forward_lstm_11_while_greater_forward_lstm_11_cast_0"I
forward_lstm_11_while_identity'forward_lstm_11/while/Identity:output:0"M
 forward_lstm_11_while_identity_1)forward_lstm_11/while/Identity_1:output:0"M
 forward_lstm_11_while_identity_2)forward_lstm_11/while/Identity_2:output:0"M
 forward_lstm_11_while_identity_3)forward_lstm_11/while/Identity_3:output:0"M
 forward_lstm_11_while_identity_4)forward_lstm_11/while/Identity_4:output:0"M
 forward_lstm_11_while_identity_5)forward_lstm_11/while/Identity_5:output:0"M
 forward_lstm_11_while_identity_6)forward_lstm_11/while/Identity_6:output:0"�
Bforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resourceDforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0"�
Cforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resourceEforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0"�
Aforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resourceCforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0"�
qforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensorsforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2v
9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp2t
8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp2x
:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:���������
�^
�
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_1702021

inputs>
+lstm_cell_35_matmul_readvariableop_resource:	�@
-lstm_cell_35_matmul_1_readvariableop_resource:	2�;
,lstm_cell_35_biasadd_readvariableop_resource:	�
identity��#lstm_cell_35/BiasAdd/ReadVariableOp�"lstm_cell_35/MatMul/ReadVariableOp�$lstm_cell_35/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packedc
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedg
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis�
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'���������������������������2
	ReverseV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_35/MatMul/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_35/MatMul/ReadVariableOp�
lstm_cell_35/MatMulMatMulstrided_slice_2:output:0*lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/MatMul�
$lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_35/MatMul_1/ReadVariableOp�
lstm_cell_35/MatMul_1MatMulzeros:output:0,lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/MatMul_1�
lstm_cell_35/addAddV2lstm_cell_35/MatMul:product:0lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/add�
#lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_35/BiasAdd/ReadVariableOp�
lstm_cell_35/BiasAddBiasAddlstm_cell_35/add:z:0+lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/BiasAdd~
lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_35/split/split_dim�
lstm_cell_35/splitSplit%lstm_cell_35/split/split_dim:output:0lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_35/split�
lstm_cell_35/SigmoidSigmoidlstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/Sigmoid�
lstm_cell_35/Sigmoid_1Sigmoidlstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_35/Sigmoid_1�
lstm_cell_35/mulMullstm_cell_35/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/mul}
lstm_cell_35/ReluRelulstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_35/Relu�
lstm_cell_35/mul_1Mullstm_cell_35/Sigmoid:y:0lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/mul_1�
lstm_cell_35/add_1AddV2lstm_cell_35/mul:z:0lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/add_1�
lstm_cell_35/Sigmoid_2Sigmoidlstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_35/Sigmoid_2|
lstm_cell_35/Relu_1Relulstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/Relu_1�
lstm_cell_35/mul_2Mullstm_cell_35/Sigmoid_2:y:0!lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_35_matmul_readvariableop_resource-lstm_cell_35_matmul_1_readvariableop_resource,lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1701937*
condR
while_cond_1701936*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity�
NoOpNoOp$^lstm_cell_35/BiasAdd/ReadVariableOp#^lstm_cell_35/MatMul/ReadVariableOp%^lstm_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_35/BiasAdd/ReadVariableOp#lstm_cell_35/BiasAdd/ReadVariableOp2H
"lstm_cell_35/MatMul/ReadVariableOp"lstm_cell_35/MatMul/ReadVariableOp2L
$lstm_cell_35/MatMul_1/ReadVariableOp$lstm_cell_35/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�?
�
while_body_1705123
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_34_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_34_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_34_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_34_matmul_readvariableop_resource:	�F
3while_lstm_cell_34_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_34_biasadd_readvariableop_resource:	���)while/lstm_cell_34/BiasAdd/ReadVariableOp�(while/lstm_cell_34/MatMul/ReadVariableOp�*while/lstm_cell_34/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_34/MatMul/ReadVariableOp�
while/lstm_cell_34/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/MatMul�
*while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_34/MatMul_1/ReadVariableOp�
while/lstm_cell_34/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/MatMul_1�
while/lstm_cell_34/addAddV2#while/lstm_cell_34/MatMul:product:0%while/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/add�
)while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_34/BiasAdd/ReadVariableOp�
while/lstm_cell_34/BiasAddBiasAddwhile/lstm_cell_34/add:z:01while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/BiasAdd�
"while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_34/split/split_dim�
while/lstm_cell_34/splitSplit+while/lstm_cell_34/split/split_dim:output:0#while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_34/split�
while/lstm_cell_34/SigmoidSigmoid!while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Sigmoid�
while/lstm_cell_34/Sigmoid_1Sigmoid!while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Sigmoid_1�
while/lstm_cell_34/mulMul while/lstm_cell_34/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/mul�
while/lstm_cell_34/ReluRelu!while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Relu�
while/lstm_cell_34/mul_1Mulwhile/lstm_cell_34/Sigmoid:y:0%while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/mul_1�
while/lstm_cell_34/add_1AddV2while/lstm_cell_34/mul:z:0while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/add_1�
while/lstm_cell_34/Sigmoid_2Sigmoid!while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Sigmoid_2�
while/lstm_cell_34/Relu_1Reluwhile/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Relu_1�
while/lstm_cell_34/mul_2Mul while/lstm_cell_34/Sigmoid_2:y:0'while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_34/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_34/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_34/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_34/BiasAdd/ReadVariableOp)^while/lstm_cell_34/MatMul/ReadVariableOp+^while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_34_biasadd_readvariableop_resource4while_lstm_cell_34_biasadd_readvariableop_resource_0"l
3while_lstm_cell_34_matmul_1_readvariableop_resource5while_lstm_cell_34_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_34_matmul_readvariableop_resource3while_lstm_cell_34_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_34/BiasAdd/ReadVariableOp)while/lstm_cell_34/BiasAdd/ReadVariableOp2T
(while/lstm_cell_34/MatMul/ReadVariableOp(while/lstm_cell_34/MatMul/ReadVariableOp2X
*while/lstm_cell_34/MatMul_1/ReadVariableOp*while/lstm_cell_34/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_1701936
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1701936___redundant_placeholder05
1while_while_cond_1701936___redundant_placeholder15
1while_while_cond_1701936___redundant_placeholder25
1while_while_cond_1701936___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
��
�
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_1702680

inputs
inputs_1	N
;forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource:	�P
=forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource:	2�K
<forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource:	�O
<backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource:	�Q
>backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource:	2�L
=backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource:	�
identity��4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp�3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp�5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp�backward_lstm_11/while�3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp�2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp�4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp�forward_lstm_11/while�
$forward_lstm_11/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$forward_lstm_11/RaggedToTensor/zeros�
$forward_lstm_11/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2&
$forward_lstm_11/RaggedToTensor/Const�
3forward_lstm_11/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-forward_lstm_11/RaggedToTensor/Const:output:0inputs-forward_lstm_11/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3forward_lstm_11/RaggedToTensor/RaggedTensorToTensor�
:forward_lstm_11/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack�
<forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1�
<forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2�
4forward_lstm_11/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cforward_lstm_11/RaggedNestedRowLengths/strided_slice/stack:output:0Eforward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1:output:0Eforward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask26
4forward_lstm_11/RaggedNestedRowLengths/strided_slice�
<forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack�
>forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2@
>forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1�
>forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2�
6forward_lstm_11/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Eforward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gforward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gforward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask28
6forward_lstm_11/RaggedNestedRowLengths/strided_slice_1�
*forward_lstm_11/RaggedNestedRowLengths/subSub=forward_lstm_11/RaggedNestedRowLengths/strided_slice:output:0?forward_lstm_11/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2,
*forward_lstm_11/RaggedNestedRowLengths/sub�
forward_lstm_11/CastCast.forward_lstm_11/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
forward_lstm_11/Cast�
forward_lstm_11/ShapeShape<forward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_11/Shape�
#forward_lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_11/strided_slice/stack�
%forward_lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_11/strided_slice/stack_1�
%forward_lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_11/strided_slice/stack_2�
forward_lstm_11/strided_sliceStridedSliceforward_lstm_11/Shape:output:0,forward_lstm_11/strided_slice/stack:output:0.forward_lstm_11/strided_slice/stack_1:output:0.forward_lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_11/strided_slice|
forward_lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_11/zeros/mul/y�
forward_lstm_11/zeros/mulMul&forward_lstm_11/strided_slice:output:0$forward_lstm_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros/mul
forward_lstm_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_11/zeros/Less/y�
forward_lstm_11/zeros/LessLessforward_lstm_11/zeros/mul:z:0%forward_lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros/Less�
forward_lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_11/zeros/packed/1�
forward_lstm_11/zeros/packedPack&forward_lstm_11/strided_slice:output:0'forward_lstm_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_11/zeros/packed�
forward_lstm_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_11/zeros/Const�
forward_lstm_11/zerosFill%forward_lstm_11/zeros/packed:output:0$forward_lstm_11/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_11/zeros�
forward_lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_11/zeros_1/mul/y�
forward_lstm_11/zeros_1/mulMul&forward_lstm_11/strided_slice:output:0&forward_lstm_11/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros_1/mul�
forward_lstm_11/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
forward_lstm_11/zeros_1/Less/y�
forward_lstm_11/zeros_1/LessLessforward_lstm_11/zeros_1/mul:z:0'forward_lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros_1/Less�
 forward_lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_11/zeros_1/packed/1�
forward_lstm_11/zeros_1/packedPack&forward_lstm_11/strided_slice:output:0)forward_lstm_11/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_11/zeros_1/packed�
forward_lstm_11/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_11/zeros_1/Const�
forward_lstm_11/zeros_1Fill'forward_lstm_11/zeros_1/packed:output:0&forward_lstm_11/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_11/zeros_1�
forward_lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_11/transpose/perm�
forward_lstm_11/transpose	Transpose<forward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0'forward_lstm_11/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
forward_lstm_11/transpose
forward_lstm_11/Shape_1Shapeforward_lstm_11/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_11/Shape_1�
%forward_lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_11/strided_slice_1/stack�
'forward_lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_1/stack_1�
'forward_lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_1/stack_2�
forward_lstm_11/strided_slice_1StridedSlice forward_lstm_11/Shape_1:output:0.forward_lstm_11/strided_slice_1/stack:output:00forward_lstm_11/strided_slice_1/stack_1:output:00forward_lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_11/strided_slice_1�
+forward_lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+forward_lstm_11/TensorArrayV2/element_shape�
forward_lstm_11/TensorArrayV2TensorListReserve4forward_lstm_11/TensorArrayV2/element_shape:output:0(forward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_11/TensorArrayV2�
Eforward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2G
Eforward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape�
7forward_lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_11/transpose:y:0Nforward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_11/TensorArrayUnstack/TensorListFromTensor�
%forward_lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_11/strided_slice_2/stack�
'forward_lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_2/stack_1�
'forward_lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_2/stack_2�
forward_lstm_11/strided_slice_2StridedSliceforward_lstm_11/transpose:y:0.forward_lstm_11/strided_slice_2/stack:output:00forward_lstm_11/strided_slice_2/stack_1:output:00forward_lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2!
forward_lstm_11/strided_slice_2�
2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp;forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp�
#forward_lstm_11/lstm_cell_34/MatMulMatMul(forward_lstm_11/strided_slice_2:output:0:forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#forward_lstm_11/lstm_cell_34/MatMul�
4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp=forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype026
4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp�
%forward_lstm_11/lstm_cell_34/MatMul_1MatMulforward_lstm_11/zeros:output:0<forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%forward_lstm_11/lstm_cell_34/MatMul_1�
 forward_lstm_11/lstm_cell_34/addAddV2-forward_lstm_11/lstm_cell_34/MatMul:product:0/forward_lstm_11/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2"
 forward_lstm_11/lstm_cell_34/add�
3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp<forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp�
$forward_lstm_11/lstm_cell_34/BiasAddBiasAdd$forward_lstm_11/lstm_cell_34/add:z:0;forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$forward_lstm_11/lstm_cell_34/BiasAdd�
,forward_lstm_11/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,forward_lstm_11/lstm_cell_34/split/split_dim�
"forward_lstm_11/lstm_cell_34/splitSplit5forward_lstm_11/lstm_cell_34/split/split_dim:output:0-forward_lstm_11/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2$
"forward_lstm_11/lstm_cell_34/split�
$forward_lstm_11/lstm_cell_34/SigmoidSigmoid+forward_lstm_11/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22&
$forward_lstm_11/lstm_cell_34/Sigmoid�
&forward_lstm_11/lstm_cell_34/Sigmoid_1Sigmoid+forward_lstm_11/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22(
&forward_lstm_11/lstm_cell_34/Sigmoid_1�
 forward_lstm_11/lstm_cell_34/mulMul*forward_lstm_11/lstm_cell_34/Sigmoid_1:y:0 forward_lstm_11/zeros_1:output:0*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/lstm_cell_34/mul�
!forward_lstm_11/lstm_cell_34/ReluRelu+forward_lstm_11/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22#
!forward_lstm_11/lstm_cell_34/Relu�
"forward_lstm_11/lstm_cell_34/mul_1Mul(forward_lstm_11/lstm_cell_34/Sigmoid:y:0/forward_lstm_11/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_11/lstm_cell_34/mul_1�
"forward_lstm_11/lstm_cell_34/add_1AddV2$forward_lstm_11/lstm_cell_34/mul:z:0&forward_lstm_11/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_11/lstm_cell_34/add_1�
&forward_lstm_11/lstm_cell_34/Sigmoid_2Sigmoid+forward_lstm_11/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22(
&forward_lstm_11/lstm_cell_34/Sigmoid_2�
#forward_lstm_11/lstm_cell_34/Relu_1Relu&forward_lstm_11/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22%
#forward_lstm_11/lstm_cell_34/Relu_1�
"forward_lstm_11/lstm_cell_34/mul_2Mul*forward_lstm_11/lstm_cell_34/Sigmoid_2:y:01forward_lstm_11/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_11/lstm_cell_34/mul_2�
-forward_lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2/
-forward_lstm_11/TensorArrayV2_1/element_shape�
forward_lstm_11/TensorArrayV2_1TensorListReserve6forward_lstm_11/TensorArrayV2_1/element_shape:output:0(forward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_11/TensorArrayV2_1n
forward_lstm_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_11/time�
forward_lstm_11/zeros_like	ZerosLike&forward_lstm_11/lstm_cell_34/mul_2:z:0*
T0*'
_output_shapes
:���������22
forward_lstm_11/zeros_like�
(forward_lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(forward_lstm_11/while/maximum_iterations�
"forward_lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_11/while/loop_counter�
forward_lstm_11/whileWhile+forward_lstm_11/while/loop_counter:output:01forward_lstm_11/while/maximum_iterations:output:0forward_lstm_11/time:output:0(forward_lstm_11/TensorArrayV2_1:handle:0forward_lstm_11/zeros_like:y:0forward_lstm_11/zeros:output:0 forward_lstm_11/zeros_1:output:0(forward_lstm_11/strided_slice_1:output:0Gforward_lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_11/Cast:y:0;forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource=forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource<forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *.
body&R$
"forward_lstm_11_while_body_1702404*.
cond&R$
"forward_lstm_11_while_cond_1702403*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
forward_lstm_11/while�
@forward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2B
@forward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape�
2forward_lstm_11/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_11/while:output:3Iforward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype024
2forward_lstm_11/TensorArrayV2Stack/TensorListStack�
%forward_lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2'
%forward_lstm_11/strided_slice_3/stack�
'forward_lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_11/strided_slice_3/stack_1�
'forward_lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_3/stack_2�
forward_lstm_11/strided_slice_3StridedSlice;forward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_11/strided_slice_3/stack:output:00forward_lstm_11/strided_slice_3/stack_1:output:00forward_lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2!
forward_lstm_11/strided_slice_3�
 forward_lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_11/transpose_1/perm�
forward_lstm_11/transpose_1	Transpose;forward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_11/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
forward_lstm_11/transpose_1�
forward_lstm_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_11/runtime�
%backward_lstm_11/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%backward_lstm_11/RaggedToTensor/zeros�
%backward_lstm_11/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2'
%backward_lstm_11/RaggedToTensor/Const�
4backward_lstm_11/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.backward_lstm_11/RaggedToTensor/Const:output:0inputs.backward_lstm_11/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4backward_lstm_11/RaggedToTensor/RaggedTensorToTensor�
;backward_lstm_11/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack�
=backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1�
=backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2�
5backward_lstm_11/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dbackward_lstm_11/RaggedNestedRowLengths/strided_slice/stack:output:0Fbackward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fbackward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask27
5backward_lstm_11/RaggedNestedRowLengths/strided_slice�
=backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack�
?backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2A
?backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1�
?backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2�
7backward_lstm_11/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fbackward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hbackward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hbackward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask29
7backward_lstm_11/RaggedNestedRowLengths/strided_slice_1�
+backward_lstm_11/RaggedNestedRowLengths/subSub>backward_lstm_11/RaggedNestedRowLengths/strided_slice:output:0@backward_lstm_11/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2-
+backward_lstm_11/RaggedNestedRowLengths/sub�
backward_lstm_11/CastCast/backward_lstm_11/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
backward_lstm_11/Cast�
backward_lstm_11/ShapeShape=backward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_11/Shape�
$backward_lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_11/strided_slice/stack�
&backward_lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_11/strided_slice/stack_1�
&backward_lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_11/strided_slice/stack_2�
backward_lstm_11/strided_sliceStridedSlicebackward_lstm_11/Shape:output:0-backward_lstm_11/strided_slice/stack:output:0/backward_lstm_11/strided_slice/stack_1:output:0/backward_lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_11/strided_slice~
backward_lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_11/zeros/mul/y�
backward_lstm_11/zeros/mulMul'backward_lstm_11/strided_slice:output:0%backward_lstm_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros/mul�
backward_lstm_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
backward_lstm_11/zeros/Less/y�
backward_lstm_11/zeros/LessLessbackward_lstm_11/zeros/mul:z:0&backward_lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros/Less�
backward_lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_11/zeros/packed/1�
backward_lstm_11/zeros/packedPack'backward_lstm_11/strided_slice:output:0(backward_lstm_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_11/zeros/packed�
backward_lstm_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_11/zeros/Const�
backward_lstm_11/zerosFill&backward_lstm_11/zeros/packed:output:0%backward_lstm_11/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_11/zeros�
backward_lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_11/zeros_1/mul/y�
backward_lstm_11/zeros_1/mulMul'backward_lstm_11/strided_slice:output:0'backward_lstm_11/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros_1/mul�
backward_lstm_11/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2!
backward_lstm_11/zeros_1/Less/y�
backward_lstm_11/zeros_1/LessLess backward_lstm_11/zeros_1/mul:z:0(backward_lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros_1/Less�
!backward_lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_11/zeros_1/packed/1�
backward_lstm_11/zeros_1/packedPack'backward_lstm_11/strided_slice:output:0*backward_lstm_11/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_11/zeros_1/packed�
backward_lstm_11/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_11/zeros_1/Const�
backward_lstm_11/zeros_1Fill(backward_lstm_11/zeros_1/packed:output:0'backward_lstm_11/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_11/zeros_1�
backward_lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_11/transpose/perm�
backward_lstm_11/transpose	Transpose=backward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0(backward_lstm_11/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_11/transpose�
backward_lstm_11/Shape_1Shapebackward_lstm_11/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_11/Shape_1�
&backward_lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_11/strided_slice_1/stack�
(backward_lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_1/stack_1�
(backward_lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_1/stack_2�
 backward_lstm_11/strided_slice_1StridedSlice!backward_lstm_11/Shape_1:output:0/backward_lstm_11/strided_slice_1/stack:output:01backward_lstm_11/strided_slice_1/stack_1:output:01backward_lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_11/strided_slice_1�
,backward_lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2.
,backward_lstm_11/TensorArrayV2/element_shape�
backward_lstm_11/TensorArrayV2TensorListReserve5backward_lstm_11/TensorArrayV2/element_shape:output:0)backward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_11/TensorArrayV2�
backward_lstm_11/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_11/ReverseV2/axis�
backward_lstm_11/ReverseV2	ReverseV2backward_lstm_11/transpose:y:0(backward_lstm_11/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_11/ReverseV2�
Fbackward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2H
Fbackward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape�
8backward_lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_11/ReverseV2:output:0Obackward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_11/TensorArrayUnstack/TensorListFromTensor�
&backward_lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_11/strided_slice_2/stack�
(backward_lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_2/stack_1�
(backward_lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_2/stack_2�
 backward_lstm_11/strided_slice_2StridedSlicebackward_lstm_11/transpose:y:0/backward_lstm_11/strided_slice_2/stack:output:01backward_lstm_11/strided_slice_2/stack_1:output:01backward_lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2"
 backward_lstm_11/strided_slice_2�
3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp<backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype025
3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp�
$backward_lstm_11/lstm_cell_35/MatMulMatMul)backward_lstm_11/strided_slice_2:output:0;backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$backward_lstm_11/lstm_cell_35/MatMul�
5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp>backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype027
5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp�
&backward_lstm_11/lstm_cell_35/MatMul_1MatMulbackward_lstm_11/zeros:output:0=backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&backward_lstm_11/lstm_cell_35/MatMul_1�
!backward_lstm_11/lstm_cell_35/addAddV2.backward_lstm_11/lstm_cell_35/MatMul:product:00backward_lstm_11/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2#
!backward_lstm_11/lstm_cell_35/add�
4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp=backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype026
4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp�
%backward_lstm_11/lstm_cell_35/BiasAddBiasAdd%backward_lstm_11/lstm_cell_35/add:z:0<backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%backward_lstm_11/lstm_cell_35/BiasAdd�
-backward_lstm_11/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-backward_lstm_11/lstm_cell_35/split/split_dim�
#backward_lstm_11/lstm_cell_35/splitSplit6backward_lstm_11/lstm_cell_35/split/split_dim:output:0.backward_lstm_11/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2%
#backward_lstm_11/lstm_cell_35/split�
%backward_lstm_11/lstm_cell_35/SigmoidSigmoid,backward_lstm_11/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22'
%backward_lstm_11/lstm_cell_35/Sigmoid�
'backward_lstm_11/lstm_cell_35/Sigmoid_1Sigmoid,backward_lstm_11/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22)
'backward_lstm_11/lstm_cell_35/Sigmoid_1�
!backward_lstm_11/lstm_cell_35/mulMul+backward_lstm_11/lstm_cell_35/Sigmoid_1:y:0!backward_lstm_11/zeros_1:output:0*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/lstm_cell_35/mul�
"backward_lstm_11/lstm_cell_35/ReluRelu,backward_lstm_11/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22$
"backward_lstm_11/lstm_cell_35/Relu�
#backward_lstm_11/lstm_cell_35/mul_1Mul)backward_lstm_11/lstm_cell_35/Sigmoid:y:00backward_lstm_11/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_11/lstm_cell_35/mul_1�
#backward_lstm_11/lstm_cell_35/add_1AddV2%backward_lstm_11/lstm_cell_35/mul:z:0'backward_lstm_11/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_11/lstm_cell_35/add_1�
'backward_lstm_11/lstm_cell_35/Sigmoid_2Sigmoid,backward_lstm_11/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22)
'backward_lstm_11/lstm_cell_35/Sigmoid_2�
$backward_lstm_11/lstm_cell_35/Relu_1Relu'backward_lstm_11/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22&
$backward_lstm_11/lstm_cell_35/Relu_1�
#backward_lstm_11/lstm_cell_35/mul_2Mul+backward_lstm_11/lstm_cell_35/Sigmoid_2:y:02backward_lstm_11/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_11/lstm_cell_35/mul_2�
.backward_lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   20
.backward_lstm_11/TensorArrayV2_1/element_shape�
 backward_lstm_11/TensorArrayV2_1TensorListReserve7backward_lstm_11/TensorArrayV2_1/element_shape:output:0)backward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_11/TensorArrayV2_1p
backward_lstm_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_11/time�
&backward_lstm_11/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_lstm_11/Max/reduction_indices�
backward_lstm_11/MaxMaxbackward_lstm_11/Cast:y:0/backward_lstm_11/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/Maxr
backward_lstm_11/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_11/sub/y�
backward_lstm_11/subSubbackward_lstm_11/Max:output:0backward_lstm_11/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/sub�
backward_lstm_11/Sub_1Subbackward_lstm_11/sub:z:0backward_lstm_11/Cast:y:0*
T0*#
_output_shapes
:���������2
backward_lstm_11/Sub_1�
backward_lstm_11/zeros_like	ZerosLike'backward_lstm_11/lstm_cell_35/mul_2:z:0*
T0*'
_output_shapes
:���������22
backward_lstm_11/zeros_like�
)backward_lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2+
)backward_lstm_11/while/maximum_iterations�
#backward_lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_11/while/loop_counter�	
backward_lstm_11/whileWhile,backward_lstm_11/while/loop_counter:output:02backward_lstm_11/while/maximum_iterations:output:0backward_lstm_11/time:output:0)backward_lstm_11/TensorArrayV2_1:handle:0backward_lstm_11/zeros_like:y:0backward_lstm_11/zeros:output:0!backward_lstm_11/zeros_1:output:0)backward_lstm_11/strided_slice_1:output:0Hbackward_lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_11/Sub_1:z:0<backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource>backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource=backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( */
body'R%
#backward_lstm_11_while_body_1702583*/
cond'R%
#backward_lstm_11_while_cond_1702582*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
backward_lstm_11/while�
Abackward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2C
Abackward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape�
3backward_lstm_11/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_11/while:output:3Jbackward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype025
3backward_lstm_11/TensorArrayV2Stack/TensorListStack�
&backward_lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2(
&backward_lstm_11/strided_slice_3/stack�
(backward_lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_11/strided_slice_3/stack_1�
(backward_lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_3/stack_2�
 backward_lstm_11/strided_slice_3StridedSlice<backward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_11/strided_slice_3/stack:output:01backward_lstm_11/strided_slice_3/stack_1:output:01backward_lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2"
 backward_lstm_11/strided_slice_3�
!backward_lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_11/transpose_1/perm�
backward_lstm_11/transpose_1	Transpose<backward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_11/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
backward_lstm_11/transpose_1�
backward_lstm_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_11/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2(forward_lstm_11/strided_slice_3:output:0)backward_lstm_11/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity�
NoOpNoOp5^backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp4^backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp6^backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp^backward_lstm_11/while4^forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp3^forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp5^forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp^forward_lstm_11/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������: : : : : : 2l
4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp2j
3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp2n
5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp20
backward_lstm_11/whilebackward_lstm_11/while2j
3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp2h
2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp2l
4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp2.
forward_lstm_11/whileforward_lstm_11/while:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�\
�
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_1705207

inputs>
+lstm_cell_34_matmul_readvariableop_resource:	�@
-lstm_cell_34_matmul_1_readvariableop_resource:	2�;
,lstm_cell_34_biasadd_readvariableop_resource:	�
identity��#lstm_cell_34/BiasAdd/ReadVariableOp�"lstm_cell_34/MatMul/ReadVariableOp�$lstm_cell_34/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packedc
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedg
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_34/MatMul/ReadVariableOpReadVariableOp+lstm_cell_34_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_34/MatMul/ReadVariableOp�
lstm_cell_34/MatMulMatMulstrided_slice_2:output:0*lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/MatMul�
$lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_34/MatMul_1/ReadVariableOp�
lstm_cell_34/MatMul_1MatMulzeros:output:0,lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/MatMul_1�
lstm_cell_34/addAddV2lstm_cell_34/MatMul:product:0lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/add�
#lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_34/BiasAdd/ReadVariableOp�
lstm_cell_34/BiasAddBiasAddlstm_cell_34/add:z:0+lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/BiasAdd~
lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_34/split/split_dim�
lstm_cell_34/splitSplit%lstm_cell_34/split/split_dim:output:0lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_34/split�
lstm_cell_34/SigmoidSigmoidlstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/Sigmoid�
lstm_cell_34/Sigmoid_1Sigmoidlstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_34/Sigmoid_1�
lstm_cell_34/mulMullstm_cell_34/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/mul}
lstm_cell_34/ReluRelulstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_34/Relu�
lstm_cell_34/mul_1Mullstm_cell_34/Sigmoid:y:0lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/mul_1�
lstm_cell_34/add_1AddV2lstm_cell_34/mul:z:0lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/add_1�
lstm_cell_34/Sigmoid_2Sigmoidlstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_34/Sigmoid_2|
lstm_cell_34/Relu_1Relulstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/Relu_1�
lstm_cell_34/mul_2Mullstm_cell_34/Sigmoid_2:y:0!lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_34_matmul_readvariableop_resource-lstm_cell_34_matmul_1_readvariableop_resource,lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1705123*
condR
while_cond_1705122*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity�
NoOpNoOp$^lstm_cell_34/BiasAdd/ReadVariableOp#^lstm_cell_34/MatMul/ReadVariableOp%^lstm_cell_34/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_34/BiasAdd/ReadVariableOp#lstm_cell_34/BiasAdd/ReadVariableOp2H
"lstm_cell_34/MatMul/ReadVariableOp"lstm_cell_34/MatMul/ReadVariableOp2L
$lstm_cell_34/MatMul_1/ReadVariableOp$lstm_cell_34/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
��
�
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_1703672
inputs_0N
;forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource:	�P
=forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource:	2�K
<forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource:	�O
<backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource:	�Q
>backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource:	2�L
=backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource:	�
identity��4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp�3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp�5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp�backward_lstm_11/while�3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp�2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp�4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp�forward_lstm_11/whilef
forward_lstm_11/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_11/Shape�
#forward_lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_11/strided_slice/stack�
%forward_lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_11/strided_slice/stack_1�
%forward_lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_11/strided_slice/stack_2�
forward_lstm_11/strided_sliceStridedSliceforward_lstm_11/Shape:output:0,forward_lstm_11/strided_slice/stack:output:0.forward_lstm_11/strided_slice/stack_1:output:0.forward_lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_11/strided_slice|
forward_lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_11/zeros/mul/y�
forward_lstm_11/zeros/mulMul&forward_lstm_11/strided_slice:output:0$forward_lstm_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros/mul
forward_lstm_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_11/zeros/Less/y�
forward_lstm_11/zeros/LessLessforward_lstm_11/zeros/mul:z:0%forward_lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros/Less�
forward_lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_11/zeros/packed/1�
forward_lstm_11/zeros/packedPack&forward_lstm_11/strided_slice:output:0'forward_lstm_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_11/zeros/packed�
forward_lstm_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_11/zeros/Const�
forward_lstm_11/zerosFill%forward_lstm_11/zeros/packed:output:0$forward_lstm_11/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_11/zeros�
forward_lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_11/zeros_1/mul/y�
forward_lstm_11/zeros_1/mulMul&forward_lstm_11/strided_slice:output:0&forward_lstm_11/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros_1/mul�
forward_lstm_11/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
forward_lstm_11/zeros_1/Less/y�
forward_lstm_11/zeros_1/LessLessforward_lstm_11/zeros_1/mul:z:0'forward_lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros_1/Less�
 forward_lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_11/zeros_1/packed/1�
forward_lstm_11/zeros_1/packedPack&forward_lstm_11/strided_slice:output:0)forward_lstm_11/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_11/zeros_1/packed�
forward_lstm_11/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_11/zeros_1/Const�
forward_lstm_11/zeros_1Fill'forward_lstm_11/zeros_1/packed:output:0&forward_lstm_11/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_11/zeros_1�
forward_lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_11/transpose/perm�
forward_lstm_11/transpose	Transposeinputs_0'forward_lstm_11/transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
forward_lstm_11/transpose
forward_lstm_11/Shape_1Shapeforward_lstm_11/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_11/Shape_1�
%forward_lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_11/strided_slice_1/stack�
'forward_lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_1/stack_1�
'forward_lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_1/stack_2�
forward_lstm_11/strided_slice_1StridedSlice forward_lstm_11/Shape_1:output:0.forward_lstm_11/strided_slice_1/stack:output:00forward_lstm_11/strided_slice_1/stack_1:output:00forward_lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_11/strided_slice_1�
+forward_lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+forward_lstm_11/TensorArrayV2/element_shape�
forward_lstm_11/TensorArrayV2TensorListReserve4forward_lstm_11/TensorArrayV2/element_shape:output:0(forward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_11/TensorArrayV2�
Eforward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2G
Eforward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape�
7forward_lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_11/transpose:y:0Nforward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_11/TensorArrayUnstack/TensorListFromTensor�
%forward_lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_11/strided_slice_2/stack�
'forward_lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_2/stack_1�
'forward_lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_2/stack_2�
forward_lstm_11/strided_slice_2StridedSliceforward_lstm_11/transpose:y:0.forward_lstm_11/strided_slice_2/stack:output:00forward_lstm_11/strided_slice_2/stack_1:output:00forward_lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2!
forward_lstm_11/strided_slice_2�
2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp;forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp�
#forward_lstm_11/lstm_cell_34/MatMulMatMul(forward_lstm_11/strided_slice_2:output:0:forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#forward_lstm_11/lstm_cell_34/MatMul�
4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp=forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype026
4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp�
%forward_lstm_11/lstm_cell_34/MatMul_1MatMulforward_lstm_11/zeros:output:0<forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%forward_lstm_11/lstm_cell_34/MatMul_1�
 forward_lstm_11/lstm_cell_34/addAddV2-forward_lstm_11/lstm_cell_34/MatMul:product:0/forward_lstm_11/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2"
 forward_lstm_11/lstm_cell_34/add�
3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp<forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp�
$forward_lstm_11/lstm_cell_34/BiasAddBiasAdd$forward_lstm_11/lstm_cell_34/add:z:0;forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$forward_lstm_11/lstm_cell_34/BiasAdd�
,forward_lstm_11/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,forward_lstm_11/lstm_cell_34/split/split_dim�
"forward_lstm_11/lstm_cell_34/splitSplit5forward_lstm_11/lstm_cell_34/split/split_dim:output:0-forward_lstm_11/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2$
"forward_lstm_11/lstm_cell_34/split�
$forward_lstm_11/lstm_cell_34/SigmoidSigmoid+forward_lstm_11/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22&
$forward_lstm_11/lstm_cell_34/Sigmoid�
&forward_lstm_11/lstm_cell_34/Sigmoid_1Sigmoid+forward_lstm_11/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22(
&forward_lstm_11/lstm_cell_34/Sigmoid_1�
 forward_lstm_11/lstm_cell_34/mulMul*forward_lstm_11/lstm_cell_34/Sigmoid_1:y:0 forward_lstm_11/zeros_1:output:0*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/lstm_cell_34/mul�
!forward_lstm_11/lstm_cell_34/ReluRelu+forward_lstm_11/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22#
!forward_lstm_11/lstm_cell_34/Relu�
"forward_lstm_11/lstm_cell_34/mul_1Mul(forward_lstm_11/lstm_cell_34/Sigmoid:y:0/forward_lstm_11/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_11/lstm_cell_34/mul_1�
"forward_lstm_11/lstm_cell_34/add_1AddV2$forward_lstm_11/lstm_cell_34/mul:z:0&forward_lstm_11/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_11/lstm_cell_34/add_1�
&forward_lstm_11/lstm_cell_34/Sigmoid_2Sigmoid+forward_lstm_11/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22(
&forward_lstm_11/lstm_cell_34/Sigmoid_2�
#forward_lstm_11/lstm_cell_34/Relu_1Relu&forward_lstm_11/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22%
#forward_lstm_11/lstm_cell_34/Relu_1�
"forward_lstm_11/lstm_cell_34/mul_2Mul*forward_lstm_11/lstm_cell_34/Sigmoid_2:y:01forward_lstm_11/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_11/lstm_cell_34/mul_2�
-forward_lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2/
-forward_lstm_11/TensorArrayV2_1/element_shape�
forward_lstm_11/TensorArrayV2_1TensorListReserve6forward_lstm_11/TensorArrayV2_1/element_shape:output:0(forward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_11/TensorArrayV2_1n
forward_lstm_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_11/time�
(forward_lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(forward_lstm_11/while/maximum_iterations�
"forward_lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_11/while/loop_counter�
forward_lstm_11/whileWhile+forward_lstm_11/while/loop_counter:output:01forward_lstm_11/while/maximum_iterations:output:0forward_lstm_11/time:output:0(forward_lstm_11/TensorArrayV2_1:handle:0forward_lstm_11/zeros:output:0 forward_lstm_11/zeros_1:output:0(forward_lstm_11/strided_slice_1:output:0Gforward_lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0;forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource=forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource<forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *.
body&R$
"forward_lstm_11_while_body_1703437*.
cond&R$
"forward_lstm_11_while_cond_1703436*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
forward_lstm_11/while�
@forward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2B
@forward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape�
2forward_lstm_11/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_11/while:output:3Iforward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype024
2forward_lstm_11/TensorArrayV2Stack/TensorListStack�
%forward_lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2'
%forward_lstm_11/strided_slice_3/stack�
'forward_lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_11/strided_slice_3/stack_1�
'forward_lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_3/stack_2�
forward_lstm_11/strided_slice_3StridedSlice;forward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_11/strided_slice_3/stack:output:00forward_lstm_11/strided_slice_3/stack_1:output:00forward_lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2!
forward_lstm_11/strided_slice_3�
 forward_lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_11/transpose_1/perm�
forward_lstm_11/transpose_1	Transpose;forward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_11/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
forward_lstm_11/transpose_1�
forward_lstm_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_11/runtimeh
backward_lstm_11/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_11/Shape�
$backward_lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_11/strided_slice/stack�
&backward_lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_11/strided_slice/stack_1�
&backward_lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_11/strided_slice/stack_2�
backward_lstm_11/strided_sliceStridedSlicebackward_lstm_11/Shape:output:0-backward_lstm_11/strided_slice/stack:output:0/backward_lstm_11/strided_slice/stack_1:output:0/backward_lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_11/strided_slice~
backward_lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_11/zeros/mul/y�
backward_lstm_11/zeros/mulMul'backward_lstm_11/strided_slice:output:0%backward_lstm_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros/mul�
backward_lstm_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
backward_lstm_11/zeros/Less/y�
backward_lstm_11/zeros/LessLessbackward_lstm_11/zeros/mul:z:0&backward_lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros/Less�
backward_lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_11/zeros/packed/1�
backward_lstm_11/zeros/packedPack'backward_lstm_11/strided_slice:output:0(backward_lstm_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_11/zeros/packed�
backward_lstm_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_11/zeros/Const�
backward_lstm_11/zerosFill&backward_lstm_11/zeros/packed:output:0%backward_lstm_11/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_11/zeros�
backward_lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_11/zeros_1/mul/y�
backward_lstm_11/zeros_1/mulMul'backward_lstm_11/strided_slice:output:0'backward_lstm_11/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros_1/mul�
backward_lstm_11/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2!
backward_lstm_11/zeros_1/Less/y�
backward_lstm_11/zeros_1/LessLess backward_lstm_11/zeros_1/mul:z:0(backward_lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros_1/Less�
!backward_lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_11/zeros_1/packed/1�
backward_lstm_11/zeros_1/packedPack'backward_lstm_11/strided_slice:output:0*backward_lstm_11/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_11/zeros_1/packed�
backward_lstm_11/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_11/zeros_1/Const�
backward_lstm_11/zeros_1Fill(backward_lstm_11/zeros_1/packed:output:0'backward_lstm_11/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_11/zeros_1�
backward_lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_11/transpose/perm�
backward_lstm_11/transpose	Transposeinputs_0(backward_lstm_11/transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
backward_lstm_11/transpose�
backward_lstm_11/Shape_1Shapebackward_lstm_11/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_11/Shape_1�
&backward_lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_11/strided_slice_1/stack�
(backward_lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_1/stack_1�
(backward_lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_1/stack_2�
 backward_lstm_11/strided_slice_1StridedSlice!backward_lstm_11/Shape_1:output:0/backward_lstm_11/strided_slice_1/stack:output:01backward_lstm_11/strided_slice_1/stack_1:output:01backward_lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_11/strided_slice_1�
,backward_lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2.
,backward_lstm_11/TensorArrayV2/element_shape�
backward_lstm_11/TensorArrayV2TensorListReserve5backward_lstm_11/TensorArrayV2/element_shape:output:0)backward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_11/TensorArrayV2�
backward_lstm_11/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_11/ReverseV2/axis�
backward_lstm_11/ReverseV2	ReverseV2backward_lstm_11/transpose:y:0(backward_lstm_11/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'���������������������������2
backward_lstm_11/ReverseV2�
Fbackward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2H
Fbackward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape�
8backward_lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_11/ReverseV2:output:0Obackward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_11/TensorArrayUnstack/TensorListFromTensor�
&backward_lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_11/strided_slice_2/stack�
(backward_lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_2/stack_1�
(backward_lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_2/stack_2�
 backward_lstm_11/strided_slice_2StridedSlicebackward_lstm_11/transpose:y:0/backward_lstm_11/strided_slice_2/stack:output:01backward_lstm_11/strided_slice_2/stack_1:output:01backward_lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2"
 backward_lstm_11/strided_slice_2�
3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp<backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype025
3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp�
$backward_lstm_11/lstm_cell_35/MatMulMatMul)backward_lstm_11/strided_slice_2:output:0;backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$backward_lstm_11/lstm_cell_35/MatMul�
5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp>backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype027
5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp�
&backward_lstm_11/lstm_cell_35/MatMul_1MatMulbackward_lstm_11/zeros:output:0=backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&backward_lstm_11/lstm_cell_35/MatMul_1�
!backward_lstm_11/lstm_cell_35/addAddV2.backward_lstm_11/lstm_cell_35/MatMul:product:00backward_lstm_11/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2#
!backward_lstm_11/lstm_cell_35/add�
4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp=backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype026
4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp�
%backward_lstm_11/lstm_cell_35/BiasAddBiasAdd%backward_lstm_11/lstm_cell_35/add:z:0<backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%backward_lstm_11/lstm_cell_35/BiasAdd�
-backward_lstm_11/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-backward_lstm_11/lstm_cell_35/split/split_dim�
#backward_lstm_11/lstm_cell_35/splitSplit6backward_lstm_11/lstm_cell_35/split/split_dim:output:0.backward_lstm_11/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2%
#backward_lstm_11/lstm_cell_35/split�
%backward_lstm_11/lstm_cell_35/SigmoidSigmoid,backward_lstm_11/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22'
%backward_lstm_11/lstm_cell_35/Sigmoid�
'backward_lstm_11/lstm_cell_35/Sigmoid_1Sigmoid,backward_lstm_11/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22)
'backward_lstm_11/lstm_cell_35/Sigmoid_1�
!backward_lstm_11/lstm_cell_35/mulMul+backward_lstm_11/lstm_cell_35/Sigmoid_1:y:0!backward_lstm_11/zeros_1:output:0*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/lstm_cell_35/mul�
"backward_lstm_11/lstm_cell_35/ReluRelu,backward_lstm_11/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22$
"backward_lstm_11/lstm_cell_35/Relu�
#backward_lstm_11/lstm_cell_35/mul_1Mul)backward_lstm_11/lstm_cell_35/Sigmoid:y:00backward_lstm_11/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_11/lstm_cell_35/mul_1�
#backward_lstm_11/lstm_cell_35/add_1AddV2%backward_lstm_11/lstm_cell_35/mul:z:0'backward_lstm_11/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_11/lstm_cell_35/add_1�
'backward_lstm_11/lstm_cell_35/Sigmoid_2Sigmoid,backward_lstm_11/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22)
'backward_lstm_11/lstm_cell_35/Sigmoid_2�
$backward_lstm_11/lstm_cell_35/Relu_1Relu'backward_lstm_11/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22&
$backward_lstm_11/lstm_cell_35/Relu_1�
#backward_lstm_11/lstm_cell_35/mul_2Mul+backward_lstm_11/lstm_cell_35/Sigmoid_2:y:02backward_lstm_11/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_11/lstm_cell_35/mul_2�
.backward_lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   20
.backward_lstm_11/TensorArrayV2_1/element_shape�
 backward_lstm_11/TensorArrayV2_1TensorListReserve7backward_lstm_11/TensorArrayV2_1/element_shape:output:0)backward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_11/TensorArrayV2_1p
backward_lstm_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_11/time�
)backward_lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2+
)backward_lstm_11/while/maximum_iterations�
#backward_lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_11/while/loop_counter�
backward_lstm_11/whileWhile,backward_lstm_11/while/loop_counter:output:02backward_lstm_11/while/maximum_iterations:output:0backward_lstm_11/time:output:0)backward_lstm_11/TensorArrayV2_1:handle:0backward_lstm_11/zeros:output:0!backward_lstm_11/zeros_1:output:0)backward_lstm_11/strided_slice_1:output:0Hbackward_lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0<backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource>backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource=backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( */
body'R%
#backward_lstm_11_while_body_1703586*/
cond'R%
#backward_lstm_11_while_cond_1703585*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
backward_lstm_11/while�
Abackward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2C
Abackward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape�
3backward_lstm_11/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_11/while:output:3Jbackward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype025
3backward_lstm_11/TensorArrayV2Stack/TensorListStack�
&backward_lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2(
&backward_lstm_11/strided_slice_3/stack�
(backward_lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_11/strided_slice_3/stack_1�
(backward_lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_3/stack_2�
 backward_lstm_11/strided_slice_3StridedSlice<backward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_11/strided_slice_3/stack:output:01backward_lstm_11/strided_slice_3/stack_1:output:01backward_lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2"
 backward_lstm_11/strided_slice_3�
!backward_lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_11/transpose_1/perm�
backward_lstm_11/transpose_1	Transpose<backward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_11/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
backward_lstm_11/transpose_1�
backward_lstm_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_11/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2(forward_lstm_11/strided_slice_3:output:0)backward_lstm_11/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity�
NoOpNoOp5^backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp4^backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp6^backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp^backward_lstm_11/while4^forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp3^forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp5^forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp^forward_lstm_11/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 2l
4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp2j
3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp2n
5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp20
backward_lstm_11/whilebackward_lstm_11/while2j
3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp2h
2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp2l
4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp2.
forward_lstm_11/whileforward_lstm_11/while:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs/0
�H
�
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_1701034

inputs'
lstm_cell_35_1700952:	�'
lstm_cell_35_1700954:	2�#
lstm_cell_35_1700956:	�
identity��$lstm_cell_35/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packedc
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedg
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis�
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������2
	ReverseV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
$lstm_cell_35/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_35_1700952lstm_cell_35_1700954lstm_cell_35_1700956*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������2:���������2:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_lstm_cell_35_layer_call_and_return_conditional_losses_17009512&
$lstm_cell_35/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_35_1700952lstm_cell_35_1700954lstm_cell_35_1700956*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1700965*
condR
while_cond_1700964*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity}
NoOpNoOp%^lstm_cell_35/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_35/StatefulPartitionedCall$lstm_cell_35/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
"forward_lstm_11_while_cond_1704413<
8forward_lstm_11_while_forward_lstm_11_while_loop_counterB
>forward_lstm_11_while_forward_lstm_11_while_maximum_iterations%
!forward_lstm_11_while_placeholder'
#forward_lstm_11_while_placeholder_1'
#forward_lstm_11_while_placeholder_2'
#forward_lstm_11_while_placeholder_3'
#forward_lstm_11_while_placeholder_4>
:forward_lstm_11_while_less_forward_lstm_11_strided_slice_1U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1704413___redundant_placeholder0U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1704413___redundant_placeholder1U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1704413___redundant_placeholder2U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1704413___redundant_placeholder3U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1704413___redundant_placeholder4"
forward_lstm_11_while_identity
�
forward_lstm_11/while/LessLess!forward_lstm_11_while_placeholder:forward_lstm_11_while_less_forward_lstm_11_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_11/while/Less�
forward_lstm_11/while/IdentityIdentityforward_lstm_11/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_11/while/Identity"I
forward_lstm_11_while_identity'forward_lstm_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :���������2:���������2:���������2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
�?
�
while_body_1701585
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_34_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_34_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_34_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_34_matmul_readvariableop_resource:	�F
3while_lstm_cell_34_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_34_biasadd_readvariableop_resource:	���)while/lstm_cell_34/BiasAdd/ReadVariableOp�(while/lstm_cell_34/MatMul/ReadVariableOp�*while/lstm_cell_34/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_34/MatMul/ReadVariableOp�
while/lstm_cell_34/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/MatMul�
*while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_34/MatMul_1/ReadVariableOp�
while/lstm_cell_34/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/MatMul_1�
while/lstm_cell_34/addAddV2#while/lstm_cell_34/MatMul:product:0%while/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/add�
)while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_34/BiasAdd/ReadVariableOp�
while/lstm_cell_34/BiasAddBiasAddwhile/lstm_cell_34/add:z:01while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/BiasAdd�
"while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_34/split/split_dim�
while/lstm_cell_34/splitSplit+while/lstm_cell_34/split/split_dim:output:0#while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_34/split�
while/lstm_cell_34/SigmoidSigmoid!while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Sigmoid�
while/lstm_cell_34/Sigmoid_1Sigmoid!while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Sigmoid_1�
while/lstm_cell_34/mulMul while/lstm_cell_34/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/mul�
while/lstm_cell_34/ReluRelu!while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Relu�
while/lstm_cell_34/mul_1Mulwhile/lstm_cell_34/Sigmoid:y:0%while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/mul_1�
while/lstm_cell_34/add_1AddV2while/lstm_cell_34/mul:z:0while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/add_1�
while/lstm_cell_34/Sigmoid_2Sigmoid!while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Sigmoid_2�
while/lstm_cell_34/Relu_1Reluwhile/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Relu_1�
while/lstm_cell_34/mul_2Mul while/lstm_cell_34/Sigmoid_2:y:0'while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_34/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_34/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_34/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_34/BiasAdd/ReadVariableOp)^while/lstm_cell_34/MatMul/ReadVariableOp+^while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_34_biasadd_readvariableop_resource4while_lstm_cell_34_biasadd_readvariableop_resource_0"l
3while_lstm_cell_34_matmul_1_readvariableop_resource5while_lstm_cell_34_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_34_matmul_readvariableop_resource3while_lstm_cell_34_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_34/BiasAdd/ReadVariableOp)while/lstm_cell_34/BiasAdd/ReadVariableOp2T
(while/lstm_cell_34/MatMul/ReadVariableOp(while/lstm_cell_34/MatMul/ReadVariableOp2X
*while/lstm_cell_34/MatMul_1/ReadVariableOp*while/lstm_cell_34/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
�

�
%__inference_signature_wrapper_1703300

args_0
args_0_1	
unknown:	�
	unknown_0:	2�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	2�
	unknown_4:	�
	unknown_5:d
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallargs_0args_0_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_17002442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameargs_0:MI
#
_output_shapes
:���������
"
_user_specified_name
args_0_1
�
�
Bsequential_11_bidirectional_11_backward_lstm_11_while_cond_1700139|
xsequential_11_bidirectional_11_backward_lstm_11_while_sequential_11_bidirectional_11_backward_lstm_11_while_loop_counter�
~sequential_11_bidirectional_11_backward_lstm_11_while_sequential_11_bidirectional_11_backward_lstm_11_while_maximum_iterationsE
Asequential_11_bidirectional_11_backward_lstm_11_while_placeholderG
Csequential_11_bidirectional_11_backward_lstm_11_while_placeholder_1G
Csequential_11_bidirectional_11_backward_lstm_11_while_placeholder_2G
Csequential_11_bidirectional_11_backward_lstm_11_while_placeholder_3G
Csequential_11_bidirectional_11_backward_lstm_11_while_placeholder_4~
zsequential_11_bidirectional_11_backward_lstm_11_while_less_sequential_11_bidirectional_11_backward_lstm_11_strided_slice_1�
�sequential_11_bidirectional_11_backward_lstm_11_while_sequential_11_bidirectional_11_backward_lstm_11_while_cond_1700139___redundant_placeholder0�
�sequential_11_bidirectional_11_backward_lstm_11_while_sequential_11_bidirectional_11_backward_lstm_11_while_cond_1700139___redundant_placeholder1�
�sequential_11_bidirectional_11_backward_lstm_11_while_sequential_11_bidirectional_11_backward_lstm_11_while_cond_1700139___redundant_placeholder2�
�sequential_11_bidirectional_11_backward_lstm_11_while_sequential_11_bidirectional_11_backward_lstm_11_while_cond_1700139___redundant_placeholder3�
�sequential_11_bidirectional_11_backward_lstm_11_while_sequential_11_bidirectional_11_backward_lstm_11_while_cond_1700139___redundant_placeholder4B
>sequential_11_bidirectional_11_backward_lstm_11_while_identity
�
:sequential_11/bidirectional_11/backward_lstm_11/while/LessLessAsequential_11_bidirectional_11_backward_lstm_11_while_placeholderzsequential_11_bidirectional_11_backward_lstm_11_while_less_sequential_11_bidirectional_11_backward_lstm_11_strided_slice_1*
T0*
_output_shapes
: 2<
:sequential_11/bidirectional_11/backward_lstm_11/while/Less�
>sequential_11/bidirectional_11/backward_lstm_11/while/IdentityIdentity>sequential_11/bidirectional_11/backward_lstm_11/while/Less:z:0*
T0
*
_output_shapes
: 2@
>sequential_11/bidirectional_11/backward_lstm_11/while/Identity"�
>sequential_11_bidirectional_11_backward_lstm_11_while_identityGsequential_11/bidirectional_11/backward_lstm_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :���������2:���������2:���������2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
�\
�
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_1705358

inputs>
+lstm_cell_34_matmul_readvariableop_resource:	�@
-lstm_cell_34_matmul_1_readvariableop_resource:	2�;
,lstm_cell_34_biasadd_readvariableop_resource:	�
identity��#lstm_cell_34/BiasAdd/ReadVariableOp�"lstm_cell_34/MatMul/ReadVariableOp�$lstm_cell_34/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packedc
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedg
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_34/MatMul/ReadVariableOpReadVariableOp+lstm_cell_34_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_34/MatMul/ReadVariableOp�
lstm_cell_34/MatMulMatMulstrided_slice_2:output:0*lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/MatMul�
$lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_34/MatMul_1/ReadVariableOp�
lstm_cell_34/MatMul_1MatMulzeros:output:0,lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/MatMul_1�
lstm_cell_34/addAddV2lstm_cell_34/MatMul:product:0lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/add�
#lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_34/BiasAdd/ReadVariableOp�
lstm_cell_34/BiasAddBiasAddlstm_cell_34/add:z:0+lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/BiasAdd~
lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_34/split/split_dim�
lstm_cell_34/splitSplit%lstm_cell_34/split/split_dim:output:0lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_34/split�
lstm_cell_34/SigmoidSigmoidlstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/Sigmoid�
lstm_cell_34/Sigmoid_1Sigmoidlstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_34/Sigmoid_1�
lstm_cell_34/mulMullstm_cell_34/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/mul}
lstm_cell_34/ReluRelulstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_34/Relu�
lstm_cell_34/mul_1Mullstm_cell_34/Sigmoid:y:0lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/mul_1�
lstm_cell_34/add_1AddV2lstm_cell_34/mul:z:0lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/add_1�
lstm_cell_34/Sigmoid_2Sigmoidlstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_34/Sigmoid_2|
lstm_cell_34/Relu_1Relulstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/Relu_1�
lstm_cell_34/mul_2Mullstm_cell_34/Sigmoid_2:y:0!lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_34_matmul_readvariableop_resource-lstm_cell_34_matmul_1_readvariableop_resource,lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1705274*
condR
while_cond_1705273*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity�
NoOpNoOp$^lstm_cell_34/BiasAdd/ReadVariableOp#^lstm_cell_34/MatMul/ReadVariableOp%^lstm_cell_34/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_34/BiasAdd/ReadVariableOp#lstm_cell_34/BiasAdd/ReadVariableOp2H
"lstm_cell_34/MatMul/ReadVariableOp"lstm_cell_34/MatMul/ReadVariableOp2L
$lstm_cell_34/MatMul_1/ReadVariableOp$lstm_cell_34/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�d
�
#backward_lstm_11_while_body_1702583>
:backward_lstm_11_while_backward_lstm_11_while_loop_counterD
@backward_lstm_11_while_backward_lstm_11_while_maximum_iterations&
"backward_lstm_11_while_placeholder(
$backward_lstm_11_while_placeholder_1(
$backward_lstm_11_while_placeholder_2(
$backward_lstm_11_while_placeholder_3(
$backward_lstm_11_while_placeholder_4=
9backward_lstm_11_while_backward_lstm_11_strided_slice_1_0y
ubackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_08
4backward_lstm_11_while_less_backward_lstm_11_sub_1_0W
Dbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0:	�Y
Fbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0:	2�T
Ebackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0:	�#
backward_lstm_11_while_identity%
!backward_lstm_11_while_identity_1%
!backward_lstm_11_while_identity_2%
!backward_lstm_11_while_identity_3%
!backward_lstm_11_while_identity_4%
!backward_lstm_11_while_identity_5%
!backward_lstm_11_while_identity_6;
7backward_lstm_11_while_backward_lstm_11_strided_slice_1w
sbackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor6
2backward_lstm_11_while_less_backward_lstm_11_sub_1U
Bbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource:	�W
Dbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource:	2�R
Cbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource:	���:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp�9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp�;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp�
Hbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2J
Hbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape�
:backward_lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_11_while_placeholderQbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02<
:backward_lstm_11/while/TensorArrayV2Read/TensorListGetItem�
backward_lstm_11/while/LessLess4backward_lstm_11_while_less_backward_lstm_11_sub_1_0"backward_lstm_11_while_placeholder*
T0*#
_output_shapes
:���������2
backward_lstm_11/while/Less�
9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOpDbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02;
9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp�
*backward_lstm_11/while/lstm_cell_35/MatMulMatMulAbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:0Abackward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*backward_lstm_11/while/lstm_cell_35/MatMul�
;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOpFbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02=
;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp�
,backward_lstm_11/while/lstm_cell_35/MatMul_1MatMul$backward_lstm_11_while_placeholder_3Cbackward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2.
,backward_lstm_11/while/lstm_cell_35/MatMul_1�
'backward_lstm_11/while/lstm_cell_35/addAddV24backward_lstm_11/while/lstm_cell_35/MatMul:product:06backward_lstm_11/while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2)
'backward_lstm_11/while/lstm_cell_35/add�
:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOpEbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02<
:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp�
+backward_lstm_11/while/lstm_cell_35/BiasAddBiasAdd+backward_lstm_11/while/lstm_cell_35/add:z:0Bbackward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+backward_lstm_11/while/lstm_cell_35/BiasAdd�
3backward_lstm_11/while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3backward_lstm_11/while/lstm_cell_35/split/split_dim�
)backward_lstm_11/while/lstm_cell_35/splitSplit<backward_lstm_11/while/lstm_cell_35/split/split_dim:output:04backward_lstm_11/while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2+
)backward_lstm_11/while/lstm_cell_35/split�
+backward_lstm_11/while/lstm_cell_35/SigmoidSigmoid2backward_lstm_11/while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22-
+backward_lstm_11/while/lstm_cell_35/Sigmoid�
-backward_lstm_11/while/lstm_cell_35/Sigmoid_1Sigmoid2backward_lstm_11/while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22/
-backward_lstm_11/while/lstm_cell_35/Sigmoid_1�
'backward_lstm_11/while/lstm_cell_35/mulMul1backward_lstm_11/while/lstm_cell_35/Sigmoid_1:y:0$backward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22)
'backward_lstm_11/while/lstm_cell_35/mul�
(backward_lstm_11/while/lstm_cell_35/ReluRelu2backward_lstm_11/while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22*
(backward_lstm_11/while/lstm_cell_35/Relu�
)backward_lstm_11/while/lstm_cell_35/mul_1Mul/backward_lstm_11/while/lstm_cell_35/Sigmoid:y:06backward_lstm_11/while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_11/while/lstm_cell_35/mul_1�
)backward_lstm_11/while/lstm_cell_35/add_1AddV2+backward_lstm_11/while/lstm_cell_35/mul:z:0-backward_lstm_11/while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_11/while/lstm_cell_35/add_1�
-backward_lstm_11/while/lstm_cell_35/Sigmoid_2Sigmoid2backward_lstm_11/while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22/
-backward_lstm_11/while/lstm_cell_35/Sigmoid_2�
*backward_lstm_11/while/lstm_cell_35/Relu_1Relu-backward_lstm_11/while/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22,
*backward_lstm_11/while/lstm_cell_35/Relu_1�
)backward_lstm_11/while/lstm_cell_35/mul_2Mul1backward_lstm_11/while/lstm_cell_35/Sigmoid_2:y:08backward_lstm_11/while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_11/while/lstm_cell_35/mul_2�
backward_lstm_11/while/SelectSelectbackward_lstm_11/while/Less:z:0-backward_lstm_11/while/lstm_cell_35/mul_2:z:0$backward_lstm_11_while_placeholder_2*
T0*'
_output_shapes
:���������22
backward_lstm_11/while/Select�
backward_lstm_11/while/Select_1Selectbackward_lstm_11/while/Less:z:0-backward_lstm_11/while/lstm_cell_35/mul_2:z:0$backward_lstm_11_while_placeholder_3*
T0*'
_output_shapes
:���������22!
backward_lstm_11/while/Select_1�
backward_lstm_11/while/Select_2Selectbackward_lstm_11/while/Less:z:0-backward_lstm_11/while/lstm_cell_35/add_1:z:0$backward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22!
backward_lstm_11/while/Select_2�
;backward_lstm_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_11_while_placeholder_1"backward_lstm_11_while_placeholder&backward_lstm_11/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_11/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_11/while/add/y�
backward_lstm_11/while/addAddV2"backward_lstm_11_while_placeholder%backward_lstm_11/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/while/add�
backward_lstm_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_11/while/add_1/y�
backward_lstm_11/while/add_1AddV2:backward_lstm_11_while_backward_lstm_11_while_loop_counter'backward_lstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/while/add_1�
backward_lstm_11/while/IdentityIdentity backward_lstm_11/while/add_1:z:0^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_11/while/Identity�
!backward_lstm_11/while/Identity_1Identity@backward_lstm_11_while_backward_lstm_11_while_maximum_iterations^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_11/while/Identity_1�
!backward_lstm_11/while/Identity_2Identitybackward_lstm_11/while/add:z:0^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_11/while/Identity_2�
!backward_lstm_11/while/Identity_3IdentityKbackward_lstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_11/while/Identity_3�
!backward_lstm_11/while/Identity_4Identity&backward_lstm_11/while/Select:output:0^backward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/while/Identity_4�
!backward_lstm_11/while/Identity_5Identity(backward_lstm_11/while/Select_1:output:0^backward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/while/Identity_5�
!backward_lstm_11/while/Identity_6Identity(backward_lstm_11/while/Select_2:output:0^backward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/while/Identity_6�
backward_lstm_11/while/NoOpNoOp;^backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp:^backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp<^backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_11/while/NoOp"t
7backward_lstm_11_while_backward_lstm_11_strided_slice_19backward_lstm_11_while_backward_lstm_11_strided_slice_1_0"K
backward_lstm_11_while_identity(backward_lstm_11/while/Identity:output:0"O
!backward_lstm_11_while_identity_1*backward_lstm_11/while/Identity_1:output:0"O
!backward_lstm_11_while_identity_2*backward_lstm_11/while/Identity_2:output:0"O
!backward_lstm_11_while_identity_3*backward_lstm_11/while/Identity_3:output:0"O
!backward_lstm_11_while_identity_4*backward_lstm_11/while/Identity_4:output:0"O
!backward_lstm_11_while_identity_5*backward_lstm_11/while/Identity_5:output:0"O
!backward_lstm_11_while_identity_6*backward_lstm_11/while/Identity_6:output:0"j
2backward_lstm_11_while_less_backward_lstm_11_sub_14backward_lstm_11_while_less_backward_lstm_11_sub_1_0"�
Cbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resourceEbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0"�
Dbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resourceFbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0"�
Bbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resourceDbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0"�
sbackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensorubackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2x
:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp2v
9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp2z
;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:���������
�
�
I__inference_lstm_cell_35_layer_call_and_return_conditional_losses_1700951

inputs

states
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	2�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identity_2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������2:���������2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������2
 
_user_specified_namestates:OK
'
_output_shapes
:���������2
 
_user_specified_namestates
�
�
I__inference_lstm_cell_35_layer_call_and_return_conditional_losses_1706210

inputs
states_0
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	2�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identity_2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������2:���������2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������2
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������2
"
_user_specified_name
states/1
�
�
"forward_lstm_11_while_cond_1703436<
8forward_lstm_11_while_forward_lstm_11_while_loop_counterB
>forward_lstm_11_while_forward_lstm_11_while_maximum_iterations%
!forward_lstm_11_while_placeholder'
#forward_lstm_11_while_placeholder_1'
#forward_lstm_11_while_placeholder_2'
#forward_lstm_11_while_placeholder_3>
:forward_lstm_11_while_less_forward_lstm_11_strided_slice_1U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1703436___redundant_placeholder0U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1703436___redundant_placeholder1U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1703436___redundant_placeholder2U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1703436___redundant_placeholder3"
forward_lstm_11_while_identity
�
forward_lstm_11/while/LessLess!forward_lstm_11_while_placeholder:forward_lstm_11_while_less_forward_lstm_11_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_11/while/Less�
forward_lstm_11/while/IdentityIdentityforward_lstm_11/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_11/while/Identity"I
forward_lstm_11_while_identity'forward_lstm_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
�
�
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_1701840

inputs*
forward_lstm_11_1701670:	�*
forward_lstm_11_1701672:	2�&
forward_lstm_11_1701674:	�+
backward_lstm_11_1701830:	�+
backward_lstm_11_1701832:	2�'
backward_lstm_11_1701834:	�
identity��(backward_lstm_11/StatefulPartitionedCall�'forward_lstm_11/StatefulPartitionedCall�
'forward_lstm_11/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_11_1701670forward_lstm_11_1701672forward_lstm_11_1701674*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_17016692)
'forward_lstm_11/StatefulPartitionedCall�
(backward_lstm_11/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_11_1701830backward_lstm_11_1701832backward_lstm_11_1701834*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_17018292*
(backward_lstm_11/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV20forward_lstm_11/StatefulPartitionedCall:output:01backward_lstm_11/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity�
NoOpNoOp)^backward_lstm_11/StatefulPartitionedCall(^forward_lstm_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 2T
(backward_lstm_11/StatefulPartitionedCall(backward_lstm_11/StatefulPartitionedCall2R
'forward_lstm_11/StatefulPartitionedCall'forward_lstm_11/StatefulPartitionedCall:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�

�
/__inference_sequential_11_layer_call_fn_1702731

inputs
inputs_1	
unknown:	�
	unknown_0:	2�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	2�
	unknown_4:	�
	unknown_5:d
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_11_layer_call_and_return_conditional_losses_17027122
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_lstm_cell_34_layer_call_and_return_conditional_losses_1700319

inputs

states
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	2�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identity_2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������2:���������2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������2
 
_user_specified_namestates:OK
'
_output_shapes
:���������2
 
_user_specified_namestates
�
�
#backward_lstm_11_while_cond_1703022>
:backward_lstm_11_while_backward_lstm_11_while_loop_counterD
@backward_lstm_11_while_backward_lstm_11_while_maximum_iterations&
"backward_lstm_11_while_placeholder(
$backward_lstm_11_while_placeholder_1(
$backward_lstm_11_while_placeholder_2(
$backward_lstm_11_while_placeholder_3(
$backward_lstm_11_while_placeholder_4@
<backward_lstm_11_while_less_backward_lstm_11_strided_slice_1W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1703022___redundant_placeholder0W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1703022___redundant_placeholder1W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1703022___redundant_placeholder2W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1703022___redundant_placeholder3W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1703022___redundant_placeholder4#
backward_lstm_11_while_identity
�
backward_lstm_11/while/LessLess"backward_lstm_11_while_placeholder<backward_lstm_11_while_less_backward_lstm_11_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_11/while/Less�
backward_lstm_11/while/IdentityIdentitybackward_lstm_11/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_11/while/Identity"K
backward_lstm_11_while_identity(backward_lstm_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :���������2:���������2:���������2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
�?
�
while_body_1705930
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_35_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_35_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_35_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_35_matmul_readvariableop_resource:	�F
3while_lstm_cell_35_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_35_biasadd_readvariableop_resource:	���)while/lstm_cell_35/BiasAdd/ReadVariableOp�(while/lstm_cell_35/MatMul/ReadVariableOp�*while/lstm_cell_35/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_35/MatMul/ReadVariableOp�
while/lstm_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/MatMul�
*while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_35/MatMul_1/ReadVariableOp�
while/lstm_cell_35/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/MatMul_1�
while/lstm_cell_35/addAddV2#while/lstm_cell_35/MatMul:product:0%while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/add�
)while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_35/BiasAdd/ReadVariableOp�
while/lstm_cell_35/BiasAddBiasAddwhile/lstm_cell_35/add:z:01while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/BiasAdd�
"while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_35/split/split_dim�
while/lstm_cell_35/splitSplit+while/lstm_cell_35/split/split_dim:output:0#while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_35/split�
while/lstm_cell_35/SigmoidSigmoid!while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Sigmoid�
while/lstm_cell_35/Sigmoid_1Sigmoid!while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Sigmoid_1�
while/lstm_cell_35/mulMul while/lstm_cell_35/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/mul�
while/lstm_cell_35/ReluRelu!while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Relu�
while/lstm_cell_35/mul_1Mulwhile/lstm_cell_35/Sigmoid:y:0%while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/mul_1�
while/lstm_cell_35/add_1AddV2while/lstm_cell_35/mul:z:0while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/add_1�
while/lstm_cell_35/Sigmoid_2Sigmoid!while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Sigmoid_2�
while/lstm_cell_35/Relu_1Reluwhile/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Relu_1�
while/lstm_cell_35/mul_2Mul while/lstm_cell_35/Sigmoid_2:y:0'while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_35/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_35/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_35/BiasAdd/ReadVariableOp)^while/lstm_cell_35/MatMul/ReadVariableOp+^while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_35_biasadd_readvariableop_resource4while_lstm_cell_35_biasadd_readvariableop_resource_0"l
3while_lstm_cell_35_matmul_1_readvariableop_resource5while_lstm_cell_35_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_35_matmul_readvariableop_resource3while_lstm_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_35/BiasAdd/ReadVariableOp)while/lstm_cell_35/BiasAdd/ReadVariableOp2T
(while/lstm_cell_35/MatMul/ReadVariableOp(while/lstm_cell_35/MatMul/ReadVariableOp2X
*while/lstm_cell_35/MatMul_1/ReadVariableOp*while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
�
�
J__inference_sequential_11_layer_call_and_return_conditional_losses_1702712

inputs
inputs_1	+
bidirectional_11_1702681:	�+
bidirectional_11_1702683:	2�'
bidirectional_11_1702685:	�+
bidirectional_11_1702687:	�+
bidirectional_11_1702689:	2�'
bidirectional_11_1702691:	�"
dense_11_1702706:d
dense_11_1702708:
identity��(bidirectional_11/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�
(bidirectional_11/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_11_1702681bidirectional_11_1702683bidirectional_11_1702685bidirectional_11_1702687bidirectional_11_1702689bidirectional_11_1702691*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_17026802*
(bidirectional_11/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_11/StatefulPartitionedCall:output:0dense_11_1702706dense_11_1702708*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_17027052"
 dense_11/StatefulPartitionedCall�
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp)^bidirectional_11/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:���������:���������: : : : : : : : 2T
(bidirectional_11/StatefulPartitionedCall(bidirectional_11/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_1701584
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1701584___redundant_placeholder05
1while_while_cond_1701584___redundant_placeholder15
1while_while_cond_1701584___redundant_placeholder25
1while_while_cond_1701584___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
�
�
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_1702242

inputs*
forward_lstm_11_1702225:	�*
forward_lstm_11_1702227:	2�&
forward_lstm_11_1702229:	�+
backward_lstm_11_1702232:	�+
backward_lstm_11_1702234:	2�'
backward_lstm_11_1702236:	�
identity��(backward_lstm_11/StatefulPartitionedCall�'forward_lstm_11/StatefulPartitionedCall�
'forward_lstm_11/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_11_1702225forward_lstm_11_1702227forward_lstm_11_1702229*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_17021942)
'forward_lstm_11/StatefulPartitionedCall�
(backward_lstm_11/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_11_1702232backward_lstm_11_1702234backward_lstm_11_1702236*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_17020212*
(backward_lstm_11/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV20forward_lstm_11/StatefulPartitionedCall:output:01backward_lstm_11/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity�
NoOpNoOp)^backward_lstm_11/StatefulPartitionedCall(^forward_lstm_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 2T
(backward_lstm_11/StatefulPartitionedCall(backward_lstm_11/StatefulPartitionedCall2R
'forward_lstm_11/StatefulPartitionedCall'forward_lstm_11/StatefulPartitionedCall:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
while_cond_1705776
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1705776___redundant_placeholder05
1while_while_cond_1705776___redundant_placeholder15
1while_while_cond_1705776___redundant_placeholder25
1while_while_cond_1705776___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
�H
�
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_1701246

inputs'
lstm_cell_35_1701164:	�'
lstm_cell_35_1701166:	2�#
lstm_cell_35_1701168:	�
identity��$lstm_cell_35/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packedc
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedg
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis�
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������2
	ReverseV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
$lstm_cell_35/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_35_1701164lstm_cell_35_1701166lstm_cell_35_1701168*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������2:���������2:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_lstm_cell_35_layer_call_and_return_conditional_losses_17010972&
$lstm_cell_35/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_35_1701164lstm_cell_35_1701166lstm_cell_35_1701168*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1701177*
condR
while_cond_1701176*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity}
NoOpNoOp%^lstm_cell_35/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_35/StatefulPartitionedCall$lstm_cell_35/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
.__inference_lstm_cell_34_layer_call_fn_1706048

inputs
states_0
states_1
unknown:	�
	unknown_0:	2�
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������2:���������2:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_lstm_cell_34_layer_call_and_return_conditional_losses_17004652
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������22

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������2:���������2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������2
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������2
"
_user_specified_name
states/1
�d
�
#backward_lstm_11_while_body_1704593>
:backward_lstm_11_while_backward_lstm_11_while_loop_counterD
@backward_lstm_11_while_backward_lstm_11_while_maximum_iterations&
"backward_lstm_11_while_placeholder(
$backward_lstm_11_while_placeholder_1(
$backward_lstm_11_while_placeholder_2(
$backward_lstm_11_while_placeholder_3(
$backward_lstm_11_while_placeholder_4=
9backward_lstm_11_while_backward_lstm_11_strided_slice_1_0y
ubackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_08
4backward_lstm_11_while_less_backward_lstm_11_sub_1_0W
Dbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0:	�Y
Fbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0:	2�T
Ebackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0:	�#
backward_lstm_11_while_identity%
!backward_lstm_11_while_identity_1%
!backward_lstm_11_while_identity_2%
!backward_lstm_11_while_identity_3%
!backward_lstm_11_while_identity_4%
!backward_lstm_11_while_identity_5%
!backward_lstm_11_while_identity_6;
7backward_lstm_11_while_backward_lstm_11_strided_slice_1w
sbackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor6
2backward_lstm_11_while_less_backward_lstm_11_sub_1U
Bbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource:	�W
Dbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource:	2�R
Cbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource:	���:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp�9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp�;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp�
Hbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2J
Hbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape�
:backward_lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_11_while_placeholderQbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02<
:backward_lstm_11/while/TensorArrayV2Read/TensorListGetItem�
backward_lstm_11/while/LessLess4backward_lstm_11_while_less_backward_lstm_11_sub_1_0"backward_lstm_11_while_placeholder*
T0*#
_output_shapes
:���������2
backward_lstm_11/while/Less�
9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOpDbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02;
9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp�
*backward_lstm_11/while/lstm_cell_35/MatMulMatMulAbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:0Abackward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*backward_lstm_11/while/lstm_cell_35/MatMul�
;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOpFbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02=
;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp�
,backward_lstm_11/while/lstm_cell_35/MatMul_1MatMul$backward_lstm_11_while_placeholder_3Cbackward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2.
,backward_lstm_11/while/lstm_cell_35/MatMul_1�
'backward_lstm_11/while/lstm_cell_35/addAddV24backward_lstm_11/while/lstm_cell_35/MatMul:product:06backward_lstm_11/while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2)
'backward_lstm_11/while/lstm_cell_35/add�
:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOpEbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02<
:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp�
+backward_lstm_11/while/lstm_cell_35/BiasAddBiasAdd+backward_lstm_11/while/lstm_cell_35/add:z:0Bbackward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+backward_lstm_11/while/lstm_cell_35/BiasAdd�
3backward_lstm_11/while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3backward_lstm_11/while/lstm_cell_35/split/split_dim�
)backward_lstm_11/while/lstm_cell_35/splitSplit<backward_lstm_11/while/lstm_cell_35/split/split_dim:output:04backward_lstm_11/while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2+
)backward_lstm_11/while/lstm_cell_35/split�
+backward_lstm_11/while/lstm_cell_35/SigmoidSigmoid2backward_lstm_11/while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22-
+backward_lstm_11/while/lstm_cell_35/Sigmoid�
-backward_lstm_11/while/lstm_cell_35/Sigmoid_1Sigmoid2backward_lstm_11/while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22/
-backward_lstm_11/while/lstm_cell_35/Sigmoid_1�
'backward_lstm_11/while/lstm_cell_35/mulMul1backward_lstm_11/while/lstm_cell_35/Sigmoid_1:y:0$backward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22)
'backward_lstm_11/while/lstm_cell_35/mul�
(backward_lstm_11/while/lstm_cell_35/ReluRelu2backward_lstm_11/while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22*
(backward_lstm_11/while/lstm_cell_35/Relu�
)backward_lstm_11/while/lstm_cell_35/mul_1Mul/backward_lstm_11/while/lstm_cell_35/Sigmoid:y:06backward_lstm_11/while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_11/while/lstm_cell_35/mul_1�
)backward_lstm_11/while/lstm_cell_35/add_1AddV2+backward_lstm_11/while/lstm_cell_35/mul:z:0-backward_lstm_11/while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_11/while/lstm_cell_35/add_1�
-backward_lstm_11/while/lstm_cell_35/Sigmoid_2Sigmoid2backward_lstm_11/while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22/
-backward_lstm_11/while/lstm_cell_35/Sigmoid_2�
*backward_lstm_11/while/lstm_cell_35/Relu_1Relu-backward_lstm_11/while/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22,
*backward_lstm_11/while/lstm_cell_35/Relu_1�
)backward_lstm_11/while/lstm_cell_35/mul_2Mul1backward_lstm_11/while/lstm_cell_35/Sigmoid_2:y:08backward_lstm_11/while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_11/while/lstm_cell_35/mul_2�
backward_lstm_11/while/SelectSelectbackward_lstm_11/while/Less:z:0-backward_lstm_11/while/lstm_cell_35/mul_2:z:0$backward_lstm_11_while_placeholder_2*
T0*'
_output_shapes
:���������22
backward_lstm_11/while/Select�
backward_lstm_11/while/Select_1Selectbackward_lstm_11/while/Less:z:0-backward_lstm_11/while/lstm_cell_35/mul_2:z:0$backward_lstm_11_while_placeholder_3*
T0*'
_output_shapes
:���������22!
backward_lstm_11/while/Select_1�
backward_lstm_11/while/Select_2Selectbackward_lstm_11/while/Less:z:0-backward_lstm_11/while/lstm_cell_35/add_1:z:0$backward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22!
backward_lstm_11/while/Select_2�
;backward_lstm_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_11_while_placeholder_1"backward_lstm_11_while_placeholder&backward_lstm_11/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_11/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_11/while/add/y�
backward_lstm_11/while/addAddV2"backward_lstm_11_while_placeholder%backward_lstm_11/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/while/add�
backward_lstm_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_11/while/add_1/y�
backward_lstm_11/while/add_1AddV2:backward_lstm_11_while_backward_lstm_11_while_loop_counter'backward_lstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/while/add_1�
backward_lstm_11/while/IdentityIdentity backward_lstm_11/while/add_1:z:0^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_11/while/Identity�
!backward_lstm_11/while/Identity_1Identity@backward_lstm_11_while_backward_lstm_11_while_maximum_iterations^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_11/while/Identity_1�
!backward_lstm_11/while/Identity_2Identitybackward_lstm_11/while/add:z:0^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_11/while/Identity_2�
!backward_lstm_11/while/Identity_3IdentityKbackward_lstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_11/while/Identity_3�
!backward_lstm_11/while/Identity_4Identity&backward_lstm_11/while/Select:output:0^backward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/while/Identity_4�
!backward_lstm_11/while/Identity_5Identity(backward_lstm_11/while/Select_1:output:0^backward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/while/Identity_5�
!backward_lstm_11/while/Identity_6Identity(backward_lstm_11/while/Select_2:output:0^backward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/while/Identity_6�
backward_lstm_11/while/NoOpNoOp;^backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp:^backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp<^backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_11/while/NoOp"t
7backward_lstm_11_while_backward_lstm_11_strided_slice_19backward_lstm_11_while_backward_lstm_11_strided_slice_1_0"K
backward_lstm_11_while_identity(backward_lstm_11/while/Identity:output:0"O
!backward_lstm_11_while_identity_1*backward_lstm_11/while/Identity_1:output:0"O
!backward_lstm_11_while_identity_2*backward_lstm_11/while/Identity_2:output:0"O
!backward_lstm_11_while_identity_3*backward_lstm_11/while/Identity_3:output:0"O
!backward_lstm_11_while_identity_4*backward_lstm_11/while/Identity_4:output:0"O
!backward_lstm_11_while_identity_5*backward_lstm_11/while/Identity_5:output:0"O
!backward_lstm_11_while_identity_6*backward_lstm_11/while/Identity_6:output:0"j
2backward_lstm_11_while_less_backward_lstm_11_sub_14backward_lstm_11_while_less_backward_lstm_11_sub_1_0"�
Cbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resourceEbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0"�
Dbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resourceFbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0"�
Bbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resourceDbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0"�
sbackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensorubackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2x
:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp2v
9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp2z
;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:���������
�
�
I__inference_lstm_cell_34_layer_call_and_return_conditional_losses_1700465

inputs

states
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	2�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identity_2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������2:���������2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������2
 
_user_specified_namestates:OK
'
_output_shapes
:���������2
 
_user_specified_namestates
�?
�
while_body_1704821
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_34_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_34_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_34_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_34_matmul_readvariableop_resource:	�F
3while_lstm_cell_34_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_34_biasadd_readvariableop_resource:	���)while/lstm_cell_34/BiasAdd/ReadVariableOp�(while/lstm_cell_34/MatMul/ReadVariableOp�*while/lstm_cell_34/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_34/MatMul/ReadVariableOp�
while/lstm_cell_34/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/MatMul�
*while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_34/MatMul_1/ReadVariableOp�
while/lstm_cell_34/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/MatMul_1�
while/lstm_cell_34/addAddV2#while/lstm_cell_34/MatMul:product:0%while/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/add�
)while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_34/BiasAdd/ReadVariableOp�
while/lstm_cell_34/BiasAddBiasAddwhile/lstm_cell_34/add:z:01while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/BiasAdd�
"while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_34/split/split_dim�
while/lstm_cell_34/splitSplit+while/lstm_cell_34/split/split_dim:output:0#while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_34/split�
while/lstm_cell_34/SigmoidSigmoid!while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Sigmoid�
while/lstm_cell_34/Sigmoid_1Sigmoid!while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Sigmoid_1�
while/lstm_cell_34/mulMul while/lstm_cell_34/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/mul�
while/lstm_cell_34/ReluRelu!while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Relu�
while/lstm_cell_34/mul_1Mulwhile/lstm_cell_34/Sigmoid:y:0%while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/mul_1�
while/lstm_cell_34/add_1AddV2while/lstm_cell_34/mul:z:0while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/add_1�
while/lstm_cell_34/Sigmoid_2Sigmoid!while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Sigmoid_2�
while/lstm_cell_34/Relu_1Reluwhile/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Relu_1�
while/lstm_cell_34/mul_2Mul while/lstm_cell_34/Sigmoid_2:y:0'while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_34/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_34/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_34/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_34/BiasAdd/ReadVariableOp)^while/lstm_cell_34/MatMul/ReadVariableOp+^while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_34_biasadd_readvariableop_resource4while_lstm_cell_34_biasadd_readvariableop_resource_0"l
3while_lstm_cell_34_matmul_1_readvariableop_resource5while_lstm_cell_34_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_34_matmul_readvariableop_resource3while_lstm_cell_34_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_34/BiasAdd/ReadVariableOp)while/lstm_cell_34/BiasAdd/ReadVariableOp2T
(while/lstm_cell_34/MatMul/ReadVariableOp(while/lstm_cell_34/MatMul/ReadVariableOp2X
*while/lstm_cell_34/MatMul_1/ReadVariableOp*while/lstm_cell_34/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_1700332
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1700332___redundant_placeholder05
1while_while_cond_1700332___redundant_placeholder15
1while_while_cond_1700332___redundant_placeholder25
1while_while_cond_1700332___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
�W
�
#backward_lstm_11_while_body_1703586>
:backward_lstm_11_while_backward_lstm_11_while_loop_counterD
@backward_lstm_11_while_backward_lstm_11_while_maximum_iterations&
"backward_lstm_11_while_placeholder(
$backward_lstm_11_while_placeholder_1(
$backward_lstm_11_while_placeholder_2(
$backward_lstm_11_while_placeholder_3=
9backward_lstm_11_while_backward_lstm_11_strided_slice_1_0y
ubackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0W
Dbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0:	�Y
Fbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0:	2�T
Ebackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0:	�#
backward_lstm_11_while_identity%
!backward_lstm_11_while_identity_1%
!backward_lstm_11_while_identity_2%
!backward_lstm_11_while_identity_3%
!backward_lstm_11_while_identity_4%
!backward_lstm_11_while_identity_5;
7backward_lstm_11_while_backward_lstm_11_strided_slice_1w
sbackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensorU
Bbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource:	�W
Dbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource:	2�R
Cbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource:	���:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp�9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp�;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp�
Hbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2J
Hbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape�
:backward_lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_11_while_placeholderQbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02<
:backward_lstm_11/while/TensorArrayV2Read/TensorListGetItem�
9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOpDbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02;
9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp�
*backward_lstm_11/while/lstm_cell_35/MatMulMatMulAbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:0Abackward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*backward_lstm_11/while/lstm_cell_35/MatMul�
;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOpFbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02=
;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp�
,backward_lstm_11/while/lstm_cell_35/MatMul_1MatMul$backward_lstm_11_while_placeholder_2Cbackward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2.
,backward_lstm_11/while/lstm_cell_35/MatMul_1�
'backward_lstm_11/while/lstm_cell_35/addAddV24backward_lstm_11/while/lstm_cell_35/MatMul:product:06backward_lstm_11/while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2)
'backward_lstm_11/while/lstm_cell_35/add�
:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOpEbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02<
:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp�
+backward_lstm_11/while/lstm_cell_35/BiasAddBiasAdd+backward_lstm_11/while/lstm_cell_35/add:z:0Bbackward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+backward_lstm_11/while/lstm_cell_35/BiasAdd�
3backward_lstm_11/while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3backward_lstm_11/while/lstm_cell_35/split/split_dim�
)backward_lstm_11/while/lstm_cell_35/splitSplit<backward_lstm_11/while/lstm_cell_35/split/split_dim:output:04backward_lstm_11/while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2+
)backward_lstm_11/while/lstm_cell_35/split�
+backward_lstm_11/while/lstm_cell_35/SigmoidSigmoid2backward_lstm_11/while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22-
+backward_lstm_11/while/lstm_cell_35/Sigmoid�
-backward_lstm_11/while/lstm_cell_35/Sigmoid_1Sigmoid2backward_lstm_11/while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22/
-backward_lstm_11/while/lstm_cell_35/Sigmoid_1�
'backward_lstm_11/while/lstm_cell_35/mulMul1backward_lstm_11/while/lstm_cell_35/Sigmoid_1:y:0$backward_lstm_11_while_placeholder_3*
T0*'
_output_shapes
:���������22)
'backward_lstm_11/while/lstm_cell_35/mul�
(backward_lstm_11/while/lstm_cell_35/ReluRelu2backward_lstm_11/while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22*
(backward_lstm_11/while/lstm_cell_35/Relu�
)backward_lstm_11/while/lstm_cell_35/mul_1Mul/backward_lstm_11/while/lstm_cell_35/Sigmoid:y:06backward_lstm_11/while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_11/while/lstm_cell_35/mul_1�
)backward_lstm_11/while/lstm_cell_35/add_1AddV2+backward_lstm_11/while/lstm_cell_35/mul:z:0-backward_lstm_11/while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_11/while/lstm_cell_35/add_1�
-backward_lstm_11/while/lstm_cell_35/Sigmoid_2Sigmoid2backward_lstm_11/while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22/
-backward_lstm_11/while/lstm_cell_35/Sigmoid_2�
*backward_lstm_11/while/lstm_cell_35/Relu_1Relu-backward_lstm_11/while/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22,
*backward_lstm_11/while/lstm_cell_35/Relu_1�
)backward_lstm_11/while/lstm_cell_35/mul_2Mul1backward_lstm_11/while/lstm_cell_35/Sigmoid_2:y:08backward_lstm_11/while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_11/while/lstm_cell_35/mul_2�
;backward_lstm_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_11_while_placeholder_1"backward_lstm_11_while_placeholder-backward_lstm_11/while/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_11/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_11/while/add/y�
backward_lstm_11/while/addAddV2"backward_lstm_11_while_placeholder%backward_lstm_11/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/while/add�
backward_lstm_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_11/while/add_1/y�
backward_lstm_11/while/add_1AddV2:backward_lstm_11_while_backward_lstm_11_while_loop_counter'backward_lstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/while/add_1�
backward_lstm_11/while/IdentityIdentity backward_lstm_11/while/add_1:z:0^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_11/while/Identity�
!backward_lstm_11/while/Identity_1Identity@backward_lstm_11_while_backward_lstm_11_while_maximum_iterations^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_11/while/Identity_1�
!backward_lstm_11/while/Identity_2Identitybackward_lstm_11/while/add:z:0^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_11/while/Identity_2�
!backward_lstm_11/while/Identity_3IdentityKbackward_lstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_11/while/Identity_3�
!backward_lstm_11/while/Identity_4Identity-backward_lstm_11/while/lstm_cell_35/mul_2:z:0^backward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/while/Identity_4�
!backward_lstm_11/while/Identity_5Identity-backward_lstm_11/while/lstm_cell_35/add_1:z:0^backward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/while/Identity_5�
backward_lstm_11/while/NoOpNoOp;^backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp:^backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp<^backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_11/while/NoOp"t
7backward_lstm_11_while_backward_lstm_11_strided_slice_19backward_lstm_11_while_backward_lstm_11_strided_slice_1_0"K
backward_lstm_11_while_identity(backward_lstm_11/while/Identity:output:0"O
!backward_lstm_11_while_identity_1*backward_lstm_11/while/Identity_1:output:0"O
!backward_lstm_11_while_identity_2*backward_lstm_11/while/Identity_2:output:0"O
!backward_lstm_11_while_identity_3*backward_lstm_11/while/Identity_3:output:0"O
!backward_lstm_11_while_identity_4*backward_lstm_11/while/Identity_4:output:0"O
!backward_lstm_11_while_identity_5*backward_lstm_11/while/Identity_5:output:0"�
Cbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resourceEbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0"�
Dbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resourceFbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0"�
Bbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resourceDbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0"�
sbackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensorubackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2x
:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp2v
9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp2z
;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
�\
�
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_1702194

inputs>
+lstm_cell_34_matmul_readvariableop_resource:	�@
-lstm_cell_34_matmul_1_readvariableop_resource:	2�;
,lstm_cell_34_biasadd_readvariableop_resource:	�
identity��#lstm_cell_34/BiasAdd/ReadVariableOp�"lstm_cell_34/MatMul/ReadVariableOp�$lstm_cell_34/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packedc
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedg
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_34/MatMul/ReadVariableOpReadVariableOp+lstm_cell_34_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_34/MatMul/ReadVariableOp�
lstm_cell_34/MatMulMatMulstrided_slice_2:output:0*lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/MatMul�
$lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_34/MatMul_1/ReadVariableOp�
lstm_cell_34/MatMul_1MatMulzeros:output:0,lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/MatMul_1�
lstm_cell_34/addAddV2lstm_cell_34/MatMul:product:0lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/add�
#lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_34/BiasAdd/ReadVariableOp�
lstm_cell_34/BiasAddBiasAddlstm_cell_34/add:z:0+lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/BiasAdd~
lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_34/split/split_dim�
lstm_cell_34/splitSplit%lstm_cell_34/split/split_dim:output:0lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_34/split�
lstm_cell_34/SigmoidSigmoidlstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/Sigmoid�
lstm_cell_34/Sigmoid_1Sigmoidlstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_34/Sigmoid_1�
lstm_cell_34/mulMullstm_cell_34/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/mul}
lstm_cell_34/ReluRelulstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_34/Relu�
lstm_cell_34/mul_1Mullstm_cell_34/Sigmoid:y:0lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/mul_1�
lstm_cell_34/add_1AddV2lstm_cell_34/mul:z:0lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/add_1�
lstm_cell_34/Sigmoid_2Sigmoidlstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_34/Sigmoid_2|
lstm_cell_34/Relu_1Relulstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/Relu_1�
lstm_cell_34/mul_2Mullstm_cell_34/Sigmoid_2:y:0!lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_34_matmul_readvariableop_resource-lstm_cell_34_matmul_1_readvariableop_resource,lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1702110*
condR
while_cond_1702109*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity�
NoOpNoOp$^lstm_cell_34/BiasAdd/ReadVariableOp#^lstm_cell_34/MatMul/ReadVariableOp%^lstm_cell_34/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_34/BiasAdd/ReadVariableOp#lstm_cell_34/BiasAdd/ReadVariableOp2H
"lstm_cell_34/MatMul/ReadVariableOp"lstm_cell_34/MatMul/ReadVariableOp2L
$lstm_cell_34/MatMul_1/ReadVariableOp$lstm_cell_34/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
while_cond_1705623
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1705623___redundant_placeholder05
1while_while_cond_1705623___redundant_placeholder15
1while_while_cond_1705623___redundant_placeholder25
1while_while_cond_1705623___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
�^
�
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_1705708
inputs_0>
+lstm_cell_35_matmul_readvariableop_resource:	�@
-lstm_cell_35_matmul_1_readvariableop_resource:	2�;
,lstm_cell_35_biasadd_readvariableop_resource:	�
identity��#lstm_cell_35/BiasAdd/ReadVariableOp�"lstm_cell_35/MatMul/ReadVariableOp�$lstm_cell_35/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packedc
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedg
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis�
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������2
	ReverseV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_35/MatMul/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_35/MatMul/ReadVariableOp�
lstm_cell_35/MatMulMatMulstrided_slice_2:output:0*lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/MatMul�
$lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_35/MatMul_1/ReadVariableOp�
lstm_cell_35/MatMul_1MatMulzeros:output:0,lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/MatMul_1�
lstm_cell_35/addAddV2lstm_cell_35/MatMul:product:0lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/add�
#lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_35/BiasAdd/ReadVariableOp�
lstm_cell_35/BiasAddBiasAddlstm_cell_35/add:z:0+lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/BiasAdd~
lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_35/split/split_dim�
lstm_cell_35/splitSplit%lstm_cell_35/split/split_dim:output:0lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_35/split�
lstm_cell_35/SigmoidSigmoidlstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/Sigmoid�
lstm_cell_35/Sigmoid_1Sigmoidlstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_35/Sigmoid_1�
lstm_cell_35/mulMullstm_cell_35/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/mul}
lstm_cell_35/ReluRelulstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_35/Relu�
lstm_cell_35/mul_1Mullstm_cell_35/Sigmoid:y:0lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/mul_1�
lstm_cell_35/add_1AddV2lstm_cell_35/mul:z:0lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/add_1�
lstm_cell_35/Sigmoid_2Sigmoidlstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_35/Sigmoid_2|
lstm_cell_35/Relu_1Relulstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/Relu_1�
lstm_cell_35/mul_2Mullstm_cell_35/Sigmoid_2:y:0!lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_35_matmul_readvariableop_resource-lstm_cell_35_matmul_1_readvariableop_resource,lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1705624*
condR
while_cond_1705623*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity�
NoOpNoOp$^lstm_cell_35/BiasAdd/ReadVariableOp#^lstm_cell_35/MatMul/ReadVariableOp%^lstm_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_35/BiasAdd/ReadVariableOp#lstm_cell_35/BiasAdd/ReadVariableOp2H
"lstm_cell_35/MatMul/ReadVariableOp"lstm_cell_35/MatMul/ReadVariableOp2L
$lstm_cell_35/MatMul_1/ReadVariableOp$lstm_cell_35/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
I__inference_lstm_cell_34_layer_call_and_return_conditional_losses_1706080

inputs
states_0
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	2�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identity_2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������2:���������2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������2
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������2
"
_user_specified_name
states/1
�

�
2__inference_bidirectional_11_layer_call_fn_1703352

inputs
inputs_1	
unknown:	�
	unknown_0:	2�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	2�
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_17026802
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�?
�
while_body_1701745
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_35_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_35_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_35_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_35_matmul_readvariableop_resource:	�F
3while_lstm_cell_35_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_35_biasadd_readvariableop_resource:	���)while/lstm_cell_35/BiasAdd/ReadVariableOp�(while/lstm_cell_35/MatMul/ReadVariableOp�*while/lstm_cell_35/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_35/MatMul/ReadVariableOp�
while/lstm_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/MatMul�
*while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_35/MatMul_1/ReadVariableOp�
while/lstm_cell_35/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/MatMul_1�
while/lstm_cell_35/addAddV2#while/lstm_cell_35/MatMul:product:0%while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/add�
)while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_35/BiasAdd/ReadVariableOp�
while/lstm_cell_35/BiasAddBiasAddwhile/lstm_cell_35/add:z:01while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/BiasAdd�
"while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_35/split/split_dim�
while/lstm_cell_35/splitSplit+while/lstm_cell_35/split/split_dim:output:0#while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_35/split�
while/lstm_cell_35/SigmoidSigmoid!while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Sigmoid�
while/lstm_cell_35/Sigmoid_1Sigmoid!while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Sigmoid_1�
while/lstm_cell_35/mulMul while/lstm_cell_35/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/mul�
while/lstm_cell_35/ReluRelu!while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Relu�
while/lstm_cell_35/mul_1Mulwhile/lstm_cell_35/Sigmoid:y:0%while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/mul_1�
while/lstm_cell_35/add_1AddV2while/lstm_cell_35/mul:z:0while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/add_1�
while/lstm_cell_35/Sigmoid_2Sigmoid!while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Sigmoid_2�
while/lstm_cell_35/Relu_1Reluwhile/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Relu_1�
while/lstm_cell_35/mul_2Mul while/lstm_cell_35/Sigmoid_2:y:0'while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_35/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_35/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_35/BiasAdd/ReadVariableOp)^while/lstm_cell_35/MatMul/ReadVariableOp+^while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_35_biasadd_readvariableop_resource4while_lstm_cell_35_biasadd_readvariableop_resource_0"l
3while_lstm_cell_35_matmul_1_readvariableop_resource5while_lstm_cell_35_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_35_matmul_readvariableop_resource3while_lstm_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_35/BiasAdd/ReadVariableOp)while/lstm_cell_35/BiasAdd/ReadVariableOp2T
(while/lstm_cell_35/MatMul/ReadVariableOp(while/lstm_cell_35/MatMul/ReadVariableOp2X
*while/lstm_cell_35/MatMul_1/ReadVariableOp*while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
�?
�
while_body_1705624
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_35_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_35_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_35_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_35_matmul_readvariableop_resource:	�F
3while_lstm_cell_35_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_35_biasadd_readvariableop_resource:	���)while/lstm_cell_35/BiasAdd/ReadVariableOp�(while/lstm_cell_35/MatMul/ReadVariableOp�*while/lstm_cell_35/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_35/MatMul/ReadVariableOp�
while/lstm_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/MatMul�
*while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_35/MatMul_1/ReadVariableOp�
while/lstm_cell_35/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/MatMul_1�
while/lstm_cell_35/addAddV2#while/lstm_cell_35/MatMul:product:0%while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/add�
)while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_35/BiasAdd/ReadVariableOp�
while/lstm_cell_35/BiasAddBiasAddwhile/lstm_cell_35/add:z:01while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/BiasAdd�
"while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_35/split/split_dim�
while/lstm_cell_35/splitSplit+while/lstm_cell_35/split/split_dim:output:0#while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_35/split�
while/lstm_cell_35/SigmoidSigmoid!while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Sigmoid�
while/lstm_cell_35/Sigmoid_1Sigmoid!while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Sigmoid_1�
while/lstm_cell_35/mulMul while/lstm_cell_35/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/mul�
while/lstm_cell_35/ReluRelu!while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Relu�
while/lstm_cell_35/mul_1Mulwhile/lstm_cell_35/Sigmoid:y:0%while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/mul_1�
while/lstm_cell_35/add_1AddV2while/lstm_cell_35/mul:z:0while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/add_1�
while/lstm_cell_35/Sigmoid_2Sigmoid!while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Sigmoid_2�
while/lstm_cell_35/Relu_1Reluwhile/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Relu_1�
while/lstm_cell_35/mul_2Mul while/lstm_cell_35/Sigmoid_2:y:0'while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_35/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_35/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_35/BiasAdd/ReadVariableOp)^while/lstm_cell_35/MatMul/ReadVariableOp+^while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_35_biasadd_readvariableop_resource4while_lstm_cell_35_biasadd_readvariableop_resource_0"l
3while_lstm_cell_35_matmul_1_readvariableop_resource5while_lstm_cell_35_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_35_matmul_readvariableop_resource3while_lstm_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_35/BiasAdd/ReadVariableOp)while/lstm_cell_35/BiasAdd/ReadVariableOp2T
(while/lstm_cell_35/MatMul/ReadVariableOp(while/lstm_cell_35/MatMul/ReadVariableOp2X
*while/lstm_cell_35/MatMul_1/ReadVariableOp*while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
�
�
2__inference_backward_lstm_11_layer_call_fn_1705402

inputs
unknown:	�
	unknown_0:	2�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_17020212
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
��
�
#__inference__traced_restore_1706478
file_prefix2
 assignvariableop_dense_11_kernel:d.
 assignvariableop_1_dense_11_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: Z
Gassignvariableop_7_bidirectional_11_forward_lstm_11_lstm_cell_34_kernel:	�d
Qassignvariableop_8_bidirectional_11_forward_lstm_11_lstm_cell_34_recurrent_kernel:	2�T
Eassignvariableop_9_bidirectional_11_forward_lstm_11_lstm_cell_34_bias:	�\
Iassignvariableop_10_bidirectional_11_backward_lstm_11_lstm_cell_35_kernel:	�f
Sassignvariableop_11_bidirectional_11_backward_lstm_11_lstm_cell_35_recurrent_kernel:	2�V
Gassignvariableop_12_bidirectional_11_backward_lstm_11_lstm_cell_35_bias:	�#
assignvariableop_13_total: #
assignvariableop_14_count: <
*assignvariableop_15_adam_dense_11_kernel_m:d6
(assignvariableop_16_adam_dense_11_bias_m:b
Oassignvariableop_17_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_kernel_m:	�l
Yassignvariableop_18_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_recurrent_kernel_m:	2�\
Massignvariableop_19_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_bias_m:	�c
Passignvariableop_20_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_kernel_m:	�m
Zassignvariableop_21_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_recurrent_kernel_m:	2�]
Nassignvariableop_22_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_bias_m:	�<
*assignvariableop_23_adam_dense_11_kernel_v:d6
(assignvariableop_24_adam_dense_11_bias_v:b
Oassignvariableop_25_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_kernel_v:	�l
Yassignvariableop_26_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_recurrent_kernel_v:	2�\
Massignvariableop_27_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_bias_v:	�c
Passignvariableop_28_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_kernel_v:	�m
Zassignvariableop_29_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_recurrent_kernel_v:	2�]
Nassignvariableop_30_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_bias_v:	�?
-assignvariableop_31_adam_dense_11_kernel_vhat:d9
+assignvariableop_32_adam_dense_11_bias_vhat:e
Rassignvariableop_33_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_kernel_vhat:	�o
\assignvariableop_34_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_recurrent_kernel_vhat:	2�_
Passignvariableop_35_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_bias_vhat:	�f
Sassignvariableop_36_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_kernel_vhat:	�p
]assignvariableop_37_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_recurrent_kernel_vhat:	2�`
Qassignvariableop_38_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_bias_vhat:	�
identity_40��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*�
value�B�(B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_dense_11_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_11_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpGassignvariableop_7_bidirectional_11_forward_lstm_11_lstm_cell_34_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpQassignvariableop_8_bidirectional_11_forward_lstm_11_lstm_cell_34_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpEassignvariableop_9_bidirectional_11_forward_lstm_11_lstm_cell_34_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpIassignvariableop_10_bidirectional_11_backward_lstm_11_lstm_cell_35_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpSassignvariableop_11_bidirectional_11_backward_lstm_11_lstm_cell_35_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpGassignvariableop_12_bidirectional_11_backward_lstm_11_lstm_cell_35_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_11_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_11_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpOassignvariableop_17_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpYassignvariableop_18_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpMassignvariableop_19_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpPassignvariableop_20_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpZassignvariableop_21_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpNassignvariableop_22_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_11_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_11_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOpOassignvariableop_25_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOpYassignvariableop_26_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOpMassignvariableop_27_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOpPassignvariableop_28_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOpZassignvariableop_29_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOpNassignvariableop_30_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp-assignvariableop_31_adam_dense_11_kernel_vhatIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_dense_11_bias_vhatIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOpRassignvariableop_33_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_kernel_vhatIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp\assignvariableop_34_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_recurrent_kernel_vhatIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOpPassignvariableop_35_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_bias_vhatIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOpSassignvariableop_36_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_kernel_vhatIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp]assignvariableop_37_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_recurrent_kernel_vhatIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOpQassignvariableop_38_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_bias_vhatIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_39f
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_40�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�?
�
while_body_1705777
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_35_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_35_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_35_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_35_matmul_readvariableop_resource:	�F
3while_lstm_cell_35_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_35_biasadd_readvariableop_resource:	���)while/lstm_cell_35/BiasAdd/ReadVariableOp�(while/lstm_cell_35/MatMul/ReadVariableOp�*while/lstm_cell_35/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_35/MatMul/ReadVariableOp�
while/lstm_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/MatMul�
*while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_35/MatMul_1/ReadVariableOp�
while/lstm_cell_35/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/MatMul_1�
while/lstm_cell_35/addAddV2#while/lstm_cell_35/MatMul:product:0%while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/add�
)while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_35/BiasAdd/ReadVariableOp�
while/lstm_cell_35/BiasAddBiasAddwhile/lstm_cell_35/add:z:01while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/BiasAdd�
"while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_35/split/split_dim�
while/lstm_cell_35/splitSplit+while/lstm_cell_35/split/split_dim:output:0#while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_35/split�
while/lstm_cell_35/SigmoidSigmoid!while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Sigmoid�
while/lstm_cell_35/Sigmoid_1Sigmoid!while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Sigmoid_1�
while/lstm_cell_35/mulMul while/lstm_cell_35/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/mul�
while/lstm_cell_35/ReluRelu!while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Relu�
while/lstm_cell_35/mul_1Mulwhile/lstm_cell_35/Sigmoid:y:0%while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/mul_1�
while/lstm_cell_35/add_1AddV2while/lstm_cell_35/mul:z:0while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/add_1�
while/lstm_cell_35/Sigmoid_2Sigmoid!while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Sigmoid_2�
while/lstm_cell_35/Relu_1Reluwhile/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Relu_1�
while/lstm_cell_35/mul_2Mul while/lstm_cell_35/Sigmoid_2:y:0'while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_35/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_35/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_35/BiasAdd/ReadVariableOp)^while/lstm_cell_35/MatMul/ReadVariableOp+^while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_35_biasadd_readvariableop_resource4while_lstm_cell_35_biasadd_readvariableop_resource_0"l
3while_lstm_cell_35_matmul_1_readvariableop_resource5while_lstm_cell_35_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_35_matmul_readvariableop_resource3while_lstm_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_35/BiasAdd/ReadVariableOp)while/lstm_cell_35/BiasAdd/ReadVariableOp2T
(while/lstm_cell_35/MatMul/ReadVariableOp(while/lstm_cell_35/MatMul/ReadVariableOp2X
*while/lstm_cell_35/MatMul_1/ReadVariableOp*while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_1701176
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1701176___redundant_placeholder05
1while_while_cond_1701176___redundant_placeholder15
1while_while_cond_1701176___redundant_placeholder25
1while_while_cond_1701176___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
�
�
"forward_lstm_11_while_cond_1702403<
8forward_lstm_11_while_forward_lstm_11_while_loop_counterB
>forward_lstm_11_while_forward_lstm_11_while_maximum_iterations%
!forward_lstm_11_while_placeholder'
#forward_lstm_11_while_placeholder_1'
#forward_lstm_11_while_placeholder_2'
#forward_lstm_11_while_placeholder_3'
#forward_lstm_11_while_placeholder_4>
:forward_lstm_11_while_less_forward_lstm_11_strided_slice_1U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1702403___redundant_placeholder0U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1702403___redundant_placeholder1U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1702403___redundant_placeholder2U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1702403___redundant_placeholder3U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1702403___redundant_placeholder4"
forward_lstm_11_while_identity
�
forward_lstm_11/while/LessLess!forward_lstm_11_while_placeholder:forward_lstm_11_while_less_forward_lstm_11_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_11/while/Less�
forward_lstm_11/while/IdentityIdentityforward_lstm_11/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_11/while/Identity"I
forward_lstm_11_while_identity'forward_lstm_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :���������2:���������2:���������2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
�^
�
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_1701829

inputs>
+lstm_cell_35_matmul_readvariableop_resource:	�@
-lstm_cell_35_matmul_1_readvariableop_resource:	2�;
,lstm_cell_35_biasadd_readvariableop_resource:	�
identity��#lstm_cell_35/BiasAdd/ReadVariableOp�"lstm_cell_35/MatMul/ReadVariableOp�$lstm_cell_35/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packedc
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedg
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis�
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'���������������������������2
	ReverseV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_35/MatMul/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_35/MatMul/ReadVariableOp�
lstm_cell_35/MatMulMatMulstrided_slice_2:output:0*lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/MatMul�
$lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_35/MatMul_1/ReadVariableOp�
lstm_cell_35/MatMul_1MatMulzeros:output:0,lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/MatMul_1�
lstm_cell_35/addAddV2lstm_cell_35/MatMul:product:0lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/add�
#lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_35/BiasAdd/ReadVariableOp�
lstm_cell_35/BiasAddBiasAddlstm_cell_35/add:z:0+lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/BiasAdd~
lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_35/split/split_dim�
lstm_cell_35/splitSplit%lstm_cell_35/split/split_dim:output:0lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_35/split�
lstm_cell_35/SigmoidSigmoidlstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/Sigmoid�
lstm_cell_35/Sigmoid_1Sigmoidlstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_35/Sigmoid_1�
lstm_cell_35/mulMullstm_cell_35/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/mul}
lstm_cell_35/ReluRelulstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_35/Relu�
lstm_cell_35/mul_1Mullstm_cell_35/Sigmoid:y:0lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/mul_1�
lstm_cell_35/add_1AddV2lstm_cell_35/mul:z:0lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/add_1�
lstm_cell_35/Sigmoid_2Sigmoidlstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_35/Sigmoid_2|
lstm_cell_35/Relu_1Relulstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/Relu_1�
lstm_cell_35/mul_2Mullstm_cell_35/Sigmoid_2:y:0!lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_35_matmul_readvariableop_resource-lstm_cell_35_matmul_1_readvariableop_resource,lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1701745*
condR
while_cond_1701744*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity�
NoOpNoOp$^lstm_cell_35/BiasAdd/ReadVariableOp#^lstm_cell_35/MatMul/ReadVariableOp%^lstm_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_35/BiasAdd/ReadVariableOp#lstm_cell_35/BiasAdd/ReadVariableOp2H
"lstm_cell_35/MatMul/ReadVariableOp"lstm_cell_35/MatMul/ReadVariableOp2L
$lstm_cell_35/MatMul_1/ReadVariableOp$lstm_cell_35/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
��
�
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_1703974
inputs_0N
;forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource:	�P
=forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource:	2�K
<forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource:	�O
<backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource:	�Q
>backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource:	2�L
=backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource:	�
identity��4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp�3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp�5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp�backward_lstm_11/while�3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp�2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp�4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp�forward_lstm_11/whilef
forward_lstm_11/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_11/Shape�
#forward_lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_11/strided_slice/stack�
%forward_lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_11/strided_slice/stack_1�
%forward_lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_11/strided_slice/stack_2�
forward_lstm_11/strided_sliceStridedSliceforward_lstm_11/Shape:output:0,forward_lstm_11/strided_slice/stack:output:0.forward_lstm_11/strided_slice/stack_1:output:0.forward_lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_11/strided_slice|
forward_lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_11/zeros/mul/y�
forward_lstm_11/zeros/mulMul&forward_lstm_11/strided_slice:output:0$forward_lstm_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros/mul
forward_lstm_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_11/zeros/Less/y�
forward_lstm_11/zeros/LessLessforward_lstm_11/zeros/mul:z:0%forward_lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros/Less�
forward_lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_11/zeros/packed/1�
forward_lstm_11/zeros/packedPack&forward_lstm_11/strided_slice:output:0'forward_lstm_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_11/zeros/packed�
forward_lstm_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_11/zeros/Const�
forward_lstm_11/zerosFill%forward_lstm_11/zeros/packed:output:0$forward_lstm_11/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_11/zeros�
forward_lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_11/zeros_1/mul/y�
forward_lstm_11/zeros_1/mulMul&forward_lstm_11/strided_slice:output:0&forward_lstm_11/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros_1/mul�
forward_lstm_11/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
forward_lstm_11/zeros_1/Less/y�
forward_lstm_11/zeros_1/LessLessforward_lstm_11/zeros_1/mul:z:0'forward_lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros_1/Less�
 forward_lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_11/zeros_1/packed/1�
forward_lstm_11/zeros_1/packedPack&forward_lstm_11/strided_slice:output:0)forward_lstm_11/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_11/zeros_1/packed�
forward_lstm_11/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_11/zeros_1/Const�
forward_lstm_11/zeros_1Fill'forward_lstm_11/zeros_1/packed:output:0&forward_lstm_11/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_11/zeros_1�
forward_lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_11/transpose/perm�
forward_lstm_11/transpose	Transposeinputs_0'forward_lstm_11/transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
forward_lstm_11/transpose
forward_lstm_11/Shape_1Shapeforward_lstm_11/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_11/Shape_1�
%forward_lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_11/strided_slice_1/stack�
'forward_lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_1/stack_1�
'forward_lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_1/stack_2�
forward_lstm_11/strided_slice_1StridedSlice forward_lstm_11/Shape_1:output:0.forward_lstm_11/strided_slice_1/stack:output:00forward_lstm_11/strided_slice_1/stack_1:output:00forward_lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_11/strided_slice_1�
+forward_lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+forward_lstm_11/TensorArrayV2/element_shape�
forward_lstm_11/TensorArrayV2TensorListReserve4forward_lstm_11/TensorArrayV2/element_shape:output:0(forward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_11/TensorArrayV2�
Eforward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2G
Eforward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape�
7forward_lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_11/transpose:y:0Nforward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_11/TensorArrayUnstack/TensorListFromTensor�
%forward_lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_11/strided_slice_2/stack�
'forward_lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_2/stack_1�
'forward_lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_2/stack_2�
forward_lstm_11/strided_slice_2StridedSliceforward_lstm_11/transpose:y:0.forward_lstm_11/strided_slice_2/stack:output:00forward_lstm_11/strided_slice_2/stack_1:output:00forward_lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2!
forward_lstm_11/strided_slice_2�
2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp;forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp�
#forward_lstm_11/lstm_cell_34/MatMulMatMul(forward_lstm_11/strided_slice_2:output:0:forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#forward_lstm_11/lstm_cell_34/MatMul�
4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp=forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype026
4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp�
%forward_lstm_11/lstm_cell_34/MatMul_1MatMulforward_lstm_11/zeros:output:0<forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%forward_lstm_11/lstm_cell_34/MatMul_1�
 forward_lstm_11/lstm_cell_34/addAddV2-forward_lstm_11/lstm_cell_34/MatMul:product:0/forward_lstm_11/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2"
 forward_lstm_11/lstm_cell_34/add�
3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp<forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp�
$forward_lstm_11/lstm_cell_34/BiasAddBiasAdd$forward_lstm_11/lstm_cell_34/add:z:0;forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$forward_lstm_11/lstm_cell_34/BiasAdd�
,forward_lstm_11/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,forward_lstm_11/lstm_cell_34/split/split_dim�
"forward_lstm_11/lstm_cell_34/splitSplit5forward_lstm_11/lstm_cell_34/split/split_dim:output:0-forward_lstm_11/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2$
"forward_lstm_11/lstm_cell_34/split�
$forward_lstm_11/lstm_cell_34/SigmoidSigmoid+forward_lstm_11/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22&
$forward_lstm_11/lstm_cell_34/Sigmoid�
&forward_lstm_11/lstm_cell_34/Sigmoid_1Sigmoid+forward_lstm_11/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22(
&forward_lstm_11/lstm_cell_34/Sigmoid_1�
 forward_lstm_11/lstm_cell_34/mulMul*forward_lstm_11/lstm_cell_34/Sigmoid_1:y:0 forward_lstm_11/zeros_1:output:0*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/lstm_cell_34/mul�
!forward_lstm_11/lstm_cell_34/ReluRelu+forward_lstm_11/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22#
!forward_lstm_11/lstm_cell_34/Relu�
"forward_lstm_11/lstm_cell_34/mul_1Mul(forward_lstm_11/lstm_cell_34/Sigmoid:y:0/forward_lstm_11/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_11/lstm_cell_34/mul_1�
"forward_lstm_11/lstm_cell_34/add_1AddV2$forward_lstm_11/lstm_cell_34/mul:z:0&forward_lstm_11/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_11/lstm_cell_34/add_1�
&forward_lstm_11/lstm_cell_34/Sigmoid_2Sigmoid+forward_lstm_11/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22(
&forward_lstm_11/lstm_cell_34/Sigmoid_2�
#forward_lstm_11/lstm_cell_34/Relu_1Relu&forward_lstm_11/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22%
#forward_lstm_11/lstm_cell_34/Relu_1�
"forward_lstm_11/lstm_cell_34/mul_2Mul*forward_lstm_11/lstm_cell_34/Sigmoid_2:y:01forward_lstm_11/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_11/lstm_cell_34/mul_2�
-forward_lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2/
-forward_lstm_11/TensorArrayV2_1/element_shape�
forward_lstm_11/TensorArrayV2_1TensorListReserve6forward_lstm_11/TensorArrayV2_1/element_shape:output:0(forward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_11/TensorArrayV2_1n
forward_lstm_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_11/time�
(forward_lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(forward_lstm_11/while/maximum_iterations�
"forward_lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_11/while/loop_counter�
forward_lstm_11/whileWhile+forward_lstm_11/while/loop_counter:output:01forward_lstm_11/while/maximum_iterations:output:0forward_lstm_11/time:output:0(forward_lstm_11/TensorArrayV2_1:handle:0forward_lstm_11/zeros:output:0 forward_lstm_11/zeros_1:output:0(forward_lstm_11/strided_slice_1:output:0Gforward_lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0;forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource=forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource<forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *.
body&R$
"forward_lstm_11_while_body_1703739*.
cond&R$
"forward_lstm_11_while_cond_1703738*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
forward_lstm_11/while�
@forward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2B
@forward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape�
2forward_lstm_11/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_11/while:output:3Iforward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype024
2forward_lstm_11/TensorArrayV2Stack/TensorListStack�
%forward_lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2'
%forward_lstm_11/strided_slice_3/stack�
'forward_lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_11/strided_slice_3/stack_1�
'forward_lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_3/stack_2�
forward_lstm_11/strided_slice_3StridedSlice;forward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_11/strided_slice_3/stack:output:00forward_lstm_11/strided_slice_3/stack_1:output:00forward_lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2!
forward_lstm_11/strided_slice_3�
 forward_lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_11/transpose_1/perm�
forward_lstm_11/transpose_1	Transpose;forward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_11/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
forward_lstm_11/transpose_1�
forward_lstm_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_11/runtimeh
backward_lstm_11/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_11/Shape�
$backward_lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_11/strided_slice/stack�
&backward_lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_11/strided_slice/stack_1�
&backward_lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_11/strided_slice/stack_2�
backward_lstm_11/strided_sliceStridedSlicebackward_lstm_11/Shape:output:0-backward_lstm_11/strided_slice/stack:output:0/backward_lstm_11/strided_slice/stack_1:output:0/backward_lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_11/strided_slice~
backward_lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_11/zeros/mul/y�
backward_lstm_11/zeros/mulMul'backward_lstm_11/strided_slice:output:0%backward_lstm_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros/mul�
backward_lstm_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
backward_lstm_11/zeros/Less/y�
backward_lstm_11/zeros/LessLessbackward_lstm_11/zeros/mul:z:0&backward_lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros/Less�
backward_lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_11/zeros/packed/1�
backward_lstm_11/zeros/packedPack'backward_lstm_11/strided_slice:output:0(backward_lstm_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_11/zeros/packed�
backward_lstm_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_11/zeros/Const�
backward_lstm_11/zerosFill&backward_lstm_11/zeros/packed:output:0%backward_lstm_11/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_11/zeros�
backward_lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_11/zeros_1/mul/y�
backward_lstm_11/zeros_1/mulMul'backward_lstm_11/strided_slice:output:0'backward_lstm_11/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros_1/mul�
backward_lstm_11/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2!
backward_lstm_11/zeros_1/Less/y�
backward_lstm_11/zeros_1/LessLess backward_lstm_11/zeros_1/mul:z:0(backward_lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros_1/Less�
!backward_lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_11/zeros_1/packed/1�
backward_lstm_11/zeros_1/packedPack'backward_lstm_11/strided_slice:output:0*backward_lstm_11/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_11/zeros_1/packed�
backward_lstm_11/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_11/zeros_1/Const�
backward_lstm_11/zeros_1Fill(backward_lstm_11/zeros_1/packed:output:0'backward_lstm_11/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_11/zeros_1�
backward_lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_11/transpose/perm�
backward_lstm_11/transpose	Transposeinputs_0(backward_lstm_11/transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
backward_lstm_11/transpose�
backward_lstm_11/Shape_1Shapebackward_lstm_11/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_11/Shape_1�
&backward_lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_11/strided_slice_1/stack�
(backward_lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_1/stack_1�
(backward_lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_1/stack_2�
 backward_lstm_11/strided_slice_1StridedSlice!backward_lstm_11/Shape_1:output:0/backward_lstm_11/strided_slice_1/stack:output:01backward_lstm_11/strided_slice_1/stack_1:output:01backward_lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_11/strided_slice_1�
,backward_lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2.
,backward_lstm_11/TensorArrayV2/element_shape�
backward_lstm_11/TensorArrayV2TensorListReserve5backward_lstm_11/TensorArrayV2/element_shape:output:0)backward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_11/TensorArrayV2�
backward_lstm_11/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_11/ReverseV2/axis�
backward_lstm_11/ReverseV2	ReverseV2backward_lstm_11/transpose:y:0(backward_lstm_11/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'���������������������������2
backward_lstm_11/ReverseV2�
Fbackward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2H
Fbackward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape�
8backward_lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_11/ReverseV2:output:0Obackward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_11/TensorArrayUnstack/TensorListFromTensor�
&backward_lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_11/strided_slice_2/stack�
(backward_lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_2/stack_1�
(backward_lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_2/stack_2�
 backward_lstm_11/strided_slice_2StridedSlicebackward_lstm_11/transpose:y:0/backward_lstm_11/strided_slice_2/stack:output:01backward_lstm_11/strided_slice_2/stack_1:output:01backward_lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2"
 backward_lstm_11/strided_slice_2�
3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp<backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype025
3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp�
$backward_lstm_11/lstm_cell_35/MatMulMatMul)backward_lstm_11/strided_slice_2:output:0;backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$backward_lstm_11/lstm_cell_35/MatMul�
5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp>backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype027
5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp�
&backward_lstm_11/lstm_cell_35/MatMul_1MatMulbackward_lstm_11/zeros:output:0=backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&backward_lstm_11/lstm_cell_35/MatMul_1�
!backward_lstm_11/lstm_cell_35/addAddV2.backward_lstm_11/lstm_cell_35/MatMul:product:00backward_lstm_11/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2#
!backward_lstm_11/lstm_cell_35/add�
4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp=backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype026
4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp�
%backward_lstm_11/lstm_cell_35/BiasAddBiasAdd%backward_lstm_11/lstm_cell_35/add:z:0<backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%backward_lstm_11/lstm_cell_35/BiasAdd�
-backward_lstm_11/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-backward_lstm_11/lstm_cell_35/split/split_dim�
#backward_lstm_11/lstm_cell_35/splitSplit6backward_lstm_11/lstm_cell_35/split/split_dim:output:0.backward_lstm_11/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2%
#backward_lstm_11/lstm_cell_35/split�
%backward_lstm_11/lstm_cell_35/SigmoidSigmoid,backward_lstm_11/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22'
%backward_lstm_11/lstm_cell_35/Sigmoid�
'backward_lstm_11/lstm_cell_35/Sigmoid_1Sigmoid,backward_lstm_11/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22)
'backward_lstm_11/lstm_cell_35/Sigmoid_1�
!backward_lstm_11/lstm_cell_35/mulMul+backward_lstm_11/lstm_cell_35/Sigmoid_1:y:0!backward_lstm_11/zeros_1:output:0*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/lstm_cell_35/mul�
"backward_lstm_11/lstm_cell_35/ReluRelu,backward_lstm_11/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22$
"backward_lstm_11/lstm_cell_35/Relu�
#backward_lstm_11/lstm_cell_35/mul_1Mul)backward_lstm_11/lstm_cell_35/Sigmoid:y:00backward_lstm_11/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_11/lstm_cell_35/mul_1�
#backward_lstm_11/lstm_cell_35/add_1AddV2%backward_lstm_11/lstm_cell_35/mul:z:0'backward_lstm_11/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_11/lstm_cell_35/add_1�
'backward_lstm_11/lstm_cell_35/Sigmoid_2Sigmoid,backward_lstm_11/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22)
'backward_lstm_11/lstm_cell_35/Sigmoid_2�
$backward_lstm_11/lstm_cell_35/Relu_1Relu'backward_lstm_11/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22&
$backward_lstm_11/lstm_cell_35/Relu_1�
#backward_lstm_11/lstm_cell_35/mul_2Mul+backward_lstm_11/lstm_cell_35/Sigmoid_2:y:02backward_lstm_11/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_11/lstm_cell_35/mul_2�
.backward_lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   20
.backward_lstm_11/TensorArrayV2_1/element_shape�
 backward_lstm_11/TensorArrayV2_1TensorListReserve7backward_lstm_11/TensorArrayV2_1/element_shape:output:0)backward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_11/TensorArrayV2_1p
backward_lstm_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_11/time�
)backward_lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2+
)backward_lstm_11/while/maximum_iterations�
#backward_lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_11/while/loop_counter�
backward_lstm_11/whileWhile,backward_lstm_11/while/loop_counter:output:02backward_lstm_11/while/maximum_iterations:output:0backward_lstm_11/time:output:0)backward_lstm_11/TensorArrayV2_1:handle:0backward_lstm_11/zeros:output:0!backward_lstm_11/zeros_1:output:0)backward_lstm_11/strided_slice_1:output:0Hbackward_lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0<backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource>backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource=backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( */
body'R%
#backward_lstm_11_while_body_1703888*/
cond'R%
#backward_lstm_11_while_cond_1703887*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
backward_lstm_11/while�
Abackward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2C
Abackward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape�
3backward_lstm_11/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_11/while:output:3Jbackward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype025
3backward_lstm_11/TensorArrayV2Stack/TensorListStack�
&backward_lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2(
&backward_lstm_11/strided_slice_3/stack�
(backward_lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_11/strided_slice_3/stack_1�
(backward_lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_3/stack_2�
 backward_lstm_11/strided_slice_3StridedSlice<backward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_11/strided_slice_3/stack:output:01backward_lstm_11/strided_slice_3/stack_1:output:01backward_lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2"
 backward_lstm_11/strided_slice_3�
!backward_lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_11/transpose_1/perm�
backward_lstm_11/transpose_1	Transpose<backward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_11/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
backward_lstm_11/transpose_1�
backward_lstm_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_11/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2(forward_lstm_11/strided_slice_3:output:0)backward_lstm_11/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity�
NoOpNoOp5^backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp4^backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp6^backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp^backward_lstm_11/while4^forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp3^forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp5^forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp^forward_lstm_11/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 2l
4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp2j
3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp2n
5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp20
backward_lstm_11/whilebackward_lstm_11/while2j
3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp2h
2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp2l
4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp2.
forward_lstm_11/whileforward_lstm_11/while:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs/0
�
�
1__inference_forward_lstm_11_layer_call_fn_1704732
inputs_0
unknown:	�
	unknown_0:	2�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_17006122
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
*__inference_dense_11_layer_call_fn_1704699

inputs
unknown:d
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_17027052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�?
�
while_body_1702110
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_34_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_34_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_34_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_34_matmul_readvariableop_resource:	�F
3while_lstm_cell_34_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_34_biasadd_readvariableop_resource:	���)while/lstm_cell_34/BiasAdd/ReadVariableOp�(while/lstm_cell_34/MatMul/ReadVariableOp�*while/lstm_cell_34/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_34/MatMul/ReadVariableOp�
while/lstm_cell_34/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/MatMul�
*while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_34/MatMul_1/ReadVariableOp�
while/lstm_cell_34/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/MatMul_1�
while/lstm_cell_34/addAddV2#while/lstm_cell_34/MatMul:product:0%while/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/add�
)while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_34/BiasAdd/ReadVariableOp�
while/lstm_cell_34/BiasAddBiasAddwhile/lstm_cell_34/add:z:01while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/BiasAdd�
"while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_34/split/split_dim�
while/lstm_cell_34/splitSplit+while/lstm_cell_34/split/split_dim:output:0#while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_34/split�
while/lstm_cell_34/SigmoidSigmoid!while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Sigmoid�
while/lstm_cell_34/Sigmoid_1Sigmoid!while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Sigmoid_1�
while/lstm_cell_34/mulMul while/lstm_cell_34/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/mul�
while/lstm_cell_34/ReluRelu!while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Relu�
while/lstm_cell_34/mul_1Mulwhile/lstm_cell_34/Sigmoid:y:0%while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/mul_1�
while/lstm_cell_34/add_1AddV2while/lstm_cell_34/mul:z:0while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/add_1�
while/lstm_cell_34/Sigmoid_2Sigmoid!while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Sigmoid_2�
while/lstm_cell_34/Relu_1Reluwhile/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Relu_1�
while/lstm_cell_34/mul_2Mul while/lstm_cell_34/Sigmoid_2:y:0'while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_34/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_34/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_34/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_34/BiasAdd/ReadVariableOp)^while/lstm_cell_34/MatMul/ReadVariableOp+^while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_34_biasadd_readvariableop_resource4while_lstm_cell_34_biasadd_readvariableop_resource_0"l
3while_lstm_cell_34_matmul_1_readvariableop_resource5while_lstm_cell_34_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_34_matmul_readvariableop_resource3while_lstm_cell_34_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_34/BiasAdd/ReadVariableOp)while/lstm_cell_34/BiasAdd/ReadVariableOp2T
(while/lstm_cell_34/MatMul/ReadVariableOp(while/lstm_cell_34/MatMul/ReadVariableOp2X
*while/lstm_cell_34/MatMul_1/ReadVariableOp*while/lstm_cell_34/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
�
�
J__inference_sequential_11_layer_call_and_return_conditional_losses_1703183

inputs
inputs_1	+
bidirectional_11_1703164:	�+
bidirectional_11_1703166:	2�'
bidirectional_11_1703168:	�+
bidirectional_11_1703170:	�+
bidirectional_11_1703172:	2�'
bidirectional_11_1703174:	�"
dense_11_1703177:d
dense_11_1703179:
identity��(bidirectional_11/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�
(bidirectional_11/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_11_1703164bidirectional_11_1703166bidirectional_11_1703168bidirectional_11_1703170bidirectional_11_1703172bidirectional_11_1703174*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_17031202*
(bidirectional_11/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_11/StatefulPartitionedCall:output:0dense_11_1703177dense_11_1703179*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_17027052"
 dense_11/StatefulPartitionedCall�
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp)^bidirectional_11/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:���������:���������: : : : : : : : 2T
(bidirectional_11/StatefulPartitionedCall(bidirectional_11/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_1705273
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1705273___redundant_placeholder05
1while_while_cond_1705273___redundant_placeholder15
1while_while_cond_1705273___redundant_placeholder25
1while_while_cond_1705273___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
�
�
I__inference_lstm_cell_35_layer_call_and_return_conditional_losses_1701097

inputs

states
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	2�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identity_2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������2:���������2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������2
 
_user_specified_namestates:OK
'
_output_shapes
:���������2
 
_user_specified_namestates
�d
�
#backward_lstm_11_while_body_1703023>
:backward_lstm_11_while_backward_lstm_11_while_loop_counterD
@backward_lstm_11_while_backward_lstm_11_while_maximum_iterations&
"backward_lstm_11_while_placeholder(
$backward_lstm_11_while_placeholder_1(
$backward_lstm_11_while_placeholder_2(
$backward_lstm_11_while_placeholder_3(
$backward_lstm_11_while_placeholder_4=
9backward_lstm_11_while_backward_lstm_11_strided_slice_1_0y
ubackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_08
4backward_lstm_11_while_less_backward_lstm_11_sub_1_0W
Dbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0:	�Y
Fbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0:	2�T
Ebackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0:	�#
backward_lstm_11_while_identity%
!backward_lstm_11_while_identity_1%
!backward_lstm_11_while_identity_2%
!backward_lstm_11_while_identity_3%
!backward_lstm_11_while_identity_4%
!backward_lstm_11_while_identity_5%
!backward_lstm_11_while_identity_6;
7backward_lstm_11_while_backward_lstm_11_strided_slice_1w
sbackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor6
2backward_lstm_11_while_less_backward_lstm_11_sub_1U
Bbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource:	�W
Dbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource:	2�R
Cbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource:	���:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp�9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp�;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp�
Hbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2J
Hbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape�
:backward_lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_11_while_placeholderQbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02<
:backward_lstm_11/while/TensorArrayV2Read/TensorListGetItem�
backward_lstm_11/while/LessLess4backward_lstm_11_while_less_backward_lstm_11_sub_1_0"backward_lstm_11_while_placeholder*
T0*#
_output_shapes
:���������2
backward_lstm_11/while/Less�
9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOpDbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02;
9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp�
*backward_lstm_11/while/lstm_cell_35/MatMulMatMulAbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:0Abackward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*backward_lstm_11/while/lstm_cell_35/MatMul�
;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOpFbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02=
;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp�
,backward_lstm_11/while/lstm_cell_35/MatMul_1MatMul$backward_lstm_11_while_placeholder_3Cbackward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2.
,backward_lstm_11/while/lstm_cell_35/MatMul_1�
'backward_lstm_11/while/lstm_cell_35/addAddV24backward_lstm_11/while/lstm_cell_35/MatMul:product:06backward_lstm_11/while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2)
'backward_lstm_11/while/lstm_cell_35/add�
:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOpEbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02<
:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp�
+backward_lstm_11/while/lstm_cell_35/BiasAddBiasAdd+backward_lstm_11/while/lstm_cell_35/add:z:0Bbackward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+backward_lstm_11/while/lstm_cell_35/BiasAdd�
3backward_lstm_11/while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3backward_lstm_11/while/lstm_cell_35/split/split_dim�
)backward_lstm_11/while/lstm_cell_35/splitSplit<backward_lstm_11/while/lstm_cell_35/split/split_dim:output:04backward_lstm_11/while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2+
)backward_lstm_11/while/lstm_cell_35/split�
+backward_lstm_11/while/lstm_cell_35/SigmoidSigmoid2backward_lstm_11/while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22-
+backward_lstm_11/while/lstm_cell_35/Sigmoid�
-backward_lstm_11/while/lstm_cell_35/Sigmoid_1Sigmoid2backward_lstm_11/while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22/
-backward_lstm_11/while/lstm_cell_35/Sigmoid_1�
'backward_lstm_11/while/lstm_cell_35/mulMul1backward_lstm_11/while/lstm_cell_35/Sigmoid_1:y:0$backward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22)
'backward_lstm_11/while/lstm_cell_35/mul�
(backward_lstm_11/while/lstm_cell_35/ReluRelu2backward_lstm_11/while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22*
(backward_lstm_11/while/lstm_cell_35/Relu�
)backward_lstm_11/while/lstm_cell_35/mul_1Mul/backward_lstm_11/while/lstm_cell_35/Sigmoid:y:06backward_lstm_11/while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_11/while/lstm_cell_35/mul_1�
)backward_lstm_11/while/lstm_cell_35/add_1AddV2+backward_lstm_11/while/lstm_cell_35/mul:z:0-backward_lstm_11/while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_11/while/lstm_cell_35/add_1�
-backward_lstm_11/while/lstm_cell_35/Sigmoid_2Sigmoid2backward_lstm_11/while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22/
-backward_lstm_11/while/lstm_cell_35/Sigmoid_2�
*backward_lstm_11/while/lstm_cell_35/Relu_1Relu-backward_lstm_11/while/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22,
*backward_lstm_11/while/lstm_cell_35/Relu_1�
)backward_lstm_11/while/lstm_cell_35/mul_2Mul1backward_lstm_11/while/lstm_cell_35/Sigmoid_2:y:08backward_lstm_11/while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_11/while/lstm_cell_35/mul_2�
backward_lstm_11/while/SelectSelectbackward_lstm_11/while/Less:z:0-backward_lstm_11/while/lstm_cell_35/mul_2:z:0$backward_lstm_11_while_placeholder_2*
T0*'
_output_shapes
:���������22
backward_lstm_11/while/Select�
backward_lstm_11/while/Select_1Selectbackward_lstm_11/while/Less:z:0-backward_lstm_11/while/lstm_cell_35/mul_2:z:0$backward_lstm_11_while_placeholder_3*
T0*'
_output_shapes
:���������22!
backward_lstm_11/while/Select_1�
backward_lstm_11/while/Select_2Selectbackward_lstm_11/while/Less:z:0-backward_lstm_11/while/lstm_cell_35/add_1:z:0$backward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22!
backward_lstm_11/while/Select_2�
;backward_lstm_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_11_while_placeholder_1"backward_lstm_11_while_placeholder&backward_lstm_11/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_11/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_11/while/add/y�
backward_lstm_11/while/addAddV2"backward_lstm_11_while_placeholder%backward_lstm_11/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/while/add�
backward_lstm_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_11/while/add_1/y�
backward_lstm_11/while/add_1AddV2:backward_lstm_11_while_backward_lstm_11_while_loop_counter'backward_lstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/while/add_1�
backward_lstm_11/while/IdentityIdentity backward_lstm_11/while/add_1:z:0^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_11/while/Identity�
!backward_lstm_11/while/Identity_1Identity@backward_lstm_11_while_backward_lstm_11_while_maximum_iterations^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_11/while/Identity_1�
!backward_lstm_11/while/Identity_2Identitybackward_lstm_11/while/add:z:0^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_11/while/Identity_2�
!backward_lstm_11/while/Identity_3IdentityKbackward_lstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_11/while/Identity_3�
!backward_lstm_11/while/Identity_4Identity&backward_lstm_11/while/Select:output:0^backward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/while/Identity_4�
!backward_lstm_11/while/Identity_5Identity(backward_lstm_11/while/Select_1:output:0^backward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/while/Identity_5�
!backward_lstm_11/while/Identity_6Identity(backward_lstm_11/while/Select_2:output:0^backward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/while/Identity_6�
backward_lstm_11/while/NoOpNoOp;^backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp:^backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp<^backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_11/while/NoOp"t
7backward_lstm_11_while_backward_lstm_11_strided_slice_19backward_lstm_11_while_backward_lstm_11_strided_slice_1_0"K
backward_lstm_11_while_identity(backward_lstm_11/while/Identity:output:0"O
!backward_lstm_11_while_identity_1*backward_lstm_11/while/Identity_1:output:0"O
!backward_lstm_11_while_identity_2*backward_lstm_11/while/Identity_2:output:0"O
!backward_lstm_11_while_identity_3*backward_lstm_11/while/Identity_3:output:0"O
!backward_lstm_11_while_identity_4*backward_lstm_11/while/Identity_4:output:0"O
!backward_lstm_11_while_identity_5*backward_lstm_11/while/Identity_5:output:0"O
!backward_lstm_11_while_identity_6*backward_lstm_11/while/Identity_6:output:0"j
2backward_lstm_11_while_less_backward_lstm_11_sub_14backward_lstm_11_while_less_backward_lstm_11_sub_1_0"�
Cbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resourceEbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0"�
Dbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resourceFbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0"�
Bbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resourceDbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0"�
sbackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensorubackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2x
:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp2v
9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp2z
;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:���������
�	
�
2__inference_bidirectional_11_layer_call_fn_1703317
inputs_0
unknown:	�
	unknown_0:	2�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	2�
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_17018402
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs/0
�
�
1__inference_forward_lstm_11_layer_call_fn_1704743

inputs
unknown:	�
	unknown_0:	2�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_17016692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�b
�
"forward_lstm_11_while_body_1702404<
8forward_lstm_11_while_forward_lstm_11_while_loop_counterB
>forward_lstm_11_while_forward_lstm_11_while_maximum_iterations%
!forward_lstm_11_while_placeholder'
#forward_lstm_11_while_placeholder_1'
#forward_lstm_11_while_placeholder_2'
#forward_lstm_11_while_placeholder_3'
#forward_lstm_11_while_placeholder_4;
7forward_lstm_11_while_forward_lstm_11_strided_slice_1_0w
sforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_08
4forward_lstm_11_while_greater_forward_lstm_11_cast_0V
Cforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0:	�X
Eforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0:	2�S
Dforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0:	�"
forward_lstm_11_while_identity$
 forward_lstm_11_while_identity_1$
 forward_lstm_11_while_identity_2$
 forward_lstm_11_while_identity_3$
 forward_lstm_11_while_identity_4$
 forward_lstm_11_while_identity_5$
 forward_lstm_11_while_identity_69
5forward_lstm_11_while_forward_lstm_11_strided_slice_1u
qforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor6
2forward_lstm_11_while_greater_forward_lstm_11_castT
Aforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource:	�V
Cforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource:	2�Q
Bforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource:	���9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp�8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp�:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp�
Gforward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2I
Gforward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape�
9forward_lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_11_while_placeholderPforward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02;
9forward_lstm_11/while/TensorArrayV2Read/TensorListGetItem�
forward_lstm_11/while/GreaterGreater4forward_lstm_11_while_greater_forward_lstm_11_cast_0!forward_lstm_11_while_placeholder*
T0*#
_output_shapes
:���������2
forward_lstm_11/while/Greater�
8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOpCforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02:
8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp�
)forward_lstm_11/while/lstm_cell_34/MatMulMatMul@forward_lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:0@forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)forward_lstm_11/while/lstm_cell_34/MatMul�
:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOpEforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02<
:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp�
+forward_lstm_11/while/lstm_cell_34/MatMul_1MatMul#forward_lstm_11_while_placeholder_3Bforward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+forward_lstm_11/while/lstm_cell_34/MatMul_1�
&forward_lstm_11/while/lstm_cell_34/addAddV23forward_lstm_11/while/lstm_cell_34/MatMul:product:05forward_lstm_11/while/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&forward_lstm_11/while/lstm_cell_34/add�
9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOpDforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02;
9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp�
*forward_lstm_11/while/lstm_cell_34/BiasAddBiasAdd*forward_lstm_11/while/lstm_cell_34/add:z:0Aforward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*forward_lstm_11/while/lstm_cell_34/BiasAdd�
2forward_lstm_11/while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2forward_lstm_11/while/lstm_cell_34/split/split_dim�
(forward_lstm_11/while/lstm_cell_34/splitSplit;forward_lstm_11/while/lstm_cell_34/split/split_dim:output:03forward_lstm_11/while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2*
(forward_lstm_11/while/lstm_cell_34/split�
*forward_lstm_11/while/lstm_cell_34/SigmoidSigmoid1forward_lstm_11/while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22,
*forward_lstm_11/while/lstm_cell_34/Sigmoid�
,forward_lstm_11/while/lstm_cell_34/Sigmoid_1Sigmoid1forward_lstm_11/while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22.
,forward_lstm_11/while/lstm_cell_34/Sigmoid_1�
&forward_lstm_11/while/lstm_cell_34/mulMul0forward_lstm_11/while/lstm_cell_34/Sigmoid_1:y:0#forward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22(
&forward_lstm_11/while/lstm_cell_34/mul�
'forward_lstm_11/while/lstm_cell_34/ReluRelu1forward_lstm_11/while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22)
'forward_lstm_11/while/lstm_cell_34/Relu�
(forward_lstm_11/while/lstm_cell_34/mul_1Mul.forward_lstm_11/while/lstm_cell_34/Sigmoid:y:05forward_lstm_11/while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_11/while/lstm_cell_34/mul_1�
(forward_lstm_11/while/lstm_cell_34/add_1AddV2*forward_lstm_11/while/lstm_cell_34/mul:z:0,forward_lstm_11/while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_11/while/lstm_cell_34/add_1�
,forward_lstm_11/while/lstm_cell_34/Sigmoid_2Sigmoid1forward_lstm_11/while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22.
,forward_lstm_11/while/lstm_cell_34/Sigmoid_2�
)forward_lstm_11/while/lstm_cell_34/Relu_1Relu,forward_lstm_11/while/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22+
)forward_lstm_11/while/lstm_cell_34/Relu_1�
(forward_lstm_11/while/lstm_cell_34/mul_2Mul0forward_lstm_11/while/lstm_cell_34/Sigmoid_2:y:07forward_lstm_11/while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_11/while/lstm_cell_34/mul_2�
forward_lstm_11/while/SelectSelect!forward_lstm_11/while/Greater:z:0,forward_lstm_11/while/lstm_cell_34/mul_2:z:0#forward_lstm_11_while_placeholder_2*
T0*'
_output_shapes
:���������22
forward_lstm_11/while/Select�
forward_lstm_11/while/Select_1Select!forward_lstm_11/while/Greater:z:0,forward_lstm_11/while/lstm_cell_34/mul_2:z:0#forward_lstm_11_while_placeholder_3*
T0*'
_output_shapes
:���������22 
forward_lstm_11/while/Select_1�
forward_lstm_11/while/Select_2Select!forward_lstm_11/while/Greater:z:0,forward_lstm_11/while/lstm_cell_34/add_1:z:0#forward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22 
forward_lstm_11/while/Select_2�
:forward_lstm_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_11_while_placeholder_1!forward_lstm_11_while_placeholder%forward_lstm_11/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_11/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_11/while/add/y�
forward_lstm_11/while/addAddV2!forward_lstm_11_while_placeholder$forward_lstm_11/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/while/add�
forward_lstm_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_11/while/add_1/y�
forward_lstm_11/while/add_1AddV28forward_lstm_11_while_forward_lstm_11_while_loop_counter&forward_lstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/while/add_1�
forward_lstm_11/while/IdentityIdentityforward_lstm_11/while/add_1:z:0^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_11/while/Identity�
 forward_lstm_11/while/Identity_1Identity>forward_lstm_11_while_forward_lstm_11_while_maximum_iterations^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_11/while/Identity_1�
 forward_lstm_11/while/Identity_2Identityforward_lstm_11/while/add:z:0^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_11/while/Identity_2�
 forward_lstm_11/while/Identity_3IdentityJforward_lstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_11/while/Identity_3�
 forward_lstm_11/while/Identity_4Identity%forward_lstm_11/while/Select:output:0^forward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/while/Identity_4�
 forward_lstm_11/while/Identity_5Identity'forward_lstm_11/while/Select_1:output:0^forward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/while/Identity_5�
 forward_lstm_11/while/Identity_6Identity'forward_lstm_11/while/Select_2:output:0^forward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/while/Identity_6�
forward_lstm_11/while/NoOpNoOp:^forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp9^forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp;^forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_11/while/NoOp"p
5forward_lstm_11_while_forward_lstm_11_strided_slice_17forward_lstm_11_while_forward_lstm_11_strided_slice_1_0"j
2forward_lstm_11_while_greater_forward_lstm_11_cast4forward_lstm_11_while_greater_forward_lstm_11_cast_0"I
forward_lstm_11_while_identity'forward_lstm_11/while/Identity:output:0"M
 forward_lstm_11_while_identity_1)forward_lstm_11/while/Identity_1:output:0"M
 forward_lstm_11_while_identity_2)forward_lstm_11/while/Identity_2:output:0"M
 forward_lstm_11_while_identity_3)forward_lstm_11/while/Identity_3:output:0"M
 forward_lstm_11_while_identity_4)forward_lstm_11/while/Identity_4:output:0"M
 forward_lstm_11_while_identity_5)forward_lstm_11/while/Identity_5:output:0"M
 forward_lstm_11_while_identity_6)forward_lstm_11/while/Identity_6:output:0"�
Bforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resourceDforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0"�
Cforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resourceEforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0"�
Aforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resourceCforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0"�
qforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensorsforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2v
9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp2t
8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp2x
:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:���������
�
�
#backward_lstm_11_while_cond_1702582>
:backward_lstm_11_while_backward_lstm_11_while_loop_counterD
@backward_lstm_11_while_backward_lstm_11_while_maximum_iterations&
"backward_lstm_11_while_placeholder(
$backward_lstm_11_while_placeholder_1(
$backward_lstm_11_while_placeholder_2(
$backward_lstm_11_while_placeholder_3(
$backward_lstm_11_while_placeholder_4@
<backward_lstm_11_while_less_backward_lstm_11_strided_slice_1W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1702582___redundant_placeholder0W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1702582___redundant_placeholder1W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1702582___redundant_placeholder2W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1702582___redundant_placeholder3W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1702582___redundant_placeholder4#
backward_lstm_11_while_identity
�
backward_lstm_11/while/LessLess"backward_lstm_11_while_placeholder<backward_lstm_11_while_less_backward_lstm_11_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_11/while/Less�
backward_lstm_11/while/IdentityIdentitybackward_lstm_11/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_11/while/Identity"K
backward_lstm_11_while_identity(backward_lstm_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :���������2:���������2:���������2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
�?
�
while_body_1704972
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_34_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_34_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_34_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_34_matmul_readvariableop_resource:	�F
3while_lstm_cell_34_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_34_biasadd_readvariableop_resource:	���)while/lstm_cell_34/BiasAdd/ReadVariableOp�(while/lstm_cell_34/MatMul/ReadVariableOp�*while/lstm_cell_34/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_34/MatMul/ReadVariableOp�
while/lstm_cell_34/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/MatMul�
*while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_34/MatMul_1/ReadVariableOp�
while/lstm_cell_34/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/MatMul_1�
while/lstm_cell_34/addAddV2#while/lstm_cell_34/MatMul:product:0%while/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/add�
)while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_34/BiasAdd/ReadVariableOp�
while/lstm_cell_34/BiasAddBiasAddwhile/lstm_cell_34/add:z:01while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/BiasAdd�
"while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_34/split/split_dim�
while/lstm_cell_34/splitSplit+while/lstm_cell_34/split/split_dim:output:0#while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_34/split�
while/lstm_cell_34/SigmoidSigmoid!while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Sigmoid�
while/lstm_cell_34/Sigmoid_1Sigmoid!while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Sigmoid_1�
while/lstm_cell_34/mulMul while/lstm_cell_34/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/mul�
while/lstm_cell_34/ReluRelu!while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Relu�
while/lstm_cell_34/mul_1Mulwhile/lstm_cell_34/Sigmoid:y:0%while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/mul_1�
while/lstm_cell_34/add_1AddV2while/lstm_cell_34/mul:z:0while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/add_1�
while/lstm_cell_34/Sigmoid_2Sigmoid!while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Sigmoid_2�
while/lstm_cell_34/Relu_1Reluwhile/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Relu_1�
while/lstm_cell_34/mul_2Mul while/lstm_cell_34/Sigmoid_2:y:0'while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_34/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_34/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_34/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_34/BiasAdd/ReadVariableOp)^while/lstm_cell_34/MatMul/ReadVariableOp+^while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_34_biasadd_readvariableop_resource4while_lstm_cell_34_biasadd_readvariableop_resource_0"l
3while_lstm_cell_34_matmul_1_readvariableop_resource5while_lstm_cell_34_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_34_matmul_readvariableop_resource3while_lstm_cell_34_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_34/BiasAdd/ReadVariableOp)while/lstm_cell_34/BiasAdd/ReadVariableOp2T
(while/lstm_cell_34/MatMul/ReadVariableOp(while/lstm_cell_34/MatMul/ReadVariableOp2X
*while/lstm_cell_34/MatMul_1/ReadVariableOp*while/lstm_cell_34/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
�^
�
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_1705555
inputs_0>
+lstm_cell_35_matmul_readvariableop_resource:	�@
-lstm_cell_35_matmul_1_readvariableop_resource:	2�;
,lstm_cell_35_biasadd_readvariableop_resource:	�
identity��#lstm_cell_35/BiasAdd/ReadVariableOp�"lstm_cell_35/MatMul/ReadVariableOp�$lstm_cell_35/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packedc
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedg
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis�
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������2
	ReverseV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_35/MatMul/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_35/MatMul/ReadVariableOp�
lstm_cell_35/MatMulMatMulstrided_slice_2:output:0*lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/MatMul�
$lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_35/MatMul_1/ReadVariableOp�
lstm_cell_35/MatMul_1MatMulzeros:output:0,lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/MatMul_1�
lstm_cell_35/addAddV2lstm_cell_35/MatMul:product:0lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/add�
#lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_35/BiasAdd/ReadVariableOp�
lstm_cell_35/BiasAddBiasAddlstm_cell_35/add:z:0+lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/BiasAdd~
lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_35/split/split_dim�
lstm_cell_35/splitSplit%lstm_cell_35/split/split_dim:output:0lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_35/split�
lstm_cell_35/SigmoidSigmoidlstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/Sigmoid�
lstm_cell_35/Sigmoid_1Sigmoidlstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_35/Sigmoid_1�
lstm_cell_35/mulMullstm_cell_35/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/mul}
lstm_cell_35/ReluRelulstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_35/Relu�
lstm_cell_35/mul_1Mullstm_cell_35/Sigmoid:y:0lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/mul_1�
lstm_cell_35/add_1AddV2lstm_cell_35/mul:z:0lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/add_1�
lstm_cell_35/Sigmoid_2Sigmoidlstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_35/Sigmoid_2|
lstm_cell_35/Relu_1Relulstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/Relu_1�
lstm_cell_35/mul_2Mullstm_cell_35/Sigmoid_2:y:0!lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_35_matmul_readvariableop_resource-lstm_cell_35_matmul_1_readvariableop_resource,lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1705471*
condR
while_cond_1705470*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity�
NoOpNoOp$^lstm_cell_35/BiasAdd/ReadVariableOp#^lstm_cell_35/MatMul/ReadVariableOp%^lstm_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_35/BiasAdd/ReadVariableOp#lstm_cell_35/BiasAdd/ReadVariableOp2H
"lstm_cell_35/MatMul/ReadVariableOp"lstm_cell_35/MatMul/ReadVariableOp2L
$lstm_cell_35/MatMul_1/ReadVariableOp$lstm_cell_35/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
I__inference_lstm_cell_35_layer_call_and_return_conditional_losses_1706178

inputs
states_0
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	2�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identity_2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������2:���������2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������2
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������2
"
_user_specified_name
states/1
�V
�
"forward_lstm_11_while_body_1703739<
8forward_lstm_11_while_forward_lstm_11_while_loop_counterB
>forward_lstm_11_while_forward_lstm_11_while_maximum_iterations%
!forward_lstm_11_while_placeholder'
#forward_lstm_11_while_placeholder_1'
#forward_lstm_11_while_placeholder_2'
#forward_lstm_11_while_placeholder_3;
7forward_lstm_11_while_forward_lstm_11_strided_slice_1_0w
sforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0V
Cforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0:	�X
Eforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0:	2�S
Dforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0:	�"
forward_lstm_11_while_identity$
 forward_lstm_11_while_identity_1$
 forward_lstm_11_while_identity_2$
 forward_lstm_11_while_identity_3$
 forward_lstm_11_while_identity_4$
 forward_lstm_11_while_identity_59
5forward_lstm_11_while_forward_lstm_11_strided_slice_1u
qforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensorT
Aforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource:	�V
Cforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource:	2�Q
Bforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource:	���9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp�8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp�:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp�
Gforward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2I
Gforward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape�
9forward_lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_11_while_placeholderPforward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02;
9forward_lstm_11/while/TensorArrayV2Read/TensorListGetItem�
8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOpCforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02:
8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp�
)forward_lstm_11/while/lstm_cell_34/MatMulMatMul@forward_lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:0@forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)forward_lstm_11/while/lstm_cell_34/MatMul�
:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOpEforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02<
:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp�
+forward_lstm_11/while/lstm_cell_34/MatMul_1MatMul#forward_lstm_11_while_placeholder_2Bforward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+forward_lstm_11/while/lstm_cell_34/MatMul_1�
&forward_lstm_11/while/lstm_cell_34/addAddV23forward_lstm_11/while/lstm_cell_34/MatMul:product:05forward_lstm_11/while/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&forward_lstm_11/while/lstm_cell_34/add�
9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOpDforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02;
9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp�
*forward_lstm_11/while/lstm_cell_34/BiasAddBiasAdd*forward_lstm_11/while/lstm_cell_34/add:z:0Aforward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*forward_lstm_11/while/lstm_cell_34/BiasAdd�
2forward_lstm_11/while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2forward_lstm_11/while/lstm_cell_34/split/split_dim�
(forward_lstm_11/while/lstm_cell_34/splitSplit;forward_lstm_11/while/lstm_cell_34/split/split_dim:output:03forward_lstm_11/while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2*
(forward_lstm_11/while/lstm_cell_34/split�
*forward_lstm_11/while/lstm_cell_34/SigmoidSigmoid1forward_lstm_11/while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22,
*forward_lstm_11/while/lstm_cell_34/Sigmoid�
,forward_lstm_11/while/lstm_cell_34/Sigmoid_1Sigmoid1forward_lstm_11/while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22.
,forward_lstm_11/while/lstm_cell_34/Sigmoid_1�
&forward_lstm_11/while/lstm_cell_34/mulMul0forward_lstm_11/while/lstm_cell_34/Sigmoid_1:y:0#forward_lstm_11_while_placeholder_3*
T0*'
_output_shapes
:���������22(
&forward_lstm_11/while/lstm_cell_34/mul�
'forward_lstm_11/while/lstm_cell_34/ReluRelu1forward_lstm_11/while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22)
'forward_lstm_11/while/lstm_cell_34/Relu�
(forward_lstm_11/while/lstm_cell_34/mul_1Mul.forward_lstm_11/while/lstm_cell_34/Sigmoid:y:05forward_lstm_11/while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_11/while/lstm_cell_34/mul_1�
(forward_lstm_11/while/lstm_cell_34/add_1AddV2*forward_lstm_11/while/lstm_cell_34/mul:z:0,forward_lstm_11/while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_11/while/lstm_cell_34/add_1�
,forward_lstm_11/while/lstm_cell_34/Sigmoid_2Sigmoid1forward_lstm_11/while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22.
,forward_lstm_11/while/lstm_cell_34/Sigmoid_2�
)forward_lstm_11/while/lstm_cell_34/Relu_1Relu,forward_lstm_11/while/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22+
)forward_lstm_11/while/lstm_cell_34/Relu_1�
(forward_lstm_11/while/lstm_cell_34/mul_2Mul0forward_lstm_11/while/lstm_cell_34/Sigmoid_2:y:07forward_lstm_11/while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_11/while/lstm_cell_34/mul_2�
:forward_lstm_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_11_while_placeholder_1!forward_lstm_11_while_placeholder,forward_lstm_11/while/lstm_cell_34/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_11/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_11/while/add/y�
forward_lstm_11/while/addAddV2!forward_lstm_11_while_placeholder$forward_lstm_11/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/while/add�
forward_lstm_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_11/while/add_1/y�
forward_lstm_11/while/add_1AddV28forward_lstm_11_while_forward_lstm_11_while_loop_counter&forward_lstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/while/add_1�
forward_lstm_11/while/IdentityIdentityforward_lstm_11/while/add_1:z:0^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_11/while/Identity�
 forward_lstm_11/while/Identity_1Identity>forward_lstm_11_while_forward_lstm_11_while_maximum_iterations^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_11/while/Identity_1�
 forward_lstm_11/while/Identity_2Identityforward_lstm_11/while/add:z:0^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_11/while/Identity_2�
 forward_lstm_11/while/Identity_3IdentityJforward_lstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_11/while/Identity_3�
 forward_lstm_11/while/Identity_4Identity,forward_lstm_11/while/lstm_cell_34/mul_2:z:0^forward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/while/Identity_4�
 forward_lstm_11/while/Identity_5Identity,forward_lstm_11/while/lstm_cell_34/add_1:z:0^forward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/while/Identity_5�
forward_lstm_11/while/NoOpNoOp:^forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp9^forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp;^forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_11/while/NoOp"p
5forward_lstm_11_while_forward_lstm_11_strided_slice_17forward_lstm_11_while_forward_lstm_11_strided_slice_1_0"I
forward_lstm_11_while_identity'forward_lstm_11/while/Identity:output:0"M
 forward_lstm_11_while_identity_1)forward_lstm_11/while/Identity_1:output:0"M
 forward_lstm_11_while_identity_2)forward_lstm_11/while/Identity_2:output:0"M
 forward_lstm_11_while_identity_3)forward_lstm_11/while/Identity_3:output:0"M
 forward_lstm_11_while_identity_4)forward_lstm_11/while/Identity_4:output:0"M
 forward_lstm_11_while_identity_5)forward_lstm_11/while/Identity_5:output:0"�
Bforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resourceDforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0"�
Cforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resourceEforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0"�
Aforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resourceCforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0"�
qforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensorsforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2v
9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp2t
8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp2x
:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
�
�
1__inference_forward_lstm_11_layer_call_fn_1704754

inputs
unknown:	�
	unknown_0:	2�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_17021942
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
#backward_lstm_11_while_cond_1704234>
:backward_lstm_11_while_backward_lstm_11_while_loop_counterD
@backward_lstm_11_while_backward_lstm_11_while_maximum_iterations&
"backward_lstm_11_while_placeholder(
$backward_lstm_11_while_placeholder_1(
$backward_lstm_11_while_placeholder_2(
$backward_lstm_11_while_placeholder_3(
$backward_lstm_11_while_placeholder_4@
<backward_lstm_11_while_less_backward_lstm_11_strided_slice_1W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1704234___redundant_placeholder0W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1704234___redundant_placeholder1W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1704234___redundant_placeholder2W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1704234___redundant_placeholder3W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1704234___redundant_placeholder4#
backward_lstm_11_while_identity
�
backward_lstm_11/while/LessLess"backward_lstm_11_while_placeholder<backward_lstm_11_while_less_backward_lstm_11_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_11/while/Less�
backward_lstm_11/while/IdentityIdentitybackward_lstm_11/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_11/while/Identity"K
backward_lstm_11_while_identity(backward_lstm_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :���������2:���������2:���������2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
�
�
2__inference_backward_lstm_11_layer_call_fn_1705380
inputs_0
unknown:	�
	unknown_0:	2�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_17012462
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�?
�
while_body_1701937
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_35_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_35_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_35_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_35_matmul_readvariableop_resource:	�F
3while_lstm_cell_35_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_35_biasadd_readvariableop_resource:	���)while/lstm_cell_35/BiasAdd/ReadVariableOp�(while/lstm_cell_35/MatMul/ReadVariableOp�*while/lstm_cell_35/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_35/MatMul/ReadVariableOp�
while/lstm_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/MatMul�
*while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_35/MatMul_1/ReadVariableOp�
while/lstm_cell_35/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/MatMul_1�
while/lstm_cell_35/addAddV2#while/lstm_cell_35/MatMul:product:0%while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/add�
)while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_35/BiasAdd/ReadVariableOp�
while/lstm_cell_35/BiasAddBiasAddwhile/lstm_cell_35/add:z:01while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/BiasAdd�
"while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_35/split/split_dim�
while/lstm_cell_35/splitSplit+while/lstm_cell_35/split/split_dim:output:0#while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_35/split�
while/lstm_cell_35/SigmoidSigmoid!while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Sigmoid�
while/lstm_cell_35/Sigmoid_1Sigmoid!while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Sigmoid_1�
while/lstm_cell_35/mulMul while/lstm_cell_35/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/mul�
while/lstm_cell_35/ReluRelu!while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Relu�
while/lstm_cell_35/mul_1Mulwhile/lstm_cell_35/Sigmoid:y:0%while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/mul_1�
while/lstm_cell_35/add_1AddV2while/lstm_cell_35/mul:z:0while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/add_1�
while/lstm_cell_35/Sigmoid_2Sigmoid!while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Sigmoid_2�
while/lstm_cell_35/Relu_1Reluwhile/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Relu_1�
while/lstm_cell_35/mul_2Mul while/lstm_cell_35/Sigmoid_2:y:0'while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_35/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_35/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_35/BiasAdd/ReadVariableOp)^while/lstm_cell_35/MatMul/ReadVariableOp+^while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_35_biasadd_readvariableop_resource4while_lstm_cell_35_biasadd_readvariableop_resource_0"l
3while_lstm_cell_35_matmul_1_readvariableop_resource5while_lstm_cell_35_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_35_matmul_readvariableop_resource3while_lstm_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_35/BiasAdd/ReadVariableOp)while/lstm_cell_35/BiasAdd/ReadVariableOp2T
(while/lstm_cell_35/MatMul/ReadVariableOp(while/lstm_cell_35/MatMul/ReadVariableOp2X
*while/lstm_cell_35/MatMul_1/ReadVariableOp*while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
�d
�
#backward_lstm_11_while_body_1704235>
:backward_lstm_11_while_backward_lstm_11_while_loop_counterD
@backward_lstm_11_while_backward_lstm_11_while_maximum_iterations&
"backward_lstm_11_while_placeholder(
$backward_lstm_11_while_placeholder_1(
$backward_lstm_11_while_placeholder_2(
$backward_lstm_11_while_placeholder_3(
$backward_lstm_11_while_placeholder_4=
9backward_lstm_11_while_backward_lstm_11_strided_slice_1_0y
ubackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_08
4backward_lstm_11_while_less_backward_lstm_11_sub_1_0W
Dbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0:	�Y
Fbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0:	2�T
Ebackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0:	�#
backward_lstm_11_while_identity%
!backward_lstm_11_while_identity_1%
!backward_lstm_11_while_identity_2%
!backward_lstm_11_while_identity_3%
!backward_lstm_11_while_identity_4%
!backward_lstm_11_while_identity_5%
!backward_lstm_11_while_identity_6;
7backward_lstm_11_while_backward_lstm_11_strided_slice_1w
sbackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor6
2backward_lstm_11_while_less_backward_lstm_11_sub_1U
Bbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource:	�W
Dbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource:	2�R
Cbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource:	���:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp�9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp�;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp�
Hbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2J
Hbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape�
:backward_lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_11_while_placeholderQbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02<
:backward_lstm_11/while/TensorArrayV2Read/TensorListGetItem�
backward_lstm_11/while/LessLess4backward_lstm_11_while_less_backward_lstm_11_sub_1_0"backward_lstm_11_while_placeholder*
T0*#
_output_shapes
:���������2
backward_lstm_11/while/Less�
9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOpDbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02;
9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp�
*backward_lstm_11/while/lstm_cell_35/MatMulMatMulAbackward_lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:0Abackward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*backward_lstm_11/while/lstm_cell_35/MatMul�
;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOpFbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02=
;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp�
,backward_lstm_11/while/lstm_cell_35/MatMul_1MatMul$backward_lstm_11_while_placeholder_3Cbackward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2.
,backward_lstm_11/while/lstm_cell_35/MatMul_1�
'backward_lstm_11/while/lstm_cell_35/addAddV24backward_lstm_11/while/lstm_cell_35/MatMul:product:06backward_lstm_11/while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2)
'backward_lstm_11/while/lstm_cell_35/add�
:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOpEbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02<
:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp�
+backward_lstm_11/while/lstm_cell_35/BiasAddBiasAdd+backward_lstm_11/while/lstm_cell_35/add:z:0Bbackward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+backward_lstm_11/while/lstm_cell_35/BiasAdd�
3backward_lstm_11/while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3backward_lstm_11/while/lstm_cell_35/split/split_dim�
)backward_lstm_11/while/lstm_cell_35/splitSplit<backward_lstm_11/while/lstm_cell_35/split/split_dim:output:04backward_lstm_11/while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2+
)backward_lstm_11/while/lstm_cell_35/split�
+backward_lstm_11/while/lstm_cell_35/SigmoidSigmoid2backward_lstm_11/while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22-
+backward_lstm_11/while/lstm_cell_35/Sigmoid�
-backward_lstm_11/while/lstm_cell_35/Sigmoid_1Sigmoid2backward_lstm_11/while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22/
-backward_lstm_11/while/lstm_cell_35/Sigmoid_1�
'backward_lstm_11/while/lstm_cell_35/mulMul1backward_lstm_11/while/lstm_cell_35/Sigmoid_1:y:0$backward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22)
'backward_lstm_11/while/lstm_cell_35/mul�
(backward_lstm_11/while/lstm_cell_35/ReluRelu2backward_lstm_11/while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22*
(backward_lstm_11/while/lstm_cell_35/Relu�
)backward_lstm_11/while/lstm_cell_35/mul_1Mul/backward_lstm_11/while/lstm_cell_35/Sigmoid:y:06backward_lstm_11/while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_11/while/lstm_cell_35/mul_1�
)backward_lstm_11/while/lstm_cell_35/add_1AddV2+backward_lstm_11/while/lstm_cell_35/mul:z:0-backward_lstm_11/while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_11/while/lstm_cell_35/add_1�
-backward_lstm_11/while/lstm_cell_35/Sigmoid_2Sigmoid2backward_lstm_11/while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22/
-backward_lstm_11/while/lstm_cell_35/Sigmoid_2�
*backward_lstm_11/while/lstm_cell_35/Relu_1Relu-backward_lstm_11/while/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22,
*backward_lstm_11/while/lstm_cell_35/Relu_1�
)backward_lstm_11/while/lstm_cell_35/mul_2Mul1backward_lstm_11/while/lstm_cell_35/Sigmoid_2:y:08backward_lstm_11/while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_11/while/lstm_cell_35/mul_2�
backward_lstm_11/while/SelectSelectbackward_lstm_11/while/Less:z:0-backward_lstm_11/while/lstm_cell_35/mul_2:z:0$backward_lstm_11_while_placeholder_2*
T0*'
_output_shapes
:���������22
backward_lstm_11/while/Select�
backward_lstm_11/while/Select_1Selectbackward_lstm_11/while/Less:z:0-backward_lstm_11/while/lstm_cell_35/mul_2:z:0$backward_lstm_11_while_placeholder_3*
T0*'
_output_shapes
:���������22!
backward_lstm_11/while/Select_1�
backward_lstm_11/while/Select_2Selectbackward_lstm_11/while/Less:z:0-backward_lstm_11/while/lstm_cell_35/add_1:z:0$backward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22!
backward_lstm_11/while/Select_2�
;backward_lstm_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_11_while_placeholder_1"backward_lstm_11_while_placeholder&backward_lstm_11/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_11/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_11/while/add/y�
backward_lstm_11/while/addAddV2"backward_lstm_11_while_placeholder%backward_lstm_11/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/while/add�
backward_lstm_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_11/while/add_1/y�
backward_lstm_11/while/add_1AddV2:backward_lstm_11_while_backward_lstm_11_while_loop_counter'backward_lstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/while/add_1�
backward_lstm_11/while/IdentityIdentity backward_lstm_11/while/add_1:z:0^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_11/while/Identity�
!backward_lstm_11/while/Identity_1Identity@backward_lstm_11_while_backward_lstm_11_while_maximum_iterations^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_11/while/Identity_1�
!backward_lstm_11/while/Identity_2Identitybackward_lstm_11/while/add:z:0^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_11/while/Identity_2�
!backward_lstm_11/while/Identity_3IdentityKbackward_lstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_11/while/Identity_3�
!backward_lstm_11/while/Identity_4Identity&backward_lstm_11/while/Select:output:0^backward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/while/Identity_4�
!backward_lstm_11/while/Identity_5Identity(backward_lstm_11/while/Select_1:output:0^backward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/while/Identity_5�
!backward_lstm_11/while/Identity_6Identity(backward_lstm_11/while/Select_2:output:0^backward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/while/Identity_6�
backward_lstm_11/while/NoOpNoOp;^backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp:^backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp<^backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_11/while/NoOp"t
7backward_lstm_11_while_backward_lstm_11_strided_slice_19backward_lstm_11_while_backward_lstm_11_strided_slice_1_0"K
backward_lstm_11_while_identity(backward_lstm_11/while/Identity:output:0"O
!backward_lstm_11_while_identity_1*backward_lstm_11/while/Identity_1:output:0"O
!backward_lstm_11_while_identity_2*backward_lstm_11/while/Identity_2:output:0"O
!backward_lstm_11_while_identity_3*backward_lstm_11/while/Identity_3:output:0"O
!backward_lstm_11_while_identity_4*backward_lstm_11/while/Identity_4:output:0"O
!backward_lstm_11_while_identity_5*backward_lstm_11/while/Identity_5:output:0"O
!backward_lstm_11_while_identity_6*backward_lstm_11/while/Identity_6:output:0"j
2backward_lstm_11_while_less_backward_lstm_11_sub_14backward_lstm_11_while_less_backward_lstm_11_sub_1_0"�
Cbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resourceEbackward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0"�
Dbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resourceFbackward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0"�
Bbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resourceDbackward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0"�
sbackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensorubackward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2x
:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp:backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp2v
9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp9backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp2z
;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp;backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:���������
�	
�
2__inference_bidirectional_11_layer_call_fn_1703334
inputs_0
unknown:	�
	unknown_0:	2�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	2�
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_17022422
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs/0
�F
�
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_1700612

inputs'
lstm_cell_34_1700530:	�'
lstm_cell_34_1700532:	2�#
lstm_cell_34_1700534:	�
identity��$lstm_cell_34/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packedc
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedg
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
$lstm_cell_34/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_34_1700530lstm_cell_34_1700532lstm_cell_34_1700534*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������2:���������2:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_lstm_cell_34_layer_call_and_return_conditional_losses_17004652&
$lstm_cell_34/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_34_1700530lstm_cell_34_1700532lstm_cell_34_1700534*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1700543*
condR
while_cond_1700542*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity}
NoOpNoOp%^lstm_cell_34/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_34/StatefulPartitionedCall$lstm_cell_34/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
"forward_lstm_11_while_cond_1703738<
8forward_lstm_11_while_forward_lstm_11_while_loop_counterB
>forward_lstm_11_while_forward_lstm_11_while_maximum_iterations%
!forward_lstm_11_while_placeholder'
#forward_lstm_11_while_placeholder_1'
#forward_lstm_11_while_placeholder_2'
#forward_lstm_11_while_placeholder_3>
:forward_lstm_11_while_less_forward_lstm_11_strided_slice_1U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1703738___redundant_placeholder0U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1703738___redundant_placeholder1U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1703738___redundant_placeholder2U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1703738___redundant_placeholder3"
forward_lstm_11_while_identity
�
forward_lstm_11/while/LessLess!forward_lstm_11_while_placeholder:forward_lstm_11_while_less_forward_lstm_11_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_11/while/Less�
forward_lstm_11/while/IdentityIdentityforward_lstm_11/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_11/while/Identity"I
forward_lstm_11_while_identity'forward_lstm_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
�
�
2__inference_backward_lstm_11_layer_call_fn_1705369
inputs_0
unknown:	�
	unknown_0:	2�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_17010342
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
I__inference_lstm_cell_34_layer_call_and_return_conditional_losses_1706112

inputs
states_0
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	2�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������22

Identity_2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������2:���������2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������2
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������2
"
_user_specified_name
states/1
�
�
.__inference_lstm_cell_35_layer_call_fn_1706129

inputs
states_0
states_1
unknown:	�
	unknown_0:	2�
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������2:���������2:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_lstm_cell_35_layer_call_and_return_conditional_losses_17009512
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������22

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������2:���������2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������2
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������2
"
_user_specified_name
states/1
�^
�
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_1705861

inputs>
+lstm_cell_35_matmul_readvariableop_resource:	�@
-lstm_cell_35_matmul_1_readvariableop_resource:	2�;
,lstm_cell_35_biasadd_readvariableop_resource:	�
identity��#lstm_cell_35/BiasAdd/ReadVariableOp�"lstm_cell_35/MatMul/ReadVariableOp�$lstm_cell_35/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packedc
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedg
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis�
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'���������������������������2
	ReverseV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_35/MatMul/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_35/MatMul/ReadVariableOp�
lstm_cell_35/MatMulMatMulstrided_slice_2:output:0*lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/MatMul�
$lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_35/MatMul_1/ReadVariableOp�
lstm_cell_35/MatMul_1MatMulzeros:output:0,lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/MatMul_1�
lstm_cell_35/addAddV2lstm_cell_35/MatMul:product:0lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/add�
#lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_35/BiasAdd/ReadVariableOp�
lstm_cell_35/BiasAddBiasAddlstm_cell_35/add:z:0+lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/BiasAdd~
lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_35/split/split_dim�
lstm_cell_35/splitSplit%lstm_cell_35/split/split_dim:output:0lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_35/split�
lstm_cell_35/SigmoidSigmoidlstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/Sigmoid�
lstm_cell_35/Sigmoid_1Sigmoidlstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_35/Sigmoid_1�
lstm_cell_35/mulMullstm_cell_35/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/mul}
lstm_cell_35/ReluRelulstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_35/Relu�
lstm_cell_35/mul_1Mullstm_cell_35/Sigmoid:y:0lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/mul_1�
lstm_cell_35/add_1AddV2lstm_cell_35/mul:z:0lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/add_1�
lstm_cell_35/Sigmoid_2Sigmoidlstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_35/Sigmoid_2|
lstm_cell_35/Relu_1Relulstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/Relu_1�
lstm_cell_35/mul_2Mullstm_cell_35/Sigmoid_2:y:0!lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_35_matmul_readvariableop_resource-lstm_cell_35_matmul_1_readvariableop_resource,lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1705777*
condR
while_cond_1705776*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity�
NoOpNoOp$^lstm_cell_35/BiasAdd/ReadVariableOp#^lstm_cell_35/MatMul/ReadVariableOp%^lstm_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_35/BiasAdd/ReadVariableOp#lstm_cell_35/BiasAdd/ReadVariableOp2H
"lstm_cell_35/MatMul/ReadVariableOp"lstm_cell_35/MatMul/ReadVariableOp2L
$lstm_cell_35/MatMul_1/ReadVariableOp$lstm_cell_35/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�

�
/__inference_sequential_11_layer_call_fn_1703224

inputs
inputs_1	
unknown:	�
	unknown_0:	2�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	2�
	unknown_4:	�
	unknown_5:d
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_11_layer_call_and_return_conditional_losses_17031832
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�b
�
"forward_lstm_11_while_body_1704414<
8forward_lstm_11_while_forward_lstm_11_while_loop_counterB
>forward_lstm_11_while_forward_lstm_11_while_maximum_iterations%
!forward_lstm_11_while_placeholder'
#forward_lstm_11_while_placeholder_1'
#forward_lstm_11_while_placeholder_2'
#forward_lstm_11_while_placeholder_3'
#forward_lstm_11_while_placeholder_4;
7forward_lstm_11_while_forward_lstm_11_strided_slice_1_0w
sforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_08
4forward_lstm_11_while_greater_forward_lstm_11_cast_0V
Cforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0:	�X
Eforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0:	2�S
Dforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0:	�"
forward_lstm_11_while_identity$
 forward_lstm_11_while_identity_1$
 forward_lstm_11_while_identity_2$
 forward_lstm_11_while_identity_3$
 forward_lstm_11_while_identity_4$
 forward_lstm_11_while_identity_5$
 forward_lstm_11_while_identity_69
5forward_lstm_11_while_forward_lstm_11_strided_slice_1u
qforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor6
2forward_lstm_11_while_greater_forward_lstm_11_castT
Aforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource:	�V
Cforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource:	2�Q
Bforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource:	���9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp�8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp�:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp�
Gforward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2I
Gforward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape�
9forward_lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_11_while_placeholderPforward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02;
9forward_lstm_11/while/TensorArrayV2Read/TensorListGetItem�
forward_lstm_11/while/GreaterGreater4forward_lstm_11_while_greater_forward_lstm_11_cast_0!forward_lstm_11_while_placeholder*
T0*#
_output_shapes
:���������2
forward_lstm_11/while/Greater�
8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOpCforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02:
8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp�
)forward_lstm_11/while/lstm_cell_34/MatMulMatMul@forward_lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:0@forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)forward_lstm_11/while/lstm_cell_34/MatMul�
:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOpEforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02<
:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp�
+forward_lstm_11/while/lstm_cell_34/MatMul_1MatMul#forward_lstm_11_while_placeholder_3Bforward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+forward_lstm_11/while/lstm_cell_34/MatMul_1�
&forward_lstm_11/while/lstm_cell_34/addAddV23forward_lstm_11/while/lstm_cell_34/MatMul:product:05forward_lstm_11/while/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&forward_lstm_11/while/lstm_cell_34/add�
9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOpDforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02;
9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp�
*forward_lstm_11/while/lstm_cell_34/BiasAddBiasAdd*forward_lstm_11/while/lstm_cell_34/add:z:0Aforward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*forward_lstm_11/while/lstm_cell_34/BiasAdd�
2forward_lstm_11/while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2forward_lstm_11/while/lstm_cell_34/split/split_dim�
(forward_lstm_11/while/lstm_cell_34/splitSplit;forward_lstm_11/while/lstm_cell_34/split/split_dim:output:03forward_lstm_11/while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2*
(forward_lstm_11/while/lstm_cell_34/split�
*forward_lstm_11/while/lstm_cell_34/SigmoidSigmoid1forward_lstm_11/while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22,
*forward_lstm_11/while/lstm_cell_34/Sigmoid�
,forward_lstm_11/while/lstm_cell_34/Sigmoid_1Sigmoid1forward_lstm_11/while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22.
,forward_lstm_11/while/lstm_cell_34/Sigmoid_1�
&forward_lstm_11/while/lstm_cell_34/mulMul0forward_lstm_11/while/lstm_cell_34/Sigmoid_1:y:0#forward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22(
&forward_lstm_11/while/lstm_cell_34/mul�
'forward_lstm_11/while/lstm_cell_34/ReluRelu1forward_lstm_11/while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22)
'forward_lstm_11/while/lstm_cell_34/Relu�
(forward_lstm_11/while/lstm_cell_34/mul_1Mul.forward_lstm_11/while/lstm_cell_34/Sigmoid:y:05forward_lstm_11/while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_11/while/lstm_cell_34/mul_1�
(forward_lstm_11/while/lstm_cell_34/add_1AddV2*forward_lstm_11/while/lstm_cell_34/mul:z:0,forward_lstm_11/while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_11/while/lstm_cell_34/add_1�
,forward_lstm_11/while/lstm_cell_34/Sigmoid_2Sigmoid1forward_lstm_11/while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22.
,forward_lstm_11/while/lstm_cell_34/Sigmoid_2�
)forward_lstm_11/while/lstm_cell_34/Relu_1Relu,forward_lstm_11/while/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22+
)forward_lstm_11/while/lstm_cell_34/Relu_1�
(forward_lstm_11/while/lstm_cell_34/mul_2Mul0forward_lstm_11/while/lstm_cell_34/Sigmoid_2:y:07forward_lstm_11/while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_11/while/lstm_cell_34/mul_2�
forward_lstm_11/while/SelectSelect!forward_lstm_11/while/Greater:z:0,forward_lstm_11/while/lstm_cell_34/mul_2:z:0#forward_lstm_11_while_placeholder_2*
T0*'
_output_shapes
:���������22
forward_lstm_11/while/Select�
forward_lstm_11/while/Select_1Select!forward_lstm_11/while/Greater:z:0,forward_lstm_11/while/lstm_cell_34/mul_2:z:0#forward_lstm_11_while_placeholder_3*
T0*'
_output_shapes
:���������22 
forward_lstm_11/while/Select_1�
forward_lstm_11/while/Select_2Select!forward_lstm_11/while/Greater:z:0,forward_lstm_11/while/lstm_cell_34/add_1:z:0#forward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22 
forward_lstm_11/while/Select_2�
:forward_lstm_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_11_while_placeholder_1!forward_lstm_11_while_placeholder%forward_lstm_11/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_11/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_11/while/add/y�
forward_lstm_11/while/addAddV2!forward_lstm_11_while_placeholder$forward_lstm_11/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/while/add�
forward_lstm_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_11/while/add_1/y�
forward_lstm_11/while/add_1AddV28forward_lstm_11_while_forward_lstm_11_while_loop_counter&forward_lstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/while/add_1�
forward_lstm_11/while/IdentityIdentityforward_lstm_11/while/add_1:z:0^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_11/while/Identity�
 forward_lstm_11/while/Identity_1Identity>forward_lstm_11_while_forward_lstm_11_while_maximum_iterations^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_11/while/Identity_1�
 forward_lstm_11/while/Identity_2Identityforward_lstm_11/while/add:z:0^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_11/while/Identity_2�
 forward_lstm_11/while/Identity_3IdentityJforward_lstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_11/while/Identity_3�
 forward_lstm_11/while/Identity_4Identity%forward_lstm_11/while/Select:output:0^forward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/while/Identity_4�
 forward_lstm_11/while/Identity_5Identity'forward_lstm_11/while/Select_1:output:0^forward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/while/Identity_5�
 forward_lstm_11/while/Identity_6Identity'forward_lstm_11/while/Select_2:output:0^forward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/while/Identity_6�
forward_lstm_11/while/NoOpNoOp:^forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp9^forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp;^forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_11/while/NoOp"p
5forward_lstm_11_while_forward_lstm_11_strided_slice_17forward_lstm_11_while_forward_lstm_11_strided_slice_1_0"j
2forward_lstm_11_while_greater_forward_lstm_11_cast4forward_lstm_11_while_greater_forward_lstm_11_cast_0"I
forward_lstm_11_while_identity'forward_lstm_11/while/Identity:output:0"M
 forward_lstm_11_while_identity_1)forward_lstm_11/while/Identity_1:output:0"M
 forward_lstm_11_while_identity_2)forward_lstm_11/while/Identity_2:output:0"M
 forward_lstm_11_while_identity_3)forward_lstm_11/while/Identity_3:output:0"M
 forward_lstm_11_while_identity_4)forward_lstm_11/while/Identity_4:output:0"M
 forward_lstm_11_while_identity_5)forward_lstm_11/while/Identity_5:output:0"M
 forward_lstm_11_while_identity_6)forward_lstm_11/while/Identity_6:output:0"�
Bforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resourceDforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0"�
Cforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resourceEforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0"�
Aforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resourceCforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0"�
qforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensorsforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2v
9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp2t
8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp2x
:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:���������
�
�
while_cond_1705929
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1705929___redundant_placeholder05
1while_while_cond_1705929___redundant_placeholder15
1while_while_cond_1705929___redundant_placeholder25
1while_while_cond_1705929___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
��
�
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_1703120

inputs
inputs_1	N
;forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource:	�P
=forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource:	2�K
<forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource:	�O
<backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource:	�Q
>backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource:	2�L
=backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource:	�
identity��4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp�3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp�5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp�backward_lstm_11/while�3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp�2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp�4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp�forward_lstm_11/while�
$forward_lstm_11/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$forward_lstm_11/RaggedToTensor/zeros�
$forward_lstm_11/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2&
$forward_lstm_11/RaggedToTensor/Const�
3forward_lstm_11/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-forward_lstm_11/RaggedToTensor/Const:output:0inputs-forward_lstm_11/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3forward_lstm_11/RaggedToTensor/RaggedTensorToTensor�
:forward_lstm_11/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack�
<forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1�
<forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2�
4forward_lstm_11/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cforward_lstm_11/RaggedNestedRowLengths/strided_slice/stack:output:0Eforward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1:output:0Eforward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask26
4forward_lstm_11/RaggedNestedRowLengths/strided_slice�
<forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack�
>forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2@
>forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1�
>forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2�
6forward_lstm_11/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Eforward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gforward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gforward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask28
6forward_lstm_11/RaggedNestedRowLengths/strided_slice_1�
*forward_lstm_11/RaggedNestedRowLengths/subSub=forward_lstm_11/RaggedNestedRowLengths/strided_slice:output:0?forward_lstm_11/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2,
*forward_lstm_11/RaggedNestedRowLengths/sub�
forward_lstm_11/CastCast.forward_lstm_11/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
forward_lstm_11/Cast�
forward_lstm_11/ShapeShape<forward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_11/Shape�
#forward_lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_11/strided_slice/stack�
%forward_lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_11/strided_slice/stack_1�
%forward_lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_11/strided_slice/stack_2�
forward_lstm_11/strided_sliceStridedSliceforward_lstm_11/Shape:output:0,forward_lstm_11/strided_slice/stack:output:0.forward_lstm_11/strided_slice/stack_1:output:0.forward_lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_11/strided_slice|
forward_lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_11/zeros/mul/y�
forward_lstm_11/zeros/mulMul&forward_lstm_11/strided_slice:output:0$forward_lstm_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros/mul
forward_lstm_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_11/zeros/Less/y�
forward_lstm_11/zeros/LessLessforward_lstm_11/zeros/mul:z:0%forward_lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros/Less�
forward_lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_11/zeros/packed/1�
forward_lstm_11/zeros/packedPack&forward_lstm_11/strided_slice:output:0'forward_lstm_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_11/zeros/packed�
forward_lstm_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_11/zeros/Const�
forward_lstm_11/zerosFill%forward_lstm_11/zeros/packed:output:0$forward_lstm_11/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_11/zeros�
forward_lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_11/zeros_1/mul/y�
forward_lstm_11/zeros_1/mulMul&forward_lstm_11/strided_slice:output:0&forward_lstm_11/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros_1/mul�
forward_lstm_11/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
forward_lstm_11/zeros_1/Less/y�
forward_lstm_11/zeros_1/LessLessforward_lstm_11/zeros_1/mul:z:0'forward_lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros_1/Less�
 forward_lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_11/zeros_1/packed/1�
forward_lstm_11/zeros_1/packedPack&forward_lstm_11/strided_slice:output:0)forward_lstm_11/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_11/zeros_1/packed�
forward_lstm_11/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_11/zeros_1/Const�
forward_lstm_11/zeros_1Fill'forward_lstm_11/zeros_1/packed:output:0&forward_lstm_11/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_11/zeros_1�
forward_lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_11/transpose/perm�
forward_lstm_11/transpose	Transpose<forward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0'forward_lstm_11/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
forward_lstm_11/transpose
forward_lstm_11/Shape_1Shapeforward_lstm_11/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_11/Shape_1�
%forward_lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_11/strided_slice_1/stack�
'forward_lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_1/stack_1�
'forward_lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_1/stack_2�
forward_lstm_11/strided_slice_1StridedSlice forward_lstm_11/Shape_1:output:0.forward_lstm_11/strided_slice_1/stack:output:00forward_lstm_11/strided_slice_1/stack_1:output:00forward_lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_11/strided_slice_1�
+forward_lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+forward_lstm_11/TensorArrayV2/element_shape�
forward_lstm_11/TensorArrayV2TensorListReserve4forward_lstm_11/TensorArrayV2/element_shape:output:0(forward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_11/TensorArrayV2�
Eforward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2G
Eforward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape�
7forward_lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_11/transpose:y:0Nforward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_11/TensorArrayUnstack/TensorListFromTensor�
%forward_lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_11/strided_slice_2/stack�
'forward_lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_2/stack_1�
'forward_lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_2/stack_2�
forward_lstm_11/strided_slice_2StridedSliceforward_lstm_11/transpose:y:0.forward_lstm_11/strided_slice_2/stack:output:00forward_lstm_11/strided_slice_2/stack_1:output:00forward_lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2!
forward_lstm_11/strided_slice_2�
2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp;forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp�
#forward_lstm_11/lstm_cell_34/MatMulMatMul(forward_lstm_11/strided_slice_2:output:0:forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#forward_lstm_11/lstm_cell_34/MatMul�
4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp=forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype026
4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp�
%forward_lstm_11/lstm_cell_34/MatMul_1MatMulforward_lstm_11/zeros:output:0<forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%forward_lstm_11/lstm_cell_34/MatMul_1�
 forward_lstm_11/lstm_cell_34/addAddV2-forward_lstm_11/lstm_cell_34/MatMul:product:0/forward_lstm_11/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2"
 forward_lstm_11/lstm_cell_34/add�
3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp<forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp�
$forward_lstm_11/lstm_cell_34/BiasAddBiasAdd$forward_lstm_11/lstm_cell_34/add:z:0;forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$forward_lstm_11/lstm_cell_34/BiasAdd�
,forward_lstm_11/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,forward_lstm_11/lstm_cell_34/split/split_dim�
"forward_lstm_11/lstm_cell_34/splitSplit5forward_lstm_11/lstm_cell_34/split/split_dim:output:0-forward_lstm_11/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2$
"forward_lstm_11/lstm_cell_34/split�
$forward_lstm_11/lstm_cell_34/SigmoidSigmoid+forward_lstm_11/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22&
$forward_lstm_11/lstm_cell_34/Sigmoid�
&forward_lstm_11/lstm_cell_34/Sigmoid_1Sigmoid+forward_lstm_11/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22(
&forward_lstm_11/lstm_cell_34/Sigmoid_1�
 forward_lstm_11/lstm_cell_34/mulMul*forward_lstm_11/lstm_cell_34/Sigmoid_1:y:0 forward_lstm_11/zeros_1:output:0*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/lstm_cell_34/mul�
!forward_lstm_11/lstm_cell_34/ReluRelu+forward_lstm_11/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22#
!forward_lstm_11/lstm_cell_34/Relu�
"forward_lstm_11/lstm_cell_34/mul_1Mul(forward_lstm_11/lstm_cell_34/Sigmoid:y:0/forward_lstm_11/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_11/lstm_cell_34/mul_1�
"forward_lstm_11/lstm_cell_34/add_1AddV2$forward_lstm_11/lstm_cell_34/mul:z:0&forward_lstm_11/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_11/lstm_cell_34/add_1�
&forward_lstm_11/lstm_cell_34/Sigmoid_2Sigmoid+forward_lstm_11/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22(
&forward_lstm_11/lstm_cell_34/Sigmoid_2�
#forward_lstm_11/lstm_cell_34/Relu_1Relu&forward_lstm_11/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22%
#forward_lstm_11/lstm_cell_34/Relu_1�
"forward_lstm_11/lstm_cell_34/mul_2Mul*forward_lstm_11/lstm_cell_34/Sigmoid_2:y:01forward_lstm_11/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_11/lstm_cell_34/mul_2�
-forward_lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2/
-forward_lstm_11/TensorArrayV2_1/element_shape�
forward_lstm_11/TensorArrayV2_1TensorListReserve6forward_lstm_11/TensorArrayV2_1/element_shape:output:0(forward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_11/TensorArrayV2_1n
forward_lstm_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_11/time�
forward_lstm_11/zeros_like	ZerosLike&forward_lstm_11/lstm_cell_34/mul_2:z:0*
T0*'
_output_shapes
:���������22
forward_lstm_11/zeros_like�
(forward_lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(forward_lstm_11/while/maximum_iterations�
"forward_lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_11/while/loop_counter�
forward_lstm_11/whileWhile+forward_lstm_11/while/loop_counter:output:01forward_lstm_11/while/maximum_iterations:output:0forward_lstm_11/time:output:0(forward_lstm_11/TensorArrayV2_1:handle:0forward_lstm_11/zeros_like:y:0forward_lstm_11/zeros:output:0 forward_lstm_11/zeros_1:output:0(forward_lstm_11/strided_slice_1:output:0Gforward_lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_11/Cast:y:0;forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource=forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource<forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *.
body&R$
"forward_lstm_11_while_body_1702844*.
cond&R$
"forward_lstm_11_while_cond_1702843*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
forward_lstm_11/while�
@forward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2B
@forward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape�
2forward_lstm_11/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_11/while:output:3Iforward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype024
2forward_lstm_11/TensorArrayV2Stack/TensorListStack�
%forward_lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2'
%forward_lstm_11/strided_slice_3/stack�
'forward_lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_11/strided_slice_3/stack_1�
'forward_lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_3/stack_2�
forward_lstm_11/strided_slice_3StridedSlice;forward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_11/strided_slice_3/stack:output:00forward_lstm_11/strided_slice_3/stack_1:output:00forward_lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2!
forward_lstm_11/strided_slice_3�
 forward_lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_11/transpose_1/perm�
forward_lstm_11/transpose_1	Transpose;forward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_11/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
forward_lstm_11/transpose_1�
forward_lstm_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_11/runtime�
%backward_lstm_11/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%backward_lstm_11/RaggedToTensor/zeros�
%backward_lstm_11/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2'
%backward_lstm_11/RaggedToTensor/Const�
4backward_lstm_11/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.backward_lstm_11/RaggedToTensor/Const:output:0inputs.backward_lstm_11/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4backward_lstm_11/RaggedToTensor/RaggedTensorToTensor�
;backward_lstm_11/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack�
=backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1�
=backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2�
5backward_lstm_11/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dbackward_lstm_11/RaggedNestedRowLengths/strided_slice/stack:output:0Fbackward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fbackward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask27
5backward_lstm_11/RaggedNestedRowLengths/strided_slice�
=backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack�
?backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2A
?backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1�
?backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2�
7backward_lstm_11/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fbackward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hbackward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hbackward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask29
7backward_lstm_11/RaggedNestedRowLengths/strided_slice_1�
+backward_lstm_11/RaggedNestedRowLengths/subSub>backward_lstm_11/RaggedNestedRowLengths/strided_slice:output:0@backward_lstm_11/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2-
+backward_lstm_11/RaggedNestedRowLengths/sub�
backward_lstm_11/CastCast/backward_lstm_11/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
backward_lstm_11/Cast�
backward_lstm_11/ShapeShape=backward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_11/Shape�
$backward_lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_11/strided_slice/stack�
&backward_lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_11/strided_slice/stack_1�
&backward_lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_11/strided_slice/stack_2�
backward_lstm_11/strided_sliceStridedSlicebackward_lstm_11/Shape:output:0-backward_lstm_11/strided_slice/stack:output:0/backward_lstm_11/strided_slice/stack_1:output:0/backward_lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_11/strided_slice~
backward_lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_11/zeros/mul/y�
backward_lstm_11/zeros/mulMul'backward_lstm_11/strided_slice:output:0%backward_lstm_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros/mul�
backward_lstm_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
backward_lstm_11/zeros/Less/y�
backward_lstm_11/zeros/LessLessbackward_lstm_11/zeros/mul:z:0&backward_lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros/Less�
backward_lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_11/zeros/packed/1�
backward_lstm_11/zeros/packedPack'backward_lstm_11/strided_slice:output:0(backward_lstm_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_11/zeros/packed�
backward_lstm_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_11/zeros/Const�
backward_lstm_11/zerosFill&backward_lstm_11/zeros/packed:output:0%backward_lstm_11/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_11/zeros�
backward_lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_11/zeros_1/mul/y�
backward_lstm_11/zeros_1/mulMul'backward_lstm_11/strided_slice:output:0'backward_lstm_11/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros_1/mul�
backward_lstm_11/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2!
backward_lstm_11/zeros_1/Less/y�
backward_lstm_11/zeros_1/LessLess backward_lstm_11/zeros_1/mul:z:0(backward_lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros_1/Less�
!backward_lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_11/zeros_1/packed/1�
backward_lstm_11/zeros_1/packedPack'backward_lstm_11/strided_slice:output:0*backward_lstm_11/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_11/zeros_1/packed�
backward_lstm_11/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_11/zeros_1/Const�
backward_lstm_11/zeros_1Fill(backward_lstm_11/zeros_1/packed:output:0'backward_lstm_11/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_11/zeros_1�
backward_lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_11/transpose/perm�
backward_lstm_11/transpose	Transpose=backward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0(backward_lstm_11/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_11/transpose�
backward_lstm_11/Shape_1Shapebackward_lstm_11/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_11/Shape_1�
&backward_lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_11/strided_slice_1/stack�
(backward_lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_1/stack_1�
(backward_lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_1/stack_2�
 backward_lstm_11/strided_slice_1StridedSlice!backward_lstm_11/Shape_1:output:0/backward_lstm_11/strided_slice_1/stack:output:01backward_lstm_11/strided_slice_1/stack_1:output:01backward_lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_11/strided_slice_1�
,backward_lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2.
,backward_lstm_11/TensorArrayV2/element_shape�
backward_lstm_11/TensorArrayV2TensorListReserve5backward_lstm_11/TensorArrayV2/element_shape:output:0)backward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_11/TensorArrayV2�
backward_lstm_11/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_11/ReverseV2/axis�
backward_lstm_11/ReverseV2	ReverseV2backward_lstm_11/transpose:y:0(backward_lstm_11/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_11/ReverseV2�
Fbackward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2H
Fbackward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape�
8backward_lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_11/ReverseV2:output:0Obackward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_11/TensorArrayUnstack/TensorListFromTensor�
&backward_lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_11/strided_slice_2/stack�
(backward_lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_2/stack_1�
(backward_lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_2/stack_2�
 backward_lstm_11/strided_slice_2StridedSlicebackward_lstm_11/transpose:y:0/backward_lstm_11/strided_slice_2/stack:output:01backward_lstm_11/strided_slice_2/stack_1:output:01backward_lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2"
 backward_lstm_11/strided_slice_2�
3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp<backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype025
3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp�
$backward_lstm_11/lstm_cell_35/MatMulMatMul)backward_lstm_11/strided_slice_2:output:0;backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$backward_lstm_11/lstm_cell_35/MatMul�
5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp>backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype027
5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp�
&backward_lstm_11/lstm_cell_35/MatMul_1MatMulbackward_lstm_11/zeros:output:0=backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&backward_lstm_11/lstm_cell_35/MatMul_1�
!backward_lstm_11/lstm_cell_35/addAddV2.backward_lstm_11/lstm_cell_35/MatMul:product:00backward_lstm_11/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2#
!backward_lstm_11/lstm_cell_35/add�
4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp=backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype026
4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp�
%backward_lstm_11/lstm_cell_35/BiasAddBiasAdd%backward_lstm_11/lstm_cell_35/add:z:0<backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%backward_lstm_11/lstm_cell_35/BiasAdd�
-backward_lstm_11/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-backward_lstm_11/lstm_cell_35/split/split_dim�
#backward_lstm_11/lstm_cell_35/splitSplit6backward_lstm_11/lstm_cell_35/split/split_dim:output:0.backward_lstm_11/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2%
#backward_lstm_11/lstm_cell_35/split�
%backward_lstm_11/lstm_cell_35/SigmoidSigmoid,backward_lstm_11/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22'
%backward_lstm_11/lstm_cell_35/Sigmoid�
'backward_lstm_11/lstm_cell_35/Sigmoid_1Sigmoid,backward_lstm_11/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22)
'backward_lstm_11/lstm_cell_35/Sigmoid_1�
!backward_lstm_11/lstm_cell_35/mulMul+backward_lstm_11/lstm_cell_35/Sigmoid_1:y:0!backward_lstm_11/zeros_1:output:0*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/lstm_cell_35/mul�
"backward_lstm_11/lstm_cell_35/ReluRelu,backward_lstm_11/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22$
"backward_lstm_11/lstm_cell_35/Relu�
#backward_lstm_11/lstm_cell_35/mul_1Mul)backward_lstm_11/lstm_cell_35/Sigmoid:y:00backward_lstm_11/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_11/lstm_cell_35/mul_1�
#backward_lstm_11/lstm_cell_35/add_1AddV2%backward_lstm_11/lstm_cell_35/mul:z:0'backward_lstm_11/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_11/lstm_cell_35/add_1�
'backward_lstm_11/lstm_cell_35/Sigmoid_2Sigmoid,backward_lstm_11/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22)
'backward_lstm_11/lstm_cell_35/Sigmoid_2�
$backward_lstm_11/lstm_cell_35/Relu_1Relu'backward_lstm_11/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22&
$backward_lstm_11/lstm_cell_35/Relu_1�
#backward_lstm_11/lstm_cell_35/mul_2Mul+backward_lstm_11/lstm_cell_35/Sigmoid_2:y:02backward_lstm_11/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_11/lstm_cell_35/mul_2�
.backward_lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   20
.backward_lstm_11/TensorArrayV2_1/element_shape�
 backward_lstm_11/TensorArrayV2_1TensorListReserve7backward_lstm_11/TensorArrayV2_1/element_shape:output:0)backward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_11/TensorArrayV2_1p
backward_lstm_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_11/time�
&backward_lstm_11/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_lstm_11/Max/reduction_indices�
backward_lstm_11/MaxMaxbackward_lstm_11/Cast:y:0/backward_lstm_11/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/Maxr
backward_lstm_11/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_11/sub/y�
backward_lstm_11/subSubbackward_lstm_11/Max:output:0backward_lstm_11/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/sub�
backward_lstm_11/Sub_1Subbackward_lstm_11/sub:z:0backward_lstm_11/Cast:y:0*
T0*#
_output_shapes
:���������2
backward_lstm_11/Sub_1�
backward_lstm_11/zeros_like	ZerosLike'backward_lstm_11/lstm_cell_35/mul_2:z:0*
T0*'
_output_shapes
:���������22
backward_lstm_11/zeros_like�
)backward_lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2+
)backward_lstm_11/while/maximum_iterations�
#backward_lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_11/while/loop_counter�	
backward_lstm_11/whileWhile,backward_lstm_11/while/loop_counter:output:02backward_lstm_11/while/maximum_iterations:output:0backward_lstm_11/time:output:0)backward_lstm_11/TensorArrayV2_1:handle:0backward_lstm_11/zeros_like:y:0backward_lstm_11/zeros:output:0!backward_lstm_11/zeros_1:output:0)backward_lstm_11/strided_slice_1:output:0Hbackward_lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_11/Sub_1:z:0<backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource>backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource=backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( */
body'R%
#backward_lstm_11_while_body_1703023*/
cond'R%
#backward_lstm_11_while_cond_1703022*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
backward_lstm_11/while�
Abackward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2C
Abackward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape�
3backward_lstm_11/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_11/while:output:3Jbackward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype025
3backward_lstm_11/TensorArrayV2Stack/TensorListStack�
&backward_lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2(
&backward_lstm_11/strided_slice_3/stack�
(backward_lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_11/strided_slice_3/stack_1�
(backward_lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_3/stack_2�
 backward_lstm_11/strided_slice_3StridedSlice<backward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_11/strided_slice_3/stack:output:01backward_lstm_11/strided_slice_3/stack_1:output:01backward_lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2"
 backward_lstm_11/strided_slice_3�
!backward_lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_11/transpose_1/perm�
backward_lstm_11/transpose_1	Transpose<backward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_11/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
backward_lstm_11/transpose_1�
backward_lstm_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_11/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2(forward_lstm_11/strided_slice_3:output:0)backward_lstm_11/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity�
NoOpNoOp5^backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp4^backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp6^backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp^backward_lstm_11/while4^forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp3^forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp5^forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp^forward_lstm_11/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������: : : : : : 2l
4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp2j
3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp2n
5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp20
backward_lstm_11/whilebackward_lstm_11/while2j
3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp2h
2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp2l
4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp2.
forward_lstm_11/whileforward_lstm_11/while:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
#backward_lstm_11_while_cond_1703887>
:backward_lstm_11_while_backward_lstm_11_while_loop_counterD
@backward_lstm_11_while_backward_lstm_11_while_maximum_iterations&
"backward_lstm_11_while_placeholder(
$backward_lstm_11_while_placeholder_1(
$backward_lstm_11_while_placeholder_2(
$backward_lstm_11_while_placeholder_3@
<backward_lstm_11_while_less_backward_lstm_11_strided_slice_1W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1703887___redundant_placeholder0W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1703887___redundant_placeholder1W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1703887___redundant_placeholder2W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1703887___redundant_placeholder3#
backward_lstm_11_while_identity
�
backward_lstm_11/while/LessLess"backward_lstm_11_while_placeholder<backward_lstm_11_while_less_backward_lstm_11_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_11/while/Less�
backward_lstm_11/while/IdentityIdentitybackward_lstm_11/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_11/while/Identity"K
backward_lstm_11_while_identity(backward_lstm_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
�
�
J__inference_sequential_11_layer_call_and_return_conditional_losses_1703270

inputs
inputs_1	+
bidirectional_11_1703251:	�+
bidirectional_11_1703253:	2�'
bidirectional_11_1703255:	�+
bidirectional_11_1703257:	�+
bidirectional_11_1703259:	2�'
bidirectional_11_1703261:	�"
dense_11_1703264:d
dense_11_1703266:
identity��(bidirectional_11/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�
(bidirectional_11/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_11_1703251bidirectional_11_1703253bidirectional_11_1703255bidirectional_11_1703257bidirectional_11_1703259bidirectional_11_1703261*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_17031202*
(bidirectional_11/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_11/StatefulPartitionedCall:output:0dense_11_1703264dense_11_1703266*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_17027052"
 dense_11/StatefulPartitionedCall�
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp)^bidirectional_11/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:���������:���������: : : : : : : : 2T
(bidirectional_11/StatefulPartitionedCall(bidirectional_11/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_1704332

inputs
inputs_1	N
;forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource:	�P
=forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource:	2�K
<forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource:	�O
<backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource:	�Q
>backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource:	2�L
=backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource:	�
identity��4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp�3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp�5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp�backward_lstm_11/while�3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp�2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp�4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp�forward_lstm_11/while�
$forward_lstm_11/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$forward_lstm_11/RaggedToTensor/zeros�
$forward_lstm_11/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2&
$forward_lstm_11/RaggedToTensor/Const�
3forward_lstm_11/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-forward_lstm_11/RaggedToTensor/Const:output:0inputs-forward_lstm_11/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3forward_lstm_11/RaggedToTensor/RaggedTensorToTensor�
:forward_lstm_11/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack�
<forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1�
<forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2�
4forward_lstm_11/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cforward_lstm_11/RaggedNestedRowLengths/strided_slice/stack:output:0Eforward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1:output:0Eforward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask26
4forward_lstm_11/RaggedNestedRowLengths/strided_slice�
<forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack�
>forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2@
>forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1�
>forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2�
6forward_lstm_11/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Eforward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gforward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gforward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask28
6forward_lstm_11/RaggedNestedRowLengths/strided_slice_1�
*forward_lstm_11/RaggedNestedRowLengths/subSub=forward_lstm_11/RaggedNestedRowLengths/strided_slice:output:0?forward_lstm_11/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2,
*forward_lstm_11/RaggedNestedRowLengths/sub�
forward_lstm_11/CastCast.forward_lstm_11/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
forward_lstm_11/Cast�
forward_lstm_11/ShapeShape<forward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_11/Shape�
#forward_lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_11/strided_slice/stack�
%forward_lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_11/strided_slice/stack_1�
%forward_lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_11/strided_slice/stack_2�
forward_lstm_11/strided_sliceStridedSliceforward_lstm_11/Shape:output:0,forward_lstm_11/strided_slice/stack:output:0.forward_lstm_11/strided_slice/stack_1:output:0.forward_lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_11/strided_slice|
forward_lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_11/zeros/mul/y�
forward_lstm_11/zeros/mulMul&forward_lstm_11/strided_slice:output:0$forward_lstm_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros/mul
forward_lstm_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_11/zeros/Less/y�
forward_lstm_11/zeros/LessLessforward_lstm_11/zeros/mul:z:0%forward_lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros/Less�
forward_lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_11/zeros/packed/1�
forward_lstm_11/zeros/packedPack&forward_lstm_11/strided_slice:output:0'forward_lstm_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_11/zeros/packed�
forward_lstm_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_11/zeros/Const�
forward_lstm_11/zerosFill%forward_lstm_11/zeros/packed:output:0$forward_lstm_11/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_11/zeros�
forward_lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_11/zeros_1/mul/y�
forward_lstm_11/zeros_1/mulMul&forward_lstm_11/strided_slice:output:0&forward_lstm_11/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros_1/mul�
forward_lstm_11/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
forward_lstm_11/zeros_1/Less/y�
forward_lstm_11/zeros_1/LessLessforward_lstm_11/zeros_1/mul:z:0'forward_lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros_1/Less�
 forward_lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_11/zeros_1/packed/1�
forward_lstm_11/zeros_1/packedPack&forward_lstm_11/strided_slice:output:0)forward_lstm_11/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_11/zeros_1/packed�
forward_lstm_11/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_11/zeros_1/Const�
forward_lstm_11/zeros_1Fill'forward_lstm_11/zeros_1/packed:output:0&forward_lstm_11/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_11/zeros_1�
forward_lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_11/transpose/perm�
forward_lstm_11/transpose	Transpose<forward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0'forward_lstm_11/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
forward_lstm_11/transpose
forward_lstm_11/Shape_1Shapeforward_lstm_11/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_11/Shape_1�
%forward_lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_11/strided_slice_1/stack�
'forward_lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_1/stack_1�
'forward_lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_1/stack_2�
forward_lstm_11/strided_slice_1StridedSlice forward_lstm_11/Shape_1:output:0.forward_lstm_11/strided_slice_1/stack:output:00forward_lstm_11/strided_slice_1/stack_1:output:00forward_lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_11/strided_slice_1�
+forward_lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+forward_lstm_11/TensorArrayV2/element_shape�
forward_lstm_11/TensorArrayV2TensorListReserve4forward_lstm_11/TensorArrayV2/element_shape:output:0(forward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_11/TensorArrayV2�
Eforward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2G
Eforward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape�
7forward_lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_11/transpose:y:0Nforward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_11/TensorArrayUnstack/TensorListFromTensor�
%forward_lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_11/strided_slice_2/stack�
'forward_lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_2/stack_1�
'forward_lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_2/stack_2�
forward_lstm_11/strided_slice_2StridedSliceforward_lstm_11/transpose:y:0.forward_lstm_11/strided_slice_2/stack:output:00forward_lstm_11/strided_slice_2/stack_1:output:00forward_lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2!
forward_lstm_11/strided_slice_2�
2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp;forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp�
#forward_lstm_11/lstm_cell_34/MatMulMatMul(forward_lstm_11/strided_slice_2:output:0:forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#forward_lstm_11/lstm_cell_34/MatMul�
4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp=forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype026
4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp�
%forward_lstm_11/lstm_cell_34/MatMul_1MatMulforward_lstm_11/zeros:output:0<forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%forward_lstm_11/lstm_cell_34/MatMul_1�
 forward_lstm_11/lstm_cell_34/addAddV2-forward_lstm_11/lstm_cell_34/MatMul:product:0/forward_lstm_11/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2"
 forward_lstm_11/lstm_cell_34/add�
3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp<forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp�
$forward_lstm_11/lstm_cell_34/BiasAddBiasAdd$forward_lstm_11/lstm_cell_34/add:z:0;forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$forward_lstm_11/lstm_cell_34/BiasAdd�
,forward_lstm_11/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,forward_lstm_11/lstm_cell_34/split/split_dim�
"forward_lstm_11/lstm_cell_34/splitSplit5forward_lstm_11/lstm_cell_34/split/split_dim:output:0-forward_lstm_11/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2$
"forward_lstm_11/lstm_cell_34/split�
$forward_lstm_11/lstm_cell_34/SigmoidSigmoid+forward_lstm_11/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22&
$forward_lstm_11/lstm_cell_34/Sigmoid�
&forward_lstm_11/lstm_cell_34/Sigmoid_1Sigmoid+forward_lstm_11/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22(
&forward_lstm_11/lstm_cell_34/Sigmoid_1�
 forward_lstm_11/lstm_cell_34/mulMul*forward_lstm_11/lstm_cell_34/Sigmoid_1:y:0 forward_lstm_11/zeros_1:output:0*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/lstm_cell_34/mul�
!forward_lstm_11/lstm_cell_34/ReluRelu+forward_lstm_11/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22#
!forward_lstm_11/lstm_cell_34/Relu�
"forward_lstm_11/lstm_cell_34/mul_1Mul(forward_lstm_11/lstm_cell_34/Sigmoid:y:0/forward_lstm_11/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_11/lstm_cell_34/mul_1�
"forward_lstm_11/lstm_cell_34/add_1AddV2$forward_lstm_11/lstm_cell_34/mul:z:0&forward_lstm_11/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_11/lstm_cell_34/add_1�
&forward_lstm_11/lstm_cell_34/Sigmoid_2Sigmoid+forward_lstm_11/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22(
&forward_lstm_11/lstm_cell_34/Sigmoid_2�
#forward_lstm_11/lstm_cell_34/Relu_1Relu&forward_lstm_11/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22%
#forward_lstm_11/lstm_cell_34/Relu_1�
"forward_lstm_11/lstm_cell_34/mul_2Mul*forward_lstm_11/lstm_cell_34/Sigmoid_2:y:01forward_lstm_11/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_11/lstm_cell_34/mul_2�
-forward_lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2/
-forward_lstm_11/TensorArrayV2_1/element_shape�
forward_lstm_11/TensorArrayV2_1TensorListReserve6forward_lstm_11/TensorArrayV2_1/element_shape:output:0(forward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_11/TensorArrayV2_1n
forward_lstm_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_11/time�
forward_lstm_11/zeros_like	ZerosLike&forward_lstm_11/lstm_cell_34/mul_2:z:0*
T0*'
_output_shapes
:���������22
forward_lstm_11/zeros_like�
(forward_lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(forward_lstm_11/while/maximum_iterations�
"forward_lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_11/while/loop_counter�
forward_lstm_11/whileWhile+forward_lstm_11/while/loop_counter:output:01forward_lstm_11/while/maximum_iterations:output:0forward_lstm_11/time:output:0(forward_lstm_11/TensorArrayV2_1:handle:0forward_lstm_11/zeros_like:y:0forward_lstm_11/zeros:output:0 forward_lstm_11/zeros_1:output:0(forward_lstm_11/strided_slice_1:output:0Gforward_lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_11/Cast:y:0;forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource=forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource<forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *.
body&R$
"forward_lstm_11_while_body_1704056*.
cond&R$
"forward_lstm_11_while_cond_1704055*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
forward_lstm_11/while�
@forward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2B
@forward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape�
2forward_lstm_11/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_11/while:output:3Iforward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype024
2forward_lstm_11/TensorArrayV2Stack/TensorListStack�
%forward_lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2'
%forward_lstm_11/strided_slice_3/stack�
'forward_lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_11/strided_slice_3/stack_1�
'forward_lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_3/stack_2�
forward_lstm_11/strided_slice_3StridedSlice;forward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_11/strided_slice_3/stack:output:00forward_lstm_11/strided_slice_3/stack_1:output:00forward_lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2!
forward_lstm_11/strided_slice_3�
 forward_lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_11/transpose_1/perm�
forward_lstm_11/transpose_1	Transpose;forward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_11/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
forward_lstm_11/transpose_1�
forward_lstm_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_11/runtime�
%backward_lstm_11/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%backward_lstm_11/RaggedToTensor/zeros�
%backward_lstm_11/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2'
%backward_lstm_11/RaggedToTensor/Const�
4backward_lstm_11/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.backward_lstm_11/RaggedToTensor/Const:output:0inputs.backward_lstm_11/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4backward_lstm_11/RaggedToTensor/RaggedTensorToTensor�
;backward_lstm_11/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack�
=backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1�
=backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2�
5backward_lstm_11/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dbackward_lstm_11/RaggedNestedRowLengths/strided_slice/stack:output:0Fbackward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fbackward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask27
5backward_lstm_11/RaggedNestedRowLengths/strided_slice�
=backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack�
?backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2A
?backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1�
?backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2�
7backward_lstm_11/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fbackward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hbackward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hbackward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask29
7backward_lstm_11/RaggedNestedRowLengths/strided_slice_1�
+backward_lstm_11/RaggedNestedRowLengths/subSub>backward_lstm_11/RaggedNestedRowLengths/strided_slice:output:0@backward_lstm_11/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2-
+backward_lstm_11/RaggedNestedRowLengths/sub�
backward_lstm_11/CastCast/backward_lstm_11/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
backward_lstm_11/Cast�
backward_lstm_11/ShapeShape=backward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_11/Shape�
$backward_lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_11/strided_slice/stack�
&backward_lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_11/strided_slice/stack_1�
&backward_lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_11/strided_slice/stack_2�
backward_lstm_11/strided_sliceStridedSlicebackward_lstm_11/Shape:output:0-backward_lstm_11/strided_slice/stack:output:0/backward_lstm_11/strided_slice/stack_1:output:0/backward_lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_11/strided_slice~
backward_lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_11/zeros/mul/y�
backward_lstm_11/zeros/mulMul'backward_lstm_11/strided_slice:output:0%backward_lstm_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros/mul�
backward_lstm_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
backward_lstm_11/zeros/Less/y�
backward_lstm_11/zeros/LessLessbackward_lstm_11/zeros/mul:z:0&backward_lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros/Less�
backward_lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_11/zeros/packed/1�
backward_lstm_11/zeros/packedPack'backward_lstm_11/strided_slice:output:0(backward_lstm_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_11/zeros/packed�
backward_lstm_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_11/zeros/Const�
backward_lstm_11/zerosFill&backward_lstm_11/zeros/packed:output:0%backward_lstm_11/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_11/zeros�
backward_lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_11/zeros_1/mul/y�
backward_lstm_11/zeros_1/mulMul'backward_lstm_11/strided_slice:output:0'backward_lstm_11/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros_1/mul�
backward_lstm_11/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2!
backward_lstm_11/zeros_1/Less/y�
backward_lstm_11/zeros_1/LessLess backward_lstm_11/zeros_1/mul:z:0(backward_lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros_1/Less�
!backward_lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_11/zeros_1/packed/1�
backward_lstm_11/zeros_1/packedPack'backward_lstm_11/strided_slice:output:0*backward_lstm_11/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_11/zeros_1/packed�
backward_lstm_11/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_11/zeros_1/Const�
backward_lstm_11/zeros_1Fill(backward_lstm_11/zeros_1/packed:output:0'backward_lstm_11/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_11/zeros_1�
backward_lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_11/transpose/perm�
backward_lstm_11/transpose	Transpose=backward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0(backward_lstm_11/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_11/transpose�
backward_lstm_11/Shape_1Shapebackward_lstm_11/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_11/Shape_1�
&backward_lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_11/strided_slice_1/stack�
(backward_lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_1/stack_1�
(backward_lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_1/stack_2�
 backward_lstm_11/strided_slice_1StridedSlice!backward_lstm_11/Shape_1:output:0/backward_lstm_11/strided_slice_1/stack:output:01backward_lstm_11/strided_slice_1/stack_1:output:01backward_lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_11/strided_slice_1�
,backward_lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2.
,backward_lstm_11/TensorArrayV2/element_shape�
backward_lstm_11/TensorArrayV2TensorListReserve5backward_lstm_11/TensorArrayV2/element_shape:output:0)backward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_11/TensorArrayV2�
backward_lstm_11/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_11/ReverseV2/axis�
backward_lstm_11/ReverseV2	ReverseV2backward_lstm_11/transpose:y:0(backward_lstm_11/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_11/ReverseV2�
Fbackward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2H
Fbackward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape�
8backward_lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_11/ReverseV2:output:0Obackward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_11/TensorArrayUnstack/TensorListFromTensor�
&backward_lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_11/strided_slice_2/stack�
(backward_lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_2/stack_1�
(backward_lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_2/stack_2�
 backward_lstm_11/strided_slice_2StridedSlicebackward_lstm_11/transpose:y:0/backward_lstm_11/strided_slice_2/stack:output:01backward_lstm_11/strided_slice_2/stack_1:output:01backward_lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2"
 backward_lstm_11/strided_slice_2�
3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp<backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype025
3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp�
$backward_lstm_11/lstm_cell_35/MatMulMatMul)backward_lstm_11/strided_slice_2:output:0;backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$backward_lstm_11/lstm_cell_35/MatMul�
5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp>backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype027
5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp�
&backward_lstm_11/lstm_cell_35/MatMul_1MatMulbackward_lstm_11/zeros:output:0=backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&backward_lstm_11/lstm_cell_35/MatMul_1�
!backward_lstm_11/lstm_cell_35/addAddV2.backward_lstm_11/lstm_cell_35/MatMul:product:00backward_lstm_11/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2#
!backward_lstm_11/lstm_cell_35/add�
4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp=backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype026
4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp�
%backward_lstm_11/lstm_cell_35/BiasAddBiasAdd%backward_lstm_11/lstm_cell_35/add:z:0<backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%backward_lstm_11/lstm_cell_35/BiasAdd�
-backward_lstm_11/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-backward_lstm_11/lstm_cell_35/split/split_dim�
#backward_lstm_11/lstm_cell_35/splitSplit6backward_lstm_11/lstm_cell_35/split/split_dim:output:0.backward_lstm_11/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2%
#backward_lstm_11/lstm_cell_35/split�
%backward_lstm_11/lstm_cell_35/SigmoidSigmoid,backward_lstm_11/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22'
%backward_lstm_11/lstm_cell_35/Sigmoid�
'backward_lstm_11/lstm_cell_35/Sigmoid_1Sigmoid,backward_lstm_11/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22)
'backward_lstm_11/lstm_cell_35/Sigmoid_1�
!backward_lstm_11/lstm_cell_35/mulMul+backward_lstm_11/lstm_cell_35/Sigmoid_1:y:0!backward_lstm_11/zeros_1:output:0*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/lstm_cell_35/mul�
"backward_lstm_11/lstm_cell_35/ReluRelu,backward_lstm_11/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22$
"backward_lstm_11/lstm_cell_35/Relu�
#backward_lstm_11/lstm_cell_35/mul_1Mul)backward_lstm_11/lstm_cell_35/Sigmoid:y:00backward_lstm_11/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_11/lstm_cell_35/mul_1�
#backward_lstm_11/lstm_cell_35/add_1AddV2%backward_lstm_11/lstm_cell_35/mul:z:0'backward_lstm_11/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_11/lstm_cell_35/add_1�
'backward_lstm_11/lstm_cell_35/Sigmoid_2Sigmoid,backward_lstm_11/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22)
'backward_lstm_11/lstm_cell_35/Sigmoid_2�
$backward_lstm_11/lstm_cell_35/Relu_1Relu'backward_lstm_11/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22&
$backward_lstm_11/lstm_cell_35/Relu_1�
#backward_lstm_11/lstm_cell_35/mul_2Mul+backward_lstm_11/lstm_cell_35/Sigmoid_2:y:02backward_lstm_11/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_11/lstm_cell_35/mul_2�
.backward_lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   20
.backward_lstm_11/TensorArrayV2_1/element_shape�
 backward_lstm_11/TensorArrayV2_1TensorListReserve7backward_lstm_11/TensorArrayV2_1/element_shape:output:0)backward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_11/TensorArrayV2_1p
backward_lstm_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_11/time�
&backward_lstm_11/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_lstm_11/Max/reduction_indices�
backward_lstm_11/MaxMaxbackward_lstm_11/Cast:y:0/backward_lstm_11/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/Maxr
backward_lstm_11/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_11/sub/y�
backward_lstm_11/subSubbackward_lstm_11/Max:output:0backward_lstm_11/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/sub�
backward_lstm_11/Sub_1Subbackward_lstm_11/sub:z:0backward_lstm_11/Cast:y:0*
T0*#
_output_shapes
:���������2
backward_lstm_11/Sub_1�
backward_lstm_11/zeros_like	ZerosLike'backward_lstm_11/lstm_cell_35/mul_2:z:0*
T0*'
_output_shapes
:���������22
backward_lstm_11/zeros_like�
)backward_lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2+
)backward_lstm_11/while/maximum_iterations�
#backward_lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_11/while/loop_counter�	
backward_lstm_11/whileWhile,backward_lstm_11/while/loop_counter:output:02backward_lstm_11/while/maximum_iterations:output:0backward_lstm_11/time:output:0)backward_lstm_11/TensorArrayV2_1:handle:0backward_lstm_11/zeros_like:y:0backward_lstm_11/zeros:output:0!backward_lstm_11/zeros_1:output:0)backward_lstm_11/strided_slice_1:output:0Hbackward_lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_11/Sub_1:z:0<backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource>backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource=backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( */
body'R%
#backward_lstm_11_while_body_1704235*/
cond'R%
#backward_lstm_11_while_cond_1704234*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
backward_lstm_11/while�
Abackward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2C
Abackward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape�
3backward_lstm_11/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_11/while:output:3Jbackward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype025
3backward_lstm_11/TensorArrayV2Stack/TensorListStack�
&backward_lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2(
&backward_lstm_11/strided_slice_3/stack�
(backward_lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_11/strided_slice_3/stack_1�
(backward_lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_3/stack_2�
 backward_lstm_11/strided_slice_3StridedSlice<backward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_11/strided_slice_3/stack:output:01backward_lstm_11/strided_slice_3/stack_1:output:01backward_lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2"
 backward_lstm_11/strided_slice_3�
!backward_lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_11/transpose_1/perm�
backward_lstm_11/transpose_1	Transpose<backward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_11/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
backward_lstm_11/transpose_1�
backward_lstm_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_11/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2(forward_lstm_11/strided_slice_3:output:0)backward_lstm_11/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity�
NoOpNoOp5^backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp4^backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp6^backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp^backward_lstm_11/while4^forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp3^forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp5^forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp^forward_lstm_11/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������: : : : : : 2l
4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp2j
3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp2n
5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp20
backward_lstm_11/whilebackward_lstm_11/while2j
3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp2h
2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp2l
4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp2.
forward_lstm_11/whileforward_lstm_11/while:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_1704971
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1704971___redundant_placeholder05
1while_while_cond_1704971___redundant_placeholder15
1while_while_cond_1704971___redundant_placeholder25
1while_while_cond_1704971___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
��
�
"__inference__wrapped_model_1700244

args_0
args_0_1	m
Zsequential_11_bidirectional_11_forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource:	�o
\sequential_11_bidirectional_11_forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource:	2�j
[sequential_11_bidirectional_11_forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource:	�n
[sequential_11_bidirectional_11_backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource:	�p
]sequential_11_bidirectional_11_backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource:	2�k
\sequential_11_bidirectional_11_backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource:	�G
5sequential_11_dense_11_matmul_readvariableop_resource:dD
6sequential_11_dense_11_biasadd_readvariableop_resource:
identity��Ssequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp�Rsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp�Tsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp�5sequential_11/bidirectional_11/backward_lstm_11/while�Rsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp�Qsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp�Ssequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp�4sequential_11/bidirectional_11/forward_lstm_11/while�-sequential_11/dense_11/BiasAdd/ReadVariableOp�,sequential_11/dense_11/MatMul/ReadVariableOp�
Csequential_11/bidirectional_11/forward_lstm_11/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2E
Csequential_11/bidirectional_11/forward_lstm_11/RaggedToTensor/zeros�
Csequential_11/bidirectional_11/forward_lstm_11/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2E
Csequential_11/bidirectional_11/forward_lstm_11/RaggedToTensor/Const�
Rsequential_11/bidirectional_11/forward_lstm_11/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorLsequential_11/bidirectional_11/forward_lstm_11/RaggedToTensor/Const:output:0args_0Lsequential_11/bidirectional_11/forward_lstm_11/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2T
Rsequential_11/bidirectional_11/forward_lstm_11/RaggedToTensor/RaggedTensorToTensor�
Ysequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2[
Ysequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack�
[sequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2]
[sequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1�
[sequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2]
[sequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2�
Ssequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1bsequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack:output:0dsequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1:output:0dsequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask2U
Ssequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice�
[sequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2]
[sequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack�
]sequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2_
]sequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1�
]sequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2_
]sequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2�
Usequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1dsequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack:output:0fsequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0fsequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask2W
Usequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice_1�
Isequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/subSub\sequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice:output:0^sequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2K
Isequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/sub�
3sequential_11/bidirectional_11/forward_lstm_11/CastCastMsequential_11/bidirectional_11/forward_lstm_11/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������25
3sequential_11/bidirectional_11/forward_lstm_11/Cast�
4sequential_11/bidirectional_11/forward_lstm_11/ShapeShape[sequential_11/bidirectional_11/forward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:26
4sequential_11/bidirectional_11/forward_lstm_11/Shape�
Bsequential_11/bidirectional_11/forward_lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_11/bidirectional_11/forward_lstm_11/strided_slice/stack�
Dsequential_11/bidirectional_11/forward_lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_11/bidirectional_11/forward_lstm_11/strided_slice/stack_1�
Dsequential_11/bidirectional_11/forward_lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_11/bidirectional_11/forward_lstm_11/strided_slice/stack_2�
<sequential_11/bidirectional_11/forward_lstm_11/strided_sliceStridedSlice=sequential_11/bidirectional_11/forward_lstm_11/Shape:output:0Ksequential_11/bidirectional_11/forward_lstm_11/strided_slice/stack:output:0Msequential_11/bidirectional_11/forward_lstm_11/strided_slice/stack_1:output:0Msequential_11/bidirectional_11/forward_lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_11/bidirectional_11/forward_lstm_11/strided_slice�
:sequential_11/bidirectional_11/forward_lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22<
:sequential_11/bidirectional_11/forward_lstm_11/zeros/mul/y�
8sequential_11/bidirectional_11/forward_lstm_11/zeros/mulMulEsequential_11/bidirectional_11/forward_lstm_11/strided_slice:output:0Csequential_11/bidirectional_11/forward_lstm_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2:
8sequential_11/bidirectional_11/forward_lstm_11/zeros/mul�
;sequential_11/bidirectional_11/forward_lstm_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2=
;sequential_11/bidirectional_11/forward_lstm_11/zeros/Less/y�
9sequential_11/bidirectional_11/forward_lstm_11/zeros/LessLess<sequential_11/bidirectional_11/forward_lstm_11/zeros/mul:z:0Dsequential_11/bidirectional_11/forward_lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2;
9sequential_11/bidirectional_11/forward_lstm_11/zeros/Less�
=sequential_11/bidirectional_11/forward_lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22?
=sequential_11/bidirectional_11/forward_lstm_11/zeros/packed/1�
;sequential_11/bidirectional_11/forward_lstm_11/zeros/packedPackEsequential_11/bidirectional_11/forward_lstm_11/strided_slice:output:0Fsequential_11/bidirectional_11/forward_lstm_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2=
;sequential_11/bidirectional_11/forward_lstm_11/zeros/packed�
:sequential_11/bidirectional_11/forward_lstm_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2<
:sequential_11/bidirectional_11/forward_lstm_11/zeros/Const�
4sequential_11/bidirectional_11/forward_lstm_11/zerosFillDsequential_11/bidirectional_11/forward_lstm_11/zeros/packed:output:0Csequential_11/bidirectional_11/forward_lstm_11/zeros/Const:output:0*
T0*'
_output_shapes
:���������226
4sequential_11/bidirectional_11/forward_lstm_11/zeros�
<sequential_11/bidirectional_11/forward_lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22>
<sequential_11/bidirectional_11/forward_lstm_11/zeros_1/mul/y�
:sequential_11/bidirectional_11/forward_lstm_11/zeros_1/mulMulEsequential_11/bidirectional_11/forward_lstm_11/strided_slice:output:0Esequential_11/bidirectional_11/forward_lstm_11/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2<
:sequential_11/bidirectional_11/forward_lstm_11/zeros_1/mul�
=sequential_11/bidirectional_11/forward_lstm_11/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2?
=sequential_11/bidirectional_11/forward_lstm_11/zeros_1/Less/y�
;sequential_11/bidirectional_11/forward_lstm_11/zeros_1/LessLess>sequential_11/bidirectional_11/forward_lstm_11/zeros_1/mul:z:0Fsequential_11/bidirectional_11/forward_lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2=
;sequential_11/bidirectional_11/forward_lstm_11/zeros_1/Less�
?sequential_11/bidirectional_11/forward_lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22A
?sequential_11/bidirectional_11/forward_lstm_11/zeros_1/packed/1�
=sequential_11/bidirectional_11/forward_lstm_11/zeros_1/packedPackEsequential_11/bidirectional_11/forward_lstm_11/strided_slice:output:0Hsequential_11/bidirectional_11/forward_lstm_11/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2?
=sequential_11/bidirectional_11/forward_lstm_11/zeros_1/packed�
<sequential_11/bidirectional_11/forward_lstm_11/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2>
<sequential_11/bidirectional_11/forward_lstm_11/zeros_1/Const�
6sequential_11/bidirectional_11/forward_lstm_11/zeros_1FillFsequential_11/bidirectional_11/forward_lstm_11/zeros_1/packed:output:0Esequential_11/bidirectional_11/forward_lstm_11/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������228
6sequential_11/bidirectional_11/forward_lstm_11/zeros_1�
=sequential_11/bidirectional_11/forward_lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2?
=sequential_11/bidirectional_11/forward_lstm_11/transpose/perm�
8sequential_11/bidirectional_11/forward_lstm_11/transpose	Transpose[sequential_11/bidirectional_11/forward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0Fsequential_11/bidirectional_11/forward_lstm_11/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2:
8sequential_11/bidirectional_11/forward_lstm_11/transpose�
6sequential_11/bidirectional_11/forward_lstm_11/Shape_1Shape<sequential_11/bidirectional_11/forward_lstm_11/transpose:y:0*
T0*
_output_shapes
:28
6sequential_11/bidirectional_11/forward_lstm_11/Shape_1�
Dsequential_11/bidirectional_11/forward_lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dsequential_11/bidirectional_11/forward_lstm_11/strided_slice_1/stack�
Fsequential_11/bidirectional_11/forward_lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_11/bidirectional_11/forward_lstm_11/strided_slice_1/stack_1�
Fsequential_11/bidirectional_11/forward_lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_11/bidirectional_11/forward_lstm_11/strided_slice_1/stack_2�
>sequential_11/bidirectional_11/forward_lstm_11/strided_slice_1StridedSlice?sequential_11/bidirectional_11/forward_lstm_11/Shape_1:output:0Msequential_11/bidirectional_11/forward_lstm_11/strided_slice_1/stack:output:0Osequential_11/bidirectional_11/forward_lstm_11/strided_slice_1/stack_1:output:0Osequential_11/bidirectional_11/forward_lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2@
>sequential_11/bidirectional_11/forward_lstm_11/strided_slice_1�
Jsequential_11/bidirectional_11/forward_lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2L
Jsequential_11/bidirectional_11/forward_lstm_11/TensorArrayV2/element_shape�
<sequential_11/bidirectional_11/forward_lstm_11/TensorArrayV2TensorListReserveSsequential_11/bidirectional_11/forward_lstm_11/TensorArrayV2/element_shape:output:0Gsequential_11/bidirectional_11/forward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_11/bidirectional_11/forward_lstm_11/TensorArrayV2�
dsequential_11/bidirectional_11/forward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2f
dsequential_11/bidirectional_11/forward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape�
Vsequential_11/bidirectional_11/forward_lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor<sequential_11/bidirectional_11/forward_lstm_11/transpose:y:0msequential_11/bidirectional_11/forward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02X
Vsequential_11/bidirectional_11/forward_lstm_11/TensorArrayUnstack/TensorListFromTensor�
Dsequential_11/bidirectional_11/forward_lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dsequential_11/bidirectional_11/forward_lstm_11/strided_slice_2/stack�
Fsequential_11/bidirectional_11/forward_lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_11/bidirectional_11/forward_lstm_11/strided_slice_2/stack_1�
Fsequential_11/bidirectional_11/forward_lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_11/bidirectional_11/forward_lstm_11/strided_slice_2/stack_2�
>sequential_11/bidirectional_11/forward_lstm_11/strided_slice_2StridedSlice<sequential_11/bidirectional_11/forward_lstm_11/transpose:y:0Msequential_11/bidirectional_11/forward_lstm_11/strided_slice_2/stack:output:0Osequential_11/bidirectional_11/forward_lstm_11/strided_slice_2/stack_1:output:0Osequential_11/bidirectional_11/forward_lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2@
>sequential_11/bidirectional_11/forward_lstm_11/strided_slice_2�
Qsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOpReadVariableOpZsequential_11_bidirectional_11_forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02S
Qsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp�
Bsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMulMatMulGsequential_11/bidirectional_11/forward_lstm_11/strided_slice_2:output:0Ysequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2D
Bsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMul�
Ssequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp\sequential_11_bidirectional_11_forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02U
Ssequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp�
Dsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMul_1MatMul=sequential_11/bidirectional_11/forward_lstm_11/zeros:output:0[sequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2F
Dsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMul_1�
?sequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/addAddV2Lsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMul:product:0Nsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2A
?sequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/add�
Rsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp[sequential_11_bidirectional_11_forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02T
Rsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp�
Csequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/BiasAddBiasAddCsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/add:z:0Zsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2E
Csequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/BiasAdd�
Ksequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2M
Ksequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/split/split_dim�
Asequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/splitSplitTsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/split/split_dim:output:0Lsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2C
Asequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/split�
Csequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/SigmoidSigmoidJsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22E
Csequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/Sigmoid�
Esequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/Sigmoid_1SigmoidJsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22G
Esequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/Sigmoid_1�
?sequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/mulMulIsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/Sigmoid_1:y:0?sequential_11/bidirectional_11/forward_lstm_11/zeros_1:output:0*
T0*'
_output_shapes
:���������22A
?sequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/mul�
@sequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/ReluReluJsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22B
@sequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/Relu�
Asequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/mul_1MulGsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/Sigmoid:y:0Nsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22C
Asequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/mul_1�
Asequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/add_1AddV2Csequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/mul:z:0Esequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22C
Asequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/add_1�
Esequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/Sigmoid_2SigmoidJsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22G
Esequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/Sigmoid_2�
Bsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/Relu_1ReluEsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22D
Bsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/Relu_1�
Asequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/mul_2MulIsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/Sigmoid_2:y:0Psequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22C
Asequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/mul_2�
Lsequential_11/bidirectional_11/forward_lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2N
Lsequential_11/bidirectional_11/forward_lstm_11/TensorArrayV2_1/element_shape�
>sequential_11/bidirectional_11/forward_lstm_11/TensorArrayV2_1TensorListReserveUsequential_11/bidirectional_11/forward_lstm_11/TensorArrayV2_1/element_shape:output:0Gsequential_11/bidirectional_11/forward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02@
>sequential_11/bidirectional_11/forward_lstm_11/TensorArrayV2_1�
3sequential_11/bidirectional_11/forward_lstm_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 25
3sequential_11/bidirectional_11/forward_lstm_11/time�
9sequential_11/bidirectional_11/forward_lstm_11/zeros_like	ZerosLikeEsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/mul_2:z:0*
T0*'
_output_shapes
:���������22;
9sequential_11/bidirectional_11/forward_lstm_11/zeros_like�
Gsequential_11/bidirectional_11/forward_lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2I
Gsequential_11/bidirectional_11/forward_lstm_11/while/maximum_iterations�
Asequential_11/bidirectional_11/forward_lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_11/bidirectional_11/forward_lstm_11/while/loop_counter�
4sequential_11/bidirectional_11/forward_lstm_11/whileWhileJsequential_11/bidirectional_11/forward_lstm_11/while/loop_counter:output:0Psequential_11/bidirectional_11/forward_lstm_11/while/maximum_iterations:output:0<sequential_11/bidirectional_11/forward_lstm_11/time:output:0Gsequential_11/bidirectional_11/forward_lstm_11/TensorArrayV2_1:handle:0=sequential_11/bidirectional_11/forward_lstm_11/zeros_like:y:0=sequential_11/bidirectional_11/forward_lstm_11/zeros:output:0?sequential_11/bidirectional_11/forward_lstm_11/zeros_1:output:0Gsequential_11/bidirectional_11/forward_lstm_11/strided_slice_1:output:0fsequential_11/bidirectional_11/forward_lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:07sequential_11/bidirectional_11/forward_lstm_11/Cast:y:0Zsequential_11_bidirectional_11_forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource\sequential_11_bidirectional_11_forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource[sequential_11_bidirectional_11_forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *M
bodyERC
Asequential_11_bidirectional_11_forward_lstm_11_while_body_1699961*M
condERC
Asequential_11_bidirectional_11_forward_lstm_11_while_cond_1699960*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 26
4sequential_11/bidirectional_11/forward_lstm_11/while�
_sequential_11/bidirectional_11/forward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2a
_sequential_11/bidirectional_11/forward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape�
Qsequential_11/bidirectional_11/forward_lstm_11/TensorArrayV2Stack/TensorListStackTensorListStack=sequential_11/bidirectional_11/forward_lstm_11/while:output:3hsequential_11/bidirectional_11/forward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02S
Qsequential_11/bidirectional_11/forward_lstm_11/TensorArrayV2Stack/TensorListStack�
Dsequential_11/bidirectional_11/forward_lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2F
Dsequential_11/bidirectional_11/forward_lstm_11/strided_slice_3/stack�
Fsequential_11/bidirectional_11/forward_lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2H
Fsequential_11/bidirectional_11/forward_lstm_11/strided_slice_3/stack_1�
Fsequential_11/bidirectional_11/forward_lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_11/bidirectional_11/forward_lstm_11/strided_slice_3/stack_2�
>sequential_11/bidirectional_11/forward_lstm_11/strided_slice_3StridedSliceZsequential_11/bidirectional_11/forward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0Msequential_11/bidirectional_11/forward_lstm_11/strided_slice_3/stack:output:0Osequential_11/bidirectional_11/forward_lstm_11/strided_slice_3/stack_1:output:0Osequential_11/bidirectional_11/forward_lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2@
>sequential_11/bidirectional_11/forward_lstm_11/strided_slice_3�
?sequential_11/bidirectional_11/forward_lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2A
?sequential_11/bidirectional_11/forward_lstm_11/transpose_1/perm�
:sequential_11/bidirectional_11/forward_lstm_11/transpose_1	TransposeZsequential_11/bidirectional_11/forward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0Hsequential_11/bidirectional_11/forward_lstm_11/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22<
:sequential_11/bidirectional_11/forward_lstm_11/transpose_1�
6sequential_11/bidirectional_11/forward_lstm_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    28
6sequential_11/bidirectional_11/forward_lstm_11/runtime�
Dsequential_11/bidirectional_11/backward_lstm_11/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2F
Dsequential_11/bidirectional_11/backward_lstm_11/RaggedToTensor/zeros�
Dsequential_11/bidirectional_11/backward_lstm_11/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2F
Dsequential_11/bidirectional_11/backward_lstm_11/RaggedToTensor/Const�
Ssequential_11/bidirectional_11/backward_lstm_11/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorMsequential_11/bidirectional_11/backward_lstm_11/RaggedToTensor/Const:output:0args_0Msequential_11/bidirectional_11/backward_lstm_11/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2U
Ssequential_11/bidirectional_11/backward_lstm_11/RaggedToTensor/RaggedTensorToTensor�
Zsequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2\
Zsequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack�
\sequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2^
\sequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1�
\sequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2^
\sequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2�
Tsequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1csequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack:output:0esequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1:output:0esequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask2V
Tsequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice�
\sequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2^
\sequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack�
^sequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2`
^sequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1�
^sequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2`
^sequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2�
Vsequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1esequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack:output:0gsequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0gsequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask2X
Vsequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice_1�
Jsequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/subSub]sequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice:output:0_sequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2L
Jsequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/sub�
4sequential_11/bidirectional_11/backward_lstm_11/CastCastNsequential_11/bidirectional_11/backward_lstm_11/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������26
4sequential_11/bidirectional_11/backward_lstm_11/Cast�
5sequential_11/bidirectional_11/backward_lstm_11/ShapeShape\sequential_11/bidirectional_11/backward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:27
5sequential_11/bidirectional_11/backward_lstm_11/Shape�
Csequential_11/bidirectional_11/backward_lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Csequential_11/bidirectional_11/backward_lstm_11/strided_slice/stack�
Esequential_11/bidirectional_11/backward_lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_11/bidirectional_11/backward_lstm_11/strided_slice/stack_1�
Esequential_11/bidirectional_11/backward_lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_11/bidirectional_11/backward_lstm_11/strided_slice/stack_2�
=sequential_11/bidirectional_11/backward_lstm_11/strided_sliceStridedSlice>sequential_11/bidirectional_11/backward_lstm_11/Shape:output:0Lsequential_11/bidirectional_11/backward_lstm_11/strided_slice/stack:output:0Nsequential_11/bidirectional_11/backward_lstm_11/strided_slice/stack_1:output:0Nsequential_11/bidirectional_11/backward_lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=sequential_11/bidirectional_11/backward_lstm_11/strided_slice�
;sequential_11/bidirectional_11/backward_lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22=
;sequential_11/bidirectional_11/backward_lstm_11/zeros/mul/y�
9sequential_11/bidirectional_11/backward_lstm_11/zeros/mulMulFsequential_11/bidirectional_11/backward_lstm_11/strided_slice:output:0Dsequential_11/bidirectional_11/backward_lstm_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2;
9sequential_11/bidirectional_11/backward_lstm_11/zeros/mul�
<sequential_11/bidirectional_11/backward_lstm_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2>
<sequential_11/bidirectional_11/backward_lstm_11/zeros/Less/y�
:sequential_11/bidirectional_11/backward_lstm_11/zeros/LessLess=sequential_11/bidirectional_11/backward_lstm_11/zeros/mul:z:0Esequential_11/bidirectional_11/backward_lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2<
:sequential_11/bidirectional_11/backward_lstm_11/zeros/Less�
>sequential_11/bidirectional_11/backward_lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22@
>sequential_11/bidirectional_11/backward_lstm_11/zeros/packed/1�
<sequential_11/bidirectional_11/backward_lstm_11/zeros/packedPackFsequential_11/bidirectional_11/backward_lstm_11/strided_slice:output:0Gsequential_11/bidirectional_11/backward_lstm_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2>
<sequential_11/bidirectional_11/backward_lstm_11/zeros/packed�
;sequential_11/bidirectional_11/backward_lstm_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2=
;sequential_11/bidirectional_11/backward_lstm_11/zeros/Const�
5sequential_11/bidirectional_11/backward_lstm_11/zerosFillEsequential_11/bidirectional_11/backward_lstm_11/zeros/packed:output:0Dsequential_11/bidirectional_11/backward_lstm_11/zeros/Const:output:0*
T0*'
_output_shapes
:���������227
5sequential_11/bidirectional_11/backward_lstm_11/zeros�
=sequential_11/bidirectional_11/backward_lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22?
=sequential_11/bidirectional_11/backward_lstm_11/zeros_1/mul/y�
;sequential_11/bidirectional_11/backward_lstm_11/zeros_1/mulMulFsequential_11/bidirectional_11/backward_lstm_11/strided_slice:output:0Fsequential_11/bidirectional_11/backward_lstm_11/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2=
;sequential_11/bidirectional_11/backward_lstm_11/zeros_1/mul�
>sequential_11/bidirectional_11/backward_lstm_11/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2@
>sequential_11/bidirectional_11/backward_lstm_11/zeros_1/Less/y�
<sequential_11/bidirectional_11/backward_lstm_11/zeros_1/LessLess?sequential_11/bidirectional_11/backward_lstm_11/zeros_1/mul:z:0Gsequential_11/bidirectional_11/backward_lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2>
<sequential_11/bidirectional_11/backward_lstm_11/zeros_1/Less�
@sequential_11/bidirectional_11/backward_lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22B
@sequential_11/bidirectional_11/backward_lstm_11/zeros_1/packed/1�
>sequential_11/bidirectional_11/backward_lstm_11/zeros_1/packedPackFsequential_11/bidirectional_11/backward_lstm_11/strided_slice:output:0Isequential_11/bidirectional_11/backward_lstm_11/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2@
>sequential_11/bidirectional_11/backward_lstm_11/zeros_1/packed�
=sequential_11/bidirectional_11/backward_lstm_11/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2?
=sequential_11/bidirectional_11/backward_lstm_11/zeros_1/Const�
7sequential_11/bidirectional_11/backward_lstm_11/zeros_1FillGsequential_11/bidirectional_11/backward_lstm_11/zeros_1/packed:output:0Fsequential_11/bidirectional_11/backward_lstm_11/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������229
7sequential_11/bidirectional_11/backward_lstm_11/zeros_1�
>sequential_11/bidirectional_11/backward_lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2@
>sequential_11/bidirectional_11/backward_lstm_11/transpose/perm�
9sequential_11/bidirectional_11/backward_lstm_11/transpose	Transpose\sequential_11/bidirectional_11/backward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0Gsequential_11/bidirectional_11/backward_lstm_11/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2;
9sequential_11/bidirectional_11/backward_lstm_11/transpose�
7sequential_11/bidirectional_11/backward_lstm_11/Shape_1Shape=sequential_11/bidirectional_11/backward_lstm_11/transpose:y:0*
T0*
_output_shapes
:29
7sequential_11/bidirectional_11/backward_lstm_11/Shape_1�
Esequential_11/bidirectional_11/backward_lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_11/bidirectional_11/backward_lstm_11/strided_slice_1/stack�
Gsequential_11/bidirectional_11/backward_lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_11/bidirectional_11/backward_lstm_11/strided_slice_1/stack_1�
Gsequential_11/bidirectional_11/backward_lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_11/bidirectional_11/backward_lstm_11/strided_slice_1/stack_2�
?sequential_11/bidirectional_11/backward_lstm_11/strided_slice_1StridedSlice@sequential_11/bidirectional_11/backward_lstm_11/Shape_1:output:0Nsequential_11/bidirectional_11/backward_lstm_11/strided_slice_1/stack:output:0Psequential_11/bidirectional_11/backward_lstm_11/strided_slice_1/stack_1:output:0Psequential_11/bidirectional_11/backward_lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?sequential_11/bidirectional_11/backward_lstm_11/strided_slice_1�
Ksequential_11/bidirectional_11/backward_lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2M
Ksequential_11/bidirectional_11/backward_lstm_11/TensorArrayV2/element_shape�
=sequential_11/bidirectional_11/backward_lstm_11/TensorArrayV2TensorListReserveTsequential_11/bidirectional_11/backward_lstm_11/TensorArrayV2/element_shape:output:0Hsequential_11/bidirectional_11/backward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_11/bidirectional_11/backward_lstm_11/TensorArrayV2�
>sequential_11/bidirectional_11/backward_lstm_11/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_11/bidirectional_11/backward_lstm_11/ReverseV2/axis�
9sequential_11/bidirectional_11/backward_lstm_11/ReverseV2	ReverseV2=sequential_11/bidirectional_11/backward_lstm_11/transpose:y:0Gsequential_11/bidirectional_11/backward_lstm_11/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������2;
9sequential_11/bidirectional_11/backward_lstm_11/ReverseV2�
esequential_11/bidirectional_11/backward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2g
esequential_11/bidirectional_11/backward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape�
Wsequential_11/bidirectional_11/backward_lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorBsequential_11/bidirectional_11/backward_lstm_11/ReverseV2:output:0nsequential_11/bidirectional_11/backward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02Y
Wsequential_11/bidirectional_11/backward_lstm_11/TensorArrayUnstack/TensorListFromTensor�
Esequential_11/bidirectional_11/backward_lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_11/bidirectional_11/backward_lstm_11/strided_slice_2/stack�
Gsequential_11/bidirectional_11/backward_lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_11/bidirectional_11/backward_lstm_11/strided_slice_2/stack_1�
Gsequential_11/bidirectional_11/backward_lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_11/bidirectional_11/backward_lstm_11/strided_slice_2/stack_2�
?sequential_11/bidirectional_11/backward_lstm_11/strided_slice_2StridedSlice=sequential_11/bidirectional_11/backward_lstm_11/transpose:y:0Nsequential_11/bidirectional_11/backward_lstm_11/strided_slice_2/stack:output:0Psequential_11/bidirectional_11/backward_lstm_11/strided_slice_2/stack_1:output:0Psequential_11/bidirectional_11/backward_lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2A
?sequential_11/bidirectional_11/backward_lstm_11/strided_slice_2�
Rsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp[sequential_11_bidirectional_11_backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02T
Rsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp�
Csequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMulMatMulHsequential_11/bidirectional_11/backward_lstm_11/strided_slice_2:output:0Zsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2E
Csequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMul�
Tsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp]sequential_11_bidirectional_11_backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02V
Tsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp�
Esequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMul_1MatMul>sequential_11/bidirectional_11/backward_lstm_11/zeros:output:0\sequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2G
Esequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMul_1�
@sequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/addAddV2Msequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMul:product:0Osequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2B
@sequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/add�
Ssequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp\sequential_11_bidirectional_11_backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02U
Ssequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp�
Dsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/BiasAddBiasAddDsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/add:z:0[sequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2F
Dsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/BiasAdd�
Lsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2N
Lsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/split/split_dim�
Bsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/splitSplitUsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/split/split_dim:output:0Msequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2D
Bsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/split�
Dsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/SigmoidSigmoidKsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22F
Dsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/Sigmoid�
Fsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/Sigmoid_1SigmoidKsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22H
Fsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/Sigmoid_1�
@sequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/mulMulJsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/Sigmoid_1:y:0@sequential_11/bidirectional_11/backward_lstm_11/zeros_1:output:0*
T0*'
_output_shapes
:���������22B
@sequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/mul�
Asequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/ReluReluKsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22C
Asequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/Relu�
Bsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/mul_1MulHsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/Sigmoid:y:0Osequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22D
Bsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/mul_1�
Bsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/add_1AddV2Dsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/mul:z:0Fsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22D
Bsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/add_1�
Fsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/Sigmoid_2SigmoidKsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22H
Fsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/Sigmoid_2�
Csequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/Relu_1ReluFsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22E
Csequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/Relu_1�
Bsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/mul_2MulJsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/Sigmoid_2:y:0Qsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22D
Bsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/mul_2�
Msequential_11/bidirectional_11/backward_lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2O
Msequential_11/bidirectional_11/backward_lstm_11/TensorArrayV2_1/element_shape�
?sequential_11/bidirectional_11/backward_lstm_11/TensorArrayV2_1TensorListReserveVsequential_11/bidirectional_11/backward_lstm_11/TensorArrayV2_1/element_shape:output:0Hsequential_11/bidirectional_11/backward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?sequential_11/bidirectional_11/backward_lstm_11/TensorArrayV2_1�
4sequential_11/bidirectional_11/backward_lstm_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 26
4sequential_11/bidirectional_11/backward_lstm_11/time�
Esequential_11/bidirectional_11/backward_lstm_11/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2G
Esequential_11/bidirectional_11/backward_lstm_11/Max/reduction_indices�
3sequential_11/bidirectional_11/backward_lstm_11/MaxMax8sequential_11/bidirectional_11/backward_lstm_11/Cast:y:0Nsequential_11/bidirectional_11/backward_lstm_11/Max/reduction_indices:output:0*
T0*
_output_shapes
: 25
3sequential_11/bidirectional_11/backward_lstm_11/Max�
5sequential_11/bidirectional_11/backward_lstm_11/sub/yConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_11/bidirectional_11/backward_lstm_11/sub/y�
3sequential_11/bidirectional_11/backward_lstm_11/subSub<sequential_11/bidirectional_11/backward_lstm_11/Max:output:0>sequential_11/bidirectional_11/backward_lstm_11/sub/y:output:0*
T0*
_output_shapes
: 25
3sequential_11/bidirectional_11/backward_lstm_11/sub�
5sequential_11/bidirectional_11/backward_lstm_11/Sub_1Sub7sequential_11/bidirectional_11/backward_lstm_11/sub:z:08sequential_11/bidirectional_11/backward_lstm_11/Cast:y:0*
T0*#
_output_shapes
:���������27
5sequential_11/bidirectional_11/backward_lstm_11/Sub_1�
:sequential_11/bidirectional_11/backward_lstm_11/zeros_like	ZerosLikeFsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/mul_2:z:0*
T0*'
_output_shapes
:���������22<
:sequential_11/bidirectional_11/backward_lstm_11/zeros_like�
Hsequential_11/bidirectional_11/backward_lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2J
Hsequential_11/bidirectional_11/backward_lstm_11/while/maximum_iterations�
Bsequential_11/bidirectional_11/backward_lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bsequential_11/bidirectional_11/backward_lstm_11/while/loop_counter�
5sequential_11/bidirectional_11/backward_lstm_11/whileWhileKsequential_11/bidirectional_11/backward_lstm_11/while/loop_counter:output:0Qsequential_11/bidirectional_11/backward_lstm_11/while/maximum_iterations:output:0=sequential_11/bidirectional_11/backward_lstm_11/time:output:0Hsequential_11/bidirectional_11/backward_lstm_11/TensorArrayV2_1:handle:0>sequential_11/bidirectional_11/backward_lstm_11/zeros_like:y:0>sequential_11/bidirectional_11/backward_lstm_11/zeros:output:0@sequential_11/bidirectional_11/backward_lstm_11/zeros_1:output:0Hsequential_11/bidirectional_11/backward_lstm_11/strided_slice_1:output:0gsequential_11/bidirectional_11/backward_lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:09sequential_11/bidirectional_11/backward_lstm_11/Sub_1:z:0[sequential_11_bidirectional_11_backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource]sequential_11_bidirectional_11_backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource\sequential_11_bidirectional_11_backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *N
bodyFRD
Bsequential_11_bidirectional_11_backward_lstm_11_while_body_1700140*N
condFRD
Bsequential_11_bidirectional_11_backward_lstm_11_while_cond_1700139*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 27
5sequential_11/bidirectional_11/backward_lstm_11/while�
`sequential_11/bidirectional_11/backward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2b
`sequential_11/bidirectional_11/backward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape�
Rsequential_11/bidirectional_11/backward_lstm_11/TensorArrayV2Stack/TensorListStackTensorListStack>sequential_11/bidirectional_11/backward_lstm_11/while:output:3isequential_11/bidirectional_11/backward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02T
Rsequential_11/bidirectional_11/backward_lstm_11/TensorArrayV2Stack/TensorListStack�
Esequential_11/bidirectional_11/backward_lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2G
Esequential_11/bidirectional_11/backward_lstm_11/strided_slice_3/stack�
Gsequential_11/bidirectional_11/backward_lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2I
Gsequential_11/bidirectional_11/backward_lstm_11/strided_slice_3/stack_1�
Gsequential_11/bidirectional_11/backward_lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_11/bidirectional_11/backward_lstm_11/strided_slice_3/stack_2�
?sequential_11/bidirectional_11/backward_lstm_11/strided_slice_3StridedSlice[sequential_11/bidirectional_11/backward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0Nsequential_11/bidirectional_11/backward_lstm_11/strided_slice_3/stack:output:0Psequential_11/bidirectional_11/backward_lstm_11/strided_slice_3/stack_1:output:0Psequential_11/bidirectional_11/backward_lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2A
?sequential_11/bidirectional_11/backward_lstm_11/strided_slice_3�
@sequential_11/bidirectional_11/backward_lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2B
@sequential_11/bidirectional_11/backward_lstm_11/transpose_1/perm�
;sequential_11/bidirectional_11/backward_lstm_11/transpose_1	Transpose[sequential_11/bidirectional_11/backward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0Isequential_11/bidirectional_11/backward_lstm_11/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22=
;sequential_11/bidirectional_11/backward_lstm_11/transpose_1�
7sequential_11/bidirectional_11/backward_lstm_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    29
7sequential_11/bidirectional_11/backward_lstm_11/runtime�
*sequential_11/bidirectional_11/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2,
*sequential_11/bidirectional_11/concat/axis�
%sequential_11/bidirectional_11/concatConcatV2Gsequential_11/bidirectional_11/forward_lstm_11/strided_slice_3:output:0Hsequential_11/bidirectional_11/backward_lstm_11/strided_slice_3:output:03sequential_11/bidirectional_11/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������d2'
%sequential_11/bidirectional_11/concat�
,sequential_11/dense_11/MatMul/ReadVariableOpReadVariableOp5sequential_11_dense_11_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,sequential_11/dense_11/MatMul/ReadVariableOp�
sequential_11/dense_11/MatMulMatMul.sequential_11/bidirectional_11/concat:output:04sequential_11/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_11/dense_11/MatMul�
-sequential_11/dense_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_11/dense_11/BiasAdd/ReadVariableOp�
sequential_11/dense_11/BiasAddBiasAdd'sequential_11/dense_11/MatMul:product:05sequential_11/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
sequential_11/dense_11/BiasAdd�
sequential_11/dense_11/SigmoidSigmoid'sequential_11/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:���������2 
sequential_11/dense_11/Sigmoid}
IdentityIdentity"sequential_11/dense_11/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOpT^sequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOpS^sequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOpU^sequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp6^sequential_11/bidirectional_11/backward_lstm_11/whileS^sequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOpR^sequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOpT^sequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp5^sequential_11/bidirectional_11/forward_lstm_11/while.^sequential_11/dense_11/BiasAdd/ReadVariableOp-^sequential_11/dense_11/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:���������:���������: : : : : : : : 2�
Ssequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOpSsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp2�
Rsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOpRsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp2�
Tsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOpTsequential_11/bidirectional_11/backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp2n
5sequential_11/bidirectional_11/backward_lstm_11/while5sequential_11/bidirectional_11/backward_lstm_11/while2�
Rsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOpRsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp2�
Qsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOpQsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp2�
Ssequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOpSsequential_11/bidirectional_11/forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp2l
4sequential_11/bidirectional_11/forward_lstm_11/while4sequential_11/bidirectional_11/forward_lstm_11/while2^
-sequential_11/dense_11/BiasAdd/ReadVariableOp-sequential_11/dense_11/BiasAdd/ReadVariableOp2\
,sequential_11/dense_11/MatMul/ReadVariableOp,sequential_11/dense_11/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameargs_0:KG
#
_output_shapes
:���������
 
_user_specified_nameargs_0
�
�
E__inference_dense_11_layer_call_and_return_conditional_losses_1704710

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
while_cond_1705122
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1705122___redundant_placeholder05
1while_while_cond_1705122___redundant_placeholder15
1while_while_cond_1705122___redundant_placeholder25
1while_while_cond_1705122___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
�
�
.__inference_lstm_cell_34_layer_call_fn_1706031

inputs
states_0
states_1
unknown:	�
	unknown_0:	2�
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������2:���������2:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_lstm_cell_34_layer_call_and_return_conditional_losses_17003192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������22

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������2:���������2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������2
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������2
"
_user_specified_name
states/1
�V
�
"forward_lstm_11_while_body_1703437<
8forward_lstm_11_while_forward_lstm_11_while_loop_counterB
>forward_lstm_11_while_forward_lstm_11_while_maximum_iterations%
!forward_lstm_11_while_placeholder'
#forward_lstm_11_while_placeholder_1'
#forward_lstm_11_while_placeholder_2'
#forward_lstm_11_while_placeholder_3;
7forward_lstm_11_while_forward_lstm_11_strided_slice_1_0w
sforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0V
Cforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0:	�X
Eforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0:	2�S
Dforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0:	�"
forward_lstm_11_while_identity$
 forward_lstm_11_while_identity_1$
 forward_lstm_11_while_identity_2$
 forward_lstm_11_while_identity_3$
 forward_lstm_11_while_identity_4$
 forward_lstm_11_while_identity_59
5forward_lstm_11_while_forward_lstm_11_strided_slice_1u
qforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensorT
Aforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource:	�V
Cforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource:	2�Q
Bforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource:	���9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp�8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp�:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp�
Gforward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2I
Gforward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape�
9forward_lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_11_while_placeholderPforward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02;
9forward_lstm_11/while/TensorArrayV2Read/TensorListGetItem�
8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOpCforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02:
8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp�
)forward_lstm_11/while/lstm_cell_34/MatMulMatMul@forward_lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:0@forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)forward_lstm_11/while/lstm_cell_34/MatMul�
:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOpEforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02<
:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp�
+forward_lstm_11/while/lstm_cell_34/MatMul_1MatMul#forward_lstm_11_while_placeholder_2Bforward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+forward_lstm_11/while/lstm_cell_34/MatMul_1�
&forward_lstm_11/while/lstm_cell_34/addAddV23forward_lstm_11/while/lstm_cell_34/MatMul:product:05forward_lstm_11/while/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&forward_lstm_11/while/lstm_cell_34/add�
9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOpDforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02;
9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp�
*forward_lstm_11/while/lstm_cell_34/BiasAddBiasAdd*forward_lstm_11/while/lstm_cell_34/add:z:0Aforward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*forward_lstm_11/while/lstm_cell_34/BiasAdd�
2forward_lstm_11/while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2forward_lstm_11/while/lstm_cell_34/split/split_dim�
(forward_lstm_11/while/lstm_cell_34/splitSplit;forward_lstm_11/while/lstm_cell_34/split/split_dim:output:03forward_lstm_11/while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2*
(forward_lstm_11/while/lstm_cell_34/split�
*forward_lstm_11/while/lstm_cell_34/SigmoidSigmoid1forward_lstm_11/while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22,
*forward_lstm_11/while/lstm_cell_34/Sigmoid�
,forward_lstm_11/while/lstm_cell_34/Sigmoid_1Sigmoid1forward_lstm_11/while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22.
,forward_lstm_11/while/lstm_cell_34/Sigmoid_1�
&forward_lstm_11/while/lstm_cell_34/mulMul0forward_lstm_11/while/lstm_cell_34/Sigmoid_1:y:0#forward_lstm_11_while_placeholder_3*
T0*'
_output_shapes
:���������22(
&forward_lstm_11/while/lstm_cell_34/mul�
'forward_lstm_11/while/lstm_cell_34/ReluRelu1forward_lstm_11/while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22)
'forward_lstm_11/while/lstm_cell_34/Relu�
(forward_lstm_11/while/lstm_cell_34/mul_1Mul.forward_lstm_11/while/lstm_cell_34/Sigmoid:y:05forward_lstm_11/while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_11/while/lstm_cell_34/mul_1�
(forward_lstm_11/while/lstm_cell_34/add_1AddV2*forward_lstm_11/while/lstm_cell_34/mul:z:0,forward_lstm_11/while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_11/while/lstm_cell_34/add_1�
,forward_lstm_11/while/lstm_cell_34/Sigmoid_2Sigmoid1forward_lstm_11/while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22.
,forward_lstm_11/while/lstm_cell_34/Sigmoid_2�
)forward_lstm_11/while/lstm_cell_34/Relu_1Relu,forward_lstm_11/while/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22+
)forward_lstm_11/while/lstm_cell_34/Relu_1�
(forward_lstm_11/while/lstm_cell_34/mul_2Mul0forward_lstm_11/while/lstm_cell_34/Sigmoid_2:y:07forward_lstm_11/while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_11/while/lstm_cell_34/mul_2�
:forward_lstm_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_11_while_placeholder_1!forward_lstm_11_while_placeholder,forward_lstm_11/while/lstm_cell_34/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_11/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_11/while/add/y�
forward_lstm_11/while/addAddV2!forward_lstm_11_while_placeholder$forward_lstm_11/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/while/add�
forward_lstm_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_11/while/add_1/y�
forward_lstm_11/while/add_1AddV28forward_lstm_11_while_forward_lstm_11_while_loop_counter&forward_lstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/while/add_1�
forward_lstm_11/while/IdentityIdentityforward_lstm_11/while/add_1:z:0^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_11/while/Identity�
 forward_lstm_11/while/Identity_1Identity>forward_lstm_11_while_forward_lstm_11_while_maximum_iterations^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_11/while/Identity_1�
 forward_lstm_11/while/Identity_2Identityforward_lstm_11/while/add:z:0^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_11/while/Identity_2�
 forward_lstm_11/while/Identity_3IdentityJforward_lstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_11/while/Identity_3�
 forward_lstm_11/while/Identity_4Identity,forward_lstm_11/while/lstm_cell_34/mul_2:z:0^forward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/while/Identity_4�
 forward_lstm_11/while/Identity_5Identity,forward_lstm_11/while/lstm_cell_34/add_1:z:0^forward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/while/Identity_5�
forward_lstm_11/while/NoOpNoOp:^forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp9^forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp;^forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_11/while/NoOp"p
5forward_lstm_11_while_forward_lstm_11_strided_slice_17forward_lstm_11_while_forward_lstm_11_strided_slice_1_0"I
forward_lstm_11_while_identity'forward_lstm_11/while/Identity:output:0"M
 forward_lstm_11_while_identity_1)forward_lstm_11/while/Identity_1:output:0"M
 forward_lstm_11_while_identity_2)forward_lstm_11/while/Identity_2:output:0"M
 forward_lstm_11_while_identity_3)forward_lstm_11/while/Identity_3:output:0"M
 forward_lstm_11_while_identity_4)forward_lstm_11/while/Identity_4:output:0"M
 forward_lstm_11_while_identity_5)forward_lstm_11/while/Identity_5:output:0"�
Bforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resourceDforward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0"�
Cforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resourceEforward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0"�
Aforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resourceCforward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0"�
qforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensorsforward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2v
9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp9forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp2t
8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp8forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp2x
:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp:forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
�
�
#backward_lstm_11_while_cond_1703585>
:backward_lstm_11_while_backward_lstm_11_while_loop_counterD
@backward_lstm_11_while_backward_lstm_11_while_maximum_iterations&
"backward_lstm_11_while_placeholder(
$backward_lstm_11_while_placeholder_1(
$backward_lstm_11_while_placeholder_2(
$backward_lstm_11_while_placeholder_3@
<backward_lstm_11_while_less_backward_lstm_11_strided_slice_1W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1703585___redundant_placeholder0W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1703585___redundant_placeholder1W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1703585___redundant_placeholder2W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1703585___redundant_placeholder3#
backward_lstm_11_while_identity
�
backward_lstm_11/while/LessLess"backward_lstm_11_while_placeholder<backward_lstm_11_while_less_backward_lstm_11_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_11/while/Less�
backward_lstm_11/while/IdentityIdentitybackward_lstm_11/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_11/while/Identity"K
backward_lstm_11_while_identity(backward_lstm_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
�
�
J__inference_sequential_11_layer_call_and_return_conditional_losses_1703247

inputs
inputs_1	+
bidirectional_11_1703228:	�+
bidirectional_11_1703230:	2�'
bidirectional_11_1703232:	�+
bidirectional_11_1703234:	�+
bidirectional_11_1703236:	2�'
bidirectional_11_1703238:	�"
dense_11_1703241:d
dense_11_1703243:
identity��(bidirectional_11/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�
(bidirectional_11/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_11_1703228bidirectional_11_1703230bidirectional_11_1703232bidirectional_11_1703234bidirectional_11_1703236bidirectional_11_1703238*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_17026802*
(bidirectional_11/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_11/StatefulPartitionedCall:output:0dense_11_1703241dense_11_1703243*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_17027052"
 dense_11/StatefulPartitionedCall�
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp)^bidirectional_11/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:���������:���������: : : : : : : : 2T
(bidirectional_11/StatefulPartitionedCall(bidirectional_11/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
while_body_1700965
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_35_1700989_0:	�/
while_lstm_cell_35_1700991_0:	2�+
while_lstm_cell_35_1700993_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_35_1700989:	�-
while_lstm_cell_35_1700991:	2�)
while_lstm_cell_35_1700993:	���*while/lstm_cell_35/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
*while/lstm_cell_35/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_35_1700989_0while_lstm_cell_35_1700991_0while_lstm_cell_35_1700993_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������2:���������2:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_lstm_cell_35_layer_call_and_return_conditional_losses_17009512,
*while/lstm_cell_35/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_35/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity3while/lstm_cell_35/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_35/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_35_1700989while_lstm_cell_35_1700989_0":
while_lstm_cell_35_1700991while_lstm_cell_35_1700991_0":
while_lstm_cell_35_1700993while_lstm_cell_35_1700993_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2X
*while/lstm_cell_35/StatefulPartitionedCall*while/lstm_cell_35/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
�\
�
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_1701669

inputs>
+lstm_cell_34_matmul_readvariableop_resource:	�@
-lstm_cell_34_matmul_1_readvariableop_resource:	2�;
,lstm_cell_34_biasadd_readvariableop_resource:	�
identity��#lstm_cell_34/BiasAdd/ReadVariableOp�"lstm_cell_34/MatMul/ReadVariableOp�$lstm_cell_34/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packedc
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedg
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_34/MatMul/ReadVariableOpReadVariableOp+lstm_cell_34_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_34/MatMul/ReadVariableOp�
lstm_cell_34/MatMulMatMulstrided_slice_2:output:0*lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/MatMul�
$lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_34/MatMul_1/ReadVariableOp�
lstm_cell_34/MatMul_1MatMulzeros:output:0,lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/MatMul_1�
lstm_cell_34/addAddV2lstm_cell_34/MatMul:product:0lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/add�
#lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_34/BiasAdd/ReadVariableOp�
lstm_cell_34/BiasAddBiasAddlstm_cell_34/add:z:0+lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/BiasAdd~
lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_34/split/split_dim�
lstm_cell_34/splitSplit%lstm_cell_34/split/split_dim:output:0lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_34/split�
lstm_cell_34/SigmoidSigmoidlstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/Sigmoid�
lstm_cell_34/Sigmoid_1Sigmoidlstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_34/Sigmoid_1�
lstm_cell_34/mulMullstm_cell_34/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/mul}
lstm_cell_34/ReluRelulstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_34/Relu�
lstm_cell_34/mul_1Mullstm_cell_34/Sigmoid:y:0lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/mul_1�
lstm_cell_34/add_1AddV2lstm_cell_34/mul:z:0lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/add_1�
lstm_cell_34/Sigmoid_2Sigmoidlstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_34/Sigmoid_2|
lstm_cell_34/Relu_1Relulstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/Relu_1�
lstm_cell_34/mul_2Mullstm_cell_34/Sigmoid_2:y:0!lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_34_matmul_readvariableop_resource-lstm_cell_34_matmul_1_readvariableop_resource,lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1701585*
condR
while_cond_1701584*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity�
NoOpNoOp$^lstm_cell_34/BiasAdd/ReadVariableOp#^lstm_cell_34/MatMul/ReadVariableOp%^lstm_cell_34/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_34/BiasAdd/ReadVariableOp#lstm_cell_34/BiasAdd/ReadVariableOp2H
"lstm_cell_34/MatMul/ReadVariableOp"lstm_cell_34/MatMul/ReadVariableOp2L
$lstm_cell_34/MatMul_1/ReadVariableOp$lstm_cell_34/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�b
�
 __inference__traced_save_1706351
file_prefix.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopS
Osavev2_bidirectional_11_forward_lstm_11_lstm_cell_34_kernel_read_readvariableop]
Ysavev2_bidirectional_11_forward_lstm_11_lstm_cell_34_recurrent_kernel_read_readvariableopQ
Msavev2_bidirectional_11_forward_lstm_11_lstm_cell_34_bias_read_readvariableopT
Psavev2_bidirectional_11_backward_lstm_11_lstm_cell_35_kernel_read_readvariableop^
Zsavev2_bidirectional_11_backward_lstm_11_lstm_cell_35_recurrent_kernel_read_readvariableopR
Nsavev2_bidirectional_11_backward_lstm_11_lstm_cell_35_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_11_kernel_m_read_readvariableop3
/savev2_adam_dense_11_bias_m_read_readvariableopZ
Vsavev2_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_kernel_m_read_readvariableopd
`savev2_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_recurrent_kernel_m_read_readvariableopX
Tsavev2_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_bias_m_read_readvariableop[
Wsavev2_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_kernel_m_read_readvariableope
asavev2_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_recurrent_kernel_m_read_readvariableopY
Usavev2_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_bias_m_read_readvariableop5
1savev2_adam_dense_11_kernel_v_read_readvariableop3
/savev2_adam_dense_11_bias_v_read_readvariableopZ
Vsavev2_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_kernel_v_read_readvariableopd
`savev2_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_recurrent_kernel_v_read_readvariableopX
Tsavev2_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_bias_v_read_readvariableop[
Wsavev2_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_kernel_v_read_readvariableope
asavev2_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_recurrent_kernel_v_read_readvariableopY
Usavev2_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_bias_v_read_readvariableop8
4savev2_adam_dense_11_kernel_vhat_read_readvariableop6
2savev2_adam_dense_11_bias_vhat_read_readvariableop]
Ysavev2_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_kernel_vhat_read_readvariableopg
csavev2_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_recurrent_kernel_vhat_read_readvariableop[
Wsavev2_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_bias_vhat_read_readvariableop^
Zsavev2_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_kernel_vhat_read_readvariableoph
dsavev2_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_recurrent_kernel_vhat_read_readvariableop\
Xsavev2_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_bias_vhat_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*�
value�B�(B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopOsavev2_bidirectional_11_forward_lstm_11_lstm_cell_34_kernel_read_readvariableopYsavev2_bidirectional_11_forward_lstm_11_lstm_cell_34_recurrent_kernel_read_readvariableopMsavev2_bidirectional_11_forward_lstm_11_lstm_cell_34_bias_read_readvariableopPsavev2_bidirectional_11_backward_lstm_11_lstm_cell_35_kernel_read_readvariableopZsavev2_bidirectional_11_backward_lstm_11_lstm_cell_35_recurrent_kernel_read_readvariableopNsavev2_bidirectional_11_backward_lstm_11_lstm_cell_35_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_11_kernel_m_read_readvariableop/savev2_adam_dense_11_bias_m_read_readvariableopVsavev2_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_kernel_m_read_readvariableop`savev2_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_recurrent_kernel_m_read_readvariableopTsavev2_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_bias_m_read_readvariableopWsavev2_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_kernel_m_read_readvariableopasavev2_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_recurrent_kernel_m_read_readvariableopUsavev2_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_bias_m_read_readvariableop1savev2_adam_dense_11_kernel_v_read_readvariableop/savev2_adam_dense_11_bias_v_read_readvariableopVsavev2_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_kernel_v_read_readvariableop`savev2_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_recurrent_kernel_v_read_readvariableopTsavev2_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_bias_v_read_readvariableopWsavev2_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_kernel_v_read_readvariableopasavev2_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_recurrent_kernel_v_read_readvariableopUsavev2_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_bias_v_read_readvariableop4savev2_adam_dense_11_kernel_vhat_read_readvariableop2savev2_adam_dense_11_bias_vhat_read_readvariableopYsavev2_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_kernel_vhat_read_readvariableopcsavev2_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_recurrent_kernel_vhat_read_readvariableopWsavev2_adam_bidirectional_11_forward_lstm_11_lstm_cell_34_bias_vhat_read_readvariableopZsavev2_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_kernel_vhat_read_readvariableopdsavev2_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_recurrent_kernel_vhat_read_readvariableopXsavev2_adam_bidirectional_11_backward_lstm_11_lstm_cell_35_bias_vhat_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :d:: : : : : :	�:	2�:�:	�:	2�:�: : :d::	�:	2�:�:	�:	2�:�:d::	�:	2�:�:	�:	2�:�:d::	�:	2�:�:	�:	2�:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�:%	!

_output_shapes
:	2�:!


_output_shapes	
:�:%!

_output_shapes
:	�:%!

_output_shapes
:	2�:!

_output_shapes	
:�:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	2�:!

_output_shapes	
:�:%!

_output_shapes
:	�:%!

_output_shapes
:	2�:!

_output_shapes	
:�:$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	2�:!

_output_shapes	
:�:%!

_output_shapes
:	�:%!

_output_shapes
:	2�:!

_output_shapes	
:�:$  

_output_shapes

:d: !

_output_shapes
::%"!

_output_shapes
:	�:%#!

_output_shapes
:	2�:!$

_output_shapes	
:�:%%!

_output_shapes
:	�:%&!

_output_shapes
:	2�:!'

_output_shapes	
:�:(

_output_shapes
: 
�
�
"forward_lstm_11_while_cond_1704055<
8forward_lstm_11_while_forward_lstm_11_while_loop_counterB
>forward_lstm_11_while_forward_lstm_11_while_maximum_iterations%
!forward_lstm_11_while_placeholder'
#forward_lstm_11_while_placeholder_1'
#forward_lstm_11_while_placeholder_2'
#forward_lstm_11_while_placeholder_3'
#forward_lstm_11_while_placeholder_4>
:forward_lstm_11_while_less_forward_lstm_11_strided_slice_1U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1704055___redundant_placeholder0U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1704055___redundant_placeholder1U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1704055___redundant_placeholder2U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1704055___redundant_placeholder3U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1704055___redundant_placeholder4"
forward_lstm_11_while_identity
�
forward_lstm_11/while/LessLess!forward_lstm_11_while_placeholder:forward_lstm_11_while_less_forward_lstm_11_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_11/while/Less�
forward_lstm_11/while/IdentityIdentityforward_lstm_11/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_11/while/Identity"I
forward_lstm_11_while_identity'forward_lstm_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :���������2:���������2:���������2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
��
�
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_1704690

inputs
inputs_1	N
;forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource:	�P
=forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource:	2�K
<forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource:	�O
<backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource:	�Q
>backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource:	2�L
=backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource:	�
identity��4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp�3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp�5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp�backward_lstm_11/while�3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp�2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp�4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp�forward_lstm_11/while�
$forward_lstm_11/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$forward_lstm_11/RaggedToTensor/zeros�
$forward_lstm_11/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2&
$forward_lstm_11/RaggedToTensor/Const�
3forward_lstm_11/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-forward_lstm_11/RaggedToTensor/Const:output:0inputs-forward_lstm_11/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3forward_lstm_11/RaggedToTensor/RaggedTensorToTensor�
:forward_lstm_11/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack�
<forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1�
<forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<forward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2�
4forward_lstm_11/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cforward_lstm_11/RaggedNestedRowLengths/strided_slice/stack:output:0Eforward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1:output:0Eforward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask26
4forward_lstm_11/RaggedNestedRowLengths/strided_slice�
<forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack�
>forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2@
>forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1�
>forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>forward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2�
6forward_lstm_11/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Eforward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gforward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gforward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask28
6forward_lstm_11/RaggedNestedRowLengths/strided_slice_1�
*forward_lstm_11/RaggedNestedRowLengths/subSub=forward_lstm_11/RaggedNestedRowLengths/strided_slice:output:0?forward_lstm_11/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2,
*forward_lstm_11/RaggedNestedRowLengths/sub�
forward_lstm_11/CastCast.forward_lstm_11/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
forward_lstm_11/Cast�
forward_lstm_11/ShapeShape<forward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_11/Shape�
#forward_lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_11/strided_slice/stack�
%forward_lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_11/strided_slice/stack_1�
%forward_lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_11/strided_slice/stack_2�
forward_lstm_11/strided_sliceStridedSliceforward_lstm_11/Shape:output:0,forward_lstm_11/strided_slice/stack:output:0.forward_lstm_11/strided_slice/stack_1:output:0.forward_lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_11/strided_slice|
forward_lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_11/zeros/mul/y�
forward_lstm_11/zeros/mulMul&forward_lstm_11/strided_slice:output:0$forward_lstm_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros/mul
forward_lstm_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_11/zeros/Less/y�
forward_lstm_11/zeros/LessLessforward_lstm_11/zeros/mul:z:0%forward_lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros/Less�
forward_lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_11/zeros/packed/1�
forward_lstm_11/zeros/packedPack&forward_lstm_11/strided_slice:output:0'forward_lstm_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_11/zeros/packed�
forward_lstm_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_11/zeros/Const�
forward_lstm_11/zerosFill%forward_lstm_11/zeros/packed:output:0$forward_lstm_11/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_11/zeros�
forward_lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_11/zeros_1/mul/y�
forward_lstm_11/zeros_1/mulMul&forward_lstm_11/strided_slice:output:0&forward_lstm_11/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros_1/mul�
forward_lstm_11/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
forward_lstm_11/zeros_1/Less/y�
forward_lstm_11/zeros_1/LessLessforward_lstm_11/zeros_1/mul:z:0'forward_lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_11/zeros_1/Less�
 forward_lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_11/zeros_1/packed/1�
forward_lstm_11/zeros_1/packedPack&forward_lstm_11/strided_slice:output:0)forward_lstm_11/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_11/zeros_1/packed�
forward_lstm_11/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_11/zeros_1/Const�
forward_lstm_11/zeros_1Fill'forward_lstm_11/zeros_1/packed:output:0&forward_lstm_11/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_11/zeros_1�
forward_lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_11/transpose/perm�
forward_lstm_11/transpose	Transpose<forward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0'forward_lstm_11/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
forward_lstm_11/transpose
forward_lstm_11/Shape_1Shapeforward_lstm_11/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_11/Shape_1�
%forward_lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_11/strided_slice_1/stack�
'forward_lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_1/stack_1�
'forward_lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_1/stack_2�
forward_lstm_11/strided_slice_1StridedSlice forward_lstm_11/Shape_1:output:0.forward_lstm_11/strided_slice_1/stack:output:00forward_lstm_11/strided_slice_1/stack_1:output:00forward_lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_11/strided_slice_1�
+forward_lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+forward_lstm_11/TensorArrayV2/element_shape�
forward_lstm_11/TensorArrayV2TensorListReserve4forward_lstm_11/TensorArrayV2/element_shape:output:0(forward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_11/TensorArrayV2�
Eforward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2G
Eforward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape�
7forward_lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_11/transpose:y:0Nforward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_11/TensorArrayUnstack/TensorListFromTensor�
%forward_lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_11/strided_slice_2/stack�
'forward_lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_2/stack_1�
'forward_lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_2/stack_2�
forward_lstm_11/strided_slice_2StridedSliceforward_lstm_11/transpose:y:0.forward_lstm_11/strided_slice_2/stack:output:00forward_lstm_11/strided_slice_2/stack_1:output:00forward_lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2!
forward_lstm_11/strided_slice_2�
2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp;forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp�
#forward_lstm_11/lstm_cell_34/MatMulMatMul(forward_lstm_11/strided_slice_2:output:0:forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#forward_lstm_11/lstm_cell_34/MatMul�
4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp=forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype026
4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp�
%forward_lstm_11/lstm_cell_34/MatMul_1MatMulforward_lstm_11/zeros:output:0<forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%forward_lstm_11/lstm_cell_34/MatMul_1�
 forward_lstm_11/lstm_cell_34/addAddV2-forward_lstm_11/lstm_cell_34/MatMul:product:0/forward_lstm_11/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2"
 forward_lstm_11/lstm_cell_34/add�
3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp<forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp�
$forward_lstm_11/lstm_cell_34/BiasAddBiasAdd$forward_lstm_11/lstm_cell_34/add:z:0;forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$forward_lstm_11/lstm_cell_34/BiasAdd�
,forward_lstm_11/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,forward_lstm_11/lstm_cell_34/split/split_dim�
"forward_lstm_11/lstm_cell_34/splitSplit5forward_lstm_11/lstm_cell_34/split/split_dim:output:0-forward_lstm_11/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2$
"forward_lstm_11/lstm_cell_34/split�
$forward_lstm_11/lstm_cell_34/SigmoidSigmoid+forward_lstm_11/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22&
$forward_lstm_11/lstm_cell_34/Sigmoid�
&forward_lstm_11/lstm_cell_34/Sigmoid_1Sigmoid+forward_lstm_11/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22(
&forward_lstm_11/lstm_cell_34/Sigmoid_1�
 forward_lstm_11/lstm_cell_34/mulMul*forward_lstm_11/lstm_cell_34/Sigmoid_1:y:0 forward_lstm_11/zeros_1:output:0*
T0*'
_output_shapes
:���������22"
 forward_lstm_11/lstm_cell_34/mul�
!forward_lstm_11/lstm_cell_34/ReluRelu+forward_lstm_11/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22#
!forward_lstm_11/lstm_cell_34/Relu�
"forward_lstm_11/lstm_cell_34/mul_1Mul(forward_lstm_11/lstm_cell_34/Sigmoid:y:0/forward_lstm_11/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_11/lstm_cell_34/mul_1�
"forward_lstm_11/lstm_cell_34/add_1AddV2$forward_lstm_11/lstm_cell_34/mul:z:0&forward_lstm_11/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_11/lstm_cell_34/add_1�
&forward_lstm_11/lstm_cell_34/Sigmoid_2Sigmoid+forward_lstm_11/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22(
&forward_lstm_11/lstm_cell_34/Sigmoid_2�
#forward_lstm_11/lstm_cell_34/Relu_1Relu&forward_lstm_11/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22%
#forward_lstm_11/lstm_cell_34/Relu_1�
"forward_lstm_11/lstm_cell_34/mul_2Mul*forward_lstm_11/lstm_cell_34/Sigmoid_2:y:01forward_lstm_11/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_11/lstm_cell_34/mul_2�
-forward_lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2/
-forward_lstm_11/TensorArrayV2_1/element_shape�
forward_lstm_11/TensorArrayV2_1TensorListReserve6forward_lstm_11/TensorArrayV2_1/element_shape:output:0(forward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_11/TensorArrayV2_1n
forward_lstm_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_11/time�
forward_lstm_11/zeros_like	ZerosLike&forward_lstm_11/lstm_cell_34/mul_2:z:0*
T0*'
_output_shapes
:���������22
forward_lstm_11/zeros_like�
(forward_lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(forward_lstm_11/while/maximum_iterations�
"forward_lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_11/while/loop_counter�
forward_lstm_11/whileWhile+forward_lstm_11/while/loop_counter:output:01forward_lstm_11/while/maximum_iterations:output:0forward_lstm_11/time:output:0(forward_lstm_11/TensorArrayV2_1:handle:0forward_lstm_11/zeros_like:y:0forward_lstm_11/zeros:output:0 forward_lstm_11/zeros_1:output:0(forward_lstm_11/strided_slice_1:output:0Gforward_lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_11/Cast:y:0;forward_lstm_11_lstm_cell_34_matmul_readvariableop_resource=forward_lstm_11_lstm_cell_34_matmul_1_readvariableop_resource<forward_lstm_11_lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *.
body&R$
"forward_lstm_11_while_body_1704414*.
cond&R$
"forward_lstm_11_while_cond_1704413*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
forward_lstm_11/while�
@forward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2B
@forward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape�
2forward_lstm_11/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_11/while:output:3Iforward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype024
2forward_lstm_11/TensorArrayV2Stack/TensorListStack�
%forward_lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2'
%forward_lstm_11/strided_slice_3/stack�
'forward_lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_11/strided_slice_3/stack_1�
'forward_lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_11/strided_slice_3/stack_2�
forward_lstm_11/strided_slice_3StridedSlice;forward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_11/strided_slice_3/stack:output:00forward_lstm_11/strided_slice_3/stack_1:output:00forward_lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2!
forward_lstm_11/strided_slice_3�
 forward_lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_11/transpose_1/perm�
forward_lstm_11/transpose_1	Transpose;forward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_11/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
forward_lstm_11/transpose_1�
forward_lstm_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_11/runtime�
%backward_lstm_11/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%backward_lstm_11/RaggedToTensor/zeros�
%backward_lstm_11/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2'
%backward_lstm_11/RaggedToTensor/Const�
4backward_lstm_11/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.backward_lstm_11/RaggedToTensor/Const:output:0inputs.backward_lstm_11/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4backward_lstm_11/RaggedToTensor/RaggedTensorToTensor�
;backward_lstm_11/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack�
=backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1�
=backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=backward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2�
5backward_lstm_11/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dbackward_lstm_11/RaggedNestedRowLengths/strided_slice/stack:output:0Fbackward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fbackward_lstm_11/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask27
5backward_lstm_11/RaggedNestedRowLengths/strided_slice�
=backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack�
?backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2A
?backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1�
?backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?backward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2�
7backward_lstm_11/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fbackward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hbackward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hbackward_lstm_11/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask29
7backward_lstm_11/RaggedNestedRowLengths/strided_slice_1�
+backward_lstm_11/RaggedNestedRowLengths/subSub>backward_lstm_11/RaggedNestedRowLengths/strided_slice:output:0@backward_lstm_11/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2-
+backward_lstm_11/RaggedNestedRowLengths/sub�
backward_lstm_11/CastCast/backward_lstm_11/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
backward_lstm_11/Cast�
backward_lstm_11/ShapeShape=backward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_11/Shape�
$backward_lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_11/strided_slice/stack�
&backward_lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_11/strided_slice/stack_1�
&backward_lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_11/strided_slice/stack_2�
backward_lstm_11/strided_sliceStridedSlicebackward_lstm_11/Shape:output:0-backward_lstm_11/strided_slice/stack:output:0/backward_lstm_11/strided_slice/stack_1:output:0/backward_lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_11/strided_slice~
backward_lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_11/zeros/mul/y�
backward_lstm_11/zeros/mulMul'backward_lstm_11/strided_slice:output:0%backward_lstm_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros/mul�
backward_lstm_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
backward_lstm_11/zeros/Less/y�
backward_lstm_11/zeros/LessLessbackward_lstm_11/zeros/mul:z:0&backward_lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros/Less�
backward_lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_11/zeros/packed/1�
backward_lstm_11/zeros/packedPack'backward_lstm_11/strided_slice:output:0(backward_lstm_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_11/zeros/packed�
backward_lstm_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_11/zeros/Const�
backward_lstm_11/zerosFill&backward_lstm_11/zeros/packed:output:0%backward_lstm_11/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_11/zeros�
backward_lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_11/zeros_1/mul/y�
backward_lstm_11/zeros_1/mulMul'backward_lstm_11/strided_slice:output:0'backward_lstm_11/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros_1/mul�
backward_lstm_11/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2!
backward_lstm_11/zeros_1/Less/y�
backward_lstm_11/zeros_1/LessLess backward_lstm_11/zeros_1/mul:z:0(backward_lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/zeros_1/Less�
!backward_lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_11/zeros_1/packed/1�
backward_lstm_11/zeros_1/packedPack'backward_lstm_11/strided_slice:output:0*backward_lstm_11/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_11/zeros_1/packed�
backward_lstm_11/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_11/zeros_1/Const�
backward_lstm_11/zeros_1Fill(backward_lstm_11/zeros_1/packed:output:0'backward_lstm_11/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_11/zeros_1�
backward_lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_11/transpose/perm�
backward_lstm_11/transpose	Transpose=backward_lstm_11/RaggedToTensor/RaggedTensorToTensor:result:0(backward_lstm_11/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_11/transpose�
backward_lstm_11/Shape_1Shapebackward_lstm_11/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_11/Shape_1�
&backward_lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_11/strided_slice_1/stack�
(backward_lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_1/stack_1�
(backward_lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_1/stack_2�
 backward_lstm_11/strided_slice_1StridedSlice!backward_lstm_11/Shape_1:output:0/backward_lstm_11/strided_slice_1/stack:output:01backward_lstm_11/strided_slice_1/stack_1:output:01backward_lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_11/strided_slice_1�
,backward_lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2.
,backward_lstm_11/TensorArrayV2/element_shape�
backward_lstm_11/TensorArrayV2TensorListReserve5backward_lstm_11/TensorArrayV2/element_shape:output:0)backward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_11/TensorArrayV2�
backward_lstm_11/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_11/ReverseV2/axis�
backward_lstm_11/ReverseV2	ReverseV2backward_lstm_11/transpose:y:0(backward_lstm_11/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_11/ReverseV2�
Fbackward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2H
Fbackward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape�
8backward_lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_11/ReverseV2:output:0Obackward_lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_11/TensorArrayUnstack/TensorListFromTensor�
&backward_lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_11/strided_slice_2/stack�
(backward_lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_2/stack_1�
(backward_lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_2/stack_2�
 backward_lstm_11/strided_slice_2StridedSlicebackward_lstm_11/transpose:y:0/backward_lstm_11/strided_slice_2/stack:output:01backward_lstm_11/strided_slice_2/stack_1:output:01backward_lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2"
 backward_lstm_11/strided_slice_2�
3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp<backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype025
3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp�
$backward_lstm_11/lstm_cell_35/MatMulMatMul)backward_lstm_11/strided_slice_2:output:0;backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$backward_lstm_11/lstm_cell_35/MatMul�
5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp>backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype027
5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp�
&backward_lstm_11/lstm_cell_35/MatMul_1MatMulbackward_lstm_11/zeros:output:0=backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&backward_lstm_11/lstm_cell_35/MatMul_1�
!backward_lstm_11/lstm_cell_35/addAddV2.backward_lstm_11/lstm_cell_35/MatMul:product:00backward_lstm_11/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2#
!backward_lstm_11/lstm_cell_35/add�
4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp=backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype026
4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp�
%backward_lstm_11/lstm_cell_35/BiasAddBiasAdd%backward_lstm_11/lstm_cell_35/add:z:0<backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%backward_lstm_11/lstm_cell_35/BiasAdd�
-backward_lstm_11/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-backward_lstm_11/lstm_cell_35/split/split_dim�
#backward_lstm_11/lstm_cell_35/splitSplit6backward_lstm_11/lstm_cell_35/split/split_dim:output:0.backward_lstm_11/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2%
#backward_lstm_11/lstm_cell_35/split�
%backward_lstm_11/lstm_cell_35/SigmoidSigmoid,backward_lstm_11/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22'
%backward_lstm_11/lstm_cell_35/Sigmoid�
'backward_lstm_11/lstm_cell_35/Sigmoid_1Sigmoid,backward_lstm_11/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22)
'backward_lstm_11/lstm_cell_35/Sigmoid_1�
!backward_lstm_11/lstm_cell_35/mulMul+backward_lstm_11/lstm_cell_35/Sigmoid_1:y:0!backward_lstm_11/zeros_1:output:0*
T0*'
_output_shapes
:���������22#
!backward_lstm_11/lstm_cell_35/mul�
"backward_lstm_11/lstm_cell_35/ReluRelu,backward_lstm_11/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22$
"backward_lstm_11/lstm_cell_35/Relu�
#backward_lstm_11/lstm_cell_35/mul_1Mul)backward_lstm_11/lstm_cell_35/Sigmoid:y:00backward_lstm_11/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_11/lstm_cell_35/mul_1�
#backward_lstm_11/lstm_cell_35/add_1AddV2%backward_lstm_11/lstm_cell_35/mul:z:0'backward_lstm_11/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_11/lstm_cell_35/add_1�
'backward_lstm_11/lstm_cell_35/Sigmoid_2Sigmoid,backward_lstm_11/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22)
'backward_lstm_11/lstm_cell_35/Sigmoid_2�
$backward_lstm_11/lstm_cell_35/Relu_1Relu'backward_lstm_11/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22&
$backward_lstm_11/lstm_cell_35/Relu_1�
#backward_lstm_11/lstm_cell_35/mul_2Mul+backward_lstm_11/lstm_cell_35/Sigmoid_2:y:02backward_lstm_11/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_11/lstm_cell_35/mul_2�
.backward_lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   20
.backward_lstm_11/TensorArrayV2_1/element_shape�
 backward_lstm_11/TensorArrayV2_1TensorListReserve7backward_lstm_11/TensorArrayV2_1/element_shape:output:0)backward_lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_11/TensorArrayV2_1p
backward_lstm_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_11/time�
&backward_lstm_11/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_lstm_11/Max/reduction_indices�
backward_lstm_11/MaxMaxbackward_lstm_11/Cast:y:0/backward_lstm_11/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/Maxr
backward_lstm_11/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_11/sub/y�
backward_lstm_11/subSubbackward_lstm_11/Max:output:0backward_lstm_11/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_11/sub�
backward_lstm_11/Sub_1Subbackward_lstm_11/sub:z:0backward_lstm_11/Cast:y:0*
T0*#
_output_shapes
:���������2
backward_lstm_11/Sub_1�
backward_lstm_11/zeros_like	ZerosLike'backward_lstm_11/lstm_cell_35/mul_2:z:0*
T0*'
_output_shapes
:���������22
backward_lstm_11/zeros_like�
)backward_lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2+
)backward_lstm_11/while/maximum_iterations�
#backward_lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_11/while/loop_counter�	
backward_lstm_11/whileWhile,backward_lstm_11/while/loop_counter:output:02backward_lstm_11/while/maximum_iterations:output:0backward_lstm_11/time:output:0)backward_lstm_11/TensorArrayV2_1:handle:0backward_lstm_11/zeros_like:y:0backward_lstm_11/zeros:output:0!backward_lstm_11/zeros_1:output:0)backward_lstm_11/strided_slice_1:output:0Hbackward_lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_11/Sub_1:z:0<backward_lstm_11_lstm_cell_35_matmul_readvariableop_resource>backward_lstm_11_lstm_cell_35_matmul_1_readvariableop_resource=backward_lstm_11_lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( */
body'R%
#backward_lstm_11_while_body_1704593*/
cond'R%
#backward_lstm_11_while_cond_1704592*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
backward_lstm_11/while�
Abackward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2C
Abackward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape�
3backward_lstm_11/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_11/while:output:3Jbackward_lstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype025
3backward_lstm_11/TensorArrayV2Stack/TensorListStack�
&backward_lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2(
&backward_lstm_11/strided_slice_3/stack�
(backward_lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_11/strided_slice_3/stack_1�
(backward_lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_11/strided_slice_3/stack_2�
 backward_lstm_11/strided_slice_3StridedSlice<backward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_11/strided_slice_3/stack:output:01backward_lstm_11/strided_slice_3/stack_1:output:01backward_lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2"
 backward_lstm_11/strided_slice_3�
!backward_lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_11/transpose_1/perm�
backward_lstm_11/transpose_1	Transpose<backward_lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_11/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
backward_lstm_11/transpose_1�
backward_lstm_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_11/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2(forward_lstm_11/strided_slice_3:output:0)backward_lstm_11/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity�
NoOpNoOp5^backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp4^backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp6^backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp^backward_lstm_11/while4^forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp3^forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp5^forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp^forward_lstm_11/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������: : : : : : 2l
4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp4backward_lstm_11/lstm_cell_35/BiasAdd/ReadVariableOp2j
3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp3backward_lstm_11/lstm_cell_35/MatMul/ReadVariableOp2n
5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp5backward_lstm_11/lstm_cell_35/MatMul_1/ReadVariableOp20
backward_lstm_11/whilebackward_lstm_11/while2j
3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp3forward_lstm_11/lstm_cell_34/BiasAdd/ReadVariableOp2h
2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp2forward_lstm_11/lstm_cell_34/MatMul/ReadVariableOp2l
4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp4forward_lstm_11/lstm_cell_34/MatMul_1/ReadVariableOp2.
forward_lstm_11/whileforward_lstm_11/while:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
while_body_1700543
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_34_1700567_0:	�/
while_lstm_cell_34_1700569_0:	2�+
while_lstm_cell_34_1700571_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_34_1700567:	�-
while_lstm_cell_34_1700569:	2�)
while_lstm_cell_34_1700571:	���*while/lstm_cell_34/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
*while/lstm_cell_34/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_34_1700567_0while_lstm_cell_34_1700569_0while_lstm_cell_34_1700571_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������2:���������2:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_lstm_cell_34_layer_call_and_return_conditional_losses_17004652,
*while/lstm_cell_34/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_34/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity3while/lstm_cell_34/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_34/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_34/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_34_1700567while_lstm_cell_34_1700567_0":
while_lstm_cell_34_1700569while_lstm_cell_34_1700569_0":
while_lstm_cell_34_1700571while_lstm_cell_34_1700571_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2X
*while/lstm_cell_34/StatefulPartitionedCall*while/lstm_cell_34/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
�\
�
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_1704905
inputs_0>
+lstm_cell_34_matmul_readvariableop_resource:	�@
-lstm_cell_34_matmul_1_readvariableop_resource:	2�;
,lstm_cell_34_biasadd_readvariableop_resource:	�
identity��#lstm_cell_34/BiasAdd/ReadVariableOp�"lstm_cell_34/MatMul/ReadVariableOp�$lstm_cell_34/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packedc
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedg
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_34/MatMul/ReadVariableOpReadVariableOp+lstm_cell_34_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_34/MatMul/ReadVariableOp�
lstm_cell_34/MatMulMatMulstrided_slice_2:output:0*lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/MatMul�
$lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_34/MatMul_1/ReadVariableOp�
lstm_cell_34/MatMul_1MatMulzeros:output:0,lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/MatMul_1�
lstm_cell_34/addAddV2lstm_cell_34/MatMul:product:0lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/add�
#lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_34/BiasAdd/ReadVariableOp�
lstm_cell_34/BiasAddBiasAddlstm_cell_34/add:z:0+lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/BiasAdd~
lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_34/split/split_dim�
lstm_cell_34/splitSplit%lstm_cell_34/split/split_dim:output:0lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_34/split�
lstm_cell_34/SigmoidSigmoidlstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/Sigmoid�
lstm_cell_34/Sigmoid_1Sigmoidlstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_34/Sigmoid_1�
lstm_cell_34/mulMullstm_cell_34/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/mul}
lstm_cell_34/ReluRelulstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_34/Relu�
lstm_cell_34/mul_1Mullstm_cell_34/Sigmoid:y:0lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/mul_1�
lstm_cell_34/add_1AddV2lstm_cell_34/mul:z:0lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/add_1�
lstm_cell_34/Sigmoid_2Sigmoidlstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_34/Sigmoid_2|
lstm_cell_34/Relu_1Relulstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/Relu_1�
lstm_cell_34/mul_2Mullstm_cell_34/Sigmoid_2:y:0!lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_34_matmul_readvariableop_resource-lstm_cell_34_matmul_1_readvariableop_resource,lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1704821*
condR
while_cond_1704820*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity�
NoOpNoOp$^lstm_cell_34/BiasAdd/ReadVariableOp#^lstm_cell_34/MatMul/ReadVariableOp%^lstm_cell_34/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_34/BiasAdd/ReadVariableOp#lstm_cell_34/BiasAdd/ReadVariableOp2H
"lstm_cell_34/MatMul/ReadVariableOp"lstm_cell_34/MatMul/ReadVariableOp2L
$lstm_cell_34/MatMul_1/ReadVariableOp$lstm_cell_34/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
.__inference_lstm_cell_35_layer_call_fn_1706146

inputs
states_0
states_1
unknown:	�
	unknown_0:	2�
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������2:���������2:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_lstm_cell_35_layer_call_and_return_conditional_losses_17010972
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������22

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������2:���������2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������2
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������2
"
_user_specified_name
states/1
�
�
"forward_lstm_11_while_cond_1702843<
8forward_lstm_11_while_forward_lstm_11_while_loop_counterB
>forward_lstm_11_while_forward_lstm_11_while_maximum_iterations%
!forward_lstm_11_while_placeholder'
#forward_lstm_11_while_placeholder_1'
#forward_lstm_11_while_placeholder_2'
#forward_lstm_11_while_placeholder_3'
#forward_lstm_11_while_placeholder_4>
:forward_lstm_11_while_less_forward_lstm_11_strided_slice_1U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1702843___redundant_placeholder0U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1702843___redundant_placeholder1U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1702843___redundant_placeholder2U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1702843___redundant_placeholder3U
Qforward_lstm_11_while_forward_lstm_11_while_cond_1702843___redundant_placeholder4"
forward_lstm_11_while_identity
�
forward_lstm_11/while/LessLess!forward_lstm_11_while_placeholder:forward_lstm_11_while_less_forward_lstm_11_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_11/while/Less�
forward_lstm_11/while/IdentityIdentityforward_lstm_11/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_11/while/Identity"I
forward_lstm_11_while_identity'forward_lstm_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :���������2:���������2:���������2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
�F
�
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_1700402

inputs'
lstm_cell_34_1700320:	�'
lstm_cell_34_1700322:	2�#
lstm_cell_34_1700324:	�
identity��$lstm_cell_34/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packedc
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedg
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
$lstm_cell_34/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_34_1700320lstm_cell_34_1700322lstm_cell_34_1700324*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������2:���������2:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_lstm_cell_34_layer_call_and_return_conditional_losses_17003192&
$lstm_cell_34/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_34_1700320lstm_cell_34_1700322lstm_cell_34_1700324*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1700333*
condR
while_cond_1700332*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity}
NoOpNoOp%^lstm_cell_34/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_34/StatefulPartitionedCall$lstm_cell_34/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
1__inference_forward_lstm_11_layer_call_fn_1704721
inputs_0
unknown:	�
	unknown_0:	2�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_17004022
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
E__inference_dense_11_layer_call_and_return_conditional_losses_1702705

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�%
�
while_body_1701177
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_35_1701201_0:	�/
while_lstm_cell_35_1701203_0:	2�+
while_lstm_cell_35_1701205_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_35_1701201:	�-
while_lstm_cell_35_1701203:	2�)
while_lstm_cell_35_1701205:	���*while/lstm_cell_35/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
*while/lstm_cell_35/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_35_1701201_0while_lstm_cell_35_1701203_0while_lstm_cell_35_1701205_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������2:���������2:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_lstm_cell_35_layer_call_and_return_conditional_losses_17010972,
*while/lstm_cell_35/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_35/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity3while/lstm_cell_35/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_35/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_35_1701201while_lstm_cell_35_1701201_0":
while_lstm_cell_35_1701203while_lstm_cell_35_1701203_0":
while_lstm_cell_35_1701205while_lstm_cell_35_1701205_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2X
*while/lstm_cell_35/StatefulPartitionedCall*while/lstm_cell_35/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
ۗ
�
Asequential_11_bidirectional_11_forward_lstm_11_while_body_1699961z
vsequential_11_bidirectional_11_forward_lstm_11_while_sequential_11_bidirectional_11_forward_lstm_11_while_loop_counter�
|sequential_11_bidirectional_11_forward_lstm_11_while_sequential_11_bidirectional_11_forward_lstm_11_while_maximum_iterationsD
@sequential_11_bidirectional_11_forward_lstm_11_while_placeholderF
Bsequential_11_bidirectional_11_forward_lstm_11_while_placeholder_1F
Bsequential_11_bidirectional_11_forward_lstm_11_while_placeholder_2F
Bsequential_11_bidirectional_11_forward_lstm_11_while_placeholder_3F
Bsequential_11_bidirectional_11_forward_lstm_11_while_placeholder_4y
usequential_11_bidirectional_11_forward_lstm_11_while_sequential_11_bidirectional_11_forward_lstm_11_strided_slice_1_0�
�sequential_11_bidirectional_11_forward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_11_bidirectional_11_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0v
rsequential_11_bidirectional_11_forward_lstm_11_while_greater_sequential_11_bidirectional_11_forward_lstm_11_cast_0u
bsequential_11_bidirectional_11_forward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0:	�w
dsequential_11_bidirectional_11_forward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0:	2�r
csequential_11_bidirectional_11_forward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0:	�A
=sequential_11_bidirectional_11_forward_lstm_11_while_identityC
?sequential_11_bidirectional_11_forward_lstm_11_while_identity_1C
?sequential_11_bidirectional_11_forward_lstm_11_while_identity_2C
?sequential_11_bidirectional_11_forward_lstm_11_while_identity_3C
?sequential_11_bidirectional_11_forward_lstm_11_while_identity_4C
?sequential_11_bidirectional_11_forward_lstm_11_while_identity_5C
?sequential_11_bidirectional_11_forward_lstm_11_while_identity_6w
ssequential_11_bidirectional_11_forward_lstm_11_while_sequential_11_bidirectional_11_forward_lstm_11_strided_slice_1�
�sequential_11_bidirectional_11_forward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_11_bidirectional_11_forward_lstm_11_tensorarrayunstack_tensorlistfromtensort
psequential_11_bidirectional_11_forward_lstm_11_while_greater_sequential_11_bidirectional_11_forward_lstm_11_casts
`sequential_11_bidirectional_11_forward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource:	�u
bsequential_11_bidirectional_11_forward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource:	2�p
asequential_11_bidirectional_11_forward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource:	���Xsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp�Wsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp�Ysequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp�
fsequential_11/bidirectional_11/forward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2h
fsequential_11/bidirectional_11/forward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape�
Xsequential_11/bidirectional_11/forward_lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_11_bidirectional_11_forward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_11_bidirectional_11_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0@sequential_11_bidirectional_11_forward_lstm_11_while_placeholderosequential_11/bidirectional_11/forward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02Z
Xsequential_11/bidirectional_11/forward_lstm_11/while/TensorArrayV2Read/TensorListGetItem�
<sequential_11/bidirectional_11/forward_lstm_11/while/GreaterGreaterrsequential_11_bidirectional_11_forward_lstm_11_while_greater_sequential_11_bidirectional_11_forward_lstm_11_cast_0@sequential_11_bidirectional_11_forward_lstm_11_while_placeholder*
T0*#
_output_shapes
:���������2>
<sequential_11/bidirectional_11/forward_lstm_11/while/Greater�
Wsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOpbsequential_11_bidirectional_11_forward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02Y
Wsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp�
Hsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMulMatMul_sequential_11/bidirectional_11/forward_lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:0_sequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2J
Hsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMul�
Ysequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOpdsequential_11_bidirectional_11_forward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02[
Ysequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp�
Jsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMul_1MatMulBsequential_11_bidirectional_11_forward_lstm_11_while_placeholder_3asequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2L
Jsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMul_1�
Esequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/addAddV2Rsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMul:product:0Tsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2G
Esequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/add�
Xsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOpcsequential_11_bidirectional_11_forward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02Z
Xsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp�
Isequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/BiasAddBiasAddIsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/add:z:0`sequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2K
Isequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/BiasAdd�
Qsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2S
Qsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/split/split_dim�
Gsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/splitSplitZsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/split/split_dim:output:0Rsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2I
Gsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/split�
Isequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/SigmoidSigmoidPsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22K
Isequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/Sigmoid�
Ksequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/Sigmoid_1SigmoidPsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22M
Ksequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/Sigmoid_1�
Esequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/mulMulOsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/Sigmoid_1:y:0Bsequential_11_bidirectional_11_forward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22G
Esequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/mul�
Fsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/ReluReluPsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22H
Fsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/Relu�
Gsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/mul_1MulMsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/Sigmoid:y:0Tsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22I
Gsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/mul_1�
Gsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/add_1AddV2Isequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/mul:z:0Ksequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22I
Gsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/add_1�
Ksequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/Sigmoid_2SigmoidPsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22M
Ksequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/Sigmoid_2�
Hsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/Relu_1ReluKsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22J
Hsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/Relu_1�
Gsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/mul_2MulOsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/Sigmoid_2:y:0Vsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22I
Gsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/mul_2�
;sequential_11/bidirectional_11/forward_lstm_11/while/SelectSelect@sequential_11/bidirectional_11/forward_lstm_11/while/Greater:z:0Ksequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/mul_2:z:0Bsequential_11_bidirectional_11_forward_lstm_11_while_placeholder_2*
T0*'
_output_shapes
:���������22=
;sequential_11/bidirectional_11/forward_lstm_11/while/Select�
=sequential_11/bidirectional_11/forward_lstm_11/while/Select_1Select@sequential_11/bidirectional_11/forward_lstm_11/while/Greater:z:0Ksequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/mul_2:z:0Bsequential_11_bidirectional_11_forward_lstm_11_while_placeholder_3*
T0*'
_output_shapes
:���������22?
=sequential_11/bidirectional_11/forward_lstm_11/while/Select_1�
=sequential_11/bidirectional_11/forward_lstm_11/while/Select_2Select@sequential_11/bidirectional_11/forward_lstm_11/while/Greater:z:0Ksequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/add_1:z:0Bsequential_11_bidirectional_11_forward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22?
=sequential_11/bidirectional_11/forward_lstm_11/while/Select_2�
Ysequential_11/bidirectional_11/forward_lstm_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemBsequential_11_bidirectional_11_forward_lstm_11_while_placeholder_1@sequential_11_bidirectional_11_forward_lstm_11_while_placeholderDsequential_11/bidirectional_11/forward_lstm_11/while/Select:output:0*
_output_shapes
: *
element_dtype02[
Ysequential_11/bidirectional_11/forward_lstm_11/while/TensorArrayV2Write/TensorListSetItem�
:sequential_11/bidirectional_11/forward_lstm_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_11/bidirectional_11/forward_lstm_11/while/add/y�
8sequential_11/bidirectional_11/forward_lstm_11/while/addAddV2@sequential_11_bidirectional_11_forward_lstm_11_while_placeholderCsequential_11/bidirectional_11/forward_lstm_11/while/add/y:output:0*
T0*
_output_shapes
: 2:
8sequential_11/bidirectional_11/forward_lstm_11/while/add�
<sequential_11/bidirectional_11/forward_lstm_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2>
<sequential_11/bidirectional_11/forward_lstm_11/while/add_1/y�
:sequential_11/bidirectional_11/forward_lstm_11/while/add_1AddV2vsequential_11_bidirectional_11_forward_lstm_11_while_sequential_11_bidirectional_11_forward_lstm_11_while_loop_counterEsequential_11/bidirectional_11/forward_lstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2<
:sequential_11/bidirectional_11/forward_lstm_11/while/add_1�
=sequential_11/bidirectional_11/forward_lstm_11/while/IdentityIdentity>sequential_11/bidirectional_11/forward_lstm_11/while/add_1:z:0:^sequential_11/bidirectional_11/forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2?
=sequential_11/bidirectional_11/forward_lstm_11/while/Identity�
?sequential_11/bidirectional_11/forward_lstm_11/while/Identity_1Identity|sequential_11_bidirectional_11_forward_lstm_11_while_sequential_11_bidirectional_11_forward_lstm_11_while_maximum_iterations:^sequential_11/bidirectional_11/forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2A
?sequential_11/bidirectional_11/forward_lstm_11/while/Identity_1�
?sequential_11/bidirectional_11/forward_lstm_11/while/Identity_2Identity<sequential_11/bidirectional_11/forward_lstm_11/while/add:z:0:^sequential_11/bidirectional_11/forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2A
?sequential_11/bidirectional_11/forward_lstm_11/while/Identity_2�
?sequential_11/bidirectional_11/forward_lstm_11/while/Identity_3Identityisequential_11/bidirectional_11/forward_lstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0:^sequential_11/bidirectional_11/forward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2A
?sequential_11/bidirectional_11/forward_lstm_11/while/Identity_3�
?sequential_11/bidirectional_11/forward_lstm_11/while/Identity_4IdentityDsequential_11/bidirectional_11/forward_lstm_11/while/Select:output:0:^sequential_11/bidirectional_11/forward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22A
?sequential_11/bidirectional_11/forward_lstm_11/while/Identity_4�
?sequential_11/bidirectional_11/forward_lstm_11/while/Identity_5IdentityFsequential_11/bidirectional_11/forward_lstm_11/while/Select_1:output:0:^sequential_11/bidirectional_11/forward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22A
?sequential_11/bidirectional_11/forward_lstm_11/while/Identity_5�
?sequential_11/bidirectional_11/forward_lstm_11/while/Identity_6IdentityFsequential_11/bidirectional_11/forward_lstm_11/while/Select_2:output:0:^sequential_11/bidirectional_11/forward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22A
?sequential_11/bidirectional_11/forward_lstm_11/while/Identity_6�
9sequential_11/bidirectional_11/forward_lstm_11/while/NoOpNoOpY^sequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOpX^sequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOpZ^sequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2;
9sequential_11/bidirectional_11/forward_lstm_11/while/NoOp"�
psequential_11_bidirectional_11_forward_lstm_11_while_greater_sequential_11_bidirectional_11_forward_lstm_11_castrsequential_11_bidirectional_11_forward_lstm_11_while_greater_sequential_11_bidirectional_11_forward_lstm_11_cast_0"�
=sequential_11_bidirectional_11_forward_lstm_11_while_identityFsequential_11/bidirectional_11/forward_lstm_11/while/Identity:output:0"�
?sequential_11_bidirectional_11_forward_lstm_11_while_identity_1Hsequential_11/bidirectional_11/forward_lstm_11/while/Identity_1:output:0"�
?sequential_11_bidirectional_11_forward_lstm_11_while_identity_2Hsequential_11/bidirectional_11/forward_lstm_11/while/Identity_2:output:0"�
?sequential_11_bidirectional_11_forward_lstm_11_while_identity_3Hsequential_11/bidirectional_11/forward_lstm_11/while/Identity_3:output:0"�
?sequential_11_bidirectional_11_forward_lstm_11_while_identity_4Hsequential_11/bidirectional_11/forward_lstm_11/while/Identity_4:output:0"�
?sequential_11_bidirectional_11_forward_lstm_11_while_identity_5Hsequential_11/bidirectional_11/forward_lstm_11/while/Identity_5:output:0"�
?sequential_11_bidirectional_11_forward_lstm_11_while_identity_6Hsequential_11/bidirectional_11/forward_lstm_11/while/Identity_6:output:0"�
asequential_11_bidirectional_11_forward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resourcecsequential_11_bidirectional_11_forward_lstm_11_while_lstm_cell_34_biasadd_readvariableop_resource_0"�
bsequential_11_bidirectional_11_forward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resourcedsequential_11_bidirectional_11_forward_lstm_11_while_lstm_cell_34_matmul_1_readvariableop_resource_0"�
`sequential_11_bidirectional_11_forward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resourcebsequential_11_bidirectional_11_forward_lstm_11_while_lstm_cell_34_matmul_readvariableop_resource_0"�
ssequential_11_bidirectional_11_forward_lstm_11_while_sequential_11_bidirectional_11_forward_lstm_11_strided_slice_1usequential_11_bidirectional_11_forward_lstm_11_while_sequential_11_bidirectional_11_forward_lstm_11_strided_slice_1_0"�
�sequential_11_bidirectional_11_forward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_11_bidirectional_11_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor�sequential_11_bidirectional_11_forward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_11_bidirectional_11_forward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2�
Xsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOpXsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/BiasAdd/ReadVariableOp2�
Wsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOpWsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMul/ReadVariableOp2�
Ysequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOpYsequential_11/bidirectional_11/forward_lstm_11/while/lstm_cell_34/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:���������
�
�
2__inference_backward_lstm_11_layer_call_fn_1705391

inputs
unknown:	�
	unknown_0:	2�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_17018292
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�%
�
while_body_1700333
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_34_1700357_0:	�/
while_lstm_cell_34_1700359_0:	2�+
while_lstm_cell_34_1700361_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_34_1700357:	�-
while_lstm_cell_34_1700359:	2�)
while_lstm_cell_34_1700361:	���*while/lstm_cell_34/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
*while/lstm_cell_34/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_34_1700357_0while_lstm_cell_34_1700359_0while_lstm_cell_34_1700361_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������2:���������2:���������2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_lstm_cell_34_layer_call_and_return_conditional_losses_17003192,
*while/lstm_cell_34/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_34/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity3while/lstm_cell_34/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_34/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_34/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_34_1700357while_lstm_cell_34_1700357_0":
while_lstm_cell_34_1700359while_lstm_cell_34_1700359_0":
while_lstm_cell_34_1700361while_lstm_cell_34_1700361_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2X
*while/lstm_cell_34/StatefulPartitionedCall*while/lstm_cell_34/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
�\
�
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_1705056
inputs_0>
+lstm_cell_34_matmul_readvariableop_resource:	�@
-lstm_cell_34_matmul_1_readvariableop_resource:	2�;
,lstm_cell_34_biasadd_readvariableop_resource:	�
identity��#lstm_cell_34/BiasAdd/ReadVariableOp�"lstm_cell_34/MatMul/ReadVariableOp�$lstm_cell_34/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packedc
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedg
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_34/MatMul/ReadVariableOpReadVariableOp+lstm_cell_34_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_34/MatMul/ReadVariableOp�
lstm_cell_34/MatMulMatMulstrided_slice_2:output:0*lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/MatMul�
$lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_34_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_34/MatMul_1/ReadVariableOp�
lstm_cell_34/MatMul_1MatMulzeros:output:0,lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/MatMul_1�
lstm_cell_34/addAddV2lstm_cell_34/MatMul:product:0lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/add�
#lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_34_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_34/BiasAdd/ReadVariableOp�
lstm_cell_34/BiasAddBiasAddlstm_cell_34/add:z:0+lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_34/BiasAdd~
lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_34/split/split_dim�
lstm_cell_34/splitSplit%lstm_cell_34/split/split_dim:output:0lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_34/split�
lstm_cell_34/SigmoidSigmoidlstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/Sigmoid�
lstm_cell_34/Sigmoid_1Sigmoidlstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_34/Sigmoid_1�
lstm_cell_34/mulMullstm_cell_34/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/mul}
lstm_cell_34/ReluRelulstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_34/Relu�
lstm_cell_34/mul_1Mullstm_cell_34/Sigmoid:y:0lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/mul_1�
lstm_cell_34/add_1AddV2lstm_cell_34/mul:z:0lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/add_1�
lstm_cell_34/Sigmoid_2Sigmoidlstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_34/Sigmoid_2|
lstm_cell_34/Relu_1Relulstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/Relu_1�
lstm_cell_34/mul_2Mullstm_cell_34/Sigmoid_2:y:0!lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_34/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_34_matmul_readvariableop_resource-lstm_cell_34_matmul_1_readvariableop_resource,lstm_cell_34_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1704972*
condR
while_cond_1704971*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity�
NoOpNoOp$^lstm_cell_34/BiasAdd/ReadVariableOp#^lstm_cell_34/MatMul/ReadVariableOp%^lstm_cell_34/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_34/BiasAdd/ReadVariableOp#lstm_cell_34/BiasAdd/ReadVariableOp2H
"lstm_cell_34/MatMul/ReadVariableOp"lstm_cell_34/MatMul/ReadVariableOp2L
$lstm_cell_34/MatMul_1/ReadVariableOp$lstm_cell_34/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
Asequential_11_bidirectional_11_forward_lstm_11_while_cond_1699960z
vsequential_11_bidirectional_11_forward_lstm_11_while_sequential_11_bidirectional_11_forward_lstm_11_while_loop_counter�
|sequential_11_bidirectional_11_forward_lstm_11_while_sequential_11_bidirectional_11_forward_lstm_11_while_maximum_iterationsD
@sequential_11_bidirectional_11_forward_lstm_11_while_placeholderF
Bsequential_11_bidirectional_11_forward_lstm_11_while_placeholder_1F
Bsequential_11_bidirectional_11_forward_lstm_11_while_placeholder_2F
Bsequential_11_bidirectional_11_forward_lstm_11_while_placeholder_3F
Bsequential_11_bidirectional_11_forward_lstm_11_while_placeholder_4|
xsequential_11_bidirectional_11_forward_lstm_11_while_less_sequential_11_bidirectional_11_forward_lstm_11_strided_slice_1�
�sequential_11_bidirectional_11_forward_lstm_11_while_sequential_11_bidirectional_11_forward_lstm_11_while_cond_1699960___redundant_placeholder0�
�sequential_11_bidirectional_11_forward_lstm_11_while_sequential_11_bidirectional_11_forward_lstm_11_while_cond_1699960___redundant_placeholder1�
�sequential_11_bidirectional_11_forward_lstm_11_while_sequential_11_bidirectional_11_forward_lstm_11_while_cond_1699960___redundant_placeholder2�
�sequential_11_bidirectional_11_forward_lstm_11_while_sequential_11_bidirectional_11_forward_lstm_11_while_cond_1699960___redundant_placeholder3�
�sequential_11_bidirectional_11_forward_lstm_11_while_sequential_11_bidirectional_11_forward_lstm_11_while_cond_1699960___redundant_placeholder4A
=sequential_11_bidirectional_11_forward_lstm_11_while_identity
�
9sequential_11/bidirectional_11/forward_lstm_11/while/LessLess@sequential_11_bidirectional_11_forward_lstm_11_while_placeholderxsequential_11_bidirectional_11_forward_lstm_11_while_less_sequential_11_bidirectional_11_forward_lstm_11_strided_slice_1*
T0*
_output_shapes
: 2;
9sequential_11/bidirectional_11/forward_lstm_11/while/Less�
=sequential_11/bidirectional_11/forward_lstm_11/while/IdentityIdentity=sequential_11/bidirectional_11/forward_lstm_11/while/Less:z:0*
T0
*
_output_shapes
: 2?
=sequential_11/bidirectional_11/forward_lstm_11/while/Identity"�
=sequential_11_bidirectional_11_forward_lstm_11_while_identityFsequential_11/bidirectional_11/forward_lstm_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :���������2:���������2:���������2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
�
�
while_cond_1702109
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1702109___redundant_placeholder05
1while_while_cond_1702109___redundant_placeholder15
1while_while_cond_1702109___redundant_placeholder25
1while_while_cond_1702109___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_1704820
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1704820___redundant_placeholder05
1while_while_cond_1704820___redundant_placeholder15
1while_while_cond_1704820___redundant_placeholder25
1while_while_cond_1704820___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
�?
�
while_body_1705274
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_34_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_34_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_34_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_34_matmul_readvariableop_resource:	�F
3while_lstm_cell_34_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_34_biasadd_readvariableop_resource:	���)while/lstm_cell_34/BiasAdd/ReadVariableOp�(while/lstm_cell_34/MatMul/ReadVariableOp�*while/lstm_cell_34/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_34/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_34_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_34/MatMul/ReadVariableOp�
while/lstm_cell_34/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/MatMul�
*while/lstm_cell_34/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_34_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_34/MatMul_1/ReadVariableOp�
while/lstm_cell_34/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_34/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/MatMul_1�
while/lstm_cell_34/addAddV2#while/lstm_cell_34/MatMul:product:0%while/lstm_cell_34/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/add�
)while/lstm_cell_34/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_34_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_34/BiasAdd/ReadVariableOp�
while/lstm_cell_34/BiasAddBiasAddwhile/lstm_cell_34/add:z:01while/lstm_cell_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_34/BiasAdd�
"while/lstm_cell_34/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_34/split/split_dim�
while/lstm_cell_34/splitSplit+while/lstm_cell_34/split/split_dim:output:0#while/lstm_cell_34/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_34/split�
while/lstm_cell_34/SigmoidSigmoid!while/lstm_cell_34/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Sigmoid�
while/lstm_cell_34/Sigmoid_1Sigmoid!while/lstm_cell_34/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Sigmoid_1�
while/lstm_cell_34/mulMul while/lstm_cell_34/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/mul�
while/lstm_cell_34/ReluRelu!while/lstm_cell_34/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Relu�
while/lstm_cell_34/mul_1Mulwhile/lstm_cell_34/Sigmoid:y:0%while/lstm_cell_34/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/mul_1�
while/lstm_cell_34/add_1AddV2while/lstm_cell_34/mul:z:0while/lstm_cell_34/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/add_1�
while/lstm_cell_34/Sigmoid_2Sigmoid!while/lstm_cell_34/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Sigmoid_2�
while/lstm_cell_34/Relu_1Reluwhile/lstm_cell_34/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/Relu_1�
while/lstm_cell_34/mul_2Mul while/lstm_cell_34/Sigmoid_2:y:0'while/lstm_cell_34/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_34/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_34/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_34/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_34/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_34/BiasAdd/ReadVariableOp)^while/lstm_cell_34/MatMul/ReadVariableOp+^while/lstm_cell_34/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_34_biasadd_readvariableop_resource4while_lstm_cell_34_biasadd_readvariableop_resource_0"l
3while_lstm_cell_34_matmul_1_readvariableop_resource5while_lstm_cell_34_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_34_matmul_readvariableop_resource3while_lstm_cell_34_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_34/BiasAdd/ReadVariableOp)while/lstm_cell_34/BiasAdd/ReadVariableOp2T
(while/lstm_cell_34/MatMul/ReadVariableOp(while/lstm_cell_34/MatMul/ReadVariableOp2X
*while/lstm_cell_34/MatMul_1/ReadVariableOp*while/lstm_cell_34/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
�
�
#backward_lstm_11_while_cond_1704592>
:backward_lstm_11_while_backward_lstm_11_while_loop_counterD
@backward_lstm_11_while_backward_lstm_11_while_maximum_iterations&
"backward_lstm_11_while_placeholder(
$backward_lstm_11_while_placeholder_1(
$backward_lstm_11_while_placeholder_2(
$backward_lstm_11_while_placeholder_3(
$backward_lstm_11_while_placeholder_4@
<backward_lstm_11_while_less_backward_lstm_11_strided_slice_1W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1704592___redundant_placeholder0W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1704592___redundant_placeholder1W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1704592___redundant_placeholder2W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1704592___redundant_placeholder3W
Sbackward_lstm_11_while_backward_lstm_11_while_cond_1704592___redundant_placeholder4#
backward_lstm_11_while_identity
�
backward_lstm_11/while/LessLess"backward_lstm_11_while_placeholder<backward_lstm_11_while_less_backward_lstm_11_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_11/while/Less�
backward_lstm_11/while/IdentityIdentitybackward_lstm_11/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_11/while/Identity"K
backward_lstm_11_while_identity(backward_lstm_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :���������2:���������2:���������2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
�?
�
while_body_1705471
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_35_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_35_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_35_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_35_matmul_readvariableop_resource:	�F
3while_lstm_cell_35_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_35_biasadd_readvariableop_resource:	���)while/lstm_cell_35/BiasAdd/ReadVariableOp�(while/lstm_cell_35/MatMul/ReadVariableOp�*while/lstm_cell_35/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_35/MatMul/ReadVariableOp�
while/lstm_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/MatMul�
*while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_35/MatMul_1/ReadVariableOp�
while/lstm_cell_35/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/MatMul_1�
while/lstm_cell_35/addAddV2#while/lstm_cell_35/MatMul:product:0%while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/add�
)while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_35/BiasAdd/ReadVariableOp�
while/lstm_cell_35/BiasAddBiasAddwhile/lstm_cell_35/add:z:01while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_35/BiasAdd�
"while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_35/split/split_dim�
while/lstm_cell_35/splitSplit+while/lstm_cell_35/split/split_dim:output:0#while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_35/split�
while/lstm_cell_35/SigmoidSigmoid!while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Sigmoid�
while/lstm_cell_35/Sigmoid_1Sigmoid!while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Sigmoid_1�
while/lstm_cell_35/mulMul while/lstm_cell_35/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/mul�
while/lstm_cell_35/ReluRelu!while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Relu�
while/lstm_cell_35/mul_1Mulwhile/lstm_cell_35/Sigmoid:y:0%while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/mul_1�
while/lstm_cell_35/add_1AddV2while/lstm_cell_35/mul:z:0while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/add_1�
while/lstm_cell_35/Sigmoid_2Sigmoid!while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Sigmoid_2�
while/lstm_cell_35/Relu_1Reluwhile/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/Relu_1�
while/lstm_cell_35/mul_2Mul while/lstm_cell_35/Sigmoid_2:y:0'while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_35/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_35/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_35/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_35/BiasAdd/ReadVariableOp)^while/lstm_cell_35/MatMul/ReadVariableOp+^while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_35_biasadd_readvariableop_resource4while_lstm_cell_35_biasadd_readvariableop_resource_0"l
3while_lstm_cell_35_matmul_1_readvariableop_resource5while_lstm_cell_35_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_35_matmul_readvariableop_resource3while_lstm_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_35/BiasAdd/ReadVariableOp)while/lstm_cell_35/BiasAdd/ReadVariableOp2T
(while/lstm_cell_35/MatMul/ReadVariableOp(while/lstm_cell_35/MatMul/ReadVariableOp2X
*while/lstm_cell_35/MatMul_1/ReadVariableOp*while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_1700542
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1700542___redundant_placeholder05
1while_while_cond_1700542___redundant_placeholder15
1while_while_cond_1700542___redundant_placeholder25
1while_while_cond_1700542___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
��
�
Bsequential_11_bidirectional_11_backward_lstm_11_while_body_1700140|
xsequential_11_bidirectional_11_backward_lstm_11_while_sequential_11_bidirectional_11_backward_lstm_11_while_loop_counter�
~sequential_11_bidirectional_11_backward_lstm_11_while_sequential_11_bidirectional_11_backward_lstm_11_while_maximum_iterationsE
Asequential_11_bidirectional_11_backward_lstm_11_while_placeholderG
Csequential_11_bidirectional_11_backward_lstm_11_while_placeholder_1G
Csequential_11_bidirectional_11_backward_lstm_11_while_placeholder_2G
Csequential_11_bidirectional_11_backward_lstm_11_while_placeholder_3G
Csequential_11_bidirectional_11_backward_lstm_11_while_placeholder_4{
wsequential_11_bidirectional_11_backward_lstm_11_while_sequential_11_bidirectional_11_backward_lstm_11_strided_slice_1_0�
�sequential_11_bidirectional_11_backward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_11_bidirectional_11_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0v
rsequential_11_bidirectional_11_backward_lstm_11_while_less_sequential_11_bidirectional_11_backward_lstm_11_sub_1_0v
csequential_11_bidirectional_11_backward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0:	�x
esequential_11_bidirectional_11_backward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0:	2�s
dsequential_11_bidirectional_11_backward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0:	�B
>sequential_11_bidirectional_11_backward_lstm_11_while_identityD
@sequential_11_bidirectional_11_backward_lstm_11_while_identity_1D
@sequential_11_bidirectional_11_backward_lstm_11_while_identity_2D
@sequential_11_bidirectional_11_backward_lstm_11_while_identity_3D
@sequential_11_bidirectional_11_backward_lstm_11_while_identity_4D
@sequential_11_bidirectional_11_backward_lstm_11_while_identity_5D
@sequential_11_bidirectional_11_backward_lstm_11_while_identity_6y
usequential_11_bidirectional_11_backward_lstm_11_while_sequential_11_bidirectional_11_backward_lstm_11_strided_slice_1�
�sequential_11_bidirectional_11_backward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_11_bidirectional_11_backward_lstm_11_tensorarrayunstack_tensorlistfromtensort
psequential_11_bidirectional_11_backward_lstm_11_while_less_sequential_11_bidirectional_11_backward_lstm_11_sub_1t
asequential_11_bidirectional_11_backward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource:	�v
csequential_11_bidirectional_11_backward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource:	2�q
bsequential_11_bidirectional_11_backward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource:	���Ysequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp�Xsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp�Zsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp�
gsequential_11/bidirectional_11/backward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2i
gsequential_11/bidirectional_11/backward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape�
Ysequential_11/bidirectional_11/backward_lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_11_bidirectional_11_backward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_11_bidirectional_11_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0Asequential_11_bidirectional_11_backward_lstm_11_while_placeholderpsequential_11/bidirectional_11/backward_lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02[
Ysequential_11/bidirectional_11/backward_lstm_11/while/TensorArrayV2Read/TensorListGetItem�
:sequential_11/bidirectional_11/backward_lstm_11/while/LessLessrsequential_11_bidirectional_11_backward_lstm_11_while_less_sequential_11_bidirectional_11_backward_lstm_11_sub_1_0Asequential_11_bidirectional_11_backward_lstm_11_while_placeholder*
T0*#
_output_shapes
:���������2<
:sequential_11/bidirectional_11/backward_lstm_11/while/Less�
Xsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOpcsequential_11_bidirectional_11_backward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02Z
Xsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp�
Isequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMulMatMul`sequential_11/bidirectional_11/backward_lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:0`sequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2K
Isequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMul�
Zsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOpesequential_11_bidirectional_11_backward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02\
Zsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp�
Ksequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMul_1MatMulCsequential_11_bidirectional_11_backward_lstm_11_while_placeholder_3bsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2M
Ksequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMul_1�
Fsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/addAddV2Ssequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMul:product:0Usequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2H
Fsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/add�
Ysequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOpdsequential_11_bidirectional_11_backward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02[
Ysequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp�
Jsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/BiasAddBiasAddJsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/add:z:0asequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2L
Jsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/BiasAdd�
Rsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2T
Rsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/split/split_dim�
Hsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/splitSplit[sequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/split/split_dim:output:0Ssequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2J
Hsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/split�
Jsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/SigmoidSigmoidQsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22L
Jsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/Sigmoid�
Lsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/Sigmoid_1SigmoidQsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22N
Lsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/Sigmoid_1�
Fsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/mulMulPsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/Sigmoid_1:y:0Csequential_11_bidirectional_11_backward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22H
Fsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/mul�
Gsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/ReluReluQsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22I
Gsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/Relu�
Hsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/mul_1MulNsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/Sigmoid:y:0Usequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22J
Hsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/mul_1�
Hsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/add_1AddV2Jsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/mul:z:0Lsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22J
Hsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/add_1�
Lsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/Sigmoid_2SigmoidQsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22N
Lsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/Sigmoid_2�
Isequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/Relu_1ReluLsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22K
Isequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/Relu_1�
Hsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/mul_2MulPsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/Sigmoid_2:y:0Wsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22J
Hsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/mul_2�
<sequential_11/bidirectional_11/backward_lstm_11/while/SelectSelect>sequential_11/bidirectional_11/backward_lstm_11/while/Less:z:0Lsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/mul_2:z:0Csequential_11_bidirectional_11_backward_lstm_11_while_placeholder_2*
T0*'
_output_shapes
:���������22>
<sequential_11/bidirectional_11/backward_lstm_11/while/Select�
>sequential_11/bidirectional_11/backward_lstm_11/while/Select_1Select>sequential_11/bidirectional_11/backward_lstm_11/while/Less:z:0Lsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/mul_2:z:0Csequential_11_bidirectional_11_backward_lstm_11_while_placeholder_3*
T0*'
_output_shapes
:���������22@
>sequential_11/bidirectional_11/backward_lstm_11/while/Select_1�
>sequential_11/bidirectional_11/backward_lstm_11/while/Select_2Select>sequential_11/bidirectional_11/backward_lstm_11/while/Less:z:0Lsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/add_1:z:0Csequential_11_bidirectional_11_backward_lstm_11_while_placeholder_4*
T0*'
_output_shapes
:���������22@
>sequential_11/bidirectional_11/backward_lstm_11/while/Select_2�
Zsequential_11/bidirectional_11/backward_lstm_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemCsequential_11_bidirectional_11_backward_lstm_11_while_placeholder_1Asequential_11_bidirectional_11_backward_lstm_11_while_placeholderEsequential_11/bidirectional_11/backward_lstm_11/while/Select:output:0*
_output_shapes
: *
element_dtype02\
Zsequential_11/bidirectional_11/backward_lstm_11/while/TensorArrayV2Write/TensorListSetItem�
;sequential_11/bidirectional_11/backward_lstm_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2=
;sequential_11/bidirectional_11/backward_lstm_11/while/add/y�
9sequential_11/bidirectional_11/backward_lstm_11/while/addAddV2Asequential_11_bidirectional_11_backward_lstm_11_while_placeholderDsequential_11/bidirectional_11/backward_lstm_11/while/add/y:output:0*
T0*
_output_shapes
: 2;
9sequential_11/bidirectional_11/backward_lstm_11/while/add�
=sequential_11/bidirectional_11/backward_lstm_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2?
=sequential_11/bidirectional_11/backward_lstm_11/while/add_1/y�
;sequential_11/bidirectional_11/backward_lstm_11/while/add_1AddV2xsequential_11_bidirectional_11_backward_lstm_11_while_sequential_11_bidirectional_11_backward_lstm_11_while_loop_counterFsequential_11/bidirectional_11/backward_lstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2=
;sequential_11/bidirectional_11/backward_lstm_11/while/add_1�
>sequential_11/bidirectional_11/backward_lstm_11/while/IdentityIdentity?sequential_11/bidirectional_11/backward_lstm_11/while/add_1:z:0;^sequential_11/bidirectional_11/backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2@
>sequential_11/bidirectional_11/backward_lstm_11/while/Identity�
@sequential_11/bidirectional_11/backward_lstm_11/while/Identity_1Identity~sequential_11_bidirectional_11_backward_lstm_11_while_sequential_11_bidirectional_11_backward_lstm_11_while_maximum_iterations;^sequential_11/bidirectional_11/backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2B
@sequential_11/bidirectional_11/backward_lstm_11/while/Identity_1�
@sequential_11/bidirectional_11/backward_lstm_11/while/Identity_2Identity=sequential_11/bidirectional_11/backward_lstm_11/while/add:z:0;^sequential_11/bidirectional_11/backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2B
@sequential_11/bidirectional_11/backward_lstm_11/while/Identity_2�
@sequential_11/bidirectional_11/backward_lstm_11/while/Identity_3Identityjsequential_11/bidirectional_11/backward_lstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0;^sequential_11/bidirectional_11/backward_lstm_11/while/NoOp*
T0*
_output_shapes
: 2B
@sequential_11/bidirectional_11/backward_lstm_11/while/Identity_3�
@sequential_11/bidirectional_11/backward_lstm_11/while/Identity_4IdentityEsequential_11/bidirectional_11/backward_lstm_11/while/Select:output:0;^sequential_11/bidirectional_11/backward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22B
@sequential_11/bidirectional_11/backward_lstm_11/while/Identity_4�
@sequential_11/bidirectional_11/backward_lstm_11/while/Identity_5IdentityGsequential_11/bidirectional_11/backward_lstm_11/while/Select_1:output:0;^sequential_11/bidirectional_11/backward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22B
@sequential_11/bidirectional_11/backward_lstm_11/while/Identity_5�
@sequential_11/bidirectional_11/backward_lstm_11/while/Identity_6IdentityGsequential_11/bidirectional_11/backward_lstm_11/while/Select_2:output:0;^sequential_11/bidirectional_11/backward_lstm_11/while/NoOp*
T0*'
_output_shapes
:���������22B
@sequential_11/bidirectional_11/backward_lstm_11/while/Identity_6�
:sequential_11/bidirectional_11/backward_lstm_11/while/NoOpNoOpZ^sequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOpY^sequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp[^sequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2<
:sequential_11/bidirectional_11/backward_lstm_11/while/NoOp"�
>sequential_11_bidirectional_11_backward_lstm_11_while_identityGsequential_11/bidirectional_11/backward_lstm_11/while/Identity:output:0"�
@sequential_11_bidirectional_11_backward_lstm_11_while_identity_1Isequential_11/bidirectional_11/backward_lstm_11/while/Identity_1:output:0"�
@sequential_11_bidirectional_11_backward_lstm_11_while_identity_2Isequential_11/bidirectional_11/backward_lstm_11/while/Identity_2:output:0"�
@sequential_11_bidirectional_11_backward_lstm_11_while_identity_3Isequential_11/bidirectional_11/backward_lstm_11/while/Identity_3:output:0"�
@sequential_11_bidirectional_11_backward_lstm_11_while_identity_4Isequential_11/bidirectional_11/backward_lstm_11/while/Identity_4:output:0"�
@sequential_11_bidirectional_11_backward_lstm_11_while_identity_5Isequential_11/bidirectional_11/backward_lstm_11/while/Identity_5:output:0"�
@sequential_11_bidirectional_11_backward_lstm_11_while_identity_6Isequential_11/bidirectional_11/backward_lstm_11/while/Identity_6:output:0"�
psequential_11_bidirectional_11_backward_lstm_11_while_less_sequential_11_bidirectional_11_backward_lstm_11_sub_1rsequential_11_bidirectional_11_backward_lstm_11_while_less_sequential_11_bidirectional_11_backward_lstm_11_sub_1_0"�
bsequential_11_bidirectional_11_backward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resourcedsequential_11_bidirectional_11_backward_lstm_11_while_lstm_cell_35_biasadd_readvariableop_resource_0"�
csequential_11_bidirectional_11_backward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resourceesequential_11_bidirectional_11_backward_lstm_11_while_lstm_cell_35_matmul_1_readvariableop_resource_0"�
asequential_11_bidirectional_11_backward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resourcecsequential_11_bidirectional_11_backward_lstm_11_while_lstm_cell_35_matmul_readvariableop_resource_0"�
usequential_11_bidirectional_11_backward_lstm_11_while_sequential_11_bidirectional_11_backward_lstm_11_strided_slice_1wsequential_11_bidirectional_11_backward_lstm_11_while_sequential_11_bidirectional_11_backward_lstm_11_strided_slice_1_0"�
�sequential_11_bidirectional_11_backward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_11_bidirectional_11_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor�sequential_11_bidirectional_11_backward_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_11_bidirectional_11_backward_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2�
Ysequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOpYsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/BiasAdd/ReadVariableOp2�
Xsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOpXsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMul/ReadVariableOp2�
Zsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOpZsequential_11/bidirectional_11/backward_lstm_11/while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:���������
�
�
while_cond_1705470
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1705470___redundant_placeholder05
1while_while_cond_1705470___redundant_placeholder15
1while_while_cond_1705470___redundant_placeholder25
1while_while_cond_1705470___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������2:���������2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������2:-)
'
_output_shapes
:���������2:

_output_shapes
: :

_output_shapes
:
�

�
2__inference_bidirectional_11_layer_call_fn_1703370

inputs
inputs_1	
unknown:	�
	unknown_0:	2�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	2�
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_17031202
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�^
�
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_1706014

inputs>
+lstm_cell_35_matmul_readvariableop_resource:	�@
-lstm_cell_35_matmul_1_readvariableop_resource:	2�;
,lstm_cell_35_biasadd_readvariableop_resource:	�
identity��#lstm_cell_35/BiasAdd/ReadVariableOp�"lstm_cell_35/MatMul/ReadVariableOp�$lstm_cell_35/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packedc
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedg
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis�
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'���������������������������2
	ReverseV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_35/MatMul/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_35/MatMul/ReadVariableOp�
lstm_cell_35/MatMulMatMulstrided_slice_2:output:0*lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/MatMul�
$lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_35/MatMul_1/ReadVariableOp�
lstm_cell_35/MatMul_1MatMulzeros:output:0,lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/MatMul_1�
lstm_cell_35/addAddV2lstm_cell_35/MatMul:product:0lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/add�
#lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_35/BiasAdd/ReadVariableOp�
lstm_cell_35/BiasAddBiasAddlstm_cell_35/add:z:0+lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_35/BiasAdd~
lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_35/split/split_dim�
lstm_cell_35/splitSplit%lstm_cell_35/split/split_dim:output:0lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_35/split�
lstm_cell_35/SigmoidSigmoidlstm_cell_35/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/Sigmoid�
lstm_cell_35/Sigmoid_1Sigmoidlstm_cell_35/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_35/Sigmoid_1�
lstm_cell_35/mulMullstm_cell_35/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/mul}
lstm_cell_35/ReluRelulstm_cell_35/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_35/Relu�
lstm_cell_35/mul_1Mullstm_cell_35/Sigmoid:y:0lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/mul_1�
lstm_cell_35/add_1AddV2lstm_cell_35/mul:z:0lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/add_1�
lstm_cell_35/Sigmoid_2Sigmoidlstm_cell_35/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_35/Sigmoid_2|
lstm_cell_35/Relu_1Relulstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/Relu_1�
lstm_cell_35/mul_2Mullstm_cell_35/Sigmoid_2:y:0!lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_35/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_35_matmul_readvariableop_resource-lstm_cell_35_matmul_1_readvariableop_resource,lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1705930*
condR
while_cond_1705929*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identity�
NoOpNoOp$^lstm_cell_35/BiasAdd/ReadVariableOp#^lstm_cell_35/MatMul/ReadVariableOp%^lstm_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_35/BiasAdd/ReadVariableOp#lstm_cell_35/BiasAdd/ReadVariableOp2H
"lstm_cell_35/MatMul/ReadVariableOp"lstm_cell_35/MatMul/ReadVariableOp2L
$lstm_cell_35/MatMul_1/ReadVariableOp$lstm_cell_35/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
args_0/
serving_default_args_0:0���������
9
args_0_1-
serving_default_args_0_1:0	���������<
dense_110
StatefulPartitionedCall:0���������tensorflow/serving/predict:ٹ
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
x__call__
y_default_save_signature
*z&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
	forward_layer

backward_layer
regularization_losses
	variables
trainable_variables
	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
�
iter

beta_1

beta_2
	decay
learning_ratem`mambmcmdmemfmgvhvivjvkvlvmvnvo
vhatp
vhatq
vhatr
vhats
vhatt
vhatu
vhatv
vhatw"
	optimizer
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
�
 layer_metrics
!non_trainable_variables
regularization_losses
	variables
"metrics
trainable_variables
#layer_regularization_losses

$layers
x__call__
y_default_save_signature
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
�
%cell
&
state_spec
'regularization_losses
(	variables
)trainable_variables
*	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
�
+cell
,
state_spec
-regularization_losses
.	variables
/trainable_variables
0	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
�
1layer_metrics
2non_trainable_variables
regularization_losses
	variables
3metrics
trainable_variables
4layer_regularization_losses

5layers
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
!:d2dense_11/kernel
:2dense_11/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
6layer_metrics
7non_trainable_variables
regularization_losses
	variables
8metrics
trainable_variables
9layer_regularization_losses

:layers
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
G:E	�24bidirectional_11/forward_lstm_11/lstm_cell_34/kernel
Q:O	2�2>bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel
A:?�22bidirectional_11/forward_lstm_11/lstm_cell_34/bias
H:F	�25bidirectional_11/backward_lstm_11/lstm_cell_35/kernel
R:P	2�2?bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel
B:@�23bidirectional_11/backward_lstm_11/lstm_cell_35/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
;0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
<
state_size

kernel
recurrent_kernel
bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
�
Alayer_metrics
Bnon_trainable_variables

Cstates
'regularization_losses
(	variables
Dmetrics
)trainable_variables
Elayer_regularization_losses

Flayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
G
state_size

kernel
recurrent_kernel
bias
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
�
Llayer_metrics
Mnon_trainable_variables

Nstates
-regularization_losses
.	variables
Ometrics
/trainable_variables
Player_regularization_losses

Qlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
N
	Rtotal
	Scount
T	variables
U	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
�
Vlayer_metrics
Wnon_trainable_variables
=regularization_losses
>	variables
Xmetrics
?trainable_variables
Ylayer_regularization_losses

Zlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
%0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
�
[layer_metrics
\non_trainable_variables
Hregularization_losses
I	variables
]metrics
Jtrainable_variables
^layer_regularization_losses

_layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
+0"
trackable_list_wrapper
:  (2total
:  (2count
.
R0
S1"
trackable_list_wrapper
-
T	variables"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
&:$d2Adam/dense_11/kernel/m
 :2Adam/dense_11/bias/m
L:J	�2;Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/m
V:T	2�2EAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/m
F:D�29Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/m
M:K	�2<Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/m
W:U	2�2FAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/m
G:E�2:Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/m
&:$d2Adam/dense_11/kernel/v
 :2Adam/dense_11/bias/v
L:J	�2;Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/v
V:T	2�2EAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/v
F:D�29Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/v
M:K	�2<Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/v
W:U	2�2FAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/v
G:E�2:Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/v
):'d2Adam/dense_11/kernel/vhat
#:!2Adam/dense_11/bias/vhat
O:M	�2>Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/kernel/vhat
Y:W	2�2HAdam/bidirectional_11/forward_lstm_11/lstm_cell_34/recurrent_kernel/vhat
I:G�2<Adam/bidirectional_11/forward_lstm_11/lstm_cell_34/bias/vhat
P:N	�2?Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/kernel/vhat
Z:X	2�2IAdam/bidirectional_11/backward_lstm_11/lstm_cell_35/recurrent_kernel/vhat
J:H�2=Adam/bidirectional_11/backward_lstm_11/lstm_cell_35/bias/vhat
�2�
/__inference_sequential_11_layer_call_fn_1702731
/__inference_sequential_11_layer_call_fn_1703224�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
"__inference__wrapped_model_1700244args_0args_0_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_sequential_11_layer_call_and_return_conditional_losses_1703247
J__inference_sequential_11_layer_call_and_return_conditional_losses_1703270�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
2__inference_bidirectional_11_layer_call_fn_1703317
2__inference_bidirectional_11_layer_call_fn_1703334
2__inference_bidirectional_11_layer_call_fn_1703352
2__inference_bidirectional_11_layer_call_fn_1703370�
���
FullArgSpecO
argsG�D
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults�
p 

 

 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_1703672
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_1703974
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_1704332
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_1704690�
���
FullArgSpecO
argsG�D
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults�
p 

 

 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_dense_11_layer_call_fn_1704699�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_11_layer_call_and_return_conditional_losses_1704710�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_1703300args_0args_0_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_forward_lstm_11_layer_call_fn_1704721
1__inference_forward_lstm_11_layer_call_fn_1704732
1__inference_forward_lstm_11_layer_call_fn_1704743
1__inference_forward_lstm_11_layer_call_fn_1704754�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_1704905
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_1705056
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_1705207
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_1705358�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
2__inference_backward_lstm_11_layer_call_fn_1705369
2__inference_backward_lstm_11_layer_call_fn_1705380
2__inference_backward_lstm_11_layer_call_fn_1705391
2__inference_backward_lstm_11_layer_call_fn_1705402�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_1705555
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_1705708
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_1705861
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_1706014�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
.__inference_lstm_cell_34_layer_call_fn_1706031
.__inference_lstm_cell_34_layer_call_fn_1706048�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
I__inference_lstm_cell_34_layer_call_and_return_conditional_losses_1706080
I__inference_lstm_cell_34_layer_call_and_return_conditional_losses_1706112�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
.__inference_lstm_cell_35_layer_call_fn_1706129
.__inference_lstm_cell_35_layer_call_fn_1706146�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
I__inference_lstm_cell_35_layer_call_and_return_conditional_losses_1706178
I__inference_lstm_cell_35_layer_call_and_return_conditional_losses_1706210�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 �
"__inference__wrapped_model_1700244�\�Y
R�O
M�J4�1
!�������������������
�
`
�	RaggedTensorSpec
� "3�0
.
dense_11"�
dense_11����������
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_1705555}O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "%�"
�
0���������2
� �
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_1705708}O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "%�"
�
0���������2
� �
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_1705861Q�N
G�D
6�3
inputs'���������������������������

 
p 

 
� "%�"
�
0���������2
� �
M__inference_backward_lstm_11_layer_call_and_return_conditional_losses_1706014Q�N
G�D
6�3
inputs'���������������������������

 
p

 
� "%�"
�
0���������2
� �
2__inference_backward_lstm_11_layer_call_fn_1705369pO�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "����������2�
2__inference_backward_lstm_11_layer_call_fn_1705380pO�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "����������2�
2__inference_backward_lstm_11_layer_call_fn_1705391rQ�N
G�D
6�3
inputs'���������������������������

 
p 

 
� "����������2�
2__inference_backward_lstm_11_layer_call_fn_1705402rQ�N
G�D
6�3
inputs'���������������������������

 
p

 
� "����������2�
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_1703672�\�Y
R�O
=�:
8�5
inputs/0'���������������������������
p 

 

 

 
� "%�"
�
0���������d
� �
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_1703974�\�Y
R�O
=�:
8�5
inputs/0'���������������������������
p

 

 

 
� "%�"
�
0���������d
� �
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_1704332�l�i
b�_
M�J4�1
!�������������������
�
`
�	RaggedTensorSpec
p 

 

 

 
� "%�"
�
0���������d
� �
M__inference_bidirectional_11_layer_call_and_return_conditional_losses_1704690�l�i
b�_
M�J4�1
!�������������������
�
`
�	RaggedTensorSpec
p

 

 

 
� "%�"
�
0���������d
� �
2__inference_bidirectional_11_layer_call_fn_1703317�\�Y
R�O
=�:
8�5
inputs/0'���������������������������
p 

 

 

 
� "����������d�
2__inference_bidirectional_11_layer_call_fn_1703334�\�Y
R�O
=�:
8�5
inputs/0'���������������������������
p

 

 

 
� "����������d�
2__inference_bidirectional_11_layer_call_fn_1703352�l�i
b�_
M�J4�1
!�������������������
�
`
�	RaggedTensorSpec
p 

 

 

 
� "����������d�
2__inference_bidirectional_11_layer_call_fn_1703370�l�i
b�_
M�J4�1
!�������������������
�
`
�	RaggedTensorSpec
p

 

 

 
� "����������d�
E__inference_dense_11_layer_call_and_return_conditional_losses_1704710\/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� }
*__inference_dense_11_layer_call_fn_1704699O/�,
%�"
 �
inputs���������d
� "�����������
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_1704905}O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "%�"
�
0���������2
� �
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_1705056}O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "%�"
�
0���������2
� �
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_1705207Q�N
G�D
6�3
inputs'���������������������������

 
p 

 
� "%�"
�
0���������2
� �
L__inference_forward_lstm_11_layer_call_and_return_conditional_losses_1705358Q�N
G�D
6�3
inputs'���������������������������

 
p

 
� "%�"
�
0���������2
� �
1__inference_forward_lstm_11_layer_call_fn_1704721pO�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "����������2�
1__inference_forward_lstm_11_layer_call_fn_1704732pO�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "����������2�
1__inference_forward_lstm_11_layer_call_fn_1704743rQ�N
G�D
6�3
inputs'���������������������������

 
p 

 
� "����������2�
1__inference_forward_lstm_11_layer_call_fn_1704754rQ�N
G�D
6�3
inputs'���������������������������

 
p

 
� "����������2�
I__inference_lstm_cell_34_layer_call_and_return_conditional_losses_1706080���}
v�s
 �
inputs���������
K�H
"�
states/0���������2
"�
states/1���������2
p 
� "s�p
i�f
�
0/0���������2
E�B
�
0/1/0���������2
�
0/1/1���������2
� �
I__inference_lstm_cell_34_layer_call_and_return_conditional_losses_1706112���}
v�s
 �
inputs���������
K�H
"�
states/0���������2
"�
states/1���������2
p
� "s�p
i�f
�
0/0���������2
E�B
�
0/1/0���������2
�
0/1/1���������2
� �
.__inference_lstm_cell_34_layer_call_fn_1706031���}
v�s
 �
inputs���������
K�H
"�
states/0���������2
"�
states/1���������2
p 
� "c�`
�
0���������2
A�>
�
1/0���������2
�
1/1���������2�
.__inference_lstm_cell_34_layer_call_fn_1706048���}
v�s
 �
inputs���������
K�H
"�
states/0���������2
"�
states/1���������2
p
� "c�`
�
0���������2
A�>
�
1/0���������2
�
1/1���������2�
I__inference_lstm_cell_35_layer_call_and_return_conditional_losses_1706178���}
v�s
 �
inputs���������
K�H
"�
states/0���������2
"�
states/1���������2
p 
� "s�p
i�f
�
0/0���������2
E�B
�
0/1/0���������2
�
0/1/1���������2
� �
I__inference_lstm_cell_35_layer_call_and_return_conditional_losses_1706210���}
v�s
 �
inputs���������
K�H
"�
states/0���������2
"�
states/1���������2
p
� "s�p
i�f
�
0/0���������2
E�B
�
0/1/0���������2
�
0/1/1���������2
� �
.__inference_lstm_cell_35_layer_call_fn_1706129���}
v�s
 �
inputs���������
K�H
"�
states/0���������2
"�
states/1���������2
p 
� "c�`
�
0���������2
A�>
�
1/0���������2
�
1/1���������2�
.__inference_lstm_cell_35_layer_call_fn_1706146���}
v�s
 �
inputs���������
K�H
"�
states/0���������2
"�
states/1���������2
p
� "c�`
�
0���������2
A�>
�
1/0���������2
�
1/1���������2�
J__inference_sequential_11_layer_call_and_return_conditional_losses_1703247�d�a
Z�W
M�J4�1
!�������������������
�
`
�	RaggedTensorSpec
p 

 
� "%�"
�
0���������
� �
J__inference_sequential_11_layer_call_and_return_conditional_losses_1703270�d�a
Z�W
M�J4�1
!�������������������
�
`
�	RaggedTensorSpec
p

 
� "%�"
�
0���������
� �
/__inference_sequential_11_layer_call_fn_1702731�d�a
Z�W
M�J4�1
!�������������������
�
`
�	RaggedTensorSpec
p 

 
� "�����������
/__inference_sequential_11_layer_call_fn_1703224�d�a
Z�W
M�J4�1
!�������������������
�
`
�	RaggedTensorSpec
p

 
� "�����������
%__inference_signature_wrapper_1703300�e�b
� 
[�X
*
args_0 �
args_0���������
*
args_0_1�
args_0_1���������	"3�0
.
dense_11"�
dense_11���������