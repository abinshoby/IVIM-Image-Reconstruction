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
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d* 
shared_namedense_20/kernel
s
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes

:d*
dtype0
r
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
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
4bidirectional_20/forward_lstm_20/lstm_cell_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*E
shared_name64bidirectional_20/forward_lstm_20/lstm_cell_61/kernel
�
Hbidirectional_20/forward_lstm_20/lstm_cell_61/kernel/Read/ReadVariableOpReadVariableOp4bidirectional_20/forward_lstm_20/lstm_cell_61/kernel*
_output_shapes
:	�*
dtype0
�
>bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*O
shared_name@>bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel
�
Rbidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/Read/ReadVariableOpReadVariableOp>bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel*
_output_shapes
:	2�*
dtype0
�
2bidirectional_20/forward_lstm_20/lstm_cell_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*C
shared_name42bidirectional_20/forward_lstm_20/lstm_cell_61/bias
�
Fbidirectional_20/forward_lstm_20/lstm_cell_61/bias/Read/ReadVariableOpReadVariableOp2bidirectional_20/forward_lstm_20/lstm_cell_61/bias*
_output_shapes	
:�*
dtype0
�
5bidirectional_20/backward_lstm_20/lstm_cell_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*F
shared_name75bidirectional_20/backward_lstm_20/lstm_cell_62/kernel
�
Ibidirectional_20/backward_lstm_20/lstm_cell_62/kernel/Read/ReadVariableOpReadVariableOp5bidirectional_20/backward_lstm_20/lstm_cell_62/kernel*
_output_shapes
:	�*
dtype0
�
?bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*P
shared_nameA?bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel
�
Sbidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/Read/ReadVariableOpReadVariableOp?bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel*
_output_shapes
:	2�*
dtype0
�
3bidirectional_20/backward_lstm_20/lstm_cell_62/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*D
shared_name53bidirectional_20/backward_lstm_20/lstm_cell_62/bias
�
Gbidirectional_20/backward_lstm_20/lstm_cell_62/bias/Read/ReadVariableOpReadVariableOp3bidirectional_20/backward_lstm_20/lstm_cell_62/bias*
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
Adam/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_20/kernel/m
�
*Adam/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/m*
_output_shapes

:d*
dtype0
�
Adam/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_20/bias/m
y
(Adam/dense_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/m*
_output_shapes
:*
dtype0
�
;Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*L
shared_name=;Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/m
�
OAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/m/Read/ReadVariableOpReadVariableOp;Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/m*
_output_shapes
:	�*
dtype0
�
EAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*V
shared_nameGEAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/m
�
YAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpEAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/m*
_output_shapes
:	2�*
dtype0
�
9Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*J
shared_name;9Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/m
�
MAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/m/Read/ReadVariableOpReadVariableOp9Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/m*
_output_shapes	
:�*
dtype0
�
<Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*M
shared_name><Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/m
�
PAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/m/Read/ReadVariableOpReadVariableOp<Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/m*
_output_shapes
:	�*
dtype0
�
FAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*W
shared_nameHFAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/m
�
ZAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/m*
_output_shapes
:	2�*
dtype0
�
:Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*K
shared_name<:Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/m
�
NAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/m/Read/ReadVariableOpReadVariableOp:Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_20/kernel/v
�
*Adam/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/v*
_output_shapes

:d*
dtype0
�
Adam/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_20/bias/v
y
(Adam/dense_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/v*
_output_shapes
:*
dtype0
�
;Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*L
shared_name=;Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/v
�
OAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/v/Read/ReadVariableOpReadVariableOp;Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/v*
_output_shapes
:	�*
dtype0
�
EAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*V
shared_nameGEAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/v
�
YAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpEAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/v*
_output_shapes
:	2�*
dtype0
�
9Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*J
shared_name;9Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/v
�
MAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/v/Read/ReadVariableOpReadVariableOp9Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/v*
_output_shapes	
:�*
dtype0
�
<Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*M
shared_name><Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/v
�
PAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/v/Read/ReadVariableOpReadVariableOp<Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/v*
_output_shapes
:	�*
dtype0
�
FAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*W
shared_nameHFAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/v
�
ZAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/v*
_output_shapes
:	2�*
dtype0
�
:Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*K
shared_name<:Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/v
�
NAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/v/Read/ReadVariableOpReadVariableOp:Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_20/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d**
shared_nameAdam/dense_20/kernel/vhat
�
-Adam/dense_20/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/vhat*
_output_shapes

:d*
dtype0
�
Adam/dense_20/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/dense_20/bias/vhat

+Adam/dense_20/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/vhat*
_output_shapes
:*
dtype0
�
>Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*O
shared_name@>Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/vhat
�
RAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/vhat/Read/ReadVariableOpReadVariableOp>Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/vhat*
_output_shapes
:	�*
dtype0
�
HAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*Y
shared_nameJHAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/vhat
�
\Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpHAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/vhat*
_output_shapes
:	2�*
dtype0
�
<Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*M
shared_name><Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/vhat
�
PAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/vhat/Read/ReadVariableOpReadVariableOp<Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/vhat*
_output_shapes	
:�*
dtype0
�
?Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*P
shared_nameA?Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/vhat
�
SAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/vhat/Read/ReadVariableOpReadVariableOp?Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/vhat*
_output_shapes
:	�*
dtype0
�
IAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*Z
shared_nameKIAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/vhat
�
]Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpIAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/vhat*
_output_shapes
:	2�*
dtype0
�
=Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*N
shared_name?=Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/vhat
�
QAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/vhat/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/vhat*
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
VARIABLE_VALUEdense_20/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_20/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUE4bidirectional_20/forward_lstm_20/lstm_cell_61/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE>bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2bidirectional_20/forward_lstm_20/lstm_cell_61/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5bidirectional_20/backward_lstm_20/lstm_cell_62/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE?bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3bidirectional_20/backward_lstm_20/lstm_cell_62/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_20/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_20/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE;Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEEAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE9Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE<Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEFAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE:Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_20/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_20/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE;Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEEAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE9Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE<Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEFAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE:Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_20/kernel/vhatUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/dense_20/bias/vhatSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE>Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/vhatEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEHAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/vhatEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE<Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/vhatEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE?Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/vhatEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEIAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/vhatEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE=Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/vhatEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_0_14bidirectional_20/forward_lstm_20/lstm_cell_61/kernel>bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel2bidirectional_20/forward_lstm_20/lstm_cell_61/bias5bidirectional_20/backward_lstm_20/lstm_cell_62/kernel?bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel3bidirectional_20/backward_lstm_20/lstm_cell_62/biasdense_20/kerneldense_20/bias*
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
%__inference_signature_wrapper_2414816
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpHbidirectional_20/forward_lstm_20/lstm_cell_61/kernel/Read/ReadVariableOpRbidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/Read/ReadVariableOpFbidirectional_20/forward_lstm_20/lstm_cell_61/bias/Read/ReadVariableOpIbidirectional_20/backward_lstm_20/lstm_cell_62/kernel/Read/ReadVariableOpSbidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/Read/ReadVariableOpGbidirectional_20/backward_lstm_20/lstm_cell_62/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_20/kernel/m/Read/ReadVariableOp(Adam/dense_20/bias/m/Read/ReadVariableOpOAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/m/Read/ReadVariableOpYAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/m/Read/ReadVariableOpMAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/m/Read/ReadVariableOpPAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/m/Read/ReadVariableOpZAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/m/Read/ReadVariableOpNAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/m/Read/ReadVariableOp*Adam/dense_20/kernel/v/Read/ReadVariableOp(Adam/dense_20/bias/v/Read/ReadVariableOpOAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/v/Read/ReadVariableOpYAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/v/Read/ReadVariableOpMAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/v/Read/ReadVariableOpPAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/v/Read/ReadVariableOpZAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/v/Read/ReadVariableOpNAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/v/Read/ReadVariableOp-Adam/dense_20/kernel/vhat/Read/ReadVariableOp+Adam/dense_20/bias/vhat/Read/ReadVariableOpRAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/vhat/Read/ReadVariableOp\Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/vhat/Read/ReadVariableOpPAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/vhat/Read/ReadVariableOpSAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/vhat/Read/ReadVariableOp]Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/vhat/Read/ReadVariableOpQAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/vhat/Read/ReadVariableOpConst*4
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
 __inference__traced_save_2417867
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_20/kerneldense_20/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate4bidirectional_20/forward_lstm_20/lstm_cell_61/kernel>bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel2bidirectional_20/forward_lstm_20/lstm_cell_61/bias5bidirectional_20/backward_lstm_20/lstm_cell_62/kernel?bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel3bidirectional_20/backward_lstm_20/lstm_cell_62/biastotalcountAdam/dense_20/kernel/mAdam/dense_20/bias/m;Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/mEAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/m9Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/m<Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/mFAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/m:Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/mAdam/dense_20/kernel/vAdam/dense_20/bias/v;Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/vEAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/v9Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/v<Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/vFAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/v:Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/vAdam/dense_20/kernel/vhatAdam/dense_20/bias/vhat>Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/vhatHAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/vhat<Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/vhat?Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/vhatIAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/vhat=Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/vhat*3
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
#__inference__traced_restore_2417994��8
�H
�
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_2412762

inputs'
lstm_cell_62_2412680:	�'
lstm_cell_62_2412682:	2�#
lstm_cell_62_2412684:	�
identity��$lstm_cell_62/StatefulPartitionedCall�whileD
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
$lstm_cell_62/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_62_2412680lstm_cell_62_2412682lstm_cell_62_2412684*
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
I__inference_lstm_cell_62_layer_call_and_return_conditional_losses_24126132&
$lstm_cell_62/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_62_2412680lstm_cell_62_2412682lstm_cell_62_2412684*
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
while_body_2412693*
condR
while_cond_2412692*K
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
NoOpNoOp%^lstm_cell_62/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_62/StatefulPartitionedCall$lstm_cell_62/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
"forward_lstm_20_while_cond_2415929<
8forward_lstm_20_while_forward_lstm_20_while_loop_counterB
>forward_lstm_20_while_forward_lstm_20_while_maximum_iterations%
!forward_lstm_20_while_placeholder'
#forward_lstm_20_while_placeholder_1'
#forward_lstm_20_while_placeholder_2'
#forward_lstm_20_while_placeholder_3'
#forward_lstm_20_while_placeholder_4>
:forward_lstm_20_while_less_forward_lstm_20_strided_slice_1U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2415929___redundant_placeholder0U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2415929___redundant_placeholder1U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2415929___redundant_placeholder2U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2415929___redundant_placeholder3U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2415929___redundant_placeholder4"
forward_lstm_20_while_identity
�
forward_lstm_20/while/LessLess!forward_lstm_20_while_placeholder:forward_lstm_20_while_less_forward_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_20/while/Less�
forward_lstm_20/while/IdentityIdentityforward_lstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_20/while/Identity"I
forward_lstm_20_while_identity'forward_lstm_20/while/Identity:output:0*(
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
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_2416421
inputs_0>
+lstm_cell_61_matmul_readvariableop_resource:	�@
-lstm_cell_61_matmul_1_readvariableop_resource:	2�;
,lstm_cell_61_biasadd_readvariableop_resource:	�
identity��#lstm_cell_61/BiasAdd/ReadVariableOp�"lstm_cell_61/MatMul/ReadVariableOp�$lstm_cell_61/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_61/MatMul/ReadVariableOpReadVariableOp+lstm_cell_61_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_61/MatMul/ReadVariableOp�
lstm_cell_61/MatMulMatMulstrided_slice_2:output:0*lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/MatMul�
$lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_61_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_61/MatMul_1/ReadVariableOp�
lstm_cell_61/MatMul_1MatMulzeros:output:0,lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/MatMul_1�
lstm_cell_61/addAddV2lstm_cell_61/MatMul:product:0lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/add�
#lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_61_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_61/BiasAdd/ReadVariableOp�
lstm_cell_61/BiasAddBiasAddlstm_cell_61/add:z:0+lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/BiasAdd~
lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_61/split/split_dim�
lstm_cell_61/splitSplit%lstm_cell_61/split/split_dim:output:0lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_61/split�
lstm_cell_61/SigmoidSigmoidlstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/Sigmoid�
lstm_cell_61/Sigmoid_1Sigmoidlstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_61/Sigmoid_1�
lstm_cell_61/mulMullstm_cell_61/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/mul}
lstm_cell_61/ReluRelulstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_61/Relu�
lstm_cell_61/mul_1Mullstm_cell_61/Sigmoid:y:0lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/mul_1�
lstm_cell_61/add_1AddV2lstm_cell_61/mul:z:0lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/add_1�
lstm_cell_61/Sigmoid_2Sigmoidlstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_61/Sigmoid_2|
lstm_cell_61/Relu_1Relulstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/Relu_1�
lstm_cell_61/mul_2Mullstm_cell_61/Sigmoid_2:y:0!lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_61_matmul_readvariableop_resource-lstm_cell_61_matmul_1_readvariableop_resource,lstm_cell_61_biasadd_readvariableop_resource*
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
while_body_2416337*
condR
while_cond_2416336*K
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
NoOpNoOp$^lstm_cell_61/BiasAdd/ReadVariableOp#^lstm_cell_61/MatMul/ReadVariableOp%^lstm_cell_61/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_61/BiasAdd/ReadVariableOp#lstm_cell_61/BiasAdd/ReadVariableOp2H
"lstm_cell_61/MatMul/ReadVariableOp"lstm_cell_61/MatMul/ReadVariableOp2L
$lstm_cell_61/MatMul_1/ReadVariableOp$lstm_cell_61/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
"forward_lstm_20_while_cond_2414359<
8forward_lstm_20_while_forward_lstm_20_while_loop_counterB
>forward_lstm_20_while_forward_lstm_20_while_maximum_iterations%
!forward_lstm_20_while_placeholder'
#forward_lstm_20_while_placeholder_1'
#forward_lstm_20_while_placeholder_2'
#forward_lstm_20_while_placeholder_3'
#forward_lstm_20_while_placeholder_4>
:forward_lstm_20_while_less_forward_lstm_20_strided_slice_1U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2414359___redundant_placeholder0U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2414359___redundant_placeholder1U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2414359___redundant_placeholder2U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2414359___redundant_placeholder3U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2414359___redundant_placeholder4"
forward_lstm_20_while_identity
�
forward_lstm_20/while/LessLess!forward_lstm_20_while_placeholder:forward_lstm_20_while_less_forward_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_20/while/Less�
forward_lstm_20/while/IdentityIdentityforward_lstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_20/while/Identity"I
forward_lstm_20_while_identity'forward_lstm_20/while/Identity:output:0*(
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
�
�
"forward_lstm_20_while_cond_2414952<
8forward_lstm_20_while_forward_lstm_20_while_loop_counterB
>forward_lstm_20_while_forward_lstm_20_while_maximum_iterations%
!forward_lstm_20_while_placeholder'
#forward_lstm_20_while_placeholder_1'
#forward_lstm_20_while_placeholder_2'
#forward_lstm_20_while_placeholder_3>
:forward_lstm_20_while_less_forward_lstm_20_strided_slice_1U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2414952___redundant_placeholder0U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2414952___redundant_placeholder1U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2414952___redundant_placeholder2U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2414952___redundant_placeholder3"
forward_lstm_20_while_identity
�
forward_lstm_20/while/LessLess!forward_lstm_20_while_placeholder:forward_lstm_20_while_less_forward_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_20/while/Less�
forward_lstm_20/while/IdentityIdentityforward_lstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_20/while/Identity"I
forward_lstm_20_while_identity'forward_lstm_20/while/Identity:output:0*(
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
�%
�
while_body_2412693
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_62_2412717_0:	�/
while_lstm_cell_62_2412719_0:	2�+
while_lstm_cell_62_2412721_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_62_2412717:	�-
while_lstm_cell_62_2412719:	2�)
while_lstm_cell_62_2412721:	���*while/lstm_cell_62/StatefulPartitionedCall�
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
*while/lstm_cell_62/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_62_2412717_0while_lstm_cell_62_2412719_0while_lstm_cell_62_2412721_0*
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
I__inference_lstm_cell_62_layer_call_and_return_conditional_losses_24126132,
*while/lstm_cell_62/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_62/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_62/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_62/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_62/StatefulPartitionedCall*"
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
while_lstm_cell_62_2412717while_lstm_cell_62_2412717_0":
while_lstm_cell_62_2412719while_lstm_cell_62_2412719_0":
while_lstm_cell_62_2412721while_lstm_cell_62_2412721_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2X
*while/lstm_cell_62/StatefulPartitionedCall*while/lstm_cell_62/StatefulPartitionedCall: 
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
#backward_lstm_20_while_body_2415751>
:backward_lstm_20_while_backward_lstm_20_while_loop_counterD
@backward_lstm_20_while_backward_lstm_20_while_maximum_iterations&
"backward_lstm_20_while_placeholder(
$backward_lstm_20_while_placeholder_1(
$backward_lstm_20_while_placeholder_2(
$backward_lstm_20_while_placeholder_3(
$backward_lstm_20_while_placeholder_4=
9backward_lstm_20_while_backward_lstm_20_strided_slice_1_0y
ubackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_08
4backward_lstm_20_while_less_backward_lstm_20_sub_1_0W
Dbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0:	�Y
Fbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0:	2�T
Ebackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0:	�#
backward_lstm_20_while_identity%
!backward_lstm_20_while_identity_1%
!backward_lstm_20_while_identity_2%
!backward_lstm_20_while_identity_3%
!backward_lstm_20_while_identity_4%
!backward_lstm_20_while_identity_5%
!backward_lstm_20_while_identity_6;
7backward_lstm_20_while_backward_lstm_20_strided_slice_1w
sbackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor6
2backward_lstm_20_while_less_backward_lstm_20_sub_1U
Bbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource:	�W
Dbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource:	2�R
Cbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource:	���:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp�9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp�;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp�
Hbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2J
Hbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape�
:backward_lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_20_while_placeholderQbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02<
:backward_lstm_20/while/TensorArrayV2Read/TensorListGetItem�
backward_lstm_20/while/LessLess4backward_lstm_20_while_less_backward_lstm_20_sub_1_0"backward_lstm_20_while_placeholder*
T0*#
_output_shapes
:���������2
backward_lstm_20/while/Less�
9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOpReadVariableOpDbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02;
9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp�
*backward_lstm_20/while/lstm_cell_62/MatMulMatMulAbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:0Abackward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*backward_lstm_20/while/lstm_cell_62/MatMul�
;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOpFbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02=
;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp�
,backward_lstm_20/while/lstm_cell_62/MatMul_1MatMul$backward_lstm_20_while_placeholder_3Cbackward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2.
,backward_lstm_20/while/lstm_cell_62/MatMul_1�
'backward_lstm_20/while/lstm_cell_62/addAddV24backward_lstm_20/while/lstm_cell_62/MatMul:product:06backward_lstm_20/while/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2)
'backward_lstm_20/while/lstm_cell_62/add�
:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOpEbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02<
:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp�
+backward_lstm_20/while/lstm_cell_62/BiasAddBiasAdd+backward_lstm_20/while/lstm_cell_62/add:z:0Bbackward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+backward_lstm_20/while/lstm_cell_62/BiasAdd�
3backward_lstm_20/while/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3backward_lstm_20/while/lstm_cell_62/split/split_dim�
)backward_lstm_20/while/lstm_cell_62/splitSplit<backward_lstm_20/while/lstm_cell_62/split/split_dim:output:04backward_lstm_20/while/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2+
)backward_lstm_20/while/lstm_cell_62/split�
+backward_lstm_20/while/lstm_cell_62/SigmoidSigmoid2backward_lstm_20/while/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22-
+backward_lstm_20/while/lstm_cell_62/Sigmoid�
-backward_lstm_20/while/lstm_cell_62/Sigmoid_1Sigmoid2backward_lstm_20/while/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22/
-backward_lstm_20/while/lstm_cell_62/Sigmoid_1�
'backward_lstm_20/while/lstm_cell_62/mulMul1backward_lstm_20/while/lstm_cell_62/Sigmoid_1:y:0$backward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22)
'backward_lstm_20/while/lstm_cell_62/mul�
(backward_lstm_20/while/lstm_cell_62/ReluRelu2backward_lstm_20/while/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22*
(backward_lstm_20/while/lstm_cell_62/Relu�
)backward_lstm_20/while/lstm_cell_62/mul_1Mul/backward_lstm_20/while/lstm_cell_62/Sigmoid:y:06backward_lstm_20/while/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_20/while/lstm_cell_62/mul_1�
)backward_lstm_20/while/lstm_cell_62/add_1AddV2+backward_lstm_20/while/lstm_cell_62/mul:z:0-backward_lstm_20/while/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_20/while/lstm_cell_62/add_1�
-backward_lstm_20/while/lstm_cell_62/Sigmoid_2Sigmoid2backward_lstm_20/while/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22/
-backward_lstm_20/while/lstm_cell_62/Sigmoid_2�
*backward_lstm_20/while/lstm_cell_62/Relu_1Relu-backward_lstm_20/while/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22,
*backward_lstm_20/while/lstm_cell_62/Relu_1�
)backward_lstm_20/while/lstm_cell_62/mul_2Mul1backward_lstm_20/while/lstm_cell_62/Sigmoid_2:y:08backward_lstm_20/while/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_20/while/lstm_cell_62/mul_2�
backward_lstm_20/while/SelectSelectbackward_lstm_20/while/Less:z:0-backward_lstm_20/while/lstm_cell_62/mul_2:z:0$backward_lstm_20_while_placeholder_2*
T0*'
_output_shapes
:���������22
backward_lstm_20/while/Select�
backward_lstm_20/while/Select_1Selectbackward_lstm_20/while/Less:z:0-backward_lstm_20/while/lstm_cell_62/mul_2:z:0$backward_lstm_20_while_placeholder_3*
T0*'
_output_shapes
:���������22!
backward_lstm_20/while/Select_1�
backward_lstm_20/while/Select_2Selectbackward_lstm_20/while/Less:z:0-backward_lstm_20/while/lstm_cell_62/add_1:z:0$backward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22!
backward_lstm_20/while/Select_2�
;backward_lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_20_while_placeholder_1"backward_lstm_20_while_placeholder&backward_lstm_20/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_20/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_20/while/add/y�
backward_lstm_20/while/addAddV2"backward_lstm_20_while_placeholder%backward_lstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/while/add�
backward_lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_20/while/add_1/y�
backward_lstm_20/while/add_1AddV2:backward_lstm_20_while_backward_lstm_20_while_loop_counter'backward_lstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/while/add_1�
backward_lstm_20/while/IdentityIdentity backward_lstm_20/while/add_1:z:0^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_20/while/Identity�
!backward_lstm_20/while/Identity_1Identity@backward_lstm_20_while_backward_lstm_20_while_maximum_iterations^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_20/while/Identity_1�
!backward_lstm_20/while/Identity_2Identitybackward_lstm_20/while/add:z:0^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_20/while/Identity_2�
!backward_lstm_20/while/Identity_3IdentityKbackward_lstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_20/while/Identity_3�
!backward_lstm_20/while/Identity_4Identity&backward_lstm_20/while/Select:output:0^backward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/while/Identity_4�
!backward_lstm_20/while/Identity_5Identity(backward_lstm_20/while/Select_1:output:0^backward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/while/Identity_5�
!backward_lstm_20/while/Identity_6Identity(backward_lstm_20/while/Select_2:output:0^backward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/while/Identity_6�
backward_lstm_20/while/NoOpNoOp;^backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp:^backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp<^backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_20/while/NoOp"t
7backward_lstm_20_while_backward_lstm_20_strided_slice_19backward_lstm_20_while_backward_lstm_20_strided_slice_1_0"K
backward_lstm_20_while_identity(backward_lstm_20/while/Identity:output:0"O
!backward_lstm_20_while_identity_1*backward_lstm_20/while/Identity_1:output:0"O
!backward_lstm_20_while_identity_2*backward_lstm_20/while/Identity_2:output:0"O
!backward_lstm_20_while_identity_3*backward_lstm_20/while/Identity_3:output:0"O
!backward_lstm_20_while_identity_4*backward_lstm_20/while/Identity_4:output:0"O
!backward_lstm_20_while_identity_5*backward_lstm_20/while/Identity_5:output:0"O
!backward_lstm_20_while_identity_6*backward_lstm_20/while/Identity_6:output:0"j
2backward_lstm_20_while_less_backward_lstm_20_sub_14backward_lstm_20_while_less_backward_lstm_20_sub_1_0"�
Cbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resourceEbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0"�
Dbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resourceFbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0"�
Bbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resourceDbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0"�
sbackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensorubackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2x
:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp2v
9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp2z
;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp: 
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
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_2417071
inputs_0>
+lstm_cell_62_matmul_readvariableop_resource:	�@
-lstm_cell_62_matmul_1_readvariableop_resource:	2�;
,lstm_cell_62_biasadd_readvariableop_resource:	�
identity��#lstm_cell_62/BiasAdd/ReadVariableOp�"lstm_cell_62/MatMul/ReadVariableOp�$lstm_cell_62/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_62/MatMul/ReadVariableOpReadVariableOp+lstm_cell_62_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_62/MatMul/ReadVariableOp�
lstm_cell_62/MatMulMatMulstrided_slice_2:output:0*lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/MatMul�
$lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_62_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_62/MatMul_1/ReadVariableOp�
lstm_cell_62/MatMul_1MatMulzeros:output:0,lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/MatMul_1�
lstm_cell_62/addAddV2lstm_cell_62/MatMul:product:0lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/add�
#lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_62/BiasAdd/ReadVariableOp�
lstm_cell_62/BiasAddBiasAddlstm_cell_62/add:z:0+lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/BiasAdd~
lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_62/split/split_dim�
lstm_cell_62/splitSplit%lstm_cell_62/split/split_dim:output:0lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_62/split�
lstm_cell_62/SigmoidSigmoidlstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/Sigmoid�
lstm_cell_62/Sigmoid_1Sigmoidlstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_62/Sigmoid_1�
lstm_cell_62/mulMullstm_cell_62/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/mul}
lstm_cell_62/ReluRelulstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_62/Relu�
lstm_cell_62/mul_1Mullstm_cell_62/Sigmoid:y:0lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/mul_1�
lstm_cell_62/add_1AddV2lstm_cell_62/mul:z:0lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/add_1�
lstm_cell_62/Sigmoid_2Sigmoidlstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_62/Sigmoid_2|
lstm_cell_62/Relu_1Relulstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/Relu_1�
lstm_cell_62/mul_2Mullstm_cell_62/Sigmoid_2:y:0!lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_62_matmul_readvariableop_resource-lstm_cell_62_matmul_1_readvariableop_resource,lstm_cell_62_biasadd_readvariableop_resource*
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
while_body_2416987*
condR
while_cond_2416986*K
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
NoOpNoOp$^lstm_cell_62/BiasAdd/ReadVariableOp#^lstm_cell_62/MatMul/ReadVariableOp%^lstm_cell_62/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_62/BiasAdd/ReadVariableOp#lstm_cell_62/BiasAdd/ReadVariableOp2H
"lstm_cell_62/MatMul/ReadVariableOp"lstm_cell_62/MatMul/ReadVariableOp2L
$lstm_cell_62/MatMul_1/ReadVariableOp$lstm_cell_62/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
.__inference_lstm_cell_61_layer_call_fn_2417564

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
I__inference_lstm_cell_61_layer_call_and_return_conditional_losses_24119812
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
�
�
2__inference_backward_lstm_20_layer_call_fn_2416896
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
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_24127622
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
*__inference_dense_20_layer_call_fn_2416215

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
E__inference_dense_20_layer_call_and_return_conditional_losses_24142212
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
�
�
I__inference_lstm_cell_61_layer_call_and_return_conditional_losses_2411835

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
�
�
1__inference_forward_lstm_20_layer_call_fn_2416237
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
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_24119182
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
�F
�
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_2412128

inputs'
lstm_cell_61_2412046:	�'
lstm_cell_61_2412048:	2�#
lstm_cell_61_2412050:	�
identity��$lstm_cell_61/StatefulPartitionedCall�whileD
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
$lstm_cell_61/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_61_2412046lstm_cell_61_2412048lstm_cell_61_2412050*
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
I__inference_lstm_cell_61_layer_call_and_return_conditional_losses_24119812&
$lstm_cell_61/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_61_2412046lstm_cell_61_2412048lstm_cell_61_2412050*
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
while_body_2412059*
condR
while_cond_2412058*K
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
NoOpNoOp%^lstm_cell_61/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_61/StatefulPartitionedCall$lstm_cell_61/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
while_cond_2412480
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2412480___redundant_placeholder05
1while_while_cond_2412480___redundant_placeholder15
1while_while_cond_2412480___redundant_placeholder25
1while_while_cond_2412480___redundant_placeholder3
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
�
1__inference_forward_lstm_20_layer_call_fn_2416270

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
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_24137102
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
�\
�
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_2416874

inputs>
+lstm_cell_61_matmul_readvariableop_resource:	�@
-lstm_cell_61_matmul_1_readvariableop_resource:	2�;
,lstm_cell_61_biasadd_readvariableop_resource:	�
identity��#lstm_cell_61/BiasAdd/ReadVariableOp�"lstm_cell_61/MatMul/ReadVariableOp�$lstm_cell_61/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_61/MatMul/ReadVariableOpReadVariableOp+lstm_cell_61_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_61/MatMul/ReadVariableOp�
lstm_cell_61/MatMulMatMulstrided_slice_2:output:0*lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/MatMul�
$lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_61_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_61/MatMul_1/ReadVariableOp�
lstm_cell_61/MatMul_1MatMulzeros:output:0,lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/MatMul_1�
lstm_cell_61/addAddV2lstm_cell_61/MatMul:product:0lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/add�
#lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_61_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_61/BiasAdd/ReadVariableOp�
lstm_cell_61/BiasAddBiasAddlstm_cell_61/add:z:0+lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/BiasAdd~
lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_61/split/split_dim�
lstm_cell_61/splitSplit%lstm_cell_61/split/split_dim:output:0lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_61/split�
lstm_cell_61/SigmoidSigmoidlstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/Sigmoid�
lstm_cell_61/Sigmoid_1Sigmoidlstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_61/Sigmoid_1�
lstm_cell_61/mulMullstm_cell_61/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/mul}
lstm_cell_61/ReluRelulstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_61/Relu�
lstm_cell_61/mul_1Mullstm_cell_61/Sigmoid:y:0lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/mul_1�
lstm_cell_61/add_1AddV2lstm_cell_61/mul:z:0lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/add_1�
lstm_cell_61/Sigmoid_2Sigmoidlstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_61/Sigmoid_2|
lstm_cell_61/Relu_1Relulstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/Relu_1�
lstm_cell_61/mul_2Mullstm_cell_61/Sigmoid_2:y:0!lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_61_matmul_readvariableop_resource-lstm_cell_61_matmul_1_readvariableop_resource,lstm_cell_61_biasadd_readvariableop_resource*
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
while_body_2416790*
condR
while_cond_2416789*K
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
NoOpNoOp$^lstm_cell_61/BiasAdd/ReadVariableOp#^lstm_cell_61/MatMul/ReadVariableOp%^lstm_cell_61/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_61/BiasAdd/ReadVariableOp#lstm_cell_61/BiasAdd/ReadVariableOp2H
"lstm_cell_61/MatMul/ReadVariableOp"lstm_cell_61/MatMul/ReadVariableOp2L
$lstm_cell_61/MatMul_1/ReadVariableOp$lstm_cell_61/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
while_cond_2413260
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2413260___redundant_placeholder05
1while_while_cond_2413260___redundant_placeholder15
1while_while_cond_2413260___redundant_placeholder25
1while_while_cond_2413260___redundant_placeholder3
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
while_cond_2416336
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2416336___redundant_placeholder05
1while_while_cond_2416336___redundant_placeholder15
1while_while_cond_2416336___redundant_placeholder25
1while_while_cond_2416336___redundant_placeholder3
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
�d
�
#backward_lstm_20_while_body_2414099>
:backward_lstm_20_while_backward_lstm_20_while_loop_counterD
@backward_lstm_20_while_backward_lstm_20_while_maximum_iterations&
"backward_lstm_20_while_placeholder(
$backward_lstm_20_while_placeholder_1(
$backward_lstm_20_while_placeholder_2(
$backward_lstm_20_while_placeholder_3(
$backward_lstm_20_while_placeholder_4=
9backward_lstm_20_while_backward_lstm_20_strided_slice_1_0y
ubackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_08
4backward_lstm_20_while_less_backward_lstm_20_sub_1_0W
Dbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0:	�Y
Fbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0:	2�T
Ebackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0:	�#
backward_lstm_20_while_identity%
!backward_lstm_20_while_identity_1%
!backward_lstm_20_while_identity_2%
!backward_lstm_20_while_identity_3%
!backward_lstm_20_while_identity_4%
!backward_lstm_20_while_identity_5%
!backward_lstm_20_while_identity_6;
7backward_lstm_20_while_backward_lstm_20_strided_slice_1w
sbackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor6
2backward_lstm_20_while_less_backward_lstm_20_sub_1U
Bbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource:	�W
Dbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource:	2�R
Cbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource:	���:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp�9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp�;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp�
Hbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2J
Hbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape�
:backward_lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_20_while_placeholderQbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02<
:backward_lstm_20/while/TensorArrayV2Read/TensorListGetItem�
backward_lstm_20/while/LessLess4backward_lstm_20_while_less_backward_lstm_20_sub_1_0"backward_lstm_20_while_placeholder*
T0*#
_output_shapes
:���������2
backward_lstm_20/while/Less�
9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOpReadVariableOpDbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02;
9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp�
*backward_lstm_20/while/lstm_cell_62/MatMulMatMulAbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:0Abackward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*backward_lstm_20/while/lstm_cell_62/MatMul�
;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOpFbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02=
;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp�
,backward_lstm_20/while/lstm_cell_62/MatMul_1MatMul$backward_lstm_20_while_placeholder_3Cbackward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2.
,backward_lstm_20/while/lstm_cell_62/MatMul_1�
'backward_lstm_20/while/lstm_cell_62/addAddV24backward_lstm_20/while/lstm_cell_62/MatMul:product:06backward_lstm_20/while/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2)
'backward_lstm_20/while/lstm_cell_62/add�
:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOpEbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02<
:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp�
+backward_lstm_20/while/lstm_cell_62/BiasAddBiasAdd+backward_lstm_20/while/lstm_cell_62/add:z:0Bbackward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+backward_lstm_20/while/lstm_cell_62/BiasAdd�
3backward_lstm_20/while/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3backward_lstm_20/while/lstm_cell_62/split/split_dim�
)backward_lstm_20/while/lstm_cell_62/splitSplit<backward_lstm_20/while/lstm_cell_62/split/split_dim:output:04backward_lstm_20/while/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2+
)backward_lstm_20/while/lstm_cell_62/split�
+backward_lstm_20/while/lstm_cell_62/SigmoidSigmoid2backward_lstm_20/while/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22-
+backward_lstm_20/while/lstm_cell_62/Sigmoid�
-backward_lstm_20/while/lstm_cell_62/Sigmoid_1Sigmoid2backward_lstm_20/while/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22/
-backward_lstm_20/while/lstm_cell_62/Sigmoid_1�
'backward_lstm_20/while/lstm_cell_62/mulMul1backward_lstm_20/while/lstm_cell_62/Sigmoid_1:y:0$backward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22)
'backward_lstm_20/while/lstm_cell_62/mul�
(backward_lstm_20/while/lstm_cell_62/ReluRelu2backward_lstm_20/while/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22*
(backward_lstm_20/while/lstm_cell_62/Relu�
)backward_lstm_20/while/lstm_cell_62/mul_1Mul/backward_lstm_20/while/lstm_cell_62/Sigmoid:y:06backward_lstm_20/while/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_20/while/lstm_cell_62/mul_1�
)backward_lstm_20/while/lstm_cell_62/add_1AddV2+backward_lstm_20/while/lstm_cell_62/mul:z:0-backward_lstm_20/while/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_20/while/lstm_cell_62/add_1�
-backward_lstm_20/while/lstm_cell_62/Sigmoid_2Sigmoid2backward_lstm_20/while/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22/
-backward_lstm_20/while/lstm_cell_62/Sigmoid_2�
*backward_lstm_20/while/lstm_cell_62/Relu_1Relu-backward_lstm_20/while/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22,
*backward_lstm_20/while/lstm_cell_62/Relu_1�
)backward_lstm_20/while/lstm_cell_62/mul_2Mul1backward_lstm_20/while/lstm_cell_62/Sigmoid_2:y:08backward_lstm_20/while/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_20/while/lstm_cell_62/mul_2�
backward_lstm_20/while/SelectSelectbackward_lstm_20/while/Less:z:0-backward_lstm_20/while/lstm_cell_62/mul_2:z:0$backward_lstm_20_while_placeholder_2*
T0*'
_output_shapes
:���������22
backward_lstm_20/while/Select�
backward_lstm_20/while/Select_1Selectbackward_lstm_20/while/Less:z:0-backward_lstm_20/while/lstm_cell_62/mul_2:z:0$backward_lstm_20_while_placeholder_3*
T0*'
_output_shapes
:���������22!
backward_lstm_20/while/Select_1�
backward_lstm_20/while/Select_2Selectbackward_lstm_20/while/Less:z:0-backward_lstm_20/while/lstm_cell_62/add_1:z:0$backward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22!
backward_lstm_20/while/Select_2�
;backward_lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_20_while_placeholder_1"backward_lstm_20_while_placeholder&backward_lstm_20/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_20/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_20/while/add/y�
backward_lstm_20/while/addAddV2"backward_lstm_20_while_placeholder%backward_lstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/while/add�
backward_lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_20/while/add_1/y�
backward_lstm_20/while/add_1AddV2:backward_lstm_20_while_backward_lstm_20_while_loop_counter'backward_lstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/while/add_1�
backward_lstm_20/while/IdentityIdentity backward_lstm_20/while/add_1:z:0^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_20/while/Identity�
!backward_lstm_20/while/Identity_1Identity@backward_lstm_20_while_backward_lstm_20_while_maximum_iterations^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_20/while/Identity_1�
!backward_lstm_20/while/Identity_2Identitybackward_lstm_20/while/add:z:0^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_20/while/Identity_2�
!backward_lstm_20/while/Identity_3IdentityKbackward_lstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_20/while/Identity_3�
!backward_lstm_20/while/Identity_4Identity&backward_lstm_20/while/Select:output:0^backward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/while/Identity_4�
!backward_lstm_20/while/Identity_5Identity(backward_lstm_20/while/Select_1:output:0^backward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/while/Identity_5�
!backward_lstm_20/while/Identity_6Identity(backward_lstm_20/while/Select_2:output:0^backward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/while/Identity_6�
backward_lstm_20/while/NoOpNoOp;^backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp:^backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp<^backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_20/while/NoOp"t
7backward_lstm_20_while_backward_lstm_20_strided_slice_19backward_lstm_20_while_backward_lstm_20_strided_slice_1_0"K
backward_lstm_20_while_identity(backward_lstm_20/while/Identity:output:0"O
!backward_lstm_20_while_identity_1*backward_lstm_20/while/Identity_1:output:0"O
!backward_lstm_20_while_identity_2*backward_lstm_20/while/Identity_2:output:0"O
!backward_lstm_20_while_identity_3*backward_lstm_20/while/Identity_3:output:0"O
!backward_lstm_20_while_identity_4*backward_lstm_20/while/Identity_4:output:0"O
!backward_lstm_20_while_identity_5*backward_lstm_20/while/Identity_5:output:0"O
!backward_lstm_20_while_identity_6*backward_lstm_20/while/Identity_6:output:0"j
2backward_lstm_20_while_less_backward_lstm_20_sub_14backward_lstm_20_while_less_backward_lstm_20_sub_1_0"�
Cbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resourceEbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0"�
Dbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resourceFbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0"�
Bbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resourceDbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0"�
sbackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensorubackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2x
:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp2v
9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp2z
;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp: 
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
#backward_lstm_20_while_cond_2415750>
:backward_lstm_20_while_backward_lstm_20_while_loop_counterD
@backward_lstm_20_while_backward_lstm_20_while_maximum_iterations&
"backward_lstm_20_while_placeholder(
$backward_lstm_20_while_placeholder_1(
$backward_lstm_20_while_placeholder_2(
$backward_lstm_20_while_placeholder_3(
$backward_lstm_20_while_placeholder_4@
<backward_lstm_20_while_less_backward_lstm_20_strided_slice_1W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2415750___redundant_placeholder0W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2415750___redundant_placeholder1W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2415750___redundant_placeholder2W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2415750___redundant_placeholder3W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2415750___redundant_placeholder4#
backward_lstm_20_while_identity
�
backward_lstm_20/while/LessLess"backward_lstm_20_while_placeholder<backward_lstm_20_while_less_backward_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_20/while/Less�
backward_lstm_20/while/IdentityIdentitybackward_lstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_20/while/Identity"K
backward_lstm_20_while_identity(backward_lstm_20/while/Identity:output:0*(
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
�b
�
"forward_lstm_20_while_body_2415930<
8forward_lstm_20_while_forward_lstm_20_while_loop_counterB
>forward_lstm_20_while_forward_lstm_20_while_maximum_iterations%
!forward_lstm_20_while_placeholder'
#forward_lstm_20_while_placeholder_1'
#forward_lstm_20_while_placeholder_2'
#forward_lstm_20_while_placeholder_3'
#forward_lstm_20_while_placeholder_4;
7forward_lstm_20_while_forward_lstm_20_strided_slice_1_0w
sforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_08
4forward_lstm_20_while_greater_forward_lstm_20_cast_0V
Cforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0:	�X
Eforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0:	2�S
Dforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0:	�"
forward_lstm_20_while_identity$
 forward_lstm_20_while_identity_1$
 forward_lstm_20_while_identity_2$
 forward_lstm_20_while_identity_3$
 forward_lstm_20_while_identity_4$
 forward_lstm_20_while_identity_5$
 forward_lstm_20_while_identity_69
5forward_lstm_20_while_forward_lstm_20_strided_slice_1u
qforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor6
2forward_lstm_20_while_greater_forward_lstm_20_castT
Aforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource:	�V
Cforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource:	2�Q
Bforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource:	���9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp�8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp�:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp�
Gforward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2I
Gforward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape�
9forward_lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_20_while_placeholderPforward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02;
9forward_lstm_20/while/TensorArrayV2Read/TensorListGetItem�
forward_lstm_20/while/GreaterGreater4forward_lstm_20_while_greater_forward_lstm_20_cast_0!forward_lstm_20_while_placeholder*
T0*#
_output_shapes
:���������2
forward_lstm_20/while/Greater�
8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOpReadVariableOpCforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02:
8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp�
)forward_lstm_20/while/lstm_cell_61/MatMulMatMul@forward_lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:0@forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)forward_lstm_20/while/lstm_cell_61/MatMul�
:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOpEforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02<
:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp�
+forward_lstm_20/while/lstm_cell_61/MatMul_1MatMul#forward_lstm_20_while_placeholder_3Bforward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+forward_lstm_20/while/lstm_cell_61/MatMul_1�
&forward_lstm_20/while/lstm_cell_61/addAddV23forward_lstm_20/while/lstm_cell_61/MatMul:product:05forward_lstm_20/while/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&forward_lstm_20/while/lstm_cell_61/add�
9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOpDforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02;
9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp�
*forward_lstm_20/while/lstm_cell_61/BiasAddBiasAdd*forward_lstm_20/while/lstm_cell_61/add:z:0Aforward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*forward_lstm_20/while/lstm_cell_61/BiasAdd�
2forward_lstm_20/while/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2forward_lstm_20/while/lstm_cell_61/split/split_dim�
(forward_lstm_20/while/lstm_cell_61/splitSplit;forward_lstm_20/while/lstm_cell_61/split/split_dim:output:03forward_lstm_20/while/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2*
(forward_lstm_20/while/lstm_cell_61/split�
*forward_lstm_20/while/lstm_cell_61/SigmoidSigmoid1forward_lstm_20/while/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22,
*forward_lstm_20/while/lstm_cell_61/Sigmoid�
,forward_lstm_20/while/lstm_cell_61/Sigmoid_1Sigmoid1forward_lstm_20/while/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22.
,forward_lstm_20/while/lstm_cell_61/Sigmoid_1�
&forward_lstm_20/while/lstm_cell_61/mulMul0forward_lstm_20/while/lstm_cell_61/Sigmoid_1:y:0#forward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22(
&forward_lstm_20/while/lstm_cell_61/mul�
'forward_lstm_20/while/lstm_cell_61/ReluRelu1forward_lstm_20/while/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22)
'forward_lstm_20/while/lstm_cell_61/Relu�
(forward_lstm_20/while/lstm_cell_61/mul_1Mul.forward_lstm_20/while/lstm_cell_61/Sigmoid:y:05forward_lstm_20/while/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_20/while/lstm_cell_61/mul_1�
(forward_lstm_20/while/lstm_cell_61/add_1AddV2*forward_lstm_20/while/lstm_cell_61/mul:z:0,forward_lstm_20/while/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_20/while/lstm_cell_61/add_1�
,forward_lstm_20/while/lstm_cell_61/Sigmoid_2Sigmoid1forward_lstm_20/while/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22.
,forward_lstm_20/while/lstm_cell_61/Sigmoid_2�
)forward_lstm_20/while/lstm_cell_61/Relu_1Relu,forward_lstm_20/while/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22+
)forward_lstm_20/while/lstm_cell_61/Relu_1�
(forward_lstm_20/while/lstm_cell_61/mul_2Mul0forward_lstm_20/while/lstm_cell_61/Sigmoid_2:y:07forward_lstm_20/while/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_20/while/lstm_cell_61/mul_2�
forward_lstm_20/while/SelectSelect!forward_lstm_20/while/Greater:z:0,forward_lstm_20/while/lstm_cell_61/mul_2:z:0#forward_lstm_20_while_placeholder_2*
T0*'
_output_shapes
:���������22
forward_lstm_20/while/Select�
forward_lstm_20/while/Select_1Select!forward_lstm_20/while/Greater:z:0,forward_lstm_20/while/lstm_cell_61/mul_2:z:0#forward_lstm_20_while_placeholder_3*
T0*'
_output_shapes
:���������22 
forward_lstm_20/while/Select_1�
forward_lstm_20/while/Select_2Select!forward_lstm_20/while/Greater:z:0,forward_lstm_20/while/lstm_cell_61/add_1:z:0#forward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22 
forward_lstm_20/while/Select_2�
:forward_lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_20_while_placeholder_1!forward_lstm_20_while_placeholder%forward_lstm_20/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_20/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_20/while/add/y�
forward_lstm_20/while/addAddV2!forward_lstm_20_while_placeholder$forward_lstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/while/add�
forward_lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_20/while/add_1/y�
forward_lstm_20/while/add_1AddV28forward_lstm_20_while_forward_lstm_20_while_loop_counter&forward_lstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/while/add_1�
forward_lstm_20/while/IdentityIdentityforward_lstm_20/while/add_1:z:0^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_20/while/Identity�
 forward_lstm_20/while/Identity_1Identity>forward_lstm_20_while_forward_lstm_20_while_maximum_iterations^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_20/while/Identity_1�
 forward_lstm_20/while/Identity_2Identityforward_lstm_20/while/add:z:0^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_20/while/Identity_2�
 forward_lstm_20/while/Identity_3IdentityJforward_lstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_20/while/Identity_3�
 forward_lstm_20/while/Identity_4Identity%forward_lstm_20/while/Select:output:0^forward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/while/Identity_4�
 forward_lstm_20/while/Identity_5Identity'forward_lstm_20/while/Select_1:output:0^forward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/while/Identity_5�
 forward_lstm_20/while/Identity_6Identity'forward_lstm_20/while/Select_2:output:0^forward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/while/Identity_6�
forward_lstm_20/while/NoOpNoOp:^forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp9^forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp;^forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_20/while/NoOp"p
5forward_lstm_20_while_forward_lstm_20_strided_slice_17forward_lstm_20_while_forward_lstm_20_strided_slice_1_0"j
2forward_lstm_20_while_greater_forward_lstm_20_cast4forward_lstm_20_while_greater_forward_lstm_20_cast_0"I
forward_lstm_20_while_identity'forward_lstm_20/while/Identity:output:0"M
 forward_lstm_20_while_identity_1)forward_lstm_20/while/Identity_1:output:0"M
 forward_lstm_20_while_identity_2)forward_lstm_20/while/Identity_2:output:0"M
 forward_lstm_20_while_identity_3)forward_lstm_20/while/Identity_3:output:0"M
 forward_lstm_20_while_identity_4)forward_lstm_20/while/Identity_4:output:0"M
 forward_lstm_20_while_identity_5)forward_lstm_20/while/Identity_5:output:0"M
 forward_lstm_20_while_identity_6)forward_lstm_20/while/Identity_6:output:0"�
Bforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resourceDforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0"�
Cforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resourceEforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0"�
Aforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resourceCforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0"�
qforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensorsforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2v
9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp2t
8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp2x
:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp: 
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
I__inference_lstm_cell_62_layer_call_and_return_conditional_losses_2417726

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
�d
�
#backward_lstm_20_while_body_2416109>
:backward_lstm_20_while_backward_lstm_20_while_loop_counterD
@backward_lstm_20_while_backward_lstm_20_while_maximum_iterations&
"backward_lstm_20_while_placeholder(
$backward_lstm_20_while_placeholder_1(
$backward_lstm_20_while_placeholder_2(
$backward_lstm_20_while_placeholder_3(
$backward_lstm_20_while_placeholder_4=
9backward_lstm_20_while_backward_lstm_20_strided_slice_1_0y
ubackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_08
4backward_lstm_20_while_less_backward_lstm_20_sub_1_0W
Dbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0:	�Y
Fbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0:	2�T
Ebackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0:	�#
backward_lstm_20_while_identity%
!backward_lstm_20_while_identity_1%
!backward_lstm_20_while_identity_2%
!backward_lstm_20_while_identity_3%
!backward_lstm_20_while_identity_4%
!backward_lstm_20_while_identity_5%
!backward_lstm_20_while_identity_6;
7backward_lstm_20_while_backward_lstm_20_strided_slice_1w
sbackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor6
2backward_lstm_20_while_less_backward_lstm_20_sub_1U
Bbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource:	�W
Dbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource:	2�R
Cbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource:	���:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp�9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp�;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp�
Hbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2J
Hbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape�
:backward_lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_20_while_placeholderQbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02<
:backward_lstm_20/while/TensorArrayV2Read/TensorListGetItem�
backward_lstm_20/while/LessLess4backward_lstm_20_while_less_backward_lstm_20_sub_1_0"backward_lstm_20_while_placeholder*
T0*#
_output_shapes
:���������2
backward_lstm_20/while/Less�
9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOpReadVariableOpDbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02;
9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp�
*backward_lstm_20/while/lstm_cell_62/MatMulMatMulAbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:0Abackward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*backward_lstm_20/while/lstm_cell_62/MatMul�
;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOpFbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02=
;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp�
,backward_lstm_20/while/lstm_cell_62/MatMul_1MatMul$backward_lstm_20_while_placeholder_3Cbackward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2.
,backward_lstm_20/while/lstm_cell_62/MatMul_1�
'backward_lstm_20/while/lstm_cell_62/addAddV24backward_lstm_20/while/lstm_cell_62/MatMul:product:06backward_lstm_20/while/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2)
'backward_lstm_20/while/lstm_cell_62/add�
:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOpEbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02<
:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp�
+backward_lstm_20/while/lstm_cell_62/BiasAddBiasAdd+backward_lstm_20/while/lstm_cell_62/add:z:0Bbackward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+backward_lstm_20/while/lstm_cell_62/BiasAdd�
3backward_lstm_20/while/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3backward_lstm_20/while/lstm_cell_62/split/split_dim�
)backward_lstm_20/while/lstm_cell_62/splitSplit<backward_lstm_20/while/lstm_cell_62/split/split_dim:output:04backward_lstm_20/while/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2+
)backward_lstm_20/while/lstm_cell_62/split�
+backward_lstm_20/while/lstm_cell_62/SigmoidSigmoid2backward_lstm_20/while/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22-
+backward_lstm_20/while/lstm_cell_62/Sigmoid�
-backward_lstm_20/while/lstm_cell_62/Sigmoid_1Sigmoid2backward_lstm_20/while/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22/
-backward_lstm_20/while/lstm_cell_62/Sigmoid_1�
'backward_lstm_20/while/lstm_cell_62/mulMul1backward_lstm_20/while/lstm_cell_62/Sigmoid_1:y:0$backward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22)
'backward_lstm_20/while/lstm_cell_62/mul�
(backward_lstm_20/while/lstm_cell_62/ReluRelu2backward_lstm_20/while/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22*
(backward_lstm_20/while/lstm_cell_62/Relu�
)backward_lstm_20/while/lstm_cell_62/mul_1Mul/backward_lstm_20/while/lstm_cell_62/Sigmoid:y:06backward_lstm_20/while/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_20/while/lstm_cell_62/mul_1�
)backward_lstm_20/while/lstm_cell_62/add_1AddV2+backward_lstm_20/while/lstm_cell_62/mul:z:0-backward_lstm_20/while/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_20/while/lstm_cell_62/add_1�
-backward_lstm_20/while/lstm_cell_62/Sigmoid_2Sigmoid2backward_lstm_20/while/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22/
-backward_lstm_20/while/lstm_cell_62/Sigmoid_2�
*backward_lstm_20/while/lstm_cell_62/Relu_1Relu-backward_lstm_20/while/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22,
*backward_lstm_20/while/lstm_cell_62/Relu_1�
)backward_lstm_20/while/lstm_cell_62/mul_2Mul1backward_lstm_20/while/lstm_cell_62/Sigmoid_2:y:08backward_lstm_20/while/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_20/while/lstm_cell_62/mul_2�
backward_lstm_20/while/SelectSelectbackward_lstm_20/while/Less:z:0-backward_lstm_20/while/lstm_cell_62/mul_2:z:0$backward_lstm_20_while_placeholder_2*
T0*'
_output_shapes
:���������22
backward_lstm_20/while/Select�
backward_lstm_20/while/Select_1Selectbackward_lstm_20/while/Less:z:0-backward_lstm_20/while/lstm_cell_62/mul_2:z:0$backward_lstm_20_while_placeholder_3*
T0*'
_output_shapes
:���������22!
backward_lstm_20/while/Select_1�
backward_lstm_20/while/Select_2Selectbackward_lstm_20/while/Less:z:0-backward_lstm_20/while/lstm_cell_62/add_1:z:0$backward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22!
backward_lstm_20/while/Select_2�
;backward_lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_20_while_placeholder_1"backward_lstm_20_while_placeholder&backward_lstm_20/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_20/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_20/while/add/y�
backward_lstm_20/while/addAddV2"backward_lstm_20_while_placeholder%backward_lstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/while/add�
backward_lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_20/while/add_1/y�
backward_lstm_20/while/add_1AddV2:backward_lstm_20_while_backward_lstm_20_while_loop_counter'backward_lstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/while/add_1�
backward_lstm_20/while/IdentityIdentity backward_lstm_20/while/add_1:z:0^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_20/while/Identity�
!backward_lstm_20/while/Identity_1Identity@backward_lstm_20_while_backward_lstm_20_while_maximum_iterations^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_20/while/Identity_1�
!backward_lstm_20/while/Identity_2Identitybackward_lstm_20/while/add:z:0^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_20/while/Identity_2�
!backward_lstm_20/while/Identity_3IdentityKbackward_lstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_20/while/Identity_3�
!backward_lstm_20/while/Identity_4Identity&backward_lstm_20/while/Select:output:0^backward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/while/Identity_4�
!backward_lstm_20/while/Identity_5Identity(backward_lstm_20/while/Select_1:output:0^backward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/while/Identity_5�
!backward_lstm_20/while/Identity_6Identity(backward_lstm_20/while/Select_2:output:0^backward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/while/Identity_6�
backward_lstm_20/while/NoOpNoOp;^backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp:^backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp<^backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_20/while/NoOp"t
7backward_lstm_20_while_backward_lstm_20_strided_slice_19backward_lstm_20_while_backward_lstm_20_strided_slice_1_0"K
backward_lstm_20_while_identity(backward_lstm_20/while/Identity:output:0"O
!backward_lstm_20_while_identity_1*backward_lstm_20/while/Identity_1:output:0"O
!backward_lstm_20_while_identity_2*backward_lstm_20/while/Identity_2:output:0"O
!backward_lstm_20_while_identity_3*backward_lstm_20/while/Identity_3:output:0"O
!backward_lstm_20_while_identity_4*backward_lstm_20/while/Identity_4:output:0"O
!backward_lstm_20_while_identity_5*backward_lstm_20/while/Identity_5:output:0"O
!backward_lstm_20_while_identity_6*backward_lstm_20/while/Identity_6:output:0"j
2backward_lstm_20_while_less_backward_lstm_20_sub_14backward_lstm_20_while_less_backward_lstm_20_sub_1_0"�
Cbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resourceEbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0"�
Dbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resourceFbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0"�
Bbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resourceDbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0"�
sbackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensorubackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2x
:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp2v
9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp2z
;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp: 
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
�\
�
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_2413185

inputs>
+lstm_cell_61_matmul_readvariableop_resource:	�@
-lstm_cell_61_matmul_1_readvariableop_resource:	2�;
,lstm_cell_61_biasadd_readvariableop_resource:	�
identity��#lstm_cell_61/BiasAdd/ReadVariableOp�"lstm_cell_61/MatMul/ReadVariableOp�$lstm_cell_61/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_61/MatMul/ReadVariableOpReadVariableOp+lstm_cell_61_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_61/MatMul/ReadVariableOp�
lstm_cell_61/MatMulMatMulstrided_slice_2:output:0*lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/MatMul�
$lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_61_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_61/MatMul_1/ReadVariableOp�
lstm_cell_61/MatMul_1MatMulzeros:output:0,lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/MatMul_1�
lstm_cell_61/addAddV2lstm_cell_61/MatMul:product:0lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/add�
#lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_61_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_61/BiasAdd/ReadVariableOp�
lstm_cell_61/BiasAddBiasAddlstm_cell_61/add:z:0+lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/BiasAdd~
lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_61/split/split_dim�
lstm_cell_61/splitSplit%lstm_cell_61/split/split_dim:output:0lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_61/split�
lstm_cell_61/SigmoidSigmoidlstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/Sigmoid�
lstm_cell_61/Sigmoid_1Sigmoidlstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_61/Sigmoid_1�
lstm_cell_61/mulMullstm_cell_61/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/mul}
lstm_cell_61/ReluRelulstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_61/Relu�
lstm_cell_61/mul_1Mullstm_cell_61/Sigmoid:y:0lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/mul_1�
lstm_cell_61/add_1AddV2lstm_cell_61/mul:z:0lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/add_1�
lstm_cell_61/Sigmoid_2Sigmoidlstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_61/Sigmoid_2|
lstm_cell_61/Relu_1Relulstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/Relu_1�
lstm_cell_61/mul_2Mullstm_cell_61/Sigmoid_2:y:0!lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_61_matmul_readvariableop_resource-lstm_cell_61_matmul_1_readvariableop_resource,lstm_cell_61_biasadd_readvariableop_resource*
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
while_body_2413101*
condR
while_cond_2413100*K
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
NoOpNoOp$^lstm_cell_61/BiasAdd/ReadVariableOp#^lstm_cell_61/MatMul/ReadVariableOp%^lstm_cell_61/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_61/BiasAdd/ReadVariableOp#lstm_cell_61/BiasAdd/ReadVariableOp2H
"lstm_cell_61/MatMul/ReadVariableOp"lstm_cell_61/MatMul/ReadVariableOp2L
$lstm_cell_61/MatMul_1/ReadVariableOp$lstm_cell_61/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
while_cond_2416986
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2416986___redundant_placeholder05
1while_while_cond_2416986___redundant_placeholder15
1while_while_cond_2416986___redundant_placeholder25
1while_while_cond_2416986___redundant_placeholder3
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
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_2414196

inputs
inputs_1	N
;forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource:	�P
=forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource:	2�K
<forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource:	�O
<backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource:	�Q
>backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource:	2�L
=backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource:	�
identity��4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp�3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp�5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp�backward_lstm_20/while�3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp�2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp�4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp�forward_lstm_20/while�
$forward_lstm_20/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$forward_lstm_20/RaggedToTensor/zeros�
$forward_lstm_20/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2&
$forward_lstm_20/RaggedToTensor/Const�
3forward_lstm_20/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-forward_lstm_20/RaggedToTensor/Const:output:0inputs-forward_lstm_20/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3forward_lstm_20/RaggedToTensor/RaggedTensorToTensor�
:forward_lstm_20/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack�
<forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1�
<forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2�
4forward_lstm_20/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cforward_lstm_20/RaggedNestedRowLengths/strided_slice/stack:output:0Eforward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1:output:0Eforward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask26
4forward_lstm_20/RaggedNestedRowLengths/strided_slice�
<forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack�
>forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2@
>forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1�
>forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2�
6forward_lstm_20/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Eforward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gforward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gforward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask28
6forward_lstm_20/RaggedNestedRowLengths/strided_slice_1�
*forward_lstm_20/RaggedNestedRowLengths/subSub=forward_lstm_20/RaggedNestedRowLengths/strided_slice:output:0?forward_lstm_20/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2,
*forward_lstm_20/RaggedNestedRowLengths/sub�
forward_lstm_20/CastCast.forward_lstm_20/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
forward_lstm_20/Cast�
forward_lstm_20/ShapeShape<forward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_20/Shape�
#forward_lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_20/strided_slice/stack�
%forward_lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_20/strided_slice/stack_1�
%forward_lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_20/strided_slice/stack_2�
forward_lstm_20/strided_sliceStridedSliceforward_lstm_20/Shape:output:0,forward_lstm_20/strided_slice/stack:output:0.forward_lstm_20/strided_slice/stack_1:output:0.forward_lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_20/strided_slice|
forward_lstm_20/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_20/zeros/mul/y�
forward_lstm_20/zeros/mulMul&forward_lstm_20/strided_slice:output:0$forward_lstm_20/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros/mul
forward_lstm_20/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_20/zeros/Less/y�
forward_lstm_20/zeros/LessLessforward_lstm_20/zeros/mul:z:0%forward_lstm_20/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros/Less�
forward_lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_20/zeros/packed/1�
forward_lstm_20/zeros/packedPack&forward_lstm_20/strided_slice:output:0'forward_lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_20/zeros/packed�
forward_lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_20/zeros/Const�
forward_lstm_20/zerosFill%forward_lstm_20/zeros/packed:output:0$forward_lstm_20/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_20/zeros�
forward_lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_20/zeros_1/mul/y�
forward_lstm_20/zeros_1/mulMul&forward_lstm_20/strided_slice:output:0&forward_lstm_20/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros_1/mul�
forward_lstm_20/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
forward_lstm_20/zeros_1/Less/y�
forward_lstm_20/zeros_1/LessLessforward_lstm_20/zeros_1/mul:z:0'forward_lstm_20/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros_1/Less�
 forward_lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_20/zeros_1/packed/1�
forward_lstm_20/zeros_1/packedPack&forward_lstm_20/strided_slice:output:0)forward_lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_20/zeros_1/packed�
forward_lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_20/zeros_1/Const�
forward_lstm_20/zeros_1Fill'forward_lstm_20/zeros_1/packed:output:0&forward_lstm_20/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_20/zeros_1�
forward_lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_20/transpose/perm�
forward_lstm_20/transpose	Transpose<forward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0'forward_lstm_20/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
forward_lstm_20/transpose
forward_lstm_20/Shape_1Shapeforward_lstm_20/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_20/Shape_1�
%forward_lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_20/strided_slice_1/stack�
'forward_lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_1/stack_1�
'forward_lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_1/stack_2�
forward_lstm_20/strided_slice_1StridedSlice forward_lstm_20/Shape_1:output:0.forward_lstm_20/strided_slice_1/stack:output:00forward_lstm_20/strided_slice_1/stack_1:output:00forward_lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_20/strided_slice_1�
+forward_lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+forward_lstm_20/TensorArrayV2/element_shape�
forward_lstm_20/TensorArrayV2TensorListReserve4forward_lstm_20/TensorArrayV2/element_shape:output:0(forward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_20/TensorArrayV2�
Eforward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2G
Eforward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape�
7forward_lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_20/transpose:y:0Nforward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_20/TensorArrayUnstack/TensorListFromTensor�
%forward_lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_20/strided_slice_2/stack�
'forward_lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_2/stack_1�
'forward_lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_2/stack_2�
forward_lstm_20/strided_slice_2StridedSliceforward_lstm_20/transpose:y:0.forward_lstm_20/strided_slice_2/stack:output:00forward_lstm_20/strided_slice_2/stack_1:output:00forward_lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2!
forward_lstm_20/strided_slice_2�
2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOpReadVariableOp;forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp�
#forward_lstm_20/lstm_cell_61/MatMulMatMul(forward_lstm_20/strided_slice_2:output:0:forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#forward_lstm_20/lstm_cell_61/MatMul�
4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOp=forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype026
4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp�
%forward_lstm_20/lstm_cell_61/MatMul_1MatMulforward_lstm_20/zeros:output:0<forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%forward_lstm_20/lstm_cell_61/MatMul_1�
 forward_lstm_20/lstm_cell_61/addAddV2-forward_lstm_20/lstm_cell_61/MatMul:product:0/forward_lstm_20/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2"
 forward_lstm_20/lstm_cell_61/add�
3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOp<forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp�
$forward_lstm_20/lstm_cell_61/BiasAddBiasAdd$forward_lstm_20/lstm_cell_61/add:z:0;forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$forward_lstm_20/lstm_cell_61/BiasAdd�
,forward_lstm_20/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,forward_lstm_20/lstm_cell_61/split/split_dim�
"forward_lstm_20/lstm_cell_61/splitSplit5forward_lstm_20/lstm_cell_61/split/split_dim:output:0-forward_lstm_20/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2$
"forward_lstm_20/lstm_cell_61/split�
$forward_lstm_20/lstm_cell_61/SigmoidSigmoid+forward_lstm_20/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22&
$forward_lstm_20/lstm_cell_61/Sigmoid�
&forward_lstm_20/lstm_cell_61/Sigmoid_1Sigmoid+forward_lstm_20/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22(
&forward_lstm_20/lstm_cell_61/Sigmoid_1�
 forward_lstm_20/lstm_cell_61/mulMul*forward_lstm_20/lstm_cell_61/Sigmoid_1:y:0 forward_lstm_20/zeros_1:output:0*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/lstm_cell_61/mul�
!forward_lstm_20/lstm_cell_61/ReluRelu+forward_lstm_20/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22#
!forward_lstm_20/lstm_cell_61/Relu�
"forward_lstm_20/lstm_cell_61/mul_1Mul(forward_lstm_20/lstm_cell_61/Sigmoid:y:0/forward_lstm_20/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_20/lstm_cell_61/mul_1�
"forward_lstm_20/lstm_cell_61/add_1AddV2$forward_lstm_20/lstm_cell_61/mul:z:0&forward_lstm_20/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_20/lstm_cell_61/add_1�
&forward_lstm_20/lstm_cell_61/Sigmoid_2Sigmoid+forward_lstm_20/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22(
&forward_lstm_20/lstm_cell_61/Sigmoid_2�
#forward_lstm_20/lstm_cell_61/Relu_1Relu&forward_lstm_20/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22%
#forward_lstm_20/lstm_cell_61/Relu_1�
"forward_lstm_20/lstm_cell_61/mul_2Mul*forward_lstm_20/lstm_cell_61/Sigmoid_2:y:01forward_lstm_20/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_20/lstm_cell_61/mul_2�
-forward_lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2/
-forward_lstm_20/TensorArrayV2_1/element_shape�
forward_lstm_20/TensorArrayV2_1TensorListReserve6forward_lstm_20/TensorArrayV2_1/element_shape:output:0(forward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_20/TensorArrayV2_1n
forward_lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_20/time�
forward_lstm_20/zeros_like	ZerosLike&forward_lstm_20/lstm_cell_61/mul_2:z:0*
T0*'
_output_shapes
:���������22
forward_lstm_20/zeros_like�
(forward_lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(forward_lstm_20/while/maximum_iterations�
"forward_lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_20/while/loop_counter�
forward_lstm_20/whileWhile+forward_lstm_20/while/loop_counter:output:01forward_lstm_20/while/maximum_iterations:output:0forward_lstm_20/time:output:0(forward_lstm_20/TensorArrayV2_1:handle:0forward_lstm_20/zeros_like:y:0forward_lstm_20/zeros:output:0 forward_lstm_20/zeros_1:output:0(forward_lstm_20/strided_slice_1:output:0Gforward_lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_20/Cast:y:0;forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource=forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource<forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource*
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
"forward_lstm_20_while_body_2413920*.
cond&R$
"forward_lstm_20_while_cond_2413919*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
forward_lstm_20/while�
@forward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2B
@forward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape�
2forward_lstm_20/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_20/while:output:3Iforward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype024
2forward_lstm_20/TensorArrayV2Stack/TensorListStack�
%forward_lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2'
%forward_lstm_20/strided_slice_3/stack�
'forward_lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_20/strided_slice_3/stack_1�
'forward_lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_3/stack_2�
forward_lstm_20/strided_slice_3StridedSlice;forward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_20/strided_slice_3/stack:output:00forward_lstm_20/strided_slice_3/stack_1:output:00forward_lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2!
forward_lstm_20/strided_slice_3�
 forward_lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_20/transpose_1/perm�
forward_lstm_20/transpose_1	Transpose;forward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_20/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
forward_lstm_20/transpose_1�
forward_lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_20/runtime�
%backward_lstm_20/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%backward_lstm_20/RaggedToTensor/zeros�
%backward_lstm_20/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2'
%backward_lstm_20/RaggedToTensor/Const�
4backward_lstm_20/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.backward_lstm_20/RaggedToTensor/Const:output:0inputs.backward_lstm_20/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4backward_lstm_20/RaggedToTensor/RaggedTensorToTensor�
;backward_lstm_20/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack�
=backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1�
=backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2�
5backward_lstm_20/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dbackward_lstm_20/RaggedNestedRowLengths/strided_slice/stack:output:0Fbackward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fbackward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask27
5backward_lstm_20/RaggedNestedRowLengths/strided_slice�
=backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack�
?backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2A
?backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1�
?backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2�
7backward_lstm_20/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fbackward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hbackward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hbackward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask29
7backward_lstm_20/RaggedNestedRowLengths/strided_slice_1�
+backward_lstm_20/RaggedNestedRowLengths/subSub>backward_lstm_20/RaggedNestedRowLengths/strided_slice:output:0@backward_lstm_20/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2-
+backward_lstm_20/RaggedNestedRowLengths/sub�
backward_lstm_20/CastCast/backward_lstm_20/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
backward_lstm_20/Cast�
backward_lstm_20/ShapeShape=backward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_20/Shape�
$backward_lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_20/strided_slice/stack�
&backward_lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_20/strided_slice/stack_1�
&backward_lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_20/strided_slice/stack_2�
backward_lstm_20/strided_sliceStridedSlicebackward_lstm_20/Shape:output:0-backward_lstm_20/strided_slice/stack:output:0/backward_lstm_20/strided_slice/stack_1:output:0/backward_lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_20/strided_slice~
backward_lstm_20/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_20/zeros/mul/y�
backward_lstm_20/zeros/mulMul'backward_lstm_20/strided_slice:output:0%backward_lstm_20/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros/mul�
backward_lstm_20/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
backward_lstm_20/zeros/Less/y�
backward_lstm_20/zeros/LessLessbackward_lstm_20/zeros/mul:z:0&backward_lstm_20/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros/Less�
backward_lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_20/zeros/packed/1�
backward_lstm_20/zeros/packedPack'backward_lstm_20/strided_slice:output:0(backward_lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_20/zeros/packed�
backward_lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_20/zeros/Const�
backward_lstm_20/zerosFill&backward_lstm_20/zeros/packed:output:0%backward_lstm_20/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_20/zeros�
backward_lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_20/zeros_1/mul/y�
backward_lstm_20/zeros_1/mulMul'backward_lstm_20/strided_slice:output:0'backward_lstm_20/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros_1/mul�
backward_lstm_20/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2!
backward_lstm_20/zeros_1/Less/y�
backward_lstm_20/zeros_1/LessLess backward_lstm_20/zeros_1/mul:z:0(backward_lstm_20/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros_1/Less�
!backward_lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_20/zeros_1/packed/1�
backward_lstm_20/zeros_1/packedPack'backward_lstm_20/strided_slice:output:0*backward_lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_20/zeros_1/packed�
backward_lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_20/zeros_1/Const�
backward_lstm_20/zeros_1Fill(backward_lstm_20/zeros_1/packed:output:0'backward_lstm_20/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_20/zeros_1�
backward_lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_20/transpose/perm�
backward_lstm_20/transpose	Transpose=backward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0(backward_lstm_20/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_20/transpose�
backward_lstm_20/Shape_1Shapebackward_lstm_20/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_20/Shape_1�
&backward_lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_20/strided_slice_1/stack�
(backward_lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_1/stack_1�
(backward_lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_1/stack_2�
 backward_lstm_20/strided_slice_1StridedSlice!backward_lstm_20/Shape_1:output:0/backward_lstm_20/strided_slice_1/stack:output:01backward_lstm_20/strided_slice_1/stack_1:output:01backward_lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_20/strided_slice_1�
,backward_lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2.
,backward_lstm_20/TensorArrayV2/element_shape�
backward_lstm_20/TensorArrayV2TensorListReserve5backward_lstm_20/TensorArrayV2/element_shape:output:0)backward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_20/TensorArrayV2�
backward_lstm_20/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_20/ReverseV2/axis�
backward_lstm_20/ReverseV2	ReverseV2backward_lstm_20/transpose:y:0(backward_lstm_20/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_20/ReverseV2�
Fbackward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2H
Fbackward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape�
8backward_lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_20/ReverseV2:output:0Obackward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_20/TensorArrayUnstack/TensorListFromTensor�
&backward_lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_20/strided_slice_2/stack�
(backward_lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_2/stack_1�
(backward_lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_2/stack_2�
 backward_lstm_20/strided_slice_2StridedSlicebackward_lstm_20/transpose:y:0/backward_lstm_20/strided_slice_2/stack:output:01backward_lstm_20/strided_slice_2/stack_1:output:01backward_lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2"
 backward_lstm_20/strided_slice_2�
3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOpReadVariableOp<backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype025
3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp�
$backward_lstm_20/lstm_cell_62/MatMulMatMul)backward_lstm_20/strided_slice_2:output:0;backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$backward_lstm_20/lstm_cell_62/MatMul�
5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOp>backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype027
5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp�
&backward_lstm_20/lstm_cell_62/MatMul_1MatMulbackward_lstm_20/zeros:output:0=backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&backward_lstm_20/lstm_cell_62/MatMul_1�
!backward_lstm_20/lstm_cell_62/addAddV2.backward_lstm_20/lstm_cell_62/MatMul:product:00backward_lstm_20/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2#
!backward_lstm_20/lstm_cell_62/add�
4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOp=backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype026
4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp�
%backward_lstm_20/lstm_cell_62/BiasAddBiasAdd%backward_lstm_20/lstm_cell_62/add:z:0<backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%backward_lstm_20/lstm_cell_62/BiasAdd�
-backward_lstm_20/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-backward_lstm_20/lstm_cell_62/split/split_dim�
#backward_lstm_20/lstm_cell_62/splitSplit6backward_lstm_20/lstm_cell_62/split/split_dim:output:0.backward_lstm_20/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2%
#backward_lstm_20/lstm_cell_62/split�
%backward_lstm_20/lstm_cell_62/SigmoidSigmoid,backward_lstm_20/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22'
%backward_lstm_20/lstm_cell_62/Sigmoid�
'backward_lstm_20/lstm_cell_62/Sigmoid_1Sigmoid,backward_lstm_20/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22)
'backward_lstm_20/lstm_cell_62/Sigmoid_1�
!backward_lstm_20/lstm_cell_62/mulMul+backward_lstm_20/lstm_cell_62/Sigmoid_1:y:0!backward_lstm_20/zeros_1:output:0*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/lstm_cell_62/mul�
"backward_lstm_20/lstm_cell_62/ReluRelu,backward_lstm_20/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22$
"backward_lstm_20/lstm_cell_62/Relu�
#backward_lstm_20/lstm_cell_62/mul_1Mul)backward_lstm_20/lstm_cell_62/Sigmoid:y:00backward_lstm_20/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_20/lstm_cell_62/mul_1�
#backward_lstm_20/lstm_cell_62/add_1AddV2%backward_lstm_20/lstm_cell_62/mul:z:0'backward_lstm_20/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_20/lstm_cell_62/add_1�
'backward_lstm_20/lstm_cell_62/Sigmoid_2Sigmoid,backward_lstm_20/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22)
'backward_lstm_20/lstm_cell_62/Sigmoid_2�
$backward_lstm_20/lstm_cell_62/Relu_1Relu'backward_lstm_20/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22&
$backward_lstm_20/lstm_cell_62/Relu_1�
#backward_lstm_20/lstm_cell_62/mul_2Mul+backward_lstm_20/lstm_cell_62/Sigmoid_2:y:02backward_lstm_20/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_20/lstm_cell_62/mul_2�
.backward_lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   20
.backward_lstm_20/TensorArrayV2_1/element_shape�
 backward_lstm_20/TensorArrayV2_1TensorListReserve7backward_lstm_20/TensorArrayV2_1/element_shape:output:0)backward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_20/TensorArrayV2_1p
backward_lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_20/time�
&backward_lstm_20/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_lstm_20/Max/reduction_indices�
backward_lstm_20/MaxMaxbackward_lstm_20/Cast:y:0/backward_lstm_20/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/Maxr
backward_lstm_20/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_20/sub/y�
backward_lstm_20/subSubbackward_lstm_20/Max:output:0backward_lstm_20/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/sub�
backward_lstm_20/Sub_1Subbackward_lstm_20/sub:z:0backward_lstm_20/Cast:y:0*
T0*#
_output_shapes
:���������2
backward_lstm_20/Sub_1�
backward_lstm_20/zeros_like	ZerosLike'backward_lstm_20/lstm_cell_62/mul_2:z:0*
T0*'
_output_shapes
:���������22
backward_lstm_20/zeros_like�
)backward_lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2+
)backward_lstm_20/while/maximum_iterations�
#backward_lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_20/while/loop_counter�	
backward_lstm_20/whileWhile,backward_lstm_20/while/loop_counter:output:02backward_lstm_20/while/maximum_iterations:output:0backward_lstm_20/time:output:0)backward_lstm_20/TensorArrayV2_1:handle:0backward_lstm_20/zeros_like:y:0backward_lstm_20/zeros:output:0!backward_lstm_20/zeros_1:output:0)backward_lstm_20/strided_slice_1:output:0Hbackward_lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_20/Sub_1:z:0<backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource>backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource=backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource*
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
#backward_lstm_20_while_body_2414099*/
cond'R%
#backward_lstm_20_while_cond_2414098*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
backward_lstm_20/while�
Abackward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2C
Abackward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape�
3backward_lstm_20/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_20/while:output:3Jbackward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype025
3backward_lstm_20/TensorArrayV2Stack/TensorListStack�
&backward_lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2(
&backward_lstm_20/strided_slice_3/stack�
(backward_lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_20/strided_slice_3/stack_1�
(backward_lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_3/stack_2�
 backward_lstm_20/strided_slice_3StridedSlice<backward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_20/strided_slice_3/stack:output:01backward_lstm_20/strided_slice_3/stack_1:output:01backward_lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2"
 backward_lstm_20/strided_slice_3�
!backward_lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_20/transpose_1/perm�
backward_lstm_20/transpose_1	Transpose<backward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_20/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
backward_lstm_20/transpose_1�
backward_lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_20/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2(forward_lstm_20/strided_slice_3:output:0)backward_lstm_20/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp5^backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp4^backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp6^backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp^backward_lstm_20/while4^forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp3^forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp5^forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp^forward_lstm_20/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������: : : : : : 2l
4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp2j
3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp2n
5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp20
backward_lstm_20/whilebackward_lstm_20/while2j
3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp2h
2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp2l
4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp2.
forward_lstm_20/whileforward_lstm_20/while:O K
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
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_2413537

inputs>
+lstm_cell_62_matmul_readvariableop_resource:	�@
-lstm_cell_62_matmul_1_readvariableop_resource:	2�;
,lstm_cell_62_biasadd_readvariableop_resource:	�
identity��#lstm_cell_62/BiasAdd/ReadVariableOp�"lstm_cell_62/MatMul/ReadVariableOp�$lstm_cell_62/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_62/MatMul/ReadVariableOpReadVariableOp+lstm_cell_62_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_62/MatMul/ReadVariableOp�
lstm_cell_62/MatMulMatMulstrided_slice_2:output:0*lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/MatMul�
$lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_62_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_62/MatMul_1/ReadVariableOp�
lstm_cell_62/MatMul_1MatMulzeros:output:0,lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/MatMul_1�
lstm_cell_62/addAddV2lstm_cell_62/MatMul:product:0lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/add�
#lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_62/BiasAdd/ReadVariableOp�
lstm_cell_62/BiasAddBiasAddlstm_cell_62/add:z:0+lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/BiasAdd~
lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_62/split/split_dim�
lstm_cell_62/splitSplit%lstm_cell_62/split/split_dim:output:0lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_62/split�
lstm_cell_62/SigmoidSigmoidlstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/Sigmoid�
lstm_cell_62/Sigmoid_1Sigmoidlstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_62/Sigmoid_1�
lstm_cell_62/mulMullstm_cell_62/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/mul}
lstm_cell_62/ReluRelulstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_62/Relu�
lstm_cell_62/mul_1Mullstm_cell_62/Sigmoid:y:0lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/mul_1�
lstm_cell_62/add_1AddV2lstm_cell_62/mul:z:0lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/add_1�
lstm_cell_62/Sigmoid_2Sigmoidlstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_62/Sigmoid_2|
lstm_cell_62/Relu_1Relulstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/Relu_1�
lstm_cell_62/mul_2Mullstm_cell_62/Sigmoid_2:y:0!lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_62_matmul_readvariableop_resource-lstm_cell_62_matmul_1_readvariableop_resource,lstm_cell_62_biasadd_readvariableop_resource*
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
while_body_2413453*
condR
while_cond_2413452*K
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
NoOpNoOp$^lstm_cell_62/BiasAdd/ReadVariableOp#^lstm_cell_62/MatMul/ReadVariableOp%^lstm_cell_62/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_62/BiasAdd/ReadVariableOp#lstm_cell_62/BiasAdd/ReadVariableOp2H
"lstm_cell_62/MatMul/ReadVariableOp"lstm_cell_62/MatMul/ReadVariableOp2L
$lstm_cell_62/MatMul_1/ReadVariableOp$lstm_cell_62/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
#backward_lstm_20_while_cond_2415101>
:backward_lstm_20_while_backward_lstm_20_while_loop_counterD
@backward_lstm_20_while_backward_lstm_20_while_maximum_iterations&
"backward_lstm_20_while_placeholder(
$backward_lstm_20_while_placeholder_1(
$backward_lstm_20_while_placeholder_2(
$backward_lstm_20_while_placeholder_3@
<backward_lstm_20_while_less_backward_lstm_20_strided_slice_1W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2415101___redundant_placeholder0W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2415101___redundant_placeholder1W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2415101___redundant_placeholder2W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2415101___redundant_placeholder3#
backward_lstm_20_while_identity
�
backward_lstm_20/while/LessLess"backward_lstm_20_while_placeholder<backward_lstm_20_while_less_backward_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_20/while/Less�
backward_lstm_20/while/IdentityIdentitybackward_lstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_20/while/Identity"K
backward_lstm_20_while_identity(backward_lstm_20/while/Identity:output:0*(
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
while_body_2416790
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_61_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_61_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_61_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_61_matmul_readvariableop_resource:	�F
3while_lstm_cell_61_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_61_biasadd_readvariableop_resource:	���)while/lstm_cell_61/BiasAdd/ReadVariableOp�(while/lstm_cell_61/MatMul/ReadVariableOp�*while/lstm_cell_61/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_61/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_61_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_61/MatMul/ReadVariableOp�
while/lstm_cell_61/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/MatMul�
*while/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_61_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_61/MatMul_1/ReadVariableOp�
while/lstm_cell_61/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/MatMul_1�
while/lstm_cell_61/addAddV2#while/lstm_cell_61/MatMul:product:0%while/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/add�
)while/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_61_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_61/BiasAdd/ReadVariableOp�
while/lstm_cell_61/BiasAddBiasAddwhile/lstm_cell_61/add:z:01while/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/BiasAdd�
"while/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_61/split/split_dim�
while/lstm_cell_61/splitSplit+while/lstm_cell_61/split/split_dim:output:0#while/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_61/split�
while/lstm_cell_61/SigmoidSigmoid!while/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Sigmoid�
while/lstm_cell_61/Sigmoid_1Sigmoid!while/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Sigmoid_1�
while/lstm_cell_61/mulMul while/lstm_cell_61/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/mul�
while/lstm_cell_61/ReluRelu!while/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Relu�
while/lstm_cell_61/mul_1Mulwhile/lstm_cell_61/Sigmoid:y:0%while/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/mul_1�
while/lstm_cell_61/add_1AddV2while/lstm_cell_61/mul:z:0while/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/add_1�
while/lstm_cell_61/Sigmoid_2Sigmoid!while/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Sigmoid_2�
while/lstm_cell_61/Relu_1Reluwhile/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Relu_1�
while/lstm_cell_61/mul_2Mul while/lstm_cell_61/Sigmoid_2:y:0'while/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_61/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_61/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_61/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_61/BiasAdd/ReadVariableOp)^while/lstm_cell_61/MatMul/ReadVariableOp+^while/lstm_cell_61/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_61_biasadd_readvariableop_resource4while_lstm_cell_61_biasadd_readvariableop_resource_0"l
3while_lstm_cell_61_matmul_1_readvariableop_resource5while_lstm_cell_61_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_61_matmul_readvariableop_resource3while_lstm_cell_61_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_61/BiasAdd/ReadVariableOp)while/lstm_cell_61/BiasAdd/ReadVariableOp2T
(while/lstm_cell_61/MatMul/ReadVariableOp(while/lstm_cell_61/MatMul/ReadVariableOp2X
*while/lstm_cell_61/MatMul_1/ReadVariableOp*while/lstm_cell_61/MatMul_1/ReadVariableOp: 
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
�
E__inference_dense_20_layer_call_and_return_conditional_losses_2414221

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
�
�
"forward_lstm_20_while_cond_2415571<
8forward_lstm_20_while_forward_lstm_20_while_loop_counterB
>forward_lstm_20_while_forward_lstm_20_while_maximum_iterations%
!forward_lstm_20_while_placeholder'
#forward_lstm_20_while_placeholder_1'
#forward_lstm_20_while_placeholder_2'
#forward_lstm_20_while_placeholder_3'
#forward_lstm_20_while_placeholder_4>
:forward_lstm_20_while_less_forward_lstm_20_strided_slice_1U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2415571___redundant_placeholder0U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2415571___redundant_placeholder1U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2415571___redundant_placeholder2U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2415571___redundant_placeholder3U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2415571___redundant_placeholder4"
forward_lstm_20_while_identity
�
forward_lstm_20/while/LessLess!forward_lstm_20_while_placeholder:forward_lstm_20_while_less_forward_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_20/while/Less�
forward_lstm_20/while/IdentityIdentityforward_lstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_20/while/Identity"I
forward_lstm_20_while_identity'forward_lstm_20/while/Identity:output:0*(
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
�
�
E__inference_dense_20_layer_call_and_return_conditional_losses_2416226

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
�?
�
while_body_2413453
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_62_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_62_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_62_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_62_matmul_readvariableop_resource:	�F
3while_lstm_cell_62_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_62_biasadd_readvariableop_resource:	���)while/lstm_cell_62/BiasAdd/ReadVariableOp�(while/lstm_cell_62/MatMul/ReadVariableOp�*while/lstm_cell_62/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_62/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_62_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_62/MatMul/ReadVariableOp�
while/lstm_cell_62/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/MatMul�
*while/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_62_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_62/MatMul_1/ReadVariableOp�
while/lstm_cell_62/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/MatMul_1�
while/lstm_cell_62/addAddV2#while/lstm_cell_62/MatMul:product:0%while/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/add�
)while/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_62_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_62/BiasAdd/ReadVariableOp�
while/lstm_cell_62/BiasAddBiasAddwhile/lstm_cell_62/add:z:01while/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/BiasAdd�
"while/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_62/split/split_dim�
while/lstm_cell_62/splitSplit+while/lstm_cell_62/split/split_dim:output:0#while/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_62/split�
while/lstm_cell_62/SigmoidSigmoid!while/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Sigmoid�
while/lstm_cell_62/Sigmoid_1Sigmoid!while/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Sigmoid_1�
while/lstm_cell_62/mulMul while/lstm_cell_62/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/mul�
while/lstm_cell_62/ReluRelu!while/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Relu�
while/lstm_cell_62/mul_1Mulwhile/lstm_cell_62/Sigmoid:y:0%while/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/mul_1�
while/lstm_cell_62/add_1AddV2while/lstm_cell_62/mul:z:0while/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/add_1�
while/lstm_cell_62/Sigmoid_2Sigmoid!while/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Sigmoid_2�
while/lstm_cell_62/Relu_1Reluwhile/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Relu_1�
while/lstm_cell_62/mul_2Mul while/lstm_cell_62/Sigmoid_2:y:0'while/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_62/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_62/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_62/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_62/BiasAdd/ReadVariableOp)^while/lstm_cell_62/MatMul/ReadVariableOp+^while/lstm_cell_62/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_62_biasadd_readvariableop_resource4while_lstm_cell_62_biasadd_readvariableop_resource_0"l
3while_lstm_cell_62_matmul_1_readvariableop_resource5while_lstm_cell_62_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_62_matmul_readvariableop_resource3while_lstm_cell_62_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_62/BiasAdd/ReadVariableOp)while/lstm_cell_62/BiasAdd/ReadVariableOp2T
(while/lstm_cell_62/MatMul/ReadVariableOp(while/lstm_cell_62/MatMul/ReadVariableOp2X
*while/lstm_cell_62/MatMul_1/ReadVariableOp*while/lstm_cell_62/MatMul_1/ReadVariableOp: 
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
�W
�
#backward_lstm_20_while_body_2415102>
:backward_lstm_20_while_backward_lstm_20_while_loop_counterD
@backward_lstm_20_while_backward_lstm_20_while_maximum_iterations&
"backward_lstm_20_while_placeholder(
$backward_lstm_20_while_placeholder_1(
$backward_lstm_20_while_placeholder_2(
$backward_lstm_20_while_placeholder_3=
9backward_lstm_20_while_backward_lstm_20_strided_slice_1_0y
ubackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0W
Dbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0:	�Y
Fbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0:	2�T
Ebackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0:	�#
backward_lstm_20_while_identity%
!backward_lstm_20_while_identity_1%
!backward_lstm_20_while_identity_2%
!backward_lstm_20_while_identity_3%
!backward_lstm_20_while_identity_4%
!backward_lstm_20_while_identity_5;
7backward_lstm_20_while_backward_lstm_20_strided_slice_1w
sbackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensorU
Bbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource:	�W
Dbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource:	2�R
Cbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource:	���:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp�9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp�;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp�
Hbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2J
Hbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape�
:backward_lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_20_while_placeholderQbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02<
:backward_lstm_20/while/TensorArrayV2Read/TensorListGetItem�
9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOpReadVariableOpDbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02;
9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp�
*backward_lstm_20/while/lstm_cell_62/MatMulMatMulAbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:0Abackward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*backward_lstm_20/while/lstm_cell_62/MatMul�
;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOpFbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02=
;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp�
,backward_lstm_20/while/lstm_cell_62/MatMul_1MatMul$backward_lstm_20_while_placeholder_2Cbackward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2.
,backward_lstm_20/while/lstm_cell_62/MatMul_1�
'backward_lstm_20/while/lstm_cell_62/addAddV24backward_lstm_20/while/lstm_cell_62/MatMul:product:06backward_lstm_20/while/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2)
'backward_lstm_20/while/lstm_cell_62/add�
:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOpEbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02<
:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp�
+backward_lstm_20/while/lstm_cell_62/BiasAddBiasAdd+backward_lstm_20/while/lstm_cell_62/add:z:0Bbackward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+backward_lstm_20/while/lstm_cell_62/BiasAdd�
3backward_lstm_20/while/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3backward_lstm_20/while/lstm_cell_62/split/split_dim�
)backward_lstm_20/while/lstm_cell_62/splitSplit<backward_lstm_20/while/lstm_cell_62/split/split_dim:output:04backward_lstm_20/while/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2+
)backward_lstm_20/while/lstm_cell_62/split�
+backward_lstm_20/while/lstm_cell_62/SigmoidSigmoid2backward_lstm_20/while/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22-
+backward_lstm_20/while/lstm_cell_62/Sigmoid�
-backward_lstm_20/while/lstm_cell_62/Sigmoid_1Sigmoid2backward_lstm_20/while/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22/
-backward_lstm_20/while/lstm_cell_62/Sigmoid_1�
'backward_lstm_20/while/lstm_cell_62/mulMul1backward_lstm_20/while/lstm_cell_62/Sigmoid_1:y:0$backward_lstm_20_while_placeholder_3*
T0*'
_output_shapes
:���������22)
'backward_lstm_20/while/lstm_cell_62/mul�
(backward_lstm_20/while/lstm_cell_62/ReluRelu2backward_lstm_20/while/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22*
(backward_lstm_20/while/lstm_cell_62/Relu�
)backward_lstm_20/while/lstm_cell_62/mul_1Mul/backward_lstm_20/while/lstm_cell_62/Sigmoid:y:06backward_lstm_20/while/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_20/while/lstm_cell_62/mul_1�
)backward_lstm_20/while/lstm_cell_62/add_1AddV2+backward_lstm_20/while/lstm_cell_62/mul:z:0-backward_lstm_20/while/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_20/while/lstm_cell_62/add_1�
-backward_lstm_20/while/lstm_cell_62/Sigmoid_2Sigmoid2backward_lstm_20/while/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22/
-backward_lstm_20/while/lstm_cell_62/Sigmoid_2�
*backward_lstm_20/while/lstm_cell_62/Relu_1Relu-backward_lstm_20/while/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22,
*backward_lstm_20/while/lstm_cell_62/Relu_1�
)backward_lstm_20/while/lstm_cell_62/mul_2Mul1backward_lstm_20/while/lstm_cell_62/Sigmoid_2:y:08backward_lstm_20/while/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_20/while/lstm_cell_62/mul_2�
;backward_lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_20_while_placeholder_1"backward_lstm_20_while_placeholder-backward_lstm_20/while/lstm_cell_62/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_20/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_20/while/add/y�
backward_lstm_20/while/addAddV2"backward_lstm_20_while_placeholder%backward_lstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/while/add�
backward_lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_20/while/add_1/y�
backward_lstm_20/while/add_1AddV2:backward_lstm_20_while_backward_lstm_20_while_loop_counter'backward_lstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/while/add_1�
backward_lstm_20/while/IdentityIdentity backward_lstm_20/while/add_1:z:0^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_20/while/Identity�
!backward_lstm_20/while/Identity_1Identity@backward_lstm_20_while_backward_lstm_20_while_maximum_iterations^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_20/while/Identity_1�
!backward_lstm_20/while/Identity_2Identitybackward_lstm_20/while/add:z:0^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_20/while/Identity_2�
!backward_lstm_20/while/Identity_3IdentityKbackward_lstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_20/while/Identity_3�
!backward_lstm_20/while/Identity_4Identity-backward_lstm_20/while/lstm_cell_62/mul_2:z:0^backward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/while/Identity_4�
!backward_lstm_20/while/Identity_5Identity-backward_lstm_20/while/lstm_cell_62/add_1:z:0^backward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/while/Identity_5�
backward_lstm_20/while/NoOpNoOp;^backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp:^backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp<^backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_20/while/NoOp"t
7backward_lstm_20_while_backward_lstm_20_strided_slice_19backward_lstm_20_while_backward_lstm_20_strided_slice_1_0"K
backward_lstm_20_while_identity(backward_lstm_20/while/Identity:output:0"O
!backward_lstm_20_while_identity_1*backward_lstm_20/while/Identity_1:output:0"O
!backward_lstm_20_while_identity_2*backward_lstm_20/while/Identity_2:output:0"O
!backward_lstm_20_while_identity_3*backward_lstm_20/while/Identity_3:output:0"O
!backward_lstm_20_while_identity_4*backward_lstm_20/while/Identity_4:output:0"O
!backward_lstm_20_while_identity_5*backward_lstm_20/while/Identity_5:output:0"�
Cbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resourceEbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0"�
Dbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resourceFbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0"�
Bbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resourceDbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0"�
sbackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensorubackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2x
:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp2v
9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp2z
;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp: 
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
2__inference_bidirectional_20_layer_call_fn_2414868

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
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_24141962
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
�b
�
"forward_lstm_20_while_body_2415572<
8forward_lstm_20_while_forward_lstm_20_while_loop_counterB
>forward_lstm_20_while_forward_lstm_20_while_maximum_iterations%
!forward_lstm_20_while_placeholder'
#forward_lstm_20_while_placeholder_1'
#forward_lstm_20_while_placeholder_2'
#forward_lstm_20_while_placeholder_3'
#forward_lstm_20_while_placeholder_4;
7forward_lstm_20_while_forward_lstm_20_strided_slice_1_0w
sforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_08
4forward_lstm_20_while_greater_forward_lstm_20_cast_0V
Cforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0:	�X
Eforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0:	2�S
Dforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0:	�"
forward_lstm_20_while_identity$
 forward_lstm_20_while_identity_1$
 forward_lstm_20_while_identity_2$
 forward_lstm_20_while_identity_3$
 forward_lstm_20_while_identity_4$
 forward_lstm_20_while_identity_5$
 forward_lstm_20_while_identity_69
5forward_lstm_20_while_forward_lstm_20_strided_slice_1u
qforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor6
2forward_lstm_20_while_greater_forward_lstm_20_castT
Aforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource:	�V
Cforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource:	2�Q
Bforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource:	���9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp�8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp�:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp�
Gforward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2I
Gforward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape�
9forward_lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_20_while_placeholderPforward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02;
9forward_lstm_20/while/TensorArrayV2Read/TensorListGetItem�
forward_lstm_20/while/GreaterGreater4forward_lstm_20_while_greater_forward_lstm_20_cast_0!forward_lstm_20_while_placeholder*
T0*#
_output_shapes
:���������2
forward_lstm_20/while/Greater�
8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOpReadVariableOpCforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02:
8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp�
)forward_lstm_20/while/lstm_cell_61/MatMulMatMul@forward_lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:0@forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)forward_lstm_20/while/lstm_cell_61/MatMul�
:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOpEforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02<
:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp�
+forward_lstm_20/while/lstm_cell_61/MatMul_1MatMul#forward_lstm_20_while_placeholder_3Bforward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+forward_lstm_20/while/lstm_cell_61/MatMul_1�
&forward_lstm_20/while/lstm_cell_61/addAddV23forward_lstm_20/while/lstm_cell_61/MatMul:product:05forward_lstm_20/while/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&forward_lstm_20/while/lstm_cell_61/add�
9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOpDforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02;
9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp�
*forward_lstm_20/while/lstm_cell_61/BiasAddBiasAdd*forward_lstm_20/while/lstm_cell_61/add:z:0Aforward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*forward_lstm_20/while/lstm_cell_61/BiasAdd�
2forward_lstm_20/while/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2forward_lstm_20/while/lstm_cell_61/split/split_dim�
(forward_lstm_20/while/lstm_cell_61/splitSplit;forward_lstm_20/while/lstm_cell_61/split/split_dim:output:03forward_lstm_20/while/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2*
(forward_lstm_20/while/lstm_cell_61/split�
*forward_lstm_20/while/lstm_cell_61/SigmoidSigmoid1forward_lstm_20/while/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22,
*forward_lstm_20/while/lstm_cell_61/Sigmoid�
,forward_lstm_20/while/lstm_cell_61/Sigmoid_1Sigmoid1forward_lstm_20/while/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22.
,forward_lstm_20/while/lstm_cell_61/Sigmoid_1�
&forward_lstm_20/while/lstm_cell_61/mulMul0forward_lstm_20/while/lstm_cell_61/Sigmoid_1:y:0#forward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22(
&forward_lstm_20/while/lstm_cell_61/mul�
'forward_lstm_20/while/lstm_cell_61/ReluRelu1forward_lstm_20/while/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22)
'forward_lstm_20/while/lstm_cell_61/Relu�
(forward_lstm_20/while/lstm_cell_61/mul_1Mul.forward_lstm_20/while/lstm_cell_61/Sigmoid:y:05forward_lstm_20/while/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_20/while/lstm_cell_61/mul_1�
(forward_lstm_20/while/lstm_cell_61/add_1AddV2*forward_lstm_20/while/lstm_cell_61/mul:z:0,forward_lstm_20/while/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_20/while/lstm_cell_61/add_1�
,forward_lstm_20/while/lstm_cell_61/Sigmoid_2Sigmoid1forward_lstm_20/while/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22.
,forward_lstm_20/while/lstm_cell_61/Sigmoid_2�
)forward_lstm_20/while/lstm_cell_61/Relu_1Relu,forward_lstm_20/while/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22+
)forward_lstm_20/while/lstm_cell_61/Relu_1�
(forward_lstm_20/while/lstm_cell_61/mul_2Mul0forward_lstm_20/while/lstm_cell_61/Sigmoid_2:y:07forward_lstm_20/while/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_20/while/lstm_cell_61/mul_2�
forward_lstm_20/while/SelectSelect!forward_lstm_20/while/Greater:z:0,forward_lstm_20/while/lstm_cell_61/mul_2:z:0#forward_lstm_20_while_placeholder_2*
T0*'
_output_shapes
:���������22
forward_lstm_20/while/Select�
forward_lstm_20/while/Select_1Select!forward_lstm_20/while/Greater:z:0,forward_lstm_20/while/lstm_cell_61/mul_2:z:0#forward_lstm_20_while_placeholder_3*
T0*'
_output_shapes
:���������22 
forward_lstm_20/while/Select_1�
forward_lstm_20/while/Select_2Select!forward_lstm_20/while/Greater:z:0,forward_lstm_20/while/lstm_cell_61/add_1:z:0#forward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22 
forward_lstm_20/while/Select_2�
:forward_lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_20_while_placeholder_1!forward_lstm_20_while_placeholder%forward_lstm_20/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_20/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_20/while/add/y�
forward_lstm_20/while/addAddV2!forward_lstm_20_while_placeholder$forward_lstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/while/add�
forward_lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_20/while/add_1/y�
forward_lstm_20/while/add_1AddV28forward_lstm_20_while_forward_lstm_20_while_loop_counter&forward_lstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/while/add_1�
forward_lstm_20/while/IdentityIdentityforward_lstm_20/while/add_1:z:0^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_20/while/Identity�
 forward_lstm_20/while/Identity_1Identity>forward_lstm_20_while_forward_lstm_20_while_maximum_iterations^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_20/while/Identity_1�
 forward_lstm_20/while/Identity_2Identityforward_lstm_20/while/add:z:0^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_20/while/Identity_2�
 forward_lstm_20/while/Identity_3IdentityJforward_lstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_20/while/Identity_3�
 forward_lstm_20/while/Identity_4Identity%forward_lstm_20/while/Select:output:0^forward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/while/Identity_4�
 forward_lstm_20/while/Identity_5Identity'forward_lstm_20/while/Select_1:output:0^forward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/while/Identity_5�
 forward_lstm_20/while/Identity_6Identity'forward_lstm_20/while/Select_2:output:0^forward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/while/Identity_6�
forward_lstm_20/while/NoOpNoOp:^forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp9^forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp;^forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_20/while/NoOp"p
5forward_lstm_20_while_forward_lstm_20_strided_slice_17forward_lstm_20_while_forward_lstm_20_strided_slice_1_0"j
2forward_lstm_20_while_greater_forward_lstm_20_cast4forward_lstm_20_while_greater_forward_lstm_20_cast_0"I
forward_lstm_20_while_identity'forward_lstm_20/while/Identity:output:0"M
 forward_lstm_20_while_identity_1)forward_lstm_20/while/Identity_1:output:0"M
 forward_lstm_20_while_identity_2)forward_lstm_20/while/Identity_2:output:0"M
 forward_lstm_20_while_identity_3)forward_lstm_20/while/Identity_3:output:0"M
 forward_lstm_20_while_identity_4)forward_lstm_20/while/Identity_4:output:0"M
 forward_lstm_20_while_identity_5)forward_lstm_20/while/Identity_5:output:0"M
 forward_lstm_20_while_identity_6)forward_lstm_20/while/Identity_6:output:0"�
Bforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resourceDforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0"�
Cforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resourceEforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0"�
Aforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resourceCforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0"�
qforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensorsforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2v
9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp2t
8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp2x
:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp: 
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
%__inference_signature_wrapper_2414816

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
"__inference__wrapped_model_24117602
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
�?
�
while_body_2413101
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_61_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_61_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_61_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_61_matmul_readvariableop_resource:	�F
3while_lstm_cell_61_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_61_biasadd_readvariableop_resource:	���)while/lstm_cell_61/BiasAdd/ReadVariableOp�(while/lstm_cell_61/MatMul/ReadVariableOp�*while/lstm_cell_61/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_61/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_61_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_61/MatMul/ReadVariableOp�
while/lstm_cell_61/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/MatMul�
*while/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_61_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_61/MatMul_1/ReadVariableOp�
while/lstm_cell_61/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/MatMul_1�
while/lstm_cell_61/addAddV2#while/lstm_cell_61/MatMul:product:0%while/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/add�
)while/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_61_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_61/BiasAdd/ReadVariableOp�
while/lstm_cell_61/BiasAddBiasAddwhile/lstm_cell_61/add:z:01while/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/BiasAdd�
"while/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_61/split/split_dim�
while/lstm_cell_61/splitSplit+while/lstm_cell_61/split/split_dim:output:0#while/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_61/split�
while/lstm_cell_61/SigmoidSigmoid!while/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Sigmoid�
while/lstm_cell_61/Sigmoid_1Sigmoid!while/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Sigmoid_1�
while/lstm_cell_61/mulMul while/lstm_cell_61/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/mul�
while/lstm_cell_61/ReluRelu!while/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Relu�
while/lstm_cell_61/mul_1Mulwhile/lstm_cell_61/Sigmoid:y:0%while/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/mul_1�
while/lstm_cell_61/add_1AddV2while/lstm_cell_61/mul:z:0while/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/add_1�
while/lstm_cell_61/Sigmoid_2Sigmoid!while/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Sigmoid_2�
while/lstm_cell_61/Relu_1Reluwhile/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Relu_1�
while/lstm_cell_61/mul_2Mul while/lstm_cell_61/Sigmoid_2:y:0'while/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_61/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_61/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_61/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_61/BiasAdd/ReadVariableOp)^while/lstm_cell_61/MatMul/ReadVariableOp+^while/lstm_cell_61/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_61_biasadd_readvariableop_resource4while_lstm_cell_61_biasadd_readvariableop_resource_0"l
3while_lstm_cell_61_matmul_1_readvariableop_resource5while_lstm_cell_61_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_61_matmul_readvariableop_resource3while_lstm_cell_61_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_61/BiasAdd/ReadVariableOp)while/lstm_cell_61/BiasAdd/ReadVariableOp2T
(while/lstm_cell_61/MatMul/ReadVariableOp(while/lstm_cell_61/MatMul/ReadVariableOp2X
*while/lstm_cell_61/MatMul_1/ReadVariableOp*while/lstm_cell_61/MatMul_1/ReadVariableOp: 
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
�b
�
 __inference__traced_save_2417867
file_prefix.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopS
Osavev2_bidirectional_20_forward_lstm_20_lstm_cell_61_kernel_read_readvariableop]
Ysavev2_bidirectional_20_forward_lstm_20_lstm_cell_61_recurrent_kernel_read_readvariableopQ
Msavev2_bidirectional_20_forward_lstm_20_lstm_cell_61_bias_read_readvariableopT
Psavev2_bidirectional_20_backward_lstm_20_lstm_cell_62_kernel_read_readvariableop^
Zsavev2_bidirectional_20_backward_lstm_20_lstm_cell_62_recurrent_kernel_read_readvariableopR
Nsavev2_bidirectional_20_backward_lstm_20_lstm_cell_62_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_20_kernel_m_read_readvariableop3
/savev2_adam_dense_20_bias_m_read_readvariableopZ
Vsavev2_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_kernel_m_read_readvariableopd
`savev2_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_recurrent_kernel_m_read_readvariableopX
Tsavev2_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_bias_m_read_readvariableop[
Wsavev2_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_kernel_m_read_readvariableope
asavev2_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_recurrent_kernel_m_read_readvariableopY
Usavev2_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_bias_m_read_readvariableop5
1savev2_adam_dense_20_kernel_v_read_readvariableop3
/savev2_adam_dense_20_bias_v_read_readvariableopZ
Vsavev2_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_kernel_v_read_readvariableopd
`savev2_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_recurrent_kernel_v_read_readvariableopX
Tsavev2_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_bias_v_read_readvariableop[
Wsavev2_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_kernel_v_read_readvariableope
asavev2_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_recurrent_kernel_v_read_readvariableopY
Usavev2_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_bias_v_read_readvariableop8
4savev2_adam_dense_20_kernel_vhat_read_readvariableop6
2savev2_adam_dense_20_bias_vhat_read_readvariableop]
Ysavev2_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_kernel_vhat_read_readvariableopg
csavev2_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_recurrent_kernel_vhat_read_readvariableop[
Wsavev2_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_bias_vhat_read_readvariableop^
Zsavev2_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_kernel_vhat_read_readvariableoph
dsavev2_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_recurrent_kernel_vhat_read_readvariableop\
Xsavev2_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_bias_vhat_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopOsavev2_bidirectional_20_forward_lstm_20_lstm_cell_61_kernel_read_readvariableopYsavev2_bidirectional_20_forward_lstm_20_lstm_cell_61_recurrent_kernel_read_readvariableopMsavev2_bidirectional_20_forward_lstm_20_lstm_cell_61_bias_read_readvariableopPsavev2_bidirectional_20_backward_lstm_20_lstm_cell_62_kernel_read_readvariableopZsavev2_bidirectional_20_backward_lstm_20_lstm_cell_62_recurrent_kernel_read_readvariableopNsavev2_bidirectional_20_backward_lstm_20_lstm_cell_62_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_20_kernel_m_read_readvariableop/savev2_adam_dense_20_bias_m_read_readvariableopVsavev2_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_kernel_m_read_readvariableop`savev2_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_recurrent_kernel_m_read_readvariableopTsavev2_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_bias_m_read_readvariableopWsavev2_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_kernel_m_read_readvariableopasavev2_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_recurrent_kernel_m_read_readvariableopUsavev2_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_bias_m_read_readvariableop1savev2_adam_dense_20_kernel_v_read_readvariableop/savev2_adam_dense_20_bias_v_read_readvariableopVsavev2_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_kernel_v_read_readvariableop`savev2_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_recurrent_kernel_v_read_readvariableopTsavev2_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_bias_v_read_readvariableopWsavev2_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_kernel_v_read_readvariableopasavev2_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_recurrent_kernel_v_read_readvariableopUsavev2_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_bias_v_read_readvariableop4savev2_adam_dense_20_kernel_vhat_read_readvariableop2savev2_adam_dense_20_bias_vhat_read_readvariableopYsavev2_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_kernel_vhat_read_readvariableopcsavev2_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_recurrent_kernel_vhat_read_readvariableopWsavev2_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_bias_vhat_read_readvariableopZsavev2_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_kernel_vhat_read_readvariableopdsavev2_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_recurrent_kernel_vhat_read_readvariableopXsavev2_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_bias_vhat_read_readvariableopsavev2_const"/device:CPU:0*
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
�
�
1__inference_forward_lstm_20_layer_call_fn_2416248
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
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_24121282
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
�
�
J__inference_sequential_20_layer_call_and_return_conditional_losses_2414228

inputs
inputs_1	+
bidirectional_20_2414197:	�+
bidirectional_20_2414199:	2�'
bidirectional_20_2414201:	�+
bidirectional_20_2414203:	�+
bidirectional_20_2414205:	2�'
bidirectional_20_2414207:	�"
dense_20_2414222:d
dense_20_2414224:
identity��(bidirectional_20/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�
(bidirectional_20/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_20_2414197bidirectional_20_2414199bidirectional_20_2414201bidirectional_20_2414203bidirectional_20_2414205bidirectional_20_2414207*
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
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_24141962*
(bidirectional_20/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_20/StatefulPartitionedCall:output:0dense_20_2414222dense_20_2414224*
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
E__inference_dense_20_layer_call_and_return_conditional_losses_24142212"
 dense_20/StatefulPartitionedCall�
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp)^bidirectional_20/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:���������:���������: : : : : : : : 2T
(bidirectional_20/StatefulPartitionedCall(bidirectional_20/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall:O K
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
while_body_2413261
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_62_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_62_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_62_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_62_matmul_readvariableop_resource:	�F
3while_lstm_cell_62_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_62_biasadd_readvariableop_resource:	���)while/lstm_cell_62/BiasAdd/ReadVariableOp�(while/lstm_cell_62/MatMul/ReadVariableOp�*while/lstm_cell_62/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_62/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_62_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_62/MatMul/ReadVariableOp�
while/lstm_cell_62/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/MatMul�
*while/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_62_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_62/MatMul_1/ReadVariableOp�
while/lstm_cell_62/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/MatMul_1�
while/lstm_cell_62/addAddV2#while/lstm_cell_62/MatMul:product:0%while/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/add�
)while/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_62_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_62/BiasAdd/ReadVariableOp�
while/lstm_cell_62/BiasAddBiasAddwhile/lstm_cell_62/add:z:01while/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/BiasAdd�
"while/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_62/split/split_dim�
while/lstm_cell_62/splitSplit+while/lstm_cell_62/split/split_dim:output:0#while/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_62/split�
while/lstm_cell_62/SigmoidSigmoid!while/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Sigmoid�
while/lstm_cell_62/Sigmoid_1Sigmoid!while/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Sigmoid_1�
while/lstm_cell_62/mulMul while/lstm_cell_62/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/mul�
while/lstm_cell_62/ReluRelu!while/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Relu�
while/lstm_cell_62/mul_1Mulwhile/lstm_cell_62/Sigmoid:y:0%while/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/mul_1�
while/lstm_cell_62/add_1AddV2while/lstm_cell_62/mul:z:0while/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/add_1�
while/lstm_cell_62/Sigmoid_2Sigmoid!while/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Sigmoid_2�
while/lstm_cell_62/Relu_1Reluwhile/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Relu_1�
while/lstm_cell_62/mul_2Mul while/lstm_cell_62/Sigmoid_2:y:0'while/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_62/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_62/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_62/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_62/BiasAdd/ReadVariableOp)^while/lstm_cell_62/MatMul/ReadVariableOp+^while/lstm_cell_62/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_62_biasadd_readvariableop_resource4while_lstm_cell_62_biasadd_readvariableop_resource_0"l
3while_lstm_cell_62_matmul_1_readvariableop_resource5while_lstm_cell_62_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_62_matmul_readvariableop_resource3while_lstm_cell_62_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_62/BiasAdd/ReadVariableOp)while/lstm_cell_62/BiasAdd/ReadVariableOp2T
(while/lstm_cell_62/MatMul/ReadVariableOp(while/lstm_cell_62/MatMul/ReadVariableOp2X
*while/lstm_cell_62/MatMul_1/ReadVariableOp*while/lstm_cell_62/MatMul_1/ReadVariableOp: 
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
��
�
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_2416206

inputs
inputs_1	N
;forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource:	�P
=forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource:	2�K
<forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource:	�O
<backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource:	�Q
>backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource:	2�L
=backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource:	�
identity��4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp�3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp�5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp�backward_lstm_20/while�3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp�2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp�4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp�forward_lstm_20/while�
$forward_lstm_20/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$forward_lstm_20/RaggedToTensor/zeros�
$forward_lstm_20/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2&
$forward_lstm_20/RaggedToTensor/Const�
3forward_lstm_20/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-forward_lstm_20/RaggedToTensor/Const:output:0inputs-forward_lstm_20/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3forward_lstm_20/RaggedToTensor/RaggedTensorToTensor�
:forward_lstm_20/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack�
<forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1�
<forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2�
4forward_lstm_20/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cforward_lstm_20/RaggedNestedRowLengths/strided_slice/stack:output:0Eforward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1:output:0Eforward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask26
4forward_lstm_20/RaggedNestedRowLengths/strided_slice�
<forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack�
>forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2@
>forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1�
>forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2�
6forward_lstm_20/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Eforward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gforward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gforward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask28
6forward_lstm_20/RaggedNestedRowLengths/strided_slice_1�
*forward_lstm_20/RaggedNestedRowLengths/subSub=forward_lstm_20/RaggedNestedRowLengths/strided_slice:output:0?forward_lstm_20/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2,
*forward_lstm_20/RaggedNestedRowLengths/sub�
forward_lstm_20/CastCast.forward_lstm_20/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
forward_lstm_20/Cast�
forward_lstm_20/ShapeShape<forward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_20/Shape�
#forward_lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_20/strided_slice/stack�
%forward_lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_20/strided_slice/stack_1�
%forward_lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_20/strided_slice/stack_2�
forward_lstm_20/strided_sliceStridedSliceforward_lstm_20/Shape:output:0,forward_lstm_20/strided_slice/stack:output:0.forward_lstm_20/strided_slice/stack_1:output:0.forward_lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_20/strided_slice|
forward_lstm_20/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_20/zeros/mul/y�
forward_lstm_20/zeros/mulMul&forward_lstm_20/strided_slice:output:0$forward_lstm_20/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros/mul
forward_lstm_20/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_20/zeros/Less/y�
forward_lstm_20/zeros/LessLessforward_lstm_20/zeros/mul:z:0%forward_lstm_20/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros/Less�
forward_lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_20/zeros/packed/1�
forward_lstm_20/zeros/packedPack&forward_lstm_20/strided_slice:output:0'forward_lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_20/zeros/packed�
forward_lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_20/zeros/Const�
forward_lstm_20/zerosFill%forward_lstm_20/zeros/packed:output:0$forward_lstm_20/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_20/zeros�
forward_lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_20/zeros_1/mul/y�
forward_lstm_20/zeros_1/mulMul&forward_lstm_20/strided_slice:output:0&forward_lstm_20/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros_1/mul�
forward_lstm_20/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
forward_lstm_20/zeros_1/Less/y�
forward_lstm_20/zeros_1/LessLessforward_lstm_20/zeros_1/mul:z:0'forward_lstm_20/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros_1/Less�
 forward_lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_20/zeros_1/packed/1�
forward_lstm_20/zeros_1/packedPack&forward_lstm_20/strided_slice:output:0)forward_lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_20/zeros_1/packed�
forward_lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_20/zeros_1/Const�
forward_lstm_20/zeros_1Fill'forward_lstm_20/zeros_1/packed:output:0&forward_lstm_20/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_20/zeros_1�
forward_lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_20/transpose/perm�
forward_lstm_20/transpose	Transpose<forward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0'forward_lstm_20/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
forward_lstm_20/transpose
forward_lstm_20/Shape_1Shapeforward_lstm_20/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_20/Shape_1�
%forward_lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_20/strided_slice_1/stack�
'forward_lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_1/stack_1�
'forward_lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_1/stack_2�
forward_lstm_20/strided_slice_1StridedSlice forward_lstm_20/Shape_1:output:0.forward_lstm_20/strided_slice_1/stack:output:00forward_lstm_20/strided_slice_1/stack_1:output:00forward_lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_20/strided_slice_1�
+forward_lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+forward_lstm_20/TensorArrayV2/element_shape�
forward_lstm_20/TensorArrayV2TensorListReserve4forward_lstm_20/TensorArrayV2/element_shape:output:0(forward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_20/TensorArrayV2�
Eforward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2G
Eforward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape�
7forward_lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_20/transpose:y:0Nforward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_20/TensorArrayUnstack/TensorListFromTensor�
%forward_lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_20/strided_slice_2/stack�
'forward_lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_2/stack_1�
'forward_lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_2/stack_2�
forward_lstm_20/strided_slice_2StridedSliceforward_lstm_20/transpose:y:0.forward_lstm_20/strided_slice_2/stack:output:00forward_lstm_20/strided_slice_2/stack_1:output:00forward_lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2!
forward_lstm_20/strided_slice_2�
2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOpReadVariableOp;forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp�
#forward_lstm_20/lstm_cell_61/MatMulMatMul(forward_lstm_20/strided_slice_2:output:0:forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#forward_lstm_20/lstm_cell_61/MatMul�
4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOp=forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype026
4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp�
%forward_lstm_20/lstm_cell_61/MatMul_1MatMulforward_lstm_20/zeros:output:0<forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%forward_lstm_20/lstm_cell_61/MatMul_1�
 forward_lstm_20/lstm_cell_61/addAddV2-forward_lstm_20/lstm_cell_61/MatMul:product:0/forward_lstm_20/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2"
 forward_lstm_20/lstm_cell_61/add�
3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOp<forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp�
$forward_lstm_20/lstm_cell_61/BiasAddBiasAdd$forward_lstm_20/lstm_cell_61/add:z:0;forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$forward_lstm_20/lstm_cell_61/BiasAdd�
,forward_lstm_20/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,forward_lstm_20/lstm_cell_61/split/split_dim�
"forward_lstm_20/lstm_cell_61/splitSplit5forward_lstm_20/lstm_cell_61/split/split_dim:output:0-forward_lstm_20/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2$
"forward_lstm_20/lstm_cell_61/split�
$forward_lstm_20/lstm_cell_61/SigmoidSigmoid+forward_lstm_20/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22&
$forward_lstm_20/lstm_cell_61/Sigmoid�
&forward_lstm_20/lstm_cell_61/Sigmoid_1Sigmoid+forward_lstm_20/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22(
&forward_lstm_20/lstm_cell_61/Sigmoid_1�
 forward_lstm_20/lstm_cell_61/mulMul*forward_lstm_20/lstm_cell_61/Sigmoid_1:y:0 forward_lstm_20/zeros_1:output:0*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/lstm_cell_61/mul�
!forward_lstm_20/lstm_cell_61/ReluRelu+forward_lstm_20/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22#
!forward_lstm_20/lstm_cell_61/Relu�
"forward_lstm_20/lstm_cell_61/mul_1Mul(forward_lstm_20/lstm_cell_61/Sigmoid:y:0/forward_lstm_20/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_20/lstm_cell_61/mul_1�
"forward_lstm_20/lstm_cell_61/add_1AddV2$forward_lstm_20/lstm_cell_61/mul:z:0&forward_lstm_20/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_20/lstm_cell_61/add_1�
&forward_lstm_20/lstm_cell_61/Sigmoid_2Sigmoid+forward_lstm_20/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22(
&forward_lstm_20/lstm_cell_61/Sigmoid_2�
#forward_lstm_20/lstm_cell_61/Relu_1Relu&forward_lstm_20/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22%
#forward_lstm_20/lstm_cell_61/Relu_1�
"forward_lstm_20/lstm_cell_61/mul_2Mul*forward_lstm_20/lstm_cell_61/Sigmoid_2:y:01forward_lstm_20/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_20/lstm_cell_61/mul_2�
-forward_lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2/
-forward_lstm_20/TensorArrayV2_1/element_shape�
forward_lstm_20/TensorArrayV2_1TensorListReserve6forward_lstm_20/TensorArrayV2_1/element_shape:output:0(forward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_20/TensorArrayV2_1n
forward_lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_20/time�
forward_lstm_20/zeros_like	ZerosLike&forward_lstm_20/lstm_cell_61/mul_2:z:0*
T0*'
_output_shapes
:���������22
forward_lstm_20/zeros_like�
(forward_lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(forward_lstm_20/while/maximum_iterations�
"forward_lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_20/while/loop_counter�
forward_lstm_20/whileWhile+forward_lstm_20/while/loop_counter:output:01forward_lstm_20/while/maximum_iterations:output:0forward_lstm_20/time:output:0(forward_lstm_20/TensorArrayV2_1:handle:0forward_lstm_20/zeros_like:y:0forward_lstm_20/zeros:output:0 forward_lstm_20/zeros_1:output:0(forward_lstm_20/strided_slice_1:output:0Gforward_lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_20/Cast:y:0;forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource=forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource<forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource*
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
"forward_lstm_20_while_body_2415930*.
cond&R$
"forward_lstm_20_while_cond_2415929*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
forward_lstm_20/while�
@forward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2B
@forward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape�
2forward_lstm_20/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_20/while:output:3Iforward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype024
2forward_lstm_20/TensorArrayV2Stack/TensorListStack�
%forward_lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2'
%forward_lstm_20/strided_slice_3/stack�
'forward_lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_20/strided_slice_3/stack_1�
'forward_lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_3/stack_2�
forward_lstm_20/strided_slice_3StridedSlice;forward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_20/strided_slice_3/stack:output:00forward_lstm_20/strided_slice_3/stack_1:output:00forward_lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2!
forward_lstm_20/strided_slice_3�
 forward_lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_20/transpose_1/perm�
forward_lstm_20/transpose_1	Transpose;forward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_20/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
forward_lstm_20/transpose_1�
forward_lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_20/runtime�
%backward_lstm_20/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%backward_lstm_20/RaggedToTensor/zeros�
%backward_lstm_20/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2'
%backward_lstm_20/RaggedToTensor/Const�
4backward_lstm_20/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.backward_lstm_20/RaggedToTensor/Const:output:0inputs.backward_lstm_20/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4backward_lstm_20/RaggedToTensor/RaggedTensorToTensor�
;backward_lstm_20/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack�
=backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1�
=backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2�
5backward_lstm_20/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dbackward_lstm_20/RaggedNestedRowLengths/strided_slice/stack:output:0Fbackward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fbackward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask27
5backward_lstm_20/RaggedNestedRowLengths/strided_slice�
=backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack�
?backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2A
?backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1�
?backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2�
7backward_lstm_20/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fbackward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hbackward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hbackward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask29
7backward_lstm_20/RaggedNestedRowLengths/strided_slice_1�
+backward_lstm_20/RaggedNestedRowLengths/subSub>backward_lstm_20/RaggedNestedRowLengths/strided_slice:output:0@backward_lstm_20/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2-
+backward_lstm_20/RaggedNestedRowLengths/sub�
backward_lstm_20/CastCast/backward_lstm_20/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
backward_lstm_20/Cast�
backward_lstm_20/ShapeShape=backward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_20/Shape�
$backward_lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_20/strided_slice/stack�
&backward_lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_20/strided_slice/stack_1�
&backward_lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_20/strided_slice/stack_2�
backward_lstm_20/strided_sliceStridedSlicebackward_lstm_20/Shape:output:0-backward_lstm_20/strided_slice/stack:output:0/backward_lstm_20/strided_slice/stack_1:output:0/backward_lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_20/strided_slice~
backward_lstm_20/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_20/zeros/mul/y�
backward_lstm_20/zeros/mulMul'backward_lstm_20/strided_slice:output:0%backward_lstm_20/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros/mul�
backward_lstm_20/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
backward_lstm_20/zeros/Less/y�
backward_lstm_20/zeros/LessLessbackward_lstm_20/zeros/mul:z:0&backward_lstm_20/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros/Less�
backward_lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_20/zeros/packed/1�
backward_lstm_20/zeros/packedPack'backward_lstm_20/strided_slice:output:0(backward_lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_20/zeros/packed�
backward_lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_20/zeros/Const�
backward_lstm_20/zerosFill&backward_lstm_20/zeros/packed:output:0%backward_lstm_20/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_20/zeros�
backward_lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_20/zeros_1/mul/y�
backward_lstm_20/zeros_1/mulMul'backward_lstm_20/strided_slice:output:0'backward_lstm_20/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros_1/mul�
backward_lstm_20/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2!
backward_lstm_20/zeros_1/Less/y�
backward_lstm_20/zeros_1/LessLess backward_lstm_20/zeros_1/mul:z:0(backward_lstm_20/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros_1/Less�
!backward_lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_20/zeros_1/packed/1�
backward_lstm_20/zeros_1/packedPack'backward_lstm_20/strided_slice:output:0*backward_lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_20/zeros_1/packed�
backward_lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_20/zeros_1/Const�
backward_lstm_20/zeros_1Fill(backward_lstm_20/zeros_1/packed:output:0'backward_lstm_20/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_20/zeros_1�
backward_lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_20/transpose/perm�
backward_lstm_20/transpose	Transpose=backward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0(backward_lstm_20/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_20/transpose�
backward_lstm_20/Shape_1Shapebackward_lstm_20/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_20/Shape_1�
&backward_lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_20/strided_slice_1/stack�
(backward_lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_1/stack_1�
(backward_lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_1/stack_2�
 backward_lstm_20/strided_slice_1StridedSlice!backward_lstm_20/Shape_1:output:0/backward_lstm_20/strided_slice_1/stack:output:01backward_lstm_20/strided_slice_1/stack_1:output:01backward_lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_20/strided_slice_1�
,backward_lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2.
,backward_lstm_20/TensorArrayV2/element_shape�
backward_lstm_20/TensorArrayV2TensorListReserve5backward_lstm_20/TensorArrayV2/element_shape:output:0)backward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_20/TensorArrayV2�
backward_lstm_20/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_20/ReverseV2/axis�
backward_lstm_20/ReverseV2	ReverseV2backward_lstm_20/transpose:y:0(backward_lstm_20/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_20/ReverseV2�
Fbackward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2H
Fbackward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape�
8backward_lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_20/ReverseV2:output:0Obackward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_20/TensorArrayUnstack/TensorListFromTensor�
&backward_lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_20/strided_slice_2/stack�
(backward_lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_2/stack_1�
(backward_lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_2/stack_2�
 backward_lstm_20/strided_slice_2StridedSlicebackward_lstm_20/transpose:y:0/backward_lstm_20/strided_slice_2/stack:output:01backward_lstm_20/strided_slice_2/stack_1:output:01backward_lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2"
 backward_lstm_20/strided_slice_2�
3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOpReadVariableOp<backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype025
3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp�
$backward_lstm_20/lstm_cell_62/MatMulMatMul)backward_lstm_20/strided_slice_2:output:0;backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$backward_lstm_20/lstm_cell_62/MatMul�
5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOp>backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype027
5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp�
&backward_lstm_20/lstm_cell_62/MatMul_1MatMulbackward_lstm_20/zeros:output:0=backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&backward_lstm_20/lstm_cell_62/MatMul_1�
!backward_lstm_20/lstm_cell_62/addAddV2.backward_lstm_20/lstm_cell_62/MatMul:product:00backward_lstm_20/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2#
!backward_lstm_20/lstm_cell_62/add�
4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOp=backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype026
4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp�
%backward_lstm_20/lstm_cell_62/BiasAddBiasAdd%backward_lstm_20/lstm_cell_62/add:z:0<backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%backward_lstm_20/lstm_cell_62/BiasAdd�
-backward_lstm_20/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-backward_lstm_20/lstm_cell_62/split/split_dim�
#backward_lstm_20/lstm_cell_62/splitSplit6backward_lstm_20/lstm_cell_62/split/split_dim:output:0.backward_lstm_20/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2%
#backward_lstm_20/lstm_cell_62/split�
%backward_lstm_20/lstm_cell_62/SigmoidSigmoid,backward_lstm_20/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22'
%backward_lstm_20/lstm_cell_62/Sigmoid�
'backward_lstm_20/lstm_cell_62/Sigmoid_1Sigmoid,backward_lstm_20/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22)
'backward_lstm_20/lstm_cell_62/Sigmoid_1�
!backward_lstm_20/lstm_cell_62/mulMul+backward_lstm_20/lstm_cell_62/Sigmoid_1:y:0!backward_lstm_20/zeros_1:output:0*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/lstm_cell_62/mul�
"backward_lstm_20/lstm_cell_62/ReluRelu,backward_lstm_20/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22$
"backward_lstm_20/lstm_cell_62/Relu�
#backward_lstm_20/lstm_cell_62/mul_1Mul)backward_lstm_20/lstm_cell_62/Sigmoid:y:00backward_lstm_20/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_20/lstm_cell_62/mul_1�
#backward_lstm_20/lstm_cell_62/add_1AddV2%backward_lstm_20/lstm_cell_62/mul:z:0'backward_lstm_20/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_20/lstm_cell_62/add_1�
'backward_lstm_20/lstm_cell_62/Sigmoid_2Sigmoid,backward_lstm_20/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22)
'backward_lstm_20/lstm_cell_62/Sigmoid_2�
$backward_lstm_20/lstm_cell_62/Relu_1Relu'backward_lstm_20/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22&
$backward_lstm_20/lstm_cell_62/Relu_1�
#backward_lstm_20/lstm_cell_62/mul_2Mul+backward_lstm_20/lstm_cell_62/Sigmoid_2:y:02backward_lstm_20/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_20/lstm_cell_62/mul_2�
.backward_lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   20
.backward_lstm_20/TensorArrayV2_1/element_shape�
 backward_lstm_20/TensorArrayV2_1TensorListReserve7backward_lstm_20/TensorArrayV2_1/element_shape:output:0)backward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_20/TensorArrayV2_1p
backward_lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_20/time�
&backward_lstm_20/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_lstm_20/Max/reduction_indices�
backward_lstm_20/MaxMaxbackward_lstm_20/Cast:y:0/backward_lstm_20/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/Maxr
backward_lstm_20/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_20/sub/y�
backward_lstm_20/subSubbackward_lstm_20/Max:output:0backward_lstm_20/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/sub�
backward_lstm_20/Sub_1Subbackward_lstm_20/sub:z:0backward_lstm_20/Cast:y:0*
T0*#
_output_shapes
:���������2
backward_lstm_20/Sub_1�
backward_lstm_20/zeros_like	ZerosLike'backward_lstm_20/lstm_cell_62/mul_2:z:0*
T0*'
_output_shapes
:���������22
backward_lstm_20/zeros_like�
)backward_lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2+
)backward_lstm_20/while/maximum_iterations�
#backward_lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_20/while/loop_counter�	
backward_lstm_20/whileWhile,backward_lstm_20/while/loop_counter:output:02backward_lstm_20/while/maximum_iterations:output:0backward_lstm_20/time:output:0)backward_lstm_20/TensorArrayV2_1:handle:0backward_lstm_20/zeros_like:y:0backward_lstm_20/zeros:output:0!backward_lstm_20/zeros_1:output:0)backward_lstm_20/strided_slice_1:output:0Hbackward_lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_20/Sub_1:z:0<backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource>backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource=backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource*
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
#backward_lstm_20_while_body_2416109*/
cond'R%
#backward_lstm_20_while_cond_2416108*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
backward_lstm_20/while�
Abackward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2C
Abackward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape�
3backward_lstm_20/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_20/while:output:3Jbackward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype025
3backward_lstm_20/TensorArrayV2Stack/TensorListStack�
&backward_lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2(
&backward_lstm_20/strided_slice_3/stack�
(backward_lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_20/strided_slice_3/stack_1�
(backward_lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_3/stack_2�
 backward_lstm_20/strided_slice_3StridedSlice<backward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_20/strided_slice_3/stack:output:01backward_lstm_20/strided_slice_3/stack_1:output:01backward_lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2"
 backward_lstm_20/strided_slice_3�
!backward_lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_20/transpose_1/perm�
backward_lstm_20/transpose_1	Transpose<backward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_20/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
backward_lstm_20/transpose_1�
backward_lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_20/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2(forward_lstm_20/strided_slice_3:output:0)backward_lstm_20/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp5^backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp4^backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp6^backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp^backward_lstm_20/while4^forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp3^forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp5^forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp^forward_lstm_20/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������: : : : : : 2l
4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp2j
3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp2n
5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp20
backward_lstm_20/whilebackward_lstm_20/while2j
3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp2h
2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp2l
4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp2.
forward_lstm_20/whileforward_lstm_20/while:O K
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
while_cond_2416638
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2416638___redundant_placeholder05
1while_while_cond_2416638___redundant_placeholder15
1while_while_cond_2416638___redundant_placeholder25
1while_while_cond_2416638___redundant_placeholder3
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
�\
�
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_2413710

inputs>
+lstm_cell_61_matmul_readvariableop_resource:	�@
-lstm_cell_61_matmul_1_readvariableop_resource:	2�;
,lstm_cell_61_biasadd_readvariableop_resource:	�
identity��#lstm_cell_61/BiasAdd/ReadVariableOp�"lstm_cell_61/MatMul/ReadVariableOp�$lstm_cell_61/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_61/MatMul/ReadVariableOpReadVariableOp+lstm_cell_61_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_61/MatMul/ReadVariableOp�
lstm_cell_61/MatMulMatMulstrided_slice_2:output:0*lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/MatMul�
$lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_61_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_61/MatMul_1/ReadVariableOp�
lstm_cell_61/MatMul_1MatMulzeros:output:0,lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/MatMul_1�
lstm_cell_61/addAddV2lstm_cell_61/MatMul:product:0lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/add�
#lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_61_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_61/BiasAdd/ReadVariableOp�
lstm_cell_61/BiasAddBiasAddlstm_cell_61/add:z:0+lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/BiasAdd~
lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_61/split/split_dim�
lstm_cell_61/splitSplit%lstm_cell_61/split/split_dim:output:0lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_61/split�
lstm_cell_61/SigmoidSigmoidlstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/Sigmoid�
lstm_cell_61/Sigmoid_1Sigmoidlstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_61/Sigmoid_1�
lstm_cell_61/mulMullstm_cell_61/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/mul}
lstm_cell_61/ReluRelulstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_61/Relu�
lstm_cell_61/mul_1Mullstm_cell_61/Sigmoid:y:0lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/mul_1�
lstm_cell_61/add_1AddV2lstm_cell_61/mul:z:0lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/add_1�
lstm_cell_61/Sigmoid_2Sigmoidlstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_61/Sigmoid_2|
lstm_cell_61/Relu_1Relulstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/Relu_1�
lstm_cell_61/mul_2Mullstm_cell_61/Sigmoid_2:y:0!lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_61_matmul_readvariableop_resource-lstm_cell_61_matmul_1_readvariableop_resource,lstm_cell_61_biasadd_readvariableop_resource*
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
while_body_2413626*
condR
while_cond_2413625*K
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
NoOpNoOp$^lstm_cell_61/BiasAdd/ReadVariableOp#^lstm_cell_61/MatMul/ReadVariableOp%^lstm_cell_61/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_61/BiasAdd/ReadVariableOp#lstm_cell_61/BiasAdd/ReadVariableOp2H
"lstm_cell_61/MatMul/ReadVariableOp"lstm_cell_61/MatMul/ReadVariableOp2L
$lstm_cell_61/MatMul_1/ReadVariableOp$lstm_cell_61/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�W
�
#backward_lstm_20_while_body_2415404>
:backward_lstm_20_while_backward_lstm_20_while_loop_counterD
@backward_lstm_20_while_backward_lstm_20_while_maximum_iterations&
"backward_lstm_20_while_placeholder(
$backward_lstm_20_while_placeholder_1(
$backward_lstm_20_while_placeholder_2(
$backward_lstm_20_while_placeholder_3=
9backward_lstm_20_while_backward_lstm_20_strided_slice_1_0y
ubackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0W
Dbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0:	�Y
Fbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0:	2�T
Ebackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0:	�#
backward_lstm_20_while_identity%
!backward_lstm_20_while_identity_1%
!backward_lstm_20_while_identity_2%
!backward_lstm_20_while_identity_3%
!backward_lstm_20_while_identity_4%
!backward_lstm_20_while_identity_5;
7backward_lstm_20_while_backward_lstm_20_strided_slice_1w
sbackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensorU
Bbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource:	�W
Dbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource:	2�R
Cbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource:	���:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp�9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp�;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp�
Hbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2J
Hbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape�
:backward_lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_20_while_placeholderQbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02<
:backward_lstm_20/while/TensorArrayV2Read/TensorListGetItem�
9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOpReadVariableOpDbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02;
9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp�
*backward_lstm_20/while/lstm_cell_62/MatMulMatMulAbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:0Abackward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*backward_lstm_20/while/lstm_cell_62/MatMul�
;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOpFbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02=
;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp�
,backward_lstm_20/while/lstm_cell_62/MatMul_1MatMul$backward_lstm_20_while_placeholder_2Cbackward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2.
,backward_lstm_20/while/lstm_cell_62/MatMul_1�
'backward_lstm_20/while/lstm_cell_62/addAddV24backward_lstm_20/while/lstm_cell_62/MatMul:product:06backward_lstm_20/while/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2)
'backward_lstm_20/while/lstm_cell_62/add�
:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOpEbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02<
:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp�
+backward_lstm_20/while/lstm_cell_62/BiasAddBiasAdd+backward_lstm_20/while/lstm_cell_62/add:z:0Bbackward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+backward_lstm_20/while/lstm_cell_62/BiasAdd�
3backward_lstm_20/while/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3backward_lstm_20/while/lstm_cell_62/split/split_dim�
)backward_lstm_20/while/lstm_cell_62/splitSplit<backward_lstm_20/while/lstm_cell_62/split/split_dim:output:04backward_lstm_20/while/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2+
)backward_lstm_20/while/lstm_cell_62/split�
+backward_lstm_20/while/lstm_cell_62/SigmoidSigmoid2backward_lstm_20/while/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22-
+backward_lstm_20/while/lstm_cell_62/Sigmoid�
-backward_lstm_20/while/lstm_cell_62/Sigmoid_1Sigmoid2backward_lstm_20/while/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22/
-backward_lstm_20/while/lstm_cell_62/Sigmoid_1�
'backward_lstm_20/while/lstm_cell_62/mulMul1backward_lstm_20/while/lstm_cell_62/Sigmoid_1:y:0$backward_lstm_20_while_placeholder_3*
T0*'
_output_shapes
:���������22)
'backward_lstm_20/while/lstm_cell_62/mul�
(backward_lstm_20/while/lstm_cell_62/ReluRelu2backward_lstm_20/while/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22*
(backward_lstm_20/while/lstm_cell_62/Relu�
)backward_lstm_20/while/lstm_cell_62/mul_1Mul/backward_lstm_20/while/lstm_cell_62/Sigmoid:y:06backward_lstm_20/while/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_20/while/lstm_cell_62/mul_1�
)backward_lstm_20/while/lstm_cell_62/add_1AddV2+backward_lstm_20/while/lstm_cell_62/mul:z:0-backward_lstm_20/while/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_20/while/lstm_cell_62/add_1�
-backward_lstm_20/while/lstm_cell_62/Sigmoid_2Sigmoid2backward_lstm_20/while/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22/
-backward_lstm_20/while/lstm_cell_62/Sigmoid_2�
*backward_lstm_20/while/lstm_cell_62/Relu_1Relu-backward_lstm_20/while/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22,
*backward_lstm_20/while/lstm_cell_62/Relu_1�
)backward_lstm_20/while/lstm_cell_62/mul_2Mul1backward_lstm_20/while/lstm_cell_62/Sigmoid_2:y:08backward_lstm_20/while/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_20/while/lstm_cell_62/mul_2�
;backward_lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_20_while_placeholder_1"backward_lstm_20_while_placeholder-backward_lstm_20/while/lstm_cell_62/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_20/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_20/while/add/y�
backward_lstm_20/while/addAddV2"backward_lstm_20_while_placeholder%backward_lstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/while/add�
backward_lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_20/while/add_1/y�
backward_lstm_20/while/add_1AddV2:backward_lstm_20_while_backward_lstm_20_while_loop_counter'backward_lstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/while/add_1�
backward_lstm_20/while/IdentityIdentity backward_lstm_20/while/add_1:z:0^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_20/while/Identity�
!backward_lstm_20/while/Identity_1Identity@backward_lstm_20_while_backward_lstm_20_while_maximum_iterations^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_20/while/Identity_1�
!backward_lstm_20/while/Identity_2Identitybackward_lstm_20/while/add:z:0^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_20/while/Identity_2�
!backward_lstm_20/while/Identity_3IdentityKbackward_lstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_20/while/Identity_3�
!backward_lstm_20/while/Identity_4Identity-backward_lstm_20/while/lstm_cell_62/mul_2:z:0^backward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/while/Identity_4�
!backward_lstm_20/while/Identity_5Identity-backward_lstm_20/while/lstm_cell_62/add_1:z:0^backward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/while/Identity_5�
backward_lstm_20/while/NoOpNoOp;^backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp:^backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp<^backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_20/while/NoOp"t
7backward_lstm_20_while_backward_lstm_20_strided_slice_19backward_lstm_20_while_backward_lstm_20_strided_slice_1_0"K
backward_lstm_20_while_identity(backward_lstm_20/while/Identity:output:0"O
!backward_lstm_20_while_identity_1*backward_lstm_20/while/Identity_1:output:0"O
!backward_lstm_20_while_identity_2*backward_lstm_20/while/Identity_2:output:0"O
!backward_lstm_20_while_identity_3*backward_lstm_20/while/Identity_3:output:0"O
!backward_lstm_20_while_identity_4*backward_lstm_20/while/Identity_4:output:0"O
!backward_lstm_20_while_identity_5*backward_lstm_20/while/Identity_5:output:0"�
Cbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resourceEbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0"�
Dbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resourceFbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0"�
Bbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resourceDbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0"�
sbackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensorubackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2x
:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp2v
9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp2z
;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp: 
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
J__inference_sequential_20_layer_call_and_return_conditional_losses_2414699

inputs
inputs_1	+
bidirectional_20_2414680:	�+
bidirectional_20_2414682:	2�'
bidirectional_20_2414684:	�+
bidirectional_20_2414686:	�+
bidirectional_20_2414688:	2�'
bidirectional_20_2414690:	�"
dense_20_2414693:d
dense_20_2414695:
identity��(bidirectional_20/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�
(bidirectional_20/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_20_2414680bidirectional_20_2414682bidirectional_20_2414684bidirectional_20_2414686bidirectional_20_2414688bidirectional_20_2414690*
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
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_24146362*
(bidirectional_20/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_20/StatefulPartitionedCall:output:0dense_20_2414693dense_20_2414695*
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
E__inference_dense_20_layer_call_and_return_conditional_losses_24142212"
 dense_20/StatefulPartitionedCall�
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp)^bidirectional_20/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:���������:���������: : : : : : : : 2T
(bidirectional_20/StatefulPartitionedCall(bidirectional_20/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall:O K
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
while_cond_2416789
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2416789___redundant_placeholder05
1while_while_cond_2416789___redundant_placeholder15
1while_while_cond_2416789___redundant_placeholder25
1while_while_cond_2416789___redundant_placeholder3
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
�
1__inference_forward_lstm_20_layer_call_fn_2416259

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
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_24131852
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
while_body_2412481
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_62_2412505_0:	�/
while_lstm_cell_62_2412507_0:	2�+
while_lstm_cell_62_2412509_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_62_2412505:	�-
while_lstm_cell_62_2412507:	2�)
while_lstm_cell_62_2412509:	���*while/lstm_cell_62/StatefulPartitionedCall�
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
*while/lstm_cell_62/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_62_2412505_0while_lstm_cell_62_2412507_0while_lstm_cell_62_2412509_0*
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
I__inference_lstm_cell_62_layer_call_and_return_conditional_losses_24124672,
*while/lstm_cell_62/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_62/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_62/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_62/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_62/StatefulPartitionedCall*"
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
while_lstm_cell_62_2412505while_lstm_cell_62_2412505_0":
while_lstm_cell_62_2412507while_lstm_cell_62_2412507_0":
while_lstm_cell_62_2412509while_lstm_cell_62_2412509_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2X
*while/lstm_cell_62/StatefulPartitionedCall*while/lstm_cell_62/StatefulPartitionedCall: 
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
while_body_2417293
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_62_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_62_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_62_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_62_matmul_readvariableop_resource:	�F
3while_lstm_cell_62_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_62_biasadd_readvariableop_resource:	���)while/lstm_cell_62/BiasAdd/ReadVariableOp�(while/lstm_cell_62/MatMul/ReadVariableOp�*while/lstm_cell_62/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_62/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_62_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_62/MatMul/ReadVariableOp�
while/lstm_cell_62/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/MatMul�
*while/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_62_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_62/MatMul_1/ReadVariableOp�
while/lstm_cell_62/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/MatMul_1�
while/lstm_cell_62/addAddV2#while/lstm_cell_62/MatMul:product:0%while/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/add�
)while/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_62_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_62/BiasAdd/ReadVariableOp�
while/lstm_cell_62/BiasAddBiasAddwhile/lstm_cell_62/add:z:01while/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/BiasAdd�
"while/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_62/split/split_dim�
while/lstm_cell_62/splitSplit+while/lstm_cell_62/split/split_dim:output:0#while/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_62/split�
while/lstm_cell_62/SigmoidSigmoid!while/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Sigmoid�
while/lstm_cell_62/Sigmoid_1Sigmoid!while/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Sigmoid_1�
while/lstm_cell_62/mulMul while/lstm_cell_62/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/mul�
while/lstm_cell_62/ReluRelu!while/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Relu�
while/lstm_cell_62/mul_1Mulwhile/lstm_cell_62/Sigmoid:y:0%while/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/mul_1�
while/lstm_cell_62/add_1AddV2while/lstm_cell_62/mul:z:0while/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/add_1�
while/lstm_cell_62/Sigmoid_2Sigmoid!while/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Sigmoid_2�
while/lstm_cell_62/Relu_1Reluwhile/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Relu_1�
while/lstm_cell_62/mul_2Mul while/lstm_cell_62/Sigmoid_2:y:0'while/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_62/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_62/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_62/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_62/BiasAdd/ReadVariableOp)^while/lstm_cell_62/MatMul/ReadVariableOp+^while/lstm_cell_62/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_62_biasadd_readvariableop_resource4while_lstm_cell_62_biasadd_readvariableop_resource_0"l
3while_lstm_cell_62_matmul_1_readvariableop_resource5while_lstm_cell_62_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_62_matmul_readvariableop_resource3while_lstm_cell_62_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_62/BiasAdd/ReadVariableOp)while/lstm_cell_62/BiasAdd/ReadVariableOp2T
(while/lstm_cell_62/MatMul/ReadVariableOp(while/lstm_cell_62/MatMul/ReadVariableOp2X
*while/lstm_cell_62/MatMul_1/ReadVariableOp*while/lstm_cell_62/MatMul_1/ReadVariableOp: 
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
�V
�
"forward_lstm_20_while_body_2414953<
8forward_lstm_20_while_forward_lstm_20_while_loop_counterB
>forward_lstm_20_while_forward_lstm_20_while_maximum_iterations%
!forward_lstm_20_while_placeholder'
#forward_lstm_20_while_placeholder_1'
#forward_lstm_20_while_placeholder_2'
#forward_lstm_20_while_placeholder_3;
7forward_lstm_20_while_forward_lstm_20_strided_slice_1_0w
sforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0V
Cforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0:	�X
Eforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0:	2�S
Dforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0:	�"
forward_lstm_20_while_identity$
 forward_lstm_20_while_identity_1$
 forward_lstm_20_while_identity_2$
 forward_lstm_20_while_identity_3$
 forward_lstm_20_while_identity_4$
 forward_lstm_20_while_identity_59
5forward_lstm_20_while_forward_lstm_20_strided_slice_1u
qforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensorT
Aforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource:	�V
Cforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource:	2�Q
Bforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource:	���9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp�8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp�:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp�
Gforward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2I
Gforward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape�
9forward_lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_20_while_placeholderPforward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02;
9forward_lstm_20/while/TensorArrayV2Read/TensorListGetItem�
8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOpReadVariableOpCforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02:
8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp�
)forward_lstm_20/while/lstm_cell_61/MatMulMatMul@forward_lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:0@forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)forward_lstm_20/while/lstm_cell_61/MatMul�
:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOpEforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02<
:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp�
+forward_lstm_20/while/lstm_cell_61/MatMul_1MatMul#forward_lstm_20_while_placeholder_2Bforward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+forward_lstm_20/while/lstm_cell_61/MatMul_1�
&forward_lstm_20/while/lstm_cell_61/addAddV23forward_lstm_20/while/lstm_cell_61/MatMul:product:05forward_lstm_20/while/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&forward_lstm_20/while/lstm_cell_61/add�
9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOpDforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02;
9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp�
*forward_lstm_20/while/lstm_cell_61/BiasAddBiasAdd*forward_lstm_20/while/lstm_cell_61/add:z:0Aforward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*forward_lstm_20/while/lstm_cell_61/BiasAdd�
2forward_lstm_20/while/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2forward_lstm_20/while/lstm_cell_61/split/split_dim�
(forward_lstm_20/while/lstm_cell_61/splitSplit;forward_lstm_20/while/lstm_cell_61/split/split_dim:output:03forward_lstm_20/while/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2*
(forward_lstm_20/while/lstm_cell_61/split�
*forward_lstm_20/while/lstm_cell_61/SigmoidSigmoid1forward_lstm_20/while/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22,
*forward_lstm_20/while/lstm_cell_61/Sigmoid�
,forward_lstm_20/while/lstm_cell_61/Sigmoid_1Sigmoid1forward_lstm_20/while/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22.
,forward_lstm_20/while/lstm_cell_61/Sigmoid_1�
&forward_lstm_20/while/lstm_cell_61/mulMul0forward_lstm_20/while/lstm_cell_61/Sigmoid_1:y:0#forward_lstm_20_while_placeholder_3*
T0*'
_output_shapes
:���������22(
&forward_lstm_20/while/lstm_cell_61/mul�
'forward_lstm_20/while/lstm_cell_61/ReluRelu1forward_lstm_20/while/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22)
'forward_lstm_20/while/lstm_cell_61/Relu�
(forward_lstm_20/while/lstm_cell_61/mul_1Mul.forward_lstm_20/while/lstm_cell_61/Sigmoid:y:05forward_lstm_20/while/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_20/while/lstm_cell_61/mul_1�
(forward_lstm_20/while/lstm_cell_61/add_1AddV2*forward_lstm_20/while/lstm_cell_61/mul:z:0,forward_lstm_20/while/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_20/while/lstm_cell_61/add_1�
,forward_lstm_20/while/lstm_cell_61/Sigmoid_2Sigmoid1forward_lstm_20/while/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22.
,forward_lstm_20/while/lstm_cell_61/Sigmoid_2�
)forward_lstm_20/while/lstm_cell_61/Relu_1Relu,forward_lstm_20/while/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22+
)forward_lstm_20/while/lstm_cell_61/Relu_1�
(forward_lstm_20/while/lstm_cell_61/mul_2Mul0forward_lstm_20/while/lstm_cell_61/Sigmoid_2:y:07forward_lstm_20/while/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_20/while/lstm_cell_61/mul_2�
:forward_lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_20_while_placeholder_1!forward_lstm_20_while_placeholder,forward_lstm_20/while/lstm_cell_61/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_20/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_20/while/add/y�
forward_lstm_20/while/addAddV2!forward_lstm_20_while_placeholder$forward_lstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/while/add�
forward_lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_20/while/add_1/y�
forward_lstm_20/while/add_1AddV28forward_lstm_20_while_forward_lstm_20_while_loop_counter&forward_lstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/while/add_1�
forward_lstm_20/while/IdentityIdentityforward_lstm_20/while/add_1:z:0^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_20/while/Identity�
 forward_lstm_20/while/Identity_1Identity>forward_lstm_20_while_forward_lstm_20_while_maximum_iterations^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_20/while/Identity_1�
 forward_lstm_20/while/Identity_2Identityforward_lstm_20/while/add:z:0^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_20/while/Identity_2�
 forward_lstm_20/while/Identity_3IdentityJforward_lstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_20/while/Identity_3�
 forward_lstm_20/while/Identity_4Identity,forward_lstm_20/while/lstm_cell_61/mul_2:z:0^forward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/while/Identity_4�
 forward_lstm_20/while/Identity_5Identity,forward_lstm_20/while/lstm_cell_61/add_1:z:0^forward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/while/Identity_5�
forward_lstm_20/while/NoOpNoOp:^forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp9^forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp;^forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_20/while/NoOp"p
5forward_lstm_20_while_forward_lstm_20_strided_slice_17forward_lstm_20_while_forward_lstm_20_strided_slice_1_0"I
forward_lstm_20_while_identity'forward_lstm_20/while/Identity:output:0"M
 forward_lstm_20_while_identity_1)forward_lstm_20/while/Identity_1:output:0"M
 forward_lstm_20_while_identity_2)forward_lstm_20/while/Identity_2:output:0"M
 forward_lstm_20_while_identity_3)forward_lstm_20/while/Identity_3:output:0"M
 forward_lstm_20_while_identity_4)forward_lstm_20/while/Identity_4:output:0"M
 forward_lstm_20_while_identity_5)forward_lstm_20/while/Identity_5:output:0"�
Bforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resourceDforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0"�
Cforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resourceEforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0"�
Aforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resourceCforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0"�
qforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensorsforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2v
9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp2t
8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp2x
:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp: 
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
while_body_2417140
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_62_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_62_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_62_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_62_matmul_readvariableop_resource:	�F
3while_lstm_cell_62_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_62_biasadd_readvariableop_resource:	���)while/lstm_cell_62/BiasAdd/ReadVariableOp�(while/lstm_cell_62/MatMul/ReadVariableOp�*while/lstm_cell_62/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_62/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_62_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_62/MatMul/ReadVariableOp�
while/lstm_cell_62/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/MatMul�
*while/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_62_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_62/MatMul_1/ReadVariableOp�
while/lstm_cell_62/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/MatMul_1�
while/lstm_cell_62/addAddV2#while/lstm_cell_62/MatMul:product:0%while/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/add�
)while/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_62_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_62/BiasAdd/ReadVariableOp�
while/lstm_cell_62/BiasAddBiasAddwhile/lstm_cell_62/add:z:01while/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/BiasAdd�
"while/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_62/split/split_dim�
while/lstm_cell_62/splitSplit+while/lstm_cell_62/split/split_dim:output:0#while/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_62/split�
while/lstm_cell_62/SigmoidSigmoid!while/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Sigmoid�
while/lstm_cell_62/Sigmoid_1Sigmoid!while/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Sigmoid_1�
while/lstm_cell_62/mulMul while/lstm_cell_62/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/mul�
while/lstm_cell_62/ReluRelu!while/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Relu�
while/lstm_cell_62/mul_1Mulwhile/lstm_cell_62/Sigmoid:y:0%while/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/mul_1�
while/lstm_cell_62/add_1AddV2while/lstm_cell_62/mul:z:0while/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/add_1�
while/lstm_cell_62/Sigmoid_2Sigmoid!while/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Sigmoid_2�
while/lstm_cell_62/Relu_1Reluwhile/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Relu_1�
while/lstm_cell_62/mul_2Mul while/lstm_cell_62/Sigmoid_2:y:0'while/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_62/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_62/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_62/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_62/BiasAdd/ReadVariableOp)^while/lstm_cell_62/MatMul/ReadVariableOp+^while/lstm_cell_62/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_62_biasadd_readvariableop_resource4while_lstm_cell_62_biasadd_readvariableop_resource_0"l
3while_lstm_cell_62_matmul_1_readvariableop_resource5while_lstm_cell_62_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_62_matmul_readvariableop_resource3while_lstm_cell_62_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_62/BiasAdd/ReadVariableOp)while/lstm_cell_62/BiasAdd/ReadVariableOp2T
(while/lstm_cell_62/MatMul/ReadVariableOp(while/lstm_cell_62/MatMul/ReadVariableOp2X
*while/lstm_cell_62/MatMul_1/ReadVariableOp*while/lstm_cell_62/MatMul_1/ReadVariableOp: 
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
while_cond_2412058
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2412058___redundant_placeholder05
1while_while_cond_2412058___redundant_placeholder15
1while_while_cond_2412058___redundant_placeholder25
1while_while_cond_2412058___redundant_placeholder3
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
I__inference_lstm_cell_61_layer_call_and_return_conditional_losses_2417596

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
�
�
"forward_lstm_20_while_cond_2413919<
8forward_lstm_20_while_forward_lstm_20_while_loop_counterB
>forward_lstm_20_while_forward_lstm_20_while_maximum_iterations%
!forward_lstm_20_while_placeholder'
#forward_lstm_20_while_placeholder_1'
#forward_lstm_20_while_placeholder_2'
#forward_lstm_20_while_placeholder_3'
#forward_lstm_20_while_placeholder_4>
:forward_lstm_20_while_less_forward_lstm_20_strided_slice_1U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2413919___redundant_placeholder0U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2413919___redundant_placeholder1U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2413919___redundant_placeholder2U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2413919___redundant_placeholder3U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2413919___redundant_placeholder4"
forward_lstm_20_while_identity
�
forward_lstm_20/while/LessLess!forward_lstm_20_while_placeholder:forward_lstm_20_while_less_forward_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_20/while/Less�
forward_lstm_20/while/IdentityIdentityforward_lstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_20/while/Identity"I
forward_lstm_20_while_identity'forward_lstm_20/while/Identity:output:0*(
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
�
�
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_2413356

inputs*
forward_lstm_20_2413186:	�*
forward_lstm_20_2413188:	2�&
forward_lstm_20_2413190:	�+
backward_lstm_20_2413346:	�+
backward_lstm_20_2413348:	2�'
backward_lstm_20_2413350:	�
identity��(backward_lstm_20/StatefulPartitionedCall�'forward_lstm_20/StatefulPartitionedCall�
'forward_lstm_20/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_20_2413186forward_lstm_20_2413188forward_lstm_20_2413190*
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
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_24131852)
'forward_lstm_20/StatefulPartitionedCall�
(backward_lstm_20/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_20_2413346backward_lstm_20_2413348backward_lstm_20_2413350*
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
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_24133452*
(backward_lstm_20/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV20forward_lstm_20/StatefulPartitionedCall:output:01backward_lstm_20/StatefulPartitionedCall:output:0concat/axis:output:0*
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
NoOpNoOp)^backward_lstm_20/StatefulPartitionedCall(^forward_lstm_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 2T
(backward_lstm_20/StatefulPartitionedCall(backward_lstm_20/StatefulPartitionedCall2R
'forward_lstm_20/StatefulPartitionedCall'forward_lstm_20/StatefulPartitionedCall:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�^
�
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_2413345

inputs>
+lstm_cell_62_matmul_readvariableop_resource:	�@
-lstm_cell_62_matmul_1_readvariableop_resource:	2�;
,lstm_cell_62_biasadd_readvariableop_resource:	�
identity��#lstm_cell_62/BiasAdd/ReadVariableOp�"lstm_cell_62/MatMul/ReadVariableOp�$lstm_cell_62/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_62/MatMul/ReadVariableOpReadVariableOp+lstm_cell_62_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_62/MatMul/ReadVariableOp�
lstm_cell_62/MatMulMatMulstrided_slice_2:output:0*lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/MatMul�
$lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_62_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_62/MatMul_1/ReadVariableOp�
lstm_cell_62/MatMul_1MatMulzeros:output:0,lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/MatMul_1�
lstm_cell_62/addAddV2lstm_cell_62/MatMul:product:0lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/add�
#lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_62/BiasAdd/ReadVariableOp�
lstm_cell_62/BiasAddBiasAddlstm_cell_62/add:z:0+lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/BiasAdd~
lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_62/split/split_dim�
lstm_cell_62/splitSplit%lstm_cell_62/split/split_dim:output:0lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_62/split�
lstm_cell_62/SigmoidSigmoidlstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/Sigmoid�
lstm_cell_62/Sigmoid_1Sigmoidlstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_62/Sigmoid_1�
lstm_cell_62/mulMullstm_cell_62/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/mul}
lstm_cell_62/ReluRelulstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_62/Relu�
lstm_cell_62/mul_1Mullstm_cell_62/Sigmoid:y:0lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/mul_1�
lstm_cell_62/add_1AddV2lstm_cell_62/mul:z:0lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/add_1�
lstm_cell_62/Sigmoid_2Sigmoidlstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_62/Sigmoid_2|
lstm_cell_62/Relu_1Relulstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/Relu_1�
lstm_cell_62/mul_2Mullstm_cell_62/Sigmoid_2:y:0!lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_62_matmul_readvariableop_resource-lstm_cell_62_matmul_1_readvariableop_resource,lstm_cell_62_biasadd_readvariableop_resource*
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
while_body_2413261*
condR
while_cond_2413260*K
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
NoOpNoOp$^lstm_cell_62/BiasAdd/ReadVariableOp#^lstm_cell_62/MatMul/ReadVariableOp%^lstm_cell_62/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_62/BiasAdd/ReadVariableOp#lstm_cell_62/BiasAdd/ReadVariableOp2H
"lstm_cell_62/MatMul/ReadVariableOp"lstm_cell_62/MatMul/ReadVariableOp2L
$lstm_cell_62/MatMul_1/ReadVariableOp$lstm_cell_62/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�	
�
2__inference_bidirectional_20_layer_call_fn_2414850
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
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_24137582
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
�
�
I__inference_lstm_cell_61_layer_call_and_return_conditional_losses_2417628

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
�
�
while_cond_2416487
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2416487___redundant_placeholder05
1while_while_cond_2416487___redundant_placeholder15
1while_while_cond_2416487___redundant_placeholder25
1while_while_cond_2416487___redundant_placeholder3
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
"forward_lstm_20_while_body_2413920<
8forward_lstm_20_while_forward_lstm_20_while_loop_counterB
>forward_lstm_20_while_forward_lstm_20_while_maximum_iterations%
!forward_lstm_20_while_placeholder'
#forward_lstm_20_while_placeholder_1'
#forward_lstm_20_while_placeholder_2'
#forward_lstm_20_while_placeholder_3'
#forward_lstm_20_while_placeholder_4;
7forward_lstm_20_while_forward_lstm_20_strided_slice_1_0w
sforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_08
4forward_lstm_20_while_greater_forward_lstm_20_cast_0V
Cforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0:	�X
Eforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0:	2�S
Dforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0:	�"
forward_lstm_20_while_identity$
 forward_lstm_20_while_identity_1$
 forward_lstm_20_while_identity_2$
 forward_lstm_20_while_identity_3$
 forward_lstm_20_while_identity_4$
 forward_lstm_20_while_identity_5$
 forward_lstm_20_while_identity_69
5forward_lstm_20_while_forward_lstm_20_strided_slice_1u
qforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor6
2forward_lstm_20_while_greater_forward_lstm_20_castT
Aforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource:	�V
Cforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource:	2�Q
Bforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource:	���9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp�8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp�:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp�
Gforward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2I
Gforward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape�
9forward_lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_20_while_placeholderPforward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02;
9forward_lstm_20/while/TensorArrayV2Read/TensorListGetItem�
forward_lstm_20/while/GreaterGreater4forward_lstm_20_while_greater_forward_lstm_20_cast_0!forward_lstm_20_while_placeholder*
T0*#
_output_shapes
:���������2
forward_lstm_20/while/Greater�
8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOpReadVariableOpCforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02:
8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp�
)forward_lstm_20/while/lstm_cell_61/MatMulMatMul@forward_lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:0@forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)forward_lstm_20/while/lstm_cell_61/MatMul�
:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOpEforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02<
:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp�
+forward_lstm_20/while/lstm_cell_61/MatMul_1MatMul#forward_lstm_20_while_placeholder_3Bforward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+forward_lstm_20/while/lstm_cell_61/MatMul_1�
&forward_lstm_20/while/lstm_cell_61/addAddV23forward_lstm_20/while/lstm_cell_61/MatMul:product:05forward_lstm_20/while/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&forward_lstm_20/while/lstm_cell_61/add�
9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOpDforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02;
9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp�
*forward_lstm_20/while/lstm_cell_61/BiasAddBiasAdd*forward_lstm_20/while/lstm_cell_61/add:z:0Aforward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*forward_lstm_20/while/lstm_cell_61/BiasAdd�
2forward_lstm_20/while/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2forward_lstm_20/while/lstm_cell_61/split/split_dim�
(forward_lstm_20/while/lstm_cell_61/splitSplit;forward_lstm_20/while/lstm_cell_61/split/split_dim:output:03forward_lstm_20/while/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2*
(forward_lstm_20/while/lstm_cell_61/split�
*forward_lstm_20/while/lstm_cell_61/SigmoidSigmoid1forward_lstm_20/while/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22,
*forward_lstm_20/while/lstm_cell_61/Sigmoid�
,forward_lstm_20/while/lstm_cell_61/Sigmoid_1Sigmoid1forward_lstm_20/while/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22.
,forward_lstm_20/while/lstm_cell_61/Sigmoid_1�
&forward_lstm_20/while/lstm_cell_61/mulMul0forward_lstm_20/while/lstm_cell_61/Sigmoid_1:y:0#forward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22(
&forward_lstm_20/while/lstm_cell_61/mul�
'forward_lstm_20/while/lstm_cell_61/ReluRelu1forward_lstm_20/while/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22)
'forward_lstm_20/while/lstm_cell_61/Relu�
(forward_lstm_20/while/lstm_cell_61/mul_1Mul.forward_lstm_20/while/lstm_cell_61/Sigmoid:y:05forward_lstm_20/while/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_20/while/lstm_cell_61/mul_1�
(forward_lstm_20/while/lstm_cell_61/add_1AddV2*forward_lstm_20/while/lstm_cell_61/mul:z:0,forward_lstm_20/while/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_20/while/lstm_cell_61/add_1�
,forward_lstm_20/while/lstm_cell_61/Sigmoid_2Sigmoid1forward_lstm_20/while/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22.
,forward_lstm_20/while/lstm_cell_61/Sigmoid_2�
)forward_lstm_20/while/lstm_cell_61/Relu_1Relu,forward_lstm_20/while/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22+
)forward_lstm_20/while/lstm_cell_61/Relu_1�
(forward_lstm_20/while/lstm_cell_61/mul_2Mul0forward_lstm_20/while/lstm_cell_61/Sigmoid_2:y:07forward_lstm_20/while/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_20/while/lstm_cell_61/mul_2�
forward_lstm_20/while/SelectSelect!forward_lstm_20/while/Greater:z:0,forward_lstm_20/while/lstm_cell_61/mul_2:z:0#forward_lstm_20_while_placeholder_2*
T0*'
_output_shapes
:���������22
forward_lstm_20/while/Select�
forward_lstm_20/while/Select_1Select!forward_lstm_20/while/Greater:z:0,forward_lstm_20/while/lstm_cell_61/mul_2:z:0#forward_lstm_20_while_placeholder_3*
T0*'
_output_shapes
:���������22 
forward_lstm_20/while/Select_1�
forward_lstm_20/while/Select_2Select!forward_lstm_20/while/Greater:z:0,forward_lstm_20/while/lstm_cell_61/add_1:z:0#forward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22 
forward_lstm_20/while/Select_2�
:forward_lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_20_while_placeholder_1!forward_lstm_20_while_placeholder%forward_lstm_20/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_20/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_20/while/add/y�
forward_lstm_20/while/addAddV2!forward_lstm_20_while_placeholder$forward_lstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/while/add�
forward_lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_20/while/add_1/y�
forward_lstm_20/while/add_1AddV28forward_lstm_20_while_forward_lstm_20_while_loop_counter&forward_lstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/while/add_1�
forward_lstm_20/while/IdentityIdentityforward_lstm_20/while/add_1:z:0^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_20/while/Identity�
 forward_lstm_20/while/Identity_1Identity>forward_lstm_20_while_forward_lstm_20_while_maximum_iterations^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_20/while/Identity_1�
 forward_lstm_20/while/Identity_2Identityforward_lstm_20/while/add:z:0^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_20/while/Identity_2�
 forward_lstm_20/while/Identity_3IdentityJforward_lstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_20/while/Identity_3�
 forward_lstm_20/while/Identity_4Identity%forward_lstm_20/while/Select:output:0^forward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/while/Identity_4�
 forward_lstm_20/while/Identity_5Identity'forward_lstm_20/while/Select_1:output:0^forward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/while/Identity_5�
 forward_lstm_20/while/Identity_6Identity'forward_lstm_20/while/Select_2:output:0^forward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/while/Identity_6�
forward_lstm_20/while/NoOpNoOp:^forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp9^forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp;^forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_20/while/NoOp"p
5forward_lstm_20_while_forward_lstm_20_strided_slice_17forward_lstm_20_while_forward_lstm_20_strided_slice_1_0"j
2forward_lstm_20_while_greater_forward_lstm_20_cast4forward_lstm_20_while_greater_forward_lstm_20_cast_0"I
forward_lstm_20_while_identity'forward_lstm_20/while/Identity:output:0"M
 forward_lstm_20_while_identity_1)forward_lstm_20/while/Identity_1:output:0"M
 forward_lstm_20_while_identity_2)forward_lstm_20/while/Identity_2:output:0"M
 forward_lstm_20_while_identity_3)forward_lstm_20/while/Identity_3:output:0"M
 forward_lstm_20_while_identity_4)forward_lstm_20/while/Identity_4:output:0"M
 forward_lstm_20_while_identity_5)forward_lstm_20/while/Identity_5:output:0"M
 forward_lstm_20_while_identity_6)forward_lstm_20/while/Identity_6:output:0"�
Bforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resourceDforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0"�
Cforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resourceEforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0"�
Aforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resourceCforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0"�
qforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensorsforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2v
9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp2t
8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp2x
:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp: 
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
�?
�
while_body_2416488
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_61_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_61_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_61_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_61_matmul_readvariableop_resource:	�F
3while_lstm_cell_61_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_61_biasadd_readvariableop_resource:	���)while/lstm_cell_61/BiasAdd/ReadVariableOp�(while/lstm_cell_61/MatMul/ReadVariableOp�*while/lstm_cell_61/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_61/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_61_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_61/MatMul/ReadVariableOp�
while/lstm_cell_61/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/MatMul�
*while/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_61_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_61/MatMul_1/ReadVariableOp�
while/lstm_cell_61/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/MatMul_1�
while/lstm_cell_61/addAddV2#while/lstm_cell_61/MatMul:product:0%while/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/add�
)while/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_61_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_61/BiasAdd/ReadVariableOp�
while/lstm_cell_61/BiasAddBiasAddwhile/lstm_cell_61/add:z:01while/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/BiasAdd�
"while/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_61/split/split_dim�
while/lstm_cell_61/splitSplit+while/lstm_cell_61/split/split_dim:output:0#while/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_61/split�
while/lstm_cell_61/SigmoidSigmoid!while/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Sigmoid�
while/lstm_cell_61/Sigmoid_1Sigmoid!while/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Sigmoid_1�
while/lstm_cell_61/mulMul while/lstm_cell_61/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/mul�
while/lstm_cell_61/ReluRelu!while/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Relu�
while/lstm_cell_61/mul_1Mulwhile/lstm_cell_61/Sigmoid:y:0%while/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/mul_1�
while/lstm_cell_61/add_1AddV2while/lstm_cell_61/mul:z:0while/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/add_1�
while/lstm_cell_61/Sigmoid_2Sigmoid!while/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Sigmoid_2�
while/lstm_cell_61/Relu_1Reluwhile/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Relu_1�
while/lstm_cell_61/mul_2Mul while/lstm_cell_61/Sigmoid_2:y:0'while/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_61/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_61/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_61/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_61/BiasAdd/ReadVariableOp)^while/lstm_cell_61/MatMul/ReadVariableOp+^while/lstm_cell_61/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_61_biasadd_readvariableop_resource4while_lstm_cell_61_biasadd_readvariableop_resource_0"l
3while_lstm_cell_61_matmul_1_readvariableop_resource5while_lstm_cell_61_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_61_matmul_readvariableop_resource3while_lstm_cell_61_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_61/BiasAdd/ReadVariableOp)while/lstm_cell_61/BiasAdd/ReadVariableOp2T
(while/lstm_cell_61/MatMul/ReadVariableOp(while/lstm_cell_61/MatMul/ReadVariableOp2X
*while/lstm_cell_61/MatMul_1/ReadVariableOp*while/lstm_cell_61/MatMul_1/ReadVariableOp: 
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
while_cond_2417292
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2417292___redundant_placeholder05
1while_while_cond_2417292___redundant_placeholder15
1while_while_cond_2417292___redundant_placeholder25
1while_while_cond_2417292___redundant_placeholder3
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
"__inference__wrapped_model_2411760

args_0
args_0_1	m
Zsequential_20_bidirectional_20_forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource:	�o
\sequential_20_bidirectional_20_forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource:	2�j
[sequential_20_bidirectional_20_forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource:	�n
[sequential_20_bidirectional_20_backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource:	�p
]sequential_20_bidirectional_20_backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource:	2�k
\sequential_20_bidirectional_20_backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource:	�G
5sequential_20_dense_20_matmul_readvariableop_resource:dD
6sequential_20_dense_20_biasadd_readvariableop_resource:
identity��Ssequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp�Rsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp�Tsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp�5sequential_20/bidirectional_20/backward_lstm_20/while�Rsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp�Qsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp�Ssequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp�4sequential_20/bidirectional_20/forward_lstm_20/while�-sequential_20/dense_20/BiasAdd/ReadVariableOp�,sequential_20/dense_20/MatMul/ReadVariableOp�
Csequential_20/bidirectional_20/forward_lstm_20/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2E
Csequential_20/bidirectional_20/forward_lstm_20/RaggedToTensor/zeros�
Csequential_20/bidirectional_20/forward_lstm_20/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2E
Csequential_20/bidirectional_20/forward_lstm_20/RaggedToTensor/Const�
Rsequential_20/bidirectional_20/forward_lstm_20/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorLsequential_20/bidirectional_20/forward_lstm_20/RaggedToTensor/Const:output:0args_0Lsequential_20/bidirectional_20/forward_lstm_20/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2T
Rsequential_20/bidirectional_20/forward_lstm_20/RaggedToTensor/RaggedTensorToTensor�
Ysequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2[
Ysequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack�
[sequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2]
[sequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1�
[sequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2]
[sequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2�
Ssequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1bsequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack:output:0dsequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1:output:0dsequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask2U
Ssequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice�
[sequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2]
[sequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack�
]sequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2_
]sequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1�
]sequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2_
]sequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2�
Usequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1dsequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack:output:0fsequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0fsequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask2W
Usequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice_1�
Isequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/subSub\sequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice:output:0^sequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2K
Isequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/sub�
3sequential_20/bidirectional_20/forward_lstm_20/CastCastMsequential_20/bidirectional_20/forward_lstm_20/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������25
3sequential_20/bidirectional_20/forward_lstm_20/Cast�
4sequential_20/bidirectional_20/forward_lstm_20/ShapeShape[sequential_20/bidirectional_20/forward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:26
4sequential_20/bidirectional_20/forward_lstm_20/Shape�
Bsequential_20/bidirectional_20/forward_lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_20/bidirectional_20/forward_lstm_20/strided_slice/stack�
Dsequential_20/bidirectional_20/forward_lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_20/bidirectional_20/forward_lstm_20/strided_slice/stack_1�
Dsequential_20/bidirectional_20/forward_lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_20/bidirectional_20/forward_lstm_20/strided_slice/stack_2�
<sequential_20/bidirectional_20/forward_lstm_20/strided_sliceStridedSlice=sequential_20/bidirectional_20/forward_lstm_20/Shape:output:0Ksequential_20/bidirectional_20/forward_lstm_20/strided_slice/stack:output:0Msequential_20/bidirectional_20/forward_lstm_20/strided_slice/stack_1:output:0Msequential_20/bidirectional_20/forward_lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_20/bidirectional_20/forward_lstm_20/strided_slice�
:sequential_20/bidirectional_20/forward_lstm_20/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22<
:sequential_20/bidirectional_20/forward_lstm_20/zeros/mul/y�
8sequential_20/bidirectional_20/forward_lstm_20/zeros/mulMulEsequential_20/bidirectional_20/forward_lstm_20/strided_slice:output:0Csequential_20/bidirectional_20/forward_lstm_20/zeros/mul/y:output:0*
T0*
_output_shapes
: 2:
8sequential_20/bidirectional_20/forward_lstm_20/zeros/mul�
;sequential_20/bidirectional_20/forward_lstm_20/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2=
;sequential_20/bidirectional_20/forward_lstm_20/zeros/Less/y�
9sequential_20/bidirectional_20/forward_lstm_20/zeros/LessLess<sequential_20/bidirectional_20/forward_lstm_20/zeros/mul:z:0Dsequential_20/bidirectional_20/forward_lstm_20/zeros/Less/y:output:0*
T0*
_output_shapes
: 2;
9sequential_20/bidirectional_20/forward_lstm_20/zeros/Less�
=sequential_20/bidirectional_20/forward_lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22?
=sequential_20/bidirectional_20/forward_lstm_20/zeros/packed/1�
;sequential_20/bidirectional_20/forward_lstm_20/zeros/packedPackEsequential_20/bidirectional_20/forward_lstm_20/strided_slice:output:0Fsequential_20/bidirectional_20/forward_lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2=
;sequential_20/bidirectional_20/forward_lstm_20/zeros/packed�
:sequential_20/bidirectional_20/forward_lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2<
:sequential_20/bidirectional_20/forward_lstm_20/zeros/Const�
4sequential_20/bidirectional_20/forward_lstm_20/zerosFillDsequential_20/bidirectional_20/forward_lstm_20/zeros/packed:output:0Csequential_20/bidirectional_20/forward_lstm_20/zeros/Const:output:0*
T0*'
_output_shapes
:���������226
4sequential_20/bidirectional_20/forward_lstm_20/zeros�
<sequential_20/bidirectional_20/forward_lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22>
<sequential_20/bidirectional_20/forward_lstm_20/zeros_1/mul/y�
:sequential_20/bidirectional_20/forward_lstm_20/zeros_1/mulMulEsequential_20/bidirectional_20/forward_lstm_20/strided_slice:output:0Esequential_20/bidirectional_20/forward_lstm_20/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2<
:sequential_20/bidirectional_20/forward_lstm_20/zeros_1/mul�
=sequential_20/bidirectional_20/forward_lstm_20/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2?
=sequential_20/bidirectional_20/forward_lstm_20/zeros_1/Less/y�
;sequential_20/bidirectional_20/forward_lstm_20/zeros_1/LessLess>sequential_20/bidirectional_20/forward_lstm_20/zeros_1/mul:z:0Fsequential_20/bidirectional_20/forward_lstm_20/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2=
;sequential_20/bidirectional_20/forward_lstm_20/zeros_1/Less�
?sequential_20/bidirectional_20/forward_lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22A
?sequential_20/bidirectional_20/forward_lstm_20/zeros_1/packed/1�
=sequential_20/bidirectional_20/forward_lstm_20/zeros_1/packedPackEsequential_20/bidirectional_20/forward_lstm_20/strided_slice:output:0Hsequential_20/bidirectional_20/forward_lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2?
=sequential_20/bidirectional_20/forward_lstm_20/zeros_1/packed�
<sequential_20/bidirectional_20/forward_lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2>
<sequential_20/bidirectional_20/forward_lstm_20/zeros_1/Const�
6sequential_20/bidirectional_20/forward_lstm_20/zeros_1FillFsequential_20/bidirectional_20/forward_lstm_20/zeros_1/packed:output:0Esequential_20/bidirectional_20/forward_lstm_20/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������228
6sequential_20/bidirectional_20/forward_lstm_20/zeros_1�
=sequential_20/bidirectional_20/forward_lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2?
=sequential_20/bidirectional_20/forward_lstm_20/transpose/perm�
8sequential_20/bidirectional_20/forward_lstm_20/transpose	Transpose[sequential_20/bidirectional_20/forward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0Fsequential_20/bidirectional_20/forward_lstm_20/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2:
8sequential_20/bidirectional_20/forward_lstm_20/transpose�
6sequential_20/bidirectional_20/forward_lstm_20/Shape_1Shape<sequential_20/bidirectional_20/forward_lstm_20/transpose:y:0*
T0*
_output_shapes
:28
6sequential_20/bidirectional_20/forward_lstm_20/Shape_1�
Dsequential_20/bidirectional_20/forward_lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dsequential_20/bidirectional_20/forward_lstm_20/strided_slice_1/stack�
Fsequential_20/bidirectional_20/forward_lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_20/bidirectional_20/forward_lstm_20/strided_slice_1/stack_1�
Fsequential_20/bidirectional_20/forward_lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_20/bidirectional_20/forward_lstm_20/strided_slice_1/stack_2�
>sequential_20/bidirectional_20/forward_lstm_20/strided_slice_1StridedSlice?sequential_20/bidirectional_20/forward_lstm_20/Shape_1:output:0Msequential_20/bidirectional_20/forward_lstm_20/strided_slice_1/stack:output:0Osequential_20/bidirectional_20/forward_lstm_20/strided_slice_1/stack_1:output:0Osequential_20/bidirectional_20/forward_lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2@
>sequential_20/bidirectional_20/forward_lstm_20/strided_slice_1�
Jsequential_20/bidirectional_20/forward_lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2L
Jsequential_20/bidirectional_20/forward_lstm_20/TensorArrayV2/element_shape�
<sequential_20/bidirectional_20/forward_lstm_20/TensorArrayV2TensorListReserveSsequential_20/bidirectional_20/forward_lstm_20/TensorArrayV2/element_shape:output:0Gsequential_20/bidirectional_20/forward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_20/bidirectional_20/forward_lstm_20/TensorArrayV2�
dsequential_20/bidirectional_20/forward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2f
dsequential_20/bidirectional_20/forward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape�
Vsequential_20/bidirectional_20/forward_lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor<sequential_20/bidirectional_20/forward_lstm_20/transpose:y:0msequential_20/bidirectional_20/forward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02X
Vsequential_20/bidirectional_20/forward_lstm_20/TensorArrayUnstack/TensorListFromTensor�
Dsequential_20/bidirectional_20/forward_lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dsequential_20/bidirectional_20/forward_lstm_20/strided_slice_2/stack�
Fsequential_20/bidirectional_20/forward_lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_20/bidirectional_20/forward_lstm_20/strided_slice_2/stack_1�
Fsequential_20/bidirectional_20/forward_lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_20/bidirectional_20/forward_lstm_20/strided_slice_2/stack_2�
>sequential_20/bidirectional_20/forward_lstm_20/strided_slice_2StridedSlice<sequential_20/bidirectional_20/forward_lstm_20/transpose:y:0Msequential_20/bidirectional_20/forward_lstm_20/strided_slice_2/stack:output:0Osequential_20/bidirectional_20/forward_lstm_20/strided_slice_2/stack_1:output:0Osequential_20/bidirectional_20/forward_lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2@
>sequential_20/bidirectional_20/forward_lstm_20/strided_slice_2�
Qsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOpReadVariableOpZsequential_20_bidirectional_20_forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02S
Qsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp�
Bsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMulMatMulGsequential_20/bidirectional_20/forward_lstm_20/strided_slice_2:output:0Ysequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2D
Bsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMul�
Ssequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOp\sequential_20_bidirectional_20_forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02U
Ssequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp�
Dsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMul_1MatMul=sequential_20/bidirectional_20/forward_lstm_20/zeros:output:0[sequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2F
Dsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMul_1�
?sequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/addAddV2Lsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMul:product:0Nsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2A
?sequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/add�
Rsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOp[sequential_20_bidirectional_20_forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02T
Rsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp�
Csequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/BiasAddBiasAddCsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/add:z:0Zsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2E
Csequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/BiasAdd�
Ksequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2M
Ksequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/split/split_dim�
Asequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/splitSplitTsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/split/split_dim:output:0Lsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2C
Asequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/split�
Csequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/SigmoidSigmoidJsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22E
Csequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/Sigmoid�
Esequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/Sigmoid_1SigmoidJsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22G
Esequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/Sigmoid_1�
?sequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/mulMulIsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/Sigmoid_1:y:0?sequential_20/bidirectional_20/forward_lstm_20/zeros_1:output:0*
T0*'
_output_shapes
:���������22A
?sequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/mul�
@sequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/ReluReluJsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22B
@sequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/Relu�
Asequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/mul_1MulGsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/Sigmoid:y:0Nsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22C
Asequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/mul_1�
Asequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/add_1AddV2Csequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/mul:z:0Esequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22C
Asequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/add_1�
Esequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/Sigmoid_2SigmoidJsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22G
Esequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/Sigmoid_2�
Bsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/Relu_1ReluEsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22D
Bsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/Relu_1�
Asequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/mul_2MulIsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/Sigmoid_2:y:0Psequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22C
Asequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/mul_2�
Lsequential_20/bidirectional_20/forward_lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2N
Lsequential_20/bidirectional_20/forward_lstm_20/TensorArrayV2_1/element_shape�
>sequential_20/bidirectional_20/forward_lstm_20/TensorArrayV2_1TensorListReserveUsequential_20/bidirectional_20/forward_lstm_20/TensorArrayV2_1/element_shape:output:0Gsequential_20/bidirectional_20/forward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02@
>sequential_20/bidirectional_20/forward_lstm_20/TensorArrayV2_1�
3sequential_20/bidirectional_20/forward_lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 25
3sequential_20/bidirectional_20/forward_lstm_20/time�
9sequential_20/bidirectional_20/forward_lstm_20/zeros_like	ZerosLikeEsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/mul_2:z:0*
T0*'
_output_shapes
:���������22;
9sequential_20/bidirectional_20/forward_lstm_20/zeros_like�
Gsequential_20/bidirectional_20/forward_lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2I
Gsequential_20/bidirectional_20/forward_lstm_20/while/maximum_iterations�
Asequential_20/bidirectional_20/forward_lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_20/bidirectional_20/forward_lstm_20/while/loop_counter�
4sequential_20/bidirectional_20/forward_lstm_20/whileWhileJsequential_20/bidirectional_20/forward_lstm_20/while/loop_counter:output:0Psequential_20/bidirectional_20/forward_lstm_20/while/maximum_iterations:output:0<sequential_20/bidirectional_20/forward_lstm_20/time:output:0Gsequential_20/bidirectional_20/forward_lstm_20/TensorArrayV2_1:handle:0=sequential_20/bidirectional_20/forward_lstm_20/zeros_like:y:0=sequential_20/bidirectional_20/forward_lstm_20/zeros:output:0?sequential_20/bidirectional_20/forward_lstm_20/zeros_1:output:0Gsequential_20/bidirectional_20/forward_lstm_20/strided_slice_1:output:0fsequential_20/bidirectional_20/forward_lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:07sequential_20/bidirectional_20/forward_lstm_20/Cast:y:0Zsequential_20_bidirectional_20_forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource\sequential_20_bidirectional_20_forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource[sequential_20_bidirectional_20_forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource*
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
Asequential_20_bidirectional_20_forward_lstm_20_while_body_2411477*M
condERC
Asequential_20_bidirectional_20_forward_lstm_20_while_cond_2411476*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 26
4sequential_20/bidirectional_20/forward_lstm_20/while�
_sequential_20/bidirectional_20/forward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2a
_sequential_20/bidirectional_20/forward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape�
Qsequential_20/bidirectional_20/forward_lstm_20/TensorArrayV2Stack/TensorListStackTensorListStack=sequential_20/bidirectional_20/forward_lstm_20/while:output:3hsequential_20/bidirectional_20/forward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02S
Qsequential_20/bidirectional_20/forward_lstm_20/TensorArrayV2Stack/TensorListStack�
Dsequential_20/bidirectional_20/forward_lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2F
Dsequential_20/bidirectional_20/forward_lstm_20/strided_slice_3/stack�
Fsequential_20/bidirectional_20/forward_lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2H
Fsequential_20/bidirectional_20/forward_lstm_20/strided_slice_3/stack_1�
Fsequential_20/bidirectional_20/forward_lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_20/bidirectional_20/forward_lstm_20/strided_slice_3/stack_2�
>sequential_20/bidirectional_20/forward_lstm_20/strided_slice_3StridedSliceZsequential_20/bidirectional_20/forward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0Msequential_20/bidirectional_20/forward_lstm_20/strided_slice_3/stack:output:0Osequential_20/bidirectional_20/forward_lstm_20/strided_slice_3/stack_1:output:0Osequential_20/bidirectional_20/forward_lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2@
>sequential_20/bidirectional_20/forward_lstm_20/strided_slice_3�
?sequential_20/bidirectional_20/forward_lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2A
?sequential_20/bidirectional_20/forward_lstm_20/transpose_1/perm�
:sequential_20/bidirectional_20/forward_lstm_20/transpose_1	TransposeZsequential_20/bidirectional_20/forward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0Hsequential_20/bidirectional_20/forward_lstm_20/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22<
:sequential_20/bidirectional_20/forward_lstm_20/transpose_1�
6sequential_20/bidirectional_20/forward_lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    28
6sequential_20/bidirectional_20/forward_lstm_20/runtime�
Dsequential_20/bidirectional_20/backward_lstm_20/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2F
Dsequential_20/bidirectional_20/backward_lstm_20/RaggedToTensor/zeros�
Dsequential_20/bidirectional_20/backward_lstm_20/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2F
Dsequential_20/bidirectional_20/backward_lstm_20/RaggedToTensor/Const�
Ssequential_20/bidirectional_20/backward_lstm_20/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorMsequential_20/bidirectional_20/backward_lstm_20/RaggedToTensor/Const:output:0args_0Msequential_20/bidirectional_20/backward_lstm_20/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2U
Ssequential_20/bidirectional_20/backward_lstm_20/RaggedToTensor/RaggedTensorToTensor�
Zsequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2\
Zsequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack�
\sequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2^
\sequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1�
\sequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2^
\sequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2�
Tsequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1csequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack:output:0esequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1:output:0esequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask2V
Tsequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice�
\sequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2^
\sequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack�
^sequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2`
^sequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1�
^sequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2`
^sequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2�
Vsequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1esequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack:output:0gsequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0gsequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask2X
Vsequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice_1�
Jsequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/subSub]sequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice:output:0_sequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2L
Jsequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/sub�
4sequential_20/bidirectional_20/backward_lstm_20/CastCastNsequential_20/bidirectional_20/backward_lstm_20/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������26
4sequential_20/bidirectional_20/backward_lstm_20/Cast�
5sequential_20/bidirectional_20/backward_lstm_20/ShapeShape\sequential_20/bidirectional_20/backward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:27
5sequential_20/bidirectional_20/backward_lstm_20/Shape�
Csequential_20/bidirectional_20/backward_lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Csequential_20/bidirectional_20/backward_lstm_20/strided_slice/stack�
Esequential_20/bidirectional_20/backward_lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_20/bidirectional_20/backward_lstm_20/strided_slice/stack_1�
Esequential_20/bidirectional_20/backward_lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_20/bidirectional_20/backward_lstm_20/strided_slice/stack_2�
=sequential_20/bidirectional_20/backward_lstm_20/strided_sliceStridedSlice>sequential_20/bidirectional_20/backward_lstm_20/Shape:output:0Lsequential_20/bidirectional_20/backward_lstm_20/strided_slice/stack:output:0Nsequential_20/bidirectional_20/backward_lstm_20/strided_slice/stack_1:output:0Nsequential_20/bidirectional_20/backward_lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=sequential_20/bidirectional_20/backward_lstm_20/strided_slice�
;sequential_20/bidirectional_20/backward_lstm_20/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22=
;sequential_20/bidirectional_20/backward_lstm_20/zeros/mul/y�
9sequential_20/bidirectional_20/backward_lstm_20/zeros/mulMulFsequential_20/bidirectional_20/backward_lstm_20/strided_slice:output:0Dsequential_20/bidirectional_20/backward_lstm_20/zeros/mul/y:output:0*
T0*
_output_shapes
: 2;
9sequential_20/bidirectional_20/backward_lstm_20/zeros/mul�
<sequential_20/bidirectional_20/backward_lstm_20/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2>
<sequential_20/bidirectional_20/backward_lstm_20/zeros/Less/y�
:sequential_20/bidirectional_20/backward_lstm_20/zeros/LessLess=sequential_20/bidirectional_20/backward_lstm_20/zeros/mul:z:0Esequential_20/bidirectional_20/backward_lstm_20/zeros/Less/y:output:0*
T0*
_output_shapes
: 2<
:sequential_20/bidirectional_20/backward_lstm_20/zeros/Less�
>sequential_20/bidirectional_20/backward_lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22@
>sequential_20/bidirectional_20/backward_lstm_20/zeros/packed/1�
<sequential_20/bidirectional_20/backward_lstm_20/zeros/packedPackFsequential_20/bidirectional_20/backward_lstm_20/strided_slice:output:0Gsequential_20/bidirectional_20/backward_lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2>
<sequential_20/bidirectional_20/backward_lstm_20/zeros/packed�
;sequential_20/bidirectional_20/backward_lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2=
;sequential_20/bidirectional_20/backward_lstm_20/zeros/Const�
5sequential_20/bidirectional_20/backward_lstm_20/zerosFillEsequential_20/bidirectional_20/backward_lstm_20/zeros/packed:output:0Dsequential_20/bidirectional_20/backward_lstm_20/zeros/Const:output:0*
T0*'
_output_shapes
:���������227
5sequential_20/bidirectional_20/backward_lstm_20/zeros�
=sequential_20/bidirectional_20/backward_lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22?
=sequential_20/bidirectional_20/backward_lstm_20/zeros_1/mul/y�
;sequential_20/bidirectional_20/backward_lstm_20/zeros_1/mulMulFsequential_20/bidirectional_20/backward_lstm_20/strided_slice:output:0Fsequential_20/bidirectional_20/backward_lstm_20/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2=
;sequential_20/bidirectional_20/backward_lstm_20/zeros_1/mul�
>sequential_20/bidirectional_20/backward_lstm_20/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2@
>sequential_20/bidirectional_20/backward_lstm_20/zeros_1/Less/y�
<sequential_20/bidirectional_20/backward_lstm_20/zeros_1/LessLess?sequential_20/bidirectional_20/backward_lstm_20/zeros_1/mul:z:0Gsequential_20/bidirectional_20/backward_lstm_20/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2>
<sequential_20/bidirectional_20/backward_lstm_20/zeros_1/Less�
@sequential_20/bidirectional_20/backward_lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22B
@sequential_20/bidirectional_20/backward_lstm_20/zeros_1/packed/1�
>sequential_20/bidirectional_20/backward_lstm_20/zeros_1/packedPackFsequential_20/bidirectional_20/backward_lstm_20/strided_slice:output:0Isequential_20/bidirectional_20/backward_lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2@
>sequential_20/bidirectional_20/backward_lstm_20/zeros_1/packed�
=sequential_20/bidirectional_20/backward_lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2?
=sequential_20/bidirectional_20/backward_lstm_20/zeros_1/Const�
7sequential_20/bidirectional_20/backward_lstm_20/zeros_1FillGsequential_20/bidirectional_20/backward_lstm_20/zeros_1/packed:output:0Fsequential_20/bidirectional_20/backward_lstm_20/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������229
7sequential_20/bidirectional_20/backward_lstm_20/zeros_1�
>sequential_20/bidirectional_20/backward_lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2@
>sequential_20/bidirectional_20/backward_lstm_20/transpose/perm�
9sequential_20/bidirectional_20/backward_lstm_20/transpose	Transpose\sequential_20/bidirectional_20/backward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0Gsequential_20/bidirectional_20/backward_lstm_20/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2;
9sequential_20/bidirectional_20/backward_lstm_20/transpose�
7sequential_20/bidirectional_20/backward_lstm_20/Shape_1Shape=sequential_20/bidirectional_20/backward_lstm_20/transpose:y:0*
T0*
_output_shapes
:29
7sequential_20/bidirectional_20/backward_lstm_20/Shape_1�
Esequential_20/bidirectional_20/backward_lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_20/bidirectional_20/backward_lstm_20/strided_slice_1/stack�
Gsequential_20/bidirectional_20/backward_lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_20/bidirectional_20/backward_lstm_20/strided_slice_1/stack_1�
Gsequential_20/bidirectional_20/backward_lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_20/bidirectional_20/backward_lstm_20/strided_slice_1/stack_2�
?sequential_20/bidirectional_20/backward_lstm_20/strided_slice_1StridedSlice@sequential_20/bidirectional_20/backward_lstm_20/Shape_1:output:0Nsequential_20/bidirectional_20/backward_lstm_20/strided_slice_1/stack:output:0Psequential_20/bidirectional_20/backward_lstm_20/strided_slice_1/stack_1:output:0Psequential_20/bidirectional_20/backward_lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?sequential_20/bidirectional_20/backward_lstm_20/strided_slice_1�
Ksequential_20/bidirectional_20/backward_lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2M
Ksequential_20/bidirectional_20/backward_lstm_20/TensorArrayV2/element_shape�
=sequential_20/bidirectional_20/backward_lstm_20/TensorArrayV2TensorListReserveTsequential_20/bidirectional_20/backward_lstm_20/TensorArrayV2/element_shape:output:0Hsequential_20/bidirectional_20/backward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_20/bidirectional_20/backward_lstm_20/TensorArrayV2�
>sequential_20/bidirectional_20/backward_lstm_20/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_20/bidirectional_20/backward_lstm_20/ReverseV2/axis�
9sequential_20/bidirectional_20/backward_lstm_20/ReverseV2	ReverseV2=sequential_20/bidirectional_20/backward_lstm_20/transpose:y:0Gsequential_20/bidirectional_20/backward_lstm_20/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������2;
9sequential_20/bidirectional_20/backward_lstm_20/ReverseV2�
esequential_20/bidirectional_20/backward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2g
esequential_20/bidirectional_20/backward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape�
Wsequential_20/bidirectional_20/backward_lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorBsequential_20/bidirectional_20/backward_lstm_20/ReverseV2:output:0nsequential_20/bidirectional_20/backward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02Y
Wsequential_20/bidirectional_20/backward_lstm_20/TensorArrayUnstack/TensorListFromTensor�
Esequential_20/bidirectional_20/backward_lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_20/bidirectional_20/backward_lstm_20/strided_slice_2/stack�
Gsequential_20/bidirectional_20/backward_lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_20/bidirectional_20/backward_lstm_20/strided_slice_2/stack_1�
Gsequential_20/bidirectional_20/backward_lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_20/bidirectional_20/backward_lstm_20/strided_slice_2/stack_2�
?sequential_20/bidirectional_20/backward_lstm_20/strided_slice_2StridedSlice=sequential_20/bidirectional_20/backward_lstm_20/transpose:y:0Nsequential_20/bidirectional_20/backward_lstm_20/strided_slice_2/stack:output:0Psequential_20/bidirectional_20/backward_lstm_20/strided_slice_2/stack_1:output:0Psequential_20/bidirectional_20/backward_lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2A
?sequential_20/bidirectional_20/backward_lstm_20/strided_slice_2�
Rsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOpReadVariableOp[sequential_20_bidirectional_20_backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02T
Rsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp�
Csequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMulMatMulHsequential_20/bidirectional_20/backward_lstm_20/strided_slice_2:output:0Zsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2E
Csequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMul�
Tsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOp]sequential_20_bidirectional_20_backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02V
Tsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp�
Esequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMul_1MatMul>sequential_20/bidirectional_20/backward_lstm_20/zeros:output:0\sequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2G
Esequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMul_1�
@sequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/addAddV2Msequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMul:product:0Osequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2B
@sequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/add�
Ssequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOp\sequential_20_bidirectional_20_backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02U
Ssequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp�
Dsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/BiasAddBiasAddDsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/add:z:0[sequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2F
Dsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/BiasAdd�
Lsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2N
Lsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/split/split_dim�
Bsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/splitSplitUsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/split/split_dim:output:0Msequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2D
Bsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/split�
Dsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/SigmoidSigmoidKsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22F
Dsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/Sigmoid�
Fsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/Sigmoid_1SigmoidKsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22H
Fsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/Sigmoid_1�
@sequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/mulMulJsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/Sigmoid_1:y:0@sequential_20/bidirectional_20/backward_lstm_20/zeros_1:output:0*
T0*'
_output_shapes
:���������22B
@sequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/mul�
Asequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/ReluReluKsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22C
Asequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/Relu�
Bsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/mul_1MulHsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/Sigmoid:y:0Osequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22D
Bsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/mul_1�
Bsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/add_1AddV2Dsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/mul:z:0Fsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22D
Bsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/add_1�
Fsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/Sigmoid_2SigmoidKsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22H
Fsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/Sigmoid_2�
Csequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/Relu_1ReluFsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22E
Csequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/Relu_1�
Bsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/mul_2MulJsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/Sigmoid_2:y:0Qsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22D
Bsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/mul_2�
Msequential_20/bidirectional_20/backward_lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2O
Msequential_20/bidirectional_20/backward_lstm_20/TensorArrayV2_1/element_shape�
?sequential_20/bidirectional_20/backward_lstm_20/TensorArrayV2_1TensorListReserveVsequential_20/bidirectional_20/backward_lstm_20/TensorArrayV2_1/element_shape:output:0Hsequential_20/bidirectional_20/backward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?sequential_20/bidirectional_20/backward_lstm_20/TensorArrayV2_1�
4sequential_20/bidirectional_20/backward_lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 26
4sequential_20/bidirectional_20/backward_lstm_20/time�
Esequential_20/bidirectional_20/backward_lstm_20/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2G
Esequential_20/bidirectional_20/backward_lstm_20/Max/reduction_indices�
3sequential_20/bidirectional_20/backward_lstm_20/MaxMax8sequential_20/bidirectional_20/backward_lstm_20/Cast:y:0Nsequential_20/bidirectional_20/backward_lstm_20/Max/reduction_indices:output:0*
T0*
_output_shapes
: 25
3sequential_20/bidirectional_20/backward_lstm_20/Max�
5sequential_20/bidirectional_20/backward_lstm_20/sub/yConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_20/bidirectional_20/backward_lstm_20/sub/y�
3sequential_20/bidirectional_20/backward_lstm_20/subSub<sequential_20/bidirectional_20/backward_lstm_20/Max:output:0>sequential_20/bidirectional_20/backward_lstm_20/sub/y:output:0*
T0*
_output_shapes
: 25
3sequential_20/bidirectional_20/backward_lstm_20/sub�
5sequential_20/bidirectional_20/backward_lstm_20/Sub_1Sub7sequential_20/bidirectional_20/backward_lstm_20/sub:z:08sequential_20/bidirectional_20/backward_lstm_20/Cast:y:0*
T0*#
_output_shapes
:���������27
5sequential_20/bidirectional_20/backward_lstm_20/Sub_1�
:sequential_20/bidirectional_20/backward_lstm_20/zeros_like	ZerosLikeFsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/mul_2:z:0*
T0*'
_output_shapes
:���������22<
:sequential_20/bidirectional_20/backward_lstm_20/zeros_like�
Hsequential_20/bidirectional_20/backward_lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2J
Hsequential_20/bidirectional_20/backward_lstm_20/while/maximum_iterations�
Bsequential_20/bidirectional_20/backward_lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bsequential_20/bidirectional_20/backward_lstm_20/while/loop_counter�
5sequential_20/bidirectional_20/backward_lstm_20/whileWhileKsequential_20/bidirectional_20/backward_lstm_20/while/loop_counter:output:0Qsequential_20/bidirectional_20/backward_lstm_20/while/maximum_iterations:output:0=sequential_20/bidirectional_20/backward_lstm_20/time:output:0Hsequential_20/bidirectional_20/backward_lstm_20/TensorArrayV2_1:handle:0>sequential_20/bidirectional_20/backward_lstm_20/zeros_like:y:0>sequential_20/bidirectional_20/backward_lstm_20/zeros:output:0@sequential_20/bidirectional_20/backward_lstm_20/zeros_1:output:0Hsequential_20/bidirectional_20/backward_lstm_20/strided_slice_1:output:0gsequential_20/bidirectional_20/backward_lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:09sequential_20/bidirectional_20/backward_lstm_20/Sub_1:z:0[sequential_20_bidirectional_20_backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource]sequential_20_bidirectional_20_backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource\sequential_20_bidirectional_20_backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource*
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
Bsequential_20_bidirectional_20_backward_lstm_20_while_body_2411656*N
condFRD
Bsequential_20_bidirectional_20_backward_lstm_20_while_cond_2411655*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 27
5sequential_20/bidirectional_20/backward_lstm_20/while�
`sequential_20/bidirectional_20/backward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2b
`sequential_20/bidirectional_20/backward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape�
Rsequential_20/bidirectional_20/backward_lstm_20/TensorArrayV2Stack/TensorListStackTensorListStack>sequential_20/bidirectional_20/backward_lstm_20/while:output:3isequential_20/bidirectional_20/backward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02T
Rsequential_20/bidirectional_20/backward_lstm_20/TensorArrayV2Stack/TensorListStack�
Esequential_20/bidirectional_20/backward_lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2G
Esequential_20/bidirectional_20/backward_lstm_20/strided_slice_3/stack�
Gsequential_20/bidirectional_20/backward_lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2I
Gsequential_20/bidirectional_20/backward_lstm_20/strided_slice_3/stack_1�
Gsequential_20/bidirectional_20/backward_lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_20/bidirectional_20/backward_lstm_20/strided_slice_3/stack_2�
?sequential_20/bidirectional_20/backward_lstm_20/strided_slice_3StridedSlice[sequential_20/bidirectional_20/backward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0Nsequential_20/bidirectional_20/backward_lstm_20/strided_slice_3/stack:output:0Psequential_20/bidirectional_20/backward_lstm_20/strided_slice_3/stack_1:output:0Psequential_20/bidirectional_20/backward_lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2A
?sequential_20/bidirectional_20/backward_lstm_20/strided_slice_3�
@sequential_20/bidirectional_20/backward_lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2B
@sequential_20/bidirectional_20/backward_lstm_20/transpose_1/perm�
;sequential_20/bidirectional_20/backward_lstm_20/transpose_1	Transpose[sequential_20/bidirectional_20/backward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0Isequential_20/bidirectional_20/backward_lstm_20/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22=
;sequential_20/bidirectional_20/backward_lstm_20/transpose_1�
7sequential_20/bidirectional_20/backward_lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    29
7sequential_20/bidirectional_20/backward_lstm_20/runtime�
*sequential_20/bidirectional_20/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2,
*sequential_20/bidirectional_20/concat/axis�
%sequential_20/bidirectional_20/concatConcatV2Gsequential_20/bidirectional_20/forward_lstm_20/strided_slice_3:output:0Hsequential_20/bidirectional_20/backward_lstm_20/strided_slice_3:output:03sequential_20/bidirectional_20/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������d2'
%sequential_20/bidirectional_20/concat�
,sequential_20/dense_20/MatMul/ReadVariableOpReadVariableOp5sequential_20_dense_20_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,sequential_20/dense_20/MatMul/ReadVariableOp�
sequential_20/dense_20/MatMulMatMul.sequential_20/bidirectional_20/concat:output:04sequential_20/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_20/dense_20/MatMul�
-sequential_20/dense_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_20_dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_20/dense_20/BiasAdd/ReadVariableOp�
sequential_20/dense_20/BiasAddBiasAdd'sequential_20/dense_20/MatMul:product:05sequential_20/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
sequential_20/dense_20/BiasAdd�
sequential_20/dense_20/SigmoidSigmoid'sequential_20/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������2 
sequential_20/dense_20/Sigmoid}
IdentityIdentity"sequential_20/dense_20/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOpT^sequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOpS^sequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOpU^sequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp6^sequential_20/bidirectional_20/backward_lstm_20/whileS^sequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOpR^sequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOpT^sequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp5^sequential_20/bidirectional_20/forward_lstm_20/while.^sequential_20/dense_20/BiasAdd/ReadVariableOp-^sequential_20/dense_20/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:���������:���������: : : : : : : : 2�
Ssequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOpSsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp2�
Rsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOpRsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp2�
Tsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOpTsequential_20/bidirectional_20/backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp2n
5sequential_20/bidirectional_20/backward_lstm_20/while5sequential_20/bidirectional_20/backward_lstm_20/while2�
Rsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOpRsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp2�
Qsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOpQsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp2�
Ssequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOpSsequential_20/bidirectional_20/forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp2l
4sequential_20/bidirectional_20/forward_lstm_20/while4sequential_20/bidirectional_20/forward_lstm_20/while2^
-sequential_20/dense_20/BiasAdd/ReadVariableOp-sequential_20/dense_20/BiasAdd/ReadVariableOp2\
,sequential_20/dense_20/MatMul/ReadVariableOp,sequential_20/dense_20/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameargs_0:KG
#
_output_shapes
:���������
 
_user_specified_nameargs_0
�
�
#backward_lstm_20_while_cond_2414538>
:backward_lstm_20_while_backward_lstm_20_while_loop_counterD
@backward_lstm_20_while_backward_lstm_20_while_maximum_iterations&
"backward_lstm_20_while_placeholder(
$backward_lstm_20_while_placeholder_1(
$backward_lstm_20_while_placeholder_2(
$backward_lstm_20_while_placeholder_3(
$backward_lstm_20_while_placeholder_4@
<backward_lstm_20_while_less_backward_lstm_20_strided_slice_1W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2414538___redundant_placeholder0W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2414538___redundant_placeholder1W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2414538___redundant_placeholder2W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2414538___redundant_placeholder3W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2414538___redundant_placeholder4#
backward_lstm_20_while_identity
�
backward_lstm_20/while/LessLess"backward_lstm_20_while_placeholder<backward_lstm_20_while_less_backward_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_20/while/Less�
backward_lstm_20/while/IdentityIdentitybackward_lstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_20/while/Identity"K
backward_lstm_20_while_identity(backward_lstm_20/while/Identity:output:0*(
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
�V
�
"forward_lstm_20_while_body_2415255<
8forward_lstm_20_while_forward_lstm_20_while_loop_counterB
>forward_lstm_20_while_forward_lstm_20_while_maximum_iterations%
!forward_lstm_20_while_placeholder'
#forward_lstm_20_while_placeholder_1'
#forward_lstm_20_while_placeholder_2'
#forward_lstm_20_while_placeholder_3;
7forward_lstm_20_while_forward_lstm_20_strided_slice_1_0w
sforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0V
Cforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0:	�X
Eforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0:	2�S
Dforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0:	�"
forward_lstm_20_while_identity$
 forward_lstm_20_while_identity_1$
 forward_lstm_20_while_identity_2$
 forward_lstm_20_while_identity_3$
 forward_lstm_20_while_identity_4$
 forward_lstm_20_while_identity_59
5forward_lstm_20_while_forward_lstm_20_strided_slice_1u
qforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensorT
Aforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource:	�V
Cforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource:	2�Q
Bforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource:	���9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp�8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp�:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp�
Gforward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2I
Gforward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape�
9forward_lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_20_while_placeholderPforward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02;
9forward_lstm_20/while/TensorArrayV2Read/TensorListGetItem�
8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOpReadVariableOpCforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02:
8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp�
)forward_lstm_20/while/lstm_cell_61/MatMulMatMul@forward_lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:0@forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)forward_lstm_20/while/lstm_cell_61/MatMul�
:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOpEforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02<
:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp�
+forward_lstm_20/while/lstm_cell_61/MatMul_1MatMul#forward_lstm_20_while_placeholder_2Bforward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+forward_lstm_20/while/lstm_cell_61/MatMul_1�
&forward_lstm_20/while/lstm_cell_61/addAddV23forward_lstm_20/while/lstm_cell_61/MatMul:product:05forward_lstm_20/while/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&forward_lstm_20/while/lstm_cell_61/add�
9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOpDforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02;
9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp�
*forward_lstm_20/while/lstm_cell_61/BiasAddBiasAdd*forward_lstm_20/while/lstm_cell_61/add:z:0Aforward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*forward_lstm_20/while/lstm_cell_61/BiasAdd�
2forward_lstm_20/while/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2forward_lstm_20/while/lstm_cell_61/split/split_dim�
(forward_lstm_20/while/lstm_cell_61/splitSplit;forward_lstm_20/while/lstm_cell_61/split/split_dim:output:03forward_lstm_20/while/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2*
(forward_lstm_20/while/lstm_cell_61/split�
*forward_lstm_20/while/lstm_cell_61/SigmoidSigmoid1forward_lstm_20/while/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22,
*forward_lstm_20/while/lstm_cell_61/Sigmoid�
,forward_lstm_20/while/lstm_cell_61/Sigmoid_1Sigmoid1forward_lstm_20/while/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22.
,forward_lstm_20/while/lstm_cell_61/Sigmoid_1�
&forward_lstm_20/while/lstm_cell_61/mulMul0forward_lstm_20/while/lstm_cell_61/Sigmoid_1:y:0#forward_lstm_20_while_placeholder_3*
T0*'
_output_shapes
:���������22(
&forward_lstm_20/while/lstm_cell_61/mul�
'forward_lstm_20/while/lstm_cell_61/ReluRelu1forward_lstm_20/while/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22)
'forward_lstm_20/while/lstm_cell_61/Relu�
(forward_lstm_20/while/lstm_cell_61/mul_1Mul.forward_lstm_20/while/lstm_cell_61/Sigmoid:y:05forward_lstm_20/while/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_20/while/lstm_cell_61/mul_1�
(forward_lstm_20/while/lstm_cell_61/add_1AddV2*forward_lstm_20/while/lstm_cell_61/mul:z:0,forward_lstm_20/while/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_20/while/lstm_cell_61/add_1�
,forward_lstm_20/while/lstm_cell_61/Sigmoid_2Sigmoid1forward_lstm_20/while/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22.
,forward_lstm_20/while/lstm_cell_61/Sigmoid_2�
)forward_lstm_20/while/lstm_cell_61/Relu_1Relu,forward_lstm_20/while/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22+
)forward_lstm_20/while/lstm_cell_61/Relu_1�
(forward_lstm_20/while/lstm_cell_61/mul_2Mul0forward_lstm_20/while/lstm_cell_61/Sigmoid_2:y:07forward_lstm_20/while/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_20/while/lstm_cell_61/mul_2�
:forward_lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_20_while_placeholder_1!forward_lstm_20_while_placeholder,forward_lstm_20/while/lstm_cell_61/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_20/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_20/while/add/y�
forward_lstm_20/while/addAddV2!forward_lstm_20_while_placeholder$forward_lstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/while/add�
forward_lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_20/while/add_1/y�
forward_lstm_20/while/add_1AddV28forward_lstm_20_while_forward_lstm_20_while_loop_counter&forward_lstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/while/add_1�
forward_lstm_20/while/IdentityIdentityforward_lstm_20/while/add_1:z:0^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_20/while/Identity�
 forward_lstm_20/while/Identity_1Identity>forward_lstm_20_while_forward_lstm_20_while_maximum_iterations^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_20/while/Identity_1�
 forward_lstm_20/while/Identity_2Identityforward_lstm_20/while/add:z:0^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_20/while/Identity_2�
 forward_lstm_20/while/Identity_3IdentityJforward_lstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_20/while/Identity_3�
 forward_lstm_20/while/Identity_4Identity,forward_lstm_20/while/lstm_cell_61/mul_2:z:0^forward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/while/Identity_4�
 forward_lstm_20/while/Identity_5Identity,forward_lstm_20/while/lstm_cell_61/add_1:z:0^forward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/while/Identity_5�
forward_lstm_20/while/NoOpNoOp:^forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp9^forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp;^forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_20/while/NoOp"p
5forward_lstm_20_while_forward_lstm_20_strided_slice_17forward_lstm_20_while_forward_lstm_20_strided_slice_1_0"I
forward_lstm_20_while_identity'forward_lstm_20/while/Identity:output:0"M
 forward_lstm_20_while_identity_1)forward_lstm_20/while/Identity_1:output:0"M
 forward_lstm_20_while_identity_2)forward_lstm_20/while/Identity_2:output:0"M
 forward_lstm_20_while_identity_3)forward_lstm_20/while/Identity_3:output:0"M
 forward_lstm_20_while_identity_4)forward_lstm_20/while/Identity_4:output:0"M
 forward_lstm_20_while_identity_5)forward_lstm_20/while/Identity_5:output:0"�
Bforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resourceDforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0"�
Cforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resourceEforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0"�
Aforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resourceCforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0"�
qforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensorsforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2v
9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp2t
8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp2x
:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp: 
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
2__inference_backward_lstm_20_layer_call_fn_2416885
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
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_24125502
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
�
�
#backward_lstm_20_while_cond_2416108>
:backward_lstm_20_while_backward_lstm_20_while_loop_counterD
@backward_lstm_20_while_backward_lstm_20_while_maximum_iterations&
"backward_lstm_20_while_placeholder(
$backward_lstm_20_while_placeholder_1(
$backward_lstm_20_while_placeholder_2(
$backward_lstm_20_while_placeholder_3(
$backward_lstm_20_while_placeholder_4@
<backward_lstm_20_while_less_backward_lstm_20_strided_slice_1W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2416108___redundant_placeholder0W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2416108___redundant_placeholder1W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2416108___redundant_placeholder2W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2416108___redundant_placeholder3W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2416108___redundant_placeholder4#
backward_lstm_20_while_identity
�
backward_lstm_20/while/LessLess"backward_lstm_20_while_placeholder<backward_lstm_20_while_less_backward_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_20/while/Less�
backward_lstm_20/while/IdentityIdentitybackward_lstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_20/while/Identity"K
backward_lstm_20_while_identity(backward_lstm_20/while/Identity:output:0*(
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
while_body_2417446
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_62_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_62_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_62_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_62_matmul_readvariableop_resource:	�F
3while_lstm_cell_62_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_62_biasadd_readvariableop_resource:	���)while/lstm_cell_62/BiasAdd/ReadVariableOp�(while/lstm_cell_62/MatMul/ReadVariableOp�*while/lstm_cell_62/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_62/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_62_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_62/MatMul/ReadVariableOp�
while/lstm_cell_62/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/MatMul�
*while/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_62_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_62/MatMul_1/ReadVariableOp�
while/lstm_cell_62/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/MatMul_1�
while/lstm_cell_62/addAddV2#while/lstm_cell_62/MatMul:product:0%while/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/add�
)while/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_62_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_62/BiasAdd/ReadVariableOp�
while/lstm_cell_62/BiasAddBiasAddwhile/lstm_cell_62/add:z:01while/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/BiasAdd�
"while/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_62/split/split_dim�
while/lstm_cell_62/splitSplit+while/lstm_cell_62/split/split_dim:output:0#while/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_62/split�
while/lstm_cell_62/SigmoidSigmoid!while/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Sigmoid�
while/lstm_cell_62/Sigmoid_1Sigmoid!while/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Sigmoid_1�
while/lstm_cell_62/mulMul while/lstm_cell_62/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/mul�
while/lstm_cell_62/ReluRelu!while/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Relu�
while/lstm_cell_62/mul_1Mulwhile/lstm_cell_62/Sigmoid:y:0%while/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/mul_1�
while/lstm_cell_62/add_1AddV2while/lstm_cell_62/mul:z:0while/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/add_1�
while/lstm_cell_62/Sigmoid_2Sigmoid!while/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Sigmoid_2�
while/lstm_cell_62/Relu_1Reluwhile/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Relu_1�
while/lstm_cell_62/mul_2Mul while/lstm_cell_62/Sigmoid_2:y:0'while/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_62/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_62/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_62/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_62/BiasAdd/ReadVariableOp)^while/lstm_cell_62/MatMul/ReadVariableOp+^while/lstm_cell_62/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_62_biasadd_readvariableop_resource4while_lstm_cell_62_biasadd_readvariableop_resource_0"l
3while_lstm_cell_62_matmul_1_readvariableop_resource5while_lstm_cell_62_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_62_matmul_readvariableop_resource3while_lstm_cell_62_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_62/BiasAdd/ReadVariableOp)while/lstm_cell_62/BiasAdd/ReadVariableOp2T
(while/lstm_cell_62/MatMul/ReadVariableOp(while/lstm_cell_62/MatMul/ReadVariableOp2X
*while/lstm_cell_62/MatMul_1/ReadVariableOp*while/lstm_cell_62/MatMul_1/ReadVariableOp: 
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
�H
�
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_2412550

inputs'
lstm_cell_62_2412468:	�'
lstm_cell_62_2412470:	2�#
lstm_cell_62_2412472:	�
identity��$lstm_cell_62/StatefulPartitionedCall�whileD
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
$lstm_cell_62/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_62_2412468lstm_cell_62_2412470lstm_cell_62_2412472*
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
I__inference_lstm_cell_62_layer_call_and_return_conditional_losses_24124672&
$lstm_cell_62/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_62_2412468lstm_cell_62_2412470lstm_cell_62_2412472*
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
while_body_2412481*
condR
while_cond_2412480*K
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
NoOpNoOp%^lstm_cell_62/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_62/StatefulPartitionedCall$lstm_cell_62/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
while_cond_2417445
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2417445___redundant_placeholder05
1while_while_cond_2417445___redundant_placeholder15
1while_while_cond_2417445___redundant_placeholder25
1while_while_cond_2417445___redundant_placeholder3
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
�\
�
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_2416572
inputs_0>
+lstm_cell_61_matmul_readvariableop_resource:	�@
-lstm_cell_61_matmul_1_readvariableop_resource:	2�;
,lstm_cell_61_biasadd_readvariableop_resource:	�
identity��#lstm_cell_61/BiasAdd/ReadVariableOp�"lstm_cell_61/MatMul/ReadVariableOp�$lstm_cell_61/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_61/MatMul/ReadVariableOpReadVariableOp+lstm_cell_61_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_61/MatMul/ReadVariableOp�
lstm_cell_61/MatMulMatMulstrided_slice_2:output:0*lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/MatMul�
$lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_61_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_61/MatMul_1/ReadVariableOp�
lstm_cell_61/MatMul_1MatMulzeros:output:0,lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/MatMul_1�
lstm_cell_61/addAddV2lstm_cell_61/MatMul:product:0lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/add�
#lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_61_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_61/BiasAdd/ReadVariableOp�
lstm_cell_61/BiasAddBiasAddlstm_cell_61/add:z:0+lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/BiasAdd~
lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_61/split/split_dim�
lstm_cell_61/splitSplit%lstm_cell_61/split/split_dim:output:0lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_61/split�
lstm_cell_61/SigmoidSigmoidlstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/Sigmoid�
lstm_cell_61/Sigmoid_1Sigmoidlstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_61/Sigmoid_1�
lstm_cell_61/mulMullstm_cell_61/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/mul}
lstm_cell_61/ReluRelulstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_61/Relu�
lstm_cell_61/mul_1Mullstm_cell_61/Sigmoid:y:0lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/mul_1�
lstm_cell_61/add_1AddV2lstm_cell_61/mul:z:0lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/add_1�
lstm_cell_61/Sigmoid_2Sigmoidlstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_61/Sigmoid_2|
lstm_cell_61/Relu_1Relulstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/Relu_1�
lstm_cell_61/mul_2Mullstm_cell_61/Sigmoid_2:y:0!lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_61_matmul_readvariableop_resource-lstm_cell_61_matmul_1_readvariableop_resource,lstm_cell_61_biasadd_readvariableop_resource*
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
while_body_2416488*
condR
while_cond_2416487*K
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
NoOpNoOp$^lstm_cell_61/BiasAdd/ReadVariableOp#^lstm_cell_61/MatMul/ReadVariableOp%^lstm_cell_61/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_61/BiasAdd/ReadVariableOp#lstm_cell_61/BiasAdd/ReadVariableOp2H
"lstm_cell_61/MatMul/ReadVariableOp"lstm_cell_61/MatMul/ReadVariableOp2L
$lstm_cell_61/MatMul_1/ReadVariableOp$lstm_cell_61/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
Asequential_20_bidirectional_20_forward_lstm_20_while_cond_2411476z
vsequential_20_bidirectional_20_forward_lstm_20_while_sequential_20_bidirectional_20_forward_lstm_20_while_loop_counter�
|sequential_20_bidirectional_20_forward_lstm_20_while_sequential_20_bidirectional_20_forward_lstm_20_while_maximum_iterationsD
@sequential_20_bidirectional_20_forward_lstm_20_while_placeholderF
Bsequential_20_bidirectional_20_forward_lstm_20_while_placeholder_1F
Bsequential_20_bidirectional_20_forward_lstm_20_while_placeholder_2F
Bsequential_20_bidirectional_20_forward_lstm_20_while_placeholder_3F
Bsequential_20_bidirectional_20_forward_lstm_20_while_placeholder_4|
xsequential_20_bidirectional_20_forward_lstm_20_while_less_sequential_20_bidirectional_20_forward_lstm_20_strided_slice_1�
�sequential_20_bidirectional_20_forward_lstm_20_while_sequential_20_bidirectional_20_forward_lstm_20_while_cond_2411476___redundant_placeholder0�
�sequential_20_bidirectional_20_forward_lstm_20_while_sequential_20_bidirectional_20_forward_lstm_20_while_cond_2411476___redundant_placeholder1�
�sequential_20_bidirectional_20_forward_lstm_20_while_sequential_20_bidirectional_20_forward_lstm_20_while_cond_2411476___redundant_placeholder2�
�sequential_20_bidirectional_20_forward_lstm_20_while_sequential_20_bidirectional_20_forward_lstm_20_while_cond_2411476___redundant_placeholder3�
�sequential_20_bidirectional_20_forward_lstm_20_while_sequential_20_bidirectional_20_forward_lstm_20_while_cond_2411476___redundant_placeholder4A
=sequential_20_bidirectional_20_forward_lstm_20_while_identity
�
9sequential_20/bidirectional_20/forward_lstm_20/while/LessLess@sequential_20_bidirectional_20_forward_lstm_20_while_placeholderxsequential_20_bidirectional_20_forward_lstm_20_while_less_sequential_20_bidirectional_20_forward_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2;
9sequential_20/bidirectional_20/forward_lstm_20/while/Less�
=sequential_20/bidirectional_20/forward_lstm_20/while/IdentityIdentity=sequential_20/bidirectional_20/forward_lstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2?
=sequential_20/bidirectional_20/forward_lstm_20/while/Identity"�
=sequential_20_bidirectional_20_forward_lstm_20_while_identityFsequential_20/bidirectional_20/forward_lstm_20/while/Identity:output:0*(
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
�%
�
while_body_2412059
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_61_2412083_0:	�/
while_lstm_cell_61_2412085_0:	2�+
while_lstm_cell_61_2412087_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_61_2412083:	�-
while_lstm_cell_61_2412085:	2�)
while_lstm_cell_61_2412087:	���*while/lstm_cell_61/StatefulPartitionedCall�
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
*while/lstm_cell_61/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_61_2412083_0while_lstm_cell_61_2412085_0while_lstm_cell_61_2412087_0*
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
I__inference_lstm_cell_61_layer_call_and_return_conditional_losses_24119812,
*while/lstm_cell_61/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_61/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_61/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_61/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_61/StatefulPartitionedCall*"
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
while_lstm_cell_61_2412083while_lstm_cell_61_2412083_0":
while_lstm_cell_61_2412085while_lstm_cell_61_2412085_0":
while_lstm_cell_61_2412087while_lstm_cell_61_2412087_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2X
*while/lstm_cell_61/StatefulPartitionedCall*while/lstm_cell_61/StatefulPartitionedCall: 
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
�
�
I__inference_lstm_cell_62_layer_call_and_return_conditional_losses_2412467

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
while_body_2416337
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_61_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_61_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_61_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_61_matmul_readvariableop_resource:	�F
3while_lstm_cell_61_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_61_biasadd_readvariableop_resource:	���)while/lstm_cell_61/BiasAdd/ReadVariableOp�(while/lstm_cell_61/MatMul/ReadVariableOp�*while/lstm_cell_61/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_61/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_61_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_61/MatMul/ReadVariableOp�
while/lstm_cell_61/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/MatMul�
*while/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_61_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_61/MatMul_1/ReadVariableOp�
while/lstm_cell_61/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/MatMul_1�
while/lstm_cell_61/addAddV2#while/lstm_cell_61/MatMul:product:0%while/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/add�
)while/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_61_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_61/BiasAdd/ReadVariableOp�
while/lstm_cell_61/BiasAddBiasAddwhile/lstm_cell_61/add:z:01while/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/BiasAdd�
"while/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_61/split/split_dim�
while/lstm_cell_61/splitSplit+while/lstm_cell_61/split/split_dim:output:0#while/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_61/split�
while/lstm_cell_61/SigmoidSigmoid!while/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Sigmoid�
while/lstm_cell_61/Sigmoid_1Sigmoid!while/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Sigmoid_1�
while/lstm_cell_61/mulMul while/lstm_cell_61/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/mul�
while/lstm_cell_61/ReluRelu!while/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Relu�
while/lstm_cell_61/mul_1Mulwhile/lstm_cell_61/Sigmoid:y:0%while/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/mul_1�
while/lstm_cell_61/add_1AddV2while/lstm_cell_61/mul:z:0while/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/add_1�
while/lstm_cell_61/Sigmoid_2Sigmoid!while/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Sigmoid_2�
while/lstm_cell_61/Relu_1Reluwhile/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Relu_1�
while/lstm_cell_61/mul_2Mul while/lstm_cell_61/Sigmoid_2:y:0'while/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_61/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_61/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_61/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_61/BiasAdd/ReadVariableOp)^while/lstm_cell_61/MatMul/ReadVariableOp+^while/lstm_cell_61/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_61_biasadd_readvariableop_resource4while_lstm_cell_61_biasadd_readvariableop_resource_0"l
3while_lstm_cell_61_matmul_1_readvariableop_resource5while_lstm_cell_61_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_61_matmul_readvariableop_resource3while_lstm_cell_61_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_61/BiasAdd/ReadVariableOp)while/lstm_cell_61/BiasAdd/ReadVariableOp2T
(while/lstm_cell_61/MatMul/ReadVariableOp(while/lstm_cell_61/MatMul/ReadVariableOp2X
*while/lstm_cell_61/MatMul_1/ReadVariableOp*while/lstm_cell_61/MatMul_1/ReadVariableOp: 
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
while_body_2416987
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_62_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_62_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_62_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_62_matmul_readvariableop_resource:	�F
3while_lstm_cell_62_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_62_biasadd_readvariableop_resource:	���)while/lstm_cell_62/BiasAdd/ReadVariableOp�(while/lstm_cell_62/MatMul/ReadVariableOp�*while/lstm_cell_62/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_62/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_62_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_62/MatMul/ReadVariableOp�
while/lstm_cell_62/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/MatMul�
*while/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_62_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_62/MatMul_1/ReadVariableOp�
while/lstm_cell_62/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/MatMul_1�
while/lstm_cell_62/addAddV2#while/lstm_cell_62/MatMul:product:0%while/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/add�
)while/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_62_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_62/BiasAdd/ReadVariableOp�
while/lstm_cell_62/BiasAddBiasAddwhile/lstm_cell_62/add:z:01while/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_62/BiasAdd�
"while/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_62/split/split_dim�
while/lstm_cell_62/splitSplit+while/lstm_cell_62/split/split_dim:output:0#while/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_62/split�
while/lstm_cell_62/SigmoidSigmoid!while/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Sigmoid�
while/lstm_cell_62/Sigmoid_1Sigmoid!while/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Sigmoid_1�
while/lstm_cell_62/mulMul while/lstm_cell_62/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/mul�
while/lstm_cell_62/ReluRelu!while/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Relu�
while/lstm_cell_62/mul_1Mulwhile/lstm_cell_62/Sigmoid:y:0%while/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/mul_1�
while/lstm_cell_62/add_1AddV2while/lstm_cell_62/mul:z:0while/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/add_1�
while/lstm_cell_62/Sigmoid_2Sigmoid!while/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Sigmoid_2�
while/lstm_cell_62/Relu_1Reluwhile/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/Relu_1�
while/lstm_cell_62/mul_2Mul while/lstm_cell_62/Sigmoid_2:y:0'while/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_62/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_62/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_62/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_62/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_62/BiasAdd/ReadVariableOp)^while/lstm_cell_62/MatMul/ReadVariableOp+^while/lstm_cell_62/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_62_biasadd_readvariableop_resource4while_lstm_cell_62_biasadd_readvariableop_resource_0"l
3while_lstm_cell_62_matmul_1_readvariableop_resource5while_lstm_cell_62_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_62_matmul_readvariableop_resource3while_lstm_cell_62_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_62/BiasAdd/ReadVariableOp)while/lstm_cell_62/BiasAdd/ReadVariableOp2T
(while/lstm_cell_62/MatMul/ReadVariableOp(while/lstm_cell_62/MatMul/ReadVariableOp2X
*while/lstm_cell_62/MatMul_1/ReadVariableOp*while/lstm_cell_62/MatMul_1/ReadVariableOp: 
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
�
�
I__inference_lstm_cell_62_layer_call_and_return_conditional_losses_2412613

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
�

�
/__inference_sequential_20_layer_call_fn_2414247

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
J__inference_sequential_20_layer_call_and_return_conditional_losses_24142282
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
�
�
while_cond_2417139
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2417139___redundant_placeholder05
1while_while_cond_2417139___redundant_placeholder15
1while_while_cond_2417139___redundant_placeholder25
1while_while_cond_2417139___redundant_placeholder3
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
�
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_2415490
inputs_0N
;forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource:	�P
=forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource:	2�K
<forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource:	�O
<backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource:	�Q
>backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource:	2�L
=backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource:	�
identity��4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp�3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp�5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp�backward_lstm_20/while�3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp�2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp�4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp�forward_lstm_20/whilef
forward_lstm_20/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_20/Shape�
#forward_lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_20/strided_slice/stack�
%forward_lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_20/strided_slice/stack_1�
%forward_lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_20/strided_slice/stack_2�
forward_lstm_20/strided_sliceStridedSliceforward_lstm_20/Shape:output:0,forward_lstm_20/strided_slice/stack:output:0.forward_lstm_20/strided_slice/stack_1:output:0.forward_lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_20/strided_slice|
forward_lstm_20/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_20/zeros/mul/y�
forward_lstm_20/zeros/mulMul&forward_lstm_20/strided_slice:output:0$forward_lstm_20/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros/mul
forward_lstm_20/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_20/zeros/Less/y�
forward_lstm_20/zeros/LessLessforward_lstm_20/zeros/mul:z:0%forward_lstm_20/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros/Less�
forward_lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_20/zeros/packed/1�
forward_lstm_20/zeros/packedPack&forward_lstm_20/strided_slice:output:0'forward_lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_20/zeros/packed�
forward_lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_20/zeros/Const�
forward_lstm_20/zerosFill%forward_lstm_20/zeros/packed:output:0$forward_lstm_20/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_20/zeros�
forward_lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_20/zeros_1/mul/y�
forward_lstm_20/zeros_1/mulMul&forward_lstm_20/strided_slice:output:0&forward_lstm_20/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros_1/mul�
forward_lstm_20/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
forward_lstm_20/zeros_1/Less/y�
forward_lstm_20/zeros_1/LessLessforward_lstm_20/zeros_1/mul:z:0'forward_lstm_20/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros_1/Less�
 forward_lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_20/zeros_1/packed/1�
forward_lstm_20/zeros_1/packedPack&forward_lstm_20/strided_slice:output:0)forward_lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_20/zeros_1/packed�
forward_lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_20/zeros_1/Const�
forward_lstm_20/zeros_1Fill'forward_lstm_20/zeros_1/packed:output:0&forward_lstm_20/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_20/zeros_1�
forward_lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_20/transpose/perm�
forward_lstm_20/transpose	Transposeinputs_0'forward_lstm_20/transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
forward_lstm_20/transpose
forward_lstm_20/Shape_1Shapeforward_lstm_20/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_20/Shape_1�
%forward_lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_20/strided_slice_1/stack�
'forward_lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_1/stack_1�
'forward_lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_1/stack_2�
forward_lstm_20/strided_slice_1StridedSlice forward_lstm_20/Shape_1:output:0.forward_lstm_20/strided_slice_1/stack:output:00forward_lstm_20/strided_slice_1/stack_1:output:00forward_lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_20/strided_slice_1�
+forward_lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+forward_lstm_20/TensorArrayV2/element_shape�
forward_lstm_20/TensorArrayV2TensorListReserve4forward_lstm_20/TensorArrayV2/element_shape:output:0(forward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_20/TensorArrayV2�
Eforward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2G
Eforward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape�
7forward_lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_20/transpose:y:0Nforward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_20/TensorArrayUnstack/TensorListFromTensor�
%forward_lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_20/strided_slice_2/stack�
'forward_lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_2/stack_1�
'forward_lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_2/stack_2�
forward_lstm_20/strided_slice_2StridedSliceforward_lstm_20/transpose:y:0.forward_lstm_20/strided_slice_2/stack:output:00forward_lstm_20/strided_slice_2/stack_1:output:00forward_lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2!
forward_lstm_20/strided_slice_2�
2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOpReadVariableOp;forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp�
#forward_lstm_20/lstm_cell_61/MatMulMatMul(forward_lstm_20/strided_slice_2:output:0:forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#forward_lstm_20/lstm_cell_61/MatMul�
4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOp=forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype026
4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp�
%forward_lstm_20/lstm_cell_61/MatMul_1MatMulforward_lstm_20/zeros:output:0<forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%forward_lstm_20/lstm_cell_61/MatMul_1�
 forward_lstm_20/lstm_cell_61/addAddV2-forward_lstm_20/lstm_cell_61/MatMul:product:0/forward_lstm_20/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2"
 forward_lstm_20/lstm_cell_61/add�
3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOp<forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp�
$forward_lstm_20/lstm_cell_61/BiasAddBiasAdd$forward_lstm_20/lstm_cell_61/add:z:0;forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$forward_lstm_20/lstm_cell_61/BiasAdd�
,forward_lstm_20/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,forward_lstm_20/lstm_cell_61/split/split_dim�
"forward_lstm_20/lstm_cell_61/splitSplit5forward_lstm_20/lstm_cell_61/split/split_dim:output:0-forward_lstm_20/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2$
"forward_lstm_20/lstm_cell_61/split�
$forward_lstm_20/lstm_cell_61/SigmoidSigmoid+forward_lstm_20/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22&
$forward_lstm_20/lstm_cell_61/Sigmoid�
&forward_lstm_20/lstm_cell_61/Sigmoid_1Sigmoid+forward_lstm_20/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22(
&forward_lstm_20/lstm_cell_61/Sigmoid_1�
 forward_lstm_20/lstm_cell_61/mulMul*forward_lstm_20/lstm_cell_61/Sigmoid_1:y:0 forward_lstm_20/zeros_1:output:0*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/lstm_cell_61/mul�
!forward_lstm_20/lstm_cell_61/ReluRelu+forward_lstm_20/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22#
!forward_lstm_20/lstm_cell_61/Relu�
"forward_lstm_20/lstm_cell_61/mul_1Mul(forward_lstm_20/lstm_cell_61/Sigmoid:y:0/forward_lstm_20/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_20/lstm_cell_61/mul_1�
"forward_lstm_20/lstm_cell_61/add_1AddV2$forward_lstm_20/lstm_cell_61/mul:z:0&forward_lstm_20/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_20/lstm_cell_61/add_1�
&forward_lstm_20/lstm_cell_61/Sigmoid_2Sigmoid+forward_lstm_20/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22(
&forward_lstm_20/lstm_cell_61/Sigmoid_2�
#forward_lstm_20/lstm_cell_61/Relu_1Relu&forward_lstm_20/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22%
#forward_lstm_20/lstm_cell_61/Relu_1�
"forward_lstm_20/lstm_cell_61/mul_2Mul*forward_lstm_20/lstm_cell_61/Sigmoid_2:y:01forward_lstm_20/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_20/lstm_cell_61/mul_2�
-forward_lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2/
-forward_lstm_20/TensorArrayV2_1/element_shape�
forward_lstm_20/TensorArrayV2_1TensorListReserve6forward_lstm_20/TensorArrayV2_1/element_shape:output:0(forward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_20/TensorArrayV2_1n
forward_lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_20/time�
(forward_lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(forward_lstm_20/while/maximum_iterations�
"forward_lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_20/while/loop_counter�
forward_lstm_20/whileWhile+forward_lstm_20/while/loop_counter:output:01forward_lstm_20/while/maximum_iterations:output:0forward_lstm_20/time:output:0(forward_lstm_20/TensorArrayV2_1:handle:0forward_lstm_20/zeros:output:0 forward_lstm_20/zeros_1:output:0(forward_lstm_20/strided_slice_1:output:0Gforward_lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:0;forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource=forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource<forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource*
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
"forward_lstm_20_while_body_2415255*.
cond&R$
"forward_lstm_20_while_cond_2415254*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
forward_lstm_20/while�
@forward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2B
@forward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape�
2forward_lstm_20/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_20/while:output:3Iforward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype024
2forward_lstm_20/TensorArrayV2Stack/TensorListStack�
%forward_lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2'
%forward_lstm_20/strided_slice_3/stack�
'forward_lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_20/strided_slice_3/stack_1�
'forward_lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_3/stack_2�
forward_lstm_20/strided_slice_3StridedSlice;forward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_20/strided_slice_3/stack:output:00forward_lstm_20/strided_slice_3/stack_1:output:00forward_lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2!
forward_lstm_20/strided_slice_3�
 forward_lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_20/transpose_1/perm�
forward_lstm_20/transpose_1	Transpose;forward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_20/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
forward_lstm_20/transpose_1�
forward_lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_20/runtimeh
backward_lstm_20/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_20/Shape�
$backward_lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_20/strided_slice/stack�
&backward_lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_20/strided_slice/stack_1�
&backward_lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_20/strided_slice/stack_2�
backward_lstm_20/strided_sliceStridedSlicebackward_lstm_20/Shape:output:0-backward_lstm_20/strided_slice/stack:output:0/backward_lstm_20/strided_slice/stack_1:output:0/backward_lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_20/strided_slice~
backward_lstm_20/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_20/zeros/mul/y�
backward_lstm_20/zeros/mulMul'backward_lstm_20/strided_slice:output:0%backward_lstm_20/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros/mul�
backward_lstm_20/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
backward_lstm_20/zeros/Less/y�
backward_lstm_20/zeros/LessLessbackward_lstm_20/zeros/mul:z:0&backward_lstm_20/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros/Less�
backward_lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_20/zeros/packed/1�
backward_lstm_20/zeros/packedPack'backward_lstm_20/strided_slice:output:0(backward_lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_20/zeros/packed�
backward_lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_20/zeros/Const�
backward_lstm_20/zerosFill&backward_lstm_20/zeros/packed:output:0%backward_lstm_20/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_20/zeros�
backward_lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_20/zeros_1/mul/y�
backward_lstm_20/zeros_1/mulMul'backward_lstm_20/strided_slice:output:0'backward_lstm_20/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros_1/mul�
backward_lstm_20/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2!
backward_lstm_20/zeros_1/Less/y�
backward_lstm_20/zeros_1/LessLess backward_lstm_20/zeros_1/mul:z:0(backward_lstm_20/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros_1/Less�
!backward_lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_20/zeros_1/packed/1�
backward_lstm_20/zeros_1/packedPack'backward_lstm_20/strided_slice:output:0*backward_lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_20/zeros_1/packed�
backward_lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_20/zeros_1/Const�
backward_lstm_20/zeros_1Fill(backward_lstm_20/zeros_1/packed:output:0'backward_lstm_20/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_20/zeros_1�
backward_lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_20/transpose/perm�
backward_lstm_20/transpose	Transposeinputs_0(backward_lstm_20/transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
backward_lstm_20/transpose�
backward_lstm_20/Shape_1Shapebackward_lstm_20/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_20/Shape_1�
&backward_lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_20/strided_slice_1/stack�
(backward_lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_1/stack_1�
(backward_lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_1/stack_2�
 backward_lstm_20/strided_slice_1StridedSlice!backward_lstm_20/Shape_1:output:0/backward_lstm_20/strided_slice_1/stack:output:01backward_lstm_20/strided_slice_1/stack_1:output:01backward_lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_20/strided_slice_1�
,backward_lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2.
,backward_lstm_20/TensorArrayV2/element_shape�
backward_lstm_20/TensorArrayV2TensorListReserve5backward_lstm_20/TensorArrayV2/element_shape:output:0)backward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_20/TensorArrayV2�
backward_lstm_20/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_20/ReverseV2/axis�
backward_lstm_20/ReverseV2	ReverseV2backward_lstm_20/transpose:y:0(backward_lstm_20/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'���������������������������2
backward_lstm_20/ReverseV2�
Fbackward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2H
Fbackward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape�
8backward_lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_20/ReverseV2:output:0Obackward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_20/TensorArrayUnstack/TensorListFromTensor�
&backward_lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_20/strided_slice_2/stack�
(backward_lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_2/stack_1�
(backward_lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_2/stack_2�
 backward_lstm_20/strided_slice_2StridedSlicebackward_lstm_20/transpose:y:0/backward_lstm_20/strided_slice_2/stack:output:01backward_lstm_20/strided_slice_2/stack_1:output:01backward_lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2"
 backward_lstm_20/strided_slice_2�
3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOpReadVariableOp<backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype025
3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp�
$backward_lstm_20/lstm_cell_62/MatMulMatMul)backward_lstm_20/strided_slice_2:output:0;backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$backward_lstm_20/lstm_cell_62/MatMul�
5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOp>backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype027
5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp�
&backward_lstm_20/lstm_cell_62/MatMul_1MatMulbackward_lstm_20/zeros:output:0=backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&backward_lstm_20/lstm_cell_62/MatMul_1�
!backward_lstm_20/lstm_cell_62/addAddV2.backward_lstm_20/lstm_cell_62/MatMul:product:00backward_lstm_20/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2#
!backward_lstm_20/lstm_cell_62/add�
4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOp=backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype026
4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp�
%backward_lstm_20/lstm_cell_62/BiasAddBiasAdd%backward_lstm_20/lstm_cell_62/add:z:0<backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%backward_lstm_20/lstm_cell_62/BiasAdd�
-backward_lstm_20/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-backward_lstm_20/lstm_cell_62/split/split_dim�
#backward_lstm_20/lstm_cell_62/splitSplit6backward_lstm_20/lstm_cell_62/split/split_dim:output:0.backward_lstm_20/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2%
#backward_lstm_20/lstm_cell_62/split�
%backward_lstm_20/lstm_cell_62/SigmoidSigmoid,backward_lstm_20/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22'
%backward_lstm_20/lstm_cell_62/Sigmoid�
'backward_lstm_20/lstm_cell_62/Sigmoid_1Sigmoid,backward_lstm_20/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22)
'backward_lstm_20/lstm_cell_62/Sigmoid_1�
!backward_lstm_20/lstm_cell_62/mulMul+backward_lstm_20/lstm_cell_62/Sigmoid_1:y:0!backward_lstm_20/zeros_1:output:0*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/lstm_cell_62/mul�
"backward_lstm_20/lstm_cell_62/ReluRelu,backward_lstm_20/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22$
"backward_lstm_20/lstm_cell_62/Relu�
#backward_lstm_20/lstm_cell_62/mul_1Mul)backward_lstm_20/lstm_cell_62/Sigmoid:y:00backward_lstm_20/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_20/lstm_cell_62/mul_1�
#backward_lstm_20/lstm_cell_62/add_1AddV2%backward_lstm_20/lstm_cell_62/mul:z:0'backward_lstm_20/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_20/lstm_cell_62/add_1�
'backward_lstm_20/lstm_cell_62/Sigmoid_2Sigmoid,backward_lstm_20/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22)
'backward_lstm_20/lstm_cell_62/Sigmoid_2�
$backward_lstm_20/lstm_cell_62/Relu_1Relu'backward_lstm_20/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22&
$backward_lstm_20/lstm_cell_62/Relu_1�
#backward_lstm_20/lstm_cell_62/mul_2Mul+backward_lstm_20/lstm_cell_62/Sigmoid_2:y:02backward_lstm_20/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_20/lstm_cell_62/mul_2�
.backward_lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   20
.backward_lstm_20/TensorArrayV2_1/element_shape�
 backward_lstm_20/TensorArrayV2_1TensorListReserve7backward_lstm_20/TensorArrayV2_1/element_shape:output:0)backward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_20/TensorArrayV2_1p
backward_lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_20/time�
)backward_lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2+
)backward_lstm_20/while/maximum_iterations�
#backward_lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_20/while/loop_counter�
backward_lstm_20/whileWhile,backward_lstm_20/while/loop_counter:output:02backward_lstm_20/while/maximum_iterations:output:0backward_lstm_20/time:output:0)backward_lstm_20/TensorArrayV2_1:handle:0backward_lstm_20/zeros:output:0!backward_lstm_20/zeros_1:output:0)backward_lstm_20/strided_slice_1:output:0Hbackward_lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:0<backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource>backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource=backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource*
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
#backward_lstm_20_while_body_2415404*/
cond'R%
#backward_lstm_20_while_cond_2415403*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
backward_lstm_20/while�
Abackward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2C
Abackward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape�
3backward_lstm_20/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_20/while:output:3Jbackward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype025
3backward_lstm_20/TensorArrayV2Stack/TensorListStack�
&backward_lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2(
&backward_lstm_20/strided_slice_3/stack�
(backward_lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_20/strided_slice_3/stack_1�
(backward_lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_3/stack_2�
 backward_lstm_20/strided_slice_3StridedSlice<backward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_20/strided_slice_3/stack:output:01backward_lstm_20/strided_slice_3/stack_1:output:01backward_lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2"
 backward_lstm_20/strided_slice_3�
!backward_lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_20/transpose_1/perm�
backward_lstm_20/transpose_1	Transpose<backward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_20/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
backward_lstm_20/transpose_1�
backward_lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_20/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2(forward_lstm_20/strided_slice_3:output:0)backward_lstm_20/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp5^backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp4^backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp6^backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp^backward_lstm_20/while4^forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp3^forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp5^forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp^forward_lstm_20/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 2l
4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp2j
3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp2n
5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp20
backward_lstm_20/whilebackward_lstm_20/while2j
3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp2h
2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp2l
4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp2.
forward_lstm_20/whileforward_lstm_20/while:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs/0
�
�
Bsequential_20_bidirectional_20_backward_lstm_20_while_cond_2411655|
xsequential_20_bidirectional_20_backward_lstm_20_while_sequential_20_bidirectional_20_backward_lstm_20_while_loop_counter�
~sequential_20_bidirectional_20_backward_lstm_20_while_sequential_20_bidirectional_20_backward_lstm_20_while_maximum_iterationsE
Asequential_20_bidirectional_20_backward_lstm_20_while_placeholderG
Csequential_20_bidirectional_20_backward_lstm_20_while_placeholder_1G
Csequential_20_bidirectional_20_backward_lstm_20_while_placeholder_2G
Csequential_20_bidirectional_20_backward_lstm_20_while_placeholder_3G
Csequential_20_bidirectional_20_backward_lstm_20_while_placeholder_4~
zsequential_20_bidirectional_20_backward_lstm_20_while_less_sequential_20_bidirectional_20_backward_lstm_20_strided_slice_1�
�sequential_20_bidirectional_20_backward_lstm_20_while_sequential_20_bidirectional_20_backward_lstm_20_while_cond_2411655___redundant_placeholder0�
�sequential_20_bidirectional_20_backward_lstm_20_while_sequential_20_bidirectional_20_backward_lstm_20_while_cond_2411655___redundant_placeholder1�
�sequential_20_bidirectional_20_backward_lstm_20_while_sequential_20_bidirectional_20_backward_lstm_20_while_cond_2411655___redundant_placeholder2�
�sequential_20_bidirectional_20_backward_lstm_20_while_sequential_20_bidirectional_20_backward_lstm_20_while_cond_2411655___redundant_placeholder3�
�sequential_20_bidirectional_20_backward_lstm_20_while_sequential_20_bidirectional_20_backward_lstm_20_while_cond_2411655___redundant_placeholder4B
>sequential_20_bidirectional_20_backward_lstm_20_while_identity
�
:sequential_20/bidirectional_20/backward_lstm_20/while/LessLessAsequential_20_bidirectional_20_backward_lstm_20_while_placeholderzsequential_20_bidirectional_20_backward_lstm_20_while_less_sequential_20_bidirectional_20_backward_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2<
:sequential_20/bidirectional_20/backward_lstm_20/while/Less�
>sequential_20/bidirectional_20/backward_lstm_20/while/IdentityIdentity>sequential_20/bidirectional_20/backward_lstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2@
>sequential_20/bidirectional_20/backward_lstm_20/while/Identity"�
>sequential_20_bidirectional_20_backward_lstm_20_while_identityGsequential_20/bidirectional_20/backward_lstm_20/while/Identity:output:0*(
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
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_2415848

inputs
inputs_1	N
;forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource:	�P
=forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource:	2�K
<forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource:	�O
<backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource:	�Q
>backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource:	2�L
=backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource:	�
identity��4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp�3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp�5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp�backward_lstm_20/while�3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp�2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp�4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp�forward_lstm_20/while�
$forward_lstm_20/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$forward_lstm_20/RaggedToTensor/zeros�
$forward_lstm_20/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2&
$forward_lstm_20/RaggedToTensor/Const�
3forward_lstm_20/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-forward_lstm_20/RaggedToTensor/Const:output:0inputs-forward_lstm_20/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3forward_lstm_20/RaggedToTensor/RaggedTensorToTensor�
:forward_lstm_20/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack�
<forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1�
<forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2�
4forward_lstm_20/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cforward_lstm_20/RaggedNestedRowLengths/strided_slice/stack:output:0Eforward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1:output:0Eforward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask26
4forward_lstm_20/RaggedNestedRowLengths/strided_slice�
<forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack�
>forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2@
>forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1�
>forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2�
6forward_lstm_20/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Eforward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gforward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gforward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask28
6forward_lstm_20/RaggedNestedRowLengths/strided_slice_1�
*forward_lstm_20/RaggedNestedRowLengths/subSub=forward_lstm_20/RaggedNestedRowLengths/strided_slice:output:0?forward_lstm_20/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2,
*forward_lstm_20/RaggedNestedRowLengths/sub�
forward_lstm_20/CastCast.forward_lstm_20/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
forward_lstm_20/Cast�
forward_lstm_20/ShapeShape<forward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_20/Shape�
#forward_lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_20/strided_slice/stack�
%forward_lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_20/strided_slice/stack_1�
%forward_lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_20/strided_slice/stack_2�
forward_lstm_20/strided_sliceStridedSliceforward_lstm_20/Shape:output:0,forward_lstm_20/strided_slice/stack:output:0.forward_lstm_20/strided_slice/stack_1:output:0.forward_lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_20/strided_slice|
forward_lstm_20/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_20/zeros/mul/y�
forward_lstm_20/zeros/mulMul&forward_lstm_20/strided_slice:output:0$forward_lstm_20/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros/mul
forward_lstm_20/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_20/zeros/Less/y�
forward_lstm_20/zeros/LessLessforward_lstm_20/zeros/mul:z:0%forward_lstm_20/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros/Less�
forward_lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_20/zeros/packed/1�
forward_lstm_20/zeros/packedPack&forward_lstm_20/strided_slice:output:0'forward_lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_20/zeros/packed�
forward_lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_20/zeros/Const�
forward_lstm_20/zerosFill%forward_lstm_20/zeros/packed:output:0$forward_lstm_20/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_20/zeros�
forward_lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_20/zeros_1/mul/y�
forward_lstm_20/zeros_1/mulMul&forward_lstm_20/strided_slice:output:0&forward_lstm_20/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros_1/mul�
forward_lstm_20/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
forward_lstm_20/zeros_1/Less/y�
forward_lstm_20/zeros_1/LessLessforward_lstm_20/zeros_1/mul:z:0'forward_lstm_20/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros_1/Less�
 forward_lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_20/zeros_1/packed/1�
forward_lstm_20/zeros_1/packedPack&forward_lstm_20/strided_slice:output:0)forward_lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_20/zeros_1/packed�
forward_lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_20/zeros_1/Const�
forward_lstm_20/zeros_1Fill'forward_lstm_20/zeros_1/packed:output:0&forward_lstm_20/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_20/zeros_1�
forward_lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_20/transpose/perm�
forward_lstm_20/transpose	Transpose<forward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0'forward_lstm_20/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
forward_lstm_20/transpose
forward_lstm_20/Shape_1Shapeforward_lstm_20/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_20/Shape_1�
%forward_lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_20/strided_slice_1/stack�
'forward_lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_1/stack_1�
'forward_lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_1/stack_2�
forward_lstm_20/strided_slice_1StridedSlice forward_lstm_20/Shape_1:output:0.forward_lstm_20/strided_slice_1/stack:output:00forward_lstm_20/strided_slice_1/stack_1:output:00forward_lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_20/strided_slice_1�
+forward_lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+forward_lstm_20/TensorArrayV2/element_shape�
forward_lstm_20/TensorArrayV2TensorListReserve4forward_lstm_20/TensorArrayV2/element_shape:output:0(forward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_20/TensorArrayV2�
Eforward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2G
Eforward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape�
7forward_lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_20/transpose:y:0Nforward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_20/TensorArrayUnstack/TensorListFromTensor�
%forward_lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_20/strided_slice_2/stack�
'forward_lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_2/stack_1�
'forward_lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_2/stack_2�
forward_lstm_20/strided_slice_2StridedSliceforward_lstm_20/transpose:y:0.forward_lstm_20/strided_slice_2/stack:output:00forward_lstm_20/strided_slice_2/stack_1:output:00forward_lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2!
forward_lstm_20/strided_slice_2�
2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOpReadVariableOp;forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp�
#forward_lstm_20/lstm_cell_61/MatMulMatMul(forward_lstm_20/strided_slice_2:output:0:forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#forward_lstm_20/lstm_cell_61/MatMul�
4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOp=forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype026
4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp�
%forward_lstm_20/lstm_cell_61/MatMul_1MatMulforward_lstm_20/zeros:output:0<forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%forward_lstm_20/lstm_cell_61/MatMul_1�
 forward_lstm_20/lstm_cell_61/addAddV2-forward_lstm_20/lstm_cell_61/MatMul:product:0/forward_lstm_20/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2"
 forward_lstm_20/lstm_cell_61/add�
3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOp<forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp�
$forward_lstm_20/lstm_cell_61/BiasAddBiasAdd$forward_lstm_20/lstm_cell_61/add:z:0;forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$forward_lstm_20/lstm_cell_61/BiasAdd�
,forward_lstm_20/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,forward_lstm_20/lstm_cell_61/split/split_dim�
"forward_lstm_20/lstm_cell_61/splitSplit5forward_lstm_20/lstm_cell_61/split/split_dim:output:0-forward_lstm_20/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2$
"forward_lstm_20/lstm_cell_61/split�
$forward_lstm_20/lstm_cell_61/SigmoidSigmoid+forward_lstm_20/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22&
$forward_lstm_20/lstm_cell_61/Sigmoid�
&forward_lstm_20/lstm_cell_61/Sigmoid_1Sigmoid+forward_lstm_20/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22(
&forward_lstm_20/lstm_cell_61/Sigmoid_1�
 forward_lstm_20/lstm_cell_61/mulMul*forward_lstm_20/lstm_cell_61/Sigmoid_1:y:0 forward_lstm_20/zeros_1:output:0*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/lstm_cell_61/mul�
!forward_lstm_20/lstm_cell_61/ReluRelu+forward_lstm_20/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22#
!forward_lstm_20/lstm_cell_61/Relu�
"forward_lstm_20/lstm_cell_61/mul_1Mul(forward_lstm_20/lstm_cell_61/Sigmoid:y:0/forward_lstm_20/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_20/lstm_cell_61/mul_1�
"forward_lstm_20/lstm_cell_61/add_1AddV2$forward_lstm_20/lstm_cell_61/mul:z:0&forward_lstm_20/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_20/lstm_cell_61/add_1�
&forward_lstm_20/lstm_cell_61/Sigmoid_2Sigmoid+forward_lstm_20/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22(
&forward_lstm_20/lstm_cell_61/Sigmoid_2�
#forward_lstm_20/lstm_cell_61/Relu_1Relu&forward_lstm_20/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22%
#forward_lstm_20/lstm_cell_61/Relu_1�
"forward_lstm_20/lstm_cell_61/mul_2Mul*forward_lstm_20/lstm_cell_61/Sigmoid_2:y:01forward_lstm_20/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_20/lstm_cell_61/mul_2�
-forward_lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2/
-forward_lstm_20/TensorArrayV2_1/element_shape�
forward_lstm_20/TensorArrayV2_1TensorListReserve6forward_lstm_20/TensorArrayV2_1/element_shape:output:0(forward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_20/TensorArrayV2_1n
forward_lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_20/time�
forward_lstm_20/zeros_like	ZerosLike&forward_lstm_20/lstm_cell_61/mul_2:z:0*
T0*'
_output_shapes
:���������22
forward_lstm_20/zeros_like�
(forward_lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(forward_lstm_20/while/maximum_iterations�
"forward_lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_20/while/loop_counter�
forward_lstm_20/whileWhile+forward_lstm_20/while/loop_counter:output:01forward_lstm_20/while/maximum_iterations:output:0forward_lstm_20/time:output:0(forward_lstm_20/TensorArrayV2_1:handle:0forward_lstm_20/zeros_like:y:0forward_lstm_20/zeros:output:0 forward_lstm_20/zeros_1:output:0(forward_lstm_20/strided_slice_1:output:0Gforward_lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_20/Cast:y:0;forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource=forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource<forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource*
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
"forward_lstm_20_while_body_2415572*.
cond&R$
"forward_lstm_20_while_cond_2415571*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
forward_lstm_20/while�
@forward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2B
@forward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape�
2forward_lstm_20/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_20/while:output:3Iforward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype024
2forward_lstm_20/TensorArrayV2Stack/TensorListStack�
%forward_lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2'
%forward_lstm_20/strided_slice_3/stack�
'forward_lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_20/strided_slice_3/stack_1�
'forward_lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_3/stack_2�
forward_lstm_20/strided_slice_3StridedSlice;forward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_20/strided_slice_3/stack:output:00forward_lstm_20/strided_slice_3/stack_1:output:00forward_lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2!
forward_lstm_20/strided_slice_3�
 forward_lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_20/transpose_1/perm�
forward_lstm_20/transpose_1	Transpose;forward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_20/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
forward_lstm_20/transpose_1�
forward_lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_20/runtime�
%backward_lstm_20/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%backward_lstm_20/RaggedToTensor/zeros�
%backward_lstm_20/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2'
%backward_lstm_20/RaggedToTensor/Const�
4backward_lstm_20/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.backward_lstm_20/RaggedToTensor/Const:output:0inputs.backward_lstm_20/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4backward_lstm_20/RaggedToTensor/RaggedTensorToTensor�
;backward_lstm_20/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack�
=backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1�
=backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2�
5backward_lstm_20/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dbackward_lstm_20/RaggedNestedRowLengths/strided_slice/stack:output:0Fbackward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fbackward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask27
5backward_lstm_20/RaggedNestedRowLengths/strided_slice�
=backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack�
?backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2A
?backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1�
?backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2�
7backward_lstm_20/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fbackward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hbackward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hbackward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask29
7backward_lstm_20/RaggedNestedRowLengths/strided_slice_1�
+backward_lstm_20/RaggedNestedRowLengths/subSub>backward_lstm_20/RaggedNestedRowLengths/strided_slice:output:0@backward_lstm_20/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2-
+backward_lstm_20/RaggedNestedRowLengths/sub�
backward_lstm_20/CastCast/backward_lstm_20/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
backward_lstm_20/Cast�
backward_lstm_20/ShapeShape=backward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_20/Shape�
$backward_lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_20/strided_slice/stack�
&backward_lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_20/strided_slice/stack_1�
&backward_lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_20/strided_slice/stack_2�
backward_lstm_20/strided_sliceStridedSlicebackward_lstm_20/Shape:output:0-backward_lstm_20/strided_slice/stack:output:0/backward_lstm_20/strided_slice/stack_1:output:0/backward_lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_20/strided_slice~
backward_lstm_20/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_20/zeros/mul/y�
backward_lstm_20/zeros/mulMul'backward_lstm_20/strided_slice:output:0%backward_lstm_20/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros/mul�
backward_lstm_20/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
backward_lstm_20/zeros/Less/y�
backward_lstm_20/zeros/LessLessbackward_lstm_20/zeros/mul:z:0&backward_lstm_20/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros/Less�
backward_lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_20/zeros/packed/1�
backward_lstm_20/zeros/packedPack'backward_lstm_20/strided_slice:output:0(backward_lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_20/zeros/packed�
backward_lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_20/zeros/Const�
backward_lstm_20/zerosFill&backward_lstm_20/zeros/packed:output:0%backward_lstm_20/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_20/zeros�
backward_lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_20/zeros_1/mul/y�
backward_lstm_20/zeros_1/mulMul'backward_lstm_20/strided_slice:output:0'backward_lstm_20/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros_1/mul�
backward_lstm_20/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2!
backward_lstm_20/zeros_1/Less/y�
backward_lstm_20/zeros_1/LessLess backward_lstm_20/zeros_1/mul:z:0(backward_lstm_20/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros_1/Less�
!backward_lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_20/zeros_1/packed/1�
backward_lstm_20/zeros_1/packedPack'backward_lstm_20/strided_slice:output:0*backward_lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_20/zeros_1/packed�
backward_lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_20/zeros_1/Const�
backward_lstm_20/zeros_1Fill(backward_lstm_20/zeros_1/packed:output:0'backward_lstm_20/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_20/zeros_1�
backward_lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_20/transpose/perm�
backward_lstm_20/transpose	Transpose=backward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0(backward_lstm_20/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_20/transpose�
backward_lstm_20/Shape_1Shapebackward_lstm_20/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_20/Shape_1�
&backward_lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_20/strided_slice_1/stack�
(backward_lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_1/stack_1�
(backward_lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_1/stack_2�
 backward_lstm_20/strided_slice_1StridedSlice!backward_lstm_20/Shape_1:output:0/backward_lstm_20/strided_slice_1/stack:output:01backward_lstm_20/strided_slice_1/stack_1:output:01backward_lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_20/strided_slice_1�
,backward_lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2.
,backward_lstm_20/TensorArrayV2/element_shape�
backward_lstm_20/TensorArrayV2TensorListReserve5backward_lstm_20/TensorArrayV2/element_shape:output:0)backward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_20/TensorArrayV2�
backward_lstm_20/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_20/ReverseV2/axis�
backward_lstm_20/ReverseV2	ReverseV2backward_lstm_20/transpose:y:0(backward_lstm_20/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_20/ReverseV2�
Fbackward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2H
Fbackward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape�
8backward_lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_20/ReverseV2:output:0Obackward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_20/TensorArrayUnstack/TensorListFromTensor�
&backward_lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_20/strided_slice_2/stack�
(backward_lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_2/stack_1�
(backward_lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_2/stack_2�
 backward_lstm_20/strided_slice_2StridedSlicebackward_lstm_20/transpose:y:0/backward_lstm_20/strided_slice_2/stack:output:01backward_lstm_20/strided_slice_2/stack_1:output:01backward_lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2"
 backward_lstm_20/strided_slice_2�
3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOpReadVariableOp<backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype025
3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp�
$backward_lstm_20/lstm_cell_62/MatMulMatMul)backward_lstm_20/strided_slice_2:output:0;backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$backward_lstm_20/lstm_cell_62/MatMul�
5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOp>backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype027
5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp�
&backward_lstm_20/lstm_cell_62/MatMul_1MatMulbackward_lstm_20/zeros:output:0=backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&backward_lstm_20/lstm_cell_62/MatMul_1�
!backward_lstm_20/lstm_cell_62/addAddV2.backward_lstm_20/lstm_cell_62/MatMul:product:00backward_lstm_20/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2#
!backward_lstm_20/lstm_cell_62/add�
4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOp=backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype026
4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp�
%backward_lstm_20/lstm_cell_62/BiasAddBiasAdd%backward_lstm_20/lstm_cell_62/add:z:0<backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%backward_lstm_20/lstm_cell_62/BiasAdd�
-backward_lstm_20/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-backward_lstm_20/lstm_cell_62/split/split_dim�
#backward_lstm_20/lstm_cell_62/splitSplit6backward_lstm_20/lstm_cell_62/split/split_dim:output:0.backward_lstm_20/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2%
#backward_lstm_20/lstm_cell_62/split�
%backward_lstm_20/lstm_cell_62/SigmoidSigmoid,backward_lstm_20/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22'
%backward_lstm_20/lstm_cell_62/Sigmoid�
'backward_lstm_20/lstm_cell_62/Sigmoid_1Sigmoid,backward_lstm_20/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22)
'backward_lstm_20/lstm_cell_62/Sigmoid_1�
!backward_lstm_20/lstm_cell_62/mulMul+backward_lstm_20/lstm_cell_62/Sigmoid_1:y:0!backward_lstm_20/zeros_1:output:0*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/lstm_cell_62/mul�
"backward_lstm_20/lstm_cell_62/ReluRelu,backward_lstm_20/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22$
"backward_lstm_20/lstm_cell_62/Relu�
#backward_lstm_20/lstm_cell_62/mul_1Mul)backward_lstm_20/lstm_cell_62/Sigmoid:y:00backward_lstm_20/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_20/lstm_cell_62/mul_1�
#backward_lstm_20/lstm_cell_62/add_1AddV2%backward_lstm_20/lstm_cell_62/mul:z:0'backward_lstm_20/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_20/lstm_cell_62/add_1�
'backward_lstm_20/lstm_cell_62/Sigmoid_2Sigmoid,backward_lstm_20/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22)
'backward_lstm_20/lstm_cell_62/Sigmoid_2�
$backward_lstm_20/lstm_cell_62/Relu_1Relu'backward_lstm_20/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22&
$backward_lstm_20/lstm_cell_62/Relu_1�
#backward_lstm_20/lstm_cell_62/mul_2Mul+backward_lstm_20/lstm_cell_62/Sigmoid_2:y:02backward_lstm_20/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_20/lstm_cell_62/mul_2�
.backward_lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   20
.backward_lstm_20/TensorArrayV2_1/element_shape�
 backward_lstm_20/TensorArrayV2_1TensorListReserve7backward_lstm_20/TensorArrayV2_1/element_shape:output:0)backward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_20/TensorArrayV2_1p
backward_lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_20/time�
&backward_lstm_20/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_lstm_20/Max/reduction_indices�
backward_lstm_20/MaxMaxbackward_lstm_20/Cast:y:0/backward_lstm_20/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/Maxr
backward_lstm_20/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_20/sub/y�
backward_lstm_20/subSubbackward_lstm_20/Max:output:0backward_lstm_20/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/sub�
backward_lstm_20/Sub_1Subbackward_lstm_20/sub:z:0backward_lstm_20/Cast:y:0*
T0*#
_output_shapes
:���������2
backward_lstm_20/Sub_1�
backward_lstm_20/zeros_like	ZerosLike'backward_lstm_20/lstm_cell_62/mul_2:z:0*
T0*'
_output_shapes
:���������22
backward_lstm_20/zeros_like�
)backward_lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2+
)backward_lstm_20/while/maximum_iterations�
#backward_lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_20/while/loop_counter�	
backward_lstm_20/whileWhile,backward_lstm_20/while/loop_counter:output:02backward_lstm_20/while/maximum_iterations:output:0backward_lstm_20/time:output:0)backward_lstm_20/TensorArrayV2_1:handle:0backward_lstm_20/zeros_like:y:0backward_lstm_20/zeros:output:0!backward_lstm_20/zeros_1:output:0)backward_lstm_20/strided_slice_1:output:0Hbackward_lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_20/Sub_1:z:0<backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource>backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource=backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource*
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
#backward_lstm_20_while_body_2415751*/
cond'R%
#backward_lstm_20_while_cond_2415750*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
backward_lstm_20/while�
Abackward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2C
Abackward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape�
3backward_lstm_20/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_20/while:output:3Jbackward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype025
3backward_lstm_20/TensorArrayV2Stack/TensorListStack�
&backward_lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2(
&backward_lstm_20/strided_slice_3/stack�
(backward_lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_20/strided_slice_3/stack_1�
(backward_lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_3/stack_2�
 backward_lstm_20/strided_slice_3StridedSlice<backward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_20/strided_slice_3/stack:output:01backward_lstm_20/strided_slice_3/stack_1:output:01backward_lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2"
 backward_lstm_20/strided_slice_3�
!backward_lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_20/transpose_1/perm�
backward_lstm_20/transpose_1	Transpose<backward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_20/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
backward_lstm_20/transpose_1�
backward_lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_20/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2(forward_lstm_20/strided_slice_3:output:0)backward_lstm_20/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp5^backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp4^backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp6^backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp^backward_lstm_20/while4^forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp3^forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp5^forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp^forward_lstm_20/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������: : : : : : 2l
4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp2j
3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp2n
5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp20
backward_lstm_20/whilebackward_lstm_20/while2j
3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp2h
2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp2l
4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp2.
forward_lstm_20/whileforward_lstm_20/while:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_sequential_20_layer_call_and_return_conditional_losses_2414763

inputs
inputs_1	+
bidirectional_20_2414744:	�+
bidirectional_20_2414746:	2�'
bidirectional_20_2414748:	�+
bidirectional_20_2414750:	�+
bidirectional_20_2414752:	2�'
bidirectional_20_2414754:	�"
dense_20_2414757:d
dense_20_2414759:
identity��(bidirectional_20/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�
(bidirectional_20/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_20_2414744bidirectional_20_2414746bidirectional_20_2414748bidirectional_20_2414750bidirectional_20_2414752bidirectional_20_2414754*
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
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_24141962*
(bidirectional_20/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_20/StatefulPartitionedCall:output:0dense_20_2414757dense_20_2414759*
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
E__inference_dense_20_layer_call_and_return_conditional_losses_24142212"
 dense_20/StatefulPartitionedCall�
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp)^bidirectional_20/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:���������:���������: : : : : : : : 2T
(bidirectional_20/StatefulPartitionedCall(bidirectional_20/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
#__inference__traced_restore_2417994
file_prefix2
 assignvariableop_dense_20_kernel:d.
 assignvariableop_1_dense_20_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: Z
Gassignvariableop_7_bidirectional_20_forward_lstm_20_lstm_cell_61_kernel:	�d
Qassignvariableop_8_bidirectional_20_forward_lstm_20_lstm_cell_61_recurrent_kernel:	2�T
Eassignvariableop_9_bidirectional_20_forward_lstm_20_lstm_cell_61_bias:	�\
Iassignvariableop_10_bidirectional_20_backward_lstm_20_lstm_cell_62_kernel:	�f
Sassignvariableop_11_bidirectional_20_backward_lstm_20_lstm_cell_62_recurrent_kernel:	2�V
Gassignvariableop_12_bidirectional_20_backward_lstm_20_lstm_cell_62_bias:	�#
assignvariableop_13_total: #
assignvariableop_14_count: <
*assignvariableop_15_adam_dense_20_kernel_m:d6
(assignvariableop_16_adam_dense_20_bias_m:b
Oassignvariableop_17_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_kernel_m:	�l
Yassignvariableop_18_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_recurrent_kernel_m:	2�\
Massignvariableop_19_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_bias_m:	�c
Passignvariableop_20_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_kernel_m:	�m
Zassignvariableop_21_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_recurrent_kernel_m:	2�]
Nassignvariableop_22_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_bias_m:	�<
*assignvariableop_23_adam_dense_20_kernel_v:d6
(assignvariableop_24_adam_dense_20_bias_v:b
Oassignvariableop_25_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_kernel_v:	�l
Yassignvariableop_26_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_recurrent_kernel_v:	2�\
Massignvariableop_27_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_bias_v:	�c
Passignvariableop_28_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_kernel_v:	�m
Zassignvariableop_29_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_recurrent_kernel_v:	2�]
Nassignvariableop_30_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_bias_v:	�?
-assignvariableop_31_adam_dense_20_kernel_vhat:d9
+assignvariableop_32_adam_dense_20_bias_vhat:e
Rassignvariableop_33_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_kernel_vhat:	�o
\assignvariableop_34_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_recurrent_kernel_vhat:	2�_
Passignvariableop_35_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_bias_vhat:	�f
Sassignvariableop_36_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_kernel_vhat:	�p
]assignvariableop_37_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_recurrent_kernel_vhat:	2�`
Qassignvariableop_38_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_bias_vhat:	�
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
AssignVariableOpAssignVariableOp assignvariableop_dense_20_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_20_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOpGassignvariableop_7_bidirectional_20_forward_lstm_20_lstm_cell_61_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpQassignvariableop_8_bidirectional_20_forward_lstm_20_lstm_cell_61_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpEassignvariableop_9_bidirectional_20_forward_lstm_20_lstm_cell_61_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpIassignvariableop_10_bidirectional_20_backward_lstm_20_lstm_cell_62_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpSassignvariableop_11_bidirectional_20_backward_lstm_20_lstm_cell_62_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpGassignvariableop_12_bidirectional_20_backward_lstm_20_lstm_cell_62_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_20_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_20_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpOassignvariableop_17_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpYassignvariableop_18_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpMassignvariableop_19_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpPassignvariableop_20_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpZassignvariableop_21_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpNassignvariableop_22_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_20_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_20_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOpOassignvariableop_25_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOpYassignvariableop_26_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOpMassignvariableop_27_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOpPassignvariableop_28_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOpZassignvariableop_29_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOpNassignvariableop_30_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp-assignvariableop_31_adam_dense_20_kernel_vhatIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_dense_20_bias_vhatIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOpRassignvariableop_33_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_kernel_vhatIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp\assignvariableop_34_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_recurrent_kernel_vhatIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOpPassignvariableop_35_adam_bidirectional_20_forward_lstm_20_lstm_cell_61_bias_vhatIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOpSassignvariableop_36_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_kernel_vhatIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp]assignvariableop_37_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_recurrent_kernel_vhatIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOpQassignvariableop_38_adam_bidirectional_20_backward_lstm_20_lstm_cell_62_bias_vhatIdentity_38:output:0"/device:CPU:0*
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
�^
�
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_2417530

inputs>
+lstm_cell_62_matmul_readvariableop_resource:	�@
-lstm_cell_62_matmul_1_readvariableop_resource:	2�;
,lstm_cell_62_biasadd_readvariableop_resource:	�
identity��#lstm_cell_62/BiasAdd/ReadVariableOp�"lstm_cell_62/MatMul/ReadVariableOp�$lstm_cell_62/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_62/MatMul/ReadVariableOpReadVariableOp+lstm_cell_62_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_62/MatMul/ReadVariableOp�
lstm_cell_62/MatMulMatMulstrided_slice_2:output:0*lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/MatMul�
$lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_62_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_62/MatMul_1/ReadVariableOp�
lstm_cell_62/MatMul_1MatMulzeros:output:0,lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/MatMul_1�
lstm_cell_62/addAddV2lstm_cell_62/MatMul:product:0lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/add�
#lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_62/BiasAdd/ReadVariableOp�
lstm_cell_62/BiasAddBiasAddlstm_cell_62/add:z:0+lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/BiasAdd~
lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_62/split/split_dim�
lstm_cell_62/splitSplit%lstm_cell_62/split/split_dim:output:0lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_62/split�
lstm_cell_62/SigmoidSigmoidlstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/Sigmoid�
lstm_cell_62/Sigmoid_1Sigmoidlstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_62/Sigmoid_1�
lstm_cell_62/mulMullstm_cell_62/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/mul}
lstm_cell_62/ReluRelulstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_62/Relu�
lstm_cell_62/mul_1Mullstm_cell_62/Sigmoid:y:0lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/mul_1�
lstm_cell_62/add_1AddV2lstm_cell_62/mul:z:0lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/add_1�
lstm_cell_62/Sigmoid_2Sigmoidlstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_62/Sigmoid_2|
lstm_cell_62/Relu_1Relulstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/Relu_1�
lstm_cell_62/mul_2Mullstm_cell_62/Sigmoid_2:y:0!lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_62_matmul_readvariableop_resource-lstm_cell_62_matmul_1_readvariableop_resource,lstm_cell_62_biasadd_readvariableop_resource*
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
while_body_2417446*
condR
while_cond_2417445*K
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
NoOpNoOp$^lstm_cell_62/BiasAdd/ReadVariableOp#^lstm_cell_62/MatMul/ReadVariableOp%^lstm_cell_62/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_62/BiasAdd/ReadVariableOp#lstm_cell_62/BiasAdd/ReadVariableOp2H
"lstm_cell_62/MatMul/ReadVariableOp"lstm_cell_62/MatMul/ReadVariableOp2L
$lstm_cell_62/MatMul_1/ReadVariableOp$lstm_cell_62/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
.__inference_lstm_cell_62_layer_call_fn_2417645

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
I__inference_lstm_cell_62_layer_call_and_return_conditional_losses_24124672
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
�
�
I__inference_lstm_cell_61_layer_call_and_return_conditional_losses_2411981

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
�
�
while_cond_2413452
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2413452___redundant_placeholder05
1while_while_cond_2413452___redundant_placeholder15
1while_while_cond_2413452___redundant_placeholder25
1while_while_cond_2413452___redundant_placeholder3
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
2__inference_bidirectional_20_layer_call_fn_2414886

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
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_24146362
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
ۗ
�
Asequential_20_bidirectional_20_forward_lstm_20_while_body_2411477z
vsequential_20_bidirectional_20_forward_lstm_20_while_sequential_20_bidirectional_20_forward_lstm_20_while_loop_counter�
|sequential_20_bidirectional_20_forward_lstm_20_while_sequential_20_bidirectional_20_forward_lstm_20_while_maximum_iterationsD
@sequential_20_bidirectional_20_forward_lstm_20_while_placeholderF
Bsequential_20_bidirectional_20_forward_lstm_20_while_placeholder_1F
Bsequential_20_bidirectional_20_forward_lstm_20_while_placeholder_2F
Bsequential_20_bidirectional_20_forward_lstm_20_while_placeholder_3F
Bsequential_20_bidirectional_20_forward_lstm_20_while_placeholder_4y
usequential_20_bidirectional_20_forward_lstm_20_while_sequential_20_bidirectional_20_forward_lstm_20_strided_slice_1_0�
�sequential_20_bidirectional_20_forward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_20_bidirectional_20_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0v
rsequential_20_bidirectional_20_forward_lstm_20_while_greater_sequential_20_bidirectional_20_forward_lstm_20_cast_0u
bsequential_20_bidirectional_20_forward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0:	�w
dsequential_20_bidirectional_20_forward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0:	2�r
csequential_20_bidirectional_20_forward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0:	�A
=sequential_20_bidirectional_20_forward_lstm_20_while_identityC
?sequential_20_bidirectional_20_forward_lstm_20_while_identity_1C
?sequential_20_bidirectional_20_forward_lstm_20_while_identity_2C
?sequential_20_bidirectional_20_forward_lstm_20_while_identity_3C
?sequential_20_bidirectional_20_forward_lstm_20_while_identity_4C
?sequential_20_bidirectional_20_forward_lstm_20_while_identity_5C
?sequential_20_bidirectional_20_forward_lstm_20_while_identity_6w
ssequential_20_bidirectional_20_forward_lstm_20_while_sequential_20_bidirectional_20_forward_lstm_20_strided_slice_1�
�sequential_20_bidirectional_20_forward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_20_bidirectional_20_forward_lstm_20_tensorarrayunstack_tensorlistfromtensort
psequential_20_bidirectional_20_forward_lstm_20_while_greater_sequential_20_bidirectional_20_forward_lstm_20_casts
`sequential_20_bidirectional_20_forward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource:	�u
bsequential_20_bidirectional_20_forward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource:	2�p
asequential_20_bidirectional_20_forward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource:	���Xsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp�Wsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp�Ysequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp�
fsequential_20/bidirectional_20/forward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2h
fsequential_20/bidirectional_20/forward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape�
Xsequential_20/bidirectional_20/forward_lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_20_bidirectional_20_forward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_20_bidirectional_20_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0@sequential_20_bidirectional_20_forward_lstm_20_while_placeholderosequential_20/bidirectional_20/forward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02Z
Xsequential_20/bidirectional_20/forward_lstm_20/while/TensorArrayV2Read/TensorListGetItem�
<sequential_20/bidirectional_20/forward_lstm_20/while/GreaterGreaterrsequential_20_bidirectional_20_forward_lstm_20_while_greater_sequential_20_bidirectional_20_forward_lstm_20_cast_0@sequential_20_bidirectional_20_forward_lstm_20_while_placeholder*
T0*#
_output_shapes
:���������2>
<sequential_20/bidirectional_20/forward_lstm_20/while/Greater�
Wsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOpReadVariableOpbsequential_20_bidirectional_20_forward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02Y
Wsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp�
Hsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMulMatMul_sequential_20/bidirectional_20/forward_lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:0_sequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2J
Hsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMul�
Ysequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOpdsequential_20_bidirectional_20_forward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02[
Ysequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp�
Jsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMul_1MatMulBsequential_20_bidirectional_20_forward_lstm_20_while_placeholder_3asequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2L
Jsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMul_1�
Esequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/addAddV2Rsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMul:product:0Tsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2G
Esequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/add�
Xsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOpcsequential_20_bidirectional_20_forward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02Z
Xsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp�
Isequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/BiasAddBiasAddIsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/add:z:0`sequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2K
Isequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/BiasAdd�
Qsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2S
Qsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/split/split_dim�
Gsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/splitSplitZsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/split/split_dim:output:0Rsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2I
Gsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/split�
Isequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/SigmoidSigmoidPsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22K
Isequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/Sigmoid�
Ksequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/Sigmoid_1SigmoidPsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22M
Ksequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/Sigmoid_1�
Esequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/mulMulOsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/Sigmoid_1:y:0Bsequential_20_bidirectional_20_forward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22G
Esequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/mul�
Fsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/ReluReluPsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22H
Fsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/Relu�
Gsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/mul_1MulMsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/Sigmoid:y:0Tsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22I
Gsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/mul_1�
Gsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/add_1AddV2Isequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/mul:z:0Ksequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22I
Gsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/add_1�
Ksequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/Sigmoid_2SigmoidPsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22M
Ksequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/Sigmoid_2�
Hsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/Relu_1ReluKsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22J
Hsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/Relu_1�
Gsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/mul_2MulOsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/Sigmoid_2:y:0Vsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22I
Gsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/mul_2�
;sequential_20/bidirectional_20/forward_lstm_20/while/SelectSelect@sequential_20/bidirectional_20/forward_lstm_20/while/Greater:z:0Ksequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/mul_2:z:0Bsequential_20_bidirectional_20_forward_lstm_20_while_placeholder_2*
T0*'
_output_shapes
:���������22=
;sequential_20/bidirectional_20/forward_lstm_20/while/Select�
=sequential_20/bidirectional_20/forward_lstm_20/while/Select_1Select@sequential_20/bidirectional_20/forward_lstm_20/while/Greater:z:0Ksequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/mul_2:z:0Bsequential_20_bidirectional_20_forward_lstm_20_while_placeholder_3*
T0*'
_output_shapes
:���������22?
=sequential_20/bidirectional_20/forward_lstm_20/while/Select_1�
=sequential_20/bidirectional_20/forward_lstm_20/while/Select_2Select@sequential_20/bidirectional_20/forward_lstm_20/while/Greater:z:0Ksequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/add_1:z:0Bsequential_20_bidirectional_20_forward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22?
=sequential_20/bidirectional_20/forward_lstm_20/while/Select_2�
Ysequential_20/bidirectional_20/forward_lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemBsequential_20_bidirectional_20_forward_lstm_20_while_placeholder_1@sequential_20_bidirectional_20_forward_lstm_20_while_placeholderDsequential_20/bidirectional_20/forward_lstm_20/while/Select:output:0*
_output_shapes
: *
element_dtype02[
Ysequential_20/bidirectional_20/forward_lstm_20/while/TensorArrayV2Write/TensorListSetItem�
:sequential_20/bidirectional_20/forward_lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_20/bidirectional_20/forward_lstm_20/while/add/y�
8sequential_20/bidirectional_20/forward_lstm_20/while/addAddV2@sequential_20_bidirectional_20_forward_lstm_20_while_placeholderCsequential_20/bidirectional_20/forward_lstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2:
8sequential_20/bidirectional_20/forward_lstm_20/while/add�
<sequential_20/bidirectional_20/forward_lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2>
<sequential_20/bidirectional_20/forward_lstm_20/while/add_1/y�
:sequential_20/bidirectional_20/forward_lstm_20/while/add_1AddV2vsequential_20_bidirectional_20_forward_lstm_20_while_sequential_20_bidirectional_20_forward_lstm_20_while_loop_counterEsequential_20/bidirectional_20/forward_lstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2<
:sequential_20/bidirectional_20/forward_lstm_20/while/add_1�
=sequential_20/bidirectional_20/forward_lstm_20/while/IdentityIdentity>sequential_20/bidirectional_20/forward_lstm_20/while/add_1:z:0:^sequential_20/bidirectional_20/forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2?
=sequential_20/bidirectional_20/forward_lstm_20/while/Identity�
?sequential_20/bidirectional_20/forward_lstm_20/while/Identity_1Identity|sequential_20_bidirectional_20_forward_lstm_20_while_sequential_20_bidirectional_20_forward_lstm_20_while_maximum_iterations:^sequential_20/bidirectional_20/forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2A
?sequential_20/bidirectional_20/forward_lstm_20/while/Identity_1�
?sequential_20/bidirectional_20/forward_lstm_20/while/Identity_2Identity<sequential_20/bidirectional_20/forward_lstm_20/while/add:z:0:^sequential_20/bidirectional_20/forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2A
?sequential_20/bidirectional_20/forward_lstm_20/while/Identity_2�
?sequential_20/bidirectional_20/forward_lstm_20/while/Identity_3Identityisequential_20/bidirectional_20/forward_lstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0:^sequential_20/bidirectional_20/forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2A
?sequential_20/bidirectional_20/forward_lstm_20/while/Identity_3�
?sequential_20/bidirectional_20/forward_lstm_20/while/Identity_4IdentityDsequential_20/bidirectional_20/forward_lstm_20/while/Select:output:0:^sequential_20/bidirectional_20/forward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22A
?sequential_20/bidirectional_20/forward_lstm_20/while/Identity_4�
?sequential_20/bidirectional_20/forward_lstm_20/while/Identity_5IdentityFsequential_20/bidirectional_20/forward_lstm_20/while/Select_1:output:0:^sequential_20/bidirectional_20/forward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22A
?sequential_20/bidirectional_20/forward_lstm_20/while/Identity_5�
?sequential_20/bidirectional_20/forward_lstm_20/while/Identity_6IdentityFsequential_20/bidirectional_20/forward_lstm_20/while/Select_2:output:0:^sequential_20/bidirectional_20/forward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22A
?sequential_20/bidirectional_20/forward_lstm_20/while/Identity_6�
9sequential_20/bidirectional_20/forward_lstm_20/while/NoOpNoOpY^sequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOpX^sequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOpZ^sequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2;
9sequential_20/bidirectional_20/forward_lstm_20/while/NoOp"�
psequential_20_bidirectional_20_forward_lstm_20_while_greater_sequential_20_bidirectional_20_forward_lstm_20_castrsequential_20_bidirectional_20_forward_lstm_20_while_greater_sequential_20_bidirectional_20_forward_lstm_20_cast_0"�
=sequential_20_bidirectional_20_forward_lstm_20_while_identityFsequential_20/bidirectional_20/forward_lstm_20/while/Identity:output:0"�
?sequential_20_bidirectional_20_forward_lstm_20_while_identity_1Hsequential_20/bidirectional_20/forward_lstm_20/while/Identity_1:output:0"�
?sequential_20_bidirectional_20_forward_lstm_20_while_identity_2Hsequential_20/bidirectional_20/forward_lstm_20/while/Identity_2:output:0"�
?sequential_20_bidirectional_20_forward_lstm_20_while_identity_3Hsequential_20/bidirectional_20/forward_lstm_20/while/Identity_3:output:0"�
?sequential_20_bidirectional_20_forward_lstm_20_while_identity_4Hsequential_20/bidirectional_20/forward_lstm_20/while/Identity_4:output:0"�
?sequential_20_bidirectional_20_forward_lstm_20_while_identity_5Hsequential_20/bidirectional_20/forward_lstm_20/while/Identity_5:output:0"�
?sequential_20_bidirectional_20_forward_lstm_20_while_identity_6Hsequential_20/bidirectional_20/forward_lstm_20/while/Identity_6:output:0"�
asequential_20_bidirectional_20_forward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resourcecsequential_20_bidirectional_20_forward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0"�
bsequential_20_bidirectional_20_forward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resourcedsequential_20_bidirectional_20_forward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0"�
`sequential_20_bidirectional_20_forward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resourcebsequential_20_bidirectional_20_forward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0"�
ssequential_20_bidirectional_20_forward_lstm_20_while_sequential_20_bidirectional_20_forward_lstm_20_strided_slice_1usequential_20_bidirectional_20_forward_lstm_20_while_sequential_20_bidirectional_20_forward_lstm_20_strided_slice_1_0"�
�sequential_20_bidirectional_20_forward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_20_bidirectional_20_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor�sequential_20_bidirectional_20_forward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_20_bidirectional_20_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2�
Xsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOpXsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp2�
Wsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOpWsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp2�
Ysequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOpYsequential_20/bidirectional_20/forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp: 
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
2__inference_bidirectional_20_layer_call_fn_2414833
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
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_24133562
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
�?
�
while_body_2413626
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_61_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_61_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_61_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_61_matmul_readvariableop_resource:	�F
3while_lstm_cell_61_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_61_biasadd_readvariableop_resource:	���)while/lstm_cell_61/BiasAdd/ReadVariableOp�(while/lstm_cell_61/MatMul/ReadVariableOp�*while/lstm_cell_61/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_61/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_61_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_61/MatMul/ReadVariableOp�
while/lstm_cell_61/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/MatMul�
*while/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_61_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_61/MatMul_1/ReadVariableOp�
while/lstm_cell_61/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/MatMul_1�
while/lstm_cell_61/addAddV2#while/lstm_cell_61/MatMul:product:0%while/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/add�
)while/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_61_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_61/BiasAdd/ReadVariableOp�
while/lstm_cell_61/BiasAddBiasAddwhile/lstm_cell_61/add:z:01while/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/BiasAdd�
"while/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_61/split/split_dim�
while/lstm_cell_61/splitSplit+while/lstm_cell_61/split/split_dim:output:0#while/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_61/split�
while/lstm_cell_61/SigmoidSigmoid!while/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Sigmoid�
while/lstm_cell_61/Sigmoid_1Sigmoid!while/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Sigmoid_1�
while/lstm_cell_61/mulMul while/lstm_cell_61/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/mul�
while/lstm_cell_61/ReluRelu!while/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Relu�
while/lstm_cell_61/mul_1Mulwhile/lstm_cell_61/Sigmoid:y:0%while/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/mul_1�
while/lstm_cell_61/add_1AddV2while/lstm_cell_61/mul:z:0while/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/add_1�
while/lstm_cell_61/Sigmoid_2Sigmoid!while/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Sigmoid_2�
while/lstm_cell_61/Relu_1Reluwhile/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Relu_1�
while/lstm_cell_61/mul_2Mul while/lstm_cell_61/Sigmoid_2:y:0'while/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_61/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_61/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_61/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_61/BiasAdd/ReadVariableOp)^while/lstm_cell_61/MatMul/ReadVariableOp+^while/lstm_cell_61/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_61_biasadd_readvariableop_resource4while_lstm_cell_61_biasadd_readvariableop_resource_0"l
3while_lstm_cell_61_matmul_1_readvariableop_resource5while_lstm_cell_61_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_61_matmul_readvariableop_resource3while_lstm_cell_61_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_61/BiasAdd/ReadVariableOp)while/lstm_cell_61/BiasAdd/ReadVariableOp2T
(while/lstm_cell_61/MatMul/ReadVariableOp(while/lstm_cell_61/MatMul/ReadVariableOp2X
*while/lstm_cell_61/MatMul_1/ReadVariableOp*while/lstm_cell_61/MatMul_1/ReadVariableOp: 
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
while_cond_2413625
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2413625___redundant_placeholder05
1while_while_cond_2413625___redundant_placeholder15
1while_while_cond_2413625___redundant_placeholder25
1while_while_cond_2413625___redundant_placeholder3
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
�
"forward_lstm_20_while_cond_2415254<
8forward_lstm_20_while_forward_lstm_20_while_loop_counterB
>forward_lstm_20_while_forward_lstm_20_while_maximum_iterations%
!forward_lstm_20_while_placeholder'
#forward_lstm_20_while_placeholder_1'
#forward_lstm_20_while_placeholder_2'
#forward_lstm_20_while_placeholder_3>
:forward_lstm_20_while_less_forward_lstm_20_strided_slice_1U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2415254___redundant_placeholder0U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2415254___redundant_placeholder1U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2415254___redundant_placeholder2U
Qforward_lstm_20_while_forward_lstm_20_while_cond_2415254___redundant_placeholder3"
forward_lstm_20_while_identity
�
forward_lstm_20/while/LessLess!forward_lstm_20_while_placeholder:forward_lstm_20_while_less_forward_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_20/while/Less�
forward_lstm_20/while/IdentityIdentityforward_lstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_20/while/Identity"I
forward_lstm_20_while_identity'forward_lstm_20/while/Identity:output:0*(
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
�d
�
#backward_lstm_20_while_body_2414539>
:backward_lstm_20_while_backward_lstm_20_while_loop_counterD
@backward_lstm_20_while_backward_lstm_20_while_maximum_iterations&
"backward_lstm_20_while_placeholder(
$backward_lstm_20_while_placeholder_1(
$backward_lstm_20_while_placeholder_2(
$backward_lstm_20_while_placeholder_3(
$backward_lstm_20_while_placeholder_4=
9backward_lstm_20_while_backward_lstm_20_strided_slice_1_0y
ubackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_08
4backward_lstm_20_while_less_backward_lstm_20_sub_1_0W
Dbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0:	�Y
Fbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0:	2�T
Ebackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0:	�#
backward_lstm_20_while_identity%
!backward_lstm_20_while_identity_1%
!backward_lstm_20_while_identity_2%
!backward_lstm_20_while_identity_3%
!backward_lstm_20_while_identity_4%
!backward_lstm_20_while_identity_5%
!backward_lstm_20_while_identity_6;
7backward_lstm_20_while_backward_lstm_20_strided_slice_1w
sbackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor6
2backward_lstm_20_while_less_backward_lstm_20_sub_1U
Bbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource:	�W
Dbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource:	2�R
Cbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource:	���:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp�9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp�;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp�
Hbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2J
Hbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape�
:backward_lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_20_while_placeholderQbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02<
:backward_lstm_20/while/TensorArrayV2Read/TensorListGetItem�
backward_lstm_20/while/LessLess4backward_lstm_20_while_less_backward_lstm_20_sub_1_0"backward_lstm_20_while_placeholder*
T0*#
_output_shapes
:���������2
backward_lstm_20/while/Less�
9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOpReadVariableOpDbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02;
9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp�
*backward_lstm_20/while/lstm_cell_62/MatMulMatMulAbackward_lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:0Abackward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*backward_lstm_20/while/lstm_cell_62/MatMul�
;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOpFbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02=
;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp�
,backward_lstm_20/while/lstm_cell_62/MatMul_1MatMul$backward_lstm_20_while_placeholder_3Cbackward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2.
,backward_lstm_20/while/lstm_cell_62/MatMul_1�
'backward_lstm_20/while/lstm_cell_62/addAddV24backward_lstm_20/while/lstm_cell_62/MatMul:product:06backward_lstm_20/while/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2)
'backward_lstm_20/while/lstm_cell_62/add�
:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOpEbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02<
:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp�
+backward_lstm_20/while/lstm_cell_62/BiasAddBiasAdd+backward_lstm_20/while/lstm_cell_62/add:z:0Bbackward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+backward_lstm_20/while/lstm_cell_62/BiasAdd�
3backward_lstm_20/while/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3backward_lstm_20/while/lstm_cell_62/split/split_dim�
)backward_lstm_20/while/lstm_cell_62/splitSplit<backward_lstm_20/while/lstm_cell_62/split/split_dim:output:04backward_lstm_20/while/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2+
)backward_lstm_20/while/lstm_cell_62/split�
+backward_lstm_20/while/lstm_cell_62/SigmoidSigmoid2backward_lstm_20/while/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22-
+backward_lstm_20/while/lstm_cell_62/Sigmoid�
-backward_lstm_20/while/lstm_cell_62/Sigmoid_1Sigmoid2backward_lstm_20/while/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22/
-backward_lstm_20/while/lstm_cell_62/Sigmoid_1�
'backward_lstm_20/while/lstm_cell_62/mulMul1backward_lstm_20/while/lstm_cell_62/Sigmoid_1:y:0$backward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22)
'backward_lstm_20/while/lstm_cell_62/mul�
(backward_lstm_20/while/lstm_cell_62/ReluRelu2backward_lstm_20/while/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22*
(backward_lstm_20/while/lstm_cell_62/Relu�
)backward_lstm_20/while/lstm_cell_62/mul_1Mul/backward_lstm_20/while/lstm_cell_62/Sigmoid:y:06backward_lstm_20/while/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_20/while/lstm_cell_62/mul_1�
)backward_lstm_20/while/lstm_cell_62/add_1AddV2+backward_lstm_20/while/lstm_cell_62/mul:z:0-backward_lstm_20/while/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_20/while/lstm_cell_62/add_1�
-backward_lstm_20/while/lstm_cell_62/Sigmoid_2Sigmoid2backward_lstm_20/while/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22/
-backward_lstm_20/while/lstm_cell_62/Sigmoid_2�
*backward_lstm_20/while/lstm_cell_62/Relu_1Relu-backward_lstm_20/while/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22,
*backward_lstm_20/while/lstm_cell_62/Relu_1�
)backward_lstm_20/while/lstm_cell_62/mul_2Mul1backward_lstm_20/while/lstm_cell_62/Sigmoid_2:y:08backward_lstm_20/while/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_20/while/lstm_cell_62/mul_2�
backward_lstm_20/while/SelectSelectbackward_lstm_20/while/Less:z:0-backward_lstm_20/while/lstm_cell_62/mul_2:z:0$backward_lstm_20_while_placeholder_2*
T0*'
_output_shapes
:���������22
backward_lstm_20/while/Select�
backward_lstm_20/while/Select_1Selectbackward_lstm_20/while/Less:z:0-backward_lstm_20/while/lstm_cell_62/mul_2:z:0$backward_lstm_20_while_placeholder_3*
T0*'
_output_shapes
:���������22!
backward_lstm_20/while/Select_1�
backward_lstm_20/while/Select_2Selectbackward_lstm_20/while/Less:z:0-backward_lstm_20/while/lstm_cell_62/add_1:z:0$backward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22!
backward_lstm_20/while/Select_2�
;backward_lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_20_while_placeholder_1"backward_lstm_20_while_placeholder&backward_lstm_20/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_20/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_20/while/add/y�
backward_lstm_20/while/addAddV2"backward_lstm_20_while_placeholder%backward_lstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/while/add�
backward_lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_20/while/add_1/y�
backward_lstm_20/while/add_1AddV2:backward_lstm_20_while_backward_lstm_20_while_loop_counter'backward_lstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/while/add_1�
backward_lstm_20/while/IdentityIdentity backward_lstm_20/while/add_1:z:0^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_20/while/Identity�
!backward_lstm_20/while/Identity_1Identity@backward_lstm_20_while_backward_lstm_20_while_maximum_iterations^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_20/while/Identity_1�
!backward_lstm_20/while/Identity_2Identitybackward_lstm_20/while/add:z:0^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_20/while/Identity_2�
!backward_lstm_20/while/Identity_3IdentityKbackward_lstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_20/while/Identity_3�
!backward_lstm_20/while/Identity_4Identity&backward_lstm_20/while/Select:output:0^backward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/while/Identity_4�
!backward_lstm_20/while/Identity_5Identity(backward_lstm_20/while/Select_1:output:0^backward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/while/Identity_5�
!backward_lstm_20/while/Identity_6Identity(backward_lstm_20/while/Select_2:output:0^backward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/while/Identity_6�
backward_lstm_20/while/NoOpNoOp;^backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp:^backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp<^backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_20/while/NoOp"t
7backward_lstm_20_while_backward_lstm_20_strided_slice_19backward_lstm_20_while_backward_lstm_20_strided_slice_1_0"K
backward_lstm_20_while_identity(backward_lstm_20/while/Identity:output:0"O
!backward_lstm_20_while_identity_1*backward_lstm_20/while/Identity_1:output:0"O
!backward_lstm_20_while_identity_2*backward_lstm_20/while/Identity_2:output:0"O
!backward_lstm_20_while_identity_3*backward_lstm_20/while/Identity_3:output:0"O
!backward_lstm_20_while_identity_4*backward_lstm_20/while/Identity_4:output:0"O
!backward_lstm_20_while_identity_5*backward_lstm_20/while/Identity_5:output:0"O
!backward_lstm_20_while_identity_6*backward_lstm_20/while/Identity_6:output:0"j
2backward_lstm_20_while_less_backward_lstm_20_sub_14backward_lstm_20_while_less_backward_lstm_20_sub_1_0"�
Cbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resourceEbackward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0"�
Dbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resourceFbackward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0"�
Bbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resourceDbackward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0"�
sbackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensorubackward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2x
:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp:backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp2v
9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp9backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp2z
;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp;backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp: 
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
�
�
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_2413758

inputs*
forward_lstm_20_2413741:	�*
forward_lstm_20_2413743:	2�&
forward_lstm_20_2413745:	�+
backward_lstm_20_2413748:	�+
backward_lstm_20_2413750:	2�'
backward_lstm_20_2413752:	�
identity��(backward_lstm_20/StatefulPartitionedCall�'forward_lstm_20/StatefulPartitionedCall�
'forward_lstm_20/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_20_2413741forward_lstm_20_2413743forward_lstm_20_2413745*
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
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_24137102)
'forward_lstm_20/StatefulPartitionedCall�
(backward_lstm_20/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_20_2413748backward_lstm_20_2413750backward_lstm_20_2413752*
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
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_24135372*
(backward_lstm_20/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV20forward_lstm_20/StatefulPartitionedCall:output:01backward_lstm_20/StatefulPartitionedCall:output:0concat/axis:output:0*
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
NoOpNoOp)^backward_lstm_20/StatefulPartitionedCall(^forward_lstm_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 2T
(backward_lstm_20/StatefulPartitionedCall(backward_lstm_20/StatefulPartitionedCall2R
'forward_lstm_20/StatefulPartitionedCall'forward_lstm_20/StatefulPartitionedCall:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�F
�
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_2411918

inputs'
lstm_cell_61_2411836:	�'
lstm_cell_61_2411838:	2�#
lstm_cell_61_2411840:	�
identity��$lstm_cell_61/StatefulPartitionedCall�whileD
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
$lstm_cell_61/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_61_2411836lstm_cell_61_2411838lstm_cell_61_2411840*
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
I__inference_lstm_cell_61_layer_call_and_return_conditional_losses_24118352&
$lstm_cell_61/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_61_2411836lstm_cell_61_2411838lstm_cell_61_2411840*
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
while_body_2411849*
condR
while_cond_2411848*K
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
NoOpNoOp%^lstm_cell_61/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_61/StatefulPartitionedCall$lstm_cell_61/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�^
�
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_2417377

inputs>
+lstm_cell_62_matmul_readvariableop_resource:	�@
-lstm_cell_62_matmul_1_readvariableop_resource:	2�;
,lstm_cell_62_biasadd_readvariableop_resource:	�
identity��#lstm_cell_62/BiasAdd/ReadVariableOp�"lstm_cell_62/MatMul/ReadVariableOp�$lstm_cell_62/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_62/MatMul/ReadVariableOpReadVariableOp+lstm_cell_62_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_62/MatMul/ReadVariableOp�
lstm_cell_62/MatMulMatMulstrided_slice_2:output:0*lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/MatMul�
$lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_62_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_62/MatMul_1/ReadVariableOp�
lstm_cell_62/MatMul_1MatMulzeros:output:0,lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/MatMul_1�
lstm_cell_62/addAddV2lstm_cell_62/MatMul:product:0lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/add�
#lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_62/BiasAdd/ReadVariableOp�
lstm_cell_62/BiasAddBiasAddlstm_cell_62/add:z:0+lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/BiasAdd~
lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_62/split/split_dim�
lstm_cell_62/splitSplit%lstm_cell_62/split/split_dim:output:0lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_62/split�
lstm_cell_62/SigmoidSigmoidlstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/Sigmoid�
lstm_cell_62/Sigmoid_1Sigmoidlstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_62/Sigmoid_1�
lstm_cell_62/mulMullstm_cell_62/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/mul}
lstm_cell_62/ReluRelulstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_62/Relu�
lstm_cell_62/mul_1Mullstm_cell_62/Sigmoid:y:0lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/mul_1�
lstm_cell_62/add_1AddV2lstm_cell_62/mul:z:0lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/add_1�
lstm_cell_62/Sigmoid_2Sigmoidlstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_62/Sigmoid_2|
lstm_cell_62/Relu_1Relulstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/Relu_1�
lstm_cell_62/mul_2Mullstm_cell_62/Sigmoid_2:y:0!lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_62_matmul_readvariableop_resource-lstm_cell_62_matmul_1_readvariableop_resource,lstm_cell_62_biasadd_readvariableop_resource*
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
while_body_2417293*
condR
while_cond_2417292*K
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
NoOpNoOp$^lstm_cell_62/BiasAdd/ReadVariableOp#^lstm_cell_62/MatMul/ReadVariableOp%^lstm_cell_62/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_62/BiasAdd/ReadVariableOp#lstm_cell_62/BiasAdd/ReadVariableOp2H
"lstm_cell_62/MatMul/ReadVariableOp"lstm_cell_62/MatMul/ReadVariableOp2L
$lstm_cell_62/MatMul_1/ReadVariableOp$lstm_cell_62/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
while_cond_2411848
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2411848___redundant_placeholder05
1while_while_cond_2411848___redundant_placeholder15
1while_while_cond_2411848___redundant_placeholder25
1while_while_cond_2411848___redundant_placeholder3
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
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_2414636

inputs
inputs_1	N
;forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource:	�P
=forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource:	2�K
<forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource:	�O
<backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource:	�Q
>backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource:	2�L
=backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource:	�
identity��4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp�3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp�5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp�backward_lstm_20/while�3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp�2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp�4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp�forward_lstm_20/while�
$forward_lstm_20/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$forward_lstm_20/RaggedToTensor/zeros�
$forward_lstm_20/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2&
$forward_lstm_20/RaggedToTensor/Const�
3forward_lstm_20/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-forward_lstm_20/RaggedToTensor/Const:output:0inputs-forward_lstm_20/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3forward_lstm_20/RaggedToTensor/RaggedTensorToTensor�
:forward_lstm_20/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack�
<forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1�
<forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<forward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2�
4forward_lstm_20/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cforward_lstm_20/RaggedNestedRowLengths/strided_slice/stack:output:0Eforward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1:output:0Eforward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask26
4forward_lstm_20/RaggedNestedRowLengths/strided_slice�
<forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack�
>forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2@
>forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1�
>forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>forward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2�
6forward_lstm_20/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Eforward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gforward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gforward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask28
6forward_lstm_20/RaggedNestedRowLengths/strided_slice_1�
*forward_lstm_20/RaggedNestedRowLengths/subSub=forward_lstm_20/RaggedNestedRowLengths/strided_slice:output:0?forward_lstm_20/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2,
*forward_lstm_20/RaggedNestedRowLengths/sub�
forward_lstm_20/CastCast.forward_lstm_20/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
forward_lstm_20/Cast�
forward_lstm_20/ShapeShape<forward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_20/Shape�
#forward_lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_20/strided_slice/stack�
%forward_lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_20/strided_slice/stack_1�
%forward_lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_20/strided_slice/stack_2�
forward_lstm_20/strided_sliceStridedSliceforward_lstm_20/Shape:output:0,forward_lstm_20/strided_slice/stack:output:0.forward_lstm_20/strided_slice/stack_1:output:0.forward_lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_20/strided_slice|
forward_lstm_20/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_20/zeros/mul/y�
forward_lstm_20/zeros/mulMul&forward_lstm_20/strided_slice:output:0$forward_lstm_20/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros/mul
forward_lstm_20/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_20/zeros/Less/y�
forward_lstm_20/zeros/LessLessforward_lstm_20/zeros/mul:z:0%forward_lstm_20/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros/Less�
forward_lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_20/zeros/packed/1�
forward_lstm_20/zeros/packedPack&forward_lstm_20/strided_slice:output:0'forward_lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_20/zeros/packed�
forward_lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_20/zeros/Const�
forward_lstm_20/zerosFill%forward_lstm_20/zeros/packed:output:0$forward_lstm_20/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_20/zeros�
forward_lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_20/zeros_1/mul/y�
forward_lstm_20/zeros_1/mulMul&forward_lstm_20/strided_slice:output:0&forward_lstm_20/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros_1/mul�
forward_lstm_20/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
forward_lstm_20/zeros_1/Less/y�
forward_lstm_20/zeros_1/LessLessforward_lstm_20/zeros_1/mul:z:0'forward_lstm_20/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros_1/Less�
 forward_lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_20/zeros_1/packed/1�
forward_lstm_20/zeros_1/packedPack&forward_lstm_20/strided_slice:output:0)forward_lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_20/zeros_1/packed�
forward_lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_20/zeros_1/Const�
forward_lstm_20/zeros_1Fill'forward_lstm_20/zeros_1/packed:output:0&forward_lstm_20/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_20/zeros_1�
forward_lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_20/transpose/perm�
forward_lstm_20/transpose	Transpose<forward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0'forward_lstm_20/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
forward_lstm_20/transpose
forward_lstm_20/Shape_1Shapeforward_lstm_20/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_20/Shape_1�
%forward_lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_20/strided_slice_1/stack�
'forward_lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_1/stack_1�
'forward_lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_1/stack_2�
forward_lstm_20/strided_slice_1StridedSlice forward_lstm_20/Shape_1:output:0.forward_lstm_20/strided_slice_1/stack:output:00forward_lstm_20/strided_slice_1/stack_1:output:00forward_lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_20/strided_slice_1�
+forward_lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+forward_lstm_20/TensorArrayV2/element_shape�
forward_lstm_20/TensorArrayV2TensorListReserve4forward_lstm_20/TensorArrayV2/element_shape:output:0(forward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_20/TensorArrayV2�
Eforward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2G
Eforward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape�
7forward_lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_20/transpose:y:0Nforward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_20/TensorArrayUnstack/TensorListFromTensor�
%forward_lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_20/strided_slice_2/stack�
'forward_lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_2/stack_1�
'forward_lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_2/stack_2�
forward_lstm_20/strided_slice_2StridedSliceforward_lstm_20/transpose:y:0.forward_lstm_20/strided_slice_2/stack:output:00forward_lstm_20/strided_slice_2/stack_1:output:00forward_lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2!
forward_lstm_20/strided_slice_2�
2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOpReadVariableOp;forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp�
#forward_lstm_20/lstm_cell_61/MatMulMatMul(forward_lstm_20/strided_slice_2:output:0:forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#forward_lstm_20/lstm_cell_61/MatMul�
4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOp=forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype026
4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp�
%forward_lstm_20/lstm_cell_61/MatMul_1MatMulforward_lstm_20/zeros:output:0<forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%forward_lstm_20/lstm_cell_61/MatMul_1�
 forward_lstm_20/lstm_cell_61/addAddV2-forward_lstm_20/lstm_cell_61/MatMul:product:0/forward_lstm_20/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2"
 forward_lstm_20/lstm_cell_61/add�
3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOp<forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp�
$forward_lstm_20/lstm_cell_61/BiasAddBiasAdd$forward_lstm_20/lstm_cell_61/add:z:0;forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$forward_lstm_20/lstm_cell_61/BiasAdd�
,forward_lstm_20/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,forward_lstm_20/lstm_cell_61/split/split_dim�
"forward_lstm_20/lstm_cell_61/splitSplit5forward_lstm_20/lstm_cell_61/split/split_dim:output:0-forward_lstm_20/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2$
"forward_lstm_20/lstm_cell_61/split�
$forward_lstm_20/lstm_cell_61/SigmoidSigmoid+forward_lstm_20/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22&
$forward_lstm_20/lstm_cell_61/Sigmoid�
&forward_lstm_20/lstm_cell_61/Sigmoid_1Sigmoid+forward_lstm_20/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22(
&forward_lstm_20/lstm_cell_61/Sigmoid_1�
 forward_lstm_20/lstm_cell_61/mulMul*forward_lstm_20/lstm_cell_61/Sigmoid_1:y:0 forward_lstm_20/zeros_1:output:0*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/lstm_cell_61/mul�
!forward_lstm_20/lstm_cell_61/ReluRelu+forward_lstm_20/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22#
!forward_lstm_20/lstm_cell_61/Relu�
"forward_lstm_20/lstm_cell_61/mul_1Mul(forward_lstm_20/lstm_cell_61/Sigmoid:y:0/forward_lstm_20/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_20/lstm_cell_61/mul_1�
"forward_lstm_20/lstm_cell_61/add_1AddV2$forward_lstm_20/lstm_cell_61/mul:z:0&forward_lstm_20/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_20/lstm_cell_61/add_1�
&forward_lstm_20/lstm_cell_61/Sigmoid_2Sigmoid+forward_lstm_20/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22(
&forward_lstm_20/lstm_cell_61/Sigmoid_2�
#forward_lstm_20/lstm_cell_61/Relu_1Relu&forward_lstm_20/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22%
#forward_lstm_20/lstm_cell_61/Relu_1�
"forward_lstm_20/lstm_cell_61/mul_2Mul*forward_lstm_20/lstm_cell_61/Sigmoid_2:y:01forward_lstm_20/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_20/lstm_cell_61/mul_2�
-forward_lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2/
-forward_lstm_20/TensorArrayV2_1/element_shape�
forward_lstm_20/TensorArrayV2_1TensorListReserve6forward_lstm_20/TensorArrayV2_1/element_shape:output:0(forward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_20/TensorArrayV2_1n
forward_lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_20/time�
forward_lstm_20/zeros_like	ZerosLike&forward_lstm_20/lstm_cell_61/mul_2:z:0*
T0*'
_output_shapes
:���������22
forward_lstm_20/zeros_like�
(forward_lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(forward_lstm_20/while/maximum_iterations�
"forward_lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_20/while/loop_counter�
forward_lstm_20/whileWhile+forward_lstm_20/while/loop_counter:output:01forward_lstm_20/while/maximum_iterations:output:0forward_lstm_20/time:output:0(forward_lstm_20/TensorArrayV2_1:handle:0forward_lstm_20/zeros_like:y:0forward_lstm_20/zeros:output:0 forward_lstm_20/zeros_1:output:0(forward_lstm_20/strided_slice_1:output:0Gforward_lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_20/Cast:y:0;forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource=forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource<forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource*
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
"forward_lstm_20_while_body_2414360*.
cond&R$
"forward_lstm_20_while_cond_2414359*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
forward_lstm_20/while�
@forward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2B
@forward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape�
2forward_lstm_20/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_20/while:output:3Iforward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype024
2forward_lstm_20/TensorArrayV2Stack/TensorListStack�
%forward_lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2'
%forward_lstm_20/strided_slice_3/stack�
'forward_lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_20/strided_slice_3/stack_1�
'forward_lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_3/stack_2�
forward_lstm_20/strided_slice_3StridedSlice;forward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_20/strided_slice_3/stack:output:00forward_lstm_20/strided_slice_3/stack_1:output:00forward_lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2!
forward_lstm_20/strided_slice_3�
 forward_lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_20/transpose_1/perm�
forward_lstm_20/transpose_1	Transpose;forward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_20/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
forward_lstm_20/transpose_1�
forward_lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_20/runtime�
%backward_lstm_20/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%backward_lstm_20/RaggedToTensor/zeros�
%backward_lstm_20/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2'
%backward_lstm_20/RaggedToTensor/Const�
4backward_lstm_20/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.backward_lstm_20/RaggedToTensor/Const:output:0inputs.backward_lstm_20/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4backward_lstm_20/RaggedToTensor/RaggedTensorToTensor�
;backward_lstm_20/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack�
=backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1�
=backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=backward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2�
5backward_lstm_20/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dbackward_lstm_20/RaggedNestedRowLengths/strided_slice/stack:output:0Fbackward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fbackward_lstm_20/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask27
5backward_lstm_20/RaggedNestedRowLengths/strided_slice�
=backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack�
?backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2A
?backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1�
?backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?backward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2�
7backward_lstm_20/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fbackward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hbackward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hbackward_lstm_20/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask29
7backward_lstm_20/RaggedNestedRowLengths/strided_slice_1�
+backward_lstm_20/RaggedNestedRowLengths/subSub>backward_lstm_20/RaggedNestedRowLengths/strided_slice:output:0@backward_lstm_20/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2-
+backward_lstm_20/RaggedNestedRowLengths/sub�
backward_lstm_20/CastCast/backward_lstm_20/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
backward_lstm_20/Cast�
backward_lstm_20/ShapeShape=backward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_20/Shape�
$backward_lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_20/strided_slice/stack�
&backward_lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_20/strided_slice/stack_1�
&backward_lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_20/strided_slice/stack_2�
backward_lstm_20/strided_sliceStridedSlicebackward_lstm_20/Shape:output:0-backward_lstm_20/strided_slice/stack:output:0/backward_lstm_20/strided_slice/stack_1:output:0/backward_lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_20/strided_slice~
backward_lstm_20/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_20/zeros/mul/y�
backward_lstm_20/zeros/mulMul'backward_lstm_20/strided_slice:output:0%backward_lstm_20/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros/mul�
backward_lstm_20/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
backward_lstm_20/zeros/Less/y�
backward_lstm_20/zeros/LessLessbackward_lstm_20/zeros/mul:z:0&backward_lstm_20/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros/Less�
backward_lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_20/zeros/packed/1�
backward_lstm_20/zeros/packedPack'backward_lstm_20/strided_slice:output:0(backward_lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_20/zeros/packed�
backward_lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_20/zeros/Const�
backward_lstm_20/zerosFill&backward_lstm_20/zeros/packed:output:0%backward_lstm_20/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_20/zeros�
backward_lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_20/zeros_1/mul/y�
backward_lstm_20/zeros_1/mulMul'backward_lstm_20/strided_slice:output:0'backward_lstm_20/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros_1/mul�
backward_lstm_20/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2!
backward_lstm_20/zeros_1/Less/y�
backward_lstm_20/zeros_1/LessLess backward_lstm_20/zeros_1/mul:z:0(backward_lstm_20/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros_1/Less�
!backward_lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_20/zeros_1/packed/1�
backward_lstm_20/zeros_1/packedPack'backward_lstm_20/strided_slice:output:0*backward_lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_20/zeros_1/packed�
backward_lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_20/zeros_1/Const�
backward_lstm_20/zeros_1Fill(backward_lstm_20/zeros_1/packed:output:0'backward_lstm_20/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_20/zeros_1�
backward_lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_20/transpose/perm�
backward_lstm_20/transpose	Transpose=backward_lstm_20/RaggedToTensor/RaggedTensorToTensor:result:0(backward_lstm_20/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_20/transpose�
backward_lstm_20/Shape_1Shapebackward_lstm_20/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_20/Shape_1�
&backward_lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_20/strided_slice_1/stack�
(backward_lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_1/stack_1�
(backward_lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_1/stack_2�
 backward_lstm_20/strided_slice_1StridedSlice!backward_lstm_20/Shape_1:output:0/backward_lstm_20/strided_slice_1/stack:output:01backward_lstm_20/strided_slice_1/stack_1:output:01backward_lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_20/strided_slice_1�
,backward_lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2.
,backward_lstm_20/TensorArrayV2/element_shape�
backward_lstm_20/TensorArrayV2TensorListReserve5backward_lstm_20/TensorArrayV2/element_shape:output:0)backward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_20/TensorArrayV2�
backward_lstm_20/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_20/ReverseV2/axis�
backward_lstm_20/ReverseV2	ReverseV2backward_lstm_20/transpose:y:0(backward_lstm_20/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_20/ReverseV2�
Fbackward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2H
Fbackward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape�
8backward_lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_20/ReverseV2:output:0Obackward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_20/TensorArrayUnstack/TensorListFromTensor�
&backward_lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_20/strided_slice_2/stack�
(backward_lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_2/stack_1�
(backward_lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_2/stack_2�
 backward_lstm_20/strided_slice_2StridedSlicebackward_lstm_20/transpose:y:0/backward_lstm_20/strided_slice_2/stack:output:01backward_lstm_20/strided_slice_2/stack_1:output:01backward_lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2"
 backward_lstm_20/strided_slice_2�
3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOpReadVariableOp<backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype025
3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp�
$backward_lstm_20/lstm_cell_62/MatMulMatMul)backward_lstm_20/strided_slice_2:output:0;backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$backward_lstm_20/lstm_cell_62/MatMul�
5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOp>backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype027
5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp�
&backward_lstm_20/lstm_cell_62/MatMul_1MatMulbackward_lstm_20/zeros:output:0=backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&backward_lstm_20/lstm_cell_62/MatMul_1�
!backward_lstm_20/lstm_cell_62/addAddV2.backward_lstm_20/lstm_cell_62/MatMul:product:00backward_lstm_20/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2#
!backward_lstm_20/lstm_cell_62/add�
4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOp=backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype026
4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp�
%backward_lstm_20/lstm_cell_62/BiasAddBiasAdd%backward_lstm_20/lstm_cell_62/add:z:0<backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%backward_lstm_20/lstm_cell_62/BiasAdd�
-backward_lstm_20/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-backward_lstm_20/lstm_cell_62/split/split_dim�
#backward_lstm_20/lstm_cell_62/splitSplit6backward_lstm_20/lstm_cell_62/split/split_dim:output:0.backward_lstm_20/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2%
#backward_lstm_20/lstm_cell_62/split�
%backward_lstm_20/lstm_cell_62/SigmoidSigmoid,backward_lstm_20/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22'
%backward_lstm_20/lstm_cell_62/Sigmoid�
'backward_lstm_20/lstm_cell_62/Sigmoid_1Sigmoid,backward_lstm_20/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22)
'backward_lstm_20/lstm_cell_62/Sigmoid_1�
!backward_lstm_20/lstm_cell_62/mulMul+backward_lstm_20/lstm_cell_62/Sigmoid_1:y:0!backward_lstm_20/zeros_1:output:0*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/lstm_cell_62/mul�
"backward_lstm_20/lstm_cell_62/ReluRelu,backward_lstm_20/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22$
"backward_lstm_20/lstm_cell_62/Relu�
#backward_lstm_20/lstm_cell_62/mul_1Mul)backward_lstm_20/lstm_cell_62/Sigmoid:y:00backward_lstm_20/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_20/lstm_cell_62/mul_1�
#backward_lstm_20/lstm_cell_62/add_1AddV2%backward_lstm_20/lstm_cell_62/mul:z:0'backward_lstm_20/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_20/lstm_cell_62/add_1�
'backward_lstm_20/lstm_cell_62/Sigmoid_2Sigmoid,backward_lstm_20/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22)
'backward_lstm_20/lstm_cell_62/Sigmoid_2�
$backward_lstm_20/lstm_cell_62/Relu_1Relu'backward_lstm_20/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22&
$backward_lstm_20/lstm_cell_62/Relu_1�
#backward_lstm_20/lstm_cell_62/mul_2Mul+backward_lstm_20/lstm_cell_62/Sigmoid_2:y:02backward_lstm_20/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_20/lstm_cell_62/mul_2�
.backward_lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   20
.backward_lstm_20/TensorArrayV2_1/element_shape�
 backward_lstm_20/TensorArrayV2_1TensorListReserve7backward_lstm_20/TensorArrayV2_1/element_shape:output:0)backward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_20/TensorArrayV2_1p
backward_lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_20/time�
&backward_lstm_20/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_lstm_20/Max/reduction_indices�
backward_lstm_20/MaxMaxbackward_lstm_20/Cast:y:0/backward_lstm_20/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/Maxr
backward_lstm_20/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_20/sub/y�
backward_lstm_20/subSubbackward_lstm_20/Max:output:0backward_lstm_20/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/sub�
backward_lstm_20/Sub_1Subbackward_lstm_20/sub:z:0backward_lstm_20/Cast:y:0*
T0*#
_output_shapes
:���������2
backward_lstm_20/Sub_1�
backward_lstm_20/zeros_like	ZerosLike'backward_lstm_20/lstm_cell_62/mul_2:z:0*
T0*'
_output_shapes
:���������22
backward_lstm_20/zeros_like�
)backward_lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2+
)backward_lstm_20/while/maximum_iterations�
#backward_lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_20/while/loop_counter�	
backward_lstm_20/whileWhile,backward_lstm_20/while/loop_counter:output:02backward_lstm_20/while/maximum_iterations:output:0backward_lstm_20/time:output:0)backward_lstm_20/TensorArrayV2_1:handle:0backward_lstm_20/zeros_like:y:0backward_lstm_20/zeros:output:0!backward_lstm_20/zeros_1:output:0)backward_lstm_20/strided_slice_1:output:0Hbackward_lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_20/Sub_1:z:0<backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource>backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource=backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource*
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
#backward_lstm_20_while_body_2414539*/
cond'R%
#backward_lstm_20_while_cond_2414538*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
backward_lstm_20/while�
Abackward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2C
Abackward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape�
3backward_lstm_20/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_20/while:output:3Jbackward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype025
3backward_lstm_20/TensorArrayV2Stack/TensorListStack�
&backward_lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2(
&backward_lstm_20/strided_slice_3/stack�
(backward_lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_20/strided_slice_3/stack_1�
(backward_lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_3/stack_2�
 backward_lstm_20/strided_slice_3StridedSlice<backward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_20/strided_slice_3/stack:output:01backward_lstm_20/strided_slice_3/stack_1:output:01backward_lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2"
 backward_lstm_20/strided_slice_3�
!backward_lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_20/transpose_1/perm�
backward_lstm_20/transpose_1	Transpose<backward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_20/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
backward_lstm_20/transpose_1�
backward_lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_20/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2(forward_lstm_20/strided_slice_3:output:0)backward_lstm_20/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp5^backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp4^backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp6^backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp^backward_lstm_20/while4^forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp3^forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp5^forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp^forward_lstm_20/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������: : : : : : 2l
4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp2j
3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp2n
5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp20
backward_lstm_20/whilebackward_lstm_20/while2j
3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp2h
2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp2l
4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp2.
forward_lstm_20/whileforward_lstm_20/while:O K
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
I__inference_lstm_cell_62_layer_call_and_return_conditional_losses_2417694

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
�
�
J__inference_sequential_20_layer_call_and_return_conditional_losses_2414786

inputs
inputs_1	+
bidirectional_20_2414767:	�+
bidirectional_20_2414769:	2�'
bidirectional_20_2414771:	�+
bidirectional_20_2414773:	�+
bidirectional_20_2414775:	2�'
bidirectional_20_2414777:	�"
dense_20_2414780:d
dense_20_2414782:
identity��(bidirectional_20/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�
(bidirectional_20/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_20_2414767bidirectional_20_2414769bidirectional_20_2414771bidirectional_20_2414773bidirectional_20_2414775bidirectional_20_2414777*
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
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_24146362*
(bidirectional_20/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_20/StatefulPartitionedCall:output:0dense_20_2414780dense_20_2414782*
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
E__inference_dense_20_layer_call_and_return_conditional_losses_24142212"
 dense_20/StatefulPartitionedCall�
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp)^bidirectional_20/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:���������:���������: : : : : : : : 2T
(bidirectional_20/StatefulPartitionedCall(bidirectional_20/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall:O K
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
#backward_lstm_20_while_cond_2415403>
:backward_lstm_20_while_backward_lstm_20_while_loop_counterD
@backward_lstm_20_while_backward_lstm_20_while_maximum_iterations&
"backward_lstm_20_while_placeholder(
$backward_lstm_20_while_placeholder_1(
$backward_lstm_20_while_placeholder_2(
$backward_lstm_20_while_placeholder_3@
<backward_lstm_20_while_less_backward_lstm_20_strided_slice_1W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2415403___redundant_placeholder0W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2415403___redundant_placeholder1W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2415403___redundant_placeholder2W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2415403___redundant_placeholder3#
backward_lstm_20_while_identity
�
backward_lstm_20/while/LessLess"backward_lstm_20_while_placeholder<backward_lstm_20_while_less_backward_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_20/while/Less�
backward_lstm_20/while/IdentityIdentitybackward_lstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_20/while/Identity"K
backward_lstm_20_while_identity(backward_lstm_20/while/Identity:output:0*(
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
2__inference_backward_lstm_20_layer_call_fn_2416907

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
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_24133452
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
#backward_lstm_20_while_cond_2414098>
:backward_lstm_20_while_backward_lstm_20_while_loop_counterD
@backward_lstm_20_while_backward_lstm_20_while_maximum_iterations&
"backward_lstm_20_while_placeholder(
$backward_lstm_20_while_placeholder_1(
$backward_lstm_20_while_placeholder_2(
$backward_lstm_20_while_placeholder_3(
$backward_lstm_20_while_placeholder_4@
<backward_lstm_20_while_less_backward_lstm_20_strided_slice_1W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2414098___redundant_placeholder0W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2414098___redundant_placeholder1W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2414098___redundant_placeholder2W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2414098___redundant_placeholder3W
Sbackward_lstm_20_while_backward_lstm_20_while_cond_2414098___redundant_placeholder4#
backward_lstm_20_while_identity
�
backward_lstm_20/while/LessLess"backward_lstm_20_while_placeholder<backward_lstm_20_while_less_backward_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_20/while/Less�
backward_lstm_20/while/IdentityIdentitybackward_lstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_20/while/Identity"K
backward_lstm_20_while_identity(backward_lstm_20/while/Identity:output:0*(
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
�%
�
while_body_2411849
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_61_2411873_0:	�/
while_lstm_cell_61_2411875_0:	2�+
while_lstm_cell_61_2411877_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_61_2411873:	�-
while_lstm_cell_61_2411875:	2�)
while_lstm_cell_61_2411877:	���*while/lstm_cell_61/StatefulPartitionedCall�
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
*while/lstm_cell_61/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_61_2411873_0while_lstm_cell_61_2411875_0while_lstm_cell_61_2411877_0*
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
I__inference_lstm_cell_61_layer_call_and_return_conditional_losses_24118352,
*while/lstm_cell_61/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_61/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_61/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_61/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_61/StatefulPartitionedCall*"
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
while_lstm_cell_61_2411873while_lstm_cell_61_2411873_0":
while_lstm_cell_61_2411875while_lstm_cell_61_2411875_0":
while_lstm_cell_61_2411877while_lstm_cell_61_2411877_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2X
*while/lstm_cell_61/StatefulPartitionedCall*while/lstm_cell_61/StatefulPartitionedCall: 
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
while_body_2416639
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_61_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_61_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_61_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_61_matmul_readvariableop_resource:	�F
3while_lstm_cell_61_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_61_biasadd_readvariableop_resource:	���)while/lstm_cell_61/BiasAdd/ReadVariableOp�(while/lstm_cell_61/MatMul/ReadVariableOp�*while/lstm_cell_61/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_61/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_61_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_61/MatMul/ReadVariableOp�
while/lstm_cell_61/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/MatMul�
*while/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_61_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_61/MatMul_1/ReadVariableOp�
while/lstm_cell_61/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/MatMul_1�
while/lstm_cell_61/addAddV2#while/lstm_cell_61/MatMul:product:0%while/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/add�
)while/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_61_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_61/BiasAdd/ReadVariableOp�
while/lstm_cell_61/BiasAddBiasAddwhile/lstm_cell_61/add:z:01while/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_61/BiasAdd�
"while/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_61/split/split_dim�
while/lstm_cell_61/splitSplit+while/lstm_cell_61/split/split_dim:output:0#while/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_61/split�
while/lstm_cell_61/SigmoidSigmoid!while/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Sigmoid�
while/lstm_cell_61/Sigmoid_1Sigmoid!while/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Sigmoid_1�
while/lstm_cell_61/mulMul while/lstm_cell_61/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/mul�
while/lstm_cell_61/ReluRelu!while/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Relu�
while/lstm_cell_61/mul_1Mulwhile/lstm_cell_61/Sigmoid:y:0%while/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/mul_1�
while/lstm_cell_61/add_1AddV2while/lstm_cell_61/mul:z:0while/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/add_1�
while/lstm_cell_61/Sigmoid_2Sigmoid!while/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Sigmoid_2�
while/lstm_cell_61/Relu_1Reluwhile/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/Relu_1�
while/lstm_cell_61/mul_2Mul while/lstm_cell_61/Sigmoid_2:y:0'while/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_61/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_61/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_61/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_61/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_61/BiasAdd/ReadVariableOp)^while/lstm_cell_61/MatMul/ReadVariableOp+^while/lstm_cell_61/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_61_biasadd_readvariableop_resource4while_lstm_cell_61_biasadd_readvariableop_resource_0"l
3while_lstm_cell_61_matmul_1_readvariableop_resource5while_lstm_cell_61_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_61_matmul_readvariableop_resource3while_lstm_cell_61_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_61/BiasAdd/ReadVariableOp)while/lstm_cell_61/BiasAdd/ReadVariableOp2T
(while/lstm_cell_61/MatMul/ReadVariableOp(while/lstm_cell_61/MatMul/ReadVariableOp2X
*while/lstm_cell_61/MatMul_1/ReadVariableOp*while/lstm_cell_61/MatMul_1/ReadVariableOp: 
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
/__inference_sequential_20_layer_call_fn_2414740

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
J__inference_sequential_20_layer_call_and_return_conditional_losses_24146992
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
�\
�
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_2416723

inputs>
+lstm_cell_61_matmul_readvariableop_resource:	�@
-lstm_cell_61_matmul_1_readvariableop_resource:	2�;
,lstm_cell_61_biasadd_readvariableop_resource:	�
identity��#lstm_cell_61/BiasAdd/ReadVariableOp�"lstm_cell_61/MatMul/ReadVariableOp�$lstm_cell_61/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_61/MatMul/ReadVariableOpReadVariableOp+lstm_cell_61_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_61/MatMul/ReadVariableOp�
lstm_cell_61/MatMulMatMulstrided_slice_2:output:0*lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/MatMul�
$lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_61_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_61/MatMul_1/ReadVariableOp�
lstm_cell_61/MatMul_1MatMulzeros:output:0,lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/MatMul_1�
lstm_cell_61/addAddV2lstm_cell_61/MatMul:product:0lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/add�
#lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_61_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_61/BiasAdd/ReadVariableOp�
lstm_cell_61/BiasAddBiasAddlstm_cell_61/add:z:0+lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_61/BiasAdd~
lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_61/split/split_dim�
lstm_cell_61/splitSplit%lstm_cell_61/split/split_dim:output:0lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_61/split�
lstm_cell_61/SigmoidSigmoidlstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/Sigmoid�
lstm_cell_61/Sigmoid_1Sigmoidlstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_61/Sigmoid_1�
lstm_cell_61/mulMullstm_cell_61/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/mul}
lstm_cell_61/ReluRelulstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_61/Relu�
lstm_cell_61/mul_1Mullstm_cell_61/Sigmoid:y:0lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/mul_1�
lstm_cell_61/add_1AddV2lstm_cell_61/mul:z:0lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/add_1�
lstm_cell_61/Sigmoid_2Sigmoidlstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_61/Sigmoid_2|
lstm_cell_61/Relu_1Relulstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/Relu_1�
lstm_cell_61/mul_2Mullstm_cell_61/Sigmoid_2:y:0!lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_61/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_61_matmul_readvariableop_resource-lstm_cell_61_matmul_1_readvariableop_resource,lstm_cell_61_biasadd_readvariableop_resource*
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
while_body_2416639*
condR
while_cond_2416638*K
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
NoOpNoOp$^lstm_cell_61/BiasAdd/ReadVariableOp#^lstm_cell_61/MatMul/ReadVariableOp%^lstm_cell_61/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_61/BiasAdd/ReadVariableOp#lstm_cell_61/BiasAdd/ReadVariableOp2H
"lstm_cell_61/MatMul/ReadVariableOp"lstm_cell_61/MatMul/ReadVariableOp2L
$lstm_cell_61/MatMul_1/ReadVariableOp$lstm_cell_61/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
.__inference_lstm_cell_62_layer_call_fn_2417662

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
I__inference_lstm_cell_62_layer_call_and_return_conditional_losses_24126132
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
�
�
while_cond_2412692
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2412692___redundant_placeholder05
1while_while_cond_2412692___redundant_placeholder15
1while_while_cond_2412692___redundant_placeholder25
1while_while_cond_2412692___redundant_placeholder3
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
.__inference_lstm_cell_61_layer_call_fn_2417547

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
I__inference_lstm_cell_61_layer_call_and_return_conditional_losses_24118352
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
��
�
Bsequential_20_bidirectional_20_backward_lstm_20_while_body_2411656|
xsequential_20_bidirectional_20_backward_lstm_20_while_sequential_20_bidirectional_20_backward_lstm_20_while_loop_counter�
~sequential_20_bidirectional_20_backward_lstm_20_while_sequential_20_bidirectional_20_backward_lstm_20_while_maximum_iterationsE
Asequential_20_bidirectional_20_backward_lstm_20_while_placeholderG
Csequential_20_bidirectional_20_backward_lstm_20_while_placeholder_1G
Csequential_20_bidirectional_20_backward_lstm_20_while_placeholder_2G
Csequential_20_bidirectional_20_backward_lstm_20_while_placeholder_3G
Csequential_20_bidirectional_20_backward_lstm_20_while_placeholder_4{
wsequential_20_bidirectional_20_backward_lstm_20_while_sequential_20_bidirectional_20_backward_lstm_20_strided_slice_1_0�
�sequential_20_bidirectional_20_backward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_20_bidirectional_20_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0v
rsequential_20_bidirectional_20_backward_lstm_20_while_less_sequential_20_bidirectional_20_backward_lstm_20_sub_1_0v
csequential_20_bidirectional_20_backward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0:	�x
esequential_20_bidirectional_20_backward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0:	2�s
dsequential_20_bidirectional_20_backward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0:	�B
>sequential_20_bidirectional_20_backward_lstm_20_while_identityD
@sequential_20_bidirectional_20_backward_lstm_20_while_identity_1D
@sequential_20_bidirectional_20_backward_lstm_20_while_identity_2D
@sequential_20_bidirectional_20_backward_lstm_20_while_identity_3D
@sequential_20_bidirectional_20_backward_lstm_20_while_identity_4D
@sequential_20_bidirectional_20_backward_lstm_20_while_identity_5D
@sequential_20_bidirectional_20_backward_lstm_20_while_identity_6y
usequential_20_bidirectional_20_backward_lstm_20_while_sequential_20_bidirectional_20_backward_lstm_20_strided_slice_1�
�sequential_20_bidirectional_20_backward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_20_bidirectional_20_backward_lstm_20_tensorarrayunstack_tensorlistfromtensort
psequential_20_bidirectional_20_backward_lstm_20_while_less_sequential_20_bidirectional_20_backward_lstm_20_sub_1t
asequential_20_bidirectional_20_backward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource:	�v
csequential_20_bidirectional_20_backward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource:	2�q
bsequential_20_bidirectional_20_backward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource:	���Ysequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp�Xsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp�Zsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp�
gsequential_20/bidirectional_20/backward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2i
gsequential_20/bidirectional_20/backward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape�
Ysequential_20/bidirectional_20/backward_lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_20_bidirectional_20_backward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_20_bidirectional_20_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0Asequential_20_bidirectional_20_backward_lstm_20_while_placeholderpsequential_20/bidirectional_20/backward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02[
Ysequential_20/bidirectional_20/backward_lstm_20/while/TensorArrayV2Read/TensorListGetItem�
:sequential_20/bidirectional_20/backward_lstm_20/while/LessLessrsequential_20_bidirectional_20_backward_lstm_20_while_less_sequential_20_bidirectional_20_backward_lstm_20_sub_1_0Asequential_20_bidirectional_20_backward_lstm_20_while_placeholder*
T0*#
_output_shapes
:���������2<
:sequential_20/bidirectional_20/backward_lstm_20/while/Less�
Xsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOpReadVariableOpcsequential_20_bidirectional_20_backward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02Z
Xsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp�
Isequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMulMatMul`sequential_20/bidirectional_20/backward_lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:0`sequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2K
Isequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMul�
Zsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOpesequential_20_bidirectional_20_backward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02\
Zsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp�
Ksequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMul_1MatMulCsequential_20_bidirectional_20_backward_lstm_20_while_placeholder_3bsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2M
Ksequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMul_1�
Fsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/addAddV2Ssequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMul:product:0Usequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2H
Fsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/add�
Ysequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOpdsequential_20_bidirectional_20_backward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02[
Ysequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp�
Jsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/BiasAddBiasAddJsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/add:z:0asequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2L
Jsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/BiasAdd�
Rsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2T
Rsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/split/split_dim�
Hsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/splitSplit[sequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/split/split_dim:output:0Ssequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2J
Hsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/split�
Jsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/SigmoidSigmoidQsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22L
Jsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/Sigmoid�
Lsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/Sigmoid_1SigmoidQsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22N
Lsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/Sigmoid_1�
Fsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/mulMulPsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/Sigmoid_1:y:0Csequential_20_bidirectional_20_backward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22H
Fsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/mul�
Gsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/ReluReluQsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22I
Gsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/Relu�
Hsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/mul_1MulNsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/Sigmoid:y:0Usequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22J
Hsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/mul_1�
Hsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/add_1AddV2Jsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/mul:z:0Lsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22J
Hsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/add_1�
Lsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/Sigmoid_2SigmoidQsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22N
Lsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/Sigmoid_2�
Isequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/Relu_1ReluLsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22K
Isequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/Relu_1�
Hsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/mul_2MulPsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/Sigmoid_2:y:0Wsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22J
Hsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/mul_2�
<sequential_20/bidirectional_20/backward_lstm_20/while/SelectSelect>sequential_20/bidirectional_20/backward_lstm_20/while/Less:z:0Lsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/mul_2:z:0Csequential_20_bidirectional_20_backward_lstm_20_while_placeholder_2*
T0*'
_output_shapes
:���������22>
<sequential_20/bidirectional_20/backward_lstm_20/while/Select�
>sequential_20/bidirectional_20/backward_lstm_20/while/Select_1Select>sequential_20/bidirectional_20/backward_lstm_20/while/Less:z:0Lsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/mul_2:z:0Csequential_20_bidirectional_20_backward_lstm_20_while_placeholder_3*
T0*'
_output_shapes
:���������22@
>sequential_20/bidirectional_20/backward_lstm_20/while/Select_1�
>sequential_20/bidirectional_20/backward_lstm_20/while/Select_2Select>sequential_20/bidirectional_20/backward_lstm_20/while/Less:z:0Lsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/add_1:z:0Csequential_20_bidirectional_20_backward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22@
>sequential_20/bidirectional_20/backward_lstm_20/while/Select_2�
Zsequential_20/bidirectional_20/backward_lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemCsequential_20_bidirectional_20_backward_lstm_20_while_placeholder_1Asequential_20_bidirectional_20_backward_lstm_20_while_placeholderEsequential_20/bidirectional_20/backward_lstm_20/while/Select:output:0*
_output_shapes
: *
element_dtype02\
Zsequential_20/bidirectional_20/backward_lstm_20/while/TensorArrayV2Write/TensorListSetItem�
;sequential_20/bidirectional_20/backward_lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2=
;sequential_20/bidirectional_20/backward_lstm_20/while/add/y�
9sequential_20/bidirectional_20/backward_lstm_20/while/addAddV2Asequential_20_bidirectional_20_backward_lstm_20_while_placeholderDsequential_20/bidirectional_20/backward_lstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2;
9sequential_20/bidirectional_20/backward_lstm_20/while/add�
=sequential_20/bidirectional_20/backward_lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2?
=sequential_20/bidirectional_20/backward_lstm_20/while/add_1/y�
;sequential_20/bidirectional_20/backward_lstm_20/while/add_1AddV2xsequential_20_bidirectional_20_backward_lstm_20_while_sequential_20_bidirectional_20_backward_lstm_20_while_loop_counterFsequential_20/bidirectional_20/backward_lstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2=
;sequential_20/bidirectional_20/backward_lstm_20/while/add_1�
>sequential_20/bidirectional_20/backward_lstm_20/while/IdentityIdentity?sequential_20/bidirectional_20/backward_lstm_20/while/add_1:z:0;^sequential_20/bidirectional_20/backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2@
>sequential_20/bidirectional_20/backward_lstm_20/while/Identity�
@sequential_20/bidirectional_20/backward_lstm_20/while/Identity_1Identity~sequential_20_bidirectional_20_backward_lstm_20_while_sequential_20_bidirectional_20_backward_lstm_20_while_maximum_iterations;^sequential_20/bidirectional_20/backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2B
@sequential_20/bidirectional_20/backward_lstm_20/while/Identity_1�
@sequential_20/bidirectional_20/backward_lstm_20/while/Identity_2Identity=sequential_20/bidirectional_20/backward_lstm_20/while/add:z:0;^sequential_20/bidirectional_20/backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2B
@sequential_20/bidirectional_20/backward_lstm_20/while/Identity_2�
@sequential_20/bidirectional_20/backward_lstm_20/while/Identity_3Identityjsequential_20/bidirectional_20/backward_lstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0;^sequential_20/bidirectional_20/backward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2B
@sequential_20/bidirectional_20/backward_lstm_20/while/Identity_3�
@sequential_20/bidirectional_20/backward_lstm_20/while/Identity_4IdentityEsequential_20/bidirectional_20/backward_lstm_20/while/Select:output:0;^sequential_20/bidirectional_20/backward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22B
@sequential_20/bidirectional_20/backward_lstm_20/while/Identity_4�
@sequential_20/bidirectional_20/backward_lstm_20/while/Identity_5IdentityGsequential_20/bidirectional_20/backward_lstm_20/while/Select_1:output:0;^sequential_20/bidirectional_20/backward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22B
@sequential_20/bidirectional_20/backward_lstm_20/while/Identity_5�
@sequential_20/bidirectional_20/backward_lstm_20/while/Identity_6IdentityGsequential_20/bidirectional_20/backward_lstm_20/while/Select_2:output:0;^sequential_20/bidirectional_20/backward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22B
@sequential_20/bidirectional_20/backward_lstm_20/while/Identity_6�
:sequential_20/bidirectional_20/backward_lstm_20/while/NoOpNoOpZ^sequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOpY^sequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp[^sequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2<
:sequential_20/bidirectional_20/backward_lstm_20/while/NoOp"�
>sequential_20_bidirectional_20_backward_lstm_20_while_identityGsequential_20/bidirectional_20/backward_lstm_20/while/Identity:output:0"�
@sequential_20_bidirectional_20_backward_lstm_20_while_identity_1Isequential_20/bidirectional_20/backward_lstm_20/while/Identity_1:output:0"�
@sequential_20_bidirectional_20_backward_lstm_20_while_identity_2Isequential_20/bidirectional_20/backward_lstm_20/while/Identity_2:output:0"�
@sequential_20_bidirectional_20_backward_lstm_20_while_identity_3Isequential_20/bidirectional_20/backward_lstm_20/while/Identity_3:output:0"�
@sequential_20_bidirectional_20_backward_lstm_20_while_identity_4Isequential_20/bidirectional_20/backward_lstm_20/while/Identity_4:output:0"�
@sequential_20_bidirectional_20_backward_lstm_20_while_identity_5Isequential_20/bidirectional_20/backward_lstm_20/while/Identity_5:output:0"�
@sequential_20_bidirectional_20_backward_lstm_20_while_identity_6Isequential_20/bidirectional_20/backward_lstm_20/while/Identity_6:output:0"�
psequential_20_bidirectional_20_backward_lstm_20_while_less_sequential_20_bidirectional_20_backward_lstm_20_sub_1rsequential_20_bidirectional_20_backward_lstm_20_while_less_sequential_20_bidirectional_20_backward_lstm_20_sub_1_0"�
bsequential_20_bidirectional_20_backward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resourcedsequential_20_bidirectional_20_backward_lstm_20_while_lstm_cell_62_biasadd_readvariableop_resource_0"�
csequential_20_bidirectional_20_backward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resourceesequential_20_bidirectional_20_backward_lstm_20_while_lstm_cell_62_matmul_1_readvariableop_resource_0"�
asequential_20_bidirectional_20_backward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resourcecsequential_20_bidirectional_20_backward_lstm_20_while_lstm_cell_62_matmul_readvariableop_resource_0"�
usequential_20_bidirectional_20_backward_lstm_20_while_sequential_20_bidirectional_20_backward_lstm_20_strided_slice_1wsequential_20_bidirectional_20_backward_lstm_20_while_sequential_20_bidirectional_20_backward_lstm_20_strided_slice_1_0"�
�sequential_20_bidirectional_20_backward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_20_bidirectional_20_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor�sequential_20_bidirectional_20_backward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_20_bidirectional_20_backward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2�
Ysequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOpYsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/BiasAdd/ReadVariableOp2�
Xsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOpXsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMul/ReadVariableOp2�
Zsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOpZsequential_20/bidirectional_20/backward_lstm_20/while/lstm_cell_62/MatMul_1/ReadVariableOp: 
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
�b
�
"forward_lstm_20_while_body_2414360<
8forward_lstm_20_while_forward_lstm_20_while_loop_counterB
>forward_lstm_20_while_forward_lstm_20_while_maximum_iterations%
!forward_lstm_20_while_placeholder'
#forward_lstm_20_while_placeholder_1'
#forward_lstm_20_while_placeholder_2'
#forward_lstm_20_while_placeholder_3'
#forward_lstm_20_while_placeholder_4;
7forward_lstm_20_while_forward_lstm_20_strided_slice_1_0w
sforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_08
4forward_lstm_20_while_greater_forward_lstm_20_cast_0V
Cforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0:	�X
Eforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0:	2�S
Dforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0:	�"
forward_lstm_20_while_identity$
 forward_lstm_20_while_identity_1$
 forward_lstm_20_while_identity_2$
 forward_lstm_20_while_identity_3$
 forward_lstm_20_while_identity_4$
 forward_lstm_20_while_identity_5$
 forward_lstm_20_while_identity_69
5forward_lstm_20_while_forward_lstm_20_strided_slice_1u
qforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor6
2forward_lstm_20_while_greater_forward_lstm_20_castT
Aforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource:	�V
Cforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource:	2�Q
Bforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource:	���9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp�8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp�:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp�
Gforward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2I
Gforward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape�
9forward_lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_20_while_placeholderPforward_lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02;
9forward_lstm_20/while/TensorArrayV2Read/TensorListGetItem�
forward_lstm_20/while/GreaterGreater4forward_lstm_20_while_greater_forward_lstm_20_cast_0!forward_lstm_20_while_placeholder*
T0*#
_output_shapes
:���������2
forward_lstm_20/while/Greater�
8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOpReadVariableOpCforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02:
8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp�
)forward_lstm_20/while/lstm_cell_61/MatMulMatMul@forward_lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:0@forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)forward_lstm_20/while/lstm_cell_61/MatMul�
:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOpEforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02<
:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp�
+forward_lstm_20/while/lstm_cell_61/MatMul_1MatMul#forward_lstm_20_while_placeholder_3Bforward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+forward_lstm_20/while/lstm_cell_61/MatMul_1�
&forward_lstm_20/while/lstm_cell_61/addAddV23forward_lstm_20/while/lstm_cell_61/MatMul:product:05forward_lstm_20/while/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&forward_lstm_20/while/lstm_cell_61/add�
9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOpDforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02;
9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp�
*forward_lstm_20/while/lstm_cell_61/BiasAddBiasAdd*forward_lstm_20/while/lstm_cell_61/add:z:0Aforward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*forward_lstm_20/while/lstm_cell_61/BiasAdd�
2forward_lstm_20/while/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2forward_lstm_20/while/lstm_cell_61/split/split_dim�
(forward_lstm_20/while/lstm_cell_61/splitSplit;forward_lstm_20/while/lstm_cell_61/split/split_dim:output:03forward_lstm_20/while/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2*
(forward_lstm_20/while/lstm_cell_61/split�
*forward_lstm_20/while/lstm_cell_61/SigmoidSigmoid1forward_lstm_20/while/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22,
*forward_lstm_20/while/lstm_cell_61/Sigmoid�
,forward_lstm_20/while/lstm_cell_61/Sigmoid_1Sigmoid1forward_lstm_20/while/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22.
,forward_lstm_20/while/lstm_cell_61/Sigmoid_1�
&forward_lstm_20/while/lstm_cell_61/mulMul0forward_lstm_20/while/lstm_cell_61/Sigmoid_1:y:0#forward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22(
&forward_lstm_20/while/lstm_cell_61/mul�
'forward_lstm_20/while/lstm_cell_61/ReluRelu1forward_lstm_20/while/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22)
'forward_lstm_20/while/lstm_cell_61/Relu�
(forward_lstm_20/while/lstm_cell_61/mul_1Mul.forward_lstm_20/while/lstm_cell_61/Sigmoid:y:05forward_lstm_20/while/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_20/while/lstm_cell_61/mul_1�
(forward_lstm_20/while/lstm_cell_61/add_1AddV2*forward_lstm_20/while/lstm_cell_61/mul:z:0,forward_lstm_20/while/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_20/while/lstm_cell_61/add_1�
,forward_lstm_20/while/lstm_cell_61/Sigmoid_2Sigmoid1forward_lstm_20/while/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22.
,forward_lstm_20/while/lstm_cell_61/Sigmoid_2�
)forward_lstm_20/while/lstm_cell_61/Relu_1Relu,forward_lstm_20/while/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22+
)forward_lstm_20/while/lstm_cell_61/Relu_1�
(forward_lstm_20/while/lstm_cell_61/mul_2Mul0forward_lstm_20/while/lstm_cell_61/Sigmoid_2:y:07forward_lstm_20/while/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_20/while/lstm_cell_61/mul_2�
forward_lstm_20/while/SelectSelect!forward_lstm_20/while/Greater:z:0,forward_lstm_20/while/lstm_cell_61/mul_2:z:0#forward_lstm_20_while_placeholder_2*
T0*'
_output_shapes
:���������22
forward_lstm_20/while/Select�
forward_lstm_20/while/Select_1Select!forward_lstm_20/while/Greater:z:0,forward_lstm_20/while/lstm_cell_61/mul_2:z:0#forward_lstm_20_while_placeholder_3*
T0*'
_output_shapes
:���������22 
forward_lstm_20/while/Select_1�
forward_lstm_20/while/Select_2Select!forward_lstm_20/while/Greater:z:0,forward_lstm_20/while/lstm_cell_61/add_1:z:0#forward_lstm_20_while_placeholder_4*
T0*'
_output_shapes
:���������22 
forward_lstm_20/while/Select_2�
:forward_lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_20_while_placeholder_1!forward_lstm_20_while_placeholder%forward_lstm_20/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_20/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_20/while/add/y�
forward_lstm_20/while/addAddV2!forward_lstm_20_while_placeholder$forward_lstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/while/add�
forward_lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_20/while/add_1/y�
forward_lstm_20/while/add_1AddV28forward_lstm_20_while_forward_lstm_20_while_loop_counter&forward_lstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/while/add_1�
forward_lstm_20/while/IdentityIdentityforward_lstm_20/while/add_1:z:0^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_20/while/Identity�
 forward_lstm_20/while/Identity_1Identity>forward_lstm_20_while_forward_lstm_20_while_maximum_iterations^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_20/while/Identity_1�
 forward_lstm_20/while/Identity_2Identityforward_lstm_20/while/add:z:0^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_20/while/Identity_2�
 forward_lstm_20/while/Identity_3IdentityJforward_lstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_20/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_20/while/Identity_3�
 forward_lstm_20/while/Identity_4Identity%forward_lstm_20/while/Select:output:0^forward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/while/Identity_4�
 forward_lstm_20/while/Identity_5Identity'forward_lstm_20/while/Select_1:output:0^forward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/while/Identity_5�
 forward_lstm_20/while/Identity_6Identity'forward_lstm_20/while/Select_2:output:0^forward_lstm_20/while/NoOp*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/while/Identity_6�
forward_lstm_20/while/NoOpNoOp:^forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp9^forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp;^forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_20/while/NoOp"p
5forward_lstm_20_while_forward_lstm_20_strided_slice_17forward_lstm_20_while_forward_lstm_20_strided_slice_1_0"j
2forward_lstm_20_while_greater_forward_lstm_20_cast4forward_lstm_20_while_greater_forward_lstm_20_cast_0"I
forward_lstm_20_while_identity'forward_lstm_20/while/Identity:output:0"M
 forward_lstm_20_while_identity_1)forward_lstm_20/while/Identity_1:output:0"M
 forward_lstm_20_while_identity_2)forward_lstm_20/while/Identity_2:output:0"M
 forward_lstm_20_while_identity_3)forward_lstm_20/while/Identity_3:output:0"M
 forward_lstm_20_while_identity_4)forward_lstm_20/while/Identity_4:output:0"M
 forward_lstm_20_while_identity_5)forward_lstm_20/while/Identity_5:output:0"M
 forward_lstm_20_while_identity_6)forward_lstm_20/while/Identity_6:output:0"�
Bforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resourceDforward_lstm_20_while_lstm_cell_61_biasadd_readvariableop_resource_0"�
Cforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resourceEforward_lstm_20_while_lstm_cell_61_matmul_1_readvariableop_resource_0"�
Aforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resourceCforward_lstm_20_while_lstm_cell_61_matmul_readvariableop_resource_0"�
qforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensorsforward_lstm_20_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2v
9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp9forward_lstm_20/while/lstm_cell_61/BiasAdd/ReadVariableOp2t
8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp8forward_lstm_20/while/lstm_cell_61/MatMul/ReadVariableOp2x
:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp:forward_lstm_20/while/lstm_cell_61/MatMul_1/ReadVariableOp: 
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
while_cond_2413100
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2413100___redundant_placeholder05
1while_while_cond_2413100___redundant_placeholder15
1while_while_cond_2413100___redundant_placeholder25
1while_while_cond_2413100___redundant_placeholder3
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
�
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_2415188
inputs_0N
;forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource:	�P
=forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource:	2�K
<forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource:	�O
<backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource:	�Q
>backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource:	2�L
=backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource:	�
identity��4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp�3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp�5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp�backward_lstm_20/while�3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp�2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp�4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp�forward_lstm_20/whilef
forward_lstm_20/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_20/Shape�
#forward_lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_20/strided_slice/stack�
%forward_lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_20/strided_slice/stack_1�
%forward_lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_20/strided_slice/stack_2�
forward_lstm_20/strided_sliceStridedSliceforward_lstm_20/Shape:output:0,forward_lstm_20/strided_slice/stack:output:0.forward_lstm_20/strided_slice/stack_1:output:0.forward_lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_20/strided_slice|
forward_lstm_20/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_20/zeros/mul/y�
forward_lstm_20/zeros/mulMul&forward_lstm_20/strided_slice:output:0$forward_lstm_20/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros/mul
forward_lstm_20/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_20/zeros/Less/y�
forward_lstm_20/zeros/LessLessforward_lstm_20/zeros/mul:z:0%forward_lstm_20/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros/Less�
forward_lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_20/zeros/packed/1�
forward_lstm_20/zeros/packedPack&forward_lstm_20/strided_slice:output:0'forward_lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_20/zeros/packed�
forward_lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_20/zeros/Const�
forward_lstm_20/zerosFill%forward_lstm_20/zeros/packed:output:0$forward_lstm_20/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_20/zeros�
forward_lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_20/zeros_1/mul/y�
forward_lstm_20/zeros_1/mulMul&forward_lstm_20/strided_slice:output:0&forward_lstm_20/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros_1/mul�
forward_lstm_20/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
forward_lstm_20/zeros_1/Less/y�
forward_lstm_20/zeros_1/LessLessforward_lstm_20/zeros_1/mul:z:0'forward_lstm_20/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_20/zeros_1/Less�
 forward_lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_20/zeros_1/packed/1�
forward_lstm_20/zeros_1/packedPack&forward_lstm_20/strided_slice:output:0)forward_lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_20/zeros_1/packed�
forward_lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_20/zeros_1/Const�
forward_lstm_20/zeros_1Fill'forward_lstm_20/zeros_1/packed:output:0&forward_lstm_20/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_20/zeros_1�
forward_lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_20/transpose/perm�
forward_lstm_20/transpose	Transposeinputs_0'forward_lstm_20/transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
forward_lstm_20/transpose
forward_lstm_20/Shape_1Shapeforward_lstm_20/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_20/Shape_1�
%forward_lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_20/strided_slice_1/stack�
'forward_lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_1/stack_1�
'forward_lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_1/stack_2�
forward_lstm_20/strided_slice_1StridedSlice forward_lstm_20/Shape_1:output:0.forward_lstm_20/strided_slice_1/stack:output:00forward_lstm_20/strided_slice_1/stack_1:output:00forward_lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_20/strided_slice_1�
+forward_lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+forward_lstm_20/TensorArrayV2/element_shape�
forward_lstm_20/TensorArrayV2TensorListReserve4forward_lstm_20/TensorArrayV2/element_shape:output:0(forward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_20/TensorArrayV2�
Eforward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2G
Eforward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape�
7forward_lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_20/transpose:y:0Nforward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_20/TensorArrayUnstack/TensorListFromTensor�
%forward_lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_20/strided_slice_2/stack�
'forward_lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_2/stack_1�
'forward_lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_2/stack_2�
forward_lstm_20/strided_slice_2StridedSliceforward_lstm_20/transpose:y:0.forward_lstm_20/strided_slice_2/stack:output:00forward_lstm_20/strided_slice_2/stack_1:output:00forward_lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2!
forward_lstm_20/strided_slice_2�
2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOpReadVariableOp;forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp�
#forward_lstm_20/lstm_cell_61/MatMulMatMul(forward_lstm_20/strided_slice_2:output:0:forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#forward_lstm_20/lstm_cell_61/MatMul�
4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOpReadVariableOp=forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype026
4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp�
%forward_lstm_20/lstm_cell_61/MatMul_1MatMulforward_lstm_20/zeros:output:0<forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%forward_lstm_20/lstm_cell_61/MatMul_1�
 forward_lstm_20/lstm_cell_61/addAddV2-forward_lstm_20/lstm_cell_61/MatMul:product:0/forward_lstm_20/lstm_cell_61/MatMul_1:product:0*
T0*(
_output_shapes
:����������2"
 forward_lstm_20/lstm_cell_61/add�
3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOpReadVariableOp<forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp�
$forward_lstm_20/lstm_cell_61/BiasAddBiasAdd$forward_lstm_20/lstm_cell_61/add:z:0;forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$forward_lstm_20/lstm_cell_61/BiasAdd�
,forward_lstm_20/lstm_cell_61/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,forward_lstm_20/lstm_cell_61/split/split_dim�
"forward_lstm_20/lstm_cell_61/splitSplit5forward_lstm_20/lstm_cell_61/split/split_dim:output:0-forward_lstm_20/lstm_cell_61/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2$
"forward_lstm_20/lstm_cell_61/split�
$forward_lstm_20/lstm_cell_61/SigmoidSigmoid+forward_lstm_20/lstm_cell_61/split:output:0*
T0*'
_output_shapes
:���������22&
$forward_lstm_20/lstm_cell_61/Sigmoid�
&forward_lstm_20/lstm_cell_61/Sigmoid_1Sigmoid+forward_lstm_20/lstm_cell_61/split:output:1*
T0*'
_output_shapes
:���������22(
&forward_lstm_20/lstm_cell_61/Sigmoid_1�
 forward_lstm_20/lstm_cell_61/mulMul*forward_lstm_20/lstm_cell_61/Sigmoid_1:y:0 forward_lstm_20/zeros_1:output:0*
T0*'
_output_shapes
:���������22"
 forward_lstm_20/lstm_cell_61/mul�
!forward_lstm_20/lstm_cell_61/ReluRelu+forward_lstm_20/lstm_cell_61/split:output:2*
T0*'
_output_shapes
:���������22#
!forward_lstm_20/lstm_cell_61/Relu�
"forward_lstm_20/lstm_cell_61/mul_1Mul(forward_lstm_20/lstm_cell_61/Sigmoid:y:0/forward_lstm_20/lstm_cell_61/Relu:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_20/lstm_cell_61/mul_1�
"forward_lstm_20/lstm_cell_61/add_1AddV2$forward_lstm_20/lstm_cell_61/mul:z:0&forward_lstm_20/lstm_cell_61/mul_1:z:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_20/lstm_cell_61/add_1�
&forward_lstm_20/lstm_cell_61/Sigmoid_2Sigmoid+forward_lstm_20/lstm_cell_61/split:output:3*
T0*'
_output_shapes
:���������22(
&forward_lstm_20/lstm_cell_61/Sigmoid_2�
#forward_lstm_20/lstm_cell_61/Relu_1Relu&forward_lstm_20/lstm_cell_61/add_1:z:0*
T0*'
_output_shapes
:���������22%
#forward_lstm_20/lstm_cell_61/Relu_1�
"forward_lstm_20/lstm_cell_61/mul_2Mul*forward_lstm_20/lstm_cell_61/Sigmoid_2:y:01forward_lstm_20/lstm_cell_61/Relu_1:activations:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_20/lstm_cell_61/mul_2�
-forward_lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2/
-forward_lstm_20/TensorArrayV2_1/element_shape�
forward_lstm_20/TensorArrayV2_1TensorListReserve6forward_lstm_20/TensorArrayV2_1/element_shape:output:0(forward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_20/TensorArrayV2_1n
forward_lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_20/time�
(forward_lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(forward_lstm_20/while/maximum_iterations�
"forward_lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_20/while/loop_counter�
forward_lstm_20/whileWhile+forward_lstm_20/while/loop_counter:output:01forward_lstm_20/while/maximum_iterations:output:0forward_lstm_20/time:output:0(forward_lstm_20/TensorArrayV2_1:handle:0forward_lstm_20/zeros:output:0 forward_lstm_20/zeros_1:output:0(forward_lstm_20/strided_slice_1:output:0Gforward_lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:0;forward_lstm_20_lstm_cell_61_matmul_readvariableop_resource=forward_lstm_20_lstm_cell_61_matmul_1_readvariableop_resource<forward_lstm_20_lstm_cell_61_biasadd_readvariableop_resource*
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
"forward_lstm_20_while_body_2414953*.
cond&R$
"forward_lstm_20_while_cond_2414952*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
forward_lstm_20/while�
@forward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2B
@forward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape�
2forward_lstm_20/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_20/while:output:3Iforward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype024
2forward_lstm_20/TensorArrayV2Stack/TensorListStack�
%forward_lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2'
%forward_lstm_20/strided_slice_3/stack�
'forward_lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_20/strided_slice_3/stack_1�
'forward_lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_20/strided_slice_3/stack_2�
forward_lstm_20/strided_slice_3StridedSlice;forward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_20/strided_slice_3/stack:output:00forward_lstm_20/strided_slice_3/stack_1:output:00forward_lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2!
forward_lstm_20/strided_slice_3�
 forward_lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_20/transpose_1/perm�
forward_lstm_20/transpose_1	Transpose;forward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_20/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
forward_lstm_20/transpose_1�
forward_lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_20/runtimeh
backward_lstm_20/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_20/Shape�
$backward_lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_20/strided_slice/stack�
&backward_lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_20/strided_slice/stack_1�
&backward_lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_20/strided_slice/stack_2�
backward_lstm_20/strided_sliceStridedSlicebackward_lstm_20/Shape:output:0-backward_lstm_20/strided_slice/stack:output:0/backward_lstm_20/strided_slice/stack_1:output:0/backward_lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_20/strided_slice~
backward_lstm_20/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_20/zeros/mul/y�
backward_lstm_20/zeros/mulMul'backward_lstm_20/strided_slice:output:0%backward_lstm_20/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros/mul�
backward_lstm_20/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
backward_lstm_20/zeros/Less/y�
backward_lstm_20/zeros/LessLessbackward_lstm_20/zeros/mul:z:0&backward_lstm_20/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros/Less�
backward_lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_20/zeros/packed/1�
backward_lstm_20/zeros/packedPack'backward_lstm_20/strided_slice:output:0(backward_lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_20/zeros/packed�
backward_lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_20/zeros/Const�
backward_lstm_20/zerosFill&backward_lstm_20/zeros/packed:output:0%backward_lstm_20/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_20/zeros�
backward_lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_20/zeros_1/mul/y�
backward_lstm_20/zeros_1/mulMul'backward_lstm_20/strided_slice:output:0'backward_lstm_20/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros_1/mul�
backward_lstm_20/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2!
backward_lstm_20/zeros_1/Less/y�
backward_lstm_20/zeros_1/LessLess backward_lstm_20/zeros_1/mul:z:0(backward_lstm_20/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_20/zeros_1/Less�
!backward_lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_20/zeros_1/packed/1�
backward_lstm_20/zeros_1/packedPack'backward_lstm_20/strided_slice:output:0*backward_lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_20/zeros_1/packed�
backward_lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_20/zeros_1/Const�
backward_lstm_20/zeros_1Fill(backward_lstm_20/zeros_1/packed:output:0'backward_lstm_20/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_20/zeros_1�
backward_lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_20/transpose/perm�
backward_lstm_20/transpose	Transposeinputs_0(backward_lstm_20/transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
backward_lstm_20/transpose�
backward_lstm_20/Shape_1Shapebackward_lstm_20/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_20/Shape_1�
&backward_lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_20/strided_slice_1/stack�
(backward_lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_1/stack_1�
(backward_lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_1/stack_2�
 backward_lstm_20/strided_slice_1StridedSlice!backward_lstm_20/Shape_1:output:0/backward_lstm_20/strided_slice_1/stack:output:01backward_lstm_20/strided_slice_1/stack_1:output:01backward_lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_20/strided_slice_1�
,backward_lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2.
,backward_lstm_20/TensorArrayV2/element_shape�
backward_lstm_20/TensorArrayV2TensorListReserve5backward_lstm_20/TensorArrayV2/element_shape:output:0)backward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_20/TensorArrayV2�
backward_lstm_20/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_20/ReverseV2/axis�
backward_lstm_20/ReverseV2	ReverseV2backward_lstm_20/transpose:y:0(backward_lstm_20/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'���������������������������2
backward_lstm_20/ReverseV2�
Fbackward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2H
Fbackward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape�
8backward_lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_20/ReverseV2:output:0Obackward_lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_20/TensorArrayUnstack/TensorListFromTensor�
&backward_lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_20/strided_slice_2/stack�
(backward_lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_2/stack_1�
(backward_lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_2/stack_2�
 backward_lstm_20/strided_slice_2StridedSlicebackward_lstm_20/transpose:y:0/backward_lstm_20/strided_slice_2/stack:output:01backward_lstm_20/strided_slice_2/stack_1:output:01backward_lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2"
 backward_lstm_20/strided_slice_2�
3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOpReadVariableOp<backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype025
3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp�
$backward_lstm_20/lstm_cell_62/MatMulMatMul)backward_lstm_20/strided_slice_2:output:0;backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$backward_lstm_20/lstm_cell_62/MatMul�
5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOp>backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype027
5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp�
&backward_lstm_20/lstm_cell_62/MatMul_1MatMulbackward_lstm_20/zeros:output:0=backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&backward_lstm_20/lstm_cell_62/MatMul_1�
!backward_lstm_20/lstm_cell_62/addAddV2.backward_lstm_20/lstm_cell_62/MatMul:product:00backward_lstm_20/lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2#
!backward_lstm_20/lstm_cell_62/add�
4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOp=backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype026
4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp�
%backward_lstm_20/lstm_cell_62/BiasAddBiasAdd%backward_lstm_20/lstm_cell_62/add:z:0<backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%backward_lstm_20/lstm_cell_62/BiasAdd�
-backward_lstm_20/lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-backward_lstm_20/lstm_cell_62/split/split_dim�
#backward_lstm_20/lstm_cell_62/splitSplit6backward_lstm_20/lstm_cell_62/split/split_dim:output:0.backward_lstm_20/lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2%
#backward_lstm_20/lstm_cell_62/split�
%backward_lstm_20/lstm_cell_62/SigmoidSigmoid,backward_lstm_20/lstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22'
%backward_lstm_20/lstm_cell_62/Sigmoid�
'backward_lstm_20/lstm_cell_62/Sigmoid_1Sigmoid,backward_lstm_20/lstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22)
'backward_lstm_20/lstm_cell_62/Sigmoid_1�
!backward_lstm_20/lstm_cell_62/mulMul+backward_lstm_20/lstm_cell_62/Sigmoid_1:y:0!backward_lstm_20/zeros_1:output:0*
T0*'
_output_shapes
:���������22#
!backward_lstm_20/lstm_cell_62/mul�
"backward_lstm_20/lstm_cell_62/ReluRelu,backward_lstm_20/lstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22$
"backward_lstm_20/lstm_cell_62/Relu�
#backward_lstm_20/lstm_cell_62/mul_1Mul)backward_lstm_20/lstm_cell_62/Sigmoid:y:00backward_lstm_20/lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_20/lstm_cell_62/mul_1�
#backward_lstm_20/lstm_cell_62/add_1AddV2%backward_lstm_20/lstm_cell_62/mul:z:0'backward_lstm_20/lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_20/lstm_cell_62/add_1�
'backward_lstm_20/lstm_cell_62/Sigmoid_2Sigmoid,backward_lstm_20/lstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22)
'backward_lstm_20/lstm_cell_62/Sigmoid_2�
$backward_lstm_20/lstm_cell_62/Relu_1Relu'backward_lstm_20/lstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22&
$backward_lstm_20/lstm_cell_62/Relu_1�
#backward_lstm_20/lstm_cell_62/mul_2Mul+backward_lstm_20/lstm_cell_62/Sigmoid_2:y:02backward_lstm_20/lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_20/lstm_cell_62/mul_2�
.backward_lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   20
.backward_lstm_20/TensorArrayV2_1/element_shape�
 backward_lstm_20/TensorArrayV2_1TensorListReserve7backward_lstm_20/TensorArrayV2_1/element_shape:output:0)backward_lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_20/TensorArrayV2_1p
backward_lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_20/time�
)backward_lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2+
)backward_lstm_20/while/maximum_iterations�
#backward_lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_20/while/loop_counter�
backward_lstm_20/whileWhile,backward_lstm_20/while/loop_counter:output:02backward_lstm_20/while/maximum_iterations:output:0backward_lstm_20/time:output:0)backward_lstm_20/TensorArrayV2_1:handle:0backward_lstm_20/zeros:output:0!backward_lstm_20/zeros_1:output:0)backward_lstm_20/strided_slice_1:output:0Hbackward_lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:0<backward_lstm_20_lstm_cell_62_matmul_readvariableop_resource>backward_lstm_20_lstm_cell_62_matmul_1_readvariableop_resource=backward_lstm_20_lstm_cell_62_biasadd_readvariableop_resource*
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
#backward_lstm_20_while_body_2415102*/
cond'R%
#backward_lstm_20_while_cond_2415101*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
backward_lstm_20/while�
Abackward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2C
Abackward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape�
3backward_lstm_20/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_20/while:output:3Jbackward_lstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype025
3backward_lstm_20/TensorArrayV2Stack/TensorListStack�
&backward_lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2(
&backward_lstm_20/strided_slice_3/stack�
(backward_lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_20/strided_slice_3/stack_1�
(backward_lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_20/strided_slice_3/stack_2�
 backward_lstm_20/strided_slice_3StridedSlice<backward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_20/strided_slice_3/stack:output:01backward_lstm_20/strided_slice_3/stack_1:output:01backward_lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2"
 backward_lstm_20/strided_slice_3�
!backward_lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_20/transpose_1/perm�
backward_lstm_20/transpose_1	Transpose<backward_lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_20/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
backward_lstm_20/transpose_1�
backward_lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_20/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2(forward_lstm_20/strided_slice_3:output:0)backward_lstm_20/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp5^backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp4^backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp6^backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp^backward_lstm_20/while4^forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp3^forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp5^forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp^forward_lstm_20/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 2l
4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp4backward_lstm_20/lstm_cell_62/BiasAdd/ReadVariableOp2j
3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp3backward_lstm_20/lstm_cell_62/MatMul/ReadVariableOp2n
5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp5backward_lstm_20/lstm_cell_62/MatMul_1/ReadVariableOp20
backward_lstm_20/whilebackward_lstm_20/while2j
3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp3forward_lstm_20/lstm_cell_61/BiasAdd/ReadVariableOp2h
2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp2forward_lstm_20/lstm_cell_61/MatMul/ReadVariableOp2l
4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp4forward_lstm_20/lstm_cell_61/MatMul_1/ReadVariableOp2.
forward_lstm_20/whileforward_lstm_20/while:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs/0
�
�
2__inference_backward_lstm_20_layer_call_fn_2416918

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
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_24135372
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
�^
�
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_2417224
inputs_0>
+lstm_cell_62_matmul_readvariableop_resource:	�@
-lstm_cell_62_matmul_1_readvariableop_resource:	2�;
,lstm_cell_62_biasadd_readvariableop_resource:	�
identity��#lstm_cell_62/BiasAdd/ReadVariableOp�"lstm_cell_62/MatMul/ReadVariableOp�$lstm_cell_62/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_62/MatMul/ReadVariableOpReadVariableOp+lstm_cell_62_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_62/MatMul/ReadVariableOp�
lstm_cell_62/MatMulMatMulstrided_slice_2:output:0*lstm_cell_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/MatMul�
$lstm_cell_62/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_62_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_62/MatMul_1/ReadVariableOp�
lstm_cell_62/MatMul_1MatMulzeros:output:0,lstm_cell_62/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/MatMul_1�
lstm_cell_62/addAddV2lstm_cell_62/MatMul:product:0lstm_cell_62/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/add�
#lstm_cell_62/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_62/BiasAdd/ReadVariableOp�
lstm_cell_62/BiasAddBiasAddlstm_cell_62/add:z:0+lstm_cell_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_62/BiasAdd~
lstm_cell_62/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_62/split/split_dim�
lstm_cell_62/splitSplit%lstm_cell_62/split/split_dim:output:0lstm_cell_62/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_62/split�
lstm_cell_62/SigmoidSigmoidlstm_cell_62/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/Sigmoid�
lstm_cell_62/Sigmoid_1Sigmoidlstm_cell_62/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_62/Sigmoid_1�
lstm_cell_62/mulMullstm_cell_62/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/mul}
lstm_cell_62/ReluRelulstm_cell_62/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_62/Relu�
lstm_cell_62/mul_1Mullstm_cell_62/Sigmoid:y:0lstm_cell_62/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/mul_1�
lstm_cell_62/add_1AddV2lstm_cell_62/mul:z:0lstm_cell_62/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/add_1�
lstm_cell_62/Sigmoid_2Sigmoidlstm_cell_62/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_62/Sigmoid_2|
lstm_cell_62/Relu_1Relulstm_cell_62/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/Relu_1�
lstm_cell_62/mul_2Mullstm_cell_62/Sigmoid_2:y:0!lstm_cell_62/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_62/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_62_matmul_readvariableop_resource-lstm_cell_62_matmul_1_readvariableop_resource,lstm_cell_62_biasadd_readvariableop_resource*
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
while_body_2417140*
condR
while_cond_2417139*K
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
NoOpNoOp$^lstm_cell_62/BiasAdd/ReadVariableOp#^lstm_cell_62/MatMul/ReadVariableOp%^lstm_cell_62/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_62/BiasAdd/ReadVariableOp#lstm_cell_62/BiasAdd/ReadVariableOp2H
"lstm_cell_62/MatMul/ReadVariableOp"lstm_cell_62/MatMul/ReadVariableOp2L
$lstm_cell_62/MatMul_1/ReadVariableOp$lstm_cell_62/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0"�L
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
dense_200
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
!:d2dense_20/kernel
:2dense_20/bias
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
G:E	�24bidirectional_20/forward_lstm_20/lstm_cell_61/kernel
Q:O	2�2>bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel
A:?�22bidirectional_20/forward_lstm_20/lstm_cell_61/bias
H:F	�25bidirectional_20/backward_lstm_20/lstm_cell_62/kernel
R:P	2�2?bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel
B:@�23bidirectional_20/backward_lstm_20/lstm_cell_62/bias
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
&:$d2Adam/dense_20/kernel/m
 :2Adam/dense_20/bias/m
L:J	�2;Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/m
V:T	2�2EAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/m
F:D�29Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/m
M:K	�2<Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/m
W:U	2�2FAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/m
G:E�2:Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/m
&:$d2Adam/dense_20/kernel/v
 :2Adam/dense_20/bias/v
L:J	�2;Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/v
V:T	2�2EAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/v
F:D�29Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/v
M:K	�2<Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/v
W:U	2�2FAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/v
G:E�2:Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/v
):'d2Adam/dense_20/kernel/vhat
#:!2Adam/dense_20/bias/vhat
O:M	�2>Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/kernel/vhat
Y:W	2�2HAdam/bidirectional_20/forward_lstm_20/lstm_cell_61/recurrent_kernel/vhat
I:G�2<Adam/bidirectional_20/forward_lstm_20/lstm_cell_61/bias/vhat
P:N	�2?Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/kernel/vhat
Z:X	2�2IAdam/bidirectional_20/backward_lstm_20/lstm_cell_62/recurrent_kernel/vhat
J:H�2=Adam/bidirectional_20/backward_lstm_20/lstm_cell_62/bias/vhat
�2�
/__inference_sequential_20_layer_call_fn_2414247
/__inference_sequential_20_layer_call_fn_2414740�
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
"__inference__wrapped_model_2411760args_0args_0_1"�
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
J__inference_sequential_20_layer_call_and_return_conditional_losses_2414763
J__inference_sequential_20_layer_call_and_return_conditional_losses_2414786�
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
2__inference_bidirectional_20_layer_call_fn_2414833
2__inference_bidirectional_20_layer_call_fn_2414850
2__inference_bidirectional_20_layer_call_fn_2414868
2__inference_bidirectional_20_layer_call_fn_2414886�
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
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_2415188
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_2415490
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_2415848
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_2416206�
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
*__inference_dense_20_layer_call_fn_2416215�
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
E__inference_dense_20_layer_call_and_return_conditional_losses_2416226�
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
%__inference_signature_wrapper_2414816args_0args_0_1"�
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
1__inference_forward_lstm_20_layer_call_fn_2416237
1__inference_forward_lstm_20_layer_call_fn_2416248
1__inference_forward_lstm_20_layer_call_fn_2416259
1__inference_forward_lstm_20_layer_call_fn_2416270�
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
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_2416421
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_2416572
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_2416723
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_2416874�
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
2__inference_backward_lstm_20_layer_call_fn_2416885
2__inference_backward_lstm_20_layer_call_fn_2416896
2__inference_backward_lstm_20_layer_call_fn_2416907
2__inference_backward_lstm_20_layer_call_fn_2416918�
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
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_2417071
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_2417224
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_2417377
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_2417530�
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
.__inference_lstm_cell_61_layer_call_fn_2417547
.__inference_lstm_cell_61_layer_call_fn_2417564�
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
I__inference_lstm_cell_61_layer_call_and_return_conditional_losses_2417596
I__inference_lstm_cell_61_layer_call_and_return_conditional_losses_2417628�
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
.__inference_lstm_cell_62_layer_call_fn_2417645
.__inference_lstm_cell_62_layer_call_fn_2417662�
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
I__inference_lstm_cell_62_layer_call_and_return_conditional_losses_2417694
I__inference_lstm_cell_62_layer_call_and_return_conditional_losses_2417726�
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
"__inference__wrapped_model_2411760�\�Y
R�O
M�J4�1
!�������������������
�
`
�	RaggedTensorSpec
� "3�0
.
dense_20"�
dense_20����������
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_2417071}O�L
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
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_2417224}O�L
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
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_2417377Q�N
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
M__inference_backward_lstm_20_layer_call_and_return_conditional_losses_2417530Q�N
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
2__inference_backward_lstm_20_layer_call_fn_2416885pO�L
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
2__inference_backward_lstm_20_layer_call_fn_2416896pO�L
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
2__inference_backward_lstm_20_layer_call_fn_2416907rQ�N
G�D
6�3
inputs'���������������������������

 
p 

 
� "����������2�
2__inference_backward_lstm_20_layer_call_fn_2416918rQ�N
G�D
6�3
inputs'���������������������������

 
p

 
� "����������2�
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_2415188�\�Y
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
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_2415490�\�Y
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
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_2415848�l�i
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
M__inference_bidirectional_20_layer_call_and_return_conditional_losses_2416206�l�i
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
2__inference_bidirectional_20_layer_call_fn_2414833�\�Y
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
2__inference_bidirectional_20_layer_call_fn_2414850�\�Y
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
2__inference_bidirectional_20_layer_call_fn_2414868�l�i
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
2__inference_bidirectional_20_layer_call_fn_2414886�l�i
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
E__inference_dense_20_layer_call_and_return_conditional_losses_2416226\/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� }
*__inference_dense_20_layer_call_fn_2416215O/�,
%�"
 �
inputs���������d
� "�����������
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_2416421}O�L
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
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_2416572}O�L
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
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_2416723Q�N
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
L__inference_forward_lstm_20_layer_call_and_return_conditional_losses_2416874Q�N
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
1__inference_forward_lstm_20_layer_call_fn_2416237pO�L
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
1__inference_forward_lstm_20_layer_call_fn_2416248pO�L
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
1__inference_forward_lstm_20_layer_call_fn_2416259rQ�N
G�D
6�3
inputs'���������������������������

 
p 

 
� "����������2�
1__inference_forward_lstm_20_layer_call_fn_2416270rQ�N
G�D
6�3
inputs'���������������������������

 
p

 
� "����������2�
I__inference_lstm_cell_61_layer_call_and_return_conditional_losses_2417596���}
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
I__inference_lstm_cell_61_layer_call_and_return_conditional_losses_2417628���}
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
.__inference_lstm_cell_61_layer_call_fn_2417547���}
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
.__inference_lstm_cell_61_layer_call_fn_2417564���}
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
I__inference_lstm_cell_62_layer_call_and_return_conditional_losses_2417694���}
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
I__inference_lstm_cell_62_layer_call_and_return_conditional_losses_2417726���}
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
.__inference_lstm_cell_62_layer_call_fn_2417645���}
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
.__inference_lstm_cell_62_layer_call_fn_2417662���}
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
J__inference_sequential_20_layer_call_and_return_conditional_losses_2414763�d�a
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
J__inference_sequential_20_layer_call_and_return_conditional_losses_2414786�d�a
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
/__inference_sequential_20_layer_call_fn_2414247�d�a
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
/__inference_sequential_20_layer_call_fn_2414740�d�a
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
%__inference_signature_wrapper_2414816�e�b
� 
[�X
*
args_0 �
args_0���������
*
args_0_1�
args_0_1���������	"3�0
.
dense_20"�
dense_20���������