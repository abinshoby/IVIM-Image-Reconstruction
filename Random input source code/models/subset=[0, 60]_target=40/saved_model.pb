��:
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
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:d*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
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
2bidirectional_4/forward_lstm_4/lstm_cell_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*C
shared_name42bidirectional_4/forward_lstm_4/lstm_cell_13/kernel
�
Fbidirectional_4/forward_lstm_4/lstm_cell_13/kernel/Read/ReadVariableOpReadVariableOp2bidirectional_4/forward_lstm_4/lstm_cell_13/kernel*
_output_shapes
:	�*
dtype0
�
<bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*M
shared_name><bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel
�
Pbidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/Read/ReadVariableOpReadVariableOp<bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel*
_output_shapes
:	2�*
dtype0
�
0bidirectional_4/forward_lstm_4/lstm_cell_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*A
shared_name20bidirectional_4/forward_lstm_4/lstm_cell_13/bias
�
Dbidirectional_4/forward_lstm_4/lstm_cell_13/bias/Read/ReadVariableOpReadVariableOp0bidirectional_4/forward_lstm_4/lstm_cell_13/bias*
_output_shapes	
:�*
dtype0
�
3bidirectional_4/backward_lstm_4/lstm_cell_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*D
shared_name53bidirectional_4/backward_lstm_4/lstm_cell_14/kernel
�
Gbidirectional_4/backward_lstm_4/lstm_cell_14/kernel/Read/ReadVariableOpReadVariableOp3bidirectional_4/backward_lstm_4/lstm_cell_14/kernel*
_output_shapes
:	�*
dtype0
�
=bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*N
shared_name?=bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel
�
Qbidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/Read/ReadVariableOpReadVariableOp=bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel*
_output_shapes
:	2�*
dtype0
�
1bidirectional_4/backward_lstm_4/lstm_cell_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*B
shared_name31bidirectional_4/backward_lstm_4/lstm_cell_14/bias
�
Ebidirectional_4/backward_lstm_4/lstm_cell_14/bias/Read/ReadVariableOpReadVariableOp1bidirectional_4/backward_lstm_4/lstm_cell_14/bias*
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
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:d*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:*
dtype0
�
9Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*J
shared_name;9Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/m
�
MAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/m/Read/ReadVariableOpReadVariableOp9Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/m*
_output_shapes
:	�*
dtype0
�
CAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*T
shared_nameECAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/m
�
WAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpCAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/m*
_output_shapes
:	2�*
dtype0
�
7Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*H
shared_name97Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/m
�
KAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/m/Read/ReadVariableOpReadVariableOp7Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/m*
_output_shapes	
:�*
dtype0
�
:Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*K
shared_name<:Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/m
�
NAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/m/Read/ReadVariableOpReadVariableOp:Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/m*
_output_shapes
:	�*
dtype0
�
DAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*U
shared_nameFDAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/m
�
XAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpDAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/m*
_output_shapes
:	2�*
dtype0
�
8Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*I
shared_name:8Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/m
�
LAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/m/Read/ReadVariableOpReadVariableOp8Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:d*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:*
dtype0
�
9Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*J
shared_name;9Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/v
�
MAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/v/Read/ReadVariableOpReadVariableOp9Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/v*
_output_shapes
:	�*
dtype0
�
CAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*T
shared_nameECAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/v
�
WAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpCAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/v*
_output_shapes
:	2�*
dtype0
�
7Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*H
shared_name97Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/v
�
KAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/v/Read/ReadVariableOpReadVariableOp7Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/v*
_output_shapes	
:�*
dtype0
�
:Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*K
shared_name<:Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/v
�
NAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/v/Read/ReadVariableOpReadVariableOp:Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/v*
_output_shapes
:	�*
dtype0
�
DAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*U
shared_nameFDAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/v
�
XAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpDAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/v*
_output_shapes
:	2�*
dtype0
�
8Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*I
shared_name:8Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/v
�
LAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/v/Read/ReadVariableOpReadVariableOp8Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_4/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*)
shared_nameAdam/dense_4/kernel/vhat
�
,Adam/dense_4/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/vhat*
_output_shapes

:d*
dtype0
�
Adam/dense_4/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_4/bias/vhat
}
*Adam/dense_4/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/vhat*
_output_shapes
:*
dtype0
�
<Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*M
shared_name><Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/vhat
�
PAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/vhat/Read/ReadVariableOpReadVariableOp<Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/vhat*
_output_shapes
:	�*
dtype0
�
FAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*W
shared_nameHFAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/vhat
�
ZAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/vhat*
_output_shapes
:	2�*
dtype0
�
:Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*K
shared_name<:Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/vhat
�
NAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/vhat/Read/ReadVariableOpReadVariableOp:Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/vhat*
_output_shapes	
:�*
dtype0
�
=Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*N
shared_name?=Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/vhat
�
QAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/vhat/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/vhat*
_output_shapes
:	�*
dtype0
�
GAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2�*X
shared_nameIGAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/vhat
�
[Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpGAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/vhat*
_output_shapes
:	2�*
dtype0
�
;Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*L
shared_name=;Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/vhat
�
OAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/vhat/Read/ReadVariableOpReadVariableOp;Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/vhat*
_output_shapes	
:�*
dtype0

NoOpNoOp
�@
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�?
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
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
nl
VARIABLE_VALUE2bidirectional_4/forward_lstm_4/lstm_cell_13/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE<bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE0bidirectional_4/forward_lstm_4/lstm_cell_13/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3bidirectional_4/backward_lstm_4/lstm_cell_14/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE=bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1bidirectional_4/backward_lstm_4/lstm_cell_14/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
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
}{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE9Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUECAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE7Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE:Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEDAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE8Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE9Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUECAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE7Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE:Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEDAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE8Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_4/kernel/vhatUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_4/bias/vhatSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE<Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/vhatEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEFAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/vhatEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE:Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/vhatEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE=Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/vhatEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEGAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/vhatEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE;Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/vhatEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_0_12bidirectional_4/forward_lstm_4/lstm_cell_13/kernel<bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel0bidirectional_4/forward_lstm_4/lstm_cell_13/bias3bidirectional_4/backward_lstm_4/lstm_cell_14/kernel=bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel1bidirectional_4/backward_lstm_4/lstm_cell_14/biasdense_4/kerneldense_4/bias*
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
GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_693582
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpFbidirectional_4/forward_lstm_4/lstm_cell_13/kernel/Read/ReadVariableOpPbidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/Read/ReadVariableOpDbidirectional_4/forward_lstm_4/lstm_cell_13/bias/Read/ReadVariableOpGbidirectional_4/backward_lstm_4/lstm_cell_14/kernel/Read/ReadVariableOpQbidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/Read/ReadVariableOpEbidirectional_4/backward_lstm_4/lstm_cell_14/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOpMAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/m/Read/ReadVariableOpWAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/m/Read/ReadVariableOpKAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/m/Read/ReadVariableOpNAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/m/Read/ReadVariableOpXAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/m/Read/ReadVariableOpLAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOpMAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/v/Read/ReadVariableOpWAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/v/Read/ReadVariableOpKAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/v/Read/ReadVariableOpNAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/v/Read/ReadVariableOpXAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/v/Read/ReadVariableOpLAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/v/Read/ReadVariableOp,Adam/dense_4/kernel/vhat/Read/ReadVariableOp*Adam/dense_4/bias/vhat/Read/ReadVariableOpPAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/vhat/Read/ReadVariableOpZAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/vhat/Read/ReadVariableOpNAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/vhat/Read/ReadVariableOpQAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/vhat/Read/ReadVariableOp[Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/vhat/Read/ReadVariableOpOAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/vhat/Read/ReadVariableOpConst*4
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
GPU 2J 8� *(
f#R!
__inference__traced_save_696633
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_4/kerneldense_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate2bidirectional_4/forward_lstm_4/lstm_cell_13/kernel<bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel0bidirectional_4/forward_lstm_4/lstm_cell_13/bias3bidirectional_4/backward_lstm_4/lstm_cell_14/kernel=bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel1bidirectional_4/backward_lstm_4/lstm_cell_14/biastotalcountAdam/dense_4/kernel/mAdam/dense_4/bias/m9Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/mCAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/m7Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/m:Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/mDAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/m8Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/mAdam/dense_4/kernel/vAdam/dense_4/bias/v9Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/vCAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/v7Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/v:Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/vDAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/v8Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/vAdam/dense_4/kernel/vhatAdam/dense_4/bias/vhat<Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/vhatFAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/vhat:Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/vhat=Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/vhatGAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/vhat;Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/vhat*3
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
GPU 2J 8� *+
f&R$
"__inference__traced_restore_696760��7
�
�
while_cond_695253
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_695253___redundant_placeholder04
0while_while_cond_695253___redundant_placeholder14
0while_while_cond_695253___redundant_placeholder24
0while_while_cond_695253___redundant_placeholder3
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
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_692476

inputs>
+lstm_cell_13_matmul_readvariableop_resource:	�@
-lstm_cell_13_matmul_1_readvariableop_resource:	2�;
,lstm_cell_13_biasadd_readvariableop_resource:	�
identity��#lstm_cell_13/BiasAdd/ReadVariableOp�"lstm_cell_13/MatMul/ReadVariableOp�$lstm_cell_13/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp�
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/MatMul�
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp�
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/MatMul_1�
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/add�
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp�
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim�
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_13/split�
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/Sigmoid�
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_13/Sigmoid_1�
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/mul}
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_13/Relu�
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/mul_1�
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/add_1�
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_13/Sigmoid_2|
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/Relu_1�
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
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
bodyR
while_body_692392*
condR
while_cond_692391*K
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
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
Ʋ
�
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_692962

inputs
inputs_1	M
:forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource:	�O
<forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource:	2�J
;forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource:	�N
;backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource:	�P
=backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource:	2�K
<backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource:	�
identity��3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp�2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp�4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp�backward_lstm_4/while�2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp�1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp�3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp�forward_lstm_4/while�
#forward_lstm_4/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2%
#forward_lstm_4/RaggedToTensor/zeros�
#forward_lstm_4/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2%
#forward_lstm_4/RaggedToTensor/Const�
2forward_lstm_4/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor,forward_lstm_4/RaggedToTensor/Const:output:0inputs,forward_lstm_4/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS24
2forward_lstm_4/RaggedToTensor/RaggedTensorToTensor�
9forward_lstm_4/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack�
;forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1�
;forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2�
3forward_lstm_4/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Bforward_lstm_4/RaggedNestedRowLengths/strided_slice/stack:output:0Dforward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1:output:0Dforward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask25
3forward_lstm_4/RaggedNestedRowLengths/strided_slice�
;forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack�
=forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2?
=forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1�
=forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2�
5forward_lstm_4/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Dforward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack:output:0Fforward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Fforward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask27
5forward_lstm_4/RaggedNestedRowLengths/strided_slice_1�
)forward_lstm_4/RaggedNestedRowLengths/subSub<forward_lstm_4/RaggedNestedRowLengths/strided_slice:output:0>forward_lstm_4/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2+
)forward_lstm_4/RaggedNestedRowLengths/sub�
forward_lstm_4/CastCast-forward_lstm_4/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
forward_lstm_4/Cast�
forward_lstm_4/ShapeShape;forward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_4/Shape�
"forward_lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"forward_lstm_4/strided_slice/stack�
$forward_lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_4/strided_slice/stack_1�
$forward_lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_4/strided_slice/stack_2�
forward_lstm_4/strided_sliceStridedSliceforward_lstm_4/Shape:output:0+forward_lstm_4/strided_slice/stack:output:0-forward_lstm_4/strided_slice/stack_1:output:0-forward_lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_4/strided_slicez
forward_lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_4/zeros/mul/y�
forward_lstm_4/zeros/mulMul%forward_lstm_4/strided_slice:output:0#forward_lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros/mul}
forward_lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_4/zeros/Less/y�
forward_lstm_4/zeros/LessLessforward_lstm_4/zeros/mul:z:0$forward_lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros/Less�
forward_lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_4/zeros/packed/1�
forward_lstm_4/zeros/packedPack%forward_lstm_4/strided_slice:output:0&forward_lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_4/zeros/packed�
forward_lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_4/zeros/Const�
forward_lstm_4/zerosFill$forward_lstm_4/zeros/packed:output:0#forward_lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_4/zeros~
forward_lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_4/zeros_1/mul/y�
forward_lstm_4/zeros_1/mulMul%forward_lstm_4/strided_slice:output:0%forward_lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros_1/mul�
forward_lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_4/zeros_1/Less/y�
forward_lstm_4/zeros_1/LessLessforward_lstm_4/zeros_1/mul:z:0&forward_lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros_1/Less�
forward_lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_4/zeros_1/packed/1�
forward_lstm_4/zeros_1/packedPack%forward_lstm_4/strided_slice:output:0(forward_lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_4/zeros_1/packed�
forward_lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_4/zeros_1/Const�
forward_lstm_4/zeros_1Fill&forward_lstm_4/zeros_1/packed:output:0%forward_lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_4/zeros_1�
forward_lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
forward_lstm_4/transpose/perm�
forward_lstm_4/transpose	Transpose;forward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0&forward_lstm_4/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
forward_lstm_4/transpose|
forward_lstm_4/Shape_1Shapeforward_lstm_4/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_4/Shape_1�
$forward_lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_4/strided_slice_1/stack�
&forward_lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_1/stack_1�
&forward_lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_1/stack_2�
forward_lstm_4/strided_slice_1StridedSliceforward_lstm_4/Shape_1:output:0-forward_lstm_4/strided_slice_1/stack:output:0/forward_lstm_4/strided_slice_1/stack_1:output:0/forward_lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_4/strided_slice_1�
*forward_lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2,
*forward_lstm_4/TensorArrayV2/element_shape�
forward_lstm_4/TensorArrayV2TensorListReserve3forward_lstm_4/TensorArrayV2/element_shape:output:0'forward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_4/TensorArrayV2�
Dforward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2F
Dforward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape�
6forward_lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_4/transpose:y:0Mforward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6forward_lstm_4/TensorArrayUnstack/TensorListFromTensor�
$forward_lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_4/strided_slice_2/stack�
&forward_lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_2/stack_1�
&forward_lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_2/stack_2�
forward_lstm_4/strided_slice_2StridedSliceforward_lstm_4/transpose:y:0-forward_lstm_4/strided_slice_2/stack:output:0/forward_lstm_4/strided_slice_2/stack_1:output:0/forward_lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2 
forward_lstm_4/strided_slice_2�
1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp:forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype023
1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp�
"forward_lstm_4/lstm_cell_13/MatMulMatMul'forward_lstm_4/strided_slice_2:output:09forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"forward_lstm_4/lstm_cell_13/MatMul�
3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp<forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype025
3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp�
$forward_lstm_4/lstm_cell_13/MatMul_1MatMulforward_lstm_4/zeros:output:0;forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$forward_lstm_4/lstm_cell_13/MatMul_1�
forward_lstm_4/lstm_cell_13/addAddV2,forward_lstm_4/lstm_cell_13/MatMul:product:0.forward_lstm_4/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2!
forward_lstm_4/lstm_cell_13/add�
2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp;forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype024
2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp�
#forward_lstm_4/lstm_cell_13/BiasAddBiasAdd#forward_lstm_4/lstm_cell_13/add:z:0:forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#forward_lstm_4/lstm_cell_13/BiasAdd�
+forward_lstm_4/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+forward_lstm_4/lstm_cell_13/split/split_dim�
!forward_lstm_4/lstm_cell_13/splitSplit4forward_lstm_4/lstm_cell_13/split/split_dim:output:0,forward_lstm_4/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2#
!forward_lstm_4/lstm_cell_13/split�
#forward_lstm_4/lstm_cell_13/SigmoidSigmoid*forward_lstm_4/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22%
#forward_lstm_4/lstm_cell_13/Sigmoid�
%forward_lstm_4/lstm_cell_13/Sigmoid_1Sigmoid*forward_lstm_4/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22'
%forward_lstm_4/lstm_cell_13/Sigmoid_1�
forward_lstm_4/lstm_cell_13/mulMul)forward_lstm_4/lstm_cell_13/Sigmoid_1:y:0forward_lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:���������22!
forward_lstm_4/lstm_cell_13/mul�
 forward_lstm_4/lstm_cell_13/ReluRelu*forward_lstm_4/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22"
 forward_lstm_4/lstm_cell_13/Relu�
!forward_lstm_4/lstm_cell_13/mul_1Mul'forward_lstm_4/lstm_cell_13/Sigmoid:y:0.forward_lstm_4/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22#
!forward_lstm_4/lstm_cell_13/mul_1�
!forward_lstm_4/lstm_cell_13/add_1AddV2#forward_lstm_4/lstm_cell_13/mul:z:0%forward_lstm_4/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22#
!forward_lstm_4/lstm_cell_13/add_1�
%forward_lstm_4/lstm_cell_13/Sigmoid_2Sigmoid*forward_lstm_4/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22'
%forward_lstm_4/lstm_cell_13/Sigmoid_2�
"forward_lstm_4/lstm_cell_13/Relu_1Relu%forward_lstm_4/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_4/lstm_cell_13/Relu_1�
!forward_lstm_4/lstm_cell_13/mul_2Mul)forward_lstm_4/lstm_cell_13/Sigmoid_2:y:00forward_lstm_4/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22#
!forward_lstm_4/lstm_cell_13/mul_2�
,forward_lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2.
,forward_lstm_4/TensorArrayV2_1/element_shape�
forward_lstm_4/TensorArrayV2_1TensorListReserve5forward_lstm_4/TensorArrayV2_1/element_shape:output:0'forward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_4/TensorArrayV2_1l
forward_lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_4/time�
forward_lstm_4/zeros_like	ZerosLike%forward_lstm_4/lstm_cell_13/mul_2:z:0*
T0*'
_output_shapes
:���������22
forward_lstm_4/zeros_like�
'forward_lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2)
'forward_lstm_4/while/maximum_iterations�
!forward_lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!forward_lstm_4/while/loop_counter�
forward_lstm_4/whileWhile*forward_lstm_4/while/loop_counter:output:00forward_lstm_4/while/maximum_iterations:output:0forward_lstm_4/time:output:0'forward_lstm_4/TensorArrayV2_1:handle:0forward_lstm_4/zeros_like:y:0forward_lstm_4/zeros:output:0forward_lstm_4/zeros_1:output:0'forward_lstm_4/strided_slice_1:output:0Fforward_lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_4/Cast:y:0:forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource<forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource;forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *,
body$R"
 forward_lstm_4_while_body_692686*,
cond$R"
 forward_lstm_4_while_cond_692685*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
forward_lstm_4/while�
?forward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2A
?forward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape�
1forward_lstm_4/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_4/while:output:3Hforward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype023
1forward_lstm_4/TensorArrayV2Stack/TensorListStack�
$forward_lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2&
$forward_lstm_4/strided_slice_3/stack�
&forward_lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_4/strided_slice_3/stack_1�
&forward_lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_3/stack_2�
forward_lstm_4/strided_slice_3StridedSlice:forward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0-forward_lstm_4/strided_slice_3/stack:output:0/forward_lstm_4/strided_slice_3/stack_1:output:0/forward_lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2 
forward_lstm_4/strided_slice_3�
forward_lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_4/transpose_1/perm�
forward_lstm_4/transpose_1	Transpose:forward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0(forward_lstm_4/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
forward_lstm_4/transpose_1�
forward_lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_4/runtime�
$backward_lstm_4/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$backward_lstm_4/RaggedToTensor/zeros�
$backward_lstm_4/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2&
$backward_lstm_4/RaggedToTensor/Const�
3backward_lstm_4/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-backward_lstm_4/RaggedToTensor/Const:output:0inputs-backward_lstm_4/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3backward_lstm_4/RaggedToTensor/RaggedTensorToTensor�
:backward_lstm_4/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack�
<backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1�
<backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2�
4backward_lstm_4/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cbackward_lstm_4/RaggedNestedRowLengths/strided_slice/stack:output:0Ebackward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1:output:0Ebackward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask26
4backward_lstm_4/RaggedNestedRowLengths/strided_slice�
<backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack�
>backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2@
>backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1�
>backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2�
6backward_lstm_4/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Ebackward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gbackward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gbackward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask28
6backward_lstm_4/RaggedNestedRowLengths/strided_slice_1�
*backward_lstm_4/RaggedNestedRowLengths/subSub=backward_lstm_4/RaggedNestedRowLengths/strided_slice:output:0?backward_lstm_4/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2,
*backward_lstm_4/RaggedNestedRowLengths/sub�
backward_lstm_4/CastCast.backward_lstm_4/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
backward_lstm_4/Cast�
backward_lstm_4/ShapeShape<backward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_4/Shape�
#backward_lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#backward_lstm_4/strided_slice/stack�
%backward_lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_4/strided_slice/stack_1�
%backward_lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_4/strided_slice/stack_2�
backward_lstm_4/strided_sliceStridedSlicebackward_lstm_4/Shape:output:0,backward_lstm_4/strided_slice/stack:output:0.backward_lstm_4/strided_slice/stack_1:output:0.backward_lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
backward_lstm_4/strided_slice|
backward_lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_4/zeros/mul/y�
backward_lstm_4/zeros/mulMul&backward_lstm_4/strided_slice:output:0$backward_lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros/mul
backward_lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
backward_lstm_4/zeros/Less/y�
backward_lstm_4/zeros/LessLessbackward_lstm_4/zeros/mul:z:0%backward_lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros/Less�
backward_lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_4/zeros/packed/1�
backward_lstm_4/zeros/packedPack&backward_lstm_4/strided_slice:output:0'backward_lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_4/zeros/packed�
backward_lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_4/zeros/Const�
backward_lstm_4/zerosFill%backward_lstm_4/zeros/packed:output:0$backward_lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_4/zeros�
backward_lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_4/zeros_1/mul/y�
backward_lstm_4/zeros_1/mulMul&backward_lstm_4/strided_slice:output:0&backward_lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros_1/mul�
backward_lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
backward_lstm_4/zeros_1/Less/y�
backward_lstm_4/zeros_1/LessLessbackward_lstm_4/zeros_1/mul:z:0'backward_lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros_1/Less�
 backward_lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_4/zeros_1/packed/1�
backward_lstm_4/zeros_1/packedPack&backward_lstm_4/strided_slice:output:0)backward_lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_4/zeros_1/packed�
backward_lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_4/zeros_1/Const�
backward_lstm_4/zeros_1Fill'backward_lstm_4/zeros_1/packed:output:0&backward_lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_4/zeros_1�
backward_lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
backward_lstm_4/transpose/perm�
backward_lstm_4/transpose	Transpose<backward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0'backward_lstm_4/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_4/transpose
backward_lstm_4/Shape_1Shapebackward_lstm_4/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_4/Shape_1�
%backward_lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_4/strided_slice_1/stack�
'backward_lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_1/stack_1�
'backward_lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_1/stack_2�
backward_lstm_4/strided_slice_1StridedSlice backward_lstm_4/Shape_1:output:0.backward_lstm_4/strided_slice_1/stack:output:00backward_lstm_4/strided_slice_1/stack_1:output:00backward_lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_4/strided_slice_1�
+backward_lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+backward_lstm_4/TensorArrayV2/element_shape�
backward_lstm_4/TensorArrayV2TensorListReserve4backward_lstm_4/TensorArrayV2/element_shape:output:0(backward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
backward_lstm_4/TensorArrayV2�
backward_lstm_4/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2 
backward_lstm_4/ReverseV2/axis�
backward_lstm_4/ReverseV2	ReverseV2backward_lstm_4/transpose:y:0'backward_lstm_4/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_4/ReverseV2�
Ebackward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2G
Ebackward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape�
7backward_lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"backward_lstm_4/ReverseV2:output:0Nbackward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7backward_lstm_4/TensorArrayUnstack/TensorListFromTensor�
%backward_lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_4/strided_slice_2/stack�
'backward_lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_2/stack_1�
'backward_lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_2/stack_2�
backward_lstm_4/strided_slice_2StridedSlicebackward_lstm_4/transpose:y:0.backward_lstm_4/strided_slice_2/stack:output:00backward_lstm_4/strided_slice_2/stack_1:output:00backward_lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2!
backward_lstm_4/strided_slice_2�
2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp;backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp�
#backward_lstm_4/lstm_cell_14/MatMulMatMul(backward_lstm_4/strided_slice_2:output:0:backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#backward_lstm_4/lstm_cell_14/MatMul�
4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp=backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype026
4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp�
%backward_lstm_4/lstm_cell_14/MatMul_1MatMulbackward_lstm_4/zeros:output:0<backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%backward_lstm_4/lstm_cell_14/MatMul_1�
 backward_lstm_4/lstm_cell_14/addAddV2-backward_lstm_4/lstm_cell_14/MatMul:product:0/backward_lstm_4/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2"
 backward_lstm_4/lstm_cell_14/add�
3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp<backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp�
$backward_lstm_4/lstm_cell_14/BiasAddBiasAdd$backward_lstm_4/lstm_cell_14/add:z:0;backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$backward_lstm_4/lstm_cell_14/BiasAdd�
,backward_lstm_4/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,backward_lstm_4/lstm_cell_14/split/split_dim�
"backward_lstm_4/lstm_cell_14/splitSplit5backward_lstm_4/lstm_cell_14/split/split_dim:output:0-backward_lstm_4/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2$
"backward_lstm_4/lstm_cell_14/split�
$backward_lstm_4/lstm_cell_14/SigmoidSigmoid+backward_lstm_4/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22&
$backward_lstm_4/lstm_cell_14/Sigmoid�
&backward_lstm_4/lstm_cell_14/Sigmoid_1Sigmoid+backward_lstm_4/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22(
&backward_lstm_4/lstm_cell_14/Sigmoid_1�
 backward_lstm_4/lstm_cell_14/mulMul*backward_lstm_4/lstm_cell_14/Sigmoid_1:y:0 backward_lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/lstm_cell_14/mul�
!backward_lstm_4/lstm_cell_14/ReluRelu+backward_lstm_4/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22#
!backward_lstm_4/lstm_cell_14/Relu�
"backward_lstm_4/lstm_cell_14/mul_1Mul(backward_lstm_4/lstm_cell_14/Sigmoid:y:0/backward_lstm_4/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22$
"backward_lstm_4/lstm_cell_14/mul_1�
"backward_lstm_4/lstm_cell_14/add_1AddV2$backward_lstm_4/lstm_cell_14/mul:z:0&backward_lstm_4/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22$
"backward_lstm_4/lstm_cell_14/add_1�
&backward_lstm_4/lstm_cell_14/Sigmoid_2Sigmoid+backward_lstm_4/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22(
&backward_lstm_4/lstm_cell_14/Sigmoid_2�
#backward_lstm_4/lstm_cell_14/Relu_1Relu&backward_lstm_4/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_4/lstm_cell_14/Relu_1�
"backward_lstm_4/lstm_cell_14/mul_2Mul*backward_lstm_4/lstm_cell_14/Sigmoid_2:y:01backward_lstm_4/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22$
"backward_lstm_4/lstm_cell_14/mul_2�
-backward_lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2/
-backward_lstm_4/TensorArrayV2_1/element_shape�
backward_lstm_4/TensorArrayV2_1TensorListReserve6backward_lstm_4/TensorArrayV2_1/element_shape:output:0(backward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_4/TensorArrayV2_1n
backward_lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_4/time�
%backward_lstm_4/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2'
%backward_lstm_4/Max/reduction_indices�
backward_lstm_4/MaxMaxbackward_lstm_4/Cast:y:0.backward_lstm_4/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/Maxp
backward_lstm_4/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_4/sub/y�
backward_lstm_4/subSubbackward_lstm_4/Max:output:0backward_lstm_4/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/sub�
backward_lstm_4/Sub_1Subbackward_lstm_4/sub:z:0backward_lstm_4/Cast:y:0*
T0*#
_output_shapes
:���������2
backward_lstm_4/Sub_1�
backward_lstm_4/zeros_like	ZerosLike&backward_lstm_4/lstm_cell_14/mul_2:z:0*
T0*'
_output_shapes
:���������22
backward_lstm_4/zeros_like�
(backward_lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(backward_lstm_4/while/maximum_iterations�
"backward_lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"backward_lstm_4/while/loop_counter�
backward_lstm_4/whileWhile+backward_lstm_4/while/loop_counter:output:01backward_lstm_4/while/maximum_iterations:output:0backward_lstm_4/time:output:0(backward_lstm_4/TensorArrayV2_1:handle:0backward_lstm_4/zeros_like:y:0backward_lstm_4/zeros:output:0 backward_lstm_4/zeros_1:output:0(backward_lstm_4/strided_slice_1:output:0Gbackward_lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_4/Sub_1:z:0;backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource=backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource<backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *-
body%R#
!backward_lstm_4_while_body_692865*-
cond%R#
!backward_lstm_4_while_cond_692864*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
backward_lstm_4/while�
@backward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2B
@backward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape�
2backward_lstm_4/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_4/while:output:3Ibackward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype024
2backward_lstm_4/TensorArrayV2Stack/TensorListStack�
%backward_lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2'
%backward_lstm_4/strided_slice_3/stack�
'backward_lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_4/strided_slice_3/stack_1�
'backward_lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_3/stack_2�
backward_lstm_4/strided_slice_3StridedSlice;backward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0.backward_lstm_4/strided_slice_3/stack:output:00backward_lstm_4/strided_slice_3/stack_1:output:00backward_lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2!
backward_lstm_4/strided_slice_3�
 backward_lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_4/transpose_1/perm�
backward_lstm_4/transpose_1	Transpose;backward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0)backward_lstm_4/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
backward_lstm_4/transpose_1�
backward_lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_4/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2'forward_lstm_4/strided_slice_3:output:0(backward_lstm_4/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp4^backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp3^backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp5^backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp^backward_lstm_4/while3^forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp2^forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp4^forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp^forward_lstm_4/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������: : : : : : 2j
3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp2h
2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp2l
4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp2.
backward_lstm_4/whilebackward_lstm_4/while2h
2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp2f
1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp2j
3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp2,
forward_lstm_4/whileforward_lstm_4/while:O K
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
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_695187
inputs_0>
+lstm_cell_13_matmul_readvariableop_resource:	�@
-lstm_cell_13_matmul_1_readvariableop_resource:	2�;
,lstm_cell_13_biasadd_readvariableop_resource:	�
identity��#lstm_cell_13/BiasAdd/ReadVariableOp�"lstm_cell_13/MatMul/ReadVariableOp�$lstm_cell_13/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp�
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/MatMul�
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp�
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/MatMul_1�
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/add�
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp�
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim�
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_13/split�
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/Sigmoid�
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_13/Sigmoid_1�
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/mul}
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_13/Relu�
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/mul_1�
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/add_1�
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_13/Sigmoid_2|
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/Relu_1�
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
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
bodyR
while_body_695103*
condR
while_cond_695102*K
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
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�^
�
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_696143

inputs>
+lstm_cell_14_matmul_readvariableop_resource:	�@
-lstm_cell_14_matmul_1_readvariableop_resource:	2�;
,lstm_cell_14_biasadd_readvariableop_resource:	�
identity��#lstm_cell_14/BiasAdd/ReadVariableOp�"lstm_cell_14/MatMul/ReadVariableOp�$lstm_cell_14/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp�
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/MatMul�
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp�
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/MatMul_1�
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/add�
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp�
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/BiasAdd~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim�
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_14/split�
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/Sigmoid�
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_14/Sigmoid_1�
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_14/Relu�
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/mul_1�
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/add_1�
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/Relu_1�
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
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
bodyR
while_body_696059*
condR
while_cond_696058*K
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
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
/__inference_forward_lstm_4_layer_call_fn_695025

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
GPU 2J 8� *S
fNRL
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_6919512
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
�

�
-__inference_sequential_4_layer_call_fn_693013

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
GPU 2J 8� *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_6929942
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
�a
�
 forward_lstm_4_while_body_694338:
6forward_lstm_4_while_forward_lstm_4_while_loop_counter@
<forward_lstm_4_while_forward_lstm_4_while_maximum_iterations$
 forward_lstm_4_while_placeholder&
"forward_lstm_4_while_placeholder_1&
"forward_lstm_4_while_placeholder_2&
"forward_lstm_4_while_placeholder_3&
"forward_lstm_4_while_placeholder_49
5forward_lstm_4_while_forward_lstm_4_strided_slice_1_0u
qforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_06
2forward_lstm_4_while_greater_forward_lstm_4_cast_0U
Bforward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0:	�W
Dforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0:	2�R
Cforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0:	�!
forward_lstm_4_while_identity#
forward_lstm_4_while_identity_1#
forward_lstm_4_while_identity_2#
forward_lstm_4_while_identity_3#
forward_lstm_4_while_identity_4#
forward_lstm_4_while_identity_5#
forward_lstm_4_while_identity_67
3forward_lstm_4_while_forward_lstm_4_strided_slice_1s
oforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor4
0forward_lstm_4_while_greater_forward_lstm_4_castS
@forward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource:	�U
Bforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource:	2�P
Aforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource:	���8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp�7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp�9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp�
Fforward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2H
Fforward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape�
8forward_lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0 forward_lstm_4_while_placeholderOforward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02:
8forward_lstm_4/while/TensorArrayV2Read/TensorListGetItem�
forward_lstm_4/while/GreaterGreater2forward_lstm_4_while_greater_forward_lstm_4_cast_0 forward_lstm_4_while_placeholder*
T0*#
_output_shapes
:���������2
forward_lstm_4/while/Greater�
7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOpBforward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype029
7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp�
(forward_lstm_4/while/lstm_cell_13/MatMulMatMul?forward_lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0?forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(forward_lstm_4/while/lstm_cell_13/MatMul�
9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOpDforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02;
9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp�
*forward_lstm_4/while/lstm_cell_13/MatMul_1MatMul"forward_lstm_4_while_placeholder_3Aforward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*forward_lstm_4/while/lstm_cell_13/MatMul_1�
%forward_lstm_4/while/lstm_cell_13/addAddV22forward_lstm_4/while/lstm_cell_13/MatMul:product:04forward_lstm_4/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2'
%forward_lstm_4/while/lstm_cell_13/add�
8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOpCforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02:
8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp�
)forward_lstm_4/while/lstm_cell_13/BiasAddBiasAdd)forward_lstm_4/while/lstm_cell_13/add:z:0@forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)forward_lstm_4/while/lstm_cell_13/BiasAdd�
1forward_lstm_4/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1forward_lstm_4/while/lstm_cell_13/split/split_dim�
'forward_lstm_4/while/lstm_cell_13/splitSplit:forward_lstm_4/while/lstm_cell_13/split/split_dim:output:02forward_lstm_4/while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2)
'forward_lstm_4/while/lstm_cell_13/split�
)forward_lstm_4/while/lstm_cell_13/SigmoidSigmoid0forward_lstm_4/while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22+
)forward_lstm_4/while/lstm_cell_13/Sigmoid�
+forward_lstm_4/while/lstm_cell_13/Sigmoid_1Sigmoid0forward_lstm_4/while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22-
+forward_lstm_4/while/lstm_cell_13/Sigmoid_1�
%forward_lstm_4/while/lstm_cell_13/mulMul/forward_lstm_4/while/lstm_cell_13/Sigmoid_1:y:0"forward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22'
%forward_lstm_4/while/lstm_cell_13/mul�
&forward_lstm_4/while/lstm_cell_13/ReluRelu0forward_lstm_4/while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22(
&forward_lstm_4/while/lstm_cell_13/Relu�
'forward_lstm_4/while/lstm_cell_13/mul_1Mul-forward_lstm_4/while/lstm_cell_13/Sigmoid:y:04forward_lstm_4/while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22)
'forward_lstm_4/while/lstm_cell_13/mul_1�
'forward_lstm_4/while/lstm_cell_13/add_1AddV2)forward_lstm_4/while/lstm_cell_13/mul:z:0+forward_lstm_4/while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22)
'forward_lstm_4/while/lstm_cell_13/add_1�
+forward_lstm_4/while/lstm_cell_13/Sigmoid_2Sigmoid0forward_lstm_4/while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22-
+forward_lstm_4/while/lstm_cell_13/Sigmoid_2�
(forward_lstm_4/while/lstm_cell_13/Relu_1Relu+forward_lstm_4/while/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_4/while/lstm_cell_13/Relu_1�
'forward_lstm_4/while/lstm_cell_13/mul_2Mul/forward_lstm_4/while/lstm_cell_13/Sigmoid_2:y:06forward_lstm_4/while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22)
'forward_lstm_4/while/lstm_cell_13/mul_2�
forward_lstm_4/while/SelectSelect forward_lstm_4/while/Greater:z:0+forward_lstm_4/while/lstm_cell_13/mul_2:z:0"forward_lstm_4_while_placeholder_2*
T0*'
_output_shapes
:���������22
forward_lstm_4/while/Select�
forward_lstm_4/while/Select_1Select forward_lstm_4/while/Greater:z:0+forward_lstm_4/while/lstm_cell_13/mul_2:z:0"forward_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:���������22
forward_lstm_4/while/Select_1�
forward_lstm_4/while/Select_2Select forward_lstm_4/while/Greater:z:0+forward_lstm_4/while/lstm_cell_13/add_1:z:0"forward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22
forward_lstm_4/while/Select_2�
9forward_lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"forward_lstm_4_while_placeholder_1 forward_lstm_4_while_placeholder$forward_lstm_4/while/Select:output:0*
_output_shapes
: *
element_dtype02;
9forward_lstm_4/while/TensorArrayV2Write/TensorListSetItemz
forward_lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_4/while/add/y�
forward_lstm_4/while/addAddV2 forward_lstm_4_while_placeholder#forward_lstm_4/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/while/add~
forward_lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_4/while/add_1/y�
forward_lstm_4/while/add_1AddV26forward_lstm_4_while_forward_lstm_4_while_loop_counter%forward_lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/while/add_1�
forward_lstm_4/while/IdentityIdentityforward_lstm_4/while/add_1:z:0^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2
forward_lstm_4/while/Identity�
forward_lstm_4/while/Identity_1Identity<forward_lstm_4_while_forward_lstm_4_while_maximum_iterations^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_4/while/Identity_1�
forward_lstm_4/while/Identity_2Identityforward_lstm_4/while/add:z:0^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_4/while/Identity_2�
forward_lstm_4/while/Identity_3IdentityIforward_lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_4/while/Identity_3�
forward_lstm_4/while/Identity_4Identity$forward_lstm_4/while/Select:output:0^forward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22!
forward_lstm_4/while/Identity_4�
forward_lstm_4/while/Identity_5Identity&forward_lstm_4/while/Select_1:output:0^forward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22!
forward_lstm_4/while/Identity_5�
forward_lstm_4/while/Identity_6Identity&forward_lstm_4/while/Select_2:output:0^forward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22!
forward_lstm_4/while/Identity_6�
forward_lstm_4/while/NoOpNoOp9^forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp8^forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp:^forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_4/while/NoOp"l
3forward_lstm_4_while_forward_lstm_4_strided_slice_15forward_lstm_4_while_forward_lstm_4_strided_slice_1_0"f
0forward_lstm_4_while_greater_forward_lstm_4_cast2forward_lstm_4_while_greater_forward_lstm_4_cast_0"G
forward_lstm_4_while_identity&forward_lstm_4/while/Identity:output:0"K
forward_lstm_4_while_identity_1(forward_lstm_4/while/Identity_1:output:0"K
forward_lstm_4_while_identity_2(forward_lstm_4/while/Identity_2:output:0"K
forward_lstm_4_while_identity_3(forward_lstm_4/while/Identity_3:output:0"K
forward_lstm_4_while_identity_4(forward_lstm_4/while/Identity_4:output:0"K
forward_lstm_4_while_identity_5(forward_lstm_4/while/Identity_5:output:0"K
forward_lstm_4_while_identity_6(forward_lstm_4/while/Identity_6:output:0"�
Aforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resourceCforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0"�
Bforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resourceDforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0"�
@forward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resourceBforward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0"�
oforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensorqforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2t
8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp2r
7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp2v
9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp: 
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
while_cond_696058
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_696058___redundant_placeholder04
0while_while_cond_696058___redundant_placeholder14
0while_while_cond_696058___redundant_placeholder24
0while_while_cond_696058___redundant_placeholder3
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
Ʋ
�
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_693402

inputs
inputs_1	M
:forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource:	�O
<forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource:	2�J
;forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource:	�N
;backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource:	�P
=backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource:	2�K
<backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource:	�
identity��3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp�2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp�4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp�backward_lstm_4/while�2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp�1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp�3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp�forward_lstm_4/while�
#forward_lstm_4/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2%
#forward_lstm_4/RaggedToTensor/zeros�
#forward_lstm_4/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2%
#forward_lstm_4/RaggedToTensor/Const�
2forward_lstm_4/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor,forward_lstm_4/RaggedToTensor/Const:output:0inputs,forward_lstm_4/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS24
2forward_lstm_4/RaggedToTensor/RaggedTensorToTensor�
9forward_lstm_4/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack�
;forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1�
;forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2�
3forward_lstm_4/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Bforward_lstm_4/RaggedNestedRowLengths/strided_slice/stack:output:0Dforward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1:output:0Dforward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask25
3forward_lstm_4/RaggedNestedRowLengths/strided_slice�
;forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack�
=forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2?
=forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1�
=forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2�
5forward_lstm_4/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Dforward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack:output:0Fforward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Fforward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask27
5forward_lstm_4/RaggedNestedRowLengths/strided_slice_1�
)forward_lstm_4/RaggedNestedRowLengths/subSub<forward_lstm_4/RaggedNestedRowLengths/strided_slice:output:0>forward_lstm_4/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2+
)forward_lstm_4/RaggedNestedRowLengths/sub�
forward_lstm_4/CastCast-forward_lstm_4/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
forward_lstm_4/Cast�
forward_lstm_4/ShapeShape;forward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_4/Shape�
"forward_lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"forward_lstm_4/strided_slice/stack�
$forward_lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_4/strided_slice/stack_1�
$forward_lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_4/strided_slice/stack_2�
forward_lstm_4/strided_sliceStridedSliceforward_lstm_4/Shape:output:0+forward_lstm_4/strided_slice/stack:output:0-forward_lstm_4/strided_slice/stack_1:output:0-forward_lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_4/strided_slicez
forward_lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_4/zeros/mul/y�
forward_lstm_4/zeros/mulMul%forward_lstm_4/strided_slice:output:0#forward_lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros/mul}
forward_lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_4/zeros/Less/y�
forward_lstm_4/zeros/LessLessforward_lstm_4/zeros/mul:z:0$forward_lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros/Less�
forward_lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_4/zeros/packed/1�
forward_lstm_4/zeros/packedPack%forward_lstm_4/strided_slice:output:0&forward_lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_4/zeros/packed�
forward_lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_4/zeros/Const�
forward_lstm_4/zerosFill$forward_lstm_4/zeros/packed:output:0#forward_lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_4/zeros~
forward_lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_4/zeros_1/mul/y�
forward_lstm_4/zeros_1/mulMul%forward_lstm_4/strided_slice:output:0%forward_lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros_1/mul�
forward_lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_4/zeros_1/Less/y�
forward_lstm_4/zeros_1/LessLessforward_lstm_4/zeros_1/mul:z:0&forward_lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros_1/Less�
forward_lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_4/zeros_1/packed/1�
forward_lstm_4/zeros_1/packedPack%forward_lstm_4/strided_slice:output:0(forward_lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_4/zeros_1/packed�
forward_lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_4/zeros_1/Const�
forward_lstm_4/zeros_1Fill&forward_lstm_4/zeros_1/packed:output:0%forward_lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_4/zeros_1�
forward_lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
forward_lstm_4/transpose/perm�
forward_lstm_4/transpose	Transpose;forward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0&forward_lstm_4/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
forward_lstm_4/transpose|
forward_lstm_4/Shape_1Shapeforward_lstm_4/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_4/Shape_1�
$forward_lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_4/strided_slice_1/stack�
&forward_lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_1/stack_1�
&forward_lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_1/stack_2�
forward_lstm_4/strided_slice_1StridedSliceforward_lstm_4/Shape_1:output:0-forward_lstm_4/strided_slice_1/stack:output:0/forward_lstm_4/strided_slice_1/stack_1:output:0/forward_lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_4/strided_slice_1�
*forward_lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2,
*forward_lstm_4/TensorArrayV2/element_shape�
forward_lstm_4/TensorArrayV2TensorListReserve3forward_lstm_4/TensorArrayV2/element_shape:output:0'forward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_4/TensorArrayV2�
Dforward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2F
Dforward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape�
6forward_lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_4/transpose:y:0Mforward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6forward_lstm_4/TensorArrayUnstack/TensorListFromTensor�
$forward_lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_4/strided_slice_2/stack�
&forward_lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_2/stack_1�
&forward_lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_2/stack_2�
forward_lstm_4/strided_slice_2StridedSliceforward_lstm_4/transpose:y:0-forward_lstm_4/strided_slice_2/stack:output:0/forward_lstm_4/strided_slice_2/stack_1:output:0/forward_lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2 
forward_lstm_4/strided_slice_2�
1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp:forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype023
1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp�
"forward_lstm_4/lstm_cell_13/MatMulMatMul'forward_lstm_4/strided_slice_2:output:09forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"forward_lstm_4/lstm_cell_13/MatMul�
3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp<forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype025
3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp�
$forward_lstm_4/lstm_cell_13/MatMul_1MatMulforward_lstm_4/zeros:output:0;forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$forward_lstm_4/lstm_cell_13/MatMul_1�
forward_lstm_4/lstm_cell_13/addAddV2,forward_lstm_4/lstm_cell_13/MatMul:product:0.forward_lstm_4/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2!
forward_lstm_4/lstm_cell_13/add�
2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp;forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype024
2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp�
#forward_lstm_4/lstm_cell_13/BiasAddBiasAdd#forward_lstm_4/lstm_cell_13/add:z:0:forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#forward_lstm_4/lstm_cell_13/BiasAdd�
+forward_lstm_4/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+forward_lstm_4/lstm_cell_13/split/split_dim�
!forward_lstm_4/lstm_cell_13/splitSplit4forward_lstm_4/lstm_cell_13/split/split_dim:output:0,forward_lstm_4/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2#
!forward_lstm_4/lstm_cell_13/split�
#forward_lstm_4/lstm_cell_13/SigmoidSigmoid*forward_lstm_4/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22%
#forward_lstm_4/lstm_cell_13/Sigmoid�
%forward_lstm_4/lstm_cell_13/Sigmoid_1Sigmoid*forward_lstm_4/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22'
%forward_lstm_4/lstm_cell_13/Sigmoid_1�
forward_lstm_4/lstm_cell_13/mulMul)forward_lstm_4/lstm_cell_13/Sigmoid_1:y:0forward_lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:���������22!
forward_lstm_4/lstm_cell_13/mul�
 forward_lstm_4/lstm_cell_13/ReluRelu*forward_lstm_4/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22"
 forward_lstm_4/lstm_cell_13/Relu�
!forward_lstm_4/lstm_cell_13/mul_1Mul'forward_lstm_4/lstm_cell_13/Sigmoid:y:0.forward_lstm_4/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22#
!forward_lstm_4/lstm_cell_13/mul_1�
!forward_lstm_4/lstm_cell_13/add_1AddV2#forward_lstm_4/lstm_cell_13/mul:z:0%forward_lstm_4/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22#
!forward_lstm_4/lstm_cell_13/add_1�
%forward_lstm_4/lstm_cell_13/Sigmoid_2Sigmoid*forward_lstm_4/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22'
%forward_lstm_4/lstm_cell_13/Sigmoid_2�
"forward_lstm_4/lstm_cell_13/Relu_1Relu%forward_lstm_4/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_4/lstm_cell_13/Relu_1�
!forward_lstm_4/lstm_cell_13/mul_2Mul)forward_lstm_4/lstm_cell_13/Sigmoid_2:y:00forward_lstm_4/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22#
!forward_lstm_4/lstm_cell_13/mul_2�
,forward_lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2.
,forward_lstm_4/TensorArrayV2_1/element_shape�
forward_lstm_4/TensorArrayV2_1TensorListReserve5forward_lstm_4/TensorArrayV2_1/element_shape:output:0'forward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_4/TensorArrayV2_1l
forward_lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_4/time�
forward_lstm_4/zeros_like	ZerosLike%forward_lstm_4/lstm_cell_13/mul_2:z:0*
T0*'
_output_shapes
:���������22
forward_lstm_4/zeros_like�
'forward_lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2)
'forward_lstm_4/while/maximum_iterations�
!forward_lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!forward_lstm_4/while/loop_counter�
forward_lstm_4/whileWhile*forward_lstm_4/while/loop_counter:output:00forward_lstm_4/while/maximum_iterations:output:0forward_lstm_4/time:output:0'forward_lstm_4/TensorArrayV2_1:handle:0forward_lstm_4/zeros_like:y:0forward_lstm_4/zeros:output:0forward_lstm_4/zeros_1:output:0'forward_lstm_4/strided_slice_1:output:0Fforward_lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_4/Cast:y:0:forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource<forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource;forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *,
body$R"
 forward_lstm_4_while_body_693126*,
cond$R"
 forward_lstm_4_while_cond_693125*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
forward_lstm_4/while�
?forward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2A
?forward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape�
1forward_lstm_4/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_4/while:output:3Hforward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype023
1forward_lstm_4/TensorArrayV2Stack/TensorListStack�
$forward_lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2&
$forward_lstm_4/strided_slice_3/stack�
&forward_lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_4/strided_slice_3/stack_1�
&forward_lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_3/stack_2�
forward_lstm_4/strided_slice_3StridedSlice:forward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0-forward_lstm_4/strided_slice_3/stack:output:0/forward_lstm_4/strided_slice_3/stack_1:output:0/forward_lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2 
forward_lstm_4/strided_slice_3�
forward_lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_4/transpose_1/perm�
forward_lstm_4/transpose_1	Transpose:forward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0(forward_lstm_4/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
forward_lstm_4/transpose_1�
forward_lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_4/runtime�
$backward_lstm_4/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$backward_lstm_4/RaggedToTensor/zeros�
$backward_lstm_4/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2&
$backward_lstm_4/RaggedToTensor/Const�
3backward_lstm_4/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-backward_lstm_4/RaggedToTensor/Const:output:0inputs-backward_lstm_4/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3backward_lstm_4/RaggedToTensor/RaggedTensorToTensor�
:backward_lstm_4/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack�
<backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1�
<backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2�
4backward_lstm_4/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cbackward_lstm_4/RaggedNestedRowLengths/strided_slice/stack:output:0Ebackward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1:output:0Ebackward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask26
4backward_lstm_4/RaggedNestedRowLengths/strided_slice�
<backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack�
>backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2@
>backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1�
>backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2�
6backward_lstm_4/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Ebackward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gbackward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gbackward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask28
6backward_lstm_4/RaggedNestedRowLengths/strided_slice_1�
*backward_lstm_4/RaggedNestedRowLengths/subSub=backward_lstm_4/RaggedNestedRowLengths/strided_slice:output:0?backward_lstm_4/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2,
*backward_lstm_4/RaggedNestedRowLengths/sub�
backward_lstm_4/CastCast.backward_lstm_4/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
backward_lstm_4/Cast�
backward_lstm_4/ShapeShape<backward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_4/Shape�
#backward_lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#backward_lstm_4/strided_slice/stack�
%backward_lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_4/strided_slice/stack_1�
%backward_lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_4/strided_slice/stack_2�
backward_lstm_4/strided_sliceStridedSlicebackward_lstm_4/Shape:output:0,backward_lstm_4/strided_slice/stack:output:0.backward_lstm_4/strided_slice/stack_1:output:0.backward_lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
backward_lstm_4/strided_slice|
backward_lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_4/zeros/mul/y�
backward_lstm_4/zeros/mulMul&backward_lstm_4/strided_slice:output:0$backward_lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros/mul
backward_lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
backward_lstm_4/zeros/Less/y�
backward_lstm_4/zeros/LessLessbackward_lstm_4/zeros/mul:z:0%backward_lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros/Less�
backward_lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_4/zeros/packed/1�
backward_lstm_4/zeros/packedPack&backward_lstm_4/strided_slice:output:0'backward_lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_4/zeros/packed�
backward_lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_4/zeros/Const�
backward_lstm_4/zerosFill%backward_lstm_4/zeros/packed:output:0$backward_lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_4/zeros�
backward_lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_4/zeros_1/mul/y�
backward_lstm_4/zeros_1/mulMul&backward_lstm_4/strided_slice:output:0&backward_lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros_1/mul�
backward_lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
backward_lstm_4/zeros_1/Less/y�
backward_lstm_4/zeros_1/LessLessbackward_lstm_4/zeros_1/mul:z:0'backward_lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros_1/Less�
 backward_lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_4/zeros_1/packed/1�
backward_lstm_4/zeros_1/packedPack&backward_lstm_4/strided_slice:output:0)backward_lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_4/zeros_1/packed�
backward_lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_4/zeros_1/Const�
backward_lstm_4/zeros_1Fill'backward_lstm_4/zeros_1/packed:output:0&backward_lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_4/zeros_1�
backward_lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
backward_lstm_4/transpose/perm�
backward_lstm_4/transpose	Transpose<backward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0'backward_lstm_4/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_4/transpose
backward_lstm_4/Shape_1Shapebackward_lstm_4/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_4/Shape_1�
%backward_lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_4/strided_slice_1/stack�
'backward_lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_1/stack_1�
'backward_lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_1/stack_2�
backward_lstm_4/strided_slice_1StridedSlice backward_lstm_4/Shape_1:output:0.backward_lstm_4/strided_slice_1/stack:output:00backward_lstm_4/strided_slice_1/stack_1:output:00backward_lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_4/strided_slice_1�
+backward_lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+backward_lstm_4/TensorArrayV2/element_shape�
backward_lstm_4/TensorArrayV2TensorListReserve4backward_lstm_4/TensorArrayV2/element_shape:output:0(backward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
backward_lstm_4/TensorArrayV2�
backward_lstm_4/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2 
backward_lstm_4/ReverseV2/axis�
backward_lstm_4/ReverseV2	ReverseV2backward_lstm_4/transpose:y:0'backward_lstm_4/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_4/ReverseV2�
Ebackward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2G
Ebackward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape�
7backward_lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"backward_lstm_4/ReverseV2:output:0Nbackward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7backward_lstm_4/TensorArrayUnstack/TensorListFromTensor�
%backward_lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_4/strided_slice_2/stack�
'backward_lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_2/stack_1�
'backward_lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_2/stack_2�
backward_lstm_4/strided_slice_2StridedSlicebackward_lstm_4/transpose:y:0.backward_lstm_4/strided_slice_2/stack:output:00backward_lstm_4/strided_slice_2/stack_1:output:00backward_lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2!
backward_lstm_4/strided_slice_2�
2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp;backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp�
#backward_lstm_4/lstm_cell_14/MatMulMatMul(backward_lstm_4/strided_slice_2:output:0:backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#backward_lstm_4/lstm_cell_14/MatMul�
4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp=backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype026
4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp�
%backward_lstm_4/lstm_cell_14/MatMul_1MatMulbackward_lstm_4/zeros:output:0<backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%backward_lstm_4/lstm_cell_14/MatMul_1�
 backward_lstm_4/lstm_cell_14/addAddV2-backward_lstm_4/lstm_cell_14/MatMul:product:0/backward_lstm_4/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2"
 backward_lstm_4/lstm_cell_14/add�
3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp<backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp�
$backward_lstm_4/lstm_cell_14/BiasAddBiasAdd$backward_lstm_4/lstm_cell_14/add:z:0;backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$backward_lstm_4/lstm_cell_14/BiasAdd�
,backward_lstm_4/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,backward_lstm_4/lstm_cell_14/split/split_dim�
"backward_lstm_4/lstm_cell_14/splitSplit5backward_lstm_4/lstm_cell_14/split/split_dim:output:0-backward_lstm_4/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2$
"backward_lstm_4/lstm_cell_14/split�
$backward_lstm_4/lstm_cell_14/SigmoidSigmoid+backward_lstm_4/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22&
$backward_lstm_4/lstm_cell_14/Sigmoid�
&backward_lstm_4/lstm_cell_14/Sigmoid_1Sigmoid+backward_lstm_4/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22(
&backward_lstm_4/lstm_cell_14/Sigmoid_1�
 backward_lstm_4/lstm_cell_14/mulMul*backward_lstm_4/lstm_cell_14/Sigmoid_1:y:0 backward_lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/lstm_cell_14/mul�
!backward_lstm_4/lstm_cell_14/ReluRelu+backward_lstm_4/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22#
!backward_lstm_4/lstm_cell_14/Relu�
"backward_lstm_4/lstm_cell_14/mul_1Mul(backward_lstm_4/lstm_cell_14/Sigmoid:y:0/backward_lstm_4/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22$
"backward_lstm_4/lstm_cell_14/mul_1�
"backward_lstm_4/lstm_cell_14/add_1AddV2$backward_lstm_4/lstm_cell_14/mul:z:0&backward_lstm_4/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22$
"backward_lstm_4/lstm_cell_14/add_1�
&backward_lstm_4/lstm_cell_14/Sigmoid_2Sigmoid+backward_lstm_4/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22(
&backward_lstm_4/lstm_cell_14/Sigmoid_2�
#backward_lstm_4/lstm_cell_14/Relu_1Relu&backward_lstm_4/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_4/lstm_cell_14/Relu_1�
"backward_lstm_4/lstm_cell_14/mul_2Mul*backward_lstm_4/lstm_cell_14/Sigmoid_2:y:01backward_lstm_4/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22$
"backward_lstm_4/lstm_cell_14/mul_2�
-backward_lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2/
-backward_lstm_4/TensorArrayV2_1/element_shape�
backward_lstm_4/TensorArrayV2_1TensorListReserve6backward_lstm_4/TensorArrayV2_1/element_shape:output:0(backward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_4/TensorArrayV2_1n
backward_lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_4/time�
%backward_lstm_4/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2'
%backward_lstm_4/Max/reduction_indices�
backward_lstm_4/MaxMaxbackward_lstm_4/Cast:y:0.backward_lstm_4/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/Maxp
backward_lstm_4/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_4/sub/y�
backward_lstm_4/subSubbackward_lstm_4/Max:output:0backward_lstm_4/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/sub�
backward_lstm_4/Sub_1Subbackward_lstm_4/sub:z:0backward_lstm_4/Cast:y:0*
T0*#
_output_shapes
:���������2
backward_lstm_4/Sub_1�
backward_lstm_4/zeros_like	ZerosLike&backward_lstm_4/lstm_cell_14/mul_2:z:0*
T0*'
_output_shapes
:���������22
backward_lstm_4/zeros_like�
(backward_lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(backward_lstm_4/while/maximum_iterations�
"backward_lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"backward_lstm_4/while/loop_counter�
backward_lstm_4/whileWhile+backward_lstm_4/while/loop_counter:output:01backward_lstm_4/while/maximum_iterations:output:0backward_lstm_4/time:output:0(backward_lstm_4/TensorArrayV2_1:handle:0backward_lstm_4/zeros_like:y:0backward_lstm_4/zeros:output:0 backward_lstm_4/zeros_1:output:0(backward_lstm_4/strided_slice_1:output:0Gbackward_lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_4/Sub_1:z:0;backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource=backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource<backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *-
body%R#
!backward_lstm_4_while_body_693305*-
cond%R#
!backward_lstm_4_while_cond_693304*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
backward_lstm_4/while�
@backward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2B
@backward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape�
2backward_lstm_4/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_4/while:output:3Ibackward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype024
2backward_lstm_4/TensorArrayV2Stack/TensorListStack�
%backward_lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2'
%backward_lstm_4/strided_slice_3/stack�
'backward_lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_4/strided_slice_3/stack_1�
'backward_lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_3/stack_2�
backward_lstm_4/strided_slice_3StridedSlice;backward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0.backward_lstm_4/strided_slice_3/stack:output:00backward_lstm_4/strided_slice_3/stack_1:output:00backward_lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2!
backward_lstm_4/strided_slice_3�
 backward_lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_4/transpose_1/perm�
backward_lstm_4/transpose_1	Transpose;backward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0)backward_lstm_4/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
backward_lstm_4/transpose_1�
backward_lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_4/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2'forward_lstm_4/strided_slice_3:output:0(backward_lstm_4/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp4^backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp3^backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp5^backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp^backward_lstm_4/while3^forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp2^forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp4^forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp^forward_lstm_4/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������: : : : : : 2j
3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp2h
2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp2l
4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp2.
backward_lstm_4/whilebackward_lstm_4/while2h
2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp2f
1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp2j
3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp2,
forward_lstm_4/whileforward_lstm_4/while:O K
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
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_696296

inputs>
+lstm_cell_14_matmul_readvariableop_resource:	�@
-lstm_cell_14_matmul_1_readvariableop_resource:	2�;
,lstm_cell_14_biasadd_readvariableop_resource:	�
identity��#lstm_cell_14/BiasAdd/ReadVariableOp�"lstm_cell_14/MatMul/ReadVariableOp�$lstm_cell_14/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp�
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/MatMul�
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp�
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/MatMul_1�
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/add�
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp�
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/BiasAdd~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim�
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_14/split�
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/Sigmoid�
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_14/Sigmoid_1�
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_14/Relu�
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/mul_1�
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/add_1�
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/Relu_1�
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
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
bodyR
while_body_696212*
condR
while_cond_696211*K
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
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
 forward_lstm_4_while_cond_694337:
6forward_lstm_4_while_forward_lstm_4_while_loop_counter@
<forward_lstm_4_while_forward_lstm_4_while_maximum_iterations$
 forward_lstm_4_while_placeholder&
"forward_lstm_4_while_placeholder_1&
"forward_lstm_4_while_placeholder_2&
"forward_lstm_4_while_placeholder_3&
"forward_lstm_4_while_placeholder_4<
8forward_lstm_4_while_less_forward_lstm_4_strided_slice_1R
Nforward_lstm_4_while_forward_lstm_4_while_cond_694337___redundant_placeholder0R
Nforward_lstm_4_while_forward_lstm_4_while_cond_694337___redundant_placeholder1R
Nforward_lstm_4_while_forward_lstm_4_while_cond_694337___redundant_placeholder2R
Nforward_lstm_4_while_forward_lstm_4_while_cond_694337___redundant_placeholder3R
Nforward_lstm_4_while_forward_lstm_4_while_cond_694337___redundant_placeholder4!
forward_lstm_4_while_identity
�
forward_lstm_4/while/LessLess forward_lstm_4_while_placeholder8forward_lstm_4_while_less_forward_lstm_4_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_4/while/Less�
forward_lstm_4/while/IdentityIdentityforward_lstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2
forward_lstm_4/while/Identity"G
forward_lstm_4_while_identity&forward_lstm_4/while/Identity:output:0*(
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
��
�
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_693954
inputs_0M
:forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource:	�O
<forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource:	2�J
;forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource:	�N
;backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource:	�P
=backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource:	2�K
<backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource:	�
identity��3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp�2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp�4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp�backward_lstm_4/while�2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp�1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp�3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp�forward_lstm_4/whiled
forward_lstm_4/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_4/Shape�
"forward_lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"forward_lstm_4/strided_slice/stack�
$forward_lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_4/strided_slice/stack_1�
$forward_lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_4/strided_slice/stack_2�
forward_lstm_4/strided_sliceStridedSliceforward_lstm_4/Shape:output:0+forward_lstm_4/strided_slice/stack:output:0-forward_lstm_4/strided_slice/stack_1:output:0-forward_lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_4/strided_slicez
forward_lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_4/zeros/mul/y�
forward_lstm_4/zeros/mulMul%forward_lstm_4/strided_slice:output:0#forward_lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros/mul}
forward_lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_4/zeros/Less/y�
forward_lstm_4/zeros/LessLessforward_lstm_4/zeros/mul:z:0$forward_lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros/Less�
forward_lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_4/zeros/packed/1�
forward_lstm_4/zeros/packedPack%forward_lstm_4/strided_slice:output:0&forward_lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_4/zeros/packed�
forward_lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_4/zeros/Const�
forward_lstm_4/zerosFill$forward_lstm_4/zeros/packed:output:0#forward_lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_4/zeros~
forward_lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_4/zeros_1/mul/y�
forward_lstm_4/zeros_1/mulMul%forward_lstm_4/strided_slice:output:0%forward_lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros_1/mul�
forward_lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_4/zeros_1/Less/y�
forward_lstm_4/zeros_1/LessLessforward_lstm_4/zeros_1/mul:z:0&forward_lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros_1/Less�
forward_lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_4/zeros_1/packed/1�
forward_lstm_4/zeros_1/packedPack%forward_lstm_4/strided_slice:output:0(forward_lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_4/zeros_1/packed�
forward_lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_4/zeros_1/Const�
forward_lstm_4/zeros_1Fill&forward_lstm_4/zeros_1/packed:output:0%forward_lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_4/zeros_1�
forward_lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
forward_lstm_4/transpose/perm�
forward_lstm_4/transpose	Transposeinputs_0&forward_lstm_4/transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
forward_lstm_4/transpose|
forward_lstm_4/Shape_1Shapeforward_lstm_4/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_4/Shape_1�
$forward_lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_4/strided_slice_1/stack�
&forward_lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_1/stack_1�
&forward_lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_1/stack_2�
forward_lstm_4/strided_slice_1StridedSliceforward_lstm_4/Shape_1:output:0-forward_lstm_4/strided_slice_1/stack:output:0/forward_lstm_4/strided_slice_1/stack_1:output:0/forward_lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_4/strided_slice_1�
*forward_lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2,
*forward_lstm_4/TensorArrayV2/element_shape�
forward_lstm_4/TensorArrayV2TensorListReserve3forward_lstm_4/TensorArrayV2/element_shape:output:0'forward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_4/TensorArrayV2�
Dforward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2F
Dforward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape�
6forward_lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_4/transpose:y:0Mforward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6forward_lstm_4/TensorArrayUnstack/TensorListFromTensor�
$forward_lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_4/strided_slice_2/stack�
&forward_lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_2/stack_1�
&forward_lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_2/stack_2�
forward_lstm_4/strided_slice_2StridedSliceforward_lstm_4/transpose:y:0-forward_lstm_4/strided_slice_2/stack:output:0/forward_lstm_4/strided_slice_2/stack_1:output:0/forward_lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2 
forward_lstm_4/strided_slice_2�
1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp:forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype023
1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp�
"forward_lstm_4/lstm_cell_13/MatMulMatMul'forward_lstm_4/strided_slice_2:output:09forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"forward_lstm_4/lstm_cell_13/MatMul�
3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp<forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype025
3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp�
$forward_lstm_4/lstm_cell_13/MatMul_1MatMulforward_lstm_4/zeros:output:0;forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$forward_lstm_4/lstm_cell_13/MatMul_1�
forward_lstm_4/lstm_cell_13/addAddV2,forward_lstm_4/lstm_cell_13/MatMul:product:0.forward_lstm_4/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2!
forward_lstm_4/lstm_cell_13/add�
2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp;forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype024
2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp�
#forward_lstm_4/lstm_cell_13/BiasAddBiasAdd#forward_lstm_4/lstm_cell_13/add:z:0:forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#forward_lstm_4/lstm_cell_13/BiasAdd�
+forward_lstm_4/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+forward_lstm_4/lstm_cell_13/split/split_dim�
!forward_lstm_4/lstm_cell_13/splitSplit4forward_lstm_4/lstm_cell_13/split/split_dim:output:0,forward_lstm_4/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2#
!forward_lstm_4/lstm_cell_13/split�
#forward_lstm_4/lstm_cell_13/SigmoidSigmoid*forward_lstm_4/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22%
#forward_lstm_4/lstm_cell_13/Sigmoid�
%forward_lstm_4/lstm_cell_13/Sigmoid_1Sigmoid*forward_lstm_4/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22'
%forward_lstm_4/lstm_cell_13/Sigmoid_1�
forward_lstm_4/lstm_cell_13/mulMul)forward_lstm_4/lstm_cell_13/Sigmoid_1:y:0forward_lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:���������22!
forward_lstm_4/lstm_cell_13/mul�
 forward_lstm_4/lstm_cell_13/ReluRelu*forward_lstm_4/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22"
 forward_lstm_4/lstm_cell_13/Relu�
!forward_lstm_4/lstm_cell_13/mul_1Mul'forward_lstm_4/lstm_cell_13/Sigmoid:y:0.forward_lstm_4/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22#
!forward_lstm_4/lstm_cell_13/mul_1�
!forward_lstm_4/lstm_cell_13/add_1AddV2#forward_lstm_4/lstm_cell_13/mul:z:0%forward_lstm_4/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22#
!forward_lstm_4/lstm_cell_13/add_1�
%forward_lstm_4/lstm_cell_13/Sigmoid_2Sigmoid*forward_lstm_4/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22'
%forward_lstm_4/lstm_cell_13/Sigmoid_2�
"forward_lstm_4/lstm_cell_13/Relu_1Relu%forward_lstm_4/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_4/lstm_cell_13/Relu_1�
!forward_lstm_4/lstm_cell_13/mul_2Mul)forward_lstm_4/lstm_cell_13/Sigmoid_2:y:00forward_lstm_4/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22#
!forward_lstm_4/lstm_cell_13/mul_2�
,forward_lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2.
,forward_lstm_4/TensorArrayV2_1/element_shape�
forward_lstm_4/TensorArrayV2_1TensorListReserve5forward_lstm_4/TensorArrayV2_1/element_shape:output:0'forward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_4/TensorArrayV2_1l
forward_lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_4/time�
'forward_lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2)
'forward_lstm_4/while/maximum_iterations�
!forward_lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!forward_lstm_4/while/loop_counter�
forward_lstm_4/whileWhile*forward_lstm_4/while/loop_counter:output:00forward_lstm_4/while/maximum_iterations:output:0forward_lstm_4/time:output:0'forward_lstm_4/TensorArrayV2_1:handle:0forward_lstm_4/zeros:output:0forward_lstm_4/zeros_1:output:0'forward_lstm_4/strided_slice_1:output:0Fforward_lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0:forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource<forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource;forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *,
body$R"
 forward_lstm_4_while_body_693719*,
cond$R"
 forward_lstm_4_while_cond_693718*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
forward_lstm_4/while�
?forward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2A
?forward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape�
1forward_lstm_4/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_4/while:output:3Hforward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype023
1forward_lstm_4/TensorArrayV2Stack/TensorListStack�
$forward_lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2&
$forward_lstm_4/strided_slice_3/stack�
&forward_lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_4/strided_slice_3/stack_1�
&forward_lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_3/stack_2�
forward_lstm_4/strided_slice_3StridedSlice:forward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0-forward_lstm_4/strided_slice_3/stack:output:0/forward_lstm_4/strided_slice_3/stack_1:output:0/forward_lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2 
forward_lstm_4/strided_slice_3�
forward_lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_4/transpose_1/perm�
forward_lstm_4/transpose_1	Transpose:forward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0(forward_lstm_4/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
forward_lstm_4/transpose_1�
forward_lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_4/runtimef
backward_lstm_4/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_4/Shape�
#backward_lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#backward_lstm_4/strided_slice/stack�
%backward_lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_4/strided_slice/stack_1�
%backward_lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_4/strided_slice/stack_2�
backward_lstm_4/strided_sliceStridedSlicebackward_lstm_4/Shape:output:0,backward_lstm_4/strided_slice/stack:output:0.backward_lstm_4/strided_slice/stack_1:output:0.backward_lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
backward_lstm_4/strided_slice|
backward_lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_4/zeros/mul/y�
backward_lstm_4/zeros/mulMul&backward_lstm_4/strided_slice:output:0$backward_lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros/mul
backward_lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
backward_lstm_4/zeros/Less/y�
backward_lstm_4/zeros/LessLessbackward_lstm_4/zeros/mul:z:0%backward_lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros/Less�
backward_lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_4/zeros/packed/1�
backward_lstm_4/zeros/packedPack&backward_lstm_4/strided_slice:output:0'backward_lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_4/zeros/packed�
backward_lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_4/zeros/Const�
backward_lstm_4/zerosFill%backward_lstm_4/zeros/packed:output:0$backward_lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_4/zeros�
backward_lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_4/zeros_1/mul/y�
backward_lstm_4/zeros_1/mulMul&backward_lstm_4/strided_slice:output:0&backward_lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros_1/mul�
backward_lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
backward_lstm_4/zeros_1/Less/y�
backward_lstm_4/zeros_1/LessLessbackward_lstm_4/zeros_1/mul:z:0'backward_lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros_1/Less�
 backward_lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_4/zeros_1/packed/1�
backward_lstm_4/zeros_1/packedPack&backward_lstm_4/strided_slice:output:0)backward_lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_4/zeros_1/packed�
backward_lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_4/zeros_1/Const�
backward_lstm_4/zeros_1Fill'backward_lstm_4/zeros_1/packed:output:0&backward_lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_4/zeros_1�
backward_lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
backward_lstm_4/transpose/perm�
backward_lstm_4/transpose	Transposeinputs_0'backward_lstm_4/transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
backward_lstm_4/transpose
backward_lstm_4/Shape_1Shapebackward_lstm_4/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_4/Shape_1�
%backward_lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_4/strided_slice_1/stack�
'backward_lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_1/stack_1�
'backward_lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_1/stack_2�
backward_lstm_4/strided_slice_1StridedSlice backward_lstm_4/Shape_1:output:0.backward_lstm_4/strided_slice_1/stack:output:00backward_lstm_4/strided_slice_1/stack_1:output:00backward_lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_4/strided_slice_1�
+backward_lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+backward_lstm_4/TensorArrayV2/element_shape�
backward_lstm_4/TensorArrayV2TensorListReserve4backward_lstm_4/TensorArrayV2/element_shape:output:0(backward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
backward_lstm_4/TensorArrayV2�
backward_lstm_4/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2 
backward_lstm_4/ReverseV2/axis�
backward_lstm_4/ReverseV2	ReverseV2backward_lstm_4/transpose:y:0'backward_lstm_4/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'���������������������������2
backward_lstm_4/ReverseV2�
Ebackward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2G
Ebackward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape�
7backward_lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"backward_lstm_4/ReverseV2:output:0Nbackward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7backward_lstm_4/TensorArrayUnstack/TensorListFromTensor�
%backward_lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_4/strided_slice_2/stack�
'backward_lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_2/stack_1�
'backward_lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_2/stack_2�
backward_lstm_4/strided_slice_2StridedSlicebackward_lstm_4/transpose:y:0.backward_lstm_4/strided_slice_2/stack:output:00backward_lstm_4/strided_slice_2/stack_1:output:00backward_lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2!
backward_lstm_4/strided_slice_2�
2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp;backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp�
#backward_lstm_4/lstm_cell_14/MatMulMatMul(backward_lstm_4/strided_slice_2:output:0:backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#backward_lstm_4/lstm_cell_14/MatMul�
4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp=backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype026
4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp�
%backward_lstm_4/lstm_cell_14/MatMul_1MatMulbackward_lstm_4/zeros:output:0<backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%backward_lstm_4/lstm_cell_14/MatMul_1�
 backward_lstm_4/lstm_cell_14/addAddV2-backward_lstm_4/lstm_cell_14/MatMul:product:0/backward_lstm_4/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2"
 backward_lstm_4/lstm_cell_14/add�
3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp<backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp�
$backward_lstm_4/lstm_cell_14/BiasAddBiasAdd$backward_lstm_4/lstm_cell_14/add:z:0;backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$backward_lstm_4/lstm_cell_14/BiasAdd�
,backward_lstm_4/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,backward_lstm_4/lstm_cell_14/split/split_dim�
"backward_lstm_4/lstm_cell_14/splitSplit5backward_lstm_4/lstm_cell_14/split/split_dim:output:0-backward_lstm_4/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2$
"backward_lstm_4/lstm_cell_14/split�
$backward_lstm_4/lstm_cell_14/SigmoidSigmoid+backward_lstm_4/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22&
$backward_lstm_4/lstm_cell_14/Sigmoid�
&backward_lstm_4/lstm_cell_14/Sigmoid_1Sigmoid+backward_lstm_4/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22(
&backward_lstm_4/lstm_cell_14/Sigmoid_1�
 backward_lstm_4/lstm_cell_14/mulMul*backward_lstm_4/lstm_cell_14/Sigmoid_1:y:0 backward_lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/lstm_cell_14/mul�
!backward_lstm_4/lstm_cell_14/ReluRelu+backward_lstm_4/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22#
!backward_lstm_4/lstm_cell_14/Relu�
"backward_lstm_4/lstm_cell_14/mul_1Mul(backward_lstm_4/lstm_cell_14/Sigmoid:y:0/backward_lstm_4/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22$
"backward_lstm_4/lstm_cell_14/mul_1�
"backward_lstm_4/lstm_cell_14/add_1AddV2$backward_lstm_4/lstm_cell_14/mul:z:0&backward_lstm_4/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22$
"backward_lstm_4/lstm_cell_14/add_1�
&backward_lstm_4/lstm_cell_14/Sigmoid_2Sigmoid+backward_lstm_4/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22(
&backward_lstm_4/lstm_cell_14/Sigmoid_2�
#backward_lstm_4/lstm_cell_14/Relu_1Relu&backward_lstm_4/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_4/lstm_cell_14/Relu_1�
"backward_lstm_4/lstm_cell_14/mul_2Mul*backward_lstm_4/lstm_cell_14/Sigmoid_2:y:01backward_lstm_4/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22$
"backward_lstm_4/lstm_cell_14/mul_2�
-backward_lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2/
-backward_lstm_4/TensorArrayV2_1/element_shape�
backward_lstm_4/TensorArrayV2_1TensorListReserve6backward_lstm_4/TensorArrayV2_1/element_shape:output:0(backward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_4/TensorArrayV2_1n
backward_lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_4/time�
(backward_lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(backward_lstm_4/while/maximum_iterations�
"backward_lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"backward_lstm_4/while/loop_counter�
backward_lstm_4/whileWhile+backward_lstm_4/while/loop_counter:output:01backward_lstm_4/while/maximum_iterations:output:0backward_lstm_4/time:output:0(backward_lstm_4/TensorArrayV2_1:handle:0backward_lstm_4/zeros:output:0 backward_lstm_4/zeros_1:output:0(backward_lstm_4/strided_slice_1:output:0Gbackward_lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0;backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource=backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource<backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *-
body%R#
!backward_lstm_4_while_body_693868*-
cond%R#
!backward_lstm_4_while_cond_693867*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
backward_lstm_4/while�
@backward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2B
@backward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape�
2backward_lstm_4/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_4/while:output:3Ibackward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype024
2backward_lstm_4/TensorArrayV2Stack/TensorListStack�
%backward_lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2'
%backward_lstm_4/strided_slice_3/stack�
'backward_lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_4/strided_slice_3/stack_1�
'backward_lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_3/stack_2�
backward_lstm_4/strided_slice_3StridedSlice;backward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0.backward_lstm_4/strided_slice_3/stack:output:00backward_lstm_4/strided_slice_3/stack_1:output:00backward_lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2!
backward_lstm_4/strided_slice_3�
 backward_lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_4/transpose_1/perm�
backward_lstm_4/transpose_1	Transpose;backward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0)backward_lstm_4/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
backward_lstm_4/transpose_1�
backward_lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_4/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2'forward_lstm_4/strided_slice_3:output:0(backward_lstm_4/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp4^backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp3^backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp5^backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp^backward_lstm_4/while3^forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp2^forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp4^forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp^forward_lstm_4/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 2j
3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp2h
2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp2l
4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp2.
backward_lstm_4/whilebackward_lstm_4/while2h
2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp2f
1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp2j
3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp2,
forward_lstm_4/whileforward_lstm_4/while:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs/0
�?
�
while_body_695906
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_14_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_14_matmul_readvariableop_resource:	�F
3while_lstm_cell_14_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_14_biasadd_readvariableop_resource:	���)while/lstm_cell_14/BiasAdd/ReadVariableOp�(while/lstm_cell_14/MatMul/ReadVariableOp�*while/lstm_cell_14/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp�
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/MatMul�
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp�
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/MatMul_1�
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/add�
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp�
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/BiasAdd�
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim�
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_14/split�
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Sigmoid�
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Sigmoid_1�
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/mul�
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Relu�
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/mul_1�
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/add_1�
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Sigmoid_2�
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Relu_1�
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
��
�
!__inference__wrapped_model_690526

args_0
args_0_1	j
Wsequential_4_bidirectional_4_forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource:	�l
Ysequential_4_bidirectional_4_forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource:	2�g
Xsequential_4_bidirectional_4_forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource:	�k
Xsequential_4_bidirectional_4_backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource:	�m
Zsequential_4_bidirectional_4_backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource:	2�h
Ysequential_4_bidirectional_4_backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource:	�E
3sequential_4_dense_4_matmul_readvariableop_resource:dB
4sequential_4_dense_4_biasadd_readvariableop_resource:
identity��Psequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp�Osequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp�Qsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp�2sequential_4/bidirectional_4/backward_lstm_4/while�Osequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp�Nsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp�Psequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp�1sequential_4/bidirectional_4/forward_lstm_4/while�+sequential_4/dense_4/BiasAdd/ReadVariableOp�*sequential_4/dense_4/MatMul/ReadVariableOp�
@sequential_4/bidirectional_4/forward_lstm_4/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2B
@sequential_4/bidirectional_4/forward_lstm_4/RaggedToTensor/zeros�
@sequential_4/bidirectional_4/forward_lstm_4/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2B
@sequential_4/bidirectional_4/forward_lstm_4/RaggedToTensor/Const�
Osequential_4/bidirectional_4/forward_lstm_4/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorIsequential_4/bidirectional_4/forward_lstm_4/RaggedToTensor/Const:output:0args_0Isequential_4/bidirectional_4/forward_lstm_4/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2Q
Osequential_4/bidirectional_4/forward_lstm_4/RaggedToTensor/RaggedTensorToTensor�
Vsequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2X
Vsequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack�
Xsequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Z
Xsequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1�
Xsequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Z
Xsequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2�
Psequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1_sequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack:output:0asequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1:output:0asequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask2R
Psequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice�
Xsequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Z
Xsequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack�
Zsequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2\
Zsequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1�
Zsequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2\
Zsequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2�
Rsequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1asequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack:output:0csequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0csequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask2T
Rsequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice_1�
Fsequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/subSubYsequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice:output:0[sequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2H
Fsequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/sub�
0sequential_4/bidirectional_4/forward_lstm_4/CastCastJsequential_4/bidirectional_4/forward_lstm_4/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������22
0sequential_4/bidirectional_4/forward_lstm_4/Cast�
1sequential_4/bidirectional_4/forward_lstm_4/ShapeShapeXsequential_4/bidirectional_4/forward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:23
1sequential_4/bidirectional_4/forward_lstm_4/Shape�
?sequential_4/bidirectional_4/forward_lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?sequential_4/bidirectional_4/forward_lstm_4/strided_slice/stack�
Asequential_4/bidirectional_4/forward_lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential_4/bidirectional_4/forward_lstm_4/strided_slice/stack_1�
Asequential_4/bidirectional_4/forward_lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential_4/bidirectional_4/forward_lstm_4/strided_slice/stack_2�
9sequential_4/bidirectional_4/forward_lstm_4/strided_sliceStridedSlice:sequential_4/bidirectional_4/forward_lstm_4/Shape:output:0Hsequential_4/bidirectional_4/forward_lstm_4/strided_slice/stack:output:0Jsequential_4/bidirectional_4/forward_lstm_4/strided_slice/stack_1:output:0Jsequential_4/bidirectional_4/forward_lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9sequential_4/bidirectional_4/forward_lstm_4/strided_slice�
7sequential_4/bidirectional_4/forward_lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :229
7sequential_4/bidirectional_4/forward_lstm_4/zeros/mul/y�
5sequential_4/bidirectional_4/forward_lstm_4/zeros/mulMulBsequential_4/bidirectional_4/forward_lstm_4/strided_slice:output:0@sequential_4/bidirectional_4/forward_lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 27
5sequential_4/bidirectional_4/forward_lstm_4/zeros/mul�
8sequential_4/bidirectional_4/forward_lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2:
8sequential_4/bidirectional_4/forward_lstm_4/zeros/Less/y�
6sequential_4/bidirectional_4/forward_lstm_4/zeros/LessLess9sequential_4/bidirectional_4/forward_lstm_4/zeros/mul:z:0Asequential_4/bidirectional_4/forward_lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 28
6sequential_4/bidirectional_4/forward_lstm_4/zeros/Less�
:sequential_4/bidirectional_4/forward_lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22<
:sequential_4/bidirectional_4/forward_lstm_4/zeros/packed/1�
8sequential_4/bidirectional_4/forward_lstm_4/zeros/packedPackBsequential_4/bidirectional_4/forward_lstm_4/strided_slice:output:0Csequential_4/bidirectional_4/forward_lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2:
8sequential_4/bidirectional_4/forward_lstm_4/zeros/packed�
7sequential_4/bidirectional_4/forward_lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        29
7sequential_4/bidirectional_4/forward_lstm_4/zeros/Const�
1sequential_4/bidirectional_4/forward_lstm_4/zerosFillAsequential_4/bidirectional_4/forward_lstm_4/zeros/packed:output:0@sequential_4/bidirectional_4/forward_lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:���������223
1sequential_4/bidirectional_4/forward_lstm_4/zeros�
9sequential_4/bidirectional_4/forward_lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22;
9sequential_4/bidirectional_4/forward_lstm_4/zeros_1/mul/y�
7sequential_4/bidirectional_4/forward_lstm_4/zeros_1/mulMulBsequential_4/bidirectional_4/forward_lstm_4/strided_slice:output:0Bsequential_4/bidirectional_4/forward_lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 29
7sequential_4/bidirectional_4/forward_lstm_4/zeros_1/mul�
:sequential_4/bidirectional_4/forward_lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2<
:sequential_4/bidirectional_4/forward_lstm_4/zeros_1/Less/y�
8sequential_4/bidirectional_4/forward_lstm_4/zeros_1/LessLess;sequential_4/bidirectional_4/forward_lstm_4/zeros_1/mul:z:0Csequential_4/bidirectional_4/forward_lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2:
8sequential_4/bidirectional_4/forward_lstm_4/zeros_1/Less�
<sequential_4/bidirectional_4/forward_lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22>
<sequential_4/bidirectional_4/forward_lstm_4/zeros_1/packed/1�
:sequential_4/bidirectional_4/forward_lstm_4/zeros_1/packedPackBsequential_4/bidirectional_4/forward_lstm_4/strided_slice:output:0Esequential_4/bidirectional_4/forward_lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2<
:sequential_4/bidirectional_4/forward_lstm_4/zeros_1/packed�
9sequential_4/bidirectional_4/forward_lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2;
9sequential_4/bidirectional_4/forward_lstm_4/zeros_1/Const�
3sequential_4/bidirectional_4/forward_lstm_4/zeros_1FillCsequential_4/bidirectional_4/forward_lstm_4/zeros_1/packed:output:0Bsequential_4/bidirectional_4/forward_lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������225
3sequential_4/bidirectional_4/forward_lstm_4/zeros_1�
:sequential_4/bidirectional_4/forward_lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2<
:sequential_4/bidirectional_4/forward_lstm_4/transpose/perm�
5sequential_4/bidirectional_4/forward_lstm_4/transpose	TransposeXsequential_4/bidirectional_4/forward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0Csequential_4/bidirectional_4/forward_lstm_4/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������27
5sequential_4/bidirectional_4/forward_lstm_4/transpose�
3sequential_4/bidirectional_4/forward_lstm_4/Shape_1Shape9sequential_4/bidirectional_4/forward_lstm_4/transpose:y:0*
T0*
_output_shapes
:25
3sequential_4/bidirectional_4/forward_lstm_4/Shape_1�
Asequential_4/bidirectional_4/forward_lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential_4/bidirectional_4/forward_lstm_4/strided_slice_1/stack�
Csequential_4/bidirectional_4/forward_lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_4/bidirectional_4/forward_lstm_4/strided_slice_1/stack_1�
Csequential_4/bidirectional_4/forward_lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_4/bidirectional_4/forward_lstm_4/strided_slice_1/stack_2�
;sequential_4/bidirectional_4/forward_lstm_4/strided_slice_1StridedSlice<sequential_4/bidirectional_4/forward_lstm_4/Shape_1:output:0Jsequential_4/bidirectional_4/forward_lstm_4/strided_slice_1/stack:output:0Lsequential_4/bidirectional_4/forward_lstm_4/strided_slice_1/stack_1:output:0Lsequential_4/bidirectional_4/forward_lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;sequential_4/bidirectional_4/forward_lstm_4/strided_slice_1�
Gsequential_4/bidirectional_4/forward_lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2I
Gsequential_4/bidirectional_4/forward_lstm_4/TensorArrayV2/element_shape�
9sequential_4/bidirectional_4/forward_lstm_4/TensorArrayV2TensorListReservePsequential_4/bidirectional_4/forward_lstm_4/TensorArrayV2/element_shape:output:0Dsequential_4/bidirectional_4/forward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9sequential_4/bidirectional_4/forward_lstm_4/TensorArrayV2�
asequential_4/bidirectional_4/forward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2c
asequential_4/bidirectional_4/forward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape�
Ssequential_4/bidirectional_4/forward_lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor9sequential_4/bidirectional_4/forward_lstm_4/transpose:y:0jsequential_4/bidirectional_4/forward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02U
Ssequential_4/bidirectional_4/forward_lstm_4/TensorArrayUnstack/TensorListFromTensor�
Asequential_4/bidirectional_4/forward_lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential_4/bidirectional_4/forward_lstm_4/strided_slice_2/stack�
Csequential_4/bidirectional_4/forward_lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_4/bidirectional_4/forward_lstm_4/strided_slice_2/stack_1�
Csequential_4/bidirectional_4/forward_lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_4/bidirectional_4/forward_lstm_4/strided_slice_2/stack_2�
;sequential_4/bidirectional_4/forward_lstm_4/strided_slice_2StridedSlice9sequential_4/bidirectional_4/forward_lstm_4/transpose:y:0Jsequential_4/bidirectional_4/forward_lstm_4/strided_slice_2/stack:output:0Lsequential_4/bidirectional_4/forward_lstm_4/strided_slice_2/stack_1:output:0Lsequential_4/bidirectional_4/forward_lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2=
;sequential_4/bidirectional_4/forward_lstm_4/strided_slice_2�
Nsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOpReadVariableOpWsequential_4_bidirectional_4_forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02P
Nsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp�
?sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMulMatMulDsequential_4/bidirectional_4/forward_lstm_4/strided_slice_2:output:0Vsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2A
?sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMul�
Psequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOpYsequential_4_bidirectional_4_forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02R
Psequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp�
Asequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMul_1MatMul:sequential_4/bidirectional_4/forward_lstm_4/zeros:output:0Xsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2C
Asequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMul_1�
<sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/addAddV2Isequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMul:product:0Ksequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2>
<sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/add�
Osequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOpXsequential_4_bidirectional_4_forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02Q
Osequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp�
@sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/BiasAddBiasAdd@sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/add:z:0Wsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2B
@sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/BiasAdd�
Hsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2J
Hsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/split/split_dim�
>sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/splitSplitQsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/split/split_dim:output:0Isequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2@
>sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/split�
@sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/SigmoidSigmoidGsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22B
@sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/Sigmoid�
Bsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/Sigmoid_1SigmoidGsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22D
Bsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/Sigmoid_1�
<sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/mulMulFsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/Sigmoid_1:y:0<sequential_4/bidirectional_4/forward_lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:���������22>
<sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/mul�
=sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/ReluReluGsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22?
=sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/Relu�
>sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/mul_1MulDsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/Sigmoid:y:0Ksequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22@
>sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/mul_1�
>sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/add_1AddV2@sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/mul:z:0Bsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22@
>sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/add_1�
Bsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/Sigmoid_2SigmoidGsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22D
Bsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/Sigmoid_2�
?sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/Relu_1ReluBsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22A
?sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/Relu_1�
>sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/mul_2MulFsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/Sigmoid_2:y:0Msequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22@
>sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/mul_2�
Isequential_4/bidirectional_4/forward_lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2K
Isequential_4/bidirectional_4/forward_lstm_4/TensorArrayV2_1/element_shape�
;sequential_4/bidirectional_4/forward_lstm_4/TensorArrayV2_1TensorListReserveRsequential_4/bidirectional_4/forward_lstm_4/TensorArrayV2_1/element_shape:output:0Dsequential_4/bidirectional_4/forward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_4/bidirectional_4/forward_lstm_4/TensorArrayV2_1�
0sequential_4/bidirectional_4/forward_lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_4/bidirectional_4/forward_lstm_4/time�
6sequential_4/bidirectional_4/forward_lstm_4/zeros_like	ZerosLikeBsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/mul_2:z:0*
T0*'
_output_shapes
:���������228
6sequential_4/bidirectional_4/forward_lstm_4/zeros_like�
Dsequential_4/bidirectional_4/forward_lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2F
Dsequential_4/bidirectional_4/forward_lstm_4/while/maximum_iterations�
>sequential_4/bidirectional_4/forward_lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2@
>sequential_4/bidirectional_4/forward_lstm_4/while/loop_counter�
1sequential_4/bidirectional_4/forward_lstm_4/whileWhileGsequential_4/bidirectional_4/forward_lstm_4/while/loop_counter:output:0Msequential_4/bidirectional_4/forward_lstm_4/while/maximum_iterations:output:09sequential_4/bidirectional_4/forward_lstm_4/time:output:0Dsequential_4/bidirectional_4/forward_lstm_4/TensorArrayV2_1:handle:0:sequential_4/bidirectional_4/forward_lstm_4/zeros_like:y:0:sequential_4/bidirectional_4/forward_lstm_4/zeros:output:0<sequential_4/bidirectional_4/forward_lstm_4/zeros_1:output:0Dsequential_4/bidirectional_4/forward_lstm_4/strided_slice_1:output:0csequential_4/bidirectional_4/forward_lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:04sequential_4/bidirectional_4/forward_lstm_4/Cast:y:0Wsequential_4_bidirectional_4_forward_lstm_4_lstm_cell_13_matmul_readvariableop_resourceYsequential_4_bidirectional_4_forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resourceXsequential_4_bidirectional_4_forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *I
bodyAR?
=sequential_4_bidirectional_4_forward_lstm_4_while_body_690243*I
condAR?
=sequential_4_bidirectional_4_forward_lstm_4_while_cond_690242*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 23
1sequential_4/bidirectional_4/forward_lstm_4/while�
\sequential_4/bidirectional_4/forward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2^
\sequential_4/bidirectional_4/forward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape�
Nsequential_4/bidirectional_4/forward_lstm_4/TensorArrayV2Stack/TensorListStackTensorListStack:sequential_4/bidirectional_4/forward_lstm_4/while:output:3esequential_4/bidirectional_4/forward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02P
Nsequential_4/bidirectional_4/forward_lstm_4/TensorArrayV2Stack/TensorListStack�
Asequential_4/bidirectional_4/forward_lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2C
Asequential_4/bidirectional_4/forward_lstm_4/strided_slice_3/stack�
Csequential_4/bidirectional_4/forward_lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Csequential_4/bidirectional_4/forward_lstm_4/strided_slice_3/stack_1�
Csequential_4/bidirectional_4/forward_lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_4/bidirectional_4/forward_lstm_4/strided_slice_3/stack_2�
;sequential_4/bidirectional_4/forward_lstm_4/strided_slice_3StridedSliceWsequential_4/bidirectional_4/forward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0Jsequential_4/bidirectional_4/forward_lstm_4/strided_slice_3/stack:output:0Lsequential_4/bidirectional_4/forward_lstm_4/strided_slice_3/stack_1:output:0Lsequential_4/bidirectional_4/forward_lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2=
;sequential_4/bidirectional_4/forward_lstm_4/strided_slice_3�
<sequential_4/bidirectional_4/forward_lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2>
<sequential_4/bidirectional_4/forward_lstm_4/transpose_1/perm�
7sequential_4/bidirectional_4/forward_lstm_4/transpose_1	TransposeWsequential_4/bidirectional_4/forward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0Esequential_4/bidirectional_4/forward_lstm_4/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������229
7sequential_4/bidirectional_4/forward_lstm_4/transpose_1�
3sequential_4/bidirectional_4/forward_lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    25
3sequential_4/bidirectional_4/forward_lstm_4/runtime�
Asequential_4/bidirectional_4/backward_lstm_4/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2C
Asequential_4/bidirectional_4/backward_lstm_4/RaggedToTensor/zeros�
Asequential_4/bidirectional_4/backward_lstm_4/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2C
Asequential_4/bidirectional_4/backward_lstm_4/RaggedToTensor/Const�
Psequential_4/bidirectional_4/backward_lstm_4/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorJsequential_4/bidirectional_4/backward_lstm_4/RaggedToTensor/Const:output:0args_0Jsequential_4/bidirectional_4/backward_lstm_4/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2R
Psequential_4/bidirectional_4/backward_lstm_4/RaggedToTensor/RaggedTensorToTensor�
Wsequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2Y
Wsequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack�
Ysequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2[
Ysequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1�
Ysequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2[
Ysequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2�
Qsequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1`sequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack:output:0bsequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1:output:0bsequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask2S
Qsequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice�
Ysequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2[
Ysequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack�
[sequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2]
[sequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1�
[sequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2]
[sequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2�
Ssequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1bsequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack:output:0dsequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0dsequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask2U
Ssequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice_1�
Gsequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/subSubZsequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice:output:0\sequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2I
Gsequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/sub�
1sequential_4/bidirectional_4/backward_lstm_4/CastCastKsequential_4/bidirectional_4/backward_lstm_4/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������23
1sequential_4/bidirectional_4/backward_lstm_4/Cast�
2sequential_4/bidirectional_4/backward_lstm_4/ShapeShapeYsequential_4/bidirectional_4/backward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:24
2sequential_4/bidirectional_4/backward_lstm_4/Shape�
@sequential_4/bidirectional_4/backward_lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_4/bidirectional_4/backward_lstm_4/strided_slice/stack�
Bsequential_4/bidirectional_4/backward_lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_4/bidirectional_4/backward_lstm_4/strided_slice/stack_1�
Bsequential_4/bidirectional_4/backward_lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_4/bidirectional_4/backward_lstm_4/strided_slice/stack_2�
:sequential_4/bidirectional_4/backward_lstm_4/strided_sliceStridedSlice;sequential_4/bidirectional_4/backward_lstm_4/Shape:output:0Isequential_4/bidirectional_4/backward_lstm_4/strided_slice/stack:output:0Ksequential_4/bidirectional_4/backward_lstm_4/strided_slice/stack_1:output:0Ksequential_4/bidirectional_4/backward_lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_4/bidirectional_4/backward_lstm_4/strided_slice�
8sequential_4/bidirectional_4/backward_lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22:
8sequential_4/bidirectional_4/backward_lstm_4/zeros/mul/y�
6sequential_4/bidirectional_4/backward_lstm_4/zeros/mulMulCsequential_4/bidirectional_4/backward_lstm_4/strided_slice:output:0Asequential_4/bidirectional_4/backward_lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 28
6sequential_4/bidirectional_4/backward_lstm_4/zeros/mul�
9sequential_4/bidirectional_4/backward_lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2;
9sequential_4/bidirectional_4/backward_lstm_4/zeros/Less/y�
7sequential_4/bidirectional_4/backward_lstm_4/zeros/LessLess:sequential_4/bidirectional_4/backward_lstm_4/zeros/mul:z:0Bsequential_4/bidirectional_4/backward_lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 29
7sequential_4/bidirectional_4/backward_lstm_4/zeros/Less�
;sequential_4/bidirectional_4/backward_lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22=
;sequential_4/bidirectional_4/backward_lstm_4/zeros/packed/1�
9sequential_4/bidirectional_4/backward_lstm_4/zeros/packedPackCsequential_4/bidirectional_4/backward_lstm_4/strided_slice:output:0Dsequential_4/bidirectional_4/backward_lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2;
9sequential_4/bidirectional_4/backward_lstm_4/zeros/packed�
8sequential_4/bidirectional_4/backward_lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2:
8sequential_4/bidirectional_4/backward_lstm_4/zeros/Const�
2sequential_4/bidirectional_4/backward_lstm_4/zerosFillBsequential_4/bidirectional_4/backward_lstm_4/zeros/packed:output:0Asequential_4/bidirectional_4/backward_lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:���������224
2sequential_4/bidirectional_4/backward_lstm_4/zeros�
:sequential_4/bidirectional_4/backward_lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22<
:sequential_4/bidirectional_4/backward_lstm_4/zeros_1/mul/y�
8sequential_4/bidirectional_4/backward_lstm_4/zeros_1/mulMulCsequential_4/bidirectional_4/backward_lstm_4/strided_slice:output:0Csequential_4/bidirectional_4/backward_lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2:
8sequential_4/bidirectional_4/backward_lstm_4/zeros_1/mul�
;sequential_4/bidirectional_4/backward_lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2=
;sequential_4/bidirectional_4/backward_lstm_4/zeros_1/Less/y�
9sequential_4/bidirectional_4/backward_lstm_4/zeros_1/LessLess<sequential_4/bidirectional_4/backward_lstm_4/zeros_1/mul:z:0Dsequential_4/bidirectional_4/backward_lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2;
9sequential_4/bidirectional_4/backward_lstm_4/zeros_1/Less�
=sequential_4/bidirectional_4/backward_lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22?
=sequential_4/bidirectional_4/backward_lstm_4/zeros_1/packed/1�
;sequential_4/bidirectional_4/backward_lstm_4/zeros_1/packedPackCsequential_4/bidirectional_4/backward_lstm_4/strided_slice:output:0Fsequential_4/bidirectional_4/backward_lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2=
;sequential_4/bidirectional_4/backward_lstm_4/zeros_1/packed�
:sequential_4/bidirectional_4/backward_lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2<
:sequential_4/bidirectional_4/backward_lstm_4/zeros_1/Const�
4sequential_4/bidirectional_4/backward_lstm_4/zeros_1FillDsequential_4/bidirectional_4/backward_lstm_4/zeros_1/packed:output:0Csequential_4/bidirectional_4/backward_lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������226
4sequential_4/bidirectional_4/backward_lstm_4/zeros_1�
;sequential_4/bidirectional_4/backward_lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2=
;sequential_4/bidirectional_4/backward_lstm_4/transpose/perm�
6sequential_4/bidirectional_4/backward_lstm_4/transpose	TransposeYsequential_4/bidirectional_4/backward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0Dsequential_4/bidirectional_4/backward_lstm_4/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������28
6sequential_4/bidirectional_4/backward_lstm_4/transpose�
4sequential_4/bidirectional_4/backward_lstm_4/Shape_1Shape:sequential_4/bidirectional_4/backward_lstm_4/transpose:y:0*
T0*
_output_shapes
:26
4sequential_4/bidirectional_4/backward_lstm_4/Shape_1�
Bsequential_4/bidirectional_4/backward_lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_4/bidirectional_4/backward_lstm_4/strided_slice_1/stack�
Dsequential_4/bidirectional_4/backward_lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_4/bidirectional_4/backward_lstm_4/strided_slice_1/stack_1�
Dsequential_4/bidirectional_4/backward_lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_4/bidirectional_4/backward_lstm_4/strided_slice_1/stack_2�
<sequential_4/bidirectional_4/backward_lstm_4/strided_slice_1StridedSlice=sequential_4/bidirectional_4/backward_lstm_4/Shape_1:output:0Ksequential_4/bidirectional_4/backward_lstm_4/strided_slice_1/stack:output:0Msequential_4/bidirectional_4/backward_lstm_4/strided_slice_1/stack_1:output:0Msequential_4/bidirectional_4/backward_lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_4/bidirectional_4/backward_lstm_4/strided_slice_1�
Hsequential_4/bidirectional_4/backward_lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2J
Hsequential_4/bidirectional_4/backward_lstm_4/TensorArrayV2/element_shape�
:sequential_4/bidirectional_4/backward_lstm_4/TensorArrayV2TensorListReserveQsequential_4/bidirectional_4/backward_lstm_4/TensorArrayV2/element_shape:output:0Esequential_4/bidirectional_4/backward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02<
:sequential_4/bidirectional_4/backward_lstm_4/TensorArrayV2�
;sequential_4/bidirectional_4/backward_lstm_4/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2=
;sequential_4/bidirectional_4/backward_lstm_4/ReverseV2/axis�
6sequential_4/bidirectional_4/backward_lstm_4/ReverseV2	ReverseV2:sequential_4/bidirectional_4/backward_lstm_4/transpose:y:0Dsequential_4/bidirectional_4/backward_lstm_4/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������28
6sequential_4/bidirectional_4/backward_lstm_4/ReverseV2�
bsequential_4/bidirectional_4/backward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2d
bsequential_4/bidirectional_4/backward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape�
Tsequential_4/bidirectional_4/backward_lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor?sequential_4/bidirectional_4/backward_lstm_4/ReverseV2:output:0ksequential_4/bidirectional_4/backward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02V
Tsequential_4/bidirectional_4/backward_lstm_4/TensorArrayUnstack/TensorListFromTensor�
Bsequential_4/bidirectional_4/backward_lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_4/bidirectional_4/backward_lstm_4/strided_slice_2/stack�
Dsequential_4/bidirectional_4/backward_lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_4/bidirectional_4/backward_lstm_4/strided_slice_2/stack_1�
Dsequential_4/bidirectional_4/backward_lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_4/bidirectional_4/backward_lstm_4/strided_slice_2/stack_2�
<sequential_4/bidirectional_4/backward_lstm_4/strided_slice_2StridedSlice:sequential_4/bidirectional_4/backward_lstm_4/transpose:y:0Ksequential_4/bidirectional_4/backward_lstm_4/strided_slice_2/stack:output:0Msequential_4/bidirectional_4/backward_lstm_4/strided_slice_2/stack_1:output:0Msequential_4/bidirectional_4/backward_lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2>
<sequential_4/bidirectional_4/backward_lstm_4/strided_slice_2�
Osequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOpReadVariableOpXsequential_4_bidirectional_4_backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02Q
Osequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp�
@sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMulMatMulEsequential_4/bidirectional_4/backward_lstm_4/strided_slice_2:output:0Wsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2B
@sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMul�
Qsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpZsequential_4_bidirectional_4_backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02S
Qsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp�
Bsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMul_1MatMul;sequential_4/bidirectional_4/backward_lstm_4/zeros:output:0Ysequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2D
Bsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMul_1�
=sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/addAddV2Jsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMul:product:0Lsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2?
=sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/add�
Psequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpYsequential_4_bidirectional_4_backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02R
Psequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp�
Asequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/BiasAddBiasAddAsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/add:z:0Xsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2C
Asequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/BiasAdd�
Isequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2K
Isequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/split/split_dim�
?sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/splitSplitRsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/split/split_dim:output:0Jsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2A
?sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/split�
Asequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/SigmoidSigmoidHsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22C
Asequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/Sigmoid�
Csequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/Sigmoid_1SigmoidHsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22E
Csequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/Sigmoid_1�
=sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/mulMulGsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/Sigmoid_1:y:0=sequential_4/bidirectional_4/backward_lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:���������22?
=sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/mul�
>sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/ReluReluHsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22@
>sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/Relu�
?sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/mul_1MulEsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/Sigmoid:y:0Lsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22A
?sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/mul_1�
?sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/add_1AddV2Asequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/mul:z:0Csequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22A
?sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/add_1�
Csequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/Sigmoid_2SigmoidHsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22E
Csequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/Sigmoid_2�
@sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/Relu_1ReluCsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22B
@sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/Relu_1�
?sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/mul_2MulGsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/Sigmoid_2:y:0Nsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22A
?sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/mul_2�
Jsequential_4/bidirectional_4/backward_lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2L
Jsequential_4/bidirectional_4/backward_lstm_4/TensorArrayV2_1/element_shape�
<sequential_4/bidirectional_4/backward_lstm_4/TensorArrayV2_1TensorListReserveSsequential_4/bidirectional_4/backward_lstm_4/TensorArrayV2_1/element_shape:output:0Esequential_4/bidirectional_4/backward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_4/bidirectional_4/backward_lstm_4/TensorArrayV2_1�
1sequential_4/bidirectional_4/backward_lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential_4/bidirectional_4/backward_lstm_4/time�
Bsequential_4/bidirectional_4/backward_lstm_4/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bsequential_4/bidirectional_4/backward_lstm_4/Max/reduction_indices�
0sequential_4/bidirectional_4/backward_lstm_4/MaxMax5sequential_4/bidirectional_4/backward_lstm_4/Cast:y:0Ksequential_4/bidirectional_4/backward_lstm_4/Max/reduction_indices:output:0*
T0*
_output_shapes
: 22
0sequential_4/bidirectional_4/backward_lstm_4/Max�
2sequential_4/bidirectional_4/backward_lstm_4/sub/yConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_4/bidirectional_4/backward_lstm_4/sub/y�
0sequential_4/bidirectional_4/backward_lstm_4/subSub9sequential_4/bidirectional_4/backward_lstm_4/Max:output:0;sequential_4/bidirectional_4/backward_lstm_4/sub/y:output:0*
T0*
_output_shapes
: 22
0sequential_4/bidirectional_4/backward_lstm_4/sub�
2sequential_4/bidirectional_4/backward_lstm_4/Sub_1Sub4sequential_4/bidirectional_4/backward_lstm_4/sub:z:05sequential_4/bidirectional_4/backward_lstm_4/Cast:y:0*
T0*#
_output_shapes
:���������24
2sequential_4/bidirectional_4/backward_lstm_4/Sub_1�
7sequential_4/bidirectional_4/backward_lstm_4/zeros_like	ZerosLikeCsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/mul_2:z:0*
T0*'
_output_shapes
:���������229
7sequential_4/bidirectional_4/backward_lstm_4/zeros_like�
Esequential_4/bidirectional_4/backward_lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2G
Esequential_4/bidirectional_4/backward_lstm_4/while/maximum_iterations�
?sequential_4/bidirectional_4/backward_lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2A
?sequential_4/bidirectional_4/backward_lstm_4/while/loop_counter�
2sequential_4/bidirectional_4/backward_lstm_4/whileWhileHsequential_4/bidirectional_4/backward_lstm_4/while/loop_counter:output:0Nsequential_4/bidirectional_4/backward_lstm_4/while/maximum_iterations:output:0:sequential_4/bidirectional_4/backward_lstm_4/time:output:0Esequential_4/bidirectional_4/backward_lstm_4/TensorArrayV2_1:handle:0;sequential_4/bidirectional_4/backward_lstm_4/zeros_like:y:0;sequential_4/bidirectional_4/backward_lstm_4/zeros:output:0=sequential_4/bidirectional_4/backward_lstm_4/zeros_1:output:0Esequential_4/bidirectional_4/backward_lstm_4/strided_slice_1:output:0dsequential_4/bidirectional_4/backward_lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:06sequential_4/bidirectional_4/backward_lstm_4/Sub_1:z:0Xsequential_4_bidirectional_4_backward_lstm_4_lstm_cell_14_matmul_readvariableop_resourceZsequential_4_bidirectional_4_backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resourceYsequential_4_bidirectional_4_backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *J
bodyBR@
>sequential_4_bidirectional_4_backward_lstm_4_while_body_690422*J
condBR@
>sequential_4_bidirectional_4_backward_lstm_4_while_cond_690421*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 24
2sequential_4/bidirectional_4/backward_lstm_4/while�
]sequential_4/bidirectional_4/backward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2_
]sequential_4/bidirectional_4/backward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape�
Osequential_4/bidirectional_4/backward_lstm_4/TensorArrayV2Stack/TensorListStackTensorListStack;sequential_4/bidirectional_4/backward_lstm_4/while:output:3fsequential_4/bidirectional_4/backward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype02Q
Osequential_4/bidirectional_4/backward_lstm_4/TensorArrayV2Stack/TensorListStack�
Bsequential_4/bidirectional_4/backward_lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2D
Bsequential_4/bidirectional_4/backward_lstm_4/strided_slice_3/stack�
Dsequential_4/bidirectional_4/backward_lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2F
Dsequential_4/bidirectional_4/backward_lstm_4/strided_slice_3/stack_1�
Dsequential_4/bidirectional_4/backward_lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_4/bidirectional_4/backward_lstm_4/strided_slice_3/stack_2�
<sequential_4/bidirectional_4/backward_lstm_4/strided_slice_3StridedSliceXsequential_4/bidirectional_4/backward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0Ksequential_4/bidirectional_4/backward_lstm_4/strided_slice_3/stack:output:0Msequential_4/bidirectional_4/backward_lstm_4/strided_slice_3/stack_1:output:0Msequential_4/bidirectional_4/backward_lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2>
<sequential_4/bidirectional_4/backward_lstm_4/strided_slice_3�
=sequential_4/bidirectional_4/backward_lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2?
=sequential_4/bidirectional_4/backward_lstm_4/transpose_1/perm�
8sequential_4/bidirectional_4/backward_lstm_4/transpose_1	TransposeXsequential_4/bidirectional_4/backward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0Fsequential_4/bidirectional_4/backward_lstm_4/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22:
8sequential_4/bidirectional_4/backward_lstm_4/transpose_1�
4sequential_4/bidirectional_4/backward_lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    26
4sequential_4/bidirectional_4/backward_lstm_4/runtime�
(sequential_4/bidirectional_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_4/bidirectional_4/concat/axis�
#sequential_4/bidirectional_4/concatConcatV2Dsequential_4/bidirectional_4/forward_lstm_4/strided_slice_3:output:0Esequential_4/bidirectional_4/backward_lstm_4/strided_slice_3:output:01sequential_4/bidirectional_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������d2%
#sequential_4/bidirectional_4/concat�
*sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02,
*sequential_4/dense_4/MatMul/ReadVariableOp�
sequential_4/dense_4/MatMulMatMul,sequential_4/bidirectional_4/concat:output:02sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_4/dense_4/MatMul�
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_4/dense_4/BiasAdd/ReadVariableOp�
sequential_4/dense_4/BiasAddBiasAdd%sequential_4/dense_4/MatMul:product:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_4/dense_4/BiasAdd�
sequential_4/dense_4/SigmoidSigmoid%sequential_4/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
sequential_4/dense_4/Sigmoid{
IdentityIdentity sequential_4/dense_4/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOpQ^sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOpP^sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOpR^sequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp3^sequential_4/bidirectional_4/backward_lstm_4/whileP^sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOpO^sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOpQ^sequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp2^sequential_4/bidirectional_4/forward_lstm_4/while,^sequential_4/dense_4/BiasAdd/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:���������:���������: : : : : : : : 2�
Psequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOpPsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp2�
Osequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOpOsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp2�
Qsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOpQsequential_4/bidirectional_4/backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp2h
2sequential_4/bidirectional_4/backward_lstm_4/while2sequential_4/bidirectional_4/backward_lstm_4/while2�
Osequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOpOsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp2�
Nsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOpNsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp2�
Psequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOpPsequential_4/bidirectional_4/forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp2f
1sequential_4/bidirectional_4/forward_lstm_4/while1sequential_4/bidirectional_4/forward_lstm_4/while2Z
+sequential_4/dense_4/BiasAdd/ReadVariableOp+sequential_4/dense_4/BiasAdd/ReadVariableOp2X
*sequential_4/dense_4/MatMul/ReadVariableOp*sequential_4/dense_4/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameargs_0:KG
#
_output_shapes
:���������
 
_user_specified_nameargs_0
�
�
H__inference_sequential_4_layer_call_and_return_conditional_losses_692994

inputs
inputs_1	)
bidirectional_4_692963:	�)
bidirectional_4_692965:	2�%
bidirectional_4_692967:	�)
bidirectional_4_692969:	�)
bidirectional_4_692971:	2�%
bidirectional_4_692973:	� 
dense_4_692988:d
dense_4_692990:
identity��'bidirectional_4/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�
'bidirectional_4/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_4_692963bidirectional_4_692965bidirectional_4_692967bidirectional_4_692969bidirectional_4_692971bidirectional_4_692973*
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
GPU 2J 8� *T
fORM
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_6929622)
'bidirectional_4/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall0bidirectional_4/StatefulPartitionedCall:output:0dense_4_692988dense_4_692990*
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
GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_6929872!
dense_4/StatefulPartitionedCall�
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp(^bidirectional_4/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:���������:���������: : : : : : : : 2R
'bidirectional_4/StatefulPartitionedCall'bidirectional_4/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:O K
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
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_691233

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
while_body_692392
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_13_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_13_matmul_readvariableop_resource:	�F
3while_lstm_cell_13_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_13_biasadd_readvariableop_resource:	���)while/lstm_cell_13/BiasAdd/ReadVariableOp�(while/lstm_cell_13/MatMul/ReadVariableOp�*while/lstm_cell_13/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp�
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/MatMul�
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp�
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/MatMul_1�
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/add�
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp�
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/BiasAdd�
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim�
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_13/split�
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Sigmoid�
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Sigmoid_1�
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/mul�
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Relu�
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/mul_1�
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/add_1�
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Sigmoid_2�
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Relu_1�
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 
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
 forward_lstm_4_while_cond_692685:
6forward_lstm_4_while_forward_lstm_4_while_loop_counter@
<forward_lstm_4_while_forward_lstm_4_while_maximum_iterations$
 forward_lstm_4_while_placeholder&
"forward_lstm_4_while_placeholder_1&
"forward_lstm_4_while_placeholder_2&
"forward_lstm_4_while_placeholder_3&
"forward_lstm_4_while_placeholder_4<
8forward_lstm_4_while_less_forward_lstm_4_strided_slice_1R
Nforward_lstm_4_while_forward_lstm_4_while_cond_692685___redundant_placeholder0R
Nforward_lstm_4_while_forward_lstm_4_while_cond_692685___redundant_placeholder1R
Nforward_lstm_4_while_forward_lstm_4_while_cond_692685___redundant_placeholder2R
Nforward_lstm_4_while_forward_lstm_4_while_cond_692685___redundant_placeholder3R
Nforward_lstm_4_while_forward_lstm_4_while_cond_692685___redundant_placeholder4!
forward_lstm_4_while_identity
�
forward_lstm_4/while/LessLess forward_lstm_4_while_placeholder8forward_lstm_4_while_less_forward_lstm_4_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_4/while/Less�
forward_lstm_4/while/IdentityIdentityforward_lstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2
forward_lstm_4/while/Identity"G
forward_lstm_4_while_identity&forward_lstm_4/while/Identity:output:0*(
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
while_body_692219
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_14_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_14_matmul_readvariableop_resource:	�F
3while_lstm_cell_14_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_14_biasadd_readvariableop_resource:	���)while/lstm_cell_14/BiasAdd/ReadVariableOp�(while/lstm_cell_14/MatMul/ReadVariableOp�*while/lstm_cell_14/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp�
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/MatMul�
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp�
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/MatMul_1�
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/add�
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp�
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/BiasAdd�
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim�
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_14/split�
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Sigmoid�
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Sigmoid_1�
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/mul�
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Relu�
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/mul_1�
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/add_1�
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Sigmoid_2�
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Relu_1�
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
-__inference_lstm_cell_13_layer_call_fn_696313

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
GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_6906012
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
�
>sequential_4_bidirectional_4_backward_lstm_4_while_body_690422v
rsequential_4_bidirectional_4_backward_lstm_4_while_sequential_4_bidirectional_4_backward_lstm_4_while_loop_counter|
xsequential_4_bidirectional_4_backward_lstm_4_while_sequential_4_bidirectional_4_backward_lstm_4_while_maximum_iterationsB
>sequential_4_bidirectional_4_backward_lstm_4_while_placeholderD
@sequential_4_bidirectional_4_backward_lstm_4_while_placeholder_1D
@sequential_4_bidirectional_4_backward_lstm_4_while_placeholder_2D
@sequential_4_bidirectional_4_backward_lstm_4_while_placeholder_3D
@sequential_4_bidirectional_4_backward_lstm_4_while_placeholder_4u
qsequential_4_bidirectional_4_backward_lstm_4_while_sequential_4_bidirectional_4_backward_lstm_4_strided_slice_1_0�
�sequential_4_bidirectional_4_backward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_bidirectional_4_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0p
lsequential_4_bidirectional_4_backward_lstm_4_while_less_sequential_4_bidirectional_4_backward_lstm_4_sub_1_0s
`sequential_4_bidirectional_4_backward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0:	�u
bsequential_4_bidirectional_4_backward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0:	2�p
asequential_4_bidirectional_4_backward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0:	�?
;sequential_4_bidirectional_4_backward_lstm_4_while_identityA
=sequential_4_bidirectional_4_backward_lstm_4_while_identity_1A
=sequential_4_bidirectional_4_backward_lstm_4_while_identity_2A
=sequential_4_bidirectional_4_backward_lstm_4_while_identity_3A
=sequential_4_bidirectional_4_backward_lstm_4_while_identity_4A
=sequential_4_bidirectional_4_backward_lstm_4_while_identity_5A
=sequential_4_bidirectional_4_backward_lstm_4_while_identity_6s
osequential_4_bidirectional_4_backward_lstm_4_while_sequential_4_bidirectional_4_backward_lstm_4_strided_slice_1�
�sequential_4_bidirectional_4_backward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_bidirectional_4_backward_lstm_4_tensorarrayunstack_tensorlistfromtensorn
jsequential_4_bidirectional_4_backward_lstm_4_while_less_sequential_4_bidirectional_4_backward_lstm_4_sub_1q
^sequential_4_bidirectional_4_backward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource:	�s
`sequential_4_bidirectional_4_backward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource:	2�n
_sequential_4_bidirectional_4_backward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource:	���Vsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp�Usequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp�Wsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp�
dsequential_4/bidirectional_4/backward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2f
dsequential_4/bidirectional_4/backward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape�
Vsequential_4/bidirectional_4/backward_lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_4_bidirectional_4_backward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_bidirectional_4_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0>sequential_4_bidirectional_4_backward_lstm_4_while_placeholdermsequential_4/bidirectional_4/backward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02X
Vsequential_4/bidirectional_4/backward_lstm_4/while/TensorArrayV2Read/TensorListGetItem�
7sequential_4/bidirectional_4/backward_lstm_4/while/LessLesslsequential_4_bidirectional_4_backward_lstm_4_while_less_sequential_4_bidirectional_4_backward_lstm_4_sub_1_0>sequential_4_bidirectional_4_backward_lstm_4_while_placeholder*
T0*#
_output_shapes
:���������29
7sequential_4/bidirectional_4/backward_lstm_4/while/Less�
Usequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp`sequential_4_bidirectional_4_backward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02W
Usequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp�
Fsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMulMatMul]sequential_4/bidirectional_4/backward_lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0]sequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2H
Fsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMul�
Wsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpbsequential_4_bidirectional_4_backward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02Y
Wsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp�
Hsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMul_1MatMul@sequential_4_bidirectional_4_backward_lstm_4_while_placeholder_3_sequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2J
Hsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMul_1�
Csequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/addAddV2Psequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMul:product:0Rsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2E
Csequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/add�
Vsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpasequential_4_bidirectional_4_backward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02X
Vsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp�
Gsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/BiasAddBiasAddGsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/add:z:0^sequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2I
Gsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/BiasAdd�
Osequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2Q
Osequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/split/split_dim�
Esequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/splitSplitXsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/split/split_dim:output:0Psequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2G
Esequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/split�
Gsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/SigmoidSigmoidNsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22I
Gsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/Sigmoid�
Isequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/Sigmoid_1SigmoidNsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22K
Isequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/Sigmoid_1�
Csequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/mulMulMsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/Sigmoid_1:y:0@sequential_4_bidirectional_4_backward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22E
Csequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/mul�
Dsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/ReluReluNsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22F
Dsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/Relu�
Esequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/mul_1MulKsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/Sigmoid:y:0Rsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22G
Esequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/mul_1�
Esequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/add_1AddV2Gsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/mul:z:0Isequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22G
Esequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/add_1�
Isequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/Sigmoid_2SigmoidNsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22K
Isequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/Sigmoid_2�
Fsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/Relu_1ReluIsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22H
Fsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/Relu_1�
Esequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/mul_2MulMsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/Sigmoid_2:y:0Tsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22G
Esequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/mul_2�
9sequential_4/bidirectional_4/backward_lstm_4/while/SelectSelect;sequential_4/bidirectional_4/backward_lstm_4/while/Less:z:0Isequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/mul_2:z:0@sequential_4_bidirectional_4_backward_lstm_4_while_placeholder_2*
T0*'
_output_shapes
:���������22;
9sequential_4/bidirectional_4/backward_lstm_4/while/Select�
;sequential_4/bidirectional_4/backward_lstm_4/while/Select_1Select;sequential_4/bidirectional_4/backward_lstm_4/while/Less:z:0Isequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/mul_2:z:0@sequential_4_bidirectional_4_backward_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:���������22=
;sequential_4/bidirectional_4/backward_lstm_4/while/Select_1�
;sequential_4/bidirectional_4/backward_lstm_4/while/Select_2Select;sequential_4/bidirectional_4/backward_lstm_4/while/Less:z:0Isequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/add_1:z:0@sequential_4_bidirectional_4_backward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22=
;sequential_4/bidirectional_4/backward_lstm_4/while/Select_2�
Wsequential_4/bidirectional_4/backward_lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem@sequential_4_bidirectional_4_backward_lstm_4_while_placeholder_1>sequential_4_bidirectional_4_backward_lstm_4_while_placeholderBsequential_4/bidirectional_4/backward_lstm_4/while/Select:output:0*
_output_shapes
: *
element_dtype02Y
Wsequential_4/bidirectional_4/backward_lstm_4/while/TensorArrayV2Write/TensorListSetItem�
8sequential_4/bidirectional_4/backward_lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_4/bidirectional_4/backward_lstm_4/while/add/y�
6sequential_4/bidirectional_4/backward_lstm_4/while/addAddV2>sequential_4_bidirectional_4_backward_lstm_4_while_placeholderAsequential_4/bidirectional_4/backward_lstm_4/while/add/y:output:0*
T0*
_output_shapes
: 28
6sequential_4/bidirectional_4/backward_lstm_4/while/add�
:sequential_4/bidirectional_4/backward_lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_4/bidirectional_4/backward_lstm_4/while/add_1/y�
8sequential_4/bidirectional_4/backward_lstm_4/while/add_1AddV2rsequential_4_bidirectional_4_backward_lstm_4_while_sequential_4_bidirectional_4_backward_lstm_4_while_loop_counterCsequential_4/bidirectional_4/backward_lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2:
8sequential_4/bidirectional_4/backward_lstm_4/while/add_1�
;sequential_4/bidirectional_4/backward_lstm_4/while/IdentityIdentity<sequential_4/bidirectional_4/backward_lstm_4/while/add_1:z:08^sequential_4/bidirectional_4/backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2=
;sequential_4/bidirectional_4/backward_lstm_4/while/Identity�
=sequential_4/bidirectional_4/backward_lstm_4/while/Identity_1Identityxsequential_4_bidirectional_4_backward_lstm_4_while_sequential_4_bidirectional_4_backward_lstm_4_while_maximum_iterations8^sequential_4/bidirectional_4/backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2?
=sequential_4/bidirectional_4/backward_lstm_4/while/Identity_1�
=sequential_4/bidirectional_4/backward_lstm_4/while/Identity_2Identity:sequential_4/bidirectional_4/backward_lstm_4/while/add:z:08^sequential_4/bidirectional_4/backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2?
=sequential_4/bidirectional_4/backward_lstm_4/while/Identity_2�
=sequential_4/bidirectional_4/backward_lstm_4/while/Identity_3Identitygsequential_4/bidirectional_4/backward_lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:08^sequential_4/bidirectional_4/backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2?
=sequential_4/bidirectional_4/backward_lstm_4/while/Identity_3�
=sequential_4/bidirectional_4/backward_lstm_4/while/Identity_4IdentityBsequential_4/bidirectional_4/backward_lstm_4/while/Select:output:08^sequential_4/bidirectional_4/backward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22?
=sequential_4/bidirectional_4/backward_lstm_4/while/Identity_4�
=sequential_4/bidirectional_4/backward_lstm_4/while/Identity_5IdentityDsequential_4/bidirectional_4/backward_lstm_4/while/Select_1:output:08^sequential_4/bidirectional_4/backward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22?
=sequential_4/bidirectional_4/backward_lstm_4/while/Identity_5�
=sequential_4/bidirectional_4/backward_lstm_4/while/Identity_6IdentityDsequential_4/bidirectional_4/backward_lstm_4/while/Select_2:output:08^sequential_4/bidirectional_4/backward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22?
=sequential_4/bidirectional_4/backward_lstm_4/while/Identity_6�
7sequential_4/bidirectional_4/backward_lstm_4/while/NoOpNoOpW^sequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOpV^sequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOpX^sequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 29
7sequential_4/bidirectional_4/backward_lstm_4/while/NoOp"�
;sequential_4_bidirectional_4_backward_lstm_4_while_identityDsequential_4/bidirectional_4/backward_lstm_4/while/Identity:output:0"�
=sequential_4_bidirectional_4_backward_lstm_4_while_identity_1Fsequential_4/bidirectional_4/backward_lstm_4/while/Identity_1:output:0"�
=sequential_4_bidirectional_4_backward_lstm_4_while_identity_2Fsequential_4/bidirectional_4/backward_lstm_4/while/Identity_2:output:0"�
=sequential_4_bidirectional_4_backward_lstm_4_while_identity_3Fsequential_4/bidirectional_4/backward_lstm_4/while/Identity_3:output:0"�
=sequential_4_bidirectional_4_backward_lstm_4_while_identity_4Fsequential_4/bidirectional_4/backward_lstm_4/while/Identity_4:output:0"�
=sequential_4_bidirectional_4_backward_lstm_4_while_identity_5Fsequential_4/bidirectional_4/backward_lstm_4/while/Identity_5:output:0"�
=sequential_4_bidirectional_4_backward_lstm_4_while_identity_6Fsequential_4/bidirectional_4/backward_lstm_4/while/Identity_6:output:0"�
jsequential_4_bidirectional_4_backward_lstm_4_while_less_sequential_4_bidirectional_4_backward_lstm_4_sub_1lsequential_4_bidirectional_4_backward_lstm_4_while_less_sequential_4_bidirectional_4_backward_lstm_4_sub_1_0"�
_sequential_4_bidirectional_4_backward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resourceasequential_4_bidirectional_4_backward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0"�
`sequential_4_bidirectional_4_backward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resourcebsequential_4_bidirectional_4_backward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0"�
^sequential_4_bidirectional_4_backward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource`sequential_4_bidirectional_4_backward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0"�
osequential_4_bidirectional_4_backward_lstm_4_while_sequential_4_bidirectional_4_backward_lstm_4_strided_slice_1qsequential_4_bidirectional_4_backward_lstm_4_while_sequential_4_bidirectional_4_backward_lstm_4_strided_slice_1_0"�
�sequential_4_bidirectional_4_backward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_bidirectional_4_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor�sequential_4_bidirectional_4_backward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_bidirectional_4_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2�
Vsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOpVsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp2�
Usequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOpUsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp2�
Wsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOpWsequential_4/bidirectional_4/backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
0__inference_bidirectional_4_layer_call_fn_693652

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
GPU 2J 8� *T
fORM
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_6934022
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
�a
�
 forward_lstm_4_while_body_693126:
6forward_lstm_4_while_forward_lstm_4_while_loop_counter@
<forward_lstm_4_while_forward_lstm_4_while_maximum_iterations$
 forward_lstm_4_while_placeholder&
"forward_lstm_4_while_placeholder_1&
"forward_lstm_4_while_placeholder_2&
"forward_lstm_4_while_placeholder_3&
"forward_lstm_4_while_placeholder_49
5forward_lstm_4_while_forward_lstm_4_strided_slice_1_0u
qforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_06
2forward_lstm_4_while_greater_forward_lstm_4_cast_0U
Bforward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0:	�W
Dforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0:	2�R
Cforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0:	�!
forward_lstm_4_while_identity#
forward_lstm_4_while_identity_1#
forward_lstm_4_while_identity_2#
forward_lstm_4_while_identity_3#
forward_lstm_4_while_identity_4#
forward_lstm_4_while_identity_5#
forward_lstm_4_while_identity_67
3forward_lstm_4_while_forward_lstm_4_strided_slice_1s
oforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor4
0forward_lstm_4_while_greater_forward_lstm_4_castS
@forward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource:	�U
Bforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource:	2�P
Aforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource:	���8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp�7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp�9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp�
Fforward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2H
Fforward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape�
8forward_lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0 forward_lstm_4_while_placeholderOforward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02:
8forward_lstm_4/while/TensorArrayV2Read/TensorListGetItem�
forward_lstm_4/while/GreaterGreater2forward_lstm_4_while_greater_forward_lstm_4_cast_0 forward_lstm_4_while_placeholder*
T0*#
_output_shapes
:���������2
forward_lstm_4/while/Greater�
7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOpBforward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype029
7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp�
(forward_lstm_4/while/lstm_cell_13/MatMulMatMul?forward_lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0?forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(forward_lstm_4/while/lstm_cell_13/MatMul�
9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOpDforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02;
9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp�
*forward_lstm_4/while/lstm_cell_13/MatMul_1MatMul"forward_lstm_4_while_placeholder_3Aforward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*forward_lstm_4/while/lstm_cell_13/MatMul_1�
%forward_lstm_4/while/lstm_cell_13/addAddV22forward_lstm_4/while/lstm_cell_13/MatMul:product:04forward_lstm_4/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2'
%forward_lstm_4/while/lstm_cell_13/add�
8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOpCforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02:
8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp�
)forward_lstm_4/while/lstm_cell_13/BiasAddBiasAdd)forward_lstm_4/while/lstm_cell_13/add:z:0@forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)forward_lstm_4/while/lstm_cell_13/BiasAdd�
1forward_lstm_4/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1forward_lstm_4/while/lstm_cell_13/split/split_dim�
'forward_lstm_4/while/lstm_cell_13/splitSplit:forward_lstm_4/while/lstm_cell_13/split/split_dim:output:02forward_lstm_4/while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2)
'forward_lstm_4/while/lstm_cell_13/split�
)forward_lstm_4/while/lstm_cell_13/SigmoidSigmoid0forward_lstm_4/while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22+
)forward_lstm_4/while/lstm_cell_13/Sigmoid�
+forward_lstm_4/while/lstm_cell_13/Sigmoid_1Sigmoid0forward_lstm_4/while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22-
+forward_lstm_4/while/lstm_cell_13/Sigmoid_1�
%forward_lstm_4/while/lstm_cell_13/mulMul/forward_lstm_4/while/lstm_cell_13/Sigmoid_1:y:0"forward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22'
%forward_lstm_4/while/lstm_cell_13/mul�
&forward_lstm_4/while/lstm_cell_13/ReluRelu0forward_lstm_4/while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22(
&forward_lstm_4/while/lstm_cell_13/Relu�
'forward_lstm_4/while/lstm_cell_13/mul_1Mul-forward_lstm_4/while/lstm_cell_13/Sigmoid:y:04forward_lstm_4/while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22)
'forward_lstm_4/while/lstm_cell_13/mul_1�
'forward_lstm_4/while/lstm_cell_13/add_1AddV2)forward_lstm_4/while/lstm_cell_13/mul:z:0+forward_lstm_4/while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22)
'forward_lstm_4/while/lstm_cell_13/add_1�
+forward_lstm_4/while/lstm_cell_13/Sigmoid_2Sigmoid0forward_lstm_4/while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22-
+forward_lstm_4/while/lstm_cell_13/Sigmoid_2�
(forward_lstm_4/while/lstm_cell_13/Relu_1Relu+forward_lstm_4/while/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_4/while/lstm_cell_13/Relu_1�
'forward_lstm_4/while/lstm_cell_13/mul_2Mul/forward_lstm_4/while/lstm_cell_13/Sigmoid_2:y:06forward_lstm_4/while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22)
'forward_lstm_4/while/lstm_cell_13/mul_2�
forward_lstm_4/while/SelectSelect forward_lstm_4/while/Greater:z:0+forward_lstm_4/while/lstm_cell_13/mul_2:z:0"forward_lstm_4_while_placeholder_2*
T0*'
_output_shapes
:���������22
forward_lstm_4/while/Select�
forward_lstm_4/while/Select_1Select forward_lstm_4/while/Greater:z:0+forward_lstm_4/while/lstm_cell_13/mul_2:z:0"forward_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:���������22
forward_lstm_4/while/Select_1�
forward_lstm_4/while/Select_2Select forward_lstm_4/while/Greater:z:0+forward_lstm_4/while/lstm_cell_13/add_1:z:0"forward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22
forward_lstm_4/while/Select_2�
9forward_lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"forward_lstm_4_while_placeholder_1 forward_lstm_4_while_placeholder$forward_lstm_4/while/Select:output:0*
_output_shapes
: *
element_dtype02;
9forward_lstm_4/while/TensorArrayV2Write/TensorListSetItemz
forward_lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_4/while/add/y�
forward_lstm_4/while/addAddV2 forward_lstm_4_while_placeholder#forward_lstm_4/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/while/add~
forward_lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_4/while/add_1/y�
forward_lstm_4/while/add_1AddV26forward_lstm_4_while_forward_lstm_4_while_loop_counter%forward_lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/while/add_1�
forward_lstm_4/while/IdentityIdentityforward_lstm_4/while/add_1:z:0^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2
forward_lstm_4/while/Identity�
forward_lstm_4/while/Identity_1Identity<forward_lstm_4_while_forward_lstm_4_while_maximum_iterations^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_4/while/Identity_1�
forward_lstm_4/while/Identity_2Identityforward_lstm_4/while/add:z:0^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_4/while/Identity_2�
forward_lstm_4/while/Identity_3IdentityIforward_lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_4/while/Identity_3�
forward_lstm_4/while/Identity_4Identity$forward_lstm_4/while/Select:output:0^forward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22!
forward_lstm_4/while/Identity_4�
forward_lstm_4/while/Identity_5Identity&forward_lstm_4/while/Select_1:output:0^forward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22!
forward_lstm_4/while/Identity_5�
forward_lstm_4/while/Identity_6Identity&forward_lstm_4/while/Select_2:output:0^forward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22!
forward_lstm_4/while/Identity_6�
forward_lstm_4/while/NoOpNoOp9^forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp8^forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp:^forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_4/while/NoOp"l
3forward_lstm_4_while_forward_lstm_4_strided_slice_15forward_lstm_4_while_forward_lstm_4_strided_slice_1_0"f
0forward_lstm_4_while_greater_forward_lstm_4_cast2forward_lstm_4_while_greater_forward_lstm_4_cast_0"G
forward_lstm_4_while_identity&forward_lstm_4/while/Identity:output:0"K
forward_lstm_4_while_identity_1(forward_lstm_4/while/Identity_1:output:0"K
forward_lstm_4_while_identity_2(forward_lstm_4/while/Identity_2:output:0"K
forward_lstm_4_while_identity_3(forward_lstm_4/while/Identity_3:output:0"K
forward_lstm_4_while_identity_4(forward_lstm_4/while/Identity_4:output:0"K
forward_lstm_4_while_identity_5(forward_lstm_4/while/Identity_5:output:0"K
forward_lstm_4_while_identity_6(forward_lstm_4/while/Identity_6:output:0"�
Aforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resourceCforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0"�
Bforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resourceDforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0"�
@forward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resourceBforward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0"�
oforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensorqforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2t
8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp2r
7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp2v
9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp: 
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
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_691951

inputs>
+lstm_cell_13_matmul_readvariableop_resource:	�@
-lstm_cell_13_matmul_1_readvariableop_resource:	2�;
,lstm_cell_13_biasadd_readvariableop_resource:	�
identity��#lstm_cell_13/BiasAdd/ReadVariableOp�"lstm_cell_13/MatMul/ReadVariableOp�$lstm_cell_13/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp�
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/MatMul�
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp�
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/MatMul_1�
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/add�
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp�
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim�
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_13/split�
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/Sigmoid�
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_13/Sigmoid_1�
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/mul}
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_13/Relu�
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/mul_1�
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/add_1�
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_13/Sigmoid_2|
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/Relu_1�
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
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
bodyR
while_body_691867*
condR
while_cond_691866*K
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
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
while_cond_690824
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_690824___redundant_placeholder04
0while_while_cond_690824___redundant_placeholder14
0while_while_cond_690824___redundant_placeholder24
0while_while_cond_690824___redundant_placeholder3
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
�a
�
 forward_lstm_4_while_body_692686:
6forward_lstm_4_while_forward_lstm_4_while_loop_counter@
<forward_lstm_4_while_forward_lstm_4_while_maximum_iterations$
 forward_lstm_4_while_placeholder&
"forward_lstm_4_while_placeholder_1&
"forward_lstm_4_while_placeholder_2&
"forward_lstm_4_while_placeholder_3&
"forward_lstm_4_while_placeholder_49
5forward_lstm_4_while_forward_lstm_4_strided_slice_1_0u
qforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_06
2forward_lstm_4_while_greater_forward_lstm_4_cast_0U
Bforward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0:	�W
Dforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0:	2�R
Cforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0:	�!
forward_lstm_4_while_identity#
forward_lstm_4_while_identity_1#
forward_lstm_4_while_identity_2#
forward_lstm_4_while_identity_3#
forward_lstm_4_while_identity_4#
forward_lstm_4_while_identity_5#
forward_lstm_4_while_identity_67
3forward_lstm_4_while_forward_lstm_4_strided_slice_1s
oforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor4
0forward_lstm_4_while_greater_forward_lstm_4_castS
@forward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource:	�U
Bforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource:	2�P
Aforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource:	���8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp�7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp�9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp�
Fforward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2H
Fforward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape�
8forward_lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0 forward_lstm_4_while_placeholderOforward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02:
8forward_lstm_4/while/TensorArrayV2Read/TensorListGetItem�
forward_lstm_4/while/GreaterGreater2forward_lstm_4_while_greater_forward_lstm_4_cast_0 forward_lstm_4_while_placeholder*
T0*#
_output_shapes
:���������2
forward_lstm_4/while/Greater�
7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOpBforward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype029
7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp�
(forward_lstm_4/while/lstm_cell_13/MatMulMatMul?forward_lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0?forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(forward_lstm_4/while/lstm_cell_13/MatMul�
9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOpDforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02;
9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp�
*forward_lstm_4/while/lstm_cell_13/MatMul_1MatMul"forward_lstm_4_while_placeholder_3Aforward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*forward_lstm_4/while/lstm_cell_13/MatMul_1�
%forward_lstm_4/while/lstm_cell_13/addAddV22forward_lstm_4/while/lstm_cell_13/MatMul:product:04forward_lstm_4/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2'
%forward_lstm_4/while/lstm_cell_13/add�
8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOpCforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02:
8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp�
)forward_lstm_4/while/lstm_cell_13/BiasAddBiasAdd)forward_lstm_4/while/lstm_cell_13/add:z:0@forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)forward_lstm_4/while/lstm_cell_13/BiasAdd�
1forward_lstm_4/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1forward_lstm_4/while/lstm_cell_13/split/split_dim�
'forward_lstm_4/while/lstm_cell_13/splitSplit:forward_lstm_4/while/lstm_cell_13/split/split_dim:output:02forward_lstm_4/while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2)
'forward_lstm_4/while/lstm_cell_13/split�
)forward_lstm_4/while/lstm_cell_13/SigmoidSigmoid0forward_lstm_4/while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22+
)forward_lstm_4/while/lstm_cell_13/Sigmoid�
+forward_lstm_4/while/lstm_cell_13/Sigmoid_1Sigmoid0forward_lstm_4/while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22-
+forward_lstm_4/while/lstm_cell_13/Sigmoid_1�
%forward_lstm_4/while/lstm_cell_13/mulMul/forward_lstm_4/while/lstm_cell_13/Sigmoid_1:y:0"forward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22'
%forward_lstm_4/while/lstm_cell_13/mul�
&forward_lstm_4/while/lstm_cell_13/ReluRelu0forward_lstm_4/while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22(
&forward_lstm_4/while/lstm_cell_13/Relu�
'forward_lstm_4/while/lstm_cell_13/mul_1Mul-forward_lstm_4/while/lstm_cell_13/Sigmoid:y:04forward_lstm_4/while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22)
'forward_lstm_4/while/lstm_cell_13/mul_1�
'forward_lstm_4/while/lstm_cell_13/add_1AddV2)forward_lstm_4/while/lstm_cell_13/mul:z:0+forward_lstm_4/while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22)
'forward_lstm_4/while/lstm_cell_13/add_1�
+forward_lstm_4/while/lstm_cell_13/Sigmoid_2Sigmoid0forward_lstm_4/while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22-
+forward_lstm_4/while/lstm_cell_13/Sigmoid_2�
(forward_lstm_4/while/lstm_cell_13/Relu_1Relu+forward_lstm_4/while/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_4/while/lstm_cell_13/Relu_1�
'forward_lstm_4/while/lstm_cell_13/mul_2Mul/forward_lstm_4/while/lstm_cell_13/Sigmoid_2:y:06forward_lstm_4/while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22)
'forward_lstm_4/while/lstm_cell_13/mul_2�
forward_lstm_4/while/SelectSelect forward_lstm_4/while/Greater:z:0+forward_lstm_4/while/lstm_cell_13/mul_2:z:0"forward_lstm_4_while_placeholder_2*
T0*'
_output_shapes
:���������22
forward_lstm_4/while/Select�
forward_lstm_4/while/Select_1Select forward_lstm_4/while/Greater:z:0+forward_lstm_4/while/lstm_cell_13/mul_2:z:0"forward_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:���������22
forward_lstm_4/while/Select_1�
forward_lstm_4/while/Select_2Select forward_lstm_4/while/Greater:z:0+forward_lstm_4/while/lstm_cell_13/add_1:z:0"forward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22
forward_lstm_4/while/Select_2�
9forward_lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"forward_lstm_4_while_placeholder_1 forward_lstm_4_while_placeholder$forward_lstm_4/while/Select:output:0*
_output_shapes
: *
element_dtype02;
9forward_lstm_4/while/TensorArrayV2Write/TensorListSetItemz
forward_lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_4/while/add/y�
forward_lstm_4/while/addAddV2 forward_lstm_4_while_placeholder#forward_lstm_4/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/while/add~
forward_lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_4/while/add_1/y�
forward_lstm_4/while/add_1AddV26forward_lstm_4_while_forward_lstm_4_while_loop_counter%forward_lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/while/add_1�
forward_lstm_4/while/IdentityIdentityforward_lstm_4/while/add_1:z:0^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2
forward_lstm_4/while/Identity�
forward_lstm_4/while/Identity_1Identity<forward_lstm_4_while_forward_lstm_4_while_maximum_iterations^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_4/while/Identity_1�
forward_lstm_4/while/Identity_2Identityforward_lstm_4/while/add:z:0^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_4/while/Identity_2�
forward_lstm_4/while/Identity_3IdentityIforward_lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_4/while/Identity_3�
forward_lstm_4/while/Identity_4Identity$forward_lstm_4/while/Select:output:0^forward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22!
forward_lstm_4/while/Identity_4�
forward_lstm_4/while/Identity_5Identity&forward_lstm_4/while/Select_1:output:0^forward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22!
forward_lstm_4/while/Identity_5�
forward_lstm_4/while/Identity_6Identity&forward_lstm_4/while/Select_2:output:0^forward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22!
forward_lstm_4/while/Identity_6�
forward_lstm_4/while/NoOpNoOp9^forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp8^forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp:^forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_4/while/NoOp"l
3forward_lstm_4_while_forward_lstm_4_strided_slice_15forward_lstm_4_while_forward_lstm_4_strided_slice_1_0"f
0forward_lstm_4_while_greater_forward_lstm_4_cast2forward_lstm_4_while_greater_forward_lstm_4_cast_0"G
forward_lstm_4_while_identity&forward_lstm_4/while/Identity:output:0"K
forward_lstm_4_while_identity_1(forward_lstm_4/while/Identity_1:output:0"K
forward_lstm_4_while_identity_2(forward_lstm_4/while/Identity_2:output:0"K
forward_lstm_4_while_identity_3(forward_lstm_4/while/Identity_3:output:0"K
forward_lstm_4_while_identity_4(forward_lstm_4/while/Identity_4:output:0"K
forward_lstm_4_while_identity_5(forward_lstm_4/while/Identity_5:output:0"K
forward_lstm_4_while_identity_6(forward_lstm_4/while/Identity_6:output:0"�
Aforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resourceCforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0"�
Bforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resourceDforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0"�
@forward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resourceBforward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0"�
oforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensorqforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2t
8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp2r
7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp2v
9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp: 
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
�a
�
__inference__traced_save_696633
file_prefix-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopQ
Msavev2_bidirectional_4_forward_lstm_4_lstm_cell_13_kernel_read_readvariableop[
Wsavev2_bidirectional_4_forward_lstm_4_lstm_cell_13_recurrent_kernel_read_readvariableopO
Ksavev2_bidirectional_4_forward_lstm_4_lstm_cell_13_bias_read_readvariableopR
Nsavev2_bidirectional_4_backward_lstm_4_lstm_cell_14_kernel_read_readvariableop\
Xsavev2_bidirectional_4_backward_lstm_4_lstm_cell_14_recurrent_kernel_read_readvariableopP
Lsavev2_bidirectional_4_backward_lstm_4_lstm_cell_14_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableopX
Tsavev2_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_kernel_m_read_readvariableopb
^savev2_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_recurrent_kernel_m_read_readvariableopV
Rsavev2_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_bias_m_read_readvariableopY
Usavev2_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_kernel_m_read_readvariableopc
_savev2_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_recurrent_kernel_m_read_readvariableopW
Ssavev2_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableopX
Tsavev2_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_kernel_v_read_readvariableopb
^savev2_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_recurrent_kernel_v_read_readvariableopV
Rsavev2_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_bias_v_read_readvariableopY
Usavev2_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_kernel_v_read_readvariableopc
_savev2_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_recurrent_kernel_v_read_readvariableopW
Ssavev2_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_bias_v_read_readvariableop7
3savev2_adam_dense_4_kernel_vhat_read_readvariableop5
1savev2_adam_dense_4_bias_vhat_read_readvariableop[
Wsavev2_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_kernel_vhat_read_readvariableope
asavev2_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_recurrent_kernel_vhat_read_readvariableopY
Usavev2_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_bias_vhat_read_readvariableop\
Xsavev2_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_kernel_vhat_read_readvariableopf
bsavev2_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_recurrent_kernel_vhat_read_readvariableopZ
Vsavev2_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_bias_vhat_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopMsavev2_bidirectional_4_forward_lstm_4_lstm_cell_13_kernel_read_readvariableopWsavev2_bidirectional_4_forward_lstm_4_lstm_cell_13_recurrent_kernel_read_readvariableopKsavev2_bidirectional_4_forward_lstm_4_lstm_cell_13_bias_read_readvariableopNsavev2_bidirectional_4_backward_lstm_4_lstm_cell_14_kernel_read_readvariableopXsavev2_bidirectional_4_backward_lstm_4_lstm_cell_14_recurrent_kernel_read_readvariableopLsavev2_bidirectional_4_backward_lstm_4_lstm_cell_14_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableopTsavev2_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_kernel_m_read_readvariableop^savev2_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_recurrent_kernel_m_read_readvariableopRsavev2_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_bias_m_read_readvariableopUsavev2_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_kernel_m_read_readvariableop_savev2_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_recurrent_kernel_m_read_readvariableopSsavev2_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableopTsavev2_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_kernel_v_read_readvariableop^savev2_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_recurrent_kernel_v_read_readvariableopRsavev2_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_bias_v_read_readvariableopUsavev2_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_kernel_v_read_readvariableop_savev2_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_recurrent_kernel_v_read_readvariableopSsavev2_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_bias_v_read_readvariableop3savev2_adam_dense_4_kernel_vhat_read_readvariableop1savev2_adam_dense_4_bias_vhat_read_readvariableopWsavev2_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_kernel_vhat_read_readvariableopasavev2_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_recurrent_kernel_vhat_read_readvariableopUsavev2_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_bias_vhat_read_readvariableopXsavev2_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_kernel_vhat_read_readvariableopbsavev2_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_recurrent_kernel_vhat_read_readvariableopVsavev2_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_bias_vhat_read_readvariableopsavev2_const"/device:CPU:0*
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
�
�
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_690747

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
while_body_695753
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_14_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_14_matmul_readvariableop_resource:	�F
3while_lstm_cell_14_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_14_biasadd_readvariableop_resource:	���)while/lstm_cell_14/BiasAdd/ReadVariableOp�(while/lstm_cell_14/MatMul/ReadVariableOp�*while/lstm_cell_14/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp�
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/MatMul�
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp�
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/MatMul_1�
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/add�
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp�
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/BiasAdd�
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim�
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_14/split�
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Sigmoid�
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Sigmoid_1�
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/mul�
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Relu�
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/mul_1�
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/add_1�
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Sigmoid_2�
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Relu_1�
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
while_cond_691246
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_691246___redundant_placeholder04
0while_while_cond_691246___redundant_placeholder14
0while_while_cond_691246___redundant_placeholder24
0while_while_cond_691246___redundant_placeholder3
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
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_692524

inputs(
forward_lstm_4_692507:	�(
forward_lstm_4_692509:	2�$
forward_lstm_4_692511:	�)
backward_lstm_4_692514:	�)
backward_lstm_4_692516:	2�%
backward_lstm_4_692518:	�
identity��'backward_lstm_4/StatefulPartitionedCall�&forward_lstm_4/StatefulPartitionedCall�
&forward_lstm_4/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_4_692507forward_lstm_4_692509forward_lstm_4_692511*
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
GPU 2J 8� *S
fNRL
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_6924762(
&forward_lstm_4/StatefulPartitionedCall�
'backward_lstm_4/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_4_692514backward_lstm_4_692516backward_lstm_4_692518*
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
GPU 2J 8� *T
fORM
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_6923032)
'backward_lstm_4/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2/forward_lstm_4/StatefulPartitionedCall:output:00backward_lstm_4/StatefulPartitionedCall:output:0concat/axis:output:0*
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
NoOpNoOp(^backward_lstm_4/StatefulPartitionedCall'^forward_lstm_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 2R
'backward_lstm_4/StatefulPartitionedCall'backward_lstm_4/StatefulPartitionedCall2P
&forward_lstm_4/StatefulPartitionedCall&forward_lstm_4/StatefulPartitionedCall:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_696362

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
�
�
>sequential_4_bidirectional_4_backward_lstm_4_while_cond_690421v
rsequential_4_bidirectional_4_backward_lstm_4_while_sequential_4_bidirectional_4_backward_lstm_4_while_loop_counter|
xsequential_4_bidirectional_4_backward_lstm_4_while_sequential_4_bidirectional_4_backward_lstm_4_while_maximum_iterationsB
>sequential_4_bidirectional_4_backward_lstm_4_while_placeholderD
@sequential_4_bidirectional_4_backward_lstm_4_while_placeholder_1D
@sequential_4_bidirectional_4_backward_lstm_4_while_placeholder_2D
@sequential_4_bidirectional_4_backward_lstm_4_while_placeholder_3D
@sequential_4_bidirectional_4_backward_lstm_4_while_placeholder_4x
tsequential_4_bidirectional_4_backward_lstm_4_while_less_sequential_4_bidirectional_4_backward_lstm_4_strided_slice_1�
�sequential_4_bidirectional_4_backward_lstm_4_while_sequential_4_bidirectional_4_backward_lstm_4_while_cond_690421___redundant_placeholder0�
�sequential_4_bidirectional_4_backward_lstm_4_while_sequential_4_bidirectional_4_backward_lstm_4_while_cond_690421___redundant_placeholder1�
�sequential_4_bidirectional_4_backward_lstm_4_while_sequential_4_bidirectional_4_backward_lstm_4_while_cond_690421___redundant_placeholder2�
�sequential_4_bidirectional_4_backward_lstm_4_while_sequential_4_bidirectional_4_backward_lstm_4_while_cond_690421___redundant_placeholder3�
�sequential_4_bidirectional_4_backward_lstm_4_while_sequential_4_bidirectional_4_backward_lstm_4_while_cond_690421___redundant_placeholder4?
;sequential_4_bidirectional_4_backward_lstm_4_while_identity
�
7sequential_4/bidirectional_4/backward_lstm_4/while/LessLess>sequential_4_bidirectional_4_backward_lstm_4_while_placeholdertsequential_4_bidirectional_4_backward_lstm_4_while_less_sequential_4_bidirectional_4_backward_lstm_4_strided_slice_1*
T0*
_output_shapes
: 29
7sequential_4/bidirectional_4/backward_lstm_4/while/Less�
;sequential_4/bidirectional_4/backward_lstm_4/while/IdentityIdentity;sequential_4/bidirectional_4/backward_lstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2=
;sequential_4/bidirectional_4/backward_lstm_4/while/Identity"�
;sequential_4_bidirectional_4_backward_lstm_4_while_identityDsequential_4/bidirectional_4/backward_lstm_4/while/Identity:output:0*(
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
��
�
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_694256
inputs_0M
:forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource:	�O
<forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource:	2�J
;forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource:	�N
;backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource:	�P
=backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource:	2�K
<backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource:	�
identity��3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp�2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp�4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp�backward_lstm_4/while�2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp�1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp�3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp�forward_lstm_4/whiled
forward_lstm_4/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_4/Shape�
"forward_lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"forward_lstm_4/strided_slice/stack�
$forward_lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_4/strided_slice/stack_1�
$forward_lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_4/strided_slice/stack_2�
forward_lstm_4/strided_sliceStridedSliceforward_lstm_4/Shape:output:0+forward_lstm_4/strided_slice/stack:output:0-forward_lstm_4/strided_slice/stack_1:output:0-forward_lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_4/strided_slicez
forward_lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_4/zeros/mul/y�
forward_lstm_4/zeros/mulMul%forward_lstm_4/strided_slice:output:0#forward_lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros/mul}
forward_lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_4/zeros/Less/y�
forward_lstm_4/zeros/LessLessforward_lstm_4/zeros/mul:z:0$forward_lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros/Less�
forward_lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_4/zeros/packed/1�
forward_lstm_4/zeros/packedPack%forward_lstm_4/strided_slice:output:0&forward_lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_4/zeros/packed�
forward_lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_4/zeros/Const�
forward_lstm_4/zerosFill$forward_lstm_4/zeros/packed:output:0#forward_lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_4/zeros~
forward_lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_4/zeros_1/mul/y�
forward_lstm_4/zeros_1/mulMul%forward_lstm_4/strided_slice:output:0%forward_lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros_1/mul�
forward_lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_4/zeros_1/Less/y�
forward_lstm_4/zeros_1/LessLessforward_lstm_4/zeros_1/mul:z:0&forward_lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros_1/Less�
forward_lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_4/zeros_1/packed/1�
forward_lstm_4/zeros_1/packedPack%forward_lstm_4/strided_slice:output:0(forward_lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_4/zeros_1/packed�
forward_lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_4/zeros_1/Const�
forward_lstm_4/zeros_1Fill&forward_lstm_4/zeros_1/packed:output:0%forward_lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_4/zeros_1�
forward_lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
forward_lstm_4/transpose/perm�
forward_lstm_4/transpose	Transposeinputs_0&forward_lstm_4/transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
forward_lstm_4/transpose|
forward_lstm_4/Shape_1Shapeforward_lstm_4/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_4/Shape_1�
$forward_lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_4/strided_slice_1/stack�
&forward_lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_1/stack_1�
&forward_lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_1/stack_2�
forward_lstm_4/strided_slice_1StridedSliceforward_lstm_4/Shape_1:output:0-forward_lstm_4/strided_slice_1/stack:output:0/forward_lstm_4/strided_slice_1/stack_1:output:0/forward_lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_4/strided_slice_1�
*forward_lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2,
*forward_lstm_4/TensorArrayV2/element_shape�
forward_lstm_4/TensorArrayV2TensorListReserve3forward_lstm_4/TensorArrayV2/element_shape:output:0'forward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_4/TensorArrayV2�
Dforward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2F
Dforward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape�
6forward_lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_4/transpose:y:0Mforward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6forward_lstm_4/TensorArrayUnstack/TensorListFromTensor�
$forward_lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_4/strided_slice_2/stack�
&forward_lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_2/stack_1�
&forward_lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_2/stack_2�
forward_lstm_4/strided_slice_2StridedSliceforward_lstm_4/transpose:y:0-forward_lstm_4/strided_slice_2/stack:output:0/forward_lstm_4/strided_slice_2/stack_1:output:0/forward_lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2 
forward_lstm_4/strided_slice_2�
1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp:forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype023
1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp�
"forward_lstm_4/lstm_cell_13/MatMulMatMul'forward_lstm_4/strided_slice_2:output:09forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"forward_lstm_4/lstm_cell_13/MatMul�
3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp<forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype025
3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp�
$forward_lstm_4/lstm_cell_13/MatMul_1MatMulforward_lstm_4/zeros:output:0;forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$forward_lstm_4/lstm_cell_13/MatMul_1�
forward_lstm_4/lstm_cell_13/addAddV2,forward_lstm_4/lstm_cell_13/MatMul:product:0.forward_lstm_4/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2!
forward_lstm_4/lstm_cell_13/add�
2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp;forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype024
2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp�
#forward_lstm_4/lstm_cell_13/BiasAddBiasAdd#forward_lstm_4/lstm_cell_13/add:z:0:forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#forward_lstm_4/lstm_cell_13/BiasAdd�
+forward_lstm_4/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+forward_lstm_4/lstm_cell_13/split/split_dim�
!forward_lstm_4/lstm_cell_13/splitSplit4forward_lstm_4/lstm_cell_13/split/split_dim:output:0,forward_lstm_4/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2#
!forward_lstm_4/lstm_cell_13/split�
#forward_lstm_4/lstm_cell_13/SigmoidSigmoid*forward_lstm_4/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22%
#forward_lstm_4/lstm_cell_13/Sigmoid�
%forward_lstm_4/lstm_cell_13/Sigmoid_1Sigmoid*forward_lstm_4/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22'
%forward_lstm_4/lstm_cell_13/Sigmoid_1�
forward_lstm_4/lstm_cell_13/mulMul)forward_lstm_4/lstm_cell_13/Sigmoid_1:y:0forward_lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:���������22!
forward_lstm_4/lstm_cell_13/mul�
 forward_lstm_4/lstm_cell_13/ReluRelu*forward_lstm_4/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22"
 forward_lstm_4/lstm_cell_13/Relu�
!forward_lstm_4/lstm_cell_13/mul_1Mul'forward_lstm_4/lstm_cell_13/Sigmoid:y:0.forward_lstm_4/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22#
!forward_lstm_4/lstm_cell_13/mul_1�
!forward_lstm_4/lstm_cell_13/add_1AddV2#forward_lstm_4/lstm_cell_13/mul:z:0%forward_lstm_4/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22#
!forward_lstm_4/lstm_cell_13/add_1�
%forward_lstm_4/lstm_cell_13/Sigmoid_2Sigmoid*forward_lstm_4/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22'
%forward_lstm_4/lstm_cell_13/Sigmoid_2�
"forward_lstm_4/lstm_cell_13/Relu_1Relu%forward_lstm_4/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_4/lstm_cell_13/Relu_1�
!forward_lstm_4/lstm_cell_13/mul_2Mul)forward_lstm_4/lstm_cell_13/Sigmoid_2:y:00forward_lstm_4/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22#
!forward_lstm_4/lstm_cell_13/mul_2�
,forward_lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2.
,forward_lstm_4/TensorArrayV2_1/element_shape�
forward_lstm_4/TensorArrayV2_1TensorListReserve5forward_lstm_4/TensorArrayV2_1/element_shape:output:0'forward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_4/TensorArrayV2_1l
forward_lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_4/time�
'forward_lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2)
'forward_lstm_4/while/maximum_iterations�
!forward_lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!forward_lstm_4/while/loop_counter�
forward_lstm_4/whileWhile*forward_lstm_4/while/loop_counter:output:00forward_lstm_4/while/maximum_iterations:output:0forward_lstm_4/time:output:0'forward_lstm_4/TensorArrayV2_1:handle:0forward_lstm_4/zeros:output:0forward_lstm_4/zeros_1:output:0'forward_lstm_4/strided_slice_1:output:0Fforward_lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0:forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource<forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource;forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *,
body$R"
 forward_lstm_4_while_body_694021*,
cond$R"
 forward_lstm_4_while_cond_694020*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
forward_lstm_4/while�
?forward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2A
?forward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape�
1forward_lstm_4/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_4/while:output:3Hforward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype023
1forward_lstm_4/TensorArrayV2Stack/TensorListStack�
$forward_lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2&
$forward_lstm_4/strided_slice_3/stack�
&forward_lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_4/strided_slice_3/stack_1�
&forward_lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_3/stack_2�
forward_lstm_4/strided_slice_3StridedSlice:forward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0-forward_lstm_4/strided_slice_3/stack:output:0/forward_lstm_4/strided_slice_3/stack_1:output:0/forward_lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2 
forward_lstm_4/strided_slice_3�
forward_lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_4/transpose_1/perm�
forward_lstm_4/transpose_1	Transpose:forward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0(forward_lstm_4/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
forward_lstm_4/transpose_1�
forward_lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_4/runtimef
backward_lstm_4/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_4/Shape�
#backward_lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#backward_lstm_4/strided_slice/stack�
%backward_lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_4/strided_slice/stack_1�
%backward_lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_4/strided_slice/stack_2�
backward_lstm_4/strided_sliceStridedSlicebackward_lstm_4/Shape:output:0,backward_lstm_4/strided_slice/stack:output:0.backward_lstm_4/strided_slice/stack_1:output:0.backward_lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
backward_lstm_4/strided_slice|
backward_lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_4/zeros/mul/y�
backward_lstm_4/zeros/mulMul&backward_lstm_4/strided_slice:output:0$backward_lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros/mul
backward_lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
backward_lstm_4/zeros/Less/y�
backward_lstm_4/zeros/LessLessbackward_lstm_4/zeros/mul:z:0%backward_lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros/Less�
backward_lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_4/zeros/packed/1�
backward_lstm_4/zeros/packedPack&backward_lstm_4/strided_slice:output:0'backward_lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_4/zeros/packed�
backward_lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_4/zeros/Const�
backward_lstm_4/zerosFill%backward_lstm_4/zeros/packed:output:0$backward_lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_4/zeros�
backward_lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_4/zeros_1/mul/y�
backward_lstm_4/zeros_1/mulMul&backward_lstm_4/strided_slice:output:0&backward_lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros_1/mul�
backward_lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
backward_lstm_4/zeros_1/Less/y�
backward_lstm_4/zeros_1/LessLessbackward_lstm_4/zeros_1/mul:z:0'backward_lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros_1/Less�
 backward_lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_4/zeros_1/packed/1�
backward_lstm_4/zeros_1/packedPack&backward_lstm_4/strided_slice:output:0)backward_lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_4/zeros_1/packed�
backward_lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_4/zeros_1/Const�
backward_lstm_4/zeros_1Fill'backward_lstm_4/zeros_1/packed:output:0&backward_lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_4/zeros_1�
backward_lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
backward_lstm_4/transpose/perm�
backward_lstm_4/transpose	Transposeinputs_0'backward_lstm_4/transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������2
backward_lstm_4/transpose
backward_lstm_4/Shape_1Shapebackward_lstm_4/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_4/Shape_1�
%backward_lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_4/strided_slice_1/stack�
'backward_lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_1/stack_1�
'backward_lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_1/stack_2�
backward_lstm_4/strided_slice_1StridedSlice backward_lstm_4/Shape_1:output:0.backward_lstm_4/strided_slice_1/stack:output:00backward_lstm_4/strided_slice_1/stack_1:output:00backward_lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_4/strided_slice_1�
+backward_lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+backward_lstm_4/TensorArrayV2/element_shape�
backward_lstm_4/TensorArrayV2TensorListReserve4backward_lstm_4/TensorArrayV2/element_shape:output:0(backward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
backward_lstm_4/TensorArrayV2�
backward_lstm_4/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2 
backward_lstm_4/ReverseV2/axis�
backward_lstm_4/ReverseV2	ReverseV2backward_lstm_4/transpose:y:0'backward_lstm_4/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'���������������������������2
backward_lstm_4/ReverseV2�
Ebackward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2G
Ebackward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape�
7backward_lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"backward_lstm_4/ReverseV2:output:0Nbackward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7backward_lstm_4/TensorArrayUnstack/TensorListFromTensor�
%backward_lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_4/strided_slice_2/stack�
'backward_lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_2/stack_1�
'backward_lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_2/stack_2�
backward_lstm_4/strided_slice_2StridedSlicebackward_lstm_4/transpose:y:0.backward_lstm_4/strided_slice_2/stack:output:00backward_lstm_4/strided_slice_2/stack_1:output:00backward_lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask2!
backward_lstm_4/strided_slice_2�
2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp;backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp�
#backward_lstm_4/lstm_cell_14/MatMulMatMul(backward_lstm_4/strided_slice_2:output:0:backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#backward_lstm_4/lstm_cell_14/MatMul�
4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp=backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype026
4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp�
%backward_lstm_4/lstm_cell_14/MatMul_1MatMulbackward_lstm_4/zeros:output:0<backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%backward_lstm_4/lstm_cell_14/MatMul_1�
 backward_lstm_4/lstm_cell_14/addAddV2-backward_lstm_4/lstm_cell_14/MatMul:product:0/backward_lstm_4/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2"
 backward_lstm_4/lstm_cell_14/add�
3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp<backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp�
$backward_lstm_4/lstm_cell_14/BiasAddBiasAdd$backward_lstm_4/lstm_cell_14/add:z:0;backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$backward_lstm_4/lstm_cell_14/BiasAdd�
,backward_lstm_4/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,backward_lstm_4/lstm_cell_14/split/split_dim�
"backward_lstm_4/lstm_cell_14/splitSplit5backward_lstm_4/lstm_cell_14/split/split_dim:output:0-backward_lstm_4/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2$
"backward_lstm_4/lstm_cell_14/split�
$backward_lstm_4/lstm_cell_14/SigmoidSigmoid+backward_lstm_4/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22&
$backward_lstm_4/lstm_cell_14/Sigmoid�
&backward_lstm_4/lstm_cell_14/Sigmoid_1Sigmoid+backward_lstm_4/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22(
&backward_lstm_4/lstm_cell_14/Sigmoid_1�
 backward_lstm_4/lstm_cell_14/mulMul*backward_lstm_4/lstm_cell_14/Sigmoid_1:y:0 backward_lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/lstm_cell_14/mul�
!backward_lstm_4/lstm_cell_14/ReluRelu+backward_lstm_4/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22#
!backward_lstm_4/lstm_cell_14/Relu�
"backward_lstm_4/lstm_cell_14/mul_1Mul(backward_lstm_4/lstm_cell_14/Sigmoid:y:0/backward_lstm_4/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22$
"backward_lstm_4/lstm_cell_14/mul_1�
"backward_lstm_4/lstm_cell_14/add_1AddV2$backward_lstm_4/lstm_cell_14/mul:z:0&backward_lstm_4/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22$
"backward_lstm_4/lstm_cell_14/add_1�
&backward_lstm_4/lstm_cell_14/Sigmoid_2Sigmoid+backward_lstm_4/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22(
&backward_lstm_4/lstm_cell_14/Sigmoid_2�
#backward_lstm_4/lstm_cell_14/Relu_1Relu&backward_lstm_4/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_4/lstm_cell_14/Relu_1�
"backward_lstm_4/lstm_cell_14/mul_2Mul*backward_lstm_4/lstm_cell_14/Sigmoid_2:y:01backward_lstm_4/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22$
"backward_lstm_4/lstm_cell_14/mul_2�
-backward_lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2/
-backward_lstm_4/TensorArrayV2_1/element_shape�
backward_lstm_4/TensorArrayV2_1TensorListReserve6backward_lstm_4/TensorArrayV2_1/element_shape:output:0(backward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_4/TensorArrayV2_1n
backward_lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_4/time�
(backward_lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(backward_lstm_4/while/maximum_iterations�
"backward_lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"backward_lstm_4/while/loop_counter�
backward_lstm_4/whileWhile+backward_lstm_4/while/loop_counter:output:01backward_lstm_4/while/maximum_iterations:output:0backward_lstm_4/time:output:0(backward_lstm_4/TensorArrayV2_1:handle:0backward_lstm_4/zeros:output:0 backward_lstm_4/zeros_1:output:0(backward_lstm_4/strided_slice_1:output:0Gbackward_lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0;backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource=backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource<backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������2:���������2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *-
body%R#
!backward_lstm_4_while_body_694170*-
cond%R#
!backward_lstm_4_while_cond_694169*K
output_shapes:
8: : : : :���������2:���������2: : : : : *
parallel_iterations 2
backward_lstm_4/while�
@backward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2B
@backward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape�
2backward_lstm_4/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_4/while:output:3Ibackward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype024
2backward_lstm_4/TensorArrayV2Stack/TensorListStack�
%backward_lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2'
%backward_lstm_4/strided_slice_3/stack�
'backward_lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_4/strided_slice_3/stack_1�
'backward_lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_3/stack_2�
backward_lstm_4/strided_slice_3StridedSlice;backward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0.backward_lstm_4/strided_slice_3/stack:output:00backward_lstm_4/strided_slice_3/stack_1:output:00backward_lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2!
backward_lstm_4/strided_slice_3�
 backward_lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_4/transpose_1/perm�
backward_lstm_4/transpose_1	Transpose;backward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0)backward_lstm_4/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
backward_lstm_4/transpose_1�
backward_lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_4/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2'forward_lstm_4/strided_slice_3:output:0(backward_lstm_4/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp4^backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp3^backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp5^backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp^backward_lstm_4/while3^forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp2^forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp4^forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp^forward_lstm_4/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 2j
3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp2h
2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp2l
4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp2.
backward_lstm_4/whilebackward_lstm_4/while2h
2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp2f
1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp2j
3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp2,
forward_lstm_4/whileforward_lstm_4/while:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs/0
�^
�
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_692303

inputs>
+lstm_cell_14_matmul_readvariableop_resource:	�@
-lstm_cell_14_matmul_1_readvariableop_resource:	2�;
,lstm_cell_14_biasadd_readvariableop_resource:	�
identity��#lstm_cell_14/BiasAdd/ReadVariableOp�"lstm_cell_14/MatMul/ReadVariableOp�$lstm_cell_14/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp�
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/MatMul�
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp�
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/MatMul_1�
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/add�
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp�
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/BiasAdd~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim�
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_14/split�
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/Sigmoid�
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_14/Sigmoid_1�
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_14/Relu�
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/mul_1�
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/add_1�
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/Relu_1�
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
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
bodyR
while_body_692219*
condR
while_cond_692218*K
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
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
while_cond_691458
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_691458___redundant_placeholder04
0while_while_cond_691458___redundant_placeholder14
0while_while_cond_691458___redundant_placeholder24
0while_while_cond_691458___redundant_placeholder3
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
�T
�
 forward_lstm_4_while_body_693719:
6forward_lstm_4_while_forward_lstm_4_while_loop_counter@
<forward_lstm_4_while_forward_lstm_4_while_maximum_iterations$
 forward_lstm_4_while_placeholder&
"forward_lstm_4_while_placeholder_1&
"forward_lstm_4_while_placeholder_2&
"forward_lstm_4_while_placeholder_39
5forward_lstm_4_while_forward_lstm_4_strided_slice_1_0u
qforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0U
Bforward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0:	�W
Dforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0:	2�R
Cforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0:	�!
forward_lstm_4_while_identity#
forward_lstm_4_while_identity_1#
forward_lstm_4_while_identity_2#
forward_lstm_4_while_identity_3#
forward_lstm_4_while_identity_4#
forward_lstm_4_while_identity_57
3forward_lstm_4_while_forward_lstm_4_strided_slice_1s
oforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensorS
@forward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource:	�U
Bforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource:	2�P
Aforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource:	���8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp�7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp�9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp�
Fforward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2H
Fforward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape�
8forward_lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0 forward_lstm_4_while_placeholderOforward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02:
8forward_lstm_4/while/TensorArrayV2Read/TensorListGetItem�
7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOpBforward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype029
7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp�
(forward_lstm_4/while/lstm_cell_13/MatMulMatMul?forward_lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0?forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(forward_lstm_4/while/lstm_cell_13/MatMul�
9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOpDforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02;
9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp�
*forward_lstm_4/while/lstm_cell_13/MatMul_1MatMul"forward_lstm_4_while_placeholder_2Aforward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*forward_lstm_4/while/lstm_cell_13/MatMul_1�
%forward_lstm_4/while/lstm_cell_13/addAddV22forward_lstm_4/while/lstm_cell_13/MatMul:product:04forward_lstm_4/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2'
%forward_lstm_4/while/lstm_cell_13/add�
8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOpCforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02:
8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp�
)forward_lstm_4/while/lstm_cell_13/BiasAddBiasAdd)forward_lstm_4/while/lstm_cell_13/add:z:0@forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)forward_lstm_4/while/lstm_cell_13/BiasAdd�
1forward_lstm_4/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1forward_lstm_4/while/lstm_cell_13/split/split_dim�
'forward_lstm_4/while/lstm_cell_13/splitSplit:forward_lstm_4/while/lstm_cell_13/split/split_dim:output:02forward_lstm_4/while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2)
'forward_lstm_4/while/lstm_cell_13/split�
)forward_lstm_4/while/lstm_cell_13/SigmoidSigmoid0forward_lstm_4/while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22+
)forward_lstm_4/while/lstm_cell_13/Sigmoid�
+forward_lstm_4/while/lstm_cell_13/Sigmoid_1Sigmoid0forward_lstm_4/while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22-
+forward_lstm_4/while/lstm_cell_13/Sigmoid_1�
%forward_lstm_4/while/lstm_cell_13/mulMul/forward_lstm_4/while/lstm_cell_13/Sigmoid_1:y:0"forward_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:���������22'
%forward_lstm_4/while/lstm_cell_13/mul�
&forward_lstm_4/while/lstm_cell_13/ReluRelu0forward_lstm_4/while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22(
&forward_lstm_4/while/lstm_cell_13/Relu�
'forward_lstm_4/while/lstm_cell_13/mul_1Mul-forward_lstm_4/while/lstm_cell_13/Sigmoid:y:04forward_lstm_4/while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22)
'forward_lstm_4/while/lstm_cell_13/mul_1�
'forward_lstm_4/while/lstm_cell_13/add_1AddV2)forward_lstm_4/while/lstm_cell_13/mul:z:0+forward_lstm_4/while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22)
'forward_lstm_4/while/lstm_cell_13/add_1�
+forward_lstm_4/while/lstm_cell_13/Sigmoid_2Sigmoid0forward_lstm_4/while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22-
+forward_lstm_4/while/lstm_cell_13/Sigmoid_2�
(forward_lstm_4/while/lstm_cell_13/Relu_1Relu+forward_lstm_4/while/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_4/while/lstm_cell_13/Relu_1�
'forward_lstm_4/while/lstm_cell_13/mul_2Mul/forward_lstm_4/while/lstm_cell_13/Sigmoid_2:y:06forward_lstm_4/while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22)
'forward_lstm_4/while/lstm_cell_13/mul_2�
9forward_lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"forward_lstm_4_while_placeholder_1 forward_lstm_4_while_placeholder+forward_lstm_4/while/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype02;
9forward_lstm_4/while/TensorArrayV2Write/TensorListSetItemz
forward_lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_4/while/add/y�
forward_lstm_4/while/addAddV2 forward_lstm_4_while_placeholder#forward_lstm_4/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/while/add~
forward_lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_4/while/add_1/y�
forward_lstm_4/while/add_1AddV26forward_lstm_4_while_forward_lstm_4_while_loop_counter%forward_lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/while/add_1�
forward_lstm_4/while/IdentityIdentityforward_lstm_4/while/add_1:z:0^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2
forward_lstm_4/while/Identity�
forward_lstm_4/while/Identity_1Identity<forward_lstm_4_while_forward_lstm_4_while_maximum_iterations^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_4/while/Identity_1�
forward_lstm_4/while/Identity_2Identityforward_lstm_4/while/add:z:0^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_4/while/Identity_2�
forward_lstm_4/while/Identity_3IdentityIforward_lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_4/while/Identity_3�
forward_lstm_4/while/Identity_4Identity+forward_lstm_4/while/lstm_cell_13/mul_2:z:0^forward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22!
forward_lstm_4/while/Identity_4�
forward_lstm_4/while/Identity_5Identity+forward_lstm_4/while/lstm_cell_13/add_1:z:0^forward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22!
forward_lstm_4/while/Identity_5�
forward_lstm_4/while/NoOpNoOp9^forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp8^forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp:^forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_4/while/NoOp"l
3forward_lstm_4_while_forward_lstm_4_strided_slice_15forward_lstm_4_while_forward_lstm_4_strided_slice_1_0"G
forward_lstm_4_while_identity&forward_lstm_4/while/Identity:output:0"K
forward_lstm_4_while_identity_1(forward_lstm_4/while/Identity_1:output:0"K
forward_lstm_4_while_identity_2(forward_lstm_4/while/Identity_2:output:0"K
forward_lstm_4_while_identity_3(forward_lstm_4/while/Identity_3:output:0"K
forward_lstm_4_while_identity_4(forward_lstm_4/while/Identity_4:output:0"K
forward_lstm_4_while_identity_5(forward_lstm_4/while/Identity_5:output:0"�
Aforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resourceCforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0"�
Bforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resourceDforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0"�
@forward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resourceBforward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0"�
oforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensorqforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2t
8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp2r
7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp2v
9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp: 
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
0__inference_backward_lstm_4_layer_call_fn_695662
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
GPU 2J 8� *T
fORM
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_6915282
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
�H
�
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_691316

inputs&
lstm_cell_14_691234:	�&
lstm_cell_14_691236:	2�"
lstm_cell_14_691238:	�
identity��$lstm_cell_14/StatefulPartitionedCall�whileD
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
$lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_14_691234lstm_cell_14_691236lstm_cell_14_691238*
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
GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_6912332&
$lstm_cell_14/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_14_691234lstm_cell_14_691236lstm_cell_14_691238*
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
bodyR
while_body_691247*
condR
while_cond_691246*K
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
NoOpNoOp%^lstm_cell_14/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_14/StatefulPartitionedCall$lstm_cell_14/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_696394

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
while_cond_692391
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_692391___redundant_placeholder04
0while_while_cond_692391___redundant_placeholder14
0while_while_cond_692391___redundant_placeholder24
0while_while_cond_692391___redundant_placeholder3
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
0__inference_backward_lstm_4_layer_call_fn_695651
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
GPU 2J 8� *T
fORM
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_6913162
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
�b
�
!backward_lstm_4_while_body_692865<
8backward_lstm_4_while_backward_lstm_4_while_loop_counterB
>backward_lstm_4_while_backward_lstm_4_while_maximum_iterations%
!backward_lstm_4_while_placeholder'
#backward_lstm_4_while_placeholder_1'
#backward_lstm_4_while_placeholder_2'
#backward_lstm_4_while_placeholder_3'
#backward_lstm_4_while_placeholder_4;
7backward_lstm_4_while_backward_lstm_4_strided_slice_1_0w
sbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_06
2backward_lstm_4_while_less_backward_lstm_4_sub_1_0V
Cbackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0:	�X
Ebackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0:	2�S
Dbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0:	�"
backward_lstm_4_while_identity$
 backward_lstm_4_while_identity_1$
 backward_lstm_4_while_identity_2$
 backward_lstm_4_while_identity_3$
 backward_lstm_4_while_identity_4$
 backward_lstm_4_while_identity_5$
 backward_lstm_4_while_identity_69
5backward_lstm_4_while_backward_lstm_4_strided_slice_1u
qbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor4
0backward_lstm_4_while_less_backward_lstm_4_sub_1T
Abackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource:	�V
Cbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource:	2�Q
Bbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource:	���9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp�8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp�:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp�
Gbackward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2I
Gbackward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape�
9backward_lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0!backward_lstm_4_while_placeholderPbackward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02;
9backward_lstm_4/while/TensorArrayV2Read/TensorListGetItem�
backward_lstm_4/while/LessLess2backward_lstm_4_while_less_backward_lstm_4_sub_1_0!backward_lstm_4_while_placeholder*
T0*#
_output_shapes
:���������2
backward_lstm_4/while/Less�
8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOpCbackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02:
8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp�
)backward_lstm_4/while/lstm_cell_14/MatMulMatMul@backward_lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0@backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)backward_lstm_4/while/lstm_cell_14/MatMul�
:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpEbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02<
:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp�
+backward_lstm_4/while/lstm_cell_14/MatMul_1MatMul#backward_lstm_4_while_placeholder_3Bbackward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+backward_lstm_4/while/lstm_cell_14/MatMul_1�
&backward_lstm_4/while/lstm_cell_14/addAddV23backward_lstm_4/while/lstm_cell_14/MatMul:product:05backward_lstm_4/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&backward_lstm_4/while/lstm_cell_14/add�
9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpDbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02;
9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp�
*backward_lstm_4/while/lstm_cell_14/BiasAddBiasAdd*backward_lstm_4/while/lstm_cell_14/add:z:0Abackward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*backward_lstm_4/while/lstm_cell_14/BiasAdd�
2backward_lstm_4/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2backward_lstm_4/while/lstm_cell_14/split/split_dim�
(backward_lstm_4/while/lstm_cell_14/splitSplit;backward_lstm_4/while/lstm_cell_14/split/split_dim:output:03backward_lstm_4/while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2*
(backward_lstm_4/while/lstm_cell_14/split�
*backward_lstm_4/while/lstm_cell_14/SigmoidSigmoid1backward_lstm_4/while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22,
*backward_lstm_4/while/lstm_cell_14/Sigmoid�
,backward_lstm_4/while/lstm_cell_14/Sigmoid_1Sigmoid1backward_lstm_4/while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22.
,backward_lstm_4/while/lstm_cell_14/Sigmoid_1�
&backward_lstm_4/while/lstm_cell_14/mulMul0backward_lstm_4/while/lstm_cell_14/Sigmoid_1:y:0#backward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22(
&backward_lstm_4/while/lstm_cell_14/mul�
'backward_lstm_4/while/lstm_cell_14/ReluRelu1backward_lstm_4/while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22)
'backward_lstm_4/while/lstm_cell_14/Relu�
(backward_lstm_4/while/lstm_cell_14/mul_1Mul.backward_lstm_4/while/lstm_cell_14/Sigmoid:y:05backward_lstm_4/while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22*
(backward_lstm_4/while/lstm_cell_14/mul_1�
(backward_lstm_4/while/lstm_cell_14/add_1AddV2*backward_lstm_4/while/lstm_cell_14/mul:z:0,backward_lstm_4/while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22*
(backward_lstm_4/while/lstm_cell_14/add_1�
,backward_lstm_4/while/lstm_cell_14/Sigmoid_2Sigmoid1backward_lstm_4/while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22.
,backward_lstm_4/while/lstm_cell_14/Sigmoid_2�
)backward_lstm_4/while/lstm_cell_14/Relu_1Relu,backward_lstm_4/while/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_4/while/lstm_cell_14/Relu_1�
(backward_lstm_4/while/lstm_cell_14/mul_2Mul0backward_lstm_4/while/lstm_cell_14/Sigmoid_2:y:07backward_lstm_4/while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22*
(backward_lstm_4/while/lstm_cell_14/mul_2�
backward_lstm_4/while/SelectSelectbackward_lstm_4/while/Less:z:0,backward_lstm_4/while/lstm_cell_14/mul_2:z:0#backward_lstm_4_while_placeholder_2*
T0*'
_output_shapes
:���������22
backward_lstm_4/while/Select�
backward_lstm_4/while/Select_1Selectbackward_lstm_4/while/Less:z:0,backward_lstm_4/while/lstm_cell_14/mul_2:z:0#backward_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:���������22 
backward_lstm_4/while/Select_1�
backward_lstm_4/while/Select_2Selectbackward_lstm_4/while/Less:z:0,backward_lstm_4/while/lstm_cell_14/add_1:z:0#backward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22 
backward_lstm_4/while/Select_2�
:backward_lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#backward_lstm_4_while_placeholder_1!backward_lstm_4_while_placeholder%backward_lstm_4/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:backward_lstm_4/while/TensorArrayV2Write/TensorListSetItem|
backward_lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_4/while/add/y�
backward_lstm_4/while/addAddV2!backward_lstm_4_while_placeholder$backward_lstm_4/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/while/add�
backward_lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_4/while/add_1/y�
backward_lstm_4/while/add_1AddV28backward_lstm_4_while_backward_lstm_4_while_loop_counter&backward_lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/while/add_1�
backward_lstm_4/while/IdentityIdentitybackward_lstm_4/while/add_1:z:0^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2 
backward_lstm_4/while/Identity�
 backward_lstm_4/while/Identity_1Identity>backward_lstm_4_while_backward_lstm_4_while_maximum_iterations^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_4/while/Identity_1�
 backward_lstm_4/while/Identity_2Identitybackward_lstm_4/while/add:z:0^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_4/while/Identity_2�
 backward_lstm_4/while/Identity_3IdentityJbackward_lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_4/while/Identity_3�
 backward_lstm_4/while/Identity_4Identity%backward_lstm_4/while/Select:output:0^backward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/while/Identity_4�
 backward_lstm_4/while/Identity_5Identity'backward_lstm_4/while/Select_1:output:0^backward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/while/Identity_5�
 backward_lstm_4/while/Identity_6Identity'backward_lstm_4/while/Select_2:output:0^backward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/while/Identity_6�
backward_lstm_4/while/NoOpNoOp:^backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp9^backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp;^backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_4/while/NoOp"p
5backward_lstm_4_while_backward_lstm_4_strided_slice_17backward_lstm_4_while_backward_lstm_4_strided_slice_1_0"I
backward_lstm_4_while_identity'backward_lstm_4/while/Identity:output:0"M
 backward_lstm_4_while_identity_1)backward_lstm_4/while/Identity_1:output:0"M
 backward_lstm_4_while_identity_2)backward_lstm_4/while/Identity_2:output:0"M
 backward_lstm_4_while_identity_3)backward_lstm_4/while/Identity_3:output:0"M
 backward_lstm_4_while_identity_4)backward_lstm_4/while/Identity_4:output:0"M
 backward_lstm_4_while_identity_5)backward_lstm_4/while/Identity_5:output:0"M
 backward_lstm_4_while_identity_6)backward_lstm_4/while/Identity_6:output:0"f
0backward_lstm_4_while_less_backward_lstm_4_sub_12backward_lstm_4_while_less_backward_lstm_4_sub_1_0"�
Bbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resourceDbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0"�
Cbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resourceEbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0"�
Abackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resourceCbackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0"�
qbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensorsbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2v
9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp2t
8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp2x
:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
while_body_696059
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_14_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_14_matmul_readvariableop_resource:	�F
3while_lstm_cell_14_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_14_biasadd_readvariableop_resource:	���)while/lstm_cell_14/BiasAdd/ReadVariableOp�(while/lstm_cell_14/MatMul/ReadVariableOp�*while/lstm_cell_14/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp�
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/MatMul�
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp�
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/MatMul_1�
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/add�
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp�
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/BiasAdd�
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim�
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_14/split�
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Sigmoid�
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Sigmoid_1�
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/mul�
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Relu�
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/mul_1�
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/add_1�
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Sigmoid_2�
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Relu_1�
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_696460

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
-__inference_lstm_cell_14_layer_call_fn_696411

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
GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_6912332
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
while_cond_695905
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_695905___redundant_placeholder04
0while_while_cond_695905___redundant_placeholder14
0while_while_cond_695905___redundant_placeholder24
0while_while_cond_695905___redundant_placeholder3
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
�
 forward_lstm_4_while_cond_693718:
6forward_lstm_4_while_forward_lstm_4_while_loop_counter@
<forward_lstm_4_while_forward_lstm_4_while_maximum_iterations$
 forward_lstm_4_while_placeholder&
"forward_lstm_4_while_placeholder_1&
"forward_lstm_4_while_placeholder_2&
"forward_lstm_4_while_placeholder_3<
8forward_lstm_4_while_less_forward_lstm_4_strided_slice_1R
Nforward_lstm_4_while_forward_lstm_4_while_cond_693718___redundant_placeholder0R
Nforward_lstm_4_while_forward_lstm_4_while_cond_693718___redundant_placeholder1R
Nforward_lstm_4_while_forward_lstm_4_while_cond_693718___redundant_placeholder2R
Nforward_lstm_4_while_forward_lstm_4_while_cond_693718___redundant_placeholder3!
forward_lstm_4_while_identity
�
forward_lstm_4/while/LessLess forward_lstm_4_while_placeholder8forward_lstm_4_while_less_forward_lstm_4_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_4/while/Less�
forward_lstm_4/while/IdentityIdentityforward_lstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2
forward_lstm_4/while/Identity"G
forward_lstm_4_while_identity&forward_lstm_4/while/Identity:output:0*(
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
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_692122

inputs(
forward_lstm_4_691952:	�(
forward_lstm_4_691954:	2�$
forward_lstm_4_691956:	�)
backward_lstm_4_692112:	�)
backward_lstm_4_692114:	2�%
backward_lstm_4_692116:	�
identity��'backward_lstm_4/StatefulPartitionedCall�&forward_lstm_4/StatefulPartitionedCall�
&forward_lstm_4/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_4_691952forward_lstm_4_691954forward_lstm_4_691956*
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
GPU 2J 8� *S
fNRL
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_6919512(
&forward_lstm_4/StatefulPartitionedCall�
'backward_lstm_4/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_4_692112backward_lstm_4_692114backward_lstm_4_692116*
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
GPU 2J 8� *T
fORM
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_6921112)
'backward_lstm_4/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2/forward_lstm_4/StatefulPartitionedCall:output:00backward_lstm_4/StatefulPartitionedCall:output:0concat/axis:output:0*
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
NoOpNoOp(^backward_lstm_4/StatefulPartitionedCall'^forward_lstm_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 2R
'backward_lstm_4/StatefulPartitionedCall'backward_lstm_4/StatefulPartitionedCall2P
&forward_lstm_4/StatefulPartitionedCall&forward_lstm_4/StatefulPartitionedCall:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
-__inference_lstm_cell_13_layer_call_fn_696330

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
GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_6907472
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
�?
�
while_body_695103
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_13_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_13_matmul_readvariableop_resource:	�F
3while_lstm_cell_13_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_13_biasadd_readvariableop_resource:	���)while/lstm_cell_13/BiasAdd/ReadVariableOp�(while/lstm_cell_13/MatMul/ReadVariableOp�*while/lstm_cell_13/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp�
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/MatMul�
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp�
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/MatMul_1�
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/add�
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp�
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/BiasAdd�
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim�
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_13/split�
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Sigmoid�
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Sigmoid_1�
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/mul�
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Relu�
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/mul_1�
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/add_1�
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Sigmoid_2�
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Relu_1�
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 
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
�
!backward_lstm_4_while_body_694875<
8backward_lstm_4_while_backward_lstm_4_while_loop_counterB
>backward_lstm_4_while_backward_lstm_4_while_maximum_iterations%
!backward_lstm_4_while_placeholder'
#backward_lstm_4_while_placeholder_1'
#backward_lstm_4_while_placeholder_2'
#backward_lstm_4_while_placeholder_3'
#backward_lstm_4_while_placeholder_4;
7backward_lstm_4_while_backward_lstm_4_strided_slice_1_0w
sbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_06
2backward_lstm_4_while_less_backward_lstm_4_sub_1_0V
Cbackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0:	�X
Ebackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0:	2�S
Dbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0:	�"
backward_lstm_4_while_identity$
 backward_lstm_4_while_identity_1$
 backward_lstm_4_while_identity_2$
 backward_lstm_4_while_identity_3$
 backward_lstm_4_while_identity_4$
 backward_lstm_4_while_identity_5$
 backward_lstm_4_while_identity_69
5backward_lstm_4_while_backward_lstm_4_strided_slice_1u
qbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor4
0backward_lstm_4_while_less_backward_lstm_4_sub_1T
Abackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource:	�V
Cbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource:	2�Q
Bbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource:	���9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp�8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp�:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp�
Gbackward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2I
Gbackward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape�
9backward_lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0!backward_lstm_4_while_placeholderPbackward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02;
9backward_lstm_4/while/TensorArrayV2Read/TensorListGetItem�
backward_lstm_4/while/LessLess2backward_lstm_4_while_less_backward_lstm_4_sub_1_0!backward_lstm_4_while_placeholder*
T0*#
_output_shapes
:���������2
backward_lstm_4/while/Less�
8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOpCbackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02:
8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp�
)backward_lstm_4/while/lstm_cell_14/MatMulMatMul@backward_lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0@backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)backward_lstm_4/while/lstm_cell_14/MatMul�
:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpEbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02<
:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp�
+backward_lstm_4/while/lstm_cell_14/MatMul_1MatMul#backward_lstm_4_while_placeholder_3Bbackward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+backward_lstm_4/while/lstm_cell_14/MatMul_1�
&backward_lstm_4/while/lstm_cell_14/addAddV23backward_lstm_4/while/lstm_cell_14/MatMul:product:05backward_lstm_4/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&backward_lstm_4/while/lstm_cell_14/add�
9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpDbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02;
9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp�
*backward_lstm_4/while/lstm_cell_14/BiasAddBiasAdd*backward_lstm_4/while/lstm_cell_14/add:z:0Abackward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*backward_lstm_4/while/lstm_cell_14/BiasAdd�
2backward_lstm_4/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2backward_lstm_4/while/lstm_cell_14/split/split_dim�
(backward_lstm_4/while/lstm_cell_14/splitSplit;backward_lstm_4/while/lstm_cell_14/split/split_dim:output:03backward_lstm_4/while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2*
(backward_lstm_4/while/lstm_cell_14/split�
*backward_lstm_4/while/lstm_cell_14/SigmoidSigmoid1backward_lstm_4/while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22,
*backward_lstm_4/while/lstm_cell_14/Sigmoid�
,backward_lstm_4/while/lstm_cell_14/Sigmoid_1Sigmoid1backward_lstm_4/while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22.
,backward_lstm_4/while/lstm_cell_14/Sigmoid_1�
&backward_lstm_4/while/lstm_cell_14/mulMul0backward_lstm_4/while/lstm_cell_14/Sigmoid_1:y:0#backward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22(
&backward_lstm_4/while/lstm_cell_14/mul�
'backward_lstm_4/while/lstm_cell_14/ReluRelu1backward_lstm_4/while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22)
'backward_lstm_4/while/lstm_cell_14/Relu�
(backward_lstm_4/while/lstm_cell_14/mul_1Mul.backward_lstm_4/while/lstm_cell_14/Sigmoid:y:05backward_lstm_4/while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22*
(backward_lstm_4/while/lstm_cell_14/mul_1�
(backward_lstm_4/while/lstm_cell_14/add_1AddV2*backward_lstm_4/while/lstm_cell_14/mul:z:0,backward_lstm_4/while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22*
(backward_lstm_4/while/lstm_cell_14/add_1�
,backward_lstm_4/while/lstm_cell_14/Sigmoid_2Sigmoid1backward_lstm_4/while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22.
,backward_lstm_4/while/lstm_cell_14/Sigmoid_2�
)backward_lstm_4/while/lstm_cell_14/Relu_1Relu,backward_lstm_4/while/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_4/while/lstm_cell_14/Relu_1�
(backward_lstm_4/while/lstm_cell_14/mul_2Mul0backward_lstm_4/while/lstm_cell_14/Sigmoid_2:y:07backward_lstm_4/while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22*
(backward_lstm_4/while/lstm_cell_14/mul_2�
backward_lstm_4/while/SelectSelectbackward_lstm_4/while/Less:z:0,backward_lstm_4/while/lstm_cell_14/mul_2:z:0#backward_lstm_4_while_placeholder_2*
T0*'
_output_shapes
:���������22
backward_lstm_4/while/Select�
backward_lstm_4/while/Select_1Selectbackward_lstm_4/while/Less:z:0,backward_lstm_4/while/lstm_cell_14/mul_2:z:0#backward_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:���������22 
backward_lstm_4/while/Select_1�
backward_lstm_4/while/Select_2Selectbackward_lstm_4/while/Less:z:0,backward_lstm_4/while/lstm_cell_14/add_1:z:0#backward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22 
backward_lstm_4/while/Select_2�
:backward_lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#backward_lstm_4_while_placeholder_1!backward_lstm_4_while_placeholder%backward_lstm_4/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:backward_lstm_4/while/TensorArrayV2Write/TensorListSetItem|
backward_lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_4/while/add/y�
backward_lstm_4/while/addAddV2!backward_lstm_4_while_placeholder$backward_lstm_4/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/while/add�
backward_lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_4/while/add_1/y�
backward_lstm_4/while/add_1AddV28backward_lstm_4_while_backward_lstm_4_while_loop_counter&backward_lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/while/add_1�
backward_lstm_4/while/IdentityIdentitybackward_lstm_4/while/add_1:z:0^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2 
backward_lstm_4/while/Identity�
 backward_lstm_4/while/Identity_1Identity>backward_lstm_4_while_backward_lstm_4_while_maximum_iterations^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_4/while/Identity_1�
 backward_lstm_4/while/Identity_2Identitybackward_lstm_4/while/add:z:0^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_4/while/Identity_2�
 backward_lstm_4/while/Identity_3IdentityJbackward_lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_4/while/Identity_3�
 backward_lstm_4/while/Identity_4Identity%backward_lstm_4/while/Select:output:0^backward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/while/Identity_4�
 backward_lstm_4/while/Identity_5Identity'backward_lstm_4/while/Select_1:output:0^backward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/while/Identity_5�
 backward_lstm_4/while/Identity_6Identity'backward_lstm_4/while/Select_2:output:0^backward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/while/Identity_6�
backward_lstm_4/while/NoOpNoOp:^backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp9^backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp;^backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_4/while/NoOp"p
5backward_lstm_4_while_backward_lstm_4_strided_slice_17backward_lstm_4_while_backward_lstm_4_strided_slice_1_0"I
backward_lstm_4_while_identity'backward_lstm_4/while/Identity:output:0"M
 backward_lstm_4_while_identity_1)backward_lstm_4/while/Identity_1:output:0"M
 backward_lstm_4_while_identity_2)backward_lstm_4/while/Identity_2:output:0"M
 backward_lstm_4_while_identity_3)backward_lstm_4/while/Identity_3:output:0"M
 backward_lstm_4_while_identity_4)backward_lstm_4/while/Identity_4:output:0"M
 backward_lstm_4_while_identity_5)backward_lstm_4/while/Identity_5:output:0"M
 backward_lstm_4_while_identity_6)backward_lstm_4/while/Identity_6:output:0"f
0backward_lstm_4_while_less_backward_lstm_4_sub_12backward_lstm_4_while_less_backward_lstm_4_sub_1_0"�
Bbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resourceDbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0"�
Cbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resourceEbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0"�
Abackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resourceCbackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0"�
qbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensorsbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2v
9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp2t
8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp2x
:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
0__inference_backward_lstm_4_layer_call_fn_695684

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
GPU 2J 8� *T
fORM
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_6923032
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
�
�
while_cond_692218
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_692218___redundant_placeholder04
0while_while_cond_692218___redundant_placeholder14
0while_while_cond_692218___redundant_placeholder24
0while_while_cond_692218___redundant_placeholder3
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
�
�
(__inference_dense_4_layer_call_fn_694981

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
GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_6929872
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
�

�
0__inference_bidirectional_4_layer_call_fn_693634

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
GPU 2J 8� *T
fORM
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_6929622
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
�
�
while_cond_692026
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_692026___redundant_placeholder04
0while_while_cond_692026___redundant_placeholder14
0while_while_cond_692026___redundant_placeholder24
0while_while_cond_692026___redundant_placeholder3
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
�V
�
!backward_lstm_4_while_body_693868<
8backward_lstm_4_while_backward_lstm_4_while_loop_counterB
>backward_lstm_4_while_backward_lstm_4_while_maximum_iterations%
!backward_lstm_4_while_placeholder'
#backward_lstm_4_while_placeholder_1'
#backward_lstm_4_while_placeholder_2'
#backward_lstm_4_while_placeholder_3;
7backward_lstm_4_while_backward_lstm_4_strided_slice_1_0w
sbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0V
Cbackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0:	�X
Ebackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0:	2�S
Dbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0:	�"
backward_lstm_4_while_identity$
 backward_lstm_4_while_identity_1$
 backward_lstm_4_while_identity_2$
 backward_lstm_4_while_identity_3$
 backward_lstm_4_while_identity_4$
 backward_lstm_4_while_identity_59
5backward_lstm_4_while_backward_lstm_4_strided_slice_1u
qbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensorT
Abackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource:	�V
Cbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource:	2�Q
Bbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource:	���9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp�8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp�:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp�
Gbackward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2I
Gbackward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape�
9backward_lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0!backward_lstm_4_while_placeholderPbackward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02;
9backward_lstm_4/while/TensorArrayV2Read/TensorListGetItem�
8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOpCbackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02:
8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp�
)backward_lstm_4/while/lstm_cell_14/MatMulMatMul@backward_lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0@backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)backward_lstm_4/while/lstm_cell_14/MatMul�
:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpEbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02<
:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp�
+backward_lstm_4/while/lstm_cell_14/MatMul_1MatMul#backward_lstm_4_while_placeholder_2Bbackward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+backward_lstm_4/while/lstm_cell_14/MatMul_1�
&backward_lstm_4/while/lstm_cell_14/addAddV23backward_lstm_4/while/lstm_cell_14/MatMul:product:05backward_lstm_4/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&backward_lstm_4/while/lstm_cell_14/add�
9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpDbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02;
9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp�
*backward_lstm_4/while/lstm_cell_14/BiasAddBiasAdd*backward_lstm_4/while/lstm_cell_14/add:z:0Abackward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*backward_lstm_4/while/lstm_cell_14/BiasAdd�
2backward_lstm_4/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2backward_lstm_4/while/lstm_cell_14/split/split_dim�
(backward_lstm_4/while/lstm_cell_14/splitSplit;backward_lstm_4/while/lstm_cell_14/split/split_dim:output:03backward_lstm_4/while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2*
(backward_lstm_4/while/lstm_cell_14/split�
*backward_lstm_4/while/lstm_cell_14/SigmoidSigmoid1backward_lstm_4/while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22,
*backward_lstm_4/while/lstm_cell_14/Sigmoid�
,backward_lstm_4/while/lstm_cell_14/Sigmoid_1Sigmoid1backward_lstm_4/while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22.
,backward_lstm_4/while/lstm_cell_14/Sigmoid_1�
&backward_lstm_4/while/lstm_cell_14/mulMul0backward_lstm_4/while/lstm_cell_14/Sigmoid_1:y:0#backward_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:���������22(
&backward_lstm_4/while/lstm_cell_14/mul�
'backward_lstm_4/while/lstm_cell_14/ReluRelu1backward_lstm_4/while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22)
'backward_lstm_4/while/lstm_cell_14/Relu�
(backward_lstm_4/while/lstm_cell_14/mul_1Mul.backward_lstm_4/while/lstm_cell_14/Sigmoid:y:05backward_lstm_4/while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22*
(backward_lstm_4/while/lstm_cell_14/mul_1�
(backward_lstm_4/while/lstm_cell_14/add_1AddV2*backward_lstm_4/while/lstm_cell_14/mul:z:0,backward_lstm_4/while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22*
(backward_lstm_4/while/lstm_cell_14/add_1�
,backward_lstm_4/while/lstm_cell_14/Sigmoid_2Sigmoid1backward_lstm_4/while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22.
,backward_lstm_4/while/lstm_cell_14/Sigmoid_2�
)backward_lstm_4/while/lstm_cell_14/Relu_1Relu,backward_lstm_4/while/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_4/while/lstm_cell_14/Relu_1�
(backward_lstm_4/while/lstm_cell_14/mul_2Mul0backward_lstm_4/while/lstm_cell_14/Sigmoid_2:y:07backward_lstm_4/while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22*
(backward_lstm_4/while/lstm_cell_14/mul_2�
:backward_lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#backward_lstm_4_while_placeholder_1!backward_lstm_4_while_placeholder,backward_lstm_4/while/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:backward_lstm_4/while/TensorArrayV2Write/TensorListSetItem|
backward_lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_4/while/add/y�
backward_lstm_4/while/addAddV2!backward_lstm_4_while_placeholder$backward_lstm_4/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/while/add�
backward_lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_4/while/add_1/y�
backward_lstm_4/while/add_1AddV28backward_lstm_4_while_backward_lstm_4_while_loop_counter&backward_lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/while/add_1�
backward_lstm_4/while/IdentityIdentitybackward_lstm_4/while/add_1:z:0^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2 
backward_lstm_4/while/Identity�
 backward_lstm_4/while/Identity_1Identity>backward_lstm_4_while_backward_lstm_4_while_maximum_iterations^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_4/while/Identity_1�
 backward_lstm_4/while/Identity_2Identitybackward_lstm_4/while/add:z:0^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_4/while/Identity_2�
 backward_lstm_4/while/Identity_3IdentityJbackward_lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_4/while/Identity_3�
 backward_lstm_4/while/Identity_4Identity,backward_lstm_4/while/lstm_cell_14/mul_2:z:0^backward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/while/Identity_4�
 backward_lstm_4/while/Identity_5Identity,backward_lstm_4/while/lstm_cell_14/add_1:z:0^backward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/while/Identity_5�
backward_lstm_4/while/NoOpNoOp:^backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp9^backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp;^backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_4/while/NoOp"p
5backward_lstm_4_while_backward_lstm_4_strided_slice_17backward_lstm_4_while_backward_lstm_4_strided_slice_1_0"I
backward_lstm_4_while_identity'backward_lstm_4/while/Identity:output:0"M
 backward_lstm_4_while_identity_1)backward_lstm_4/while/Identity_1:output:0"M
 backward_lstm_4_while_identity_2)backward_lstm_4/while/Identity_2:output:0"M
 backward_lstm_4_while_identity_3)backward_lstm_4/while/Identity_3:output:0"M
 backward_lstm_4_while_identity_4)backward_lstm_4/while/Identity_4:output:0"M
 backward_lstm_4_while_identity_5)backward_lstm_4/while/Identity_5:output:0"�
Bbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resourceDbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0"�
Cbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resourceEbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0"�
Abackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resourceCbackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0"�
qbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensorsbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2v
9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp2t
8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp2x
:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
�
 forward_lstm_4_while_cond_694020:
6forward_lstm_4_while_forward_lstm_4_while_loop_counter@
<forward_lstm_4_while_forward_lstm_4_while_maximum_iterations$
 forward_lstm_4_while_placeholder&
"forward_lstm_4_while_placeholder_1&
"forward_lstm_4_while_placeholder_2&
"forward_lstm_4_while_placeholder_3<
8forward_lstm_4_while_less_forward_lstm_4_strided_slice_1R
Nforward_lstm_4_while_forward_lstm_4_while_cond_694020___redundant_placeholder0R
Nforward_lstm_4_while_forward_lstm_4_while_cond_694020___redundant_placeholder1R
Nforward_lstm_4_while_forward_lstm_4_while_cond_694020___redundant_placeholder2R
Nforward_lstm_4_while_forward_lstm_4_while_cond_694020___redundant_placeholder3!
forward_lstm_4_while_identity
�
forward_lstm_4/while/LessLess forward_lstm_4_while_placeholder8forward_lstm_4_while_less_forward_lstm_4_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_4/while/Less�
forward_lstm_4/while/IdentityIdentityforward_lstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2
forward_lstm_4/while/Identity"G
forward_lstm_4_while_identity&forward_lstm_4/while/Identity:output:0*(
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
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_695489

inputs>
+lstm_cell_13_matmul_readvariableop_resource:	�@
-lstm_cell_13_matmul_1_readvariableop_resource:	2�;
,lstm_cell_13_biasadd_readvariableop_resource:	�
identity��#lstm_cell_13/BiasAdd/ReadVariableOp�"lstm_cell_13/MatMul/ReadVariableOp�$lstm_cell_13/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp�
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/MatMul�
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp�
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/MatMul_1�
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/add�
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp�
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim�
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_13/split�
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/Sigmoid�
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_13/Sigmoid_1�
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/mul}
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_13/Relu�
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/mul_1�
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/add_1�
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_13/Sigmoid_2|
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/Relu_1�
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
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
bodyR
while_body_695405*
condR
while_cond_695404*K
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
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
C__inference_dense_4_layer_call_and_return_conditional_losses_692987

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
�^
�
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_695837
inputs_0>
+lstm_cell_14_matmul_readvariableop_resource:	�@
-lstm_cell_14_matmul_1_readvariableop_resource:	2�;
,lstm_cell_14_biasadd_readvariableop_resource:	�
identity��#lstm_cell_14/BiasAdd/ReadVariableOp�"lstm_cell_14/MatMul/ReadVariableOp�$lstm_cell_14/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp�
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/MatMul�
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp�
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/MatMul_1�
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/add�
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp�
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/BiasAdd~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim�
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_14/split�
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/Sigmoid�
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_14/Sigmoid_1�
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_14/Relu�
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/mul_1�
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/add_1�
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/Relu_1�
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
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
bodyR
while_body_695753*
condR
while_cond_695752*K
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
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
!backward_lstm_4_while_cond_693867<
8backward_lstm_4_while_backward_lstm_4_while_loop_counterB
>backward_lstm_4_while_backward_lstm_4_while_maximum_iterations%
!backward_lstm_4_while_placeholder'
#backward_lstm_4_while_placeholder_1'
#backward_lstm_4_while_placeholder_2'
#backward_lstm_4_while_placeholder_3>
:backward_lstm_4_while_less_backward_lstm_4_strided_slice_1T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_693867___redundant_placeholder0T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_693867___redundant_placeholder1T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_693867___redundant_placeholder2T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_693867___redundant_placeholder3"
backward_lstm_4_while_identity
�
backward_lstm_4/while/LessLess!backward_lstm_4_while_placeholder:backward_lstm_4_while_less_backward_lstm_4_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_4/while/Less�
backward_lstm_4/while/IdentityIdentitybackward_lstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2 
backward_lstm_4/while/Identity"I
backward_lstm_4_while_identity'backward_lstm_4/while/Identity:output:0*(
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
while_cond_695752
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_695752___redundant_placeholder04
0while_while_cond_695752___redundant_placeholder14
0while_while_cond_695752___redundant_placeholder24
0while_while_cond_695752___redundant_placeholder3
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
�F
�
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_690894

inputs&
lstm_cell_13_690812:	�&
lstm_cell_13_690814:	2�"
lstm_cell_13_690816:	�
identity��$lstm_cell_13/StatefulPartitionedCall�whileD
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
$lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_13_690812lstm_cell_13_690814lstm_cell_13_690816*
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
GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_6907472&
$lstm_cell_13/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_13_690812lstm_cell_13_690814lstm_cell_13_690816*
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
bodyR
while_body_690825*
condR
while_cond_690824*K
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
NoOpNoOp%^lstm_cell_13/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_13/StatefulPartitionedCall$lstm_cell_13/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
while_cond_695555
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_695555___redundant_placeholder04
0while_while_cond_695555___redundant_placeholder14
0while_while_cond_695555___redundant_placeholder24
0while_while_cond_695555___redundant_placeholder3
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
-__inference_lstm_cell_14_layer_call_fn_696428

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
GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_6913792
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
�?
�
while_body_696212
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_14_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_14_matmul_readvariableop_resource:	�F
3while_lstm_cell_14_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_14_biasadd_readvariableop_resource:	���)while/lstm_cell_14/BiasAdd/ReadVariableOp�(while/lstm_cell_14/MatMul/ReadVariableOp�*while/lstm_cell_14/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp�
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/MatMul�
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp�
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/MatMul_1�
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/add�
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp�
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/BiasAdd�
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim�
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_14/split�
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Sigmoid�
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Sigmoid_1�
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/mul�
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Relu�
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/mul_1�
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/add_1�
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Sigmoid_2�
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Relu_1�
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
while_body_691867
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_13_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_13_matmul_readvariableop_resource:	�F
3while_lstm_cell_13_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_13_biasadd_readvariableop_resource:	���)while/lstm_cell_13/BiasAdd/ReadVariableOp�(while/lstm_cell_13/MatMul/ReadVariableOp�*while/lstm_cell_13/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp�
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/MatMul�
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp�
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/MatMul_1�
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/add�
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp�
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/BiasAdd�
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim�
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_13/split�
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Sigmoid�
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Sigmoid_1�
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/mul�
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Relu�
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/mul_1�
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/add_1�
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Sigmoid_2�
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Relu_1�
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 
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
while_cond_690614
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_690614___redundant_placeholder04
0while_while_cond_690614___redundant_placeholder14
0while_while_cond_690614___redundant_placeholder24
0while_while_cond_690614___redundant_placeholder3
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
�
�
H__inference_sequential_4_layer_call_and_return_conditional_losses_693552

inputs
inputs_1	)
bidirectional_4_693533:	�)
bidirectional_4_693535:	2�%
bidirectional_4_693537:	�)
bidirectional_4_693539:	�)
bidirectional_4_693541:	2�%
bidirectional_4_693543:	� 
dense_4_693546:d
dense_4_693548:
identity��'bidirectional_4/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�
'bidirectional_4/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_4_693533bidirectional_4_693535bidirectional_4_693537bidirectional_4_693539bidirectional_4_693541bidirectional_4_693543*
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
GPU 2J 8� *T
fORM
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_6934022)
'bidirectional_4/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall0bidirectional_4/StatefulPartitionedCall:output:0dense_4_693546dense_4_693548*
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
GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_6929872!
dense_4/StatefulPartitionedCall�
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp(^bidirectional_4/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:���������:���������: : : : : : : : 2R
'bidirectional_4/StatefulPartitionedCall'bidirectional_4/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_forward_lstm_4_layer_call_fn_695003
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
GPU 2J 8� *S
fNRL
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_6906842
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

�
-__inference_sequential_4_layer_call_fn_693506

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
GPU 2J 8� *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_6934652
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
�
�
!backward_lstm_4_while_cond_694516<
8backward_lstm_4_while_backward_lstm_4_while_loop_counterB
>backward_lstm_4_while_backward_lstm_4_while_maximum_iterations%
!backward_lstm_4_while_placeholder'
#backward_lstm_4_while_placeholder_1'
#backward_lstm_4_while_placeholder_2'
#backward_lstm_4_while_placeholder_3'
#backward_lstm_4_while_placeholder_4>
:backward_lstm_4_while_less_backward_lstm_4_strided_slice_1T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_694516___redundant_placeholder0T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_694516___redundant_placeholder1T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_694516___redundant_placeholder2T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_694516___redundant_placeholder3T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_694516___redundant_placeholder4"
backward_lstm_4_while_identity
�
backward_lstm_4/while/LessLess!backward_lstm_4_while_placeholder:backward_lstm_4_while_less_backward_lstm_4_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_4/while/Less�
backward_lstm_4/while/IdentityIdentitybackward_lstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2 
backward_lstm_4/while/Identity"I
backward_lstm_4_while_identity'backward_lstm_4/while/Identity:output:0*(
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
while_cond_695404
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_695404___redundant_placeholder04
0while_while_cond_695404___redundant_placeholder14
0while_while_cond_695404___redundant_placeholder24
0while_while_cond_695404___redundant_placeholder3
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
$__inference_signature_wrapper_693582

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
GPU 2J 8� **
f%R#
!__inference__wrapped_model_6905262
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
�a
�
 forward_lstm_4_while_body_694696:
6forward_lstm_4_while_forward_lstm_4_while_loop_counter@
<forward_lstm_4_while_forward_lstm_4_while_maximum_iterations$
 forward_lstm_4_while_placeholder&
"forward_lstm_4_while_placeholder_1&
"forward_lstm_4_while_placeholder_2&
"forward_lstm_4_while_placeholder_3&
"forward_lstm_4_while_placeholder_49
5forward_lstm_4_while_forward_lstm_4_strided_slice_1_0u
qforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_06
2forward_lstm_4_while_greater_forward_lstm_4_cast_0U
Bforward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0:	�W
Dforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0:	2�R
Cforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0:	�!
forward_lstm_4_while_identity#
forward_lstm_4_while_identity_1#
forward_lstm_4_while_identity_2#
forward_lstm_4_while_identity_3#
forward_lstm_4_while_identity_4#
forward_lstm_4_while_identity_5#
forward_lstm_4_while_identity_67
3forward_lstm_4_while_forward_lstm_4_strided_slice_1s
oforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor4
0forward_lstm_4_while_greater_forward_lstm_4_castS
@forward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource:	�U
Bforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource:	2�P
Aforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource:	���8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp�7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp�9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp�
Fforward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2H
Fforward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape�
8forward_lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0 forward_lstm_4_while_placeholderOforward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02:
8forward_lstm_4/while/TensorArrayV2Read/TensorListGetItem�
forward_lstm_4/while/GreaterGreater2forward_lstm_4_while_greater_forward_lstm_4_cast_0 forward_lstm_4_while_placeholder*
T0*#
_output_shapes
:���������2
forward_lstm_4/while/Greater�
7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOpBforward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype029
7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp�
(forward_lstm_4/while/lstm_cell_13/MatMulMatMul?forward_lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0?forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(forward_lstm_4/while/lstm_cell_13/MatMul�
9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOpDforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02;
9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp�
*forward_lstm_4/while/lstm_cell_13/MatMul_1MatMul"forward_lstm_4_while_placeholder_3Aforward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*forward_lstm_4/while/lstm_cell_13/MatMul_1�
%forward_lstm_4/while/lstm_cell_13/addAddV22forward_lstm_4/while/lstm_cell_13/MatMul:product:04forward_lstm_4/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2'
%forward_lstm_4/while/lstm_cell_13/add�
8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOpCforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02:
8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp�
)forward_lstm_4/while/lstm_cell_13/BiasAddBiasAdd)forward_lstm_4/while/lstm_cell_13/add:z:0@forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)forward_lstm_4/while/lstm_cell_13/BiasAdd�
1forward_lstm_4/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1forward_lstm_4/while/lstm_cell_13/split/split_dim�
'forward_lstm_4/while/lstm_cell_13/splitSplit:forward_lstm_4/while/lstm_cell_13/split/split_dim:output:02forward_lstm_4/while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2)
'forward_lstm_4/while/lstm_cell_13/split�
)forward_lstm_4/while/lstm_cell_13/SigmoidSigmoid0forward_lstm_4/while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22+
)forward_lstm_4/while/lstm_cell_13/Sigmoid�
+forward_lstm_4/while/lstm_cell_13/Sigmoid_1Sigmoid0forward_lstm_4/while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22-
+forward_lstm_4/while/lstm_cell_13/Sigmoid_1�
%forward_lstm_4/while/lstm_cell_13/mulMul/forward_lstm_4/while/lstm_cell_13/Sigmoid_1:y:0"forward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22'
%forward_lstm_4/while/lstm_cell_13/mul�
&forward_lstm_4/while/lstm_cell_13/ReluRelu0forward_lstm_4/while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22(
&forward_lstm_4/while/lstm_cell_13/Relu�
'forward_lstm_4/while/lstm_cell_13/mul_1Mul-forward_lstm_4/while/lstm_cell_13/Sigmoid:y:04forward_lstm_4/while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22)
'forward_lstm_4/while/lstm_cell_13/mul_1�
'forward_lstm_4/while/lstm_cell_13/add_1AddV2)forward_lstm_4/while/lstm_cell_13/mul:z:0+forward_lstm_4/while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22)
'forward_lstm_4/while/lstm_cell_13/add_1�
+forward_lstm_4/while/lstm_cell_13/Sigmoid_2Sigmoid0forward_lstm_4/while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22-
+forward_lstm_4/while/lstm_cell_13/Sigmoid_2�
(forward_lstm_4/while/lstm_cell_13/Relu_1Relu+forward_lstm_4/while/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_4/while/lstm_cell_13/Relu_1�
'forward_lstm_4/while/lstm_cell_13/mul_2Mul/forward_lstm_4/while/lstm_cell_13/Sigmoid_2:y:06forward_lstm_4/while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22)
'forward_lstm_4/while/lstm_cell_13/mul_2�
forward_lstm_4/while/SelectSelect forward_lstm_4/while/Greater:z:0+forward_lstm_4/while/lstm_cell_13/mul_2:z:0"forward_lstm_4_while_placeholder_2*
T0*'
_output_shapes
:���������22
forward_lstm_4/while/Select�
forward_lstm_4/while/Select_1Select forward_lstm_4/while/Greater:z:0+forward_lstm_4/while/lstm_cell_13/mul_2:z:0"forward_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:���������22
forward_lstm_4/while/Select_1�
forward_lstm_4/while/Select_2Select forward_lstm_4/while/Greater:z:0+forward_lstm_4/while/lstm_cell_13/add_1:z:0"forward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22
forward_lstm_4/while/Select_2�
9forward_lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"forward_lstm_4_while_placeholder_1 forward_lstm_4_while_placeholder$forward_lstm_4/while/Select:output:0*
_output_shapes
: *
element_dtype02;
9forward_lstm_4/while/TensorArrayV2Write/TensorListSetItemz
forward_lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_4/while/add/y�
forward_lstm_4/while/addAddV2 forward_lstm_4_while_placeholder#forward_lstm_4/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/while/add~
forward_lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_4/while/add_1/y�
forward_lstm_4/while/add_1AddV26forward_lstm_4_while_forward_lstm_4_while_loop_counter%forward_lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/while/add_1�
forward_lstm_4/while/IdentityIdentityforward_lstm_4/while/add_1:z:0^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2
forward_lstm_4/while/Identity�
forward_lstm_4/while/Identity_1Identity<forward_lstm_4_while_forward_lstm_4_while_maximum_iterations^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_4/while/Identity_1�
forward_lstm_4/while/Identity_2Identityforward_lstm_4/while/add:z:0^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_4/while/Identity_2�
forward_lstm_4/while/Identity_3IdentityIforward_lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_4/while/Identity_3�
forward_lstm_4/while/Identity_4Identity$forward_lstm_4/while/Select:output:0^forward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22!
forward_lstm_4/while/Identity_4�
forward_lstm_4/while/Identity_5Identity&forward_lstm_4/while/Select_1:output:0^forward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22!
forward_lstm_4/while/Identity_5�
forward_lstm_4/while/Identity_6Identity&forward_lstm_4/while/Select_2:output:0^forward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22!
forward_lstm_4/while/Identity_6�
forward_lstm_4/while/NoOpNoOp9^forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp8^forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp:^forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_4/while/NoOp"l
3forward_lstm_4_while_forward_lstm_4_strided_slice_15forward_lstm_4_while_forward_lstm_4_strided_slice_1_0"f
0forward_lstm_4_while_greater_forward_lstm_4_cast2forward_lstm_4_while_greater_forward_lstm_4_cast_0"G
forward_lstm_4_while_identity&forward_lstm_4/while/Identity:output:0"K
forward_lstm_4_while_identity_1(forward_lstm_4/while/Identity_1:output:0"K
forward_lstm_4_while_identity_2(forward_lstm_4/while/Identity_2:output:0"K
forward_lstm_4_while_identity_3(forward_lstm_4/while/Identity_3:output:0"K
forward_lstm_4_while_identity_4(forward_lstm_4/while/Identity_4:output:0"K
forward_lstm_4_while_identity_5(forward_lstm_4/while/Identity_5:output:0"K
forward_lstm_4_while_identity_6(forward_lstm_4/while/Identity_6:output:0"�
Aforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resourceCforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0"�
Bforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resourceDforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0"�
@forward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resourceBforward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0"�
oforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensorqforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2t
8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp2r
7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp2v
9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp: 
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
�
!backward_lstm_4_while_cond_694169<
8backward_lstm_4_while_backward_lstm_4_while_loop_counterB
>backward_lstm_4_while_backward_lstm_4_while_maximum_iterations%
!backward_lstm_4_while_placeholder'
#backward_lstm_4_while_placeholder_1'
#backward_lstm_4_while_placeholder_2'
#backward_lstm_4_while_placeholder_3>
:backward_lstm_4_while_less_backward_lstm_4_strided_slice_1T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_694169___redundant_placeholder0T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_694169___redundant_placeholder1T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_694169___redundant_placeholder2T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_694169___redundant_placeholder3"
backward_lstm_4_while_identity
�
backward_lstm_4/while/LessLess!backward_lstm_4_while_placeholder:backward_lstm_4_while_less_backward_lstm_4_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_4/while/Less�
backward_lstm_4/while/IdentityIdentitybackward_lstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2 
backward_lstm_4/while/Identity"I
backward_lstm_4_while_identity'backward_lstm_4/while/Identity:output:0*(
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
Ʋ
�
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_694972

inputs
inputs_1	M
:forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource:	�O
<forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource:	2�J
;forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource:	�N
;backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource:	�P
=backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource:	2�K
<backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource:	�
identity��3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp�2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp�4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp�backward_lstm_4/while�2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp�1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp�3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp�forward_lstm_4/while�
#forward_lstm_4/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2%
#forward_lstm_4/RaggedToTensor/zeros�
#forward_lstm_4/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2%
#forward_lstm_4/RaggedToTensor/Const�
2forward_lstm_4/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor,forward_lstm_4/RaggedToTensor/Const:output:0inputs,forward_lstm_4/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS24
2forward_lstm_4/RaggedToTensor/RaggedTensorToTensor�
9forward_lstm_4/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack�
;forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1�
;forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2�
3forward_lstm_4/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Bforward_lstm_4/RaggedNestedRowLengths/strided_slice/stack:output:0Dforward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1:output:0Dforward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask25
3forward_lstm_4/RaggedNestedRowLengths/strided_slice�
;forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack�
=forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2?
=forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1�
=forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2�
5forward_lstm_4/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Dforward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack:output:0Fforward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Fforward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask27
5forward_lstm_4/RaggedNestedRowLengths/strided_slice_1�
)forward_lstm_4/RaggedNestedRowLengths/subSub<forward_lstm_4/RaggedNestedRowLengths/strided_slice:output:0>forward_lstm_4/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2+
)forward_lstm_4/RaggedNestedRowLengths/sub�
forward_lstm_4/CastCast-forward_lstm_4/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
forward_lstm_4/Cast�
forward_lstm_4/ShapeShape;forward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_4/Shape�
"forward_lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"forward_lstm_4/strided_slice/stack�
$forward_lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_4/strided_slice/stack_1�
$forward_lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_4/strided_slice/stack_2�
forward_lstm_4/strided_sliceStridedSliceforward_lstm_4/Shape:output:0+forward_lstm_4/strided_slice/stack:output:0-forward_lstm_4/strided_slice/stack_1:output:0-forward_lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_4/strided_slicez
forward_lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_4/zeros/mul/y�
forward_lstm_4/zeros/mulMul%forward_lstm_4/strided_slice:output:0#forward_lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros/mul}
forward_lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_4/zeros/Less/y�
forward_lstm_4/zeros/LessLessforward_lstm_4/zeros/mul:z:0$forward_lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros/Less�
forward_lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_4/zeros/packed/1�
forward_lstm_4/zeros/packedPack%forward_lstm_4/strided_slice:output:0&forward_lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_4/zeros/packed�
forward_lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_4/zeros/Const�
forward_lstm_4/zerosFill$forward_lstm_4/zeros/packed:output:0#forward_lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_4/zeros~
forward_lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_4/zeros_1/mul/y�
forward_lstm_4/zeros_1/mulMul%forward_lstm_4/strided_slice:output:0%forward_lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros_1/mul�
forward_lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_4/zeros_1/Less/y�
forward_lstm_4/zeros_1/LessLessforward_lstm_4/zeros_1/mul:z:0&forward_lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros_1/Less�
forward_lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_4/zeros_1/packed/1�
forward_lstm_4/zeros_1/packedPack%forward_lstm_4/strided_slice:output:0(forward_lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_4/zeros_1/packed�
forward_lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_4/zeros_1/Const�
forward_lstm_4/zeros_1Fill&forward_lstm_4/zeros_1/packed:output:0%forward_lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_4/zeros_1�
forward_lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
forward_lstm_4/transpose/perm�
forward_lstm_4/transpose	Transpose;forward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0&forward_lstm_4/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
forward_lstm_4/transpose|
forward_lstm_4/Shape_1Shapeforward_lstm_4/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_4/Shape_1�
$forward_lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_4/strided_slice_1/stack�
&forward_lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_1/stack_1�
&forward_lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_1/stack_2�
forward_lstm_4/strided_slice_1StridedSliceforward_lstm_4/Shape_1:output:0-forward_lstm_4/strided_slice_1/stack:output:0/forward_lstm_4/strided_slice_1/stack_1:output:0/forward_lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_4/strided_slice_1�
*forward_lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2,
*forward_lstm_4/TensorArrayV2/element_shape�
forward_lstm_4/TensorArrayV2TensorListReserve3forward_lstm_4/TensorArrayV2/element_shape:output:0'forward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_4/TensorArrayV2�
Dforward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2F
Dforward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape�
6forward_lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_4/transpose:y:0Mforward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6forward_lstm_4/TensorArrayUnstack/TensorListFromTensor�
$forward_lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_4/strided_slice_2/stack�
&forward_lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_2/stack_1�
&forward_lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_2/stack_2�
forward_lstm_4/strided_slice_2StridedSliceforward_lstm_4/transpose:y:0-forward_lstm_4/strided_slice_2/stack:output:0/forward_lstm_4/strided_slice_2/stack_1:output:0/forward_lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2 
forward_lstm_4/strided_slice_2�
1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp:forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype023
1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp�
"forward_lstm_4/lstm_cell_13/MatMulMatMul'forward_lstm_4/strided_slice_2:output:09forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"forward_lstm_4/lstm_cell_13/MatMul�
3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp<forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype025
3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp�
$forward_lstm_4/lstm_cell_13/MatMul_1MatMulforward_lstm_4/zeros:output:0;forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$forward_lstm_4/lstm_cell_13/MatMul_1�
forward_lstm_4/lstm_cell_13/addAddV2,forward_lstm_4/lstm_cell_13/MatMul:product:0.forward_lstm_4/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2!
forward_lstm_4/lstm_cell_13/add�
2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp;forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype024
2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp�
#forward_lstm_4/lstm_cell_13/BiasAddBiasAdd#forward_lstm_4/lstm_cell_13/add:z:0:forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#forward_lstm_4/lstm_cell_13/BiasAdd�
+forward_lstm_4/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+forward_lstm_4/lstm_cell_13/split/split_dim�
!forward_lstm_4/lstm_cell_13/splitSplit4forward_lstm_4/lstm_cell_13/split/split_dim:output:0,forward_lstm_4/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2#
!forward_lstm_4/lstm_cell_13/split�
#forward_lstm_4/lstm_cell_13/SigmoidSigmoid*forward_lstm_4/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22%
#forward_lstm_4/lstm_cell_13/Sigmoid�
%forward_lstm_4/lstm_cell_13/Sigmoid_1Sigmoid*forward_lstm_4/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22'
%forward_lstm_4/lstm_cell_13/Sigmoid_1�
forward_lstm_4/lstm_cell_13/mulMul)forward_lstm_4/lstm_cell_13/Sigmoid_1:y:0forward_lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:���������22!
forward_lstm_4/lstm_cell_13/mul�
 forward_lstm_4/lstm_cell_13/ReluRelu*forward_lstm_4/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22"
 forward_lstm_4/lstm_cell_13/Relu�
!forward_lstm_4/lstm_cell_13/mul_1Mul'forward_lstm_4/lstm_cell_13/Sigmoid:y:0.forward_lstm_4/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22#
!forward_lstm_4/lstm_cell_13/mul_1�
!forward_lstm_4/lstm_cell_13/add_1AddV2#forward_lstm_4/lstm_cell_13/mul:z:0%forward_lstm_4/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22#
!forward_lstm_4/lstm_cell_13/add_1�
%forward_lstm_4/lstm_cell_13/Sigmoid_2Sigmoid*forward_lstm_4/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22'
%forward_lstm_4/lstm_cell_13/Sigmoid_2�
"forward_lstm_4/lstm_cell_13/Relu_1Relu%forward_lstm_4/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_4/lstm_cell_13/Relu_1�
!forward_lstm_4/lstm_cell_13/mul_2Mul)forward_lstm_4/lstm_cell_13/Sigmoid_2:y:00forward_lstm_4/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22#
!forward_lstm_4/lstm_cell_13/mul_2�
,forward_lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2.
,forward_lstm_4/TensorArrayV2_1/element_shape�
forward_lstm_4/TensorArrayV2_1TensorListReserve5forward_lstm_4/TensorArrayV2_1/element_shape:output:0'forward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_4/TensorArrayV2_1l
forward_lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_4/time�
forward_lstm_4/zeros_like	ZerosLike%forward_lstm_4/lstm_cell_13/mul_2:z:0*
T0*'
_output_shapes
:���������22
forward_lstm_4/zeros_like�
'forward_lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2)
'forward_lstm_4/while/maximum_iterations�
!forward_lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!forward_lstm_4/while/loop_counter�
forward_lstm_4/whileWhile*forward_lstm_4/while/loop_counter:output:00forward_lstm_4/while/maximum_iterations:output:0forward_lstm_4/time:output:0'forward_lstm_4/TensorArrayV2_1:handle:0forward_lstm_4/zeros_like:y:0forward_lstm_4/zeros:output:0forward_lstm_4/zeros_1:output:0'forward_lstm_4/strided_slice_1:output:0Fforward_lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_4/Cast:y:0:forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource<forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource;forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *,
body$R"
 forward_lstm_4_while_body_694696*,
cond$R"
 forward_lstm_4_while_cond_694695*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
forward_lstm_4/while�
?forward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2A
?forward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape�
1forward_lstm_4/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_4/while:output:3Hforward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype023
1forward_lstm_4/TensorArrayV2Stack/TensorListStack�
$forward_lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2&
$forward_lstm_4/strided_slice_3/stack�
&forward_lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_4/strided_slice_3/stack_1�
&forward_lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_3/stack_2�
forward_lstm_4/strided_slice_3StridedSlice:forward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0-forward_lstm_4/strided_slice_3/stack:output:0/forward_lstm_4/strided_slice_3/stack_1:output:0/forward_lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2 
forward_lstm_4/strided_slice_3�
forward_lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_4/transpose_1/perm�
forward_lstm_4/transpose_1	Transpose:forward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0(forward_lstm_4/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
forward_lstm_4/transpose_1�
forward_lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_4/runtime�
$backward_lstm_4/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$backward_lstm_4/RaggedToTensor/zeros�
$backward_lstm_4/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2&
$backward_lstm_4/RaggedToTensor/Const�
3backward_lstm_4/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-backward_lstm_4/RaggedToTensor/Const:output:0inputs-backward_lstm_4/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3backward_lstm_4/RaggedToTensor/RaggedTensorToTensor�
:backward_lstm_4/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack�
<backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1�
<backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2�
4backward_lstm_4/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cbackward_lstm_4/RaggedNestedRowLengths/strided_slice/stack:output:0Ebackward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1:output:0Ebackward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask26
4backward_lstm_4/RaggedNestedRowLengths/strided_slice�
<backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack�
>backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2@
>backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1�
>backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2�
6backward_lstm_4/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Ebackward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gbackward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gbackward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask28
6backward_lstm_4/RaggedNestedRowLengths/strided_slice_1�
*backward_lstm_4/RaggedNestedRowLengths/subSub=backward_lstm_4/RaggedNestedRowLengths/strided_slice:output:0?backward_lstm_4/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2,
*backward_lstm_4/RaggedNestedRowLengths/sub�
backward_lstm_4/CastCast.backward_lstm_4/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
backward_lstm_4/Cast�
backward_lstm_4/ShapeShape<backward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_4/Shape�
#backward_lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#backward_lstm_4/strided_slice/stack�
%backward_lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_4/strided_slice/stack_1�
%backward_lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_4/strided_slice/stack_2�
backward_lstm_4/strided_sliceStridedSlicebackward_lstm_4/Shape:output:0,backward_lstm_4/strided_slice/stack:output:0.backward_lstm_4/strided_slice/stack_1:output:0.backward_lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
backward_lstm_4/strided_slice|
backward_lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_4/zeros/mul/y�
backward_lstm_4/zeros/mulMul&backward_lstm_4/strided_slice:output:0$backward_lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros/mul
backward_lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
backward_lstm_4/zeros/Less/y�
backward_lstm_4/zeros/LessLessbackward_lstm_4/zeros/mul:z:0%backward_lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros/Less�
backward_lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_4/zeros/packed/1�
backward_lstm_4/zeros/packedPack&backward_lstm_4/strided_slice:output:0'backward_lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_4/zeros/packed�
backward_lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_4/zeros/Const�
backward_lstm_4/zerosFill%backward_lstm_4/zeros/packed:output:0$backward_lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_4/zeros�
backward_lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_4/zeros_1/mul/y�
backward_lstm_4/zeros_1/mulMul&backward_lstm_4/strided_slice:output:0&backward_lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros_1/mul�
backward_lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
backward_lstm_4/zeros_1/Less/y�
backward_lstm_4/zeros_1/LessLessbackward_lstm_4/zeros_1/mul:z:0'backward_lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros_1/Less�
 backward_lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_4/zeros_1/packed/1�
backward_lstm_4/zeros_1/packedPack&backward_lstm_4/strided_slice:output:0)backward_lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_4/zeros_1/packed�
backward_lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_4/zeros_1/Const�
backward_lstm_4/zeros_1Fill'backward_lstm_4/zeros_1/packed:output:0&backward_lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_4/zeros_1�
backward_lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
backward_lstm_4/transpose/perm�
backward_lstm_4/transpose	Transpose<backward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0'backward_lstm_4/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_4/transpose
backward_lstm_4/Shape_1Shapebackward_lstm_4/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_4/Shape_1�
%backward_lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_4/strided_slice_1/stack�
'backward_lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_1/stack_1�
'backward_lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_1/stack_2�
backward_lstm_4/strided_slice_1StridedSlice backward_lstm_4/Shape_1:output:0.backward_lstm_4/strided_slice_1/stack:output:00backward_lstm_4/strided_slice_1/stack_1:output:00backward_lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_4/strided_slice_1�
+backward_lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+backward_lstm_4/TensorArrayV2/element_shape�
backward_lstm_4/TensorArrayV2TensorListReserve4backward_lstm_4/TensorArrayV2/element_shape:output:0(backward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
backward_lstm_4/TensorArrayV2�
backward_lstm_4/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2 
backward_lstm_4/ReverseV2/axis�
backward_lstm_4/ReverseV2	ReverseV2backward_lstm_4/transpose:y:0'backward_lstm_4/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_4/ReverseV2�
Ebackward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2G
Ebackward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape�
7backward_lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"backward_lstm_4/ReverseV2:output:0Nbackward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7backward_lstm_4/TensorArrayUnstack/TensorListFromTensor�
%backward_lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_4/strided_slice_2/stack�
'backward_lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_2/stack_1�
'backward_lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_2/stack_2�
backward_lstm_4/strided_slice_2StridedSlicebackward_lstm_4/transpose:y:0.backward_lstm_4/strided_slice_2/stack:output:00backward_lstm_4/strided_slice_2/stack_1:output:00backward_lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2!
backward_lstm_4/strided_slice_2�
2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp;backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp�
#backward_lstm_4/lstm_cell_14/MatMulMatMul(backward_lstm_4/strided_slice_2:output:0:backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#backward_lstm_4/lstm_cell_14/MatMul�
4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp=backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype026
4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp�
%backward_lstm_4/lstm_cell_14/MatMul_1MatMulbackward_lstm_4/zeros:output:0<backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%backward_lstm_4/lstm_cell_14/MatMul_1�
 backward_lstm_4/lstm_cell_14/addAddV2-backward_lstm_4/lstm_cell_14/MatMul:product:0/backward_lstm_4/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2"
 backward_lstm_4/lstm_cell_14/add�
3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp<backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp�
$backward_lstm_4/lstm_cell_14/BiasAddBiasAdd$backward_lstm_4/lstm_cell_14/add:z:0;backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$backward_lstm_4/lstm_cell_14/BiasAdd�
,backward_lstm_4/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,backward_lstm_4/lstm_cell_14/split/split_dim�
"backward_lstm_4/lstm_cell_14/splitSplit5backward_lstm_4/lstm_cell_14/split/split_dim:output:0-backward_lstm_4/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2$
"backward_lstm_4/lstm_cell_14/split�
$backward_lstm_4/lstm_cell_14/SigmoidSigmoid+backward_lstm_4/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22&
$backward_lstm_4/lstm_cell_14/Sigmoid�
&backward_lstm_4/lstm_cell_14/Sigmoid_1Sigmoid+backward_lstm_4/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22(
&backward_lstm_4/lstm_cell_14/Sigmoid_1�
 backward_lstm_4/lstm_cell_14/mulMul*backward_lstm_4/lstm_cell_14/Sigmoid_1:y:0 backward_lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/lstm_cell_14/mul�
!backward_lstm_4/lstm_cell_14/ReluRelu+backward_lstm_4/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22#
!backward_lstm_4/lstm_cell_14/Relu�
"backward_lstm_4/lstm_cell_14/mul_1Mul(backward_lstm_4/lstm_cell_14/Sigmoid:y:0/backward_lstm_4/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22$
"backward_lstm_4/lstm_cell_14/mul_1�
"backward_lstm_4/lstm_cell_14/add_1AddV2$backward_lstm_4/lstm_cell_14/mul:z:0&backward_lstm_4/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22$
"backward_lstm_4/lstm_cell_14/add_1�
&backward_lstm_4/lstm_cell_14/Sigmoid_2Sigmoid+backward_lstm_4/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22(
&backward_lstm_4/lstm_cell_14/Sigmoid_2�
#backward_lstm_4/lstm_cell_14/Relu_1Relu&backward_lstm_4/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_4/lstm_cell_14/Relu_1�
"backward_lstm_4/lstm_cell_14/mul_2Mul*backward_lstm_4/lstm_cell_14/Sigmoid_2:y:01backward_lstm_4/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22$
"backward_lstm_4/lstm_cell_14/mul_2�
-backward_lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2/
-backward_lstm_4/TensorArrayV2_1/element_shape�
backward_lstm_4/TensorArrayV2_1TensorListReserve6backward_lstm_4/TensorArrayV2_1/element_shape:output:0(backward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_4/TensorArrayV2_1n
backward_lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_4/time�
%backward_lstm_4/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2'
%backward_lstm_4/Max/reduction_indices�
backward_lstm_4/MaxMaxbackward_lstm_4/Cast:y:0.backward_lstm_4/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/Maxp
backward_lstm_4/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_4/sub/y�
backward_lstm_4/subSubbackward_lstm_4/Max:output:0backward_lstm_4/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/sub�
backward_lstm_4/Sub_1Subbackward_lstm_4/sub:z:0backward_lstm_4/Cast:y:0*
T0*#
_output_shapes
:���������2
backward_lstm_4/Sub_1�
backward_lstm_4/zeros_like	ZerosLike&backward_lstm_4/lstm_cell_14/mul_2:z:0*
T0*'
_output_shapes
:���������22
backward_lstm_4/zeros_like�
(backward_lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(backward_lstm_4/while/maximum_iterations�
"backward_lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"backward_lstm_4/while/loop_counter�
backward_lstm_4/whileWhile+backward_lstm_4/while/loop_counter:output:01backward_lstm_4/while/maximum_iterations:output:0backward_lstm_4/time:output:0(backward_lstm_4/TensorArrayV2_1:handle:0backward_lstm_4/zeros_like:y:0backward_lstm_4/zeros:output:0 backward_lstm_4/zeros_1:output:0(backward_lstm_4/strided_slice_1:output:0Gbackward_lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_4/Sub_1:z:0;backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource=backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource<backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *-
body%R#
!backward_lstm_4_while_body_694875*-
cond%R#
!backward_lstm_4_while_cond_694874*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
backward_lstm_4/while�
@backward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2B
@backward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape�
2backward_lstm_4/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_4/while:output:3Ibackward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype024
2backward_lstm_4/TensorArrayV2Stack/TensorListStack�
%backward_lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2'
%backward_lstm_4/strided_slice_3/stack�
'backward_lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_4/strided_slice_3/stack_1�
'backward_lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_3/stack_2�
backward_lstm_4/strided_slice_3StridedSlice;backward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0.backward_lstm_4/strided_slice_3/stack:output:00backward_lstm_4/strided_slice_3/stack_1:output:00backward_lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2!
backward_lstm_4/strided_slice_3�
 backward_lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_4/transpose_1/perm�
backward_lstm_4/transpose_1	Transpose;backward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0)backward_lstm_4/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
backward_lstm_4/transpose_1�
backward_lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_4/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2'forward_lstm_4/strided_slice_3:output:0(backward_lstm_4/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp4^backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp3^backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp5^backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp^backward_lstm_4/while3^forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp2^forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp4^forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp^forward_lstm_4/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������: : : : : : 2j
3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp2h
2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp2l
4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp2.
backward_lstm_4/whilebackward_lstm_4/while2h
2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp2f
1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp2j
3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp2,
forward_lstm_4/whileforward_lstm_4/while:O K
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
while_body_692027
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_14_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_14_matmul_readvariableop_resource:	�F
3while_lstm_cell_14_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_14_biasadd_readvariableop_resource:	���)while/lstm_cell_14/BiasAdd/ReadVariableOp�(while/lstm_cell_14/MatMul/ReadVariableOp�*while/lstm_cell_14/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp�
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/MatMul�
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp�
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/MatMul_1�
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/add�
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp�
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_14/BiasAdd�
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim�
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_14/split�
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Sigmoid�
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Sigmoid_1�
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/mul�
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Relu�
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/mul_1�
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/add_1�
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Sigmoid_2�
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/Relu_1�
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_14/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
�%
�
while_body_690615
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_13_690639_0:	�.
while_lstm_cell_13_690641_0:	2�*
while_lstm_cell_13_690643_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_13_690639:	�,
while_lstm_cell_13_690641:	2�(
while_lstm_cell_13_690643:	���*while/lstm_cell_13/StatefulPartitionedCall�
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
*while/lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_13_690639_0while_lstm_cell_13_690641_0while_lstm_cell_13_690643_0*
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
GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_6906012,
*while/lstm_cell_13/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_13/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_13/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_13/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_13_690639while_lstm_cell_13_690639_0"8
while_lstm_cell_13_690641while_lstm_cell_13_690641_0"8
while_lstm_cell_13_690643while_lstm_cell_13_690643_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2X
*while/lstm_cell_13/StatefulPartitionedCall*while/lstm_cell_13/StatefulPartitionedCall: 
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
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_696492

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
while_cond_696211
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_696211___redundant_placeholder04
0while_while_cond_696211___redundant_placeholder14
0while_while_cond_696211___redundant_placeholder24
0while_while_cond_696211___redundant_placeholder3
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
!backward_lstm_4_while_cond_694874<
8backward_lstm_4_while_backward_lstm_4_while_loop_counterB
>backward_lstm_4_while_backward_lstm_4_while_maximum_iterations%
!backward_lstm_4_while_placeholder'
#backward_lstm_4_while_placeholder_1'
#backward_lstm_4_while_placeholder_2'
#backward_lstm_4_while_placeholder_3'
#backward_lstm_4_while_placeholder_4>
:backward_lstm_4_while_less_backward_lstm_4_strided_slice_1T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_694874___redundant_placeholder0T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_694874___redundant_placeholder1T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_694874___redundant_placeholder2T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_694874___redundant_placeholder3T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_694874___redundant_placeholder4"
backward_lstm_4_while_identity
�
backward_lstm_4/while/LessLess!backward_lstm_4_while_placeholder:backward_lstm_4_while_less_backward_lstm_4_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_4/while/Less�
backward_lstm_4/while/IdentityIdentitybackward_lstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2 
backward_lstm_4/while/Identity"I
backward_lstm_4_while_identity'backward_lstm_4/while/Identity:output:0*(
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
�
�
!backward_lstm_4_while_cond_693304<
8backward_lstm_4_while_backward_lstm_4_while_loop_counterB
>backward_lstm_4_while_backward_lstm_4_while_maximum_iterations%
!backward_lstm_4_while_placeholder'
#backward_lstm_4_while_placeholder_1'
#backward_lstm_4_while_placeholder_2'
#backward_lstm_4_while_placeholder_3'
#backward_lstm_4_while_placeholder_4>
:backward_lstm_4_while_less_backward_lstm_4_strided_slice_1T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_693304___redundant_placeholder0T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_693304___redundant_placeholder1T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_693304___redundant_placeholder2T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_693304___redundant_placeholder3T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_693304___redundant_placeholder4"
backward_lstm_4_while_identity
�
backward_lstm_4/while/LessLess!backward_lstm_4_while_placeholder:backward_lstm_4_while_less_backward_lstm_4_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_4/while/Less�
backward_lstm_4/while/IdentityIdentitybackward_lstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2 
backward_lstm_4/while/Identity"I
backward_lstm_4_while_identity'backward_lstm_4/while/Identity:output:0*(
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
 forward_lstm_4_while_cond_694695:
6forward_lstm_4_while_forward_lstm_4_while_loop_counter@
<forward_lstm_4_while_forward_lstm_4_while_maximum_iterations$
 forward_lstm_4_while_placeholder&
"forward_lstm_4_while_placeholder_1&
"forward_lstm_4_while_placeholder_2&
"forward_lstm_4_while_placeholder_3&
"forward_lstm_4_while_placeholder_4<
8forward_lstm_4_while_less_forward_lstm_4_strided_slice_1R
Nforward_lstm_4_while_forward_lstm_4_while_cond_694695___redundant_placeholder0R
Nforward_lstm_4_while_forward_lstm_4_while_cond_694695___redundant_placeholder1R
Nforward_lstm_4_while_forward_lstm_4_while_cond_694695___redundant_placeholder2R
Nforward_lstm_4_while_forward_lstm_4_while_cond_694695___redundant_placeholder3R
Nforward_lstm_4_while_forward_lstm_4_while_cond_694695___redundant_placeholder4!
forward_lstm_4_while_identity
�
forward_lstm_4/while/LessLess forward_lstm_4_while_placeholder8forward_lstm_4_while_less_forward_lstm_4_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_4/while/Less�
forward_lstm_4/while/IdentityIdentityforward_lstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2
forward_lstm_4/while/Identity"G
forward_lstm_4_while_identity&forward_lstm_4/while/Identity:output:0*(
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
while_body_690825
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_13_690849_0:	�.
while_lstm_cell_13_690851_0:	2�*
while_lstm_cell_13_690853_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_13_690849:	�,
while_lstm_cell_13_690851:	2�(
while_lstm_cell_13_690853:	���*while/lstm_cell_13/StatefulPartitionedCall�
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
*while/lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_13_690849_0while_lstm_cell_13_690851_0while_lstm_cell_13_690853_0*
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
GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_6907472,
*while/lstm_cell_13/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_13/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_13/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_13/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_13_690849while_lstm_cell_13_690849_0"8
while_lstm_cell_13_690851while_lstm_cell_13_690851_0"8
while_lstm_cell_13_690853while_lstm_cell_13_690853_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2X
*while/lstm_cell_13/StatefulPartitionedCall*while/lstm_cell_13/StatefulPartitionedCall: 
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
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_691528

inputs&
lstm_cell_14_691446:	�&
lstm_cell_14_691448:	2�"
lstm_cell_14_691450:	�
identity��$lstm_cell_14/StatefulPartitionedCall�whileD
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
$lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_14_691446lstm_cell_14_691448lstm_cell_14_691450*
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
GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_6913792&
$lstm_cell_14/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_14_691446lstm_cell_14_691448lstm_cell_14_691450*
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
bodyR
while_body_691459*
condR
while_cond_691458*K
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
NoOpNoOp%^lstm_cell_14/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_14/StatefulPartitionedCall$lstm_cell_14/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�b
�
!backward_lstm_4_while_body_693305<
8backward_lstm_4_while_backward_lstm_4_while_loop_counterB
>backward_lstm_4_while_backward_lstm_4_while_maximum_iterations%
!backward_lstm_4_while_placeholder'
#backward_lstm_4_while_placeholder_1'
#backward_lstm_4_while_placeholder_2'
#backward_lstm_4_while_placeholder_3'
#backward_lstm_4_while_placeholder_4;
7backward_lstm_4_while_backward_lstm_4_strided_slice_1_0w
sbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_06
2backward_lstm_4_while_less_backward_lstm_4_sub_1_0V
Cbackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0:	�X
Ebackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0:	2�S
Dbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0:	�"
backward_lstm_4_while_identity$
 backward_lstm_4_while_identity_1$
 backward_lstm_4_while_identity_2$
 backward_lstm_4_while_identity_3$
 backward_lstm_4_while_identity_4$
 backward_lstm_4_while_identity_5$
 backward_lstm_4_while_identity_69
5backward_lstm_4_while_backward_lstm_4_strided_slice_1u
qbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor4
0backward_lstm_4_while_less_backward_lstm_4_sub_1T
Abackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource:	�V
Cbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource:	2�Q
Bbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource:	���9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp�8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp�:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp�
Gbackward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2I
Gbackward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape�
9backward_lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0!backward_lstm_4_while_placeholderPbackward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02;
9backward_lstm_4/while/TensorArrayV2Read/TensorListGetItem�
backward_lstm_4/while/LessLess2backward_lstm_4_while_less_backward_lstm_4_sub_1_0!backward_lstm_4_while_placeholder*
T0*#
_output_shapes
:���������2
backward_lstm_4/while/Less�
8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOpCbackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02:
8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp�
)backward_lstm_4/while/lstm_cell_14/MatMulMatMul@backward_lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0@backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)backward_lstm_4/while/lstm_cell_14/MatMul�
:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpEbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02<
:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp�
+backward_lstm_4/while/lstm_cell_14/MatMul_1MatMul#backward_lstm_4_while_placeholder_3Bbackward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+backward_lstm_4/while/lstm_cell_14/MatMul_1�
&backward_lstm_4/while/lstm_cell_14/addAddV23backward_lstm_4/while/lstm_cell_14/MatMul:product:05backward_lstm_4/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&backward_lstm_4/while/lstm_cell_14/add�
9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpDbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02;
9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp�
*backward_lstm_4/while/lstm_cell_14/BiasAddBiasAdd*backward_lstm_4/while/lstm_cell_14/add:z:0Abackward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*backward_lstm_4/while/lstm_cell_14/BiasAdd�
2backward_lstm_4/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2backward_lstm_4/while/lstm_cell_14/split/split_dim�
(backward_lstm_4/while/lstm_cell_14/splitSplit;backward_lstm_4/while/lstm_cell_14/split/split_dim:output:03backward_lstm_4/while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2*
(backward_lstm_4/while/lstm_cell_14/split�
*backward_lstm_4/while/lstm_cell_14/SigmoidSigmoid1backward_lstm_4/while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22,
*backward_lstm_4/while/lstm_cell_14/Sigmoid�
,backward_lstm_4/while/lstm_cell_14/Sigmoid_1Sigmoid1backward_lstm_4/while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22.
,backward_lstm_4/while/lstm_cell_14/Sigmoid_1�
&backward_lstm_4/while/lstm_cell_14/mulMul0backward_lstm_4/while/lstm_cell_14/Sigmoid_1:y:0#backward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22(
&backward_lstm_4/while/lstm_cell_14/mul�
'backward_lstm_4/while/lstm_cell_14/ReluRelu1backward_lstm_4/while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22)
'backward_lstm_4/while/lstm_cell_14/Relu�
(backward_lstm_4/while/lstm_cell_14/mul_1Mul.backward_lstm_4/while/lstm_cell_14/Sigmoid:y:05backward_lstm_4/while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22*
(backward_lstm_4/while/lstm_cell_14/mul_1�
(backward_lstm_4/while/lstm_cell_14/add_1AddV2*backward_lstm_4/while/lstm_cell_14/mul:z:0,backward_lstm_4/while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22*
(backward_lstm_4/while/lstm_cell_14/add_1�
,backward_lstm_4/while/lstm_cell_14/Sigmoid_2Sigmoid1backward_lstm_4/while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22.
,backward_lstm_4/while/lstm_cell_14/Sigmoid_2�
)backward_lstm_4/while/lstm_cell_14/Relu_1Relu,backward_lstm_4/while/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_4/while/lstm_cell_14/Relu_1�
(backward_lstm_4/while/lstm_cell_14/mul_2Mul0backward_lstm_4/while/lstm_cell_14/Sigmoid_2:y:07backward_lstm_4/while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22*
(backward_lstm_4/while/lstm_cell_14/mul_2�
backward_lstm_4/while/SelectSelectbackward_lstm_4/while/Less:z:0,backward_lstm_4/while/lstm_cell_14/mul_2:z:0#backward_lstm_4_while_placeholder_2*
T0*'
_output_shapes
:���������22
backward_lstm_4/while/Select�
backward_lstm_4/while/Select_1Selectbackward_lstm_4/while/Less:z:0,backward_lstm_4/while/lstm_cell_14/mul_2:z:0#backward_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:���������22 
backward_lstm_4/while/Select_1�
backward_lstm_4/while/Select_2Selectbackward_lstm_4/while/Less:z:0,backward_lstm_4/while/lstm_cell_14/add_1:z:0#backward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22 
backward_lstm_4/while/Select_2�
:backward_lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#backward_lstm_4_while_placeholder_1!backward_lstm_4_while_placeholder%backward_lstm_4/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:backward_lstm_4/while/TensorArrayV2Write/TensorListSetItem|
backward_lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_4/while/add/y�
backward_lstm_4/while/addAddV2!backward_lstm_4_while_placeholder$backward_lstm_4/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/while/add�
backward_lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_4/while/add_1/y�
backward_lstm_4/while/add_1AddV28backward_lstm_4_while_backward_lstm_4_while_loop_counter&backward_lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/while/add_1�
backward_lstm_4/while/IdentityIdentitybackward_lstm_4/while/add_1:z:0^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2 
backward_lstm_4/while/Identity�
 backward_lstm_4/while/Identity_1Identity>backward_lstm_4_while_backward_lstm_4_while_maximum_iterations^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_4/while/Identity_1�
 backward_lstm_4/while/Identity_2Identitybackward_lstm_4/while/add:z:0^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_4/while/Identity_2�
 backward_lstm_4/while/Identity_3IdentityJbackward_lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_4/while/Identity_3�
 backward_lstm_4/while/Identity_4Identity%backward_lstm_4/while/Select:output:0^backward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/while/Identity_4�
 backward_lstm_4/while/Identity_5Identity'backward_lstm_4/while/Select_1:output:0^backward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/while/Identity_5�
 backward_lstm_4/while/Identity_6Identity'backward_lstm_4/while/Select_2:output:0^backward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/while/Identity_6�
backward_lstm_4/while/NoOpNoOp:^backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp9^backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp;^backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_4/while/NoOp"p
5backward_lstm_4_while_backward_lstm_4_strided_slice_17backward_lstm_4_while_backward_lstm_4_strided_slice_1_0"I
backward_lstm_4_while_identity'backward_lstm_4/while/Identity:output:0"M
 backward_lstm_4_while_identity_1)backward_lstm_4/while/Identity_1:output:0"M
 backward_lstm_4_while_identity_2)backward_lstm_4/while/Identity_2:output:0"M
 backward_lstm_4_while_identity_3)backward_lstm_4/while/Identity_3:output:0"M
 backward_lstm_4_while_identity_4)backward_lstm_4/while/Identity_4:output:0"M
 backward_lstm_4_while_identity_5)backward_lstm_4/while/Identity_5:output:0"M
 backward_lstm_4_while_identity_6)backward_lstm_4/while/Identity_6:output:0"f
0backward_lstm_4_while_less_backward_lstm_4_sub_12backward_lstm_4_while_less_backward_lstm_4_sub_1_0"�
Bbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resourceDbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0"�
Cbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resourceEbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0"�
Abackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resourceCbackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0"�
qbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensorsbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2v
9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp2t
8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp2x
:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
while_body_695405
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_13_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_13_matmul_readvariableop_resource:	�F
3while_lstm_cell_13_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_13_biasadd_readvariableop_resource:	���)while/lstm_cell_13/BiasAdd/ReadVariableOp�(while/lstm_cell_13/MatMul/ReadVariableOp�*while/lstm_cell_13/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp�
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/MatMul�
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp�
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/MatMul_1�
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/add�
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp�
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/BiasAdd�
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim�
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_13/split�
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Sigmoid�
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Sigmoid_1�
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/mul�
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Relu�
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/mul_1�
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/add_1�
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Sigmoid_2�
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Relu_1�
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 
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
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_695338
inputs_0>
+lstm_cell_13_matmul_readvariableop_resource:	�@
-lstm_cell_13_matmul_1_readvariableop_resource:	2�;
,lstm_cell_13_biasadd_readvariableop_resource:	�
identity��#lstm_cell_13/BiasAdd/ReadVariableOp�"lstm_cell_13/MatMul/ReadVariableOp�$lstm_cell_13/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp�
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/MatMul�
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp�
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/MatMul_1�
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/add�
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp�
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim�
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_13/split�
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/Sigmoid�
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_13/Sigmoid_1�
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/mul}
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_13/Relu�
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/mul_1�
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/add_1�
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_13/Sigmoid_2|
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/Relu_1�
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
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
bodyR
while_body_695254*
condR
while_cond_695253*K
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
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�	
�
0__inference_bidirectional_4_layer_call_fn_693599
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
GPU 2J 8� *T
fORM
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_6921222
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
�%
�
while_body_691459
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_14_691483_0:	�.
while_lstm_cell_14_691485_0:	2�*
while_lstm_cell_14_691487_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_14_691483:	�,
while_lstm_cell_14_691485:	2�(
while_lstm_cell_14_691487:	���*while/lstm_cell_14/StatefulPartitionedCall�
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
*while/lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_14_691483_0while_lstm_cell_14_691485_0while_lstm_cell_14_691487_0*
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
GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_6913792,
*while/lstm_cell_14/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_14/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_14/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_14/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_14_691483while_lstm_cell_14_691483_0"8
while_lstm_cell_14_691485while_lstm_cell_14_691485_0"8
while_lstm_cell_14_691487while_lstm_cell_14_691487_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2X
*while/lstm_cell_14/StatefulPartitionedCall*while/lstm_cell_14/StatefulPartitionedCall: 
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
˒
�
=sequential_4_bidirectional_4_forward_lstm_4_while_body_690243t
psequential_4_bidirectional_4_forward_lstm_4_while_sequential_4_bidirectional_4_forward_lstm_4_while_loop_counterz
vsequential_4_bidirectional_4_forward_lstm_4_while_sequential_4_bidirectional_4_forward_lstm_4_while_maximum_iterationsA
=sequential_4_bidirectional_4_forward_lstm_4_while_placeholderC
?sequential_4_bidirectional_4_forward_lstm_4_while_placeholder_1C
?sequential_4_bidirectional_4_forward_lstm_4_while_placeholder_2C
?sequential_4_bidirectional_4_forward_lstm_4_while_placeholder_3C
?sequential_4_bidirectional_4_forward_lstm_4_while_placeholder_4s
osequential_4_bidirectional_4_forward_lstm_4_while_sequential_4_bidirectional_4_forward_lstm_4_strided_slice_1_0�
�sequential_4_bidirectional_4_forward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_bidirectional_4_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0p
lsequential_4_bidirectional_4_forward_lstm_4_while_greater_sequential_4_bidirectional_4_forward_lstm_4_cast_0r
_sequential_4_bidirectional_4_forward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0:	�t
asequential_4_bidirectional_4_forward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0:	2�o
`sequential_4_bidirectional_4_forward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0:	�>
:sequential_4_bidirectional_4_forward_lstm_4_while_identity@
<sequential_4_bidirectional_4_forward_lstm_4_while_identity_1@
<sequential_4_bidirectional_4_forward_lstm_4_while_identity_2@
<sequential_4_bidirectional_4_forward_lstm_4_while_identity_3@
<sequential_4_bidirectional_4_forward_lstm_4_while_identity_4@
<sequential_4_bidirectional_4_forward_lstm_4_while_identity_5@
<sequential_4_bidirectional_4_forward_lstm_4_while_identity_6q
msequential_4_bidirectional_4_forward_lstm_4_while_sequential_4_bidirectional_4_forward_lstm_4_strided_slice_1�
�sequential_4_bidirectional_4_forward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_bidirectional_4_forward_lstm_4_tensorarrayunstack_tensorlistfromtensorn
jsequential_4_bidirectional_4_forward_lstm_4_while_greater_sequential_4_bidirectional_4_forward_lstm_4_castp
]sequential_4_bidirectional_4_forward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource:	�r
_sequential_4_bidirectional_4_forward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource:	2�m
^sequential_4_bidirectional_4_forward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource:	���Usequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp�Tsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp�Vsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp�
csequential_4/bidirectional_4/forward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2e
csequential_4/bidirectional_4/forward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape�
Usequential_4/bidirectional_4/forward_lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_4_bidirectional_4_forward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_bidirectional_4_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0=sequential_4_bidirectional_4_forward_lstm_4_while_placeholderlsequential_4/bidirectional_4/forward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02W
Usequential_4/bidirectional_4/forward_lstm_4/while/TensorArrayV2Read/TensorListGetItem�
9sequential_4/bidirectional_4/forward_lstm_4/while/GreaterGreaterlsequential_4_bidirectional_4_forward_lstm_4_while_greater_sequential_4_bidirectional_4_forward_lstm_4_cast_0=sequential_4_bidirectional_4_forward_lstm_4_while_placeholder*
T0*#
_output_shapes
:���������2;
9sequential_4/bidirectional_4/forward_lstm_4/while/Greater�
Tsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp_sequential_4_bidirectional_4_forward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02V
Tsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp�
Esequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMulMatMul\sequential_4/bidirectional_4/forward_lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0\sequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2G
Esequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMul�
Vsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOpasequential_4_bidirectional_4_forward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02X
Vsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp�
Gsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMul_1MatMul?sequential_4_bidirectional_4_forward_lstm_4_while_placeholder_3^sequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2I
Gsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMul_1�
Bsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/addAddV2Osequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMul:product:0Qsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2D
Bsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/add�
Usequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp`sequential_4_bidirectional_4_forward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02W
Usequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp�
Fsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/BiasAddBiasAddFsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/add:z:0]sequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2H
Fsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/BiasAdd�
Nsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2P
Nsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/split/split_dim�
Dsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/splitSplitWsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/split/split_dim:output:0Osequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2F
Dsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/split�
Fsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/SigmoidSigmoidMsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22H
Fsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/Sigmoid�
Hsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/Sigmoid_1SigmoidMsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22J
Hsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/Sigmoid_1�
Bsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/mulMulLsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/Sigmoid_1:y:0?sequential_4_bidirectional_4_forward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22D
Bsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/mul�
Csequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/ReluReluMsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22E
Csequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/Relu�
Dsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/mul_1MulJsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/Sigmoid:y:0Qsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22F
Dsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/mul_1�
Dsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/add_1AddV2Fsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/mul:z:0Hsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22F
Dsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/add_1�
Hsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/Sigmoid_2SigmoidMsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22J
Hsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/Sigmoid_2�
Esequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/Relu_1ReluHsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22G
Esequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/Relu_1�
Dsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/mul_2MulLsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/Sigmoid_2:y:0Ssequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22F
Dsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/mul_2�
8sequential_4/bidirectional_4/forward_lstm_4/while/SelectSelect=sequential_4/bidirectional_4/forward_lstm_4/while/Greater:z:0Hsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/mul_2:z:0?sequential_4_bidirectional_4_forward_lstm_4_while_placeholder_2*
T0*'
_output_shapes
:���������22:
8sequential_4/bidirectional_4/forward_lstm_4/while/Select�
:sequential_4/bidirectional_4/forward_lstm_4/while/Select_1Select=sequential_4/bidirectional_4/forward_lstm_4/while/Greater:z:0Hsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/mul_2:z:0?sequential_4_bidirectional_4_forward_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:���������22<
:sequential_4/bidirectional_4/forward_lstm_4/while/Select_1�
:sequential_4/bidirectional_4/forward_lstm_4/while/Select_2Select=sequential_4/bidirectional_4/forward_lstm_4/while/Greater:z:0Hsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/add_1:z:0?sequential_4_bidirectional_4_forward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22<
:sequential_4/bidirectional_4/forward_lstm_4/while/Select_2�
Vsequential_4/bidirectional_4/forward_lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem?sequential_4_bidirectional_4_forward_lstm_4_while_placeholder_1=sequential_4_bidirectional_4_forward_lstm_4_while_placeholderAsequential_4/bidirectional_4/forward_lstm_4/while/Select:output:0*
_output_shapes
: *
element_dtype02X
Vsequential_4/bidirectional_4/forward_lstm_4/while/TensorArrayV2Write/TensorListSetItem�
7sequential_4/bidirectional_4/forward_lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_4/bidirectional_4/forward_lstm_4/while/add/y�
5sequential_4/bidirectional_4/forward_lstm_4/while/addAddV2=sequential_4_bidirectional_4_forward_lstm_4_while_placeholder@sequential_4/bidirectional_4/forward_lstm_4/while/add/y:output:0*
T0*
_output_shapes
: 27
5sequential_4/bidirectional_4/forward_lstm_4/while/add�
9sequential_4/bidirectional_4/forward_lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2;
9sequential_4/bidirectional_4/forward_lstm_4/while/add_1/y�
7sequential_4/bidirectional_4/forward_lstm_4/while/add_1AddV2psequential_4_bidirectional_4_forward_lstm_4_while_sequential_4_bidirectional_4_forward_lstm_4_while_loop_counterBsequential_4/bidirectional_4/forward_lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 29
7sequential_4/bidirectional_4/forward_lstm_4/while/add_1�
:sequential_4/bidirectional_4/forward_lstm_4/while/IdentityIdentity;sequential_4/bidirectional_4/forward_lstm_4/while/add_1:z:07^sequential_4/bidirectional_4/forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2<
:sequential_4/bidirectional_4/forward_lstm_4/while/Identity�
<sequential_4/bidirectional_4/forward_lstm_4/while/Identity_1Identityvsequential_4_bidirectional_4_forward_lstm_4_while_sequential_4_bidirectional_4_forward_lstm_4_while_maximum_iterations7^sequential_4/bidirectional_4/forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2>
<sequential_4/bidirectional_4/forward_lstm_4/while/Identity_1�
<sequential_4/bidirectional_4/forward_lstm_4/while/Identity_2Identity9sequential_4/bidirectional_4/forward_lstm_4/while/add:z:07^sequential_4/bidirectional_4/forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2>
<sequential_4/bidirectional_4/forward_lstm_4/while/Identity_2�
<sequential_4/bidirectional_4/forward_lstm_4/while/Identity_3Identityfsequential_4/bidirectional_4/forward_lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:07^sequential_4/bidirectional_4/forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2>
<sequential_4/bidirectional_4/forward_lstm_4/while/Identity_3�
<sequential_4/bidirectional_4/forward_lstm_4/while/Identity_4IdentityAsequential_4/bidirectional_4/forward_lstm_4/while/Select:output:07^sequential_4/bidirectional_4/forward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22>
<sequential_4/bidirectional_4/forward_lstm_4/while/Identity_4�
<sequential_4/bidirectional_4/forward_lstm_4/while/Identity_5IdentityCsequential_4/bidirectional_4/forward_lstm_4/while/Select_1:output:07^sequential_4/bidirectional_4/forward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22>
<sequential_4/bidirectional_4/forward_lstm_4/while/Identity_5�
<sequential_4/bidirectional_4/forward_lstm_4/while/Identity_6IdentityCsequential_4/bidirectional_4/forward_lstm_4/while/Select_2:output:07^sequential_4/bidirectional_4/forward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22>
<sequential_4/bidirectional_4/forward_lstm_4/while/Identity_6�
6sequential_4/bidirectional_4/forward_lstm_4/while/NoOpNoOpV^sequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOpU^sequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOpW^sequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 28
6sequential_4/bidirectional_4/forward_lstm_4/while/NoOp"�
jsequential_4_bidirectional_4_forward_lstm_4_while_greater_sequential_4_bidirectional_4_forward_lstm_4_castlsequential_4_bidirectional_4_forward_lstm_4_while_greater_sequential_4_bidirectional_4_forward_lstm_4_cast_0"�
:sequential_4_bidirectional_4_forward_lstm_4_while_identityCsequential_4/bidirectional_4/forward_lstm_4/while/Identity:output:0"�
<sequential_4_bidirectional_4_forward_lstm_4_while_identity_1Esequential_4/bidirectional_4/forward_lstm_4/while/Identity_1:output:0"�
<sequential_4_bidirectional_4_forward_lstm_4_while_identity_2Esequential_4/bidirectional_4/forward_lstm_4/while/Identity_2:output:0"�
<sequential_4_bidirectional_4_forward_lstm_4_while_identity_3Esequential_4/bidirectional_4/forward_lstm_4/while/Identity_3:output:0"�
<sequential_4_bidirectional_4_forward_lstm_4_while_identity_4Esequential_4/bidirectional_4/forward_lstm_4/while/Identity_4:output:0"�
<sequential_4_bidirectional_4_forward_lstm_4_while_identity_5Esequential_4/bidirectional_4/forward_lstm_4/while/Identity_5:output:0"�
<sequential_4_bidirectional_4_forward_lstm_4_while_identity_6Esequential_4/bidirectional_4/forward_lstm_4/while/Identity_6:output:0"�
^sequential_4_bidirectional_4_forward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource`sequential_4_bidirectional_4_forward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0"�
_sequential_4_bidirectional_4_forward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resourceasequential_4_bidirectional_4_forward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0"�
]sequential_4_bidirectional_4_forward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_sequential_4_bidirectional_4_forward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0"�
msequential_4_bidirectional_4_forward_lstm_4_while_sequential_4_bidirectional_4_forward_lstm_4_strided_slice_1osequential_4_bidirectional_4_forward_lstm_4_while_sequential_4_bidirectional_4_forward_lstm_4_strided_slice_1_0"�
�sequential_4_bidirectional_4_forward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_bidirectional_4_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor�sequential_4_bidirectional_4_forward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_bidirectional_4_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2�
Usequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOpUsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp2�
Tsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOpTsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp2�
Vsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOpVsequential_4/bidirectional_4/forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp: 
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
/__inference_forward_lstm_4_layer_call_fn_695036

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
GPU 2J 8� *S
fNRL
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_6924762
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
�?
�
while_body_695254
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_13_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_13_matmul_readvariableop_resource:	�F
3while_lstm_cell_13_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_13_biasadd_readvariableop_resource:	���)while/lstm_cell_13/BiasAdd/ReadVariableOp�(while/lstm_cell_13/MatMul/ReadVariableOp�*while/lstm_cell_13/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp�
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/MatMul�
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp�
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/MatMul_1�
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/add�
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp�
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/BiasAdd�
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim�
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_13/split�
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Sigmoid�
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Sigmoid_1�
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/mul�
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Relu�
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/mul_1�
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/add_1�
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Sigmoid_2�
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Relu_1�
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 
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
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_695990
inputs_0>
+lstm_cell_14_matmul_readvariableop_resource:	�@
-lstm_cell_14_matmul_1_readvariableop_resource:	2�;
,lstm_cell_14_biasadd_readvariableop_resource:	�
identity��#lstm_cell_14/BiasAdd/ReadVariableOp�"lstm_cell_14/MatMul/ReadVariableOp�$lstm_cell_14/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp�
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/MatMul�
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp�
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/MatMul_1�
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/add�
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp�
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/BiasAdd~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim�
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_14/split�
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/Sigmoid�
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_14/Sigmoid_1�
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_14/Relu�
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/mul_1�
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/add_1�
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/Relu_1�
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
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
bodyR
while_body_695906*
condR
while_cond_695905*K
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
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�%
�
while_body_691247
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_14_691271_0:	�.
while_lstm_cell_14_691273_0:	2�*
while_lstm_cell_14_691275_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_14_691271:	�,
while_lstm_cell_14_691273:	2�(
while_lstm_cell_14_691275:	���*while/lstm_cell_14/StatefulPartitionedCall�
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
*while/lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_14_691271_0while_lstm_cell_14_691273_0while_lstm_cell_14_691275_0*
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
GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_6912332,
*while/lstm_cell_14/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_14/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_14/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_14/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_14_691271while_lstm_cell_14_691271_0"8
while_lstm_cell_14_691273while_lstm_cell_14_691273_0"8
while_lstm_cell_14_691275while_lstm_cell_14_691275_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2X
*while/lstm_cell_14/StatefulPartitionedCall*while/lstm_cell_14/StatefulPartitionedCall: 
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
��
�
"__inference__traced_restore_696760
file_prefix1
assignvariableop_dense_4_kernel:d-
assignvariableop_1_dense_4_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: X
Eassignvariableop_7_bidirectional_4_forward_lstm_4_lstm_cell_13_kernel:	�b
Oassignvariableop_8_bidirectional_4_forward_lstm_4_lstm_cell_13_recurrent_kernel:	2�R
Cassignvariableop_9_bidirectional_4_forward_lstm_4_lstm_cell_13_bias:	�Z
Gassignvariableop_10_bidirectional_4_backward_lstm_4_lstm_cell_14_kernel:	�d
Qassignvariableop_11_bidirectional_4_backward_lstm_4_lstm_cell_14_recurrent_kernel:	2�T
Eassignvariableop_12_bidirectional_4_backward_lstm_4_lstm_cell_14_bias:	�#
assignvariableop_13_total: #
assignvariableop_14_count: ;
)assignvariableop_15_adam_dense_4_kernel_m:d5
'assignvariableop_16_adam_dense_4_bias_m:`
Massignvariableop_17_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_kernel_m:	�j
Wassignvariableop_18_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_recurrent_kernel_m:	2�Z
Kassignvariableop_19_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_bias_m:	�a
Nassignvariableop_20_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_kernel_m:	�k
Xassignvariableop_21_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_recurrent_kernel_m:	2�[
Lassignvariableop_22_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_bias_m:	�;
)assignvariableop_23_adam_dense_4_kernel_v:d5
'assignvariableop_24_adam_dense_4_bias_v:`
Massignvariableop_25_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_kernel_v:	�j
Wassignvariableop_26_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_recurrent_kernel_v:	2�Z
Kassignvariableop_27_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_bias_v:	�a
Nassignvariableop_28_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_kernel_v:	�k
Xassignvariableop_29_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_recurrent_kernel_v:	2�[
Lassignvariableop_30_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_bias_v:	�>
,assignvariableop_31_adam_dense_4_kernel_vhat:d8
*assignvariableop_32_adam_dense_4_bias_vhat:c
Passignvariableop_33_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_kernel_vhat:	�m
Zassignvariableop_34_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_recurrent_kernel_vhat:	2�]
Nassignvariableop_35_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_bias_vhat:	�d
Qassignvariableop_36_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_kernel_vhat:	�n
[assignvariableop_37_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_recurrent_kernel_vhat:	2�^
Oassignvariableop_38_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_bias_vhat:	�
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
AssignVariableOpAssignVariableOpassignvariableop_dense_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_4_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOpEassignvariableop_7_bidirectional_4_forward_lstm_4_lstm_cell_13_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpOassignvariableop_8_bidirectional_4_forward_lstm_4_lstm_cell_13_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpCassignvariableop_9_bidirectional_4_forward_lstm_4_lstm_cell_13_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpGassignvariableop_10_bidirectional_4_backward_lstm_4_lstm_cell_14_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpQassignvariableop_11_bidirectional_4_backward_lstm_4_lstm_cell_14_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpEassignvariableop_12_bidirectional_4_backward_lstm_4_lstm_cell_14_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_4_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_4_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpMassignvariableop_17_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpWassignvariableop_18_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpKassignvariableop_19_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpNassignvariableop_20_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpXassignvariableop_21_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpLassignvariableop_22_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_4_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_4_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOpMassignvariableop_25_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOpWassignvariableop_26_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOpKassignvariableop_27_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOpNassignvariableop_28_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOpXassignvariableop_29_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOpLassignvariableop_30_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp,assignvariableop_31_adam_dense_4_kernel_vhatIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_dense_4_bias_vhatIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOpPassignvariableop_33_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_kernel_vhatIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOpZassignvariableop_34_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_recurrent_kernel_vhatIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOpNassignvariableop_35_adam_bidirectional_4_forward_lstm_4_lstm_cell_13_bias_vhatIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOpQassignvariableop_36_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_kernel_vhatIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp[assignvariableop_37_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_recurrent_kernel_vhatIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOpOassignvariableop_38_adam_bidirectional_4_backward_lstm_4_lstm_cell_14_bias_vhatIdentity_38:output:0"/device:CPU:0*
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
�
�
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_690601

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
while_body_695556
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_13_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:	2�C
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_13_matmul_readvariableop_resource:	�F
3while_lstm_cell_13_matmul_1_readvariableop_resource:	2�A
2while_lstm_cell_13_biasadd_readvariableop_resource:	���)while/lstm_cell_13/BiasAdd/ReadVariableOp�(while/lstm_cell_13/MatMul/ReadVariableOp�*while/lstm_cell_13/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp�
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/MatMul�
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp�
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/MatMul_1�
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/add�
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp�
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_13/BiasAdd�
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim�
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
while/lstm_cell_13/split�
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Sigmoid�
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Sigmoid_1�
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/mul�
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Relu�
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/mul_1�
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/add_1�
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Sigmoid_2�
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/Relu_1�
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
while/lstm_cell_13/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������22
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 
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
/__inference_forward_lstm_4_layer_call_fn_695014
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
GPU 2J 8� *S
fNRL
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_6908942
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
C__inference_dense_4_layer_call_and_return_conditional_losses_694992

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
�T
�
 forward_lstm_4_while_body_694021:
6forward_lstm_4_while_forward_lstm_4_while_loop_counter@
<forward_lstm_4_while_forward_lstm_4_while_maximum_iterations$
 forward_lstm_4_while_placeholder&
"forward_lstm_4_while_placeholder_1&
"forward_lstm_4_while_placeholder_2&
"forward_lstm_4_while_placeholder_39
5forward_lstm_4_while_forward_lstm_4_strided_slice_1_0u
qforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0U
Bforward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0:	�W
Dforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0:	2�R
Cforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0:	�!
forward_lstm_4_while_identity#
forward_lstm_4_while_identity_1#
forward_lstm_4_while_identity_2#
forward_lstm_4_while_identity_3#
forward_lstm_4_while_identity_4#
forward_lstm_4_while_identity_57
3forward_lstm_4_while_forward_lstm_4_strided_slice_1s
oforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensorS
@forward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource:	�U
Bforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource:	2�P
Aforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource:	���8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp�7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp�9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp�
Fforward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2H
Fforward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape�
8forward_lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0 forward_lstm_4_while_placeholderOforward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02:
8forward_lstm_4/while/TensorArrayV2Read/TensorListGetItem�
7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOpBforward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype029
7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp�
(forward_lstm_4/while/lstm_cell_13/MatMulMatMul?forward_lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0?forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(forward_lstm_4/while/lstm_cell_13/MatMul�
9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOpDforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02;
9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp�
*forward_lstm_4/while/lstm_cell_13/MatMul_1MatMul"forward_lstm_4_while_placeholder_2Aforward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*forward_lstm_4/while/lstm_cell_13/MatMul_1�
%forward_lstm_4/while/lstm_cell_13/addAddV22forward_lstm_4/while/lstm_cell_13/MatMul:product:04forward_lstm_4/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2'
%forward_lstm_4/while/lstm_cell_13/add�
8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOpCforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02:
8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp�
)forward_lstm_4/while/lstm_cell_13/BiasAddBiasAdd)forward_lstm_4/while/lstm_cell_13/add:z:0@forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)forward_lstm_4/while/lstm_cell_13/BiasAdd�
1forward_lstm_4/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1forward_lstm_4/while/lstm_cell_13/split/split_dim�
'forward_lstm_4/while/lstm_cell_13/splitSplit:forward_lstm_4/while/lstm_cell_13/split/split_dim:output:02forward_lstm_4/while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2)
'forward_lstm_4/while/lstm_cell_13/split�
)forward_lstm_4/while/lstm_cell_13/SigmoidSigmoid0forward_lstm_4/while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22+
)forward_lstm_4/while/lstm_cell_13/Sigmoid�
+forward_lstm_4/while/lstm_cell_13/Sigmoid_1Sigmoid0forward_lstm_4/while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22-
+forward_lstm_4/while/lstm_cell_13/Sigmoid_1�
%forward_lstm_4/while/lstm_cell_13/mulMul/forward_lstm_4/while/lstm_cell_13/Sigmoid_1:y:0"forward_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:���������22'
%forward_lstm_4/while/lstm_cell_13/mul�
&forward_lstm_4/while/lstm_cell_13/ReluRelu0forward_lstm_4/while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22(
&forward_lstm_4/while/lstm_cell_13/Relu�
'forward_lstm_4/while/lstm_cell_13/mul_1Mul-forward_lstm_4/while/lstm_cell_13/Sigmoid:y:04forward_lstm_4/while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22)
'forward_lstm_4/while/lstm_cell_13/mul_1�
'forward_lstm_4/while/lstm_cell_13/add_1AddV2)forward_lstm_4/while/lstm_cell_13/mul:z:0+forward_lstm_4/while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22)
'forward_lstm_4/while/lstm_cell_13/add_1�
+forward_lstm_4/while/lstm_cell_13/Sigmoid_2Sigmoid0forward_lstm_4/while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22-
+forward_lstm_4/while/lstm_cell_13/Sigmoid_2�
(forward_lstm_4/while/lstm_cell_13/Relu_1Relu+forward_lstm_4/while/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22*
(forward_lstm_4/while/lstm_cell_13/Relu_1�
'forward_lstm_4/while/lstm_cell_13/mul_2Mul/forward_lstm_4/while/lstm_cell_13/Sigmoid_2:y:06forward_lstm_4/while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22)
'forward_lstm_4/while/lstm_cell_13/mul_2�
9forward_lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"forward_lstm_4_while_placeholder_1 forward_lstm_4_while_placeholder+forward_lstm_4/while/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype02;
9forward_lstm_4/while/TensorArrayV2Write/TensorListSetItemz
forward_lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_4/while/add/y�
forward_lstm_4/while/addAddV2 forward_lstm_4_while_placeholder#forward_lstm_4/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/while/add~
forward_lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_4/while/add_1/y�
forward_lstm_4/while/add_1AddV26forward_lstm_4_while_forward_lstm_4_while_loop_counter%forward_lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/while/add_1�
forward_lstm_4/while/IdentityIdentityforward_lstm_4/while/add_1:z:0^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2
forward_lstm_4/while/Identity�
forward_lstm_4/while/Identity_1Identity<forward_lstm_4_while_forward_lstm_4_while_maximum_iterations^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_4/while/Identity_1�
forward_lstm_4/while/Identity_2Identityforward_lstm_4/while/add:z:0^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_4/while/Identity_2�
forward_lstm_4/while/Identity_3IdentityIforward_lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_4/while/Identity_3�
forward_lstm_4/while/Identity_4Identity+forward_lstm_4/while/lstm_cell_13/mul_2:z:0^forward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22!
forward_lstm_4/while/Identity_4�
forward_lstm_4/while/Identity_5Identity+forward_lstm_4/while/lstm_cell_13/add_1:z:0^forward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22!
forward_lstm_4/while/Identity_5�
forward_lstm_4/while/NoOpNoOp9^forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp8^forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp:^forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_4/while/NoOp"l
3forward_lstm_4_while_forward_lstm_4_strided_slice_15forward_lstm_4_while_forward_lstm_4_strided_slice_1_0"G
forward_lstm_4_while_identity&forward_lstm_4/while/Identity:output:0"K
forward_lstm_4_while_identity_1(forward_lstm_4/while/Identity_1:output:0"K
forward_lstm_4_while_identity_2(forward_lstm_4/while/Identity_2:output:0"K
forward_lstm_4_while_identity_3(forward_lstm_4/while/Identity_3:output:0"K
forward_lstm_4_while_identity_4(forward_lstm_4/while/Identity_4:output:0"K
forward_lstm_4_while_identity_5(forward_lstm_4/while/Identity_5:output:0"�
Aforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resourceCforward_lstm_4_while_lstm_cell_13_biasadd_readvariableop_resource_0"�
Bforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resourceDforward_lstm_4_while_lstm_cell_13_matmul_1_readvariableop_resource_0"�
@forward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resourceBforward_lstm_4_while_lstm_cell_13_matmul_readvariableop_resource_0"�
oforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensorqforward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2t
8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp8forward_lstm_4/while/lstm_cell_13/BiasAdd/ReadVariableOp2r
7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp7forward_lstm_4/while/lstm_cell_13/MatMul/ReadVariableOp2v
9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp9forward_lstm_4/while/lstm_cell_13/MatMul_1/ReadVariableOp: 
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
0__inference_bidirectional_4_layer_call_fn_693616
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
GPU 2J 8� *T
fORM
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_6925242
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
�V
�
!backward_lstm_4_while_body_694170<
8backward_lstm_4_while_backward_lstm_4_while_loop_counterB
>backward_lstm_4_while_backward_lstm_4_while_maximum_iterations%
!backward_lstm_4_while_placeholder'
#backward_lstm_4_while_placeholder_1'
#backward_lstm_4_while_placeholder_2'
#backward_lstm_4_while_placeholder_3;
7backward_lstm_4_while_backward_lstm_4_strided_slice_1_0w
sbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0V
Cbackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0:	�X
Ebackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0:	2�S
Dbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0:	�"
backward_lstm_4_while_identity$
 backward_lstm_4_while_identity_1$
 backward_lstm_4_while_identity_2$
 backward_lstm_4_while_identity_3$
 backward_lstm_4_while_identity_4$
 backward_lstm_4_while_identity_59
5backward_lstm_4_while_backward_lstm_4_strided_slice_1u
qbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensorT
Abackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource:	�V
Cbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource:	2�Q
Bbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource:	���9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp�8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp�:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp�
Gbackward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2I
Gbackward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape�
9backward_lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0!backward_lstm_4_while_placeholderPbackward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype02;
9backward_lstm_4/while/TensorArrayV2Read/TensorListGetItem�
8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOpCbackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02:
8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp�
)backward_lstm_4/while/lstm_cell_14/MatMulMatMul@backward_lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0@backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)backward_lstm_4/while/lstm_cell_14/MatMul�
:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpEbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02<
:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp�
+backward_lstm_4/while/lstm_cell_14/MatMul_1MatMul#backward_lstm_4_while_placeholder_2Bbackward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+backward_lstm_4/while/lstm_cell_14/MatMul_1�
&backward_lstm_4/while/lstm_cell_14/addAddV23backward_lstm_4/while/lstm_cell_14/MatMul:product:05backward_lstm_4/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&backward_lstm_4/while/lstm_cell_14/add�
9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpDbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02;
9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp�
*backward_lstm_4/while/lstm_cell_14/BiasAddBiasAdd*backward_lstm_4/while/lstm_cell_14/add:z:0Abackward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*backward_lstm_4/while/lstm_cell_14/BiasAdd�
2backward_lstm_4/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2backward_lstm_4/while/lstm_cell_14/split/split_dim�
(backward_lstm_4/while/lstm_cell_14/splitSplit;backward_lstm_4/while/lstm_cell_14/split/split_dim:output:03backward_lstm_4/while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2*
(backward_lstm_4/while/lstm_cell_14/split�
*backward_lstm_4/while/lstm_cell_14/SigmoidSigmoid1backward_lstm_4/while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22,
*backward_lstm_4/while/lstm_cell_14/Sigmoid�
,backward_lstm_4/while/lstm_cell_14/Sigmoid_1Sigmoid1backward_lstm_4/while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22.
,backward_lstm_4/while/lstm_cell_14/Sigmoid_1�
&backward_lstm_4/while/lstm_cell_14/mulMul0backward_lstm_4/while/lstm_cell_14/Sigmoid_1:y:0#backward_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:���������22(
&backward_lstm_4/while/lstm_cell_14/mul�
'backward_lstm_4/while/lstm_cell_14/ReluRelu1backward_lstm_4/while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22)
'backward_lstm_4/while/lstm_cell_14/Relu�
(backward_lstm_4/while/lstm_cell_14/mul_1Mul.backward_lstm_4/while/lstm_cell_14/Sigmoid:y:05backward_lstm_4/while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22*
(backward_lstm_4/while/lstm_cell_14/mul_1�
(backward_lstm_4/while/lstm_cell_14/add_1AddV2*backward_lstm_4/while/lstm_cell_14/mul:z:0,backward_lstm_4/while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22*
(backward_lstm_4/while/lstm_cell_14/add_1�
,backward_lstm_4/while/lstm_cell_14/Sigmoid_2Sigmoid1backward_lstm_4/while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22.
,backward_lstm_4/while/lstm_cell_14/Sigmoid_2�
)backward_lstm_4/while/lstm_cell_14/Relu_1Relu,backward_lstm_4/while/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_4/while/lstm_cell_14/Relu_1�
(backward_lstm_4/while/lstm_cell_14/mul_2Mul0backward_lstm_4/while/lstm_cell_14/Sigmoid_2:y:07backward_lstm_4/while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22*
(backward_lstm_4/while/lstm_cell_14/mul_2�
:backward_lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#backward_lstm_4_while_placeholder_1!backward_lstm_4_while_placeholder,backward_lstm_4/while/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:backward_lstm_4/while/TensorArrayV2Write/TensorListSetItem|
backward_lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_4/while/add/y�
backward_lstm_4/while/addAddV2!backward_lstm_4_while_placeholder$backward_lstm_4/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/while/add�
backward_lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_4/while/add_1/y�
backward_lstm_4/while/add_1AddV28backward_lstm_4_while_backward_lstm_4_while_loop_counter&backward_lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/while/add_1�
backward_lstm_4/while/IdentityIdentitybackward_lstm_4/while/add_1:z:0^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2 
backward_lstm_4/while/Identity�
 backward_lstm_4/while/Identity_1Identity>backward_lstm_4_while_backward_lstm_4_while_maximum_iterations^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_4/while/Identity_1�
 backward_lstm_4/while/Identity_2Identitybackward_lstm_4/while/add:z:0^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_4/while/Identity_2�
 backward_lstm_4/while/Identity_3IdentityJbackward_lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_4/while/Identity_3�
 backward_lstm_4/while/Identity_4Identity,backward_lstm_4/while/lstm_cell_14/mul_2:z:0^backward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/while/Identity_4�
 backward_lstm_4/while/Identity_5Identity,backward_lstm_4/while/lstm_cell_14/add_1:z:0^backward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/while/Identity_5�
backward_lstm_4/while/NoOpNoOp:^backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp9^backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp;^backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_4/while/NoOp"p
5backward_lstm_4_while_backward_lstm_4_strided_slice_17backward_lstm_4_while_backward_lstm_4_strided_slice_1_0"I
backward_lstm_4_while_identity'backward_lstm_4/while/Identity:output:0"M
 backward_lstm_4_while_identity_1)backward_lstm_4/while/Identity_1:output:0"M
 backward_lstm_4_while_identity_2)backward_lstm_4/while/Identity_2:output:0"M
 backward_lstm_4_while_identity_3)backward_lstm_4/while/Identity_3:output:0"M
 backward_lstm_4_while_identity_4)backward_lstm_4/while/Identity_4:output:0"M
 backward_lstm_4_while_identity_5)backward_lstm_4/while/Identity_5:output:0"�
Bbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resourceDbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0"�
Cbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resourceEbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0"�
Abackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resourceCbackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0"�
qbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensorsbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������2:���������2: : : : : 2v
9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp2t
8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp2x
:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
!backward_lstm_4_while_cond_692864<
8backward_lstm_4_while_backward_lstm_4_while_loop_counterB
>backward_lstm_4_while_backward_lstm_4_while_maximum_iterations%
!backward_lstm_4_while_placeholder'
#backward_lstm_4_while_placeholder_1'
#backward_lstm_4_while_placeholder_2'
#backward_lstm_4_while_placeholder_3'
#backward_lstm_4_while_placeholder_4>
:backward_lstm_4_while_less_backward_lstm_4_strided_slice_1T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_692864___redundant_placeholder0T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_692864___redundant_placeholder1T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_692864___redundant_placeholder2T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_692864___redundant_placeholder3T
Pbackward_lstm_4_while_backward_lstm_4_while_cond_692864___redundant_placeholder4"
backward_lstm_4_while_identity
�
backward_lstm_4/while/LessLess!backward_lstm_4_while_placeholder:backward_lstm_4_while_less_backward_lstm_4_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_4/while/Less�
backward_lstm_4/while/IdentityIdentitybackward_lstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2 
backward_lstm_4/while/Identity"I
backward_lstm_4_while_identity'backward_lstm_4/while/Identity:output:0*(
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
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_692111

inputs>
+lstm_cell_14_matmul_readvariableop_resource:	�@
-lstm_cell_14_matmul_1_readvariableop_resource:	2�;
,lstm_cell_14_biasadd_readvariableop_resource:	�
identity��#lstm_cell_14/BiasAdd/ReadVariableOp�"lstm_cell_14/MatMul/ReadVariableOp�$lstm_cell_14/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp�
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/MatMul�
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp�
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/MatMul_1�
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/add�
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp�
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_14/BiasAdd~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim�
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_14/split�
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/Sigmoid�
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_14/Sigmoid_1�
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_14/Relu�
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/mul_1�
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/add_1�
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/Relu_1�
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_14/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
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
bodyR
while_body_692027*
condR
while_cond_692026*K
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
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�b
�
!backward_lstm_4_while_body_694517<
8backward_lstm_4_while_backward_lstm_4_while_loop_counterB
>backward_lstm_4_while_backward_lstm_4_while_maximum_iterations%
!backward_lstm_4_while_placeholder'
#backward_lstm_4_while_placeholder_1'
#backward_lstm_4_while_placeholder_2'
#backward_lstm_4_while_placeholder_3'
#backward_lstm_4_while_placeholder_4;
7backward_lstm_4_while_backward_lstm_4_strided_slice_1_0w
sbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_06
2backward_lstm_4_while_less_backward_lstm_4_sub_1_0V
Cbackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0:	�X
Ebackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0:	2�S
Dbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0:	�"
backward_lstm_4_while_identity$
 backward_lstm_4_while_identity_1$
 backward_lstm_4_while_identity_2$
 backward_lstm_4_while_identity_3$
 backward_lstm_4_while_identity_4$
 backward_lstm_4_while_identity_5$
 backward_lstm_4_while_identity_69
5backward_lstm_4_while_backward_lstm_4_strided_slice_1u
qbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor4
0backward_lstm_4_while_less_backward_lstm_4_sub_1T
Abackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource:	�V
Cbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource:	2�Q
Bbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource:	���9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp�8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp�:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp�
Gbackward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2I
Gbackward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape�
9backward_lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0!backward_lstm_4_while_placeholderPbackward_lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02;
9backward_lstm_4/while/TensorArrayV2Read/TensorListGetItem�
backward_lstm_4/while/LessLess2backward_lstm_4_while_less_backward_lstm_4_sub_1_0!backward_lstm_4_while_placeholder*
T0*#
_output_shapes
:���������2
backward_lstm_4/while/Less�
8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOpCbackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02:
8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp�
)backward_lstm_4/while/lstm_cell_14/MatMulMatMul@backward_lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0@backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)backward_lstm_4/while/lstm_cell_14/MatMul�
:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpEbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	2�*
dtype02<
:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp�
+backward_lstm_4/while/lstm_cell_14/MatMul_1MatMul#backward_lstm_4_while_placeholder_3Bbackward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+backward_lstm_4/while/lstm_cell_14/MatMul_1�
&backward_lstm_4/while/lstm_cell_14/addAddV23backward_lstm_4/while/lstm_cell_14/MatMul:product:05backward_lstm_4/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&backward_lstm_4/while/lstm_cell_14/add�
9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpDbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02;
9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp�
*backward_lstm_4/while/lstm_cell_14/BiasAddBiasAdd*backward_lstm_4/while/lstm_cell_14/add:z:0Abackward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*backward_lstm_4/while/lstm_cell_14/BiasAdd�
2backward_lstm_4/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2backward_lstm_4/while/lstm_cell_14/split/split_dim�
(backward_lstm_4/while/lstm_cell_14/splitSplit;backward_lstm_4/while/lstm_cell_14/split/split_dim:output:03backward_lstm_4/while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2*
(backward_lstm_4/while/lstm_cell_14/split�
*backward_lstm_4/while/lstm_cell_14/SigmoidSigmoid1backward_lstm_4/while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22,
*backward_lstm_4/while/lstm_cell_14/Sigmoid�
,backward_lstm_4/while/lstm_cell_14/Sigmoid_1Sigmoid1backward_lstm_4/while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22.
,backward_lstm_4/while/lstm_cell_14/Sigmoid_1�
&backward_lstm_4/while/lstm_cell_14/mulMul0backward_lstm_4/while/lstm_cell_14/Sigmoid_1:y:0#backward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22(
&backward_lstm_4/while/lstm_cell_14/mul�
'backward_lstm_4/while/lstm_cell_14/ReluRelu1backward_lstm_4/while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22)
'backward_lstm_4/while/lstm_cell_14/Relu�
(backward_lstm_4/while/lstm_cell_14/mul_1Mul.backward_lstm_4/while/lstm_cell_14/Sigmoid:y:05backward_lstm_4/while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22*
(backward_lstm_4/while/lstm_cell_14/mul_1�
(backward_lstm_4/while/lstm_cell_14/add_1AddV2*backward_lstm_4/while/lstm_cell_14/mul:z:0,backward_lstm_4/while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22*
(backward_lstm_4/while/lstm_cell_14/add_1�
,backward_lstm_4/while/lstm_cell_14/Sigmoid_2Sigmoid1backward_lstm_4/while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22.
,backward_lstm_4/while/lstm_cell_14/Sigmoid_2�
)backward_lstm_4/while/lstm_cell_14/Relu_1Relu,backward_lstm_4/while/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22+
)backward_lstm_4/while/lstm_cell_14/Relu_1�
(backward_lstm_4/while/lstm_cell_14/mul_2Mul0backward_lstm_4/while/lstm_cell_14/Sigmoid_2:y:07backward_lstm_4/while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22*
(backward_lstm_4/while/lstm_cell_14/mul_2�
backward_lstm_4/while/SelectSelectbackward_lstm_4/while/Less:z:0,backward_lstm_4/while/lstm_cell_14/mul_2:z:0#backward_lstm_4_while_placeholder_2*
T0*'
_output_shapes
:���������22
backward_lstm_4/while/Select�
backward_lstm_4/while/Select_1Selectbackward_lstm_4/while/Less:z:0,backward_lstm_4/while/lstm_cell_14/mul_2:z:0#backward_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:���������22 
backward_lstm_4/while/Select_1�
backward_lstm_4/while/Select_2Selectbackward_lstm_4/while/Less:z:0,backward_lstm_4/while/lstm_cell_14/add_1:z:0#backward_lstm_4_while_placeholder_4*
T0*'
_output_shapes
:���������22 
backward_lstm_4/while/Select_2�
:backward_lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#backward_lstm_4_while_placeholder_1!backward_lstm_4_while_placeholder%backward_lstm_4/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:backward_lstm_4/while/TensorArrayV2Write/TensorListSetItem|
backward_lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_4/while/add/y�
backward_lstm_4/while/addAddV2!backward_lstm_4_while_placeholder$backward_lstm_4/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/while/add�
backward_lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_4/while/add_1/y�
backward_lstm_4/while/add_1AddV28backward_lstm_4_while_backward_lstm_4_while_loop_counter&backward_lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/while/add_1�
backward_lstm_4/while/IdentityIdentitybackward_lstm_4/while/add_1:z:0^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2 
backward_lstm_4/while/Identity�
 backward_lstm_4/while/Identity_1Identity>backward_lstm_4_while_backward_lstm_4_while_maximum_iterations^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_4/while/Identity_1�
 backward_lstm_4/while/Identity_2Identitybackward_lstm_4/while/add:z:0^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_4/while/Identity_2�
 backward_lstm_4/while/Identity_3IdentityJbackward_lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_4/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_4/while/Identity_3�
 backward_lstm_4/while/Identity_4Identity%backward_lstm_4/while/Select:output:0^backward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/while/Identity_4�
 backward_lstm_4/while/Identity_5Identity'backward_lstm_4/while/Select_1:output:0^backward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/while/Identity_5�
 backward_lstm_4/while/Identity_6Identity'backward_lstm_4/while/Select_2:output:0^backward_lstm_4/while/NoOp*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/while/Identity_6�
backward_lstm_4/while/NoOpNoOp:^backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp9^backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp;^backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_4/while/NoOp"p
5backward_lstm_4_while_backward_lstm_4_strided_slice_17backward_lstm_4_while_backward_lstm_4_strided_slice_1_0"I
backward_lstm_4_while_identity'backward_lstm_4/while/Identity:output:0"M
 backward_lstm_4_while_identity_1)backward_lstm_4/while/Identity_1:output:0"M
 backward_lstm_4_while_identity_2)backward_lstm_4/while/Identity_2:output:0"M
 backward_lstm_4_while_identity_3)backward_lstm_4/while/Identity_3:output:0"M
 backward_lstm_4_while_identity_4)backward_lstm_4/while/Identity_4:output:0"M
 backward_lstm_4_while_identity_5)backward_lstm_4/while/Identity_5:output:0"M
 backward_lstm_4_while_identity_6)backward_lstm_4/while/Identity_6:output:0"f
0backward_lstm_4_while_less_backward_lstm_4_sub_12backward_lstm_4_while_less_backward_lstm_4_sub_1_0"�
Bbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resourceDbackward_lstm_4_while_lstm_cell_14_biasadd_readvariableop_resource_0"�
Cbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resourceEbackward_lstm_4_while_lstm_cell_14_matmul_1_readvariableop_resource_0"�
Abackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resourceCbackward_lstm_4_while_lstm_cell_14_matmul_readvariableop_resource_0"�
qbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensorsbackward_lstm_4_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : 2v
9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp9backward_lstm_4/while/lstm_cell_14/BiasAdd/ReadVariableOp2t
8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp8backward_lstm_4/while/lstm_cell_14/MatMul/ReadVariableOp2x
:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp:backward_lstm_4/while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
while_cond_691866
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_691866___redundant_placeholder04
0while_while_cond_691866___redundant_placeholder14
0while_while_cond_691866___redundant_placeholder24
0while_while_cond_691866___redundant_placeholder3
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
0__inference_backward_lstm_4_layer_call_fn_695673

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
GPU 2J 8� *T
fORM
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_6921112
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
�
�
H__inference_sequential_4_layer_call_and_return_conditional_losses_693465

inputs
inputs_1	)
bidirectional_4_693446:	�)
bidirectional_4_693448:	2�%
bidirectional_4_693450:	�)
bidirectional_4_693452:	�)
bidirectional_4_693454:	2�%
bidirectional_4_693456:	� 
dense_4_693459:d
dense_4_693461:
identity��'bidirectional_4/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�
'bidirectional_4/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_4_693446bidirectional_4_693448bidirectional_4_693450bidirectional_4_693452bidirectional_4_693454bidirectional_4_693456*
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
GPU 2J 8� *T
fORM
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_6934022)
'bidirectional_4/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall0bidirectional_4/StatefulPartitionedCall:output:0dense_4_693459dense_4_693461*
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
GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_6929872!
dense_4/StatefulPartitionedCall�
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp(^bidirectional_4/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:���������:���������: : : : : : : : 2R
'bidirectional_4/StatefulPartitionedCall'bidirectional_4/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
=sequential_4_bidirectional_4_forward_lstm_4_while_cond_690242t
psequential_4_bidirectional_4_forward_lstm_4_while_sequential_4_bidirectional_4_forward_lstm_4_while_loop_counterz
vsequential_4_bidirectional_4_forward_lstm_4_while_sequential_4_bidirectional_4_forward_lstm_4_while_maximum_iterationsA
=sequential_4_bidirectional_4_forward_lstm_4_while_placeholderC
?sequential_4_bidirectional_4_forward_lstm_4_while_placeholder_1C
?sequential_4_bidirectional_4_forward_lstm_4_while_placeholder_2C
?sequential_4_bidirectional_4_forward_lstm_4_while_placeholder_3C
?sequential_4_bidirectional_4_forward_lstm_4_while_placeholder_4v
rsequential_4_bidirectional_4_forward_lstm_4_while_less_sequential_4_bidirectional_4_forward_lstm_4_strided_slice_1�
�sequential_4_bidirectional_4_forward_lstm_4_while_sequential_4_bidirectional_4_forward_lstm_4_while_cond_690242___redundant_placeholder0�
�sequential_4_bidirectional_4_forward_lstm_4_while_sequential_4_bidirectional_4_forward_lstm_4_while_cond_690242___redundant_placeholder1�
�sequential_4_bidirectional_4_forward_lstm_4_while_sequential_4_bidirectional_4_forward_lstm_4_while_cond_690242___redundant_placeholder2�
�sequential_4_bidirectional_4_forward_lstm_4_while_sequential_4_bidirectional_4_forward_lstm_4_while_cond_690242___redundant_placeholder3�
�sequential_4_bidirectional_4_forward_lstm_4_while_sequential_4_bidirectional_4_forward_lstm_4_while_cond_690242___redundant_placeholder4>
:sequential_4_bidirectional_4_forward_lstm_4_while_identity
�
6sequential_4/bidirectional_4/forward_lstm_4/while/LessLess=sequential_4_bidirectional_4_forward_lstm_4_while_placeholderrsequential_4_bidirectional_4_forward_lstm_4_while_less_sequential_4_bidirectional_4_forward_lstm_4_strided_slice_1*
T0*
_output_shapes
: 28
6sequential_4/bidirectional_4/forward_lstm_4/while/Less�
:sequential_4/bidirectional_4/forward_lstm_4/while/IdentityIdentity:sequential_4/bidirectional_4/forward_lstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2<
:sequential_4/bidirectional_4/forward_lstm_4/while/Identity"�
:sequential_4_bidirectional_4_forward_lstm_4_while_identityCsequential_4/bidirectional_4/forward_lstm_4/while/Identity:output:0*(
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
Ʋ
�
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_694614

inputs
inputs_1	M
:forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource:	�O
<forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource:	2�J
;forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource:	�N
;backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource:	�P
=backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource:	2�K
<backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource:	�
identity��3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp�2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp�4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp�backward_lstm_4/while�2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp�1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp�3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp�forward_lstm_4/while�
#forward_lstm_4/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2%
#forward_lstm_4/RaggedToTensor/zeros�
#forward_lstm_4/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2%
#forward_lstm_4/RaggedToTensor/Const�
2forward_lstm_4/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor,forward_lstm_4/RaggedToTensor/Const:output:0inputs,forward_lstm_4/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS24
2forward_lstm_4/RaggedToTensor/RaggedTensorToTensor�
9forward_lstm_4/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack�
;forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1�
;forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2�
3forward_lstm_4/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Bforward_lstm_4/RaggedNestedRowLengths/strided_slice/stack:output:0Dforward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1:output:0Dforward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask25
3forward_lstm_4/RaggedNestedRowLengths/strided_slice�
;forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack�
=forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2?
=forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1�
=forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2�
5forward_lstm_4/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Dforward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack:output:0Fforward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Fforward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask27
5forward_lstm_4/RaggedNestedRowLengths/strided_slice_1�
)forward_lstm_4/RaggedNestedRowLengths/subSub<forward_lstm_4/RaggedNestedRowLengths/strided_slice:output:0>forward_lstm_4/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2+
)forward_lstm_4/RaggedNestedRowLengths/sub�
forward_lstm_4/CastCast-forward_lstm_4/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
forward_lstm_4/Cast�
forward_lstm_4/ShapeShape;forward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_4/Shape�
"forward_lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"forward_lstm_4/strided_slice/stack�
$forward_lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_4/strided_slice/stack_1�
$forward_lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_4/strided_slice/stack_2�
forward_lstm_4/strided_sliceStridedSliceforward_lstm_4/Shape:output:0+forward_lstm_4/strided_slice/stack:output:0-forward_lstm_4/strided_slice/stack_1:output:0-forward_lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_4/strided_slicez
forward_lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_4/zeros/mul/y�
forward_lstm_4/zeros/mulMul%forward_lstm_4/strided_slice:output:0#forward_lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros/mul}
forward_lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_4/zeros/Less/y�
forward_lstm_4/zeros/LessLessforward_lstm_4/zeros/mul:z:0$forward_lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros/Less�
forward_lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_4/zeros/packed/1�
forward_lstm_4/zeros/packedPack%forward_lstm_4/strided_slice:output:0&forward_lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_4/zeros/packed�
forward_lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_4/zeros/Const�
forward_lstm_4/zerosFill$forward_lstm_4/zeros/packed:output:0#forward_lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_4/zeros~
forward_lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_4/zeros_1/mul/y�
forward_lstm_4/zeros_1/mulMul%forward_lstm_4/strided_slice:output:0%forward_lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros_1/mul�
forward_lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
forward_lstm_4/zeros_1/Less/y�
forward_lstm_4/zeros_1/LessLessforward_lstm_4/zeros_1/mul:z:0&forward_lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_4/zeros_1/Less�
forward_lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_4/zeros_1/packed/1�
forward_lstm_4/zeros_1/packedPack%forward_lstm_4/strided_slice:output:0(forward_lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_4/zeros_1/packed�
forward_lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_4/zeros_1/Const�
forward_lstm_4/zeros_1Fill&forward_lstm_4/zeros_1/packed:output:0%forward_lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
forward_lstm_4/zeros_1�
forward_lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
forward_lstm_4/transpose/perm�
forward_lstm_4/transpose	Transpose;forward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0&forward_lstm_4/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
forward_lstm_4/transpose|
forward_lstm_4/Shape_1Shapeforward_lstm_4/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_4/Shape_1�
$forward_lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_4/strided_slice_1/stack�
&forward_lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_1/stack_1�
&forward_lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_1/stack_2�
forward_lstm_4/strided_slice_1StridedSliceforward_lstm_4/Shape_1:output:0-forward_lstm_4/strided_slice_1/stack:output:0/forward_lstm_4/strided_slice_1/stack_1:output:0/forward_lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_4/strided_slice_1�
*forward_lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2,
*forward_lstm_4/TensorArrayV2/element_shape�
forward_lstm_4/TensorArrayV2TensorListReserve3forward_lstm_4/TensorArrayV2/element_shape:output:0'forward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_4/TensorArrayV2�
Dforward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2F
Dforward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape�
6forward_lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_4/transpose:y:0Mforward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6forward_lstm_4/TensorArrayUnstack/TensorListFromTensor�
$forward_lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_4/strided_slice_2/stack�
&forward_lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_2/stack_1�
&forward_lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_2/stack_2�
forward_lstm_4/strided_slice_2StridedSliceforward_lstm_4/transpose:y:0-forward_lstm_4/strided_slice_2/stack:output:0/forward_lstm_4/strided_slice_2/stack_1:output:0/forward_lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2 
forward_lstm_4/strided_slice_2�
1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp:forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype023
1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp�
"forward_lstm_4/lstm_cell_13/MatMulMatMul'forward_lstm_4/strided_slice_2:output:09forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"forward_lstm_4/lstm_cell_13/MatMul�
3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp<forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype025
3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp�
$forward_lstm_4/lstm_cell_13/MatMul_1MatMulforward_lstm_4/zeros:output:0;forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$forward_lstm_4/lstm_cell_13/MatMul_1�
forward_lstm_4/lstm_cell_13/addAddV2,forward_lstm_4/lstm_cell_13/MatMul:product:0.forward_lstm_4/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2!
forward_lstm_4/lstm_cell_13/add�
2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp;forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype024
2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp�
#forward_lstm_4/lstm_cell_13/BiasAddBiasAdd#forward_lstm_4/lstm_cell_13/add:z:0:forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#forward_lstm_4/lstm_cell_13/BiasAdd�
+forward_lstm_4/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+forward_lstm_4/lstm_cell_13/split/split_dim�
!forward_lstm_4/lstm_cell_13/splitSplit4forward_lstm_4/lstm_cell_13/split/split_dim:output:0,forward_lstm_4/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2#
!forward_lstm_4/lstm_cell_13/split�
#forward_lstm_4/lstm_cell_13/SigmoidSigmoid*forward_lstm_4/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22%
#forward_lstm_4/lstm_cell_13/Sigmoid�
%forward_lstm_4/lstm_cell_13/Sigmoid_1Sigmoid*forward_lstm_4/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22'
%forward_lstm_4/lstm_cell_13/Sigmoid_1�
forward_lstm_4/lstm_cell_13/mulMul)forward_lstm_4/lstm_cell_13/Sigmoid_1:y:0forward_lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:���������22!
forward_lstm_4/lstm_cell_13/mul�
 forward_lstm_4/lstm_cell_13/ReluRelu*forward_lstm_4/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22"
 forward_lstm_4/lstm_cell_13/Relu�
!forward_lstm_4/lstm_cell_13/mul_1Mul'forward_lstm_4/lstm_cell_13/Sigmoid:y:0.forward_lstm_4/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22#
!forward_lstm_4/lstm_cell_13/mul_1�
!forward_lstm_4/lstm_cell_13/add_1AddV2#forward_lstm_4/lstm_cell_13/mul:z:0%forward_lstm_4/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22#
!forward_lstm_4/lstm_cell_13/add_1�
%forward_lstm_4/lstm_cell_13/Sigmoid_2Sigmoid*forward_lstm_4/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22'
%forward_lstm_4/lstm_cell_13/Sigmoid_2�
"forward_lstm_4/lstm_cell_13/Relu_1Relu%forward_lstm_4/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22$
"forward_lstm_4/lstm_cell_13/Relu_1�
!forward_lstm_4/lstm_cell_13/mul_2Mul)forward_lstm_4/lstm_cell_13/Sigmoid_2:y:00forward_lstm_4/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22#
!forward_lstm_4/lstm_cell_13/mul_2�
,forward_lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2.
,forward_lstm_4/TensorArrayV2_1/element_shape�
forward_lstm_4/TensorArrayV2_1TensorListReserve5forward_lstm_4/TensorArrayV2_1/element_shape:output:0'forward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_4/TensorArrayV2_1l
forward_lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_4/time�
forward_lstm_4/zeros_like	ZerosLike%forward_lstm_4/lstm_cell_13/mul_2:z:0*
T0*'
_output_shapes
:���������22
forward_lstm_4/zeros_like�
'forward_lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2)
'forward_lstm_4/while/maximum_iterations�
!forward_lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!forward_lstm_4/while/loop_counter�
forward_lstm_4/whileWhile*forward_lstm_4/while/loop_counter:output:00forward_lstm_4/while/maximum_iterations:output:0forward_lstm_4/time:output:0'forward_lstm_4/TensorArrayV2_1:handle:0forward_lstm_4/zeros_like:y:0forward_lstm_4/zeros:output:0forward_lstm_4/zeros_1:output:0'forward_lstm_4/strided_slice_1:output:0Fforward_lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_4/Cast:y:0:forward_lstm_4_lstm_cell_13_matmul_readvariableop_resource<forward_lstm_4_lstm_cell_13_matmul_1_readvariableop_resource;forward_lstm_4_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *,
body$R"
 forward_lstm_4_while_body_694338*,
cond$R"
 forward_lstm_4_while_cond_694337*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
forward_lstm_4/while�
?forward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2A
?forward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape�
1forward_lstm_4/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_4/while:output:3Hforward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype023
1forward_lstm_4/TensorArrayV2Stack/TensorListStack�
$forward_lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2&
$forward_lstm_4/strided_slice_3/stack�
&forward_lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_4/strided_slice_3/stack_1�
&forward_lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_4/strided_slice_3/stack_2�
forward_lstm_4/strided_slice_3StridedSlice:forward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0-forward_lstm_4/strided_slice_3/stack:output:0/forward_lstm_4/strided_slice_3/stack_1:output:0/forward_lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2 
forward_lstm_4/strided_slice_3�
forward_lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_4/transpose_1/perm�
forward_lstm_4/transpose_1	Transpose:forward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0(forward_lstm_4/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
forward_lstm_4/transpose_1�
forward_lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_4/runtime�
$backward_lstm_4/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$backward_lstm_4/RaggedToTensor/zeros�
$backward_lstm_4/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2&
$backward_lstm_4/RaggedToTensor/Const�
3backward_lstm_4/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-backward_lstm_4/RaggedToTensor/Const:output:0inputs-backward_lstm_4/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3backward_lstm_4/RaggedToTensor/RaggedTensorToTensor�
:backward_lstm_4/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack�
<backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1�
<backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2�
4backward_lstm_4/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cbackward_lstm_4/RaggedNestedRowLengths/strided_slice/stack:output:0Ebackward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_1:output:0Ebackward_lstm_4/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*
end_mask26
4backward_lstm_4/RaggedNestedRowLengths/strided_slice�
<backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack�
>backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2@
>backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1�
>backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2�
6backward_lstm_4/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Ebackward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gbackward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gbackward_lstm_4/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask28
6backward_lstm_4/RaggedNestedRowLengths/strided_slice_1�
*backward_lstm_4/RaggedNestedRowLengths/subSub=backward_lstm_4/RaggedNestedRowLengths/strided_slice:output:0?backward_lstm_4/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:���������2,
*backward_lstm_4/RaggedNestedRowLengths/sub�
backward_lstm_4/CastCast.backward_lstm_4/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
backward_lstm_4/Cast�
backward_lstm_4/ShapeShape<backward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_4/Shape�
#backward_lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#backward_lstm_4/strided_slice/stack�
%backward_lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_4/strided_slice/stack_1�
%backward_lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_4/strided_slice/stack_2�
backward_lstm_4/strided_sliceStridedSlicebackward_lstm_4/Shape:output:0,backward_lstm_4/strided_slice/stack:output:0.backward_lstm_4/strided_slice/stack_1:output:0.backward_lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
backward_lstm_4/strided_slice|
backward_lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_4/zeros/mul/y�
backward_lstm_4/zeros/mulMul&backward_lstm_4/strided_slice:output:0$backward_lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros/mul
backward_lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
backward_lstm_4/zeros/Less/y�
backward_lstm_4/zeros/LessLessbackward_lstm_4/zeros/mul:z:0%backward_lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros/Less�
backward_lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_4/zeros/packed/1�
backward_lstm_4/zeros/packedPack&backward_lstm_4/strided_slice:output:0'backward_lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_4/zeros/packed�
backward_lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_4/zeros/Const�
backward_lstm_4/zerosFill%backward_lstm_4/zeros/packed:output:0$backward_lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_4/zeros�
backward_lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_4/zeros_1/mul/y�
backward_lstm_4/zeros_1/mulMul&backward_lstm_4/strided_slice:output:0&backward_lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros_1/mul�
backward_lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
backward_lstm_4/zeros_1/Less/y�
backward_lstm_4/zeros_1/LessLessbackward_lstm_4/zeros_1/mul:z:0'backward_lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/zeros_1/Less�
 backward_lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_4/zeros_1/packed/1�
backward_lstm_4/zeros_1/packedPack&backward_lstm_4/strided_slice:output:0)backward_lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_4/zeros_1/packed�
backward_lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_4/zeros_1/Const�
backward_lstm_4/zeros_1Fill'backward_lstm_4/zeros_1/packed:output:0&backward_lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������22
backward_lstm_4/zeros_1�
backward_lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
backward_lstm_4/transpose/perm�
backward_lstm_4/transpose	Transpose<backward_lstm_4/RaggedToTensor/RaggedTensorToTensor:result:0'backward_lstm_4/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_4/transpose
backward_lstm_4/Shape_1Shapebackward_lstm_4/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_4/Shape_1�
%backward_lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_4/strided_slice_1/stack�
'backward_lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_1/stack_1�
'backward_lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_1/stack_2�
backward_lstm_4/strided_slice_1StridedSlice backward_lstm_4/Shape_1:output:0.backward_lstm_4/strided_slice_1/stack:output:00backward_lstm_4/strided_slice_1/stack_1:output:00backward_lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_4/strided_slice_1�
+backward_lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+backward_lstm_4/TensorArrayV2/element_shape�
backward_lstm_4/TensorArrayV2TensorListReserve4backward_lstm_4/TensorArrayV2/element_shape:output:0(backward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
backward_lstm_4/TensorArrayV2�
backward_lstm_4/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2 
backward_lstm_4/ReverseV2/axis�
backward_lstm_4/ReverseV2	ReverseV2backward_lstm_4/transpose:y:0'backward_lstm_4/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������2
backward_lstm_4/ReverseV2�
Ebackward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2G
Ebackward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape�
7backward_lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"backward_lstm_4/ReverseV2:output:0Nbackward_lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7backward_lstm_4/TensorArrayUnstack/TensorListFromTensor�
%backward_lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_4/strided_slice_2/stack�
'backward_lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_2/stack_1�
'backward_lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_2/stack_2�
backward_lstm_4/strided_slice_2StridedSlicebackward_lstm_4/transpose:y:0.backward_lstm_4/strided_slice_2/stack:output:00backward_lstm_4/strided_slice_2/stack_1:output:00backward_lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2!
backward_lstm_4/strided_slice_2�
2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp;backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp�
#backward_lstm_4/lstm_cell_14/MatMulMatMul(backward_lstm_4/strided_slice_2:output:0:backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#backward_lstm_4/lstm_cell_14/MatMul�
4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp=backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype026
4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp�
%backward_lstm_4/lstm_cell_14/MatMul_1MatMulbackward_lstm_4/zeros:output:0<backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%backward_lstm_4/lstm_cell_14/MatMul_1�
 backward_lstm_4/lstm_cell_14/addAddV2-backward_lstm_4/lstm_cell_14/MatMul:product:0/backward_lstm_4/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:����������2"
 backward_lstm_4/lstm_cell_14/add�
3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp<backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp�
$backward_lstm_4/lstm_cell_14/BiasAddBiasAdd$backward_lstm_4/lstm_cell_14/add:z:0;backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$backward_lstm_4/lstm_cell_14/BiasAdd�
,backward_lstm_4/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,backward_lstm_4/lstm_cell_14/split/split_dim�
"backward_lstm_4/lstm_cell_14/splitSplit5backward_lstm_4/lstm_cell_14/split/split_dim:output:0-backward_lstm_4/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2$
"backward_lstm_4/lstm_cell_14/split�
$backward_lstm_4/lstm_cell_14/SigmoidSigmoid+backward_lstm_4/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:���������22&
$backward_lstm_4/lstm_cell_14/Sigmoid�
&backward_lstm_4/lstm_cell_14/Sigmoid_1Sigmoid+backward_lstm_4/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:���������22(
&backward_lstm_4/lstm_cell_14/Sigmoid_1�
 backward_lstm_4/lstm_cell_14/mulMul*backward_lstm_4/lstm_cell_14/Sigmoid_1:y:0 backward_lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:���������22"
 backward_lstm_4/lstm_cell_14/mul�
!backward_lstm_4/lstm_cell_14/ReluRelu+backward_lstm_4/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:���������22#
!backward_lstm_4/lstm_cell_14/Relu�
"backward_lstm_4/lstm_cell_14/mul_1Mul(backward_lstm_4/lstm_cell_14/Sigmoid:y:0/backward_lstm_4/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:���������22$
"backward_lstm_4/lstm_cell_14/mul_1�
"backward_lstm_4/lstm_cell_14/add_1AddV2$backward_lstm_4/lstm_cell_14/mul:z:0&backward_lstm_4/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:���������22$
"backward_lstm_4/lstm_cell_14/add_1�
&backward_lstm_4/lstm_cell_14/Sigmoid_2Sigmoid+backward_lstm_4/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:���������22(
&backward_lstm_4/lstm_cell_14/Sigmoid_2�
#backward_lstm_4/lstm_cell_14/Relu_1Relu&backward_lstm_4/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:���������22%
#backward_lstm_4/lstm_cell_14/Relu_1�
"backward_lstm_4/lstm_cell_14/mul_2Mul*backward_lstm_4/lstm_cell_14/Sigmoid_2:y:01backward_lstm_4/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:���������22$
"backward_lstm_4/lstm_cell_14/mul_2�
-backward_lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2/
-backward_lstm_4/TensorArrayV2_1/element_shape�
backward_lstm_4/TensorArrayV2_1TensorListReserve6backward_lstm_4/TensorArrayV2_1/element_shape:output:0(backward_lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_4/TensorArrayV2_1n
backward_lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_4/time�
%backward_lstm_4/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2'
%backward_lstm_4/Max/reduction_indices�
backward_lstm_4/MaxMaxbackward_lstm_4/Cast:y:0.backward_lstm_4/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/Maxp
backward_lstm_4/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_4/sub/y�
backward_lstm_4/subSubbackward_lstm_4/Max:output:0backward_lstm_4/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_4/sub�
backward_lstm_4/Sub_1Subbackward_lstm_4/sub:z:0backward_lstm_4/Cast:y:0*
T0*#
_output_shapes
:���������2
backward_lstm_4/Sub_1�
backward_lstm_4/zeros_like	ZerosLike&backward_lstm_4/lstm_cell_14/mul_2:z:0*
T0*'
_output_shapes
:���������22
backward_lstm_4/zeros_like�
(backward_lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(backward_lstm_4/while/maximum_iterations�
"backward_lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"backward_lstm_4/while/loop_counter�
backward_lstm_4/whileWhile+backward_lstm_4/while/loop_counter:output:01backward_lstm_4/while/maximum_iterations:output:0backward_lstm_4/time:output:0(backward_lstm_4/TensorArrayV2_1:handle:0backward_lstm_4/zeros_like:y:0backward_lstm_4/zeros:output:0 backward_lstm_4/zeros_1:output:0(backward_lstm_4/strided_slice_1:output:0Gbackward_lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_4/Sub_1:z:0;backward_lstm_4_lstm_cell_14_matmul_readvariableop_resource=backward_lstm_4_lstm_cell_14_matmul_1_readvariableop_resource<backward_lstm_4_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *-
body%R#
!backward_lstm_4_while_body_694517*-
cond%R#
!backward_lstm_4_while_cond_694516*m
output_shapes\
Z: : : : :���������2:���������2:���������2: : :���������: : : *
parallel_iterations 2
backward_lstm_4/while�
@backward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����2   2B
@backward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape�
2backward_lstm_4/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_4/while:output:3Ibackward_lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������2*
element_dtype024
2backward_lstm_4/TensorArrayV2Stack/TensorListStack�
%backward_lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2'
%backward_lstm_4/strided_slice_3/stack�
'backward_lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_4/strided_slice_3/stack_1�
'backward_lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_4/strided_slice_3/stack_2�
backward_lstm_4/strided_slice_3StridedSlice;backward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0.backward_lstm_4/strided_slice_3/stack:output:00backward_lstm_4/strided_slice_3/stack_1:output:00backward_lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������2*
shrink_axis_mask2!
backward_lstm_4/strided_slice_3�
 backward_lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_4/transpose_1/perm�
backward_lstm_4/transpose_1	Transpose;backward_lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0)backward_lstm_4/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������22
backward_lstm_4/transpose_1�
backward_lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_4/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2'forward_lstm_4/strided_slice_3:output:0(backward_lstm_4/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp4^backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp3^backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp5^backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp^backward_lstm_4/while3^forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp2^forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp4^forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp^forward_lstm_4/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������: : : : : : 2j
3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp3backward_lstm_4/lstm_cell_14/BiasAdd/ReadVariableOp2h
2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp2backward_lstm_4/lstm_cell_14/MatMul/ReadVariableOp2l
4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp4backward_lstm_4/lstm_cell_14/MatMul_1/ReadVariableOp2.
backward_lstm_4/whilebackward_lstm_4/while2h
2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp2forward_lstm_4/lstm_cell_13/BiasAdd/ReadVariableOp2f
1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp1forward_lstm_4/lstm_cell_13/MatMul/ReadVariableOp2j
3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp3forward_lstm_4/lstm_cell_13/MatMul_1/ReadVariableOp2,
forward_lstm_4/whileforward_lstm_4/while:O K
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
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_691379

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
�F
�
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_690684

inputs&
lstm_cell_13_690602:	�&
lstm_cell_13_690604:	2�"
lstm_cell_13_690606:	�
identity��$lstm_cell_13/StatefulPartitionedCall�whileD
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
$lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_13_690602lstm_cell_13_690604lstm_cell_13_690606*
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
GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_6906012&
$lstm_cell_13/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_13_690602lstm_cell_13_690604lstm_cell_13_690606*
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
bodyR
while_body_690615*
condR
while_cond_690614*K
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
NoOpNoOp%^lstm_cell_13/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_13/StatefulPartitionedCall$lstm_cell_13/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�\
�
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_695640

inputs>
+lstm_cell_13_matmul_readvariableop_resource:	�@
-lstm_cell_13_matmul_1_readvariableop_resource:	2�;
,lstm_cell_13_biasadd_readvariableop_resource:	�
identity��#lstm_cell_13/BiasAdd/ReadVariableOp�"lstm_cell_13/MatMul/ReadVariableOp�$lstm_cell_13/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp�
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/MatMul�
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	2�*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp�
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/MatMul_1�
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/add�
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp�
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim�
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������2:���������2:���������2:���������2*
	num_split2
lstm_cell_13/split�
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/Sigmoid�
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*'
_output_shapes
:���������22
lstm_cell_13/Sigmoid_1�
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/mul}
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*'
_output_shapes
:���������22
lstm_cell_13/Relu�
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/mul_1�
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/add_1�
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*'
_output_shapes
:���������22
lstm_cell_13/Sigmoid_2|
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/Relu_1�
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:���������22
lstm_cell_13/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
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
bodyR
while_body_695556*
condR
while_cond_695555*K
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
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
 forward_lstm_4_while_cond_693125:
6forward_lstm_4_while_forward_lstm_4_while_loop_counter@
<forward_lstm_4_while_forward_lstm_4_while_maximum_iterations$
 forward_lstm_4_while_placeholder&
"forward_lstm_4_while_placeholder_1&
"forward_lstm_4_while_placeholder_2&
"forward_lstm_4_while_placeholder_3&
"forward_lstm_4_while_placeholder_4<
8forward_lstm_4_while_less_forward_lstm_4_strided_slice_1R
Nforward_lstm_4_while_forward_lstm_4_while_cond_693125___redundant_placeholder0R
Nforward_lstm_4_while_forward_lstm_4_while_cond_693125___redundant_placeholder1R
Nforward_lstm_4_while_forward_lstm_4_while_cond_693125___redundant_placeholder2R
Nforward_lstm_4_while_forward_lstm_4_while_cond_693125___redundant_placeholder3R
Nforward_lstm_4_while_forward_lstm_4_while_cond_693125___redundant_placeholder4!
forward_lstm_4_while_identity
�
forward_lstm_4/while/LessLess forward_lstm_4_while_placeholder8forward_lstm_4_while_less_forward_lstm_4_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_4/while/Less�
forward_lstm_4/while/IdentityIdentityforward_lstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2
forward_lstm_4/while/Identity"G
forward_lstm_4_while_identity&forward_lstm_4/while/Identity:output:0*(
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
�
�
H__inference_sequential_4_layer_call_and_return_conditional_losses_693529

inputs
inputs_1	)
bidirectional_4_693510:	�)
bidirectional_4_693512:	2�%
bidirectional_4_693514:	�)
bidirectional_4_693516:	�)
bidirectional_4_693518:	2�%
bidirectional_4_693520:	� 
dense_4_693523:d
dense_4_693525:
identity��'bidirectional_4/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�
'bidirectional_4/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_4_693510bidirectional_4_693512bidirectional_4_693514bidirectional_4_693516bidirectional_4_693518bidirectional_4_693520*
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
GPU 2J 8� *T
fORM
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_6929622)
'bidirectional_4/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall0bidirectional_4/StatefulPartitionedCall:output:0dense_4_693523dense_4_693525*
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
GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_6929872!
dense_4/StatefulPartitionedCall�
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp(^bidirectional_4/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:���������:���������: : : : : : : : 2R
'bidirectional_4/StatefulPartitionedCall'bidirectional_4/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:O K
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
while_cond_695102
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_695102___redundant_placeholder04
0while_while_cond_695102___redundant_placeholder14
0while_while_cond_695102___redundant_placeholder24
0while_while_cond_695102___redundant_placeholder3
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
:"�L
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
serving_default_args_0_1:0	���������;
dense_40
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
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
 :d2dense_4/kernel
:2dense_4/bias
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
E:C	�22bidirectional_4/forward_lstm_4/lstm_cell_13/kernel
O:M	2�2<bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel
?:=�20bidirectional_4/forward_lstm_4/lstm_cell_13/bias
F:D	�23bidirectional_4/backward_lstm_4/lstm_cell_14/kernel
P:N	2�2=bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel
@:>�21bidirectional_4/backward_lstm_4/lstm_cell_14/bias
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
%:#d2Adam/dense_4/kernel/m
:2Adam/dense_4/bias/m
J:H	�29Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/m
T:R	2�2CAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/m
D:B�27Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/m
K:I	�2:Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/m
U:S	2�2DAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/m
E:C�28Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/m
%:#d2Adam/dense_4/kernel/v
:2Adam/dense_4/bias/v
J:H	�29Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/v
T:R	2�2CAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/v
D:B�27Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/v
K:I	�2:Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/v
U:S	2�2DAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/v
E:C�28Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/v
(:&d2Adam/dense_4/kernel/vhat
": 2Adam/dense_4/bias/vhat
M:K	�2<Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel/vhat
W:U	2�2FAdam/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel/vhat
G:E�2:Adam/bidirectional_4/forward_lstm_4/lstm_cell_13/bias/vhat
N:L	�2=Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel/vhat
X:V	2�2GAdam/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel/vhat
H:F�2;Adam/bidirectional_4/backward_lstm_4/lstm_cell_14/bias/vhat
�2�
-__inference_sequential_4_layer_call_fn_693013
-__inference_sequential_4_layer_call_fn_693506�
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
!__inference__wrapped_model_690526args_0args_0_1"�
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
H__inference_sequential_4_layer_call_and_return_conditional_losses_693529
H__inference_sequential_4_layer_call_and_return_conditional_losses_693552�
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
0__inference_bidirectional_4_layer_call_fn_693599
0__inference_bidirectional_4_layer_call_fn_693616
0__inference_bidirectional_4_layer_call_fn_693634
0__inference_bidirectional_4_layer_call_fn_693652�
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
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_693954
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_694256
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_694614
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_694972�
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
(__inference_dense_4_layer_call_fn_694981�
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
C__inference_dense_4_layer_call_and_return_conditional_losses_694992�
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
$__inference_signature_wrapper_693582args_0args_0_1"�
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
/__inference_forward_lstm_4_layer_call_fn_695003
/__inference_forward_lstm_4_layer_call_fn_695014
/__inference_forward_lstm_4_layer_call_fn_695025
/__inference_forward_lstm_4_layer_call_fn_695036�
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
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_695187
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_695338
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_695489
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_695640�
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
0__inference_backward_lstm_4_layer_call_fn_695651
0__inference_backward_lstm_4_layer_call_fn_695662
0__inference_backward_lstm_4_layer_call_fn_695673
0__inference_backward_lstm_4_layer_call_fn_695684�
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
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_695837
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_695990
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_696143
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_696296�
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
-__inference_lstm_cell_13_layer_call_fn_696313
-__inference_lstm_cell_13_layer_call_fn_696330�
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
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_696362
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_696394�
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
-__inference_lstm_cell_14_layer_call_fn_696411
-__inference_lstm_cell_14_layer_call_fn_696428�
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
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_696460
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_696492�
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
!__inference__wrapped_model_690526�\�Y
R�O
M�J4�1
!�������������������
�
`
�	RaggedTensorSpec
� "1�.
,
dense_4!�
dense_4����������
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_695837}O�L
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
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_695990}O�L
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
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_696143Q�N
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
K__inference_backward_lstm_4_layer_call_and_return_conditional_losses_696296Q�N
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
0__inference_backward_lstm_4_layer_call_fn_695651pO�L
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
0__inference_backward_lstm_4_layer_call_fn_695662pO�L
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
0__inference_backward_lstm_4_layer_call_fn_695673rQ�N
G�D
6�3
inputs'���������������������������

 
p 

 
� "����������2�
0__inference_backward_lstm_4_layer_call_fn_695684rQ�N
G�D
6�3
inputs'���������������������������

 
p

 
� "����������2�
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_693954�\�Y
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
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_694256�\�Y
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
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_694614�l�i
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
K__inference_bidirectional_4_layer_call_and_return_conditional_losses_694972�l�i
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
0__inference_bidirectional_4_layer_call_fn_693599�\�Y
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
0__inference_bidirectional_4_layer_call_fn_693616�\�Y
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
0__inference_bidirectional_4_layer_call_fn_693634�l�i
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
0__inference_bidirectional_4_layer_call_fn_693652�l�i
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
C__inference_dense_4_layer_call_and_return_conditional_losses_694992\/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� {
(__inference_dense_4_layer_call_fn_694981O/�,
%�"
 �
inputs���������d
� "�����������
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_695187}O�L
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
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_695338}O�L
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
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_695489Q�N
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
J__inference_forward_lstm_4_layer_call_and_return_conditional_losses_695640Q�N
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
/__inference_forward_lstm_4_layer_call_fn_695003pO�L
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
/__inference_forward_lstm_4_layer_call_fn_695014pO�L
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
/__inference_forward_lstm_4_layer_call_fn_695025rQ�N
G�D
6�3
inputs'���������������������������

 
p 

 
� "����������2�
/__inference_forward_lstm_4_layer_call_fn_695036rQ�N
G�D
6�3
inputs'���������������������������

 
p

 
� "����������2�
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_696362���}
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
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_696394���}
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
-__inference_lstm_cell_13_layer_call_fn_696313���}
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
-__inference_lstm_cell_13_layer_call_fn_696330���}
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
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_696460���}
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
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_696492���}
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
-__inference_lstm_cell_14_layer_call_fn_696411���}
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
-__inference_lstm_cell_14_layer_call_fn_696428���}
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
H__inference_sequential_4_layer_call_and_return_conditional_losses_693529�d�a
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
H__inference_sequential_4_layer_call_and_return_conditional_losses_693552�d�a
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
-__inference_sequential_4_layer_call_fn_693013�d�a
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
-__inference_sequential_4_layer_call_fn_693506�d�a
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
$__inference_signature_wrapper_693582�e�b
� 
[�X
*
args_0 �
args_0���������
*
args_0_1�
args_0_1���������	"1�.
,
dense_4!�
dense_4���������