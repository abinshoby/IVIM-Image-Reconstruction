тн;
Їј
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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

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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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

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
dtypetype
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
list(type)(0
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
list(type)(0
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
О
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
і
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
Ћ
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handleщelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements#
handleщelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintџџџџџџџџџ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

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

&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.6.02v2.6.0-rc2-32-g919f693420e8:
z
dense_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d* 
shared_namedense_39/kernel
s
#dense_39/kernel/Read/ReadVariableOpReadVariableOpdense_39/kernel*
_output_shapes

:d*
dtype0
r
dense_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_39/bias
k
!dense_39/bias/Read/ReadVariableOpReadVariableOpdense_39/bias*
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
Ч
5bidirectional_39/forward_lstm_39/lstm_cell_118/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*F
shared_name75bidirectional_39/forward_lstm_39/lstm_cell_118/kernel
Р
Ibidirectional_39/forward_lstm_39/lstm_cell_118/kernel/Read/ReadVariableOpReadVariableOp5bidirectional_39/forward_lstm_39/lstm_cell_118/kernel*
_output_shapes
:	Ш*
dtype0
л
?bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*P
shared_nameA?bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel
д
Sbidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/Read/ReadVariableOpReadVariableOp?bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel*
_output_shapes
:	2Ш*
dtype0
П
3bidirectional_39/forward_lstm_39/lstm_cell_118/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*D
shared_name53bidirectional_39/forward_lstm_39/lstm_cell_118/bias
И
Gbidirectional_39/forward_lstm_39/lstm_cell_118/bias/Read/ReadVariableOpReadVariableOp3bidirectional_39/forward_lstm_39/lstm_cell_118/bias*
_output_shapes	
:Ш*
dtype0
Щ
6bidirectional_39/backward_lstm_39/lstm_cell_119/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*G
shared_name86bidirectional_39/backward_lstm_39/lstm_cell_119/kernel
Т
Jbidirectional_39/backward_lstm_39/lstm_cell_119/kernel/Read/ReadVariableOpReadVariableOp6bidirectional_39/backward_lstm_39/lstm_cell_119/kernel*
_output_shapes
:	Ш*
dtype0
н
@bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*Q
shared_nameB@bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel
ж
Tbidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/Read/ReadVariableOpReadVariableOp@bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel*
_output_shapes
:	2Ш*
dtype0
С
4bidirectional_39/backward_lstm_39/lstm_cell_119/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*E
shared_name64bidirectional_39/backward_lstm_39/lstm_cell_119/bias
К
Hbidirectional_39/backward_lstm_39/lstm_cell_119/bias/Read/ReadVariableOpReadVariableOp4bidirectional_39/backward_lstm_39/lstm_cell_119/bias*
_output_shapes	
:Ш*
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

Adam/dense_39/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_39/kernel/m

*Adam/dense_39/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_39/kernel/m*
_output_shapes

:d*
dtype0

Adam/dense_39/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_39/bias/m
y
(Adam/dense_39/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_39/bias/m*
_output_shapes
:*
dtype0
е
<Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*M
shared_name><Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/m
Ю
PAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/m/Read/ReadVariableOpReadVariableOp<Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/m*
_output_shapes
:	Ш*
dtype0
щ
FAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*W
shared_nameHFAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/m
т
ZAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/m*
_output_shapes
:	2Ш*
dtype0
Э
:Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*K
shared_name<:Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/m
Ц
NAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/m/Read/ReadVariableOpReadVariableOp:Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/m*
_output_shapes	
:Ш*
dtype0
з
=Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*N
shared_name?=Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/m
а
QAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/m/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/m*
_output_shapes
:	Ш*
dtype0
ы
GAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*X
shared_nameIGAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/m
ф
[Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpGAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/m*
_output_shapes
:	2Ш*
dtype0
Я
;Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*L
shared_name=;Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/m
Ш
OAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/m/Read/ReadVariableOpReadVariableOp;Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/m*
_output_shapes	
:Ш*
dtype0

Adam/dense_39/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_39/kernel/v

*Adam/dense_39/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_39/kernel/v*
_output_shapes

:d*
dtype0

Adam/dense_39/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_39/bias/v
y
(Adam/dense_39/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_39/bias/v*
_output_shapes
:*
dtype0
е
<Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*M
shared_name><Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/v
Ю
PAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/v/Read/ReadVariableOpReadVariableOp<Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/v*
_output_shapes
:	Ш*
dtype0
щ
FAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*W
shared_nameHFAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/v
т
ZAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/v*
_output_shapes
:	2Ш*
dtype0
Э
:Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*K
shared_name<:Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/v
Ц
NAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/v/Read/ReadVariableOpReadVariableOp:Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/v*
_output_shapes	
:Ш*
dtype0
з
=Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*N
shared_name?=Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/v
а
QAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/v/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/v*
_output_shapes
:	Ш*
dtype0
ы
GAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*X
shared_nameIGAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/v
ф
[Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpGAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/v*
_output_shapes
:	2Ш*
dtype0
Я
;Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*L
shared_name=;Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/v
Ш
OAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/v/Read/ReadVariableOpReadVariableOp;Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/v*
_output_shapes	
:Ш*
dtype0

Adam/dense_39/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d**
shared_nameAdam/dense_39/kernel/vhat

-Adam/dense_39/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_39/kernel/vhat*
_output_shapes

:d*
dtype0

Adam/dense_39/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/dense_39/bias/vhat

+Adam/dense_39/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_39/bias/vhat*
_output_shapes
:*
dtype0
л
?Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*P
shared_nameA?Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/vhat
д
SAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/vhat/Read/ReadVariableOpReadVariableOp?Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/vhat*
_output_shapes
:	Ш*
dtype0
я
IAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*Z
shared_nameKIAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/vhat
ш
]Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpIAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/vhat*
_output_shapes
:	2Ш*
dtype0
г
=Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*N
shared_name?=Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/vhat
Ь
QAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/vhat/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/vhat*
_output_shapes	
:Ш*
dtype0
н
@Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*Q
shared_nameB@Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/vhat
ж
TAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/vhat/Read/ReadVariableOpReadVariableOp@Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/vhat*
_output_shapes
:	Ш*
dtype0
ё
JAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*[
shared_nameLJAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/vhat
ъ
^Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpJAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/vhat*
_output_shapes
:	2Ш*
dtype0
е
>Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*O
shared_name@>Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/vhat
Ю
RAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/vhat/Read/ReadVariableOpReadVariableOp>Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/vhat*
_output_shapes	
:Ш*
dtype0

NoOpNoOp
т@
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*@
value@B@ B@
П
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
А
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
­
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
­
1layer_metrics
2non_trainable_variables
regularization_losses
	variables
3metrics
trainable_variables
4layer_regularization_losses

5layers
[Y
VARIABLE_VALUEdense_39/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_39/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
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
qo
VARIABLE_VALUE5bidirectional_39/forward_lstm_39/lstm_cell_118/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE?bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3bidirectional_39/forward_lstm_39/lstm_cell_118/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6bidirectional_39/backward_lstm_39/lstm_cell_119/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE@bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE4bidirectional_39/backward_lstm_39/lstm_cell_119/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 

;0
 

0
1

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
Й
Alayer_metrics
Bnon_trainable_variables

Cstates
'regularization_losses
(	variables
Dmetrics
)trainable_variables
Elayer_regularization_losses

Flayers

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
Й
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
­
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
­
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
VARIABLE_VALUEAdam/dense_39/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_39/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE:Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
 
VARIABLE_VALUEGAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE;Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_39/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_39/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE:Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
 
VARIABLE_VALUEGAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE;Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_39/kernel/vhatUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/dense_39/bias/vhatSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/vhatEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ЅЂ
VARIABLE_VALUEIAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/vhatEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/vhatEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE@Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/vhatEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ІЃ
VARIABLE_VALUEJAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/vhatEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/vhatEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
y
serving_default_args_0Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
s
serving_default_args_0_1Placeholder*#
_output_shapes
:џџџџџџџџџ*
dtype0	*
shape:џџџџџџџџџ
к
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_0_15bidirectional_39/forward_lstm_39/lstm_cell_118/kernel?bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel3bidirectional_39/forward_lstm_39/lstm_cell_118/bias6bidirectional_39/backward_lstm_39/lstm_cell_119/kernel@bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel4bidirectional_39/backward_lstm_39/lstm_cell_119/biasdense_39/kerneldense_39/bias*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_4932330
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ж
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_39/kernel/Read/ReadVariableOp!dense_39/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpIbidirectional_39/forward_lstm_39/lstm_cell_118/kernel/Read/ReadVariableOpSbidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/Read/ReadVariableOpGbidirectional_39/forward_lstm_39/lstm_cell_118/bias/Read/ReadVariableOpJbidirectional_39/backward_lstm_39/lstm_cell_119/kernel/Read/ReadVariableOpTbidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/Read/ReadVariableOpHbidirectional_39/backward_lstm_39/lstm_cell_119/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_39/kernel/m/Read/ReadVariableOp(Adam/dense_39/bias/m/Read/ReadVariableOpPAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/m/Read/ReadVariableOpZAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/m/Read/ReadVariableOpNAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/m/Read/ReadVariableOpQAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/m/Read/ReadVariableOp[Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/m/Read/ReadVariableOpOAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/m/Read/ReadVariableOp*Adam/dense_39/kernel/v/Read/ReadVariableOp(Adam/dense_39/bias/v/Read/ReadVariableOpPAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/v/Read/ReadVariableOpZAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/v/Read/ReadVariableOpNAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/v/Read/ReadVariableOpQAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/v/Read/ReadVariableOp[Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/v/Read/ReadVariableOpOAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/v/Read/ReadVariableOp-Adam/dense_39/kernel/vhat/Read/ReadVariableOp+Adam/dense_39/bias/vhat/Read/ReadVariableOpSAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/vhat/Read/ReadVariableOp]Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/vhat/Read/ReadVariableOpQAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/vhat/Read/ReadVariableOpTAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/vhat/Read/ReadVariableOp^Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/vhat/Read/ReadVariableOpRAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/vhat/Read/ReadVariableOpConst*4
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_4935381
Х
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_39/kerneldense_39/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate5bidirectional_39/forward_lstm_39/lstm_cell_118/kernel?bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel3bidirectional_39/forward_lstm_39/lstm_cell_118/bias6bidirectional_39/backward_lstm_39/lstm_cell_119/kernel@bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel4bidirectional_39/backward_lstm_39/lstm_cell_119/biastotalcountAdam/dense_39/kernel/mAdam/dense_39/bias/m<Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/mFAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/m:Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/m=Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/mGAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/m;Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/mAdam/dense_39/kernel/vAdam/dense_39/bias/v<Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/vFAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/v:Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/v=Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/vGAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/v;Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/vAdam/dense_39/kernel/vhatAdam/dense_39/bias/vhat?Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/vhatIAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/vhat=Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/vhat@Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/vhatJAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/vhat>Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/vhat*3
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_4935508ЬЄ8
&
ё
while_body_4930207
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_119_4930231_0:	Ш0
while_lstm_cell_119_4930233_0:	2Ш,
while_lstm_cell_119_4930235_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_119_4930231:	Ш.
while_lstm_cell_119_4930233:	2Ш*
while_lstm_cell_119_4930235:	ШЂ+while/lstm_cell_119/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemы
+while/lstm_cell_119/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_119_4930231_0while_lstm_cell_119_4930233_0while_lstm_cell_119_4930235_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_119_layer_call_and_return_conditional_losses_49301272-
+while/lstm_cell_119/StatefulPartitionedCallј
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_119/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Ѕ
while/Identity_4Identity4while/lstm_cell_119/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѕ
while/Identity_5Identity4while/lstm_cell_119/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp,^while/lstm_cell_119/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_119_4930231while_lstm_cell_119_4930231_0"<
while_lstm_cell_119_4930233while_lstm_cell_119_4930233_0"<
while_lstm_cell_119_4930235while_lstm_cell_119_4930235_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2Z
+while/lstm_cell_119/StatefulPartitionedCall+while/lstm_cell_119/StatefulPartitionedCall: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Ђ

"forward_lstm_39_while_cond_4933443<
8forward_lstm_39_while_forward_lstm_39_while_loop_counterB
>forward_lstm_39_while_forward_lstm_39_while_maximum_iterations%
!forward_lstm_39_while_placeholder'
#forward_lstm_39_while_placeholder_1'
#forward_lstm_39_while_placeholder_2'
#forward_lstm_39_while_placeholder_3'
#forward_lstm_39_while_placeholder_4>
:forward_lstm_39_while_less_forward_lstm_39_strided_slice_1U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4933443___redundant_placeholder0U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4933443___redundant_placeholder1U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4933443___redundant_placeholder2U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4933443___redundant_placeholder3U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4933443___redundant_placeholder4"
forward_lstm_39_while_identity
Р
forward_lstm_39/while/LessLess!forward_lstm_39_while_placeholder:forward_lstm_39_while_less_forward_lstm_39_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_39/while/Less
forward_lstm_39/while/IdentityIdentityforward_lstm_39/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_39/while/Identity"I
forward_lstm_39_while_identity'forward_lstm_39/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: :::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
БЗ
 
#__inference__traced_restore_4935508
file_prefix2
 assignvariableop_dense_39_kernel:d.
 assignvariableop_1_dense_39_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: [
Hassignvariableop_7_bidirectional_39_forward_lstm_39_lstm_cell_118_kernel:	Шe
Rassignvariableop_8_bidirectional_39_forward_lstm_39_lstm_cell_118_recurrent_kernel:	2ШU
Fassignvariableop_9_bidirectional_39_forward_lstm_39_lstm_cell_118_bias:	Ш]
Jassignvariableop_10_bidirectional_39_backward_lstm_39_lstm_cell_119_kernel:	Шg
Tassignvariableop_11_bidirectional_39_backward_lstm_39_lstm_cell_119_recurrent_kernel:	2ШW
Hassignvariableop_12_bidirectional_39_backward_lstm_39_lstm_cell_119_bias:	Ш#
assignvariableop_13_total: #
assignvariableop_14_count: <
*assignvariableop_15_adam_dense_39_kernel_m:d6
(assignvariableop_16_adam_dense_39_bias_m:c
Passignvariableop_17_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_kernel_m:	Шm
Zassignvariableop_18_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_recurrent_kernel_m:	2Ш]
Nassignvariableop_19_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_bias_m:	Шd
Qassignvariableop_20_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_kernel_m:	Шn
[assignvariableop_21_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_recurrent_kernel_m:	2Ш^
Oassignvariableop_22_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_bias_m:	Ш<
*assignvariableop_23_adam_dense_39_kernel_v:d6
(assignvariableop_24_adam_dense_39_bias_v:c
Passignvariableop_25_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_kernel_v:	Шm
Zassignvariableop_26_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_recurrent_kernel_v:	2Ш]
Nassignvariableop_27_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_bias_v:	Шd
Qassignvariableop_28_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_kernel_v:	Шn
[assignvariableop_29_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_recurrent_kernel_v:	2Ш^
Oassignvariableop_30_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_bias_v:	Ш?
-assignvariableop_31_adam_dense_39_kernel_vhat:d9
+assignvariableop_32_adam_dense_39_bias_vhat:f
Sassignvariableop_33_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_kernel_vhat:	Шp
]assignvariableop_34_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_recurrent_kernel_vhat:	2Ш`
Qassignvariableop_35_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_bias_vhat:	Шg
Tassignvariableop_36_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_kernel_vhat:	Шq
^assignvariableop_37_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_recurrent_kernel_vhat:	2Шa
Rassignvariableop_38_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_bias_vhat:	Ш
identity_40ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ј
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*Д
valueЊBЇ(B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesо
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesі
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ж
_output_shapesЃ
 ::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_39_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ѕ
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_39_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2Ё
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѓ
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ѓ
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ђ
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Њ
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Э
AssignVariableOp_7AssignVariableOpHassignvariableop_7_bidirectional_39_forward_lstm_39_lstm_cell_118_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8з
AssignVariableOp_8AssignVariableOpRassignvariableop_8_bidirectional_39_forward_lstm_39_lstm_cell_118_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ы
AssignVariableOp_9AssignVariableOpFassignvariableop_9_bidirectional_39_forward_lstm_39_lstm_cell_118_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10в
AssignVariableOp_10AssignVariableOpJassignvariableop_10_bidirectional_39_backward_lstm_39_lstm_cell_119_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11м
AssignVariableOp_11AssignVariableOpTassignvariableop_11_bidirectional_39_backward_lstm_39_lstm_cell_119_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12а
AssignVariableOp_12AssignVariableOpHassignvariableop_12_bidirectional_39_backward_lstm_39_lstm_cell_119_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ё
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ё
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15В
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_39_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16А
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_39_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17и
AssignVariableOp_17AssignVariableOpPassignvariableop_17_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18т
AssignVariableOp_18AssignVariableOpZassignvariableop_18_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19ж
AssignVariableOp_19AssignVariableOpNassignvariableop_19_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20й
AssignVariableOp_20AssignVariableOpQassignvariableop_20_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21у
AssignVariableOp_21AssignVariableOp[assignvariableop_21_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22з
AssignVariableOp_22AssignVariableOpOassignvariableop_22_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23В
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_39_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24А
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_39_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25и
AssignVariableOp_25AssignVariableOpPassignvariableop_25_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26т
AssignVariableOp_26AssignVariableOpZassignvariableop_26_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27ж
AssignVariableOp_27AssignVariableOpNassignvariableop_27_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28й
AssignVariableOp_28AssignVariableOpQassignvariableop_28_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29у
AssignVariableOp_29AssignVariableOp[assignvariableop_29_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30з
AssignVariableOp_30AssignVariableOpOassignvariableop_30_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Е
AssignVariableOp_31AssignVariableOp-assignvariableop_31_adam_dense_39_kernel_vhatIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Г
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_dense_39_bias_vhatIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33л
AssignVariableOp_33AssignVariableOpSassignvariableop_33_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_kernel_vhatIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34х
AssignVariableOp_34AssignVariableOp]assignvariableop_34_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_recurrent_kernel_vhatIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35й
AssignVariableOp_35AssignVariableOpQassignvariableop_35_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_bias_vhatIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36м
AssignVariableOp_36AssignVariableOpTassignvariableop_36_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_kernel_vhatIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37ц
AssignVariableOp_37AssignVariableOp^assignvariableop_37_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_recurrent_kernel_vhatIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38к
AssignVariableOp_38AssignVariableOpRassignvariableop_38_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_bias_vhatIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpИ
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_39f
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_40 
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
ШF

L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_4929642

inputs(
lstm_cell_118_4929560:	Ш(
lstm_cell_118_4929562:	2Ш$
lstm_cell_118_4929564:	Ш
identityЂ%lstm_cell_118/StatefulPartitionedCallЂwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1
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
:џџџџџџџџџ22
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
B :ш2
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
zeros_1/packed/1
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
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
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
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Ї
%lstm_cell_118/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_118_4929560lstm_cell_118_4929562lstm_cell_118_4929564*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_118_layer_call_and_return_conditional_losses_49294952'
%lstm_cell_118/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterШ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_118_4929560lstm_cell_118_4929562lstm_cell_118_4929564*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_4929573*
condR
while_cond_4929572*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
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
:џџџџџџџџџ22

Identity~
NoOpNoOp&^lstm_cell_118/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2N
%lstm_cell_118/StatefulPartitionedCall%lstm_cell_118/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ђ

"forward_lstm_39_while_cond_4931433<
8forward_lstm_39_while_forward_lstm_39_while_loop_counterB
>forward_lstm_39_while_forward_lstm_39_while_maximum_iterations%
!forward_lstm_39_while_placeholder'
#forward_lstm_39_while_placeholder_1'
#forward_lstm_39_while_placeholder_2'
#forward_lstm_39_while_placeholder_3'
#forward_lstm_39_while_placeholder_4>
:forward_lstm_39_while_less_forward_lstm_39_strided_slice_1U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4931433___redundant_placeholder0U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4931433___redundant_placeholder1U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4931433___redundant_placeholder2U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4931433___redundant_placeholder3U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4931433___redundant_placeholder4"
forward_lstm_39_while_identity
Р
forward_lstm_39/while/LessLess!forward_lstm_39_while_placeholder:forward_lstm_39_while_less_forward_lstm_39_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_39/while/Less
forward_lstm_39/while/IdentityIdentityforward_lstm_39/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_39/while/Identity"I
forward_lstm_39_while_identity'forward_lstm_39/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: :::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
ј?
к
while_body_4934304
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_118_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_118_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_118_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_118_matmul_readvariableop_resource:	ШG
4while_lstm_cell_118_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_118_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_118/BiasAdd/ReadVariableOpЂ)while/lstm_cell_118/MatMul/ReadVariableOpЂ+while/lstm_cell_118/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeм
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЬ
)while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_118/MatMul/ReadVariableOpк
while/lstm_cell_118/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/MatMulв
+while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_118/MatMul_1/ReadVariableOpУ
while/lstm_cell_118/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/MatMul_1М
while/lstm_cell_118/addAddV2$while/lstm_cell_118/MatMul:product:0&while/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/addЫ
*while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_118/BiasAdd/ReadVariableOpЩ
while/lstm_cell_118/BiasAddBiasAddwhile/lstm_cell_118/add:z:02while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/BiasAdd
#while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_118/split/split_dim
while/lstm_cell_118/splitSplit,while/lstm_cell_118/split/split_dim:output:0$while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_118/split
while/lstm_cell_118/SigmoidSigmoid"while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Sigmoid
while/lstm_cell_118/Sigmoid_1Sigmoid"while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Sigmoid_1Ѓ
while/lstm_cell_118/mulMul!while/lstm_cell_118/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/mul
while/lstm_cell_118/ReluRelu"while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/ReluИ
while/lstm_cell_118/mul_1Mulwhile/lstm_cell_118/Sigmoid:y:0&while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/mul_1­
while/lstm_cell_118/add_1AddV2while/lstm_cell_118/mul:z:0while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/add_1
while/lstm_cell_118/Sigmoid_2Sigmoid"while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Sigmoid_2
while/lstm_cell_118/Relu_1Reluwhile/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Relu_1М
while/lstm_cell_118/mul_2Mul!while/lstm_cell_118/Sigmoid_2:y:0(while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_118/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_118/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_118/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_118/BiasAdd/ReadVariableOp*^while/lstm_cell_118/MatMul/ReadVariableOp,^while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_118_biasadd_readvariableop_resource5while_lstm_cell_118_biasadd_readvariableop_resource_0"n
4while_lstm_cell_118_matmul_1_readvariableop_resource6while_lstm_cell_118_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_118_matmul_readvariableop_resource4while_lstm_cell_118_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_118/BiasAdd/ReadVariableOp*while/lstm_cell_118/BiasAdd/ReadVariableOp2V
)while/lstm_cell_118/MatMul/ReadVariableOp)while/lstm_cell_118/MatMul/ReadVariableOp2Z
+while/lstm_cell_118/MatMul_1/ReadVariableOp+while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
єd
Н
#backward_lstm_39_while_body_4933623>
:backward_lstm_39_while_backward_lstm_39_while_loop_counterD
@backward_lstm_39_while_backward_lstm_39_while_maximum_iterations&
"backward_lstm_39_while_placeholder(
$backward_lstm_39_while_placeholder_1(
$backward_lstm_39_while_placeholder_2(
$backward_lstm_39_while_placeholder_3(
$backward_lstm_39_while_placeholder_4=
9backward_lstm_39_while_backward_lstm_39_strided_slice_1_0y
ubackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_08
4backward_lstm_39_while_less_backward_lstm_39_sub_1_0X
Ebackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0:	ШZ
Gbackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0:	2ШU
Fbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0:	Ш#
backward_lstm_39_while_identity%
!backward_lstm_39_while_identity_1%
!backward_lstm_39_while_identity_2%
!backward_lstm_39_while_identity_3%
!backward_lstm_39_while_identity_4%
!backward_lstm_39_while_identity_5%
!backward_lstm_39_while_identity_6;
7backward_lstm_39_while_backward_lstm_39_strided_slice_1w
sbackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor6
2backward_lstm_39_while_less_backward_lstm_39_sub_1V
Cbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource:	ШX
Ebackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource:	2ШS
Dbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource:	ШЂ;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOpЂ:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOpЂ<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOpх
Hbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2J
Hbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
:backward_lstm_39/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_39_while_placeholderQbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02<
:backward_lstm_39/while/TensorArrayV2Read/TensorListGetItemЪ
backward_lstm_39/while/LessLess4backward_lstm_39_while_less_backward_lstm_39_sub_1_0"backward_lstm_39_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_39/while/Lessџ
:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOpReadVariableOpEbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp
+backward_lstm_39/while/lstm_cell_119/MatMulMatMulAbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem:item:0Bbackward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+backward_lstm_39/while/lstm_cell_119/MatMul
<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOpGbackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp
-backward_lstm_39/while/lstm_cell_119/MatMul_1MatMul$backward_lstm_39_while_placeholder_3Dbackward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_39/while/lstm_cell_119/MatMul_1
(backward_lstm_39/while/lstm_cell_119/addAddV25backward_lstm_39/while/lstm_cell_119/MatMul:product:07backward_lstm_39/while/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_39/while/lstm_cell_119/addў
;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOpFbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp
,backward_lstm_39/while/lstm_cell_119/BiasAddBiasAdd,backward_lstm_39/while/lstm_cell_119/add:z:0Cbackward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_39/while/lstm_cell_119/BiasAddЎ
4backward_lstm_39/while/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4backward_lstm_39/while/lstm_cell_119/split/split_dimг
*backward_lstm_39/while/lstm_cell_119/splitSplit=backward_lstm_39/while/lstm_cell_119/split/split_dim:output:05backward_lstm_39/while/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*backward_lstm_39/while/lstm_cell_119/splitЮ
,backward_lstm_39/while/lstm_cell_119/SigmoidSigmoid3backward_lstm_39/while/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_39/while/lstm_cell_119/Sigmoidв
.backward_lstm_39/while/lstm_cell_119/Sigmoid_1Sigmoid3backward_lstm_39/while/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.backward_lstm_39/while/lstm_cell_119/Sigmoid_1ч
(backward_lstm_39/while/lstm_cell_119/mulMul2backward_lstm_39/while/lstm_cell_119/Sigmoid_1:y:0$backward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_39/while/lstm_cell_119/mulХ
)backward_lstm_39/while/lstm_cell_119/ReluRelu3backward_lstm_39/while/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_39/while/lstm_cell_119/Reluќ
*backward_lstm_39/while/lstm_cell_119/mul_1Mul0backward_lstm_39/while/lstm_cell_119/Sigmoid:y:07backward_lstm_39/while/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_39/while/lstm_cell_119/mul_1ё
*backward_lstm_39/while/lstm_cell_119/add_1AddV2,backward_lstm_39/while/lstm_cell_119/mul:z:0.backward_lstm_39/while/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_39/while/lstm_cell_119/add_1в
.backward_lstm_39/while/lstm_cell_119/Sigmoid_2Sigmoid3backward_lstm_39/while/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.backward_lstm_39/while/lstm_cell_119/Sigmoid_2Ф
+backward_lstm_39/while/lstm_cell_119/Relu_1Relu.backward_lstm_39/while/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_39/while/lstm_cell_119/Relu_1
*backward_lstm_39/while/lstm_cell_119/mul_2Mul2backward_lstm_39/while/lstm_cell_119/Sigmoid_2:y:09backward_lstm_39/while/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_39/while/lstm_cell_119/mul_2ё
backward_lstm_39/while/SelectSelectbackward_lstm_39/while/Less:z:0.backward_lstm_39/while/lstm_cell_119/mul_2:z:0$backward_lstm_39_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/while/Selectѕ
backward_lstm_39/while/Select_1Selectbackward_lstm_39/while/Less:z:0.backward_lstm_39/while/lstm_cell_119/mul_2:z:0$backward_lstm_39_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22!
backward_lstm_39/while/Select_1ѕ
backward_lstm_39/while/Select_2Selectbackward_lstm_39/while/Less:z:0.backward_lstm_39/while/lstm_cell_119/add_1:z:0$backward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22!
backward_lstm_39/while/Select_2Ў
;backward_lstm_39/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_39_while_placeholder_1"backward_lstm_39_while_placeholder&backward_lstm_39/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_39/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_39/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_39/while/add/y­
backward_lstm_39/while/addAddV2"backward_lstm_39_while_placeholder%backward_lstm_39/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/while/add
backward_lstm_39/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_39/while/add_1/yЫ
backward_lstm_39/while/add_1AddV2:backward_lstm_39_while_backward_lstm_39_while_loop_counter'backward_lstm_39/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/while/add_1Џ
backward_lstm_39/while/IdentityIdentity backward_lstm_39/while/add_1:z:0^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_39/while/Identityг
!backward_lstm_39/while/Identity_1Identity@backward_lstm_39_while_backward_lstm_39_while_maximum_iterations^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_39/while/Identity_1Б
!backward_lstm_39/while/Identity_2Identitybackward_lstm_39/while/add:z:0^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_39/while/Identity_2о
!backward_lstm_39/while/Identity_3IdentityKbackward_lstm_39/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_39/while/Identity_3Ъ
!backward_lstm_39/while/Identity_4Identity&backward_lstm_39/while/Select:output:0^backward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_39/while/Identity_4Ь
!backward_lstm_39/while/Identity_5Identity(backward_lstm_39/while/Select_1:output:0^backward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_39/while/Identity_5Ь
!backward_lstm_39/while/Identity_6Identity(backward_lstm_39/while/Select_2:output:0^backward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_39/while/Identity_6Ж
backward_lstm_39/while/NoOpNoOp<^backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp;^backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp=^backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_39/while/NoOp"t
7backward_lstm_39_while_backward_lstm_39_strided_slice_19backward_lstm_39_while_backward_lstm_39_strided_slice_1_0"K
backward_lstm_39_while_identity(backward_lstm_39/while/Identity:output:0"O
!backward_lstm_39_while_identity_1*backward_lstm_39/while/Identity_1:output:0"O
!backward_lstm_39_while_identity_2*backward_lstm_39/while/Identity_2:output:0"O
!backward_lstm_39_while_identity_3*backward_lstm_39/while/Identity_3:output:0"O
!backward_lstm_39_while_identity_4*backward_lstm_39/while/Identity_4:output:0"O
!backward_lstm_39_while_identity_5*backward_lstm_39/while/Identity_5:output:0"O
!backward_lstm_39_while_identity_6*backward_lstm_39/while/Identity_6:output:0"j
2backward_lstm_39_while_less_backward_lstm_39_sub_14backward_lstm_39_while_less_backward_lstm_39_sub_1_0"
Dbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resourceFbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0"
Ebackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resourceGbackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0"
Cbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resourceEbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0"ь
sbackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensorubackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2z
;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp2x
:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp2|
<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:џџџџџџџџџ


J__inference_sequential_39_layer_call_and_return_conditional_losses_4932300

inputs
inputs_1	+
bidirectional_39_4932281:	Ш+
bidirectional_39_4932283:	2Ш'
bidirectional_39_4932285:	Ш+
bidirectional_39_4932287:	Ш+
bidirectional_39_4932289:	2Ш'
bidirectional_39_4932291:	Ш"
dense_39_4932294:d
dense_39_4932296:
identityЂ(bidirectional_39/StatefulPartitionedCallЂ dense_39/StatefulPartitionedCallК
(bidirectional_39/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_39_4932281bidirectional_39_4932283bidirectional_39_4932285bidirectional_39_4932287bidirectional_39_4932289bidirectional_39_4932291*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_49321502*
(bidirectional_39/StatefulPartitionedCallТ
 dense_39/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_39/StatefulPartitionedCall:output:0dense_39_4932294dense_39_4932296*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_39_layer_call_and_return_conditional_losses_49317352"
 dense_39/StatefulPartitionedCall
IdentityIdentity)dense_39/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp)^bidirectional_39/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2T
(bidirectional_39/StatefulPartitionedCall(bidirectional_39/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
к
Ш
while_cond_4930774
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_4930774___redundant_placeholder05
1while_while_cond_4930774___redundant_placeholder15
1while_while_cond_4930774___redundant_placeholder25
1while_while_cond_4930774___redundant_placeholder3
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
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Ђ

"forward_lstm_39_while_cond_4931873<
8forward_lstm_39_while_forward_lstm_39_while_loop_counterB
>forward_lstm_39_while_forward_lstm_39_while_maximum_iterations%
!forward_lstm_39_while_placeholder'
#forward_lstm_39_while_placeholder_1'
#forward_lstm_39_while_placeholder_2'
#forward_lstm_39_while_placeholder_3'
#forward_lstm_39_while_placeholder_4>
:forward_lstm_39_while_less_forward_lstm_39_strided_slice_1U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4931873___redundant_placeholder0U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4931873___redundant_placeholder1U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4931873___redundant_placeholder2U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4931873___redundant_placeholder3U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4931873___redundant_placeholder4"
forward_lstm_39_while_identity
Р
forward_lstm_39/while/LessLess!forward_lstm_39_while_placeholder:forward_lstm_39_while_less_forward_lstm_39_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_39/while/Less
forward_lstm_39/while/IdentityIdentityforward_lstm_39/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_39/while/Identity"I
forward_lstm_39_while_identity'forward_lstm_39/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: :::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
Ђ_
Ћ
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_4930859

inputs?
,lstm_cell_119_matmul_readvariableop_resource:	ШA
.lstm_cell_119_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_119_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_119/BiasAdd/ReadVariableOpЂ#lstm_cell_119/MatMul/ReadVariableOpЂ%lstm_cell_119/MatMul_1/ReadVariableOpЂwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1
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
:џџџџџџџџџ22
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
B :ш2
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
zeros_1/packed/1
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
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
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
ReverseV2/axis
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
	ReverseV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ27
5TensorArrayUnstack/TensorListFromTensor/element_shape§
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
strided_slice_2/stack_2
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2
strided_slice_2И
#lstm_cell_119/MatMul/ReadVariableOpReadVariableOp,lstm_cell_119_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_119/MatMul/ReadVariableOpА
lstm_cell_119/MatMulMatMulstrided_slice_2:output:0+lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/MatMulО
%lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_119_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_119/MatMul_1/ReadVariableOpЌ
lstm_cell_119/MatMul_1MatMulzeros:output:0-lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/MatMul_1Є
lstm_cell_119/addAddV2lstm_cell_119/MatMul:product:0 lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/addЗ
$lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_119_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_119/BiasAdd/ReadVariableOpБ
lstm_cell_119/BiasAddBiasAddlstm_cell_119/add:z:0,lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/BiasAdd
lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_119/split/split_dimї
lstm_cell_119/splitSplit&lstm_cell_119/split/split_dim:output:0lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_119/split
lstm_cell_119/SigmoidSigmoidlstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Sigmoid
lstm_cell_119/Sigmoid_1Sigmoidlstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Sigmoid_1
lstm_cell_119/mulMullstm_cell_119/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/mul
lstm_cell_119/ReluRelulstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Relu 
lstm_cell_119/mul_1Mullstm_cell_119/Sigmoid:y:0 lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/mul_1
lstm_cell_119/add_1AddV2lstm_cell_119/mul:z:0lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/add_1
lstm_cell_119/Sigmoid_2Sigmoidlstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Sigmoid_2
lstm_cell_119/Relu_1Relulstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Relu_1Є
lstm_cell_119/mul_2Mullstm_cell_119/Sigmoid_2:y:0"lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_119_matmul_readvariableop_resource.lstm_cell_119_matmul_1_readvariableop_resource-lstm_cell_119_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_4930775*
condR
while_cond_4930774*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
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
:џџџџџџџџџ22

IdentityЫ
NoOpNoOp%^lstm_cell_119/BiasAdd/ReadVariableOp$^lstm_cell_119/MatMul/ReadVariableOp&^lstm_cell_119/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_119/BiasAdd/ReadVariableOp$lstm_cell_119/BiasAdd/ReadVariableOp2J
#lstm_cell_119/MatMul/ReadVariableOp#lstm_cell_119/MatMul/ReadVariableOp2N
%lstm_cell_119/MatMul_1/ReadVariableOp%lstm_cell_119/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ј?
к
while_body_4934807
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_119_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_119_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_119_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_119_matmul_readvariableop_resource:	ШG
4while_lstm_cell_119_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_119_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_119/BiasAdd/ReadVariableOpЂ)while/lstm_cell_119/MatMul/ReadVariableOpЂ+while/lstm_cell_119/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeм
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЬ
)while/lstm_cell_119/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_119_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_119/MatMul/ReadVariableOpк
while/lstm_cell_119/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/MatMulв
+while/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_119_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_119/MatMul_1/ReadVariableOpУ
while/lstm_cell_119/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/MatMul_1М
while/lstm_cell_119/addAddV2$while/lstm_cell_119/MatMul:product:0&while/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/addЫ
*while/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_119_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_119/BiasAdd/ReadVariableOpЩ
while/lstm_cell_119/BiasAddBiasAddwhile/lstm_cell_119/add:z:02while/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/BiasAdd
#while/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_119/split/split_dim
while/lstm_cell_119/splitSplit,while/lstm_cell_119/split/split_dim:output:0$while/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_119/split
while/lstm_cell_119/SigmoidSigmoid"while/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Sigmoid
while/lstm_cell_119/Sigmoid_1Sigmoid"while/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Sigmoid_1Ѓ
while/lstm_cell_119/mulMul!while/lstm_cell_119/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/mul
while/lstm_cell_119/ReluRelu"while/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/ReluИ
while/lstm_cell_119/mul_1Mulwhile/lstm_cell_119/Sigmoid:y:0&while/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/mul_1­
while/lstm_cell_119/add_1AddV2while/lstm_cell_119/mul:z:0while/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/add_1
while/lstm_cell_119/Sigmoid_2Sigmoid"while/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Sigmoid_2
while/lstm_cell_119/Relu_1Reluwhile/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Relu_1М
while/lstm_cell_119/mul_2Mul!while/lstm_cell_119/Sigmoid_2:y:0(while/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_119/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_119/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_119/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_119/BiasAdd/ReadVariableOp*^while/lstm_cell_119/MatMul/ReadVariableOp,^while/lstm_cell_119/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_119_biasadd_readvariableop_resource5while_lstm_cell_119_biasadd_readvariableop_resource_0"n
4while_lstm_cell_119_matmul_1_readvariableop_resource6while_lstm_cell_119_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_119_matmul_readvariableop_resource4while_lstm_cell_119_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_119/BiasAdd/ReadVariableOp*while/lstm_cell_119/BiasAdd/ReadVariableOp2V
)while/lstm_cell_119/MatMul/ReadVariableOp)while/lstm_cell_119/MatMul/ReadVariableOp2Z
+while/lstm_cell_119/MatMul_1/ReadVariableOp+while/lstm_cell_119/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Ђ_
Ћ
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_4934891

inputs?
,lstm_cell_119_matmul_readvariableop_resource:	ШA
.lstm_cell_119_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_119_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_119/BiasAdd/ReadVariableOpЂ#lstm_cell_119/MatMul/ReadVariableOpЂ%lstm_cell_119/MatMul_1/ReadVariableOpЂwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1
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
:џџџџџџџџџ22
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
B :ш2
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
zeros_1/packed/1
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
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
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
ReverseV2/axis
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
	ReverseV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ27
5TensorArrayUnstack/TensorListFromTensor/element_shape§
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
strided_slice_2/stack_2
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2
strided_slice_2И
#lstm_cell_119/MatMul/ReadVariableOpReadVariableOp,lstm_cell_119_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_119/MatMul/ReadVariableOpА
lstm_cell_119/MatMulMatMulstrided_slice_2:output:0+lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/MatMulО
%lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_119_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_119/MatMul_1/ReadVariableOpЌ
lstm_cell_119/MatMul_1MatMulzeros:output:0-lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/MatMul_1Є
lstm_cell_119/addAddV2lstm_cell_119/MatMul:product:0 lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/addЗ
$lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_119_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_119/BiasAdd/ReadVariableOpБ
lstm_cell_119/BiasAddBiasAddlstm_cell_119/add:z:0,lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/BiasAdd
lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_119/split/split_dimї
lstm_cell_119/splitSplit&lstm_cell_119/split/split_dim:output:0lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_119/split
lstm_cell_119/SigmoidSigmoidlstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Sigmoid
lstm_cell_119/Sigmoid_1Sigmoidlstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Sigmoid_1
lstm_cell_119/mulMullstm_cell_119/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/mul
lstm_cell_119/ReluRelulstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Relu 
lstm_cell_119/mul_1Mullstm_cell_119/Sigmoid:y:0 lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/mul_1
lstm_cell_119/add_1AddV2lstm_cell_119/mul:z:0lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/add_1
lstm_cell_119/Sigmoid_2Sigmoidlstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Sigmoid_2
lstm_cell_119/Relu_1Relulstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Relu_1Є
lstm_cell_119/mul_2Mullstm_cell_119/Sigmoid_2:y:0"lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_119_matmul_readvariableop_resource.lstm_cell_119_matmul_1_readvariableop_resource-lstm_cell_119_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_4934807*
condR
while_cond_4934806*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
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
:џџџџџџџџџ22

IdentityЫ
NoOpNoOp%^lstm_cell_119/BiasAdd/ReadVariableOp$^lstm_cell_119/MatMul/ReadVariableOp&^lstm_cell_119/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_119/BiasAdd/ReadVariableOp$lstm_cell_119/BiasAdd/ReadVariableOp2J
#lstm_cell_119/MatMul/ReadVariableOp#lstm_cell_119/MatMul/ReadVariableOp2N
%lstm_cell_119/MatMul_1/ReadVariableOp%lstm_cell_119/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ђ_
Ћ
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_4935044

inputs?
,lstm_cell_119_matmul_readvariableop_resource:	ШA
.lstm_cell_119_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_119_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_119/BiasAdd/ReadVariableOpЂ#lstm_cell_119/MatMul/ReadVariableOpЂ%lstm_cell_119/MatMul_1/ReadVariableOpЂwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1
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
:џџџџџџџџџ22
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
B :ш2
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
zeros_1/packed/1
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
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
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
ReverseV2/axis
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
	ReverseV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ27
5TensorArrayUnstack/TensorListFromTensor/element_shape§
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
strided_slice_2/stack_2
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2
strided_slice_2И
#lstm_cell_119/MatMul/ReadVariableOpReadVariableOp,lstm_cell_119_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_119/MatMul/ReadVariableOpА
lstm_cell_119/MatMulMatMulstrided_slice_2:output:0+lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/MatMulО
%lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_119_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_119/MatMul_1/ReadVariableOpЌ
lstm_cell_119/MatMul_1MatMulzeros:output:0-lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/MatMul_1Є
lstm_cell_119/addAddV2lstm_cell_119/MatMul:product:0 lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/addЗ
$lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_119_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_119/BiasAdd/ReadVariableOpБ
lstm_cell_119/BiasAddBiasAddlstm_cell_119/add:z:0,lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/BiasAdd
lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_119/split/split_dimї
lstm_cell_119/splitSplit&lstm_cell_119/split/split_dim:output:0lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_119/split
lstm_cell_119/SigmoidSigmoidlstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Sigmoid
lstm_cell_119/Sigmoid_1Sigmoidlstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Sigmoid_1
lstm_cell_119/mulMullstm_cell_119/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/mul
lstm_cell_119/ReluRelulstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Relu 
lstm_cell_119/mul_1Mullstm_cell_119/Sigmoid:y:0 lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/mul_1
lstm_cell_119/add_1AddV2lstm_cell_119/mul:z:0lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/add_1
lstm_cell_119/Sigmoid_2Sigmoidlstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Sigmoid_2
lstm_cell_119/Relu_1Relulstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Relu_1Є
lstm_cell_119/mul_2Mullstm_cell_119/Sigmoid_2:y:0"lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_119_matmul_readvariableop_resource.lstm_cell_119_matmul_1_readvariableop_resource-lstm_cell_119_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_4934960*
condR
while_cond_4934959*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
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
:џџџџџџџџџ22

IdentityЫ
NoOpNoOp%^lstm_cell_119/BiasAdd/ReadVariableOp$^lstm_cell_119/MatMul/ReadVariableOp&^lstm_cell_119/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_119/BiasAdd/ReadVariableOp$lstm_cell_119/BiasAdd/ReadVariableOp2J
#lstm_cell_119/MatMul/ReadVariableOp#lstm_cell_119/MatMul/ReadVariableOp2N
%lstm_cell_119/MatMul_1/ReadVariableOp%lstm_cell_119/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
а
Р
1__inference_forward_lstm_39_layer_call_fn_4933762
inputs_0
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_49296422
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
У

#backward_lstm_39_while_cond_4932052>
:backward_lstm_39_while_backward_lstm_39_while_loop_counterD
@backward_lstm_39_while_backward_lstm_39_while_maximum_iterations&
"backward_lstm_39_while_placeholder(
$backward_lstm_39_while_placeholder_1(
$backward_lstm_39_while_placeholder_2(
$backward_lstm_39_while_placeholder_3(
$backward_lstm_39_while_placeholder_4@
<backward_lstm_39_while_less_backward_lstm_39_strided_slice_1W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4932052___redundant_placeholder0W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4932052___redundant_placeholder1W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4932052___redundant_placeholder2W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4932052___redundant_placeholder3W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4932052___redundant_placeholder4#
backward_lstm_39_while_identity
Х
backward_lstm_39/while/LessLess"backward_lstm_39_while_placeholder<backward_lstm_39_while_less_backward_lstm_39_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_39/while/Less
backward_lstm_39/while/IdentityIdentitybackward_lstm_39/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_39/while/Identity"K
backward_lstm_39_while_identity(backward_lstm_39/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: :::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
]
Њ
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_4934388

inputs?
,lstm_cell_118_matmul_readvariableop_resource:	ШA
.lstm_cell_118_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_118_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_118/BiasAdd/ReadVariableOpЂ#lstm_cell_118/MatMul/ReadVariableOpЂ%lstm_cell_118/MatMul_1/ReadVariableOpЂwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1
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
:џџџџџџџџџ22
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
B :ш2
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
zeros_1/packed/1
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
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
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
strided_slice_2/stack_2
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2
strided_slice_2И
#lstm_cell_118/MatMul/ReadVariableOpReadVariableOp,lstm_cell_118_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_118/MatMul/ReadVariableOpА
lstm_cell_118/MatMulMatMulstrided_slice_2:output:0+lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/MatMulО
%lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_118/MatMul_1/ReadVariableOpЌ
lstm_cell_118/MatMul_1MatMulzeros:output:0-lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/MatMul_1Є
lstm_cell_118/addAddV2lstm_cell_118/MatMul:product:0 lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/addЗ
$lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_118/BiasAdd/ReadVariableOpБ
lstm_cell_118/BiasAddBiasAddlstm_cell_118/add:z:0,lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/BiasAdd
lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_118/split/split_dimї
lstm_cell_118/splitSplit&lstm_cell_118/split/split_dim:output:0lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_118/split
lstm_cell_118/SigmoidSigmoidlstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Sigmoid
lstm_cell_118/Sigmoid_1Sigmoidlstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Sigmoid_1
lstm_cell_118/mulMullstm_cell_118/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/mul
lstm_cell_118/ReluRelulstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Relu 
lstm_cell_118/mul_1Mullstm_cell_118/Sigmoid:y:0 lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/mul_1
lstm_cell_118/add_1AddV2lstm_cell_118/mul:z:0lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/add_1
lstm_cell_118/Sigmoid_2Sigmoidlstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Sigmoid_2
lstm_cell_118/Relu_1Relulstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Relu_1Є
lstm_cell_118/mul_2Mullstm_cell_118/Sigmoid_2:y:0"lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_118_matmul_readvariableop_resource.lstm_cell_118_matmul_1_readvariableop_resource-lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_4934304*
condR
while_cond_4934303*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
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
:џџџџџџџџџ22

IdentityЫ
NoOpNoOp%^lstm_cell_118/BiasAdd/ReadVariableOp$^lstm_cell_118/MatMul/ReadVariableOp&^lstm_cell_118/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_118/BiasAdd/ReadVariableOp$lstm_cell_118/BiasAdd/ReadVariableOp2J
#lstm_cell_118/MatMul/ReadVariableOp#lstm_cell_118/MatMul/ReadVariableOp2N
%lstm_cell_118/MatMul_1/ReadVariableOp%lstm_cell_118/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ШF

L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_4929432

inputs(
lstm_cell_118_4929350:	Ш(
lstm_cell_118_4929352:	2Ш$
lstm_cell_118_4929354:	Ш
identityЂ%lstm_cell_118/StatefulPartitionedCallЂwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1
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
:џџџџџџџџџ22
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
B :ш2
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
zeros_1/packed/1
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
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
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
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Ї
%lstm_cell_118/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_118_4929350lstm_cell_118_4929352lstm_cell_118_4929354*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_118_layer_call_and_return_conditional_losses_49293492'
%lstm_cell_118/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterШ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_118_4929350lstm_cell_118_4929352lstm_cell_118_4929354*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_4929363*
condR
while_cond_4929362*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
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
:џџџџџџџџџ22

Identity~
NoOpNoOp&^lstm_cell_118/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2N
%lstm_cell_118/StatefulPartitionedCall%lstm_cell_118/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ѕ

J__inference_lstm_cell_118_layer_call_and_return_conditional_losses_4929495

inputs

states
states_11
matmul_readvariableop_resource:	Ш3
 matmul_1_readvariableop_resource:	2Ш.
biasadd_readvariableop_resource:	Ш
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2
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
?:џџџџџџџџџ:џџџџџџџџџ2:џџџџџџџџџ2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates
§^
­
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_4934738
inputs_0?
,lstm_cell_119_matmul_readvariableop_resource:	ШA
.lstm_cell_119_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_119_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_119/BiasAdd/ReadVariableOpЂ#lstm_cell_119/MatMul/ReadVariableOpЂ%lstm_cell_119/MatMul_1/ReadVariableOpЂwhileF
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1
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
:џџџџџџџџџ22
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
B :ш2
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
zeros_1/packed/1
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
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
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
ReverseV2/axis
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	ReverseV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shape§
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
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2И
#lstm_cell_119/MatMul/ReadVariableOpReadVariableOp,lstm_cell_119_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_119/MatMul/ReadVariableOpА
lstm_cell_119/MatMulMatMulstrided_slice_2:output:0+lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/MatMulО
%lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_119_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_119/MatMul_1/ReadVariableOpЌ
lstm_cell_119/MatMul_1MatMulzeros:output:0-lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/MatMul_1Є
lstm_cell_119/addAddV2lstm_cell_119/MatMul:product:0 lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/addЗ
$lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_119_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_119/BiasAdd/ReadVariableOpБ
lstm_cell_119/BiasAddBiasAddlstm_cell_119/add:z:0,lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/BiasAdd
lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_119/split/split_dimї
lstm_cell_119/splitSplit&lstm_cell_119/split/split_dim:output:0lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_119/split
lstm_cell_119/SigmoidSigmoidlstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Sigmoid
lstm_cell_119/Sigmoid_1Sigmoidlstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Sigmoid_1
lstm_cell_119/mulMullstm_cell_119/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/mul
lstm_cell_119/ReluRelulstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Relu 
lstm_cell_119/mul_1Mullstm_cell_119/Sigmoid:y:0 lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/mul_1
lstm_cell_119/add_1AddV2lstm_cell_119/mul:z:0lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/add_1
lstm_cell_119/Sigmoid_2Sigmoidlstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Sigmoid_2
lstm_cell_119/Relu_1Relulstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Relu_1Є
lstm_cell_119/mul_2Mullstm_cell_119/Sigmoid_2:y:0"lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_119_matmul_readvariableop_resource.lstm_cell_119_matmul_1_readvariableop_resource-lstm_cell_119_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_4934654*
condR
while_cond_4934653*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
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
:џџџџџџџџџ22

IdentityЫ
NoOpNoOp%^lstm_cell_119/BiasAdd/ReadVariableOp$^lstm_cell_119/MatMul/ReadVariableOp&^lstm_cell_119/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_119/BiasAdd/ReadVariableOp$lstm_cell_119/BiasAdd/ReadVariableOp2J
#lstm_cell_119/MatMul/ReadVariableOp#lstm_cell_119/MatMul/ReadVariableOp2N
%lstm_cell_119/MatMul_1/ReadVariableOp%lstm_cell_119/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
­
м
Asequential_39_bidirectional_39_forward_lstm_39_while_cond_4928990z
vsequential_39_bidirectional_39_forward_lstm_39_while_sequential_39_bidirectional_39_forward_lstm_39_while_loop_counter
|sequential_39_bidirectional_39_forward_lstm_39_while_sequential_39_bidirectional_39_forward_lstm_39_while_maximum_iterationsD
@sequential_39_bidirectional_39_forward_lstm_39_while_placeholderF
Bsequential_39_bidirectional_39_forward_lstm_39_while_placeholder_1F
Bsequential_39_bidirectional_39_forward_lstm_39_while_placeholder_2F
Bsequential_39_bidirectional_39_forward_lstm_39_while_placeholder_3F
Bsequential_39_bidirectional_39_forward_lstm_39_while_placeholder_4|
xsequential_39_bidirectional_39_forward_lstm_39_while_less_sequential_39_bidirectional_39_forward_lstm_39_strided_slice_1
sequential_39_bidirectional_39_forward_lstm_39_while_sequential_39_bidirectional_39_forward_lstm_39_while_cond_4928990___redundant_placeholder0
sequential_39_bidirectional_39_forward_lstm_39_while_sequential_39_bidirectional_39_forward_lstm_39_while_cond_4928990___redundant_placeholder1
sequential_39_bidirectional_39_forward_lstm_39_while_sequential_39_bidirectional_39_forward_lstm_39_while_cond_4928990___redundant_placeholder2
sequential_39_bidirectional_39_forward_lstm_39_while_sequential_39_bidirectional_39_forward_lstm_39_while_cond_4928990___redundant_placeholder3
sequential_39_bidirectional_39_forward_lstm_39_while_sequential_39_bidirectional_39_forward_lstm_39_while_cond_4928990___redundant_placeholder4A
=sequential_39_bidirectional_39_forward_lstm_39_while_identity
л
9sequential_39/bidirectional_39/forward_lstm_39/while/LessLess@sequential_39_bidirectional_39_forward_lstm_39_while_placeholderxsequential_39_bidirectional_39_forward_lstm_39_while_less_sequential_39_bidirectional_39_forward_lstm_39_strided_slice_1*
T0*
_output_shapes
: 2;
9sequential_39/bidirectional_39/forward_lstm_39/while/Lessъ
=sequential_39/bidirectional_39/forward_lstm_39/while/IdentityIdentity=sequential_39/bidirectional_39/forward_lstm_39/while/Less:z:0*
T0
*
_output_shapes
: 2?
=sequential_39/bidirectional_39/forward_lstm_39/while/Identity"
=sequential_39_bidirectional_39_forward_lstm_39_while_identityFsequential_39/bidirectional_39/forward_lstm_39/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: :::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
к
Ш
while_cond_4930206
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_4930206___redundant_placeholder05
1while_while_cond_4930206___redundant_placeholder15
1while_while_cond_4930206___redundant_placeholder25
1while_while_cond_4930206___redundant_placeholder3
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
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
я?
к
while_body_4934654
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_119_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_119_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_119_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_119_matmul_readvariableop_resource:	ШG
4while_lstm_cell_119_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_119_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_119/BiasAdd/ReadVariableOpЂ)while/lstm_cell_119/MatMul/ReadVariableOpЂ+while/lstm_cell_119/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЬ
)while/lstm_cell_119/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_119_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_119/MatMul/ReadVariableOpк
while/lstm_cell_119/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/MatMulв
+while/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_119_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_119/MatMul_1/ReadVariableOpУ
while/lstm_cell_119/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/MatMul_1М
while/lstm_cell_119/addAddV2$while/lstm_cell_119/MatMul:product:0&while/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/addЫ
*while/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_119_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_119/BiasAdd/ReadVariableOpЩ
while/lstm_cell_119/BiasAddBiasAddwhile/lstm_cell_119/add:z:02while/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/BiasAdd
#while/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_119/split/split_dim
while/lstm_cell_119/splitSplit,while/lstm_cell_119/split/split_dim:output:0$while/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_119/split
while/lstm_cell_119/SigmoidSigmoid"while/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Sigmoid
while/lstm_cell_119/Sigmoid_1Sigmoid"while/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Sigmoid_1Ѓ
while/lstm_cell_119/mulMul!while/lstm_cell_119/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/mul
while/lstm_cell_119/ReluRelu"while/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/ReluИ
while/lstm_cell_119/mul_1Mulwhile/lstm_cell_119/Sigmoid:y:0&while/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/mul_1­
while/lstm_cell_119/add_1AddV2while/lstm_cell_119/mul:z:0while/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/add_1
while/lstm_cell_119/Sigmoid_2Sigmoid"while/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Sigmoid_2
while/lstm_cell_119/Relu_1Reluwhile/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Relu_1М
while/lstm_cell_119/mul_2Mul!while/lstm_cell_119/Sigmoid_2:y:0(while/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_119/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_119/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_119/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_119/BiasAdd/ReadVariableOp*^while/lstm_cell_119/MatMul/ReadVariableOp,^while/lstm_cell_119/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_119_biasadd_readvariableop_resource5while_lstm_cell_119_biasadd_readvariableop_resource_0"n
4while_lstm_cell_119_matmul_1_readvariableop_resource6while_lstm_cell_119_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_119_matmul_readvariableop_resource4while_lstm_cell_119_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_119/BiasAdd/ReadVariableOp*while/lstm_cell_119/BiasAdd/ReadVariableOp2V
)while/lstm_cell_119/MatMul/ReadVariableOp)while/lstm_cell_119/MatMul/ReadVariableOp2Z
+while/lstm_cell_119/MatMul_1/ReadVariableOp+while/lstm_cell_119/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
ЧH

M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_4930276

inputs(
lstm_cell_119_4930194:	Ш(
lstm_cell_119_4930196:	2Ш$
lstm_cell_119_4930198:	Ш
identityЂ%lstm_cell_119/StatefulPartitionedCallЂwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1
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
:џџџџџџџџџ22
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
B :ш2
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
zeros_1/packed/1
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
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
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
ReverseV2/axis
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	ReverseV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shape§
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
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Ї
%lstm_cell_119/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_119_4930194lstm_cell_119_4930196lstm_cell_119_4930198*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_119_layer_call_and_return_conditional_losses_49301272'
%lstm_cell_119/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterШ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_119_4930194lstm_cell_119_4930196lstm_cell_119_4930198*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_4930207*
condR
while_cond_4930206*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
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
:џџџџџџџџџ22

Identity~
NoOpNoOp&^lstm_cell_119/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2N
%lstm_cell_119/StatefulPartitionedCall%lstm_cell_119/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


J__inference_sequential_39_layer_call_and_return_conditional_losses_4932213

inputs
inputs_1	+
bidirectional_39_4932194:	Ш+
bidirectional_39_4932196:	2Ш'
bidirectional_39_4932198:	Ш+
bidirectional_39_4932200:	Ш+
bidirectional_39_4932202:	2Ш'
bidirectional_39_4932204:	Ш"
dense_39_4932207:d
dense_39_4932209:
identityЂ(bidirectional_39/StatefulPartitionedCallЂ dense_39/StatefulPartitionedCallК
(bidirectional_39/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_39_4932194bidirectional_39_4932196bidirectional_39_4932198bidirectional_39_4932200bidirectional_39_4932202bidirectional_39_4932204*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_49321502*
(bidirectional_39/StatefulPartitionedCallТ
 dense_39/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_39/StatefulPartitionedCall:output:0dense_39_4932207dense_39_4932209*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_39_layer_call_and_return_conditional_losses_49317352"
 dense_39/StatefulPartitionedCall
IdentityIdentity)dense_39/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp)^bidirectional_39/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2T
(bidirectional_39/StatefulPartitionedCall(bidirectional_39/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Жc

"forward_lstm_39_while_body_4933086<
8forward_lstm_39_while_forward_lstm_39_while_loop_counterB
>forward_lstm_39_while_forward_lstm_39_while_maximum_iterations%
!forward_lstm_39_while_placeholder'
#forward_lstm_39_while_placeholder_1'
#forward_lstm_39_while_placeholder_2'
#forward_lstm_39_while_placeholder_3'
#forward_lstm_39_while_placeholder_4;
7forward_lstm_39_while_forward_lstm_39_strided_slice_1_0w
sforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_08
4forward_lstm_39_while_greater_forward_lstm_39_cast_0W
Dforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0:	ШY
Fforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0:	2ШT
Eforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0:	Ш"
forward_lstm_39_while_identity$
 forward_lstm_39_while_identity_1$
 forward_lstm_39_while_identity_2$
 forward_lstm_39_while_identity_3$
 forward_lstm_39_while_identity_4$
 forward_lstm_39_while_identity_5$
 forward_lstm_39_while_identity_69
5forward_lstm_39_while_forward_lstm_39_strided_slice_1u
qforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor6
2forward_lstm_39_while_greater_forward_lstm_39_castU
Bforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource:	ШW
Dforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource:	2ШR
Cforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource:	ШЂ:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOpЂ9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOpЂ;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOpу
Gforward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gforward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
9forward_lstm_39/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_39_while_placeholderPforward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02;
9forward_lstm_39/while/TensorArrayV2Read/TensorListGetItemа
forward_lstm_39/while/GreaterGreater4forward_lstm_39_while_greater_forward_lstm_39_cast_0!forward_lstm_39_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_39/while/Greaterќ
9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOpDforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02;
9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp
*forward_lstm_39/while/lstm_cell_118/MatMulMatMul@forward_lstm_39/while/TensorArrayV2Read/TensorListGetItem:item:0Aforward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2,
*forward_lstm_39/while/lstm_cell_118/MatMul
;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOpFforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02=
;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp
,forward_lstm_39/while/lstm_cell_118/MatMul_1MatMul#forward_lstm_39_while_placeholder_3Cforward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_39/while/lstm_cell_118/MatMul_1ќ
'forward_lstm_39/while/lstm_cell_118/addAddV24forward_lstm_39/while/lstm_cell_118/MatMul:product:06forward_lstm_39/while/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_39/while/lstm_cell_118/addћ
:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOpEforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02<
:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp
+forward_lstm_39/while/lstm_cell_118/BiasAddBiasAdd+forward_lstm_39/while/lstm_cell_118/add:z:0Bforward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_39/while/lstm_cell_118/BiasAddЌ
3forward_lstm_39/while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3forward_lstm_39/while/lstm_cell_118/split/split_dimЯ
)forward_lstm_39/while/lstm_cell_118/splitSplit<forward_lstm_39/while/lstm_cell_118/split/split_dim:output:04forward_lstm_39/while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2+
)forward_lstm_39/while/lstm_cell_118/splitЫ
+forward_lstm_39/while/lstm_cell_118/SigmoidSigmoid2forward_lstm_39/while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_39/while/lstm_cell_118/SigmoidЯ
-forward_lstm_39/while/lstm_cell_118/Sigmoid_1Sigmoid2forward_lstm_39/while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22/
-forward_lstm_39/while/lstm_cell_118/Sigmoid_1у
'forward_lstm_39/while/lstm_cell_118/mulMul1forward_lstm_39/while/lstm_cell_118/Sigmoid_1:y:0#forward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_39/while/lstm_cell_118/mulТ
(forward_lstm_39/while/lstm_cell_118/ReluRelu2forward_lstm_39/while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_39/while/lstm_cell_118/Reluј
)forward_lstm_39/while/lstm_cell_118/mul_1Mul/forward_lstm_39/while/lstm_cell_118/Sigmoid:y:06forward_lstm_39/while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_39/while/lstm_cell_118/mul_1э
)forward_lstm_39/while/lstm_cell_118/add_1AddV2+forward_lstm_39/while/lstm_cell_118/mul:z:0-forward_lstm_39/while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_39/while/lstm_cell_118/add_1Я
-forward_lstm_39/while/lstm_cell_118/Sigmoid_2Sigmoid2forward_lstm_39/while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22/
-forward_lstm_39/while/lstm_cell_118/Sigmoid_2С
*forward_lstm_39/while/lstm_cell_118/Relu_1Relu-forward_lstm_39/while/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_39/while/lstm_cell_118/Relu_1ќ
)forward_lstm_39/while/lstm_cell_118/mul_2Mul1forward_lstm_39/while/lstm_cell_118/Sigmoid_2:y:08forward_lstm_39/while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_39/while/lstm_cell_118/mul_2я
forward_lstm_39/while/SelectSelect!forward_lstm_39/while/Greater:z:0-forward_lstm_39/while/lstm_cell_118/mul_2:z:0#forward_lstm_39_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/while/Selectѓ
forward_lstm_39/while/Select_1Select!forward_lstm_39/while/Greater:z:0-forward_lstm_39/while/lstm_cell_118/mul_2:z:0#forward_lstm_39_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22 
forward_lstm_39/while/Select_1ѓ
forward_lstm_39/while/Select_2Select!forward_lstm_39/while/Greater:z:0-forward_lstm_39/while/lstm_cell_118/add_1:z:0#forward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22 
forward_lstm_39/while/Select_2Љ
:forward_lstm_39/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_39_while_placeholder_1!forward_lstm_39_while_placeholder%forward_lstm_39/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_39/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_39/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_39/while/add/yЉ
forward_lstm_39/while/addAddV2!forward_lstm_39_while_placeholder$forward_lstm_39/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/while/add
forward_lstm_39/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_39/while/add_1/yЦ
forward_lstm_39/while/add_1AddV28forward_lstm_39_while_forward_lstm_39_while_loop_counter&forward_lstm_39/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/while/add_1Ћ
forward_lstm_39/while/IdentityIdentityforward_lstm_39/while/add_1:z:0^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_39/while/IdentityЮ
 forward_lstm_39/while/Identity_1Identity>forward_lstm_39_while_forward_lstm_39_while_maximum_iterations^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_39/while/Identity_1­
 forward_lstm_39/while/Identity_2Identityforward_lstm_39/while/add:z:0^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_39/while/Identity_2к
 forward_lstm_39/while/Identity_3IdentityJforward_lstm_39/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_39/while/Identity_3Ц
 forward_lstm_39/while/Identity_4Identity%forward_lstm_39/while/Select:output:0^forward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_39/while/Identity_4Ш
 forward_lstm_39/while/Identity_5Identity'forward_lstm_39/while/Select_1:output:0^forward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_39/while/Identity_5Ш
 forward_lstm_39/while/Identity_6Identity'forward_lstm_39/while/Select_2:output:0^forward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_39/while/Identity_6Б
forward_lstm_39/while/NoOpNoOp;^forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp:^forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp<^forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_39/while/NoOp"p
5forward_lstm_39_while_forward_lstm_39_strided_slice_17forward_lstm_39_while_forward_lstm_39_strided_slice_1_0"j
2forward_lstm_39_while_greater_forward_lstm_39_cast4forward_lstm_39_while_greater_forward_lstm_39_cast_0"I
forward_lstm_39_while_identity'forward_lstm_39/while/Identity:output:0"M
 forward_lstm_39_while_identity_1)forward_lstm_39/while/Identity_1:output:0"M
 forward_lstm_39_while_identity_2)forward_lstm_39/while/Identity_2:output:0"M
 forward_lstm_39_while_identity_3)forward_lstm_39/while/Identity_3:output:0"M
 forward_lstm_39_while_identity_4)forward_lstm_39/while/Identity_4:output:0"M
 forward_lstm_39_while_identity_5)forward_lstm_39/while/Identity_5:output:0"M
 forward_lstm_39_while_identity_6)forward_lstm_39/while/Identity_6:output:0"
Cforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resourceEforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0"
Dforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resourceFforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0"
Bforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resourceDforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0"ш
qforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensorsforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2x
:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp2v
9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp2z
;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:џџџџџџџџџ
к
Ш
while_cond_4934500
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_4934500___redundant_placeholder05
1while_while_cond_4934500___redundant_placeholder15
1while_while_cond_4934500___redundant_placeholder25
1while_while_cond_4934500___redundant_placeholder3
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
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
У

#backward_lstm_39_while_cond_4933264>
:backward_lstm_39_while_backward_lstm_39_while_loop_counterD
@backward_lstm_39_while_backward_lstm_39_while_maximum_iterations&
"backward_lstm_39_while_placeholder(
$backward_lstm_39_while_placeholder_1(
$backward_lstm_39_while_placeholder_2(
$backward_lstm_39_while_placeholder_3(
$backward_lstm_39_while_placeholder_4@
<backward_lstm_39_while_less_backward_lstm_39_strided_slice_1W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4933264___redundant_placeholder0W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4933264___redundant_placeholder1W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4933264___redundant_placeholder2W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4933264___redundant_placeholder3W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4933264___redundant_placeholder4#
backward_lstm_39_while_identity
Х
backward_lstm_39/while/LessLess"backward_lstm_39_while_placeholder<backward_lstm_39_while_less_backward_lstm_39_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_39/while/Less
backward_lstm_39/while/IdentityIdentitybackward_lstm_39/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_39/while/Identity"K
backward_lstm_39_while_identity(backward_lstm_39/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: :::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
я
м
Bsequential_39_bidirectional_39_backward_lstm_39_while_body_4929170|
xsequential_39_bidirectional_39_backward_lstm_39_while_sequential_39_bidirectional_39_backward_lstm_39_while_loop_counter
~sequential_39_bidirectional_39_backward_lstm_39_while_sequential_39_bidirectional_39_backward_lstm_39_while_maximum_iterationsE
Asequential_39_bidirectional_39_backward_lstm_39_while_placeholderG
Csequential_39_bidirectional_39_backward_lstm_39_while_placeholder_1G
Csequential_39_bidirectional_39_backward_lstm_39_while_placeholder_2G
Csequential_39_bidirectional_39_backward_lstm_39_while_placeholder_3G
Csequential_39_bidirectional_39_backward_lstm_39_while_placeholder_4{
wsequential_39_bidirectional_39_backward_lstm_39_while_sequential_39_bidirectional_39_backward_lstm_39_strided_slice_1_0И
Гsequential_39_bidirectional_39_backward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_sequential_39_bidirectional_39_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0v
rsequential_39_bidirectional_39_backward_lstm_39_while_less_sequential_39_bidirectional_39_backward_lstm_39_sub_1_0w
dsequential_39_bidirectional_39_backward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0:	Шy
fsequential_39_bidirectional_39_backward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0:	2Шt
esequential_39_bidirectional_39_backward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0:	ШB
>sequential_39_bidirectional_39_backward_lstm_39_while_identityD
@sequential_39_bidirectional_39_backward_lstm_39_while_identity_1D
@sequential_39_bidirectional_39_backward_lstm_39_while_identity_2D
@sequential_39_bidirectional_39_backward_lstm_39_while_identity_3D
@sequential_39_bidirectional_39_backward_lstm_39_while_identity_4D
@sequential_39_bidirectional_39_backward_lstm_39_while_identity_5D
@sequential_39_bidirectional_39_backward_lstm_39_while_identity_6y
usequential_39_bidirectional_39_backward_lstm_39_while_sequential_39_bidirectional_39_backward_lstm_39_strided_slice_1Ж
Бsequential_39_bidirectional_39_backward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_sequential_39_bidirectional_39_backward_lstm_39_tensorarrayunstack_tensorlistfromtensort
psequential_39_bidirectional_39_backward_lstm_39_while_less_sequential_39_bidirectional_39_backward_lstm_39_sub_1u
bsequential_39_bidirectional_39_backward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource:	Шw
dsequential_39_bidirectional_39_backward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource:	2Шr
csequential_39_bidirectional_39_backward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource:	ШЂZsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOpЂYsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOpЂ[sequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOpЃ
gsequential_39/bidirectional_39/backward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2i
gsequential_39/bidirectional_39/backward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeє
Ysequential_39/bidirectional_39/backward_lstm_39/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemГsequential_39_bidirectional_39_backward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_sequential_39_bidirectional_39_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0Asequential_39_bidirectional_39_backward_lstm_39_while_placeholderpsequential_39/bidirectional_39/backward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02[
Ysequential_39/bidirectional_39/backward_lstm_39/while/TensorArrayV2Read/TensorListGetItemх
:sequential_39/bidirectional_39/backward_lstm_39/while/LessLessrsequential_39_bidirectional_39_backward_lstm_39_while_less_sequential_39_bidirectional_39_backward_lstm_39_sub_1_0Asequential_39_bidirectional_39_backward_lstm_39_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2<
:sequential_39/bidirectional_39/backward_lstm_39/while/Lessм
Ysequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOpReadVariableOpdsequential_39_bidirectional_39_backward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02[
Ysequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp
Jsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMulMatMul`sequential_39/bidirectional_39/backward_lstm_39/while/TensorArrayV2Read/TensorListGetItem:item:0asequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2L
Jsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMulт
[sequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOpfsequential_39_bidirectional_39_backward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02]
[sequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp
Lsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMul_1MatMulCsequential_39_bidirectional_39_backward_lstm_39_while_placeholder_3csequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2N
Lsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMul_1ќ
Gsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/addAddV2Tsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMul:product:0Vsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2I
Gsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/addл
Zsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOpesequential_39_bidirectional_39_backward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02\
Zsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp
Ksequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/BiasAddBiasAddKsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/add:z:0bsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2M
Ksequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/BiasAddь
Ssequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2U
Ssequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/split/split_dimЯ
Isequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/splitSplit\sequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/split/split_dim:output:0Tsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2K
Isequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/splitЋ
Ksequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/SigmoidSigmoidRsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22M
Ksequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/SigmoidЏ
Msequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/Sigmoid_1SigmoidRsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22O
Msequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/Sigmoid_1у
Gsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/mulMulQsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/Sigmoid_1:y:0Csequential_39_bidirectional_39_backward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22I
Gsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/mulЂ
Hsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/ReluReluRsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22J
Hsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/Reluј
Isequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/mul_1MulOsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/Sigmoid:y:0Vsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22K
Isequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/mul_1э
Isequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/add_1AddV2Ksequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/mul:z:0Msequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22K
Isequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/add_1Џ
Msequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/Sigmoid_2SigmoidRsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22O
Msequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/Sigmoid_2Ё
Jsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/Relu_1ReluMsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22L
Jsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/Relu_1ќ
Isequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/mul_2MulQsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/Sigmoid_2:y:0Xsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22K
Isequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/mul_2
<sequential_39/bidirectional_39/backward_lstm_39/while/SelectSelect>sequential_39/bidirectional_39/backward_lstm_39/while/Less:z:0Msequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/mul_2:z:0Csequential_39_bidirectional_39_backward_lstm_39_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22>
<sequential_39/bidirectional_39/backward_lstm_39/while/Select
>sequential_39/bidirectional_39/backward_lstm_39/while/Select_1Select>sequential_39/bidirectional_39/backward_lstm_39/while/Less:z:0Msequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/mul_2:z:0Csequential_39_bidirectional_39_backward_lstm_39_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22@
>sequential_39/bidirectional_39/backward_lstm_39/while/Select_1
>sequential_39/bidirectional_39/backward_lstm_39/while/Select_2Select>sequential_39/bidirectional_39/backward_lstm_39/while/Less:z:0Msequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/add_1:z:0Csequential_39_bidirectional_39_backward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22@
>sequential_39/bidirectional_39/backward_lstm_39/while/Select_2Щ
Zsequential_39/bidirectional_39/backward_lstm_39/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemCsequential_39_bidirectional_39_backward_lstm_39_while_placeholder_1Asequential_39_bidirectional_39_backward_lstm_39_while_placeholderEsequential_39/bidirectional_39/backward_lstm_39/while/Select:output:0*
_output_shapes
: *
element_dtype02\
Zsequential_39/bidirectional_39/backward_lstm_39/while/TensorArrayV2Write/TensorListSetItemМ
;sequential_39/bidirectional_39/backward_lstm_39/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2=
;sequential_39/bidirectional_39/backward_lstm_39/while/add/yЉ
9sequential_39/bidirectional_39/backward_lstm_39/while/addAddV2Asequential_39_bidirectional_39_backward_lstm_39_while_placeholderDsequential_39/bidirectional_39/backward_lstm_39/while/add/y:output:0*
T0*
_output_shapes
: 2;
9sequential_39/bidirectional_39/backward_lstm_39/while/addР
=sequential_39/bidirectional_39/backward_lstm_39/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2?
=sequential_39/bidirectional_39/backward_lstm_39/while/add_1/yц
;sequential_39/bidirectional_39/backward_lstm_39/while/add_1AddV2xsequential_39_bidirectional_39_backward_lstm_39_while_sequential_39_bidirectional_39_backward_lstm_39_while_loop_counterFsequential_39/bidirectional_39/backward_lstm_39/while/add_1/y:output:0*
T0*
_output_shapes
: 2=
;sequential_39/bidirectional_39/backward_lstm_39/while/add_1Ћ
>sequential_39/bidirectional_39/backward_lstm_39/while/IdentityIdentity?sequential_39/bidirectional_39/backward_lstm_39/while/add_1:z:0;^sequential_39/bidirectional_39/backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2@
>sequential_39/bidirectional_39/backward_lstm_39/while/Identityю
@sequential_39/bidirectional_39/backward_lstm_39/while/Identity_1Identity~sequential_39_bidirectional_39_backward_lstm_39_while_sequential_39_bidirectional_39_backward_lstm_39_while_maximum_iterations;^sequential_39/bidirectional_39/backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2B
@sequential_39/bidirectional_39/backward_lstm_39/while/Identity_1­
@sequential_39/bidirectional_39/backward_lstm_39/while/Identity_2Identity=sequential_39/bidirectional_39/backward_lstm_39/while/add:z:0;^sequential_39/bidirectional_39/backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2B
@sequential_39/bidirectional_39/backward_lstm_39/while/Identity_2к
@sequential_39/bidirectional_39/backward_lstm_39/while/Identity_3Identityjsequential_39/bidirectional_39/backward_lstm_39/while/TensorArrayV2Write/TensorListSetItem:output_handle:0;^sequential_39/bidirectional_39/backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2B
@sequential_39/bidirectional_39/backward_lstm_39/while/Identity_3Ц
@sequential_39/bidirectional_39/backward_lstm_39/while/Identity_4IdentityEsequential_39/bidirectional_39/backward_lstm_39/while/Select:output:0;^sequential_39/bidirectional_39/backward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22B
@sequential_39/bidirectional_39/backward_lstm_39/while/Identity_4Ш
@sequential_39/bidirectional_39/backward_lstm_39/while/Identity_5IdentityGsequential_39/bidirectional_39/backward_lstm_39/while/Select_1:output:0;^sequential_39/bidirectional_39/backward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22B
@sequential_39/bidirectional_39/backward_lstm_39/while/Identity_5Ш
@sequential_39/bidirectional_39/backward_lstm_39/while/Identity_6IdentityGsequential_39/bidirectional_39/backward_lstm_39/while/Select_2:output:0;^sequential_39/bidirectional_39/backward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22B
@sequential_39/bidirectional_39/backward_lstm_39/while/Identity_6б
:sequential_39/bidirectional_39/backward_lstm_39/while/NoOpNoOp[^sequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOpZ^sequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp\^sequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2<
:sequential_39/bidirectional_39/backward_lstm_39/while/NoOp"
>sequential_39_bidirectional_39_backward_lstm_39_while_identityGsequential_39/bidirectional_39/backward_lstm_39/while/Identity:output:0"
@sequential_39_bidirectional_39_backward_lstm_39_while_identity_1Isequential_39/bidirectional_39/backward_lstm_39/while/Identity_1:output:0"
@sequential_39_bidirectional_39_backward_lstm_39_while_identity_2Isequential_39/bidirectional_39/backward_lstm_39/while/Identity_2:output:0"
@sequential_39_bidirectional_39_backward_lstm_39_while_identity_3Isequential_39/bidirectional_39/backward_lstm_39/while/Identity_3:output:0"
@sequential_39_bidirectional_39_backward_lstm_39_while_identity_4Isequential_39/bidirectional_39/backward_lstm_39/while/Identity_4:output:0"
@sequential_39_bidirectional_39_backward_lstm_39_while_identity_5Isequential_39/bidirectional_39/backward_lstm_39/while/Identity_5:output:0"
@sequential_39_bidirectional_39_backward_lstm_39_while_identity_6Isequential_39/bidirectional_39/backward_lstm_39/while/Identity_6:output:0"ц
psequential_39_bidirectional_39_backward_lstm_39_while_less_sequential_39_bidirectional_39_backward_lstm_39_sub_1rsequential_39_bidirectional_39_backward_lstm_39_while_less_sequential_39_bidirectional_39_backward_lstm_39_sub_1_0"Ь
csequential_39_bidirectional_39_backward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resourceesequential_39_bidirectional_39_backward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0"Ю
dsequential_39_bidirectional_39_backward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resourcefsequential_39_bidirectional_39_backward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0"Ъ
bsequential_39_bidirectional_39_backward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resourcedsequential_39_bidirectional_39_backward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0"№
usequential_39_bidirectional_39_backward_lstm_39_while_sequential_39_bidirectional_39_backward_lstm_39_strided_slice_1wsequential_39_bidirectional_39_backward_lstm_39_while_sequential_39_bidirectional_39_backward_lstm_39_strided_slice_1_0"ъ
Бsequential_39_bidirectional_39_backward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_sequential_39_bidirectional_39_backward_lstm_39_tensorarrayunstack_tensorlistfromtensorГsequential_39_bidirectional_39_backward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_sequential_39_bidirectional_39_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2И
Zsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOpZsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp2Ж
Ysequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOpYsequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp2К
[sequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp[sequential_39/bidirectional_39/backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:џџџџџџџџџ
к
Ш
while_cond_4930966
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_4930966___redundant_placeholder05
1while_while_cond_4930966___redundant_placeholder15
1while_while_cond_4930966___redundant_placeholder25
1while_while_cond_4930966___redundant_placeholder3
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
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:


J__inference_sequential_39_layer_call_and_return_conditional_losses_4932277

inputs
inputs_1	+
bidirectional_39_4932258:	Ш+
bidirectional_39_4932260:	2Ш'
bidirectional_39_4932262:	Ш+
bidirectional_39_4932264:	Ш+
bidirectional_39_4932266:	2Ш'
bidirectional_39_4932268:	Ш"
dense_39_4932271:d
dense_39_4932273:
identityЂ(bidirectional_39/StatefulPartitionedCallЂ dense_39/StatefulPartitionedCallК
(bidirectional_39/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_39_4932258bidirectional_39_4932260bidirectional_39_4932262bidirectional_39_4932264bidirectional_39_4932266bidirectional_39_4932268*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_49317102*
(bidirectional_39/StatefulPartitionedCallТ
 dense_39/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_39/StatefulPartitionedCall:output:0dense_39_4932271dense_39_4932273*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_39_layer_call_and_return_conditional_losses_49317352"
 dense_39/StatefulPartitionedCall
IdentityIdentity)dense_39/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp)^bidirectional_39/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2T
(bidirectional_39/StatefulPartitionedCall(bidirectional_39/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
к
Ш
while_cond_4930614
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_4930614___redundant_placeholder05
1while_while_cond_4930614___redundant_placeholder15
1while_while_cond_4930614___redundant_placeholder25
1while_while_cond_4930614___redundant_placeholder3
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
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
ц
Ф
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_4931272

inputs*
forward_lstm_39_4931255:	Ш*
forward_lstm_39_4931257:	2Ш&
forward_lstm_39_4931259:	Ш+
backward_lstm_39_4931262:	Ш+
backward_lstm_39_4931264:	2Ш'
backward_lstm_39_4931266:	Ш
identityЂ(backward_lstm_39/StatefulPartitionedCallЂ'forward_lstm_39/StatefulPartitionedCallе
'forward_lstm_39/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_39_4931255forward_lstm_39_4931257forward_lstm_39_4931259*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_49312242)
'forward_lstm_39/StatefulPartitionedCallл
(backward_lstm_39/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_39_4931262backward_lstm_39_4931264backward_lstm_39_4931266*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_49310512*
(backward_lstm_39/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisв
concatConcatV20forward_lstm_39/StatefulPartitionedCall:output:01backward_lstm_39/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

IdentityЃ
NoOpNoOp)^backward_lstm_39/StatefulPartitionedCall(^forward_lstm_39/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2T
(backward_lstm_39/StatefulPartitionedCall(backward_lstm_39/StatefulPartitionedCall2R
'forward_lstm_39/StatefulPartitionedCall'forward_lstm_39/StatefulPartitionedCall:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
&
ё
while_body_4929573
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_118_4929597_0:	Ш0
while_lstm_cell_118_4929599_0:	2Ш,
while_lstm_cell_118_4929601_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_118_4929597:	Ш.
while_lstm_cell_118_4929599:	2Ш*
while_lstm_cell_118_4929601:	ШЂ+while/lstm_cell_118/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemы
+while/lstm_cell_118/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_118_4929597_0while_lstm_cell_118_4929599_0while_lstm_cell_118_4929601_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_118_layer_call_and_return_conditional_losses_49294952-
+while/lstm_cell_118/StatefulPartitionedCallј
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_118/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Ѕ
while/Identity_4Identity4while/lstm_cell_118/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѕ
while/Identity_5Identity4while/lstm_cell_118/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp,^while/lstm_cell_118/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_118_4929597while_lstm_cell_118_4929597_0"<
while_lstm_cell_118_4929599while_lstm_cell_118_4929599_0"<
while_lstm_cell_118_4929601while_lstm_cell_118_4929601_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2Z
+while/lstm_cell_118/StatefulPartitionedCall+while/lstm_cell_118/StatefulPartitionedCall: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
гљ
г
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_4932702
inputs_0O
<forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource:	ШQ
>forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource:	2ШL
=forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource:	ШP
=backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource:	ШR
?backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource:	2ШM
>backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource:	Ш
identityЂ5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpЂ4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpЂ6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOpЂbackward_lstm_39/whileЂ4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpЂ3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOpЂ5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpЂforward_lstm_39/whilef
forward_lstm_39/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_39/Shape
#forward_lstm_39/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_39/strided_slice/stack
%forward_lstm_39/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_39/strided_slice/stack_1
%forward_lstm_39/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_39/strided_slice/stack_2Т
forward_lstm_39/strided_sliceStridedSliceforward_lstm_39/Shape:output:0,forward_lstm_39/strided_slice/stack:output:0.forward_lstm_39/strided_slice/stack_1:output:0.forward_lstm_39/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_39/strided_slice|
forward_lstm_39/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_39/zeros/mul/yЌ
forward_lstm_39/zeros/mulMul&forward_lstm_39/strided_slice:output:0$forward_lstm_39/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros/mul
forward_lstm_39/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_39/zeros/Less/yЇ
forward_lstm_39/zeros/LessLessforward_lstm_39/zeros/mul:z:0%forward_lstm_39/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros/Less
forward_lstm_39/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_39/zeros/packed/1У
forward_lstm_39/zeros/packedPack&forward_lstm_39/strided_slice:output:0'forward_lstm_39/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_39/zeros/packed
forward_lstm_39/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_39/zeros/ConstЕ
forward_lstm_39/zerosFill%forward_lstm_39/zeros/packed:output:0$forward_lstm_39/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/zeros
forward_lstm_39/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_39/zeros_1/mul/yВ
forward_lstm_39/zeros_1/mulMul&forward_lstm_39/strided_slice:output:0&forward_lstm_39/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros_1/mul
forward_lstm_39/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
forward_lstm_39/zeros_1/Less/yЏ
forward_lstm_39/zeros_1/LessLessforward_lstm_39/zeros_1/mul:z:0'forward_lstm_39/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros_1/Less
 forward_lstm_39/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_39/zeros_1/packed/1Щ
forward_lstm_39/zeros_1/packedPack&forward_lstm_39/strided_slice:output:0)forward_lstm_39/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_39/zeros_1/packed
forward_lstm_39/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_39/zeros_1/ConstН
forward_lstm_39/zeros_1Fill'forward_lstm_39/zeros_1/packed:output:0&forward_lstm_39/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/zeros_1
forward_lstm_39/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_39/transpose/permО
forward_lstm_39/transpose	Transposeinputs_0'forward_lstm_39/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
forward_lstm_39/transpose
forward_lstm_39/Shape_1Shapeforward_lstm_39/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_39/Shape_1
%forward_lstm_39/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_39/strided_slice_1/stack
'forward_lstm_39/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_1/stack_1
'forward_lstm_39/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_1/stack_2Ю
forward_lstm_39/strided_slice_1StridedSlice forward_lstm_39/Shape_1:output:0.forward_lstm_39/strided_slice_1/stack:output:00forward_lstm_39/strided_slice_1/stack_1:output:00forward_lstm_39/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_39/strided_slice_1Ѕ
+forward_lstm_39/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+forward_lstm_39/TensorArrayV2/element_shapeђ
forward_lstm_39/TensorArrayV2TensorListReserve4forward_lstm_39/TensorArrayV2/element_shape:output:0(forward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_39/TensorArrayV2п
Eforward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2G
Eforward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
7forward_lstm_39/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_39/transpose:y:0Nforward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_39/TensorArrayUnstack/TensorListFromTensor
%forward_lstm_39/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_39/strided_slice_2/stack
'forward_lstm_39/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_2/stack_1
'forward_lstm_39/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_2/stack_2х
forward_lstm_39/strided_slice_2StridedSliceforward_lstm_39/transpose:y:0.forward_lstm_39/strided_slice_2/stack:output:00forward_lstm_39/strided_slice_2/stack_1:output:00forward_lstm_39/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2!
forward_lstm_39/strided_slice_2ш
3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp<forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype025
3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp№
$forward_lstm_39/lstm_cell_118/MatMulMatMul(forward_lstm_39/strided_slice_2:output:0;forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$forward_lstm_39/lstm_cell_118/MatMulю
5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp>forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype027
5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpь
&forward_lstm_39/lstm_cell_118/MatMul_1MatMulforward_lstm_39/zeros:output:0=forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_39/lstm_cell_118/MatMul_1ф
!forward_lstm_39/lstm_cell_118/addAddV2.forward_lstm_39/lstm_cell_118/MatMul:product:00forward_lstm_39/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2#
!forward_lstm_39/lstm_cell_118/addч
4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp=forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype026
4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpё
%forward_lstm_39/lstm_cell_118/BiasAddBiasAdd%forward_lstm_39/lstm_cell_118/add:z:0<forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_39/lstm_cell_118/BiasAdd 
-forward_lstm_39/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-forward_lstm_39/lstm_cell_118/split/split_dimЗ
#forward_lstm_39/lstm_cell_118/splitSplit6forward_lstm_39/lstm_cell_118/split/split_dim:output:0.forward_lstm_39/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2%
#forward_lstm_39/lstm_cell_118/splitЙ
%forward_lstm_39/lstm_cell_118/SigmoidSigmoid,forward_lstm_39/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_39/lstm_cell_118/SigmoidН
'forward_lstm_39/lstm_cell_118/Sigmoid_1Sigmoid,forward_lstm_39/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_39/lstm_cell_118/Sigmoid_1Ю
!forward_lstm_39/lstm_cell_118/mulMul+forward_lstm_39/lstm_cell_118/Sigmoid_1:y:0 forward_lstm_39/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_39/lstm_cell_118/mulА
"forward_lstm_39/lstm_cell_118/ReluRelu,forward_lstm_39/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_39/lstm_cell_118/Reluр
#forward_lstm_39/lstm_cell_118/mul_1Mul)forward_lstm_39/lstm_cell_118/Sigmoid:y:00forward_lstm_39/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_39/lstm_cell_118/mul_1е
#forward_lstm_39/lstm_cell_118/add_1AddV2%forward_lstm_39/lstm_cell_118/mul:z:0'forward_lstm_39/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_39/lstm_cell_118/add_1Н
'forward_lstm_39/lstm_cell_118/Sigmoid_2Sigmoid,forward_lstm_39/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_39/lstm_cell_118/Sigmoid_2Џ
$forward_lstm_39/lstm_cell_118/Relu_1Relu'forward_lstm_39/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_39/lstm_cell_118/Relu_1ф
#forward_lstm_39/lstm_cell_118/mul_2Mul+forward_lstm_39/lstm_cell_118/Sigmoid_2:y:02forward_lstm_39/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_39/lstm_cell_118/mul_2Џ
-forward_lstm_39/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2/
-forward_lstm_39/TensorArrayV2_1/element_shapeј
forward_lstm_39/TensorArrayV2_1TensorListReserve6forward_lstm_39/TensorArrayV2_1/element_shape:output:0(forward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_39/TensorArrayV2_1n
forward_lstm_39/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_39/time
(forward_lstm_39/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(forward_lstm_39/while/maximum_iterations
"forward_lstm_39/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_39/while/loop_counter
forward_lstm_39/whileWhile+forward_lstm_39/while/loop_counter:output:01forward_lstm_39/while/maximum_iterations:output:0forward_lstm_39/time:output:0(forward_lstm_39/TensorArrayV2_1:handle:0forward_lstm_39/zeros:output:0 forward_lstm_39/zeros_1:output:0(forward_lstm_39/strided_slice_1:output:0Gforward_lstm_39/TensorArrayUnstack/TensorListFromTensor:output_handle:0<forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource>forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource=forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *.
body&R$
"forward_lstm_39_while_body_4932467*.
cond&R$
"forward_lstm_39_while_cond_4932466*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
forward_lstm_39/whileе
@forward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2B
@forward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeБ
2forward_lstm_39/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_39/while:output:3Iforward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype024
2forward_lstm_39/TensorArrayV2Stack/TensorListStackЁ
%forward_lstm_39/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2'
%forward_lstm_39/strided_slice_3/stack
'forward_lstm_39/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_39/strided_slice_3/stack_1
'forward_lstm_39/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_3/stack_2њ
forward_lstm_39/strided_slice_3StridedSlice;forward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_39/strided_slice_3/stack:output:00forward_lstm_39/strided_slice_3/stack_1:output:00forward_lstm_39/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2!
forward_lstm_39/strided_slice_3
 forward_lstm_39/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_39/transpose_1/permю
forward_lstm_39/transpose_1	Transpose;forward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_39/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_39/transpose_1
forward_lstm_39/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_39/runtimeh
backward_lstm_39/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_39/Shape
$backward_lstm_39/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_39/strided_slice/stack
&backward_lstm_39/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_39/strided_slice/stack_1
&backward_lstm_39/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_39/strided_slice/stack_2Ш
backward_lstm_39/strided_sliceStridedSlicebackward_lstm_39/Shape:output:0-backward_lstm_39/strided_slice/stack:output:0/backward_lstm_39/strided_slice/stack_1:output:0/backward_lstm_39/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_39/strided_slice~
backward_lstm_39/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_39/zeros/mul/yА
backward_lstm_39/zeros/mulMul'backward_lstm_39/strided_slice:output:0%backward_lstm_39/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros/mul
backward_lstm_39/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
backward_lstm_39/zeros/Less/yЋ
backward_lstm_39/zeros/LessLessbackward_lstm_39/zeros/mul:z:0&backward_lstm_39/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros/Less
backward_lstm_39/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_39/zeros/packed/1Ч
backward_lstm_39/zeros/packedPack'backward_lstm_39/strided_slice:output:0(backward_lstm_39/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_39/zeros/packed
backward_lstm_39/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_39/zeros/ConstЙ
backward_lstm_39/zerosFill&backward_lstm_39/zeros/packed:output:0%backward_lstm_39/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/zeros
backward_lstm_39/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_39/zeros_1/mul/yЖ
backward_lstm_39/zeros_1/mulMul'backward_lstm_39/strided_slice:output:0'backward_lstm_39/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros_1/mul
backward_lstm_39/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
backward_lstm_39/zeros_1/Less/yГ
backward_lstm_39/zeros_1/LessLess backward_lstm_39/zeros_1/mul:z:0(backward_lstm_39/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros_1/Less
!backward_lstm_39/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_39/zeros_1/packed/1Э
backward_lstm_39/zeros_1/packedPack'backward_lstm_39/strided_slice:output:0*backward_lstm_39/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_39/zeros_1/packed
backward_lstm_39/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_39/zeros_1/ConstС
backward_lstm_39/zeros_1Fill(backward_lstm_39/zeros_1/packed:output:0'backward_lstm_39/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/zeros_1
backward_lstm_39/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_39/transpose/permС
backward_lstm_39/transpose	Transposeinputs_0(backward_lstm_39/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_39/transpose
backward_lstm_39/Shape_1Shapebackward_lstm_39/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_39/Shape_1
&backward_lstm_39/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_39/strided_slice_1/stack
(backward_lstm_39/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_1/stack_1
(backward_lstm_39/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_1/stack_2д
 backward_lstm_39/strided_slice_1StridedSlice!backward_lstm_39/Shape_1:output:0/backward_lstm_39/strided_slice_1/stack:output:01backward_lstm_39/strided_slice_1/stack_1:output:01backward_lstm_39/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_39/strided_slice_1Ї
,backward_lstm_39/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,backward_lstm_39/TensorArrayV2/element_shapeі
backward_lstm_39/TensorArrayV2TensorListReserve5backward_lstm_39/TensorArrayV2/element_shape:output:0)backward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_39/TensorArrayV2
backward_lstm_39/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_39/ReverseV2/axisз
backward_lstm_39/ReverseV2	ReverseV2backward_lstm_39/transpose:y:0(backward_lstm_39/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_39/ReverseV2с
Fbackward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2H
Fbackward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeС
8backward_lstm_39/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_39/ReverseV2:output:0Obackward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_39/TensorArrayUnstack/TensorListFromTensor
&backward_lstm_39/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_39/strided_slice_2/stack
(backward_lstm_39/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_2/stack_1
(backward_lstm_39/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_2/stack_2ы
 backward_lstm_39/strided_slice_2StridedSlicebackward_lstm_39/transpose:y:0/backward_lstm_39/strided_slice_2/stack:output:01backward_lstm_39/strided_slice_2/stack_1:output:01backward_lstm_39/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2"
 backward_lstm_39/strided_slice_2ы
4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpReadVariableOp=backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpє
%backward_lstm_39/lstm_cell_119/MatMulMatMul)backward_lstm_39/strided_slice_2:output:0<backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%backward_lstm_39/lstm_cell_119/MatMulё
6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOp?backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp№
'backward_lstm_39/lstm_cell_119/MatMul_1MatMulbackward_lstm_39/zeros:output:0>backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_39/lstm_cell_119/MatMul_1ш
"backward_lstm_39/lstm_cell_119/addAddV2/backward_lstm_39/lstm_cell_119/MatMul:product:01backward_lstm_39/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"backward_lstm_39/lstm_cell_119/addъ
5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOp>backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpѕ
&backward_lstm_39/lstm_cell_119/BiasAddBiasAdd&backward_lstm_39/lstm_cell_119/add:z:0=backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_39/lstm_cell_119/BiasAddЂ
.backward_lstm_39/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.backward_lstm_39/lstm_cell_119/split/split_dimЛ
$backward_lstm_39/lstm_cell_119/splitSplit7backward_lstm_39/lstm_cell_119/split/split_dim:output:0/backward_lstm_39/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$backward_lstm_39/lstm_cell_119/splitМ
&backward_lstm_39/lstm_cell_119/SigmoidSigmoid-backward_lstm_39/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_39/lstm_cell_119/SigmoidР
(backward_lstm_39/lstm_cell_119/Sigmoid_1Sigmoid-backward_lstm_39/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_39/lstm_cell_119/Sigmoid_1в
"backward_lstm_39/lstm_cell_119/mulMul,backward_lstm_39/lstm_cell_119/Sigmoid_1:y:0!backward_lstm_39/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_39/lstm_cell_119/mulГ
#backward_lstm_39/lstm_cell_119/ReluRelu-backward_lstm_39/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_39/lstm_cell_119/Reluф
$backward_lstm_39/lstm_cell_119/mul_1Mul*backward_lstm_39/lstm_cell_119/Sigmoid:y:01backward_lstm_39/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_39/lstm_cell_119/mul_1й
$backward_lstm_39/lstm_cell_119/add_1AddV2&backward_lstm_39/lstm_cell_119/mul:z:0(backward_lstm_39/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_39/lstm_cell_119/add_1Р
(backward_lstm_39/lstm_cell_119/Sigmoid_2Sigmoid-backward_lstm_39/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_39/lstm_cell_119/Sigmoid_2В
%backward_lstm_39/lstm_cell_119/Relu_1Relu(backward_lstm_39/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_39/lstm_cell_119/Relu_1ш
$backward_lstm_39/lstm_cell_119/mul_2Mul,backward_lstm_39/lstm_cell_119/Sigmoid_2:y:03backward_lstm_39/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_39/lstm_cell_119/mul_2Б
.backward_lstm_39/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.backward_lstm_39/TensorArrayV2_1/element_shapeќ
 backward_lstm_39/TensorArrayV2_1TensorListReserve7backward_lstm_39/TensorArrayV2_1/element_shape:output:0)backward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_39/TensorArrayV2_1p
backward_lstm_39/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_39/timeЁ
)backward_lstm_39/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)backward_lstm_39/while/maximum_iterations
#backward_lstm_39/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_39/while/loop_counter
backward_lstm_39/whileWhile,backward_lstm_39/while/loop_counter:output:02backward_lstm_39/while/maximum_iterations:output:0backward_lstm_39/time:output:0)backward_lstm_39/TensorArrayV2_1:handle:0backward_lstm_39/zeros:output:0!backward_lstm_39/zeros_1:output:0)backward_lstm_39/strided_slice_1:output:0Hbackward_lstm_39/TensorArrayUnstack/TensorListFromTensor:output_handle:0=backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource?backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource>backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( */
body'R%
#backward_lstm_39_while_body_4932616*/
cond'R%
#backward_lstm_39_while_cond_4932615*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
backward_lstm_39/whileз
Abackward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Abackward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3backward_lstm_39/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_39/while:output:3Jbackward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3backward_lstm_39/TensorArrayV2Stack/TensorListStackЃ
&backward_lstm_39/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&backward_lstm_39/strided_slice_3/stack
(backward_lstm_39/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_39/strided_slice_3/stack_1
(backward_lstm_39/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_3/stack_2
 backward_lstm_39/strided_slice_3StridedSlice<backward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_39/strided_slice_3/stack:output:01backward_lstm_39/strided_slice_3/stack_1:output:01backward_lstm_39/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 backward_lstm_39/strided_slice_3
!backward_lstm_39/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_39/transpose_1/permђ
backward_lstm_39/transpose_1	Transpose<backward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_39/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_39/transpose_1
backward_lstm_39/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_39/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisТ
concatConcatV2(forward_lstm_39/strided_slice_3:output:0)backward_lstm_39/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

IdentityЬ
NoOpNoOp6^backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp5^backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp7^backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp^backward_lstm_39/while5^forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp4^forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp6^forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp^forward_lstm_39/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2n
5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp2l
4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp2p
6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp20
backward_lstm_39/whilebackward_lstm_39/while2l
4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp2j
3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp2n
5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp2.
forward_lstm_39/whileforward_lstm_39/while:g c
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
к
Ш
while_cond_4934152
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_4934152___redundant_placeholder05
1while_while_cond_4934152___redundant_placeholder15
1while_while_cond_4934152___redundant_placeholder25
1while_while_cond_4934152___redundant_placeholder3
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
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
к
Ш
while_cond_4929572
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_4929572___redundant_placeholder05
1while_while_cond_4929572___redundant_placeholder15
1while_while_cond_4929572___redundant_placeholder25
1while_while_cond_4929572___redundant_placeholder3
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
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
&
ё
while_body_4929363
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_118_4929387_0:	Ш0
while_lstm_cell_118_4929389_0:	2Ш,
while_lstm_cell_118_4929391_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_118_4929387:	Ш.
while_lstm_cell_118_4929389:	2Ш*
while_lstm_cell_118_4929391:	ШЂ+while/lstm_cell_118/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemы
+while/lstm_cell_118/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_118_4929387_0while_lstm_cell_118_4929389_0while_lstm_cell_118_4929391_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_118_layer_call_and_return_conditional_losses_49293492-
+while/lstm_cell_118/StatefulPartitionedCallј
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_118/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Ѕ
while/Identity_4Identity4while/lstm_cell_118/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѕ
while/Identity_5Identity4while/lstm_cell_118/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp,^while/lstm_cell_118/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_118_4929387while_lstm_cell_118_4929387_0"<
while_lstm_cell_118_4929389while_lstm_cell_118_4929389_0"<
while_lstm_cell_118_4929391while_lstm_cell_118_4929391_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2Z
+while/lstm_cell_118/StatefulPartitionedCall+while/lstm_cell_118/StatefulPartitionedCall: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
м
О
1__inference_forward_lstm_39_layer_call_fn_4933784

inputs
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_49312242
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
а
Р
1__inference_forward_lstm_39_layer_call_fn_4933751
inputs_0
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_49294322
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
ЛИ
п
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_4932150

inputs
inputs_1	O
<forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource:	ШQ
>forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource:	2ШL
=forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource:	ШP
=backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource:	ШR
?backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource:	2ШM
>backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource:	Ш
identityЂ5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpЂ4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpЂ6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOpЂbackward_lstm_39/whileЂ4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpЂ3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOpЂ5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpЂforward_lstm_39/while
$forward_lstm_39/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$forward_lstm_39/RaggedToTensor/zeros
$forward_lstm_39/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2&
$forward_lstm_39/RaggedToTensor/Const
3forward_lstm_39/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-forward_lstm_39/RaggedToTensor/Const:output:0inputs-forward_lstm_39/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3forward_lstm_39/RaggedToTensor/RaggedTensorToTensorТ
:forward_lstm_39/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:forward_lstm_39/RaggedNestedRowLengths/strided_slice/stackЦ
<forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1Ц
<forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2Є
4forward_lstm_39/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cforward_lstm_39/RaggedNestedRowLengths/strided_slice/stack:output:0Eforward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1:output:0Eforward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask26
4forward_lstm_39/RaggedNestedRowLengths/strided_sliceЦ
<forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stackг
>forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2@
>forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1Ъ
>forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2А
6forward_lstm_39/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Eforward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gforward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gforward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask28
6forward_lstm_39/RaggedNestedRowLengths/strided_slice_1
*forward_lstm_39/RaggedNestedRowLengths/subSub=forward_lstm_39/RaggedNestedRowLengths/strided_slice:output:0?forward_lstm_39/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2,
*forward_lstm_39/RaggedNestedRowLengths/subЁ
forward_lstm_39/CastCast.forward_lstm_39/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_39/Cast
forward_lstm_39/ShapeShape<forward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_39/Shape
#forward_lstm_39/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_39/strided_slice/stack
%forward_lstm_39/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_39/strided_slice/stack_1
%forward_lstm_39/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_39/strided_slice/stack_2Т
forward_lstm_39/strided_sliceStridedSliceforward_lstm_39/Shape:output:0,forward_lstm_39/strided_slice/stack:output:0.forward_lstm_39/strided_slice/stack_1:output:0.forward_lstm_39/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_39/strided_slice|
forward_lstm_39/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_39/zeros/mul/yЌ
forward_lstm_39/zeros/mulMul&forward_lstm_39/strided_slice:output:0$forward_lstm_39/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros/mul
forward_lstm_39/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_39/zeros/Less/yЇ
forward_lstm_39/zeros/LessLessforward_lstm_39/zeros/mul:z:0%forward_lstm_39/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros/Less
forward_lstm_39/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_39/zeros/packed/1У
forward_lstm_39/zeros/packedPack&forward_lstm_39/strided_slice:output:0'forward_lstm_39/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_39/zeros/packed
forward_lstm_39/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_39/zeros/ConstЕ
forward_lstm_39/zerosFill%forward_lstm_39/zeros/packed:output:0$forward_lstm_39/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/zeros
forward_lstm_39/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_39/zeros_1/mul/yВ
forward_lstm_39/zeros_1/mulMul&forward_lstm_39/strided_slice:output:0&forward_lstm_39/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros_1/mul
forward_lstm_39/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
forward_lstm_39/zeros_1/Less/yЏ
forward_lstm_39/zeros_1/LessLessforward_lstm_39/zeros_1/mul:z:0'forward_lstm_39/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros_1/Less
 forward_lstm_39/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_39/zeros_1/packed/1Щ
forward_lstm_39/zeros_1/packedPack&forward_lstm_39/strided_slice:output:0)forward_lstm_39/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_39/zeros_1/packed
forward_lstm_39/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_39/zeros_1/ConstН
forward_lstm_39/zeros_1Fill'forward_lstm_39/zeros_1/packed:output:0&forward_lstm_39/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/zeros_1
forward_lstm_39/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_39/transpose/permщ
forward_lstm_39/transpose	Transpose<forward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0'forward_lstm_39/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_39/transpose
forward_lstm_39/Shape_1Shapeforward_lstm_39/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_39/Shape_1
%forward_lstm_39/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_39/strided_slice_1/stack
'forward_lstm_39/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_1/stack_1
'forward_lstm_39/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_1/stack_2Ю
forward_lstm_39/strided_slice_1StridedSlice forward_lstm_39/Shape_1:output:0.forward_lstm_39/strided_slice_1/stack:output:00forward_lstm_39/strided_slice_1/stack_1:output:00forward_lstm_39/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_39/strided_slice_1Ѕ
+forward_lstm_39/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+forward_lstm_39/TensorArrayV2/element_shapeђ
forward_lstm_39/TensorArrayV2TensorListReserve4forward_lstm_39/TensorArrayV2/element_shape:output:0(forward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_39/TensorArrayV2п
Eforward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2G
Eforward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
7forward_lstm_39/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_39/transpose:y:0Nforward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_39/TensorArrayUnstack/TensorListFromTensor
%forward_lstm_39/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_39/strided_slice_2/stack
'forward_lstm_39/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_2/stack_1
'forward_lstm_39/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_2/stack_2м
forward_lstm_39/strided_slice_2StridedSliceforward_lstm_39/transpose:y:0.forward_lstm_39/strided_slice_2/stack:output:00forward_lstm_39/strided_slice_2/stack_1:output:00forward_lstm_39/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2!
forward_lstm_39/strided_slice_2ш
3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp<forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype025
3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp№
$forward_lstm_39/lstm_cell_118/MatMulMatMul(forward_lstm_39/strided_slice_2:output:0;forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$forward_lstm_39/lstm_cell_118/MatMulю
5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp>forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype027
5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpь
&forward_lstm_39/lstm_cell_118/MatMul_1MatMulforward_lstm_39/zeros:output:0=forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_39/lstm_cell_118/MatMul_1ф
!forward_lstm_39/lstm_cell_118/addAddV2.forward_lstm_39/lstm_cell_118/MatMul:product:00forward_lstm_39/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2#
!forward_lstm_39/lstm_cell_118/addч
4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp=forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype026
4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpё
%forward_lstm_39/lstm_cell_118/BiasAddBiasAdd%forward_lstm_39/lstm_cell_118/add:z:0<forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_39/lstm_cell_118/BiasAdd 
-forward_lstm_39/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-forward_lstm_39/lstm_cell_118/split/split_dimЗ
#forward_lstm_39/lstm_cell_118/splitSplit6forward_lstm_39/lstm_cell_118/split/split_dim:output:0.forward_lstm_39/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2%
#forward_lstm_39/lstm_cell_118/splitЙ
%forward_lstm_39/lstm_cell_118/SigmoidSigmoid,forward_lstm_39/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_39/lstm_cell_118/SigmoidН
'forward_lstm_39/lstm_cell_118/Sigmoid_1Sigmoid,forward_lstm_39/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_39/lstm_cell_118/Sigmoid_1Ю
!forward_lstm_39/lstm_cell_118/mulMul+forward_lstm_39/lstm_cell_118/Sigmoid_1:y:0 forward_lstm_39/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_39/lstm_cell_118/mulА
"forward_lstm_39/lstm_cell_118/ReluRelu,forward_lstm_39/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_39/lstm_cell_118/Reluр
#forward_lstm_39/lstm_cell_118/mul_1Mul)forward_lstm_39/lstm_cell_118/Sigmoid:y:00forward_lstm_39/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_39/lstm_cell_118/mul_1е
#forward_lstm_39/lstm_cell_118/add_1AddV2%forward_lstm_39/lstm_cell_118/mul:z:0'forward_lstm_39/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_39/lstm_cell_118/add_1Н
'forward_lstm_39/lstm_cell_118/Sigmoid_2Sigmoid,forward_lstm_39/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_39/lstm_cell_118/Sigmoid_2Џ
$forward_lstm_39/lstm_cell_118/Relu_1Relu'forward_lstm_39/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_39/lstm_cell_118/Relu_1ф
#forward_lstm_39/lstm_cell_118/mul_2Mul+forward_lstm_39/lstm_cell_118/Sigmoid_2:y:02forward_lstm_39/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_39/lstm_cell_118/mul_2Џ
-forward_lstm_39/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2/
-forward_lstm_39/TensorArrayV2_1/element_shapeј
forward_lstm_39/TensorArrayV2_1TensorListReserve6forward_lstm_39/TensorArrayV2_1/element_shape:output:0(forward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_39/TensorArrayV2_1n
forward_lstm_39/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_39/time 
forward_lstm_39/zeros_like	ZerosLike'forward_lstm_39/lstm_cell_118/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/zeros_like
(forward_lstm_39/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(forward_lstm_39/while/maximum_iterations
"forward_lstm_39/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_39/while/loop_counter	
forward_lstm_39/whileWhile+forward_lstm_39/while/loop_counter:output:01forward_lstm_39/while/maximum_iterations:output:0forward_lstm_39/time:output:0(forward_lstm_39/TensorArrayV2_1:handle:0forward_lstm_39/zeros_like:y:0forward_lstm_39/zeros:output:0 forward_lstm_39/zeros_1:output:0(forward_lstm_39/strided_slice_1:output:0Gforward_lstm_39/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_39/Cast:y:0<forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource>forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource=forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *.
body&R$
"forward_lstm_39_while_body_4931874*.
cond&R$
"forward_lstm_39_while_cond_4931873*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_39/whileе
@forward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2B
@forward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeБ
2forward_lstm_39/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_39/while:output:3Iforward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype024
2forward_lstm_39/TensorArrayV2Stack/TensorListStackЁ
%forward_lstm_39/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2'
%forward_lstm_39/strided_slice_3/stack
'forward_lstm_39/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_39/strided_slice_3/stack_1
'forward_lstm_39/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_3/stack_2њ
forward_lstm_39/strided_slice_3StridedSlice;forward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_39/strided_slice_3/stack:output:00forward_lstm_39/strided_slice_3/stack_1:output:00forward_lstm_39/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2!
forward_lstm_39/strided_slice_3
 forward_lstm_39/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_39/transpose_1/permю
forward_lstm_39/transpose_1	Transpose;forward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_39/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_39/transpose_1
forward_lstm_39/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_39/runtime
%backward_lstm_39/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%backward_lstm_39/RaggedToTensor/zeros
%backward_lstm_39/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2'
%backward_lstm_39/RaggedToTensor/Const
4backward_lstm_39/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.backward_lstm_39/RaggedToTensor/Const:output:0inputs.backward_lstm_39/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4backward_lstm_39/RaggedToTensor/RaggedTensorToTensorФ
;backward_lstm_39/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;backward_lstm_39/RaggedNestedRowLengths/strided_slice/stackШ
=backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1Ш
=backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2Љ
5backward_lstm_39/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dbackward_lstm_39/RaggedNestedRowLengths/strided_slice/stack:output:0Fbackward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fbackward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask27
5backward_lstm_39/RaggedNestedRowLengths/strided_sliceШ
=backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stackе
?backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2A
?backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1Ь
?backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2Е
7backward_lstm_39/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fbackward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hbackward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hbackward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask29
7backward_lstm_39/RaggedNestedRowLengths/strided_slice_1
+backward_lstm_39/RaggedNestedRowLengths/subSub>backward_lstm_39/RaggedNestedRowLengths/strided_slice:output:0@backward_lstm_39/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2-
+backward_lstm_39/RaggedNestedRowLengths/subЄ
backward_lstm_39/CastCast/backward_lstm_39/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_39/Cast
backward_lstm_39/ShapeShape=backward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_39/Shape
$backward_lstm_39/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_39/strided_slice/stack
&backward_lstm_39/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_39/strided_slice/stack_1
&backward_lstm_39/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_39/strided_slice/stack_2Ш
backward_lstm_39/strided_sliceStridedSlicebackward_lstm_39/Shape:output:0-backward_lstm_39/strided_slice/stack:output:0/backward_lstm_39/strided_slice/stack_1:output:0/backward_lstm_39/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_39/strided_slice~
backward_lstm_39/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_39/zeros/mul/yА
backward_lstm_39/zeros/mulMul'backward_lstm_39/strided_slice:output:0%backward_lstm_39/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros/mul
backward_lstm_39/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
backward_lstm_39/zeros/Less/yЋ
backward_lstm_39/zeros/LessLessbackward_lstm_39/zeros/mul:z:0&backward_lstm_39/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros/Less
backward_lstm_39/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_39/zeros/packed/1Ч
backward_lstm_39/zeros/packedPack'backward_lstm_39/strided_slice:output:0(backward_lstm_39/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_39/zeros/packed
backward_lstm_39/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_39/zeros/ConstЙ
backward_lstm_39/zerosFill&backward_lstm_39/zeros/packed:output:0%backward_lstm_39/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/zeros
backward_lstm_39/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_39/zeros_1/mul/yЖ
backward_lstm_39/zeros_1/mulMul'backward_lstm_39/strided_slice:output:0'backward_lstm_39/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros_1/mul
backward_lstm_39/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
backward_lstm_39/zeros_1/Less/yГ
backward_lstm_39/zeros_1/LessLess backward_lstm_39/zeros_1/mul:z:0(backward_lstm_39/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros_1/Less
!backward_lstm_39/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_39/zeros_1/packed/1Э
backward_lstm_39/zeros_1/packedPack'backward_lstm_39/strided_slice:output:0*backward_lstm_39/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_39/zeros_1/packed
backward_lstm_39/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_39/zeros_1/ConstС
backward_lstm_39/zeros_1Fill(backward_lstm_39/zeros_1/packed:output:0'backward_lstm_39/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/zeros_1
backward_lstm_39/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_39/transpose/permэ
backward_lstm_39/transpose	Transpose=backward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0(backward_lstm_39/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_39/transpose
backward_lstm_39/Shape_1Shapebackward_lstm_39/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_39/Shape_1
&backward_lstm_39/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_39/strided_slice_1/stack
(backward_lstm_39/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_1/stack_1
(backward_lstm_39/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_1/stack_2д
 backward_lstm_39/strided_slice_1StridedSlice!backward_lstm_39/Shape_1:output:0/backward_lstm_39/strided_slice_1/stack:output:01backward_lstm_39/strided_slice_1/stack_1:output:01backward_lstm_39/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_39/strided_slice_1Ї
,backward_lstm_39/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,backward_lstm_39/TensorArrayV2/element_shapeі
backward_lstm_39/TensorArrayV2TensorListReserve5backward_lstm_39/TensorArrayV2/element_shape:output:0)backward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_39/TensorArrayV2
backward_lstm_39/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_39/ReverseV2/axisЮ
backward_lstm_39/ReverseV2	ReverseV2backward_lstm_39/transpose:y:0(backward_lstm_39/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_39/ReverseV2с
Fbackward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fbackward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeС
8backward_lstm_39/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_39/ReverseV2:output:0Obackward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_39/TensorArrayUnstack/TensorListFromTensor
&backward_lstm_39/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_39/strided_slice_2/stack
(backward_lstm_39/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_2/stack_1
(backward_lstm_39/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_2/stack_2т
 backward_lstm_39/strided_slice_2StridedSlicebackward_lstm_39/transpose:y:0/backward_lstm_39/strided_slice_2/stack:output:01backward_lstm_39/strided_slice_2/stack_1:output:01backward_lstm_39/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 backward_lstm_39/strided_slice_2ы
4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpReadVariableOp=backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpє
%backward_lstm_39/lstm_cell_119/MatMulMatMul)backward_lstm_39/strided_slice_2:output:0<backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%backward_lstm_39/lstm_cell_119/MatMulё
6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOp?backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp№
'backward_lstm_39/lstm_cell_119/MatMul_1MatMulbackward_lstm_39/zeros:output:0>backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_39/lstm_cell_119/MatMul_1ш
"backward_lstm_39/lstm_cell_119/addAddV2/backward_lstm_39/lstm_cell_119/MatMul:product:01backward_lstm_39/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"backward_lstm_39/lstm_cell_119/addъ
5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOp>backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpѕ
&backward_lstm_39/lstm_cell_119/BiasAddBiasAdd&backward_lstm_39/lstm_cell_119/add:z:0=backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_39/lstm_cell_119/BiasAddЂ
.backward_lstm_39/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.backward_lstm_39/lstm_cell_119/split/split_dimЛ
$backward_lstm_39/lstm_cell_119/splitSplit7backward_lstm_39/lstm_cell_119/split/split_dim:output:0/backward_lstm_39/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$backward_lstm_39/lstm_cell_119/splitМ
&backward_lstm_39/lstm_cell_119/SigmoidSigmoid-backward_lstm_39/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_39/lstm_cell_119/SigmoidР
(backward_lstm_39/lstm_cell_119/Sigmoid_1Sigmoid-backward_lstm_39/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_39/lstm_cell_119/Sigmoid_1в
"backward_lstm_39/lstm_cell_119/mulMul,backward_lstm_39/lstm_cell_119/Sigmoid_1:y:0!backward_lstm_39/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_39/lstm_cell_119/mulГ
#backward_lstm_39/lstm_cell_119/ReluRelu-backward_lstm_39/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_39/lstm_cell_119/Reluф
$backward_lstm_39/lstm_cell_119/mul_1Mul*backward_lstm_39/lstm_cell_119/Sigmoid:y:01backward_lstm_39/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_39/lstm_cell_119/mul_1й
$backward_lstm_39/lstm_cell_119/add_1AddV2&backward_lstm_39/lstm_cell_119/mul:z:0(backward_lstm_39/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_39/lstm_cell_119/add_1Р
(backward_lstm_39/lstm_cell_119/Sigmoid_2Sigmoid-backward_lstm_39/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_39/lstm_cell_119/Sigmoid_2В
%backward_lstm_39/lstm_cell_119/Relu_1Relu(backward_lstm_39/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_39/lstm_cell_119/Relu_1ш
$backward_lstm_39/lstm_cell_119/mul_2Mul,backward_lstm_39/lstm_cell_119/Sigmoid_2:y:03backward_lstm_39/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_39/lstm_cell_119/mul_2Б
.backward_lstm_39/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.backward_lstm_39/TensorArrayV2_1/element_shapeќ
 backward_lstm_39/TensorArrayV2_1TensorListReserve7backward_lstm_39/TensorArrayV2_1/element_shape:output:0)backward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_39/TensorArrayV2_1p
backward_lstm_39/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_39/time
&backward_lstm_39/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_lstm_39/Max/reduction_indices 
backward_lstm_39/MaxMaxbackward_lstm_39/Cast:y:0/backward_lstm_39/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/Maxr
backward_lstm_39/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_39/sub/y
backward_lstm_39/subSubbackward_lstm_39/Max:output:0backward_lstm_39/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/sub
backward_lstm_39/Sub_1Subbackward_lstm_39/sub:z:0backward_lstm_39/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_39/Sub_1Ѓ
backward_lstm_39/zeros_like	ZerosLike(backward_lstm_39/lstm_cell_119/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/zeros_likeЁ
)backward_lstm_39/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)backward_lstm_39/while/maximum_iterations
#backward_lstm_39/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_39/while/loop_counter	
backward_lstm_39/whileWhile,backward_lstm_39/while/loop_counter:output:02backward_lstm_39/while/maximum_iterations:output:0backward_lstm_39/time:output:0)backward_lstm_39/TensorArrayV2_1:handle:0backward_lstm_39/zeros_like:y:0backward_lstm_39/zeros:output:0!backward_lstm_39/zeros_1:output:0)backward_lstm_39/strided_slice_1:output:0Hbackward_lstm_39/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_39/Sub_1:z:0=backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource?backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource>backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( */
body'R%
#backward_lstm_39_while_body_4932053*/
cond'R%
#backward_lstm_39_while_cond_4932052*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_39/whileз
Abackward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Abackward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3backward_lstm_39/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_39/while:output:3Jbackward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3backward_lstm_39/TensorArrayV2Stack/TensorListStackЃ
&backward_lstm_39/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&backward_lstm_39/strided_slice_3/stack
(backward_lstm_39/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_39/strided_slice_3/stack_1
(backward_lstm_39/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_3/stack_2
 backward_lstm_39/strided_slice_3StridedSlice<backward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_39/strided_slice_3/stack:output:01backward_lstm_39/strided_slice_3/stack_1:output:01backward_lstm_39/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 backward_lstm_39/strided_slice_3
!backward_lstm_39/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_39/transpose_1/permђ
backward_lstm_39/transpose_1	Transpose<backward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_39/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_39/transpose_1
backward_lstm_39/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_39/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisТ
concatConcatV2(forward_lstm_39/strided_slice_3:output:0)backward_lstm_39/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

IdentityЬ
NoOpNoOp6^backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp5^backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp7^backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp^backward_lstm_39/while5^forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp4^forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp6^forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp^forward_lstm_39/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2n
5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp2l
4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp2p
6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp20
backward_lstm_39/whilebackward_lstm_39/while2l
4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp2j
3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp2n
5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp2.
forward_lstm_39/whileforward_lstm_39/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
 е
д
"__inference__wrapped_model_4929274

args_0
args_0_1	n
[sequential_39_bidirectional_39_forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource:	Шp
]sequential_39_bidirectional_39_forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource:	2Шk
\sequential_39_bidirectional_39_forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource:	Шo
\sequential_39_bidirectional_39_backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource:	Шq
^sequential_39_bidirectional_39_backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource:	2Шl
]sequential_39_bidirectional_39_backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource:	ШG
5sequential_39_dense_39_matmul_readvariableop_resource:dD
6sequential_39_dense_39_biasadd_readvariableop_resource:
identityЂTsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpЂSsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpЂUsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOpЂ5sequential_39/bidirectional_39/backward_lstm_39/whileЂSsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpЂRsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOpЂTsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpЂ4sequential_39/bidirectional_39/forward_lstm_39/whileЂ-sequential_39/dense_39/BiasAdd/ReadVariableOpЂ,sequential_39/dense_39/MatMul/ReadVariableOpг
Csequential_39/bidirectional_39/forward_lstm_39/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2E
Csequential_39/bidirectional_39/forward_lstm_39/RaggedToTensor/zerosе
Csequential_39/bidirectional_39/forward_lstm_39/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2E
Csequential_39/bidirectional_39/forward_lstm_39/RaggedToTensor/Const
Rsequential_39/bidirectional_39/forward_lstm_39/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorLsequential_39/bidirectional_39/forward_lstm_39/RaggedToTensor/Const:output:0args_0Lsequential_39/bidirectional_39/forward_lstm_39/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2T
Rsequential_39/bidirectional_39/forward_lstm_39/RaggedToTensor/RaggedTensorToTensor
Ysequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2[
Ysequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack
[sequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2]
[sequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1
[sequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2]
[sequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2П
Ssequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1bsequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack:output:0dsequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1:output:0dsequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask2U
Ssequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice
[sequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2]
[sequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack
]sequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2_
]sequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1
]sequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2_
]sequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2Ы
Usequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1dsequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack:output:0fsequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0fsequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2W
Usequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice_1
Isequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/subSub\sequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice:output:0^sequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2K
Isequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/subў
3sequential_39/bidirectional_39/forward_lstm_39/CastCastMsequential_39/bidirectional_39/forward_lstm_39/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ25
3sequential_39/bidirectional_39/forward_lstm_39/Castї
4sequential_39/bidirectional_39/forward_lstm_39/ShapeShape[sequential_39/bidirectional_39/forward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:26
4sequential_39/bidirectional_39/forward_lstm_39/Shapeв
Bsequential_39/bidirectional_39/forward_lstm_39/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_39/bidirectional_39/forward_lstm_39/strided_slice/stackж
Dsequential_39/bidirectional_39/forward_lstm_39/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_39/bidirectional_39/forward_lstm_39/strided_slice/stack_1ж
Dsequential_39/bidirectional_39/forward_lstm_39/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_39/bidirectional_39/forward_lstm_39/strided_slice/stack_2ќ
<sequential_39/bidirectional_39/forward_lstm_39/strided_sliceStridedSlice=sequential_39/bidirectional_39/forward_lstm_39/Shape:output:0Ksequential_39/bidirectional_39/forward_lstm_39/strided_slice/stack:output:0Msequential_39/bidirectional_39/forward_lstm_39/strided_slice/stack_1:output:0Msequential_39/bidirectional_39/forward_lstm_39/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_39/bidirectional_39/forward_lstm_39/strided_sliceК
:sequential_39/bidirectional_39/forward_lstm_39/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22<
:sequential_39/bidirectional_39/forward_lstm_39/zeros/mul/yЈ
8sequential_39/bidirectional_39/forward_lstm_39/zeros/mulMulEsequential_39/bidirectional_39/forward_lstm_39/strided_slice:output:0Csequential_39/bidirectional_39/forward_lstm_39/zeros/mul/y:output:0*
T0*
_output_shapes
: 2:
8sequential_39/bidirectional_39/forward_lstm_39/zeros/mulН
;sequential_39/bidirectional_39/forward_lstm_39/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2=
;sequential_39/bidirectional_39/forward_lstm_39/zeros/Less/yЃ
9sequential_39/bidirectional_39/forward_lstm_39/zeros/LessLess<sequential_39/bidirectional_39/forward_lstm_39/zeros/mul:z:0Dsequential_39/bidirectional_39/forward_lstm_39/zeros/Less/y:output:0*
T0*
_output_shapes
: 2;
9sequential_39/bidirectional_39/forward_lstm_39/zeros/LessР
=sequential_39/bidirectional_39/forward_lstm_39/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22?
=sequential_39/bidirectional_39/forward_lstm_39/zeros/packed/1П
;sequential_39/bidirectional_39/forward_lstm_39/zeros/packedPackEsequential_39/bidirectional_39/forward_lstm_39/strided_slice:output:0Fsequential_39/bidirectional_39/forward_lstm_39/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2=
;sequential_39/bidirectional_39/forward_lstm_39/zeros/packedС
:sequential_39/bidirectional_39/forward_lstm_39/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2<
:sequential_39/bidirectional_39/forward_lstm_39/zeros/ConstБ
4sequential_39/bidirectional_39/forward_lstm_39/zerosFillDsequential_39/bidirectional_39/forward_lstm_39/zeros/packed:output:0Csequential_39/bidirectional_39/forward_lstm_39/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ226
4sequential_39/bidirectional_39/forward_lstm_39/zerosО
<sequential_39/bidirectional_39/forward_lstm_39/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22>
<sequential_39/bidirectional_39/forward_lstm_39/zeros_1/mul/yЎ
:sequential_39/bidirectional_39/forward_lstm_39/zeros_1/mulMulEsequential_39/bidirectional_39/forward_lstm_39/strided_slice:output:0Esequential_39/bidirectional_39/forward_lstm_39/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2<
:sequential_39/bidirectional_39/forward_lstm_39/zeros_1/mulС
=sequential_39/bidirectional_39/forward_lstm_39/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2?
=sequential_39/bidirectional_39/forward_lstm_39/zeros_1/Less/yЋ
;sequential_39/bidirectional_39/forward_lstm_39/zeros_1/LessLess>sequential_39/bidirectional_39/forward_lstm_39/zeros_1/mul:z:0Fsequential_39/bidirectional_39/forward_lstm_39/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2=
;sequential_39/bidirectional_39/forward_lstm_39/zeros_1/LessФ
?sequential_39/bidirectional_39/forward_lstm_39/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22A
?sequential_39/bidirectional_39/forward_lstm_39/zeros_1/packed/1Х
=sequential_39/bidirectional_39/forward_lstm_39/zeros_1/packedPackEsequential_39/bidirectional_39/forward_lstm_39/strided_slice:output:0Hsequential_39/bidirectional_39/forward_lstm_39/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2?
=sequential_39/bidirectional_39/forward_lstm_39/zeros_1/packedХ
<sequential_39/bidirectional_39/forward_lstm_39/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2>
<sequential_39/bidirectional_39/forward_lstm_39/zeros_1/ConstЙ
6sequential_39/bidirectional_39/forward_lstm_39/zeros_1FillFsequential_39/bidirectional_39/forward_lstm_39/zeros_1/packed:output:0Esequential_39/bidirectional_39/forward_lstm_39/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ228
6sequential_39/bidirectional_39/forward_lstm_39/zeros_1г
=sequential_39/bidirectional_39/forward_lstm_39/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2?
=sequential_39/bidirectional_39/forward_lstm_39/transpose/permх
8sequential_39/bidirectional_39/forward_lstm_39/transpose	Transpose[sequential_39/bidirectional_39/forward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0Fsequential_39/bidirectional_39/forward_lstm_39/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2:
8sequential_39/bidirectional_39/forward_lstm_39/transposeм
6sequential_39/bidirectional_39/forward_lstm_39/Shape_1Shape<sequential_39/bidirectional_39/forward_lstm_39/transpose:y:0*
T0*
_output_shapes
:28
6sequential_39/bidirectional_39/forward_lstm_39/Shape_1ж
Dsequential_39/bidirectional_39/forward_lstm_39/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dsequential_39/bidirectional_39/forward_lstm_39/strided_slice_1/stackк
Fsequential_39/bidirectional_39/forward_lstm_39/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_39/bidirectional_39/forward_lstm_39/strided_slice_1/stack_1к
Fsequential_39/bidirectional_39/forward_lstm_39/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_39/bidirectional_39/forward_lstm_39/strided_slice_1/stack_2
>sequential_39/bidirectional_39/forward_lstm_39/strided_slice_1StridedSlice?sequential_39/bidirectional_39/forward_lstm_39/Shape_1:output:0Msequential_39/bidirectional_39/forward_lstm_39/strided_slice_1/stack:output:0Osequential_39/bidirectional_39/forward_lstm_39/strided_slice_1/stack_1:output:0Osequential_39/bidirectional_39/forward_lstm_39/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2@
>sequential_39/bidirectional_39/forward_lstm_39/strided_slice_1у
Jsequential_39/bidirectional_39/forward_lstm_39/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2L
Jsequential_39/bidirectional_39/forward_lstm_39/TensorArrayV2/element_shapeю
<sequential_39/bidirectional_39/forward_lstm_39/TensorArrayV2TensorListReserveSsequential_39/bidirectional_39/forward_lstm_39/TensorArrayV2/element_shape:output:0Gsequential_39/bidirectional_39/forward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_39/bidirectional_39/forward_lstm_39/TensorArrayV2
dsequential_39/bidirectional_39/forward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2f
dsequential_39/bidirectional_39/forward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeД
Vsequential_39/bidirectional_39/forward_lstm_39/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor<sequential_39/bidirectional_39/forward_lstm_39/transpose:y:0msequential_39/bidirectional_39/forward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02X
Vsequential_39/bidirectional_39/forward_lstm_39/TensorArrayUnstack/TensorListFromTensorж
Dsequential_39/bidirectional_39/forward_lstm_39/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dsequential_39/bidirectional_39/forward_lstm_39/strided_slice_2/stackк
Fsequential_39/bidirectional_39/forward_lstm_39/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_39/bidirectional_39/forward_lstm_39/strided_slice_2/stack_1к
Fsequential_39/bidirectional_39/forward_lstm_39/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_39/bidirectional_39/forward_lstm_39/strided_slice_2/stack_2
>sequential_39/bidirectional_39/forward_lstm_39/strided_slice_2StridedSlice<sequential_39/bidirectional_39/forward_lstm_39/transpose:y:0Msequential_39/bidirectional_39/forward_lstm_39/strided_slice_2/stack:output:0Osequential_39/bidirectional_39/forward_lstm_39/strided_slice_2/stack_1:output:0Osequential_39/bidirectional_39/forward_lstm_39/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2@
>sequential_39/bidirectional_39/forward_lstm_39/strided_slice_2Х
Rsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp[sequential_39_bidirectional_39_forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02T
Rsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOpь
Csequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMulMatMulGsequential_39/bidirectional_39/forward_lstm_39/strided_slice_2:output:0Zsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2E
Csequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMulЫ
Tsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp]sequential_39_bidirectional_39_forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02V
Tsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpш
Esequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMul_1MatMul=sequential_39/bidirectional_39/forward_lstm_39/zeros:output:0\sequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2G
Esequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMul_1р
@sequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/addAddV2Msequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMul:product:0Osequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2B
@sequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/addФ
Ssequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp\sequential_39_bidirectional_39_forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02U
Ssequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpэ
Dsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/BiasAddBiasAddDsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/add:z:0[sequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2F
Dsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/BiasAddо
Lsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2N
Lsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/split/split_dimГ
Bsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/splitSplitUsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/split/split_dim:output:0Msequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2D
Bsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/split
Dsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/SigmoidSigmoidKsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22F
Dsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/Sigmoid
Fsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/Sigmoid_1SigmoidKsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22H
Fsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/Sigmoid_1Ъ
@sequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/mulMulJsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/Sigmoid_1:y:0?sequential_39/bidirectional_39/forward_lstm_39/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22B
@sequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/mul
Asequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/ReluReluKsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22C
Asequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/Reluм
Bsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/mul_1MulHsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/Sigmoid:y:0Osequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22D
Bsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/mul_1б
Bsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/add_1AddV2Dsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/mul:z:0Fsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22D
Bsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/add_1
Fsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/Sigmoid_2SigmoidKsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22H
Fsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/Sigmoid_2
Csequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/Relu_1ReluFsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/Relu_1р
Bsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/mul_2MulJsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/Sigmoid_2:y:0Qsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22D
Bsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/mul_2э
Lsequential_39/bidirectional_39/forward_lstm_39/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2N
Lsequential_39/bidirectional_39/forward_lstm_39/TensorArrayV2_1/element_shapeє
>sequential_39/bidirectional_39/forward_lstm_39/TensorArrayV2_1TensorListReserveUsequential_39/bidirectional_39/forward_lstm_39/TensorArrayV2_1/element_shape:output:0Gsequential_39/bidirectional_39/forward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02@
>sequential_39/bidirectional_39/forward_lstm_39/TensorArrayV2_1Ќ
3sequential_39/bidirectional_39/forward_lstm_39/timeConst*
_output_shapes
: *
dtype0*
value	B : 25
3sequential_39/bidirectional_39/forward_lstm_39/time§
9sequential_39/bidirectional_39/forward_lstm_39/zeros_like	ZerosLikeFsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22;
9sequential_39/bidirectional_39/forward_lstm_39/zeros_likeн
Gsequential_39/bidirectional_39/forward_lstm_39/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2I
Gsequential_39/bidirectional_39/forward_lstm_39/while/maximum_iterationsШ
Asequential_39/bidirectional_39/forward_lstm_39/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_39/bidirectional_39/forward_lstm_39/while/loop_counter
4sequential_39/bidirectional_39/forward_lstm_39/whileWhileJsequential_39/bidirectional_39/forward_lstm_39/while/loop_counter:output:0Psequential_39/bidirectional_39/forward_lstm_39/while/maximum_iterations:output:0<sequential_39/bidirectional_39/forward_lstm_39/time:output:0Gsequential_39/bidirectional_39/forward_lstm_39/TensorArrayV2_1:handle:0=sequential_39/bidirectional_39/forward_lstm_39/zeros_like:y:0=sequential_39/bidirectional_39/forward_lstm_39/zeros:output:0?sequential_39/bidirectional_39/forward_lstm_39/zeros_1:output:0Gsequential_39/bidirectional_39/forward_lstm_39/strided_slice_1:output:0fsequential_39/bidirectional_39/forward_lstm_39/TensorArrayUnstack/TensorListFromTensor:output_handle:07sequential_39/bidirectional_39/forward_lstm_39/Cast:y:0[sequential_39_bidirectional_39_forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource]sequential_39_bidirectional_39_forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource\sequential_39_bidirectional_39_forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *M
bodyERC
Asequential_39_bidirectional_39_forward_lstm_39_while_body_4928991*M
condERC
Asequential_39_bidirectional_39_forward_lstm_39_while_cond_4928990*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 26
4sequential_39/bidirectional_39/forward_lstm_39/while
_sequential_39/bidirectional_39/forward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2a
_sequential_39/bidirectional_39/forward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shape­
Qsequential_39/bidirectional_39/forward_lstm_39/TensorArrayV2Stack/TensorListStackTensorListStack=sequential_39/bidirectional_39/forward_lstm_39/while:output:3hsequential_39/bidirectional_39/forward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02S
Qsequential_39/bidirectional_39/forward_lstm_39/TensorArrayV2Stack/TensorListStackп
Dsequential_39/bidirectional_39/forward_lstm_39/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2F
Dsequential_39/bidirectional_39/forward_lstm_39/strided_slice_3/stackк
Fsequential_39/bidirectional_39/forward_lstm_39/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2H
Fsequential_39/bidirectional_39/forward_lstm_39/strided_slice_3/stack_1к
Fsequential_39/bidirectional_39/forward_lstm_39/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_39/bidirectional_39/forward_lstm_39/strided_slice_3/stack_2Д
>sequential_39/bidirectional_39/forward_lstm_39/strided_slice_3StridedSliceZsequential_39/bidirectional_39/forward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0Msequential_39/bidirectional_39/forward_lstm_39/strided_slice_3/stack:output:0Osequential_39/bidirectional_39/forward_lstm_39/strided_slice_3/stack_1:output:0Osequential_39/bidirectional_39/forward_lstm_39/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2@
>sequential_39/bidirectional_39/forward_lstm_39/strided_slice_3з
?sequential_39/bidirectional_39/forward_lstm_39/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2A
?sequential_39/bidirectional_39/forward_lstm_39/transpose_1/permъ
:sequential_39/bidirectional_39/forward_lstm_39/transpose_1	TransposeZsequential_39/bidirectional_39/forward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0Hsequential_39/bidirectional_39/forward_lstm_39/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22<
:sequential_39/bidirectional_39/forward_lstm_39/transpose_1Ф
6sequential_39/bidirectional_39/forward_lstm_39/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    28
6sequential_39/bidirectional_39/forward_lstm_39/runtimeе
Dsequential_39/bidirectional_39/backward_lstm_39/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2F
Dsequential_39/bidirectional_39/backward_lstm_39/RaggedToTensor/zerosз
Dsequential_39/bidirectional_39/backward_lstm_39/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2F
Dsequential_39/bidirectional_39/backward_lstm_39/RaggedToTensor/Const
Ssequential_39/bidirectional_39/backward_lstm_39/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorMsequential_39/bidirectional_39/backward_lstm_39/RaggedToTensor/Const:output:0args_0Msequential_39/bidirectional_39/backward_lstm_39/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2U
Ssequential_39/bidirectional_39/backward_lstm_39/RaggedToTensor/RaggedTensorToTensor
Zsequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2\
Zsequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack
\sequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2^
\sequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1
\sequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2^
\sequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2Ф
Tsequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1csequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack:output:0esequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1:output:0esequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask2V
Tsequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice
\sequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2^
\sequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack
^sequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2`
^sequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1
^sequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2`
^sequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2а
Vsequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1esequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack:output:0gsequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0gsequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2X
Vsequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice_1
Jsequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/subSub]sequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice:output:0_sequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2L
Jsequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/sub
4sequential_39/bidirectional_39/backward_lstm_39/CastCastNsequential_39/bidirectional_39/backward_lstm_39/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ26
4sequential_39/bidirectional_39/backward_lstm_39/Castњ
5sequential_39/bidirectional_39/backward_lstm_39/ShapeShape\sequential_39/bidirectional_39/backward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:27
5sequential_39/bidirectional_39/backward_lstm_39/Shapeд
Csequential_39/bidirectional_39/backward_lstm_39/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Csequential_39/bidirectional_39/backward_lstm_39/strided_slice/stackи
Esequential_39/bidirectional_39/backward_lstm_39/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_39/bidirectional_39/backward_lstm_39/strided_slice/stack_1и
Esequential_39/bidirectional_39/backward_lstm_39/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_39/bidirectional_39/backward_lstm_39/strided_slice/stack_2
=sequential_39/bidirectional_39/backward_lstm_39/strided_sliceStridedSlice>sequential_39/bidirectional_39/backward_lstm_39/Shape:output:0Lsequential_39/bidirectional_39/backward_lstm_39/strided_slice/stack:output:0Nsequential_39/bidirectional_39/backward_lstm_39/strided_slice/stack_1:output:0Nsequential_39/bidirectional_39/backward_lstm_39/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=sequential_39/bidirectional_39/backward_lstm_39/strided_sliceМ
;sequential_39/bidirectional_39/backward_lstm_39/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22=
;sequential_39/bidirectional_39/backward_lstm_39/zeros/mul/yЌ
9sequential_39/bidirectional_39/backward_lstm_39/zeros/mulMulFsequential_39/bidirectional_39/backward_lstm_39/strided_slice:output:0Dsequential_39/bidirectional_39/backward_lstm_39/zeros/mul/y:output:0*
T0*
_output_shapes
: 2;
9sequential_39/bidirectional_39/backward_lstm_39/zeros/mulП
<sequential_39/bidirectional_39/backward_lstm_39/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2>
<sequential_39/bidirectional_39/backward_lstm_39/zeros/Less/yЇ
:sequential_39/bidirectional_39/backward_lstm_39/zeros/LessLess=sequential_39/bidirectional_39/backward_lstm_39/zeros/mul:z:0Esequential_39/bidirectional_39/backward_lstm_39/zeros/Less/y:output:0*
T0*
_output_shapes
: 2<
:sequential_39/bidirectional_39/backward_lstm_39/zeros/LessТ
>sequential_39/bidirectional_39/backward_lstm_39/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22@
>sequential_39/bidirectional_39/backward_lstm_39/zeros/packed/1У
<sequential_39/bidirectional_39/backward_lstm_39/zeros/packedPackFsequential_39/bidirectional_39/backward_lstm_39/strided_slice:output:0Gsequential_39/bidirectional_39/backward_lstm_39/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2>
<sequential_39/bidirectional_39/backward_lstm_39/zeros/packedУ
;sequential_39/bidirectional_39/backward_lstm_39/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2=
;sequential_39/bidirectional_39/backward_lstm_39/zeros/ConstЕ
5sequential_39/bidirectional_39/backward_lstm_39/zerosFillEsequential_39/bidirectional_39/backward_lstm_39/zeros/packed:output:0Dsequential_39/bidirectional_39/backward_lstm_39/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ227
5sequential_39/bidirectional_39/backward_lstm_39/zerosР
=sequential_39/bidirectional_39/backward_lstm_39/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22?
=sequential_39/bidirectional_39/backward_lstm_39/zeros_1/mul/yВ
;sequential_39/bidirectional_39/backward_lstm_39/zeros_1/mulMulFsequential_39/bidirectional_39/backward_lstm_39/strided_slice:output:0Fsequential_39/bidirectional_39/backward_lstm_39/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2=
;sequential_39/bidirectional_39/backward_lstm_39/zeros_1/mulУ
>sequential_39/bidirectional_39/backward_lstm_39/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2@
>sequential_39/bidirectional_39/backward_lstm_39/zeros_1/Less/yЏ
<sequential_39/bidirectional_39/backward_lstm_39/zeros_1/LessLess?sequential_39/bidirectional_39/backward_lstm_39/zeros_1/mul:z:0Gsequential_39/bidirectional_39/backward_lstm_39/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2>
<sequential_39/bidirectional_39/backward_lstm_39/zeros_1/LessЦ
@sequential_39/bidirectional_39/backward_lstm_39/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22B
@sequential_39/bidirectional_39/backward_lstm_39/zeros_1/packed/1Щ
>sequential_39/bidirectional_39/backward_lstm_39/zeros_1/packedPackFsequential_39/bidirectional_39/backward_lstm_39/strided_slice:output:0Isequential_39/bidirectional_39/backward_lstm_39/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2@
>sequential_39/bidirectional_39/backward_lstm_39/zeros_1/packedЧ
=sequential_39/bidirectional_39/backward_lstm_39/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2?
=sequential_39/bidirectional_39/backward_lstm_39/zeros_1/ConstН
7sequential_39/bidirectional_39/backward_lstm_39/zeros_1FillGsequential_39/bidirectional_39/backward_lstm_39/zeros_1/packed:output:0Fsequential_39/bidirectional_39/backward_lstm_39/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ229
7sequential_39/bidirectional_39/backward_lstm_39/zeros_1е
>sequential_39/bidirectional_39/backward_lstm_39/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2@
>sequential_39/bidirectional_39/backward_lstm_39/transpose/permщ
9sequential_39/bidirectional_39/backward_lstm_39/transpose	Transpose\sequential_39/bidirectional_39/backward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0Gsequential_39/bidirectional_39/backward_lstm_39/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2;
9sequential_39/bidirectional_39/backward_lstm_39/transposeп
7sequential_39/bidirectional_39/backward_lstm_39/Shape_1Shape=sequential_39/bidirectional_39/backward_lstm_39/transpose:y:0*
T0*
_output_shapes
:29
7sequential_39/bidirectional_39/backward_lstm_39/Shape_1и
Esequential_39/bidirectional_39/backward_lstm_39/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_39/bidirectional_39/backward_lstm_39/strided_slice_1/stackм
Gsequential_39/bidirectional_39/backward_lstm_39/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_39/bidirectional_39/backward_lstm_39/strided_slice_1/stack_1м
Gsequential_39/bidirectional_39/backward_lstm_39/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_39/bidirectional_39/backward_lstm_39/strided_slice_1/stack_2
?sequential_39/bidirectional_39/backward_lstm_39/strided_slice_1StridedSlice@sequential_39/bidirectional_39/backward_lstm_39/Shape_1:output:0Nsequential_39/bidirectional_39/backward_lstm_39/strided_slice_1/stack:output:0Psequential_39/bidirectional_39/backward_lstm_39/strided_slice_1/stack_1:output:0Psequential_39/bidirectional_39/backward_lstm_39/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?sequential_39/bidirectional_39/backward_lstm_39/strided_slice_1х
Ksequential_39/bidirectional_39/backward_lstm_39/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2M
Ksequential_39/bidirectional_39/backward_lstm_39/TensorArrayV2/element_shapeђ
=sequential_39/bidirectional_39/backward_lstm_39/TensorArrayV2TensorListReserveTsequential_39/bidirectional_39/backward_lstm_39/TensorArrayV2/element_shape:output:0Hsequential_39/bidirectional_39/backward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_39/bidirectional_39/backward_lstm_39/TensorArrayV2Ъ
>sequential_39/bidirectional_39/backward_lstm_39/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_39/bidirectional_39/backward_lstm_39/ReverseV2/axisЪ
9sequential_39/bidirectional_39/backward_lstm_39/ReverseV2	ReverseV2=sequential_39/bidirectional_39/backward_lstm_39/transpose:y:0Gsequential_39/bidirectional_39/backward_lstm_39/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2;
9sequential_39/bidirectional_39/backward_lstm_39/ReverseV2
esequential_39/bidirectional_39/backward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2g
esequential_39/bidirectional_39/backward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeН
Wsequential_39/bidirectional_39/backward_lstm_39/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorBsequential_39/bidirectional_39/backward_lstm_39/ReverseV2:output:0nsequential_39/bidirectional_39/backward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02Y
Wsequential_39/bidirectional_39/backward_lstm_39/TensorArrayUnstack/TensorListFromTensorи
Esequential_39/bidirectional_39/backward_lstm_39/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_39/bidirectional_39/backward_lstm_39/strided_slice_2/stackм
Gsequential_39/bidirectional_39/backward_lstm_39/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_39/bidirectional_39/backward_lstm_39/strided_slice_2/stack_1м
Gsequential_39/bidirectional_39/backward_lstm_39/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_39/bidirectional_39/backward_lstm_39/strided_slice_2/stack_2
?sequential_39/bidirectional_39/backward_lstm_39/strided_slice_2StridedSlice=sequential_39/bidirectional_39/backward_lstm_39/transpose:y:0Nsequential_39/bidirectional_39/backward_lstm_39/strided_slice_2/stack:output:0Psequential_39/bidirectional_39/backward_lstm_39/strided_slice_2/stack_1:output:0Psequential_39/bidirectional_39/backward_lstm_39/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2A
?sequential_39/bidirectional_39/backward_lstm_39/strided_slice_2Ш
Ssequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpReadVariableOp\sequential_39_bidirectional_39_backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02U
Ssequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp№
Dsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMulMatMulHsequential_39/bidirectional_39/backward_lstm_39/strided_slice_2:output:0[sequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2F
Dsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMulЮ
Usequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOp^sequential_39_bidirectional_39_backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02W
Usequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOpь
Fsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMul_1MatMul>sequential_39/bidirectional_39/backward_lstm_39/zeros:output:0]sequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2H
Fsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMul_1ф
Asequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/addAddV2Nsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMul:product:0Psequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2C
Asequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/addЧ
Tsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOp]sequential_39_bidirectional_39_backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02V
Tsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpё
Esequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/BiasAddBiasAddEsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/add:z:0\sequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2G
Esequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/BiasAddр
Msequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2O
Msequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/split/split_dimЗ
Csequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/splitSplitVsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/split/split_dim:output:0Nsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2E
Csequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/split
Esequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/SigmoidSigmoidLsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22G
Esequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/Sigmoid
Gsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/Sigmoid_1SigmoidLsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22I
Gsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/Sigmoid_1Ю
Asequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/mulMulKsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/Sigmoid_1:y:0@sequential_39/bidirectional_39/backward_lstm_39/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22C
Asequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/mul
Bsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/ReluReluLsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22D
Bsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/Reluр
Csequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/mul_1MulIsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/Sigmoid:y:0Psequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/mul_1е
Csequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/add_1AddV2Esequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/mul:z:0Gsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/add_1
Gsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/Sigmoid_2SigmoidLsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22I
Gsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/Sigmoid_2
Dsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/Relu_1ReluGsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22F
Dsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/Relu_1ф
Csequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/mul_2MulKsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/Sigmoid_2:y:0Rsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/mul_2я
Msequential_39/bidirectional_39/backward_lstm_39/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2O
Msequential_39/bidirectional_39/backward_lstm_39/TensorArrayV2_1/element_shapeј
?sequential_39/bidirectional_39/backward_lstm_39/TensorArrayV2_1TensorListReserveVsequential_39/bidirectional_39/backward_lstm_39/TensorArrayV2_1/element_shape:output:0Hsequential_39/bidirectional_39/backward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?sequential_39/bidirectional_39/backward_lstm_39/TensorArrayV2_1Ў
4sequential_39/bidirectional_39/backward_lstm_39/timeConst*
_output_shapes
: *
dtype0*
value	B : 26
4sequential_39/bidirectional_39/backward_lstm_39/timeа
Esequential_39/bidirectional_39/backward_lstm_39/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2G
Esequential_39/bidirectional_39/backward_lstm_39/Max/reduction_indices
3sequential_39/bidirectional_39/backward_lstm_39/MaxMax8sequential_39/bidirectional_39/backward_lstm_39/Cast:y:0Nsequential_39/bidirectional_39/backward_lstm_39/Max/reduction_indices:output:0*
T0*
_output_shapes
: 25
3sequential_39/bidirectional_39/backward_lstm_39/MaxА
5sequential_39/bidirectional_39/backward_lstm_39/sub/yConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_39/bidirectional_39/backward_lstm_39/sub/y
3sequential_39/bidirectional_39/backward_lstm_39/subSub<sequential_39/bidirectional_39/backward_lstm_39/Max:output:0>sequential_39/bidirectional_39/backward_lstm_39/sub/y:output:0*
T0*
_output_shapes
: 25
3sequential_39/bidirectional_39/backward_lstm_39/sub
5sequential_39/bidirectional_39/backward_lstm_39/Sub_1Sub7sequential_39/bidirectional_39/backward_lstm_39/sub:z:08sequential_39/bidirectional_39/backward_lstm_39/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ27
5sequential_39/bidirectional_39/backward_lstm_39/Sub_1
:sequential_39/bidirectional_39/backward_lstm_39/zeros_like	ZerosLikeGsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22<
:sequential_39/bidirectional_39/backward_lstm_39/zeros_likeп
Hsequential_39/bidirectional_39/backward_lstm_39/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2J
Hsequential_39/bidirectional_39/backward_lstm_39/while/maximum_iterationsЪ
Bsequential_39/bidirectional_39/backward_lstm_39/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bsequential_39/bidirectional_39/backward_lstm_39/while/loop_counterЁ
5sequential_39/bidirectional_39/backward_lstm_39/whileWhileKsequential_39/bidirectional_39/backward_lstm_39/while/loop_counter:output:0Qsequential_39/bidirectional_39/backward_lstm_39/while/maximum_iterations:output:0=sequential_39/bidirectional_39/backward_lstm_39/time:output:0Hsequential_39/bidirectional_39/backward_lstm_39/TensorArrayV2_1:handle:0>sequential_39/bidirectional_39/backward_lstm_39/zeros_like:y:0>sequential_39/bidirectional_39/backward_lstm_39/zeros:output:0@sequential_39/bidirectional_39/backward_lstm_39/zeros_1:output:0Hsequential_39/bidirectional_39/backward_lstm_39/strided_slice_1:output:0gsequential_39/bidirectional_39/backward_lstm_39/TensorArrayUnstack/TensorListFromTensor:output_handle:09sequential_39/bidirectional_39/backward_lstm_39/Sub_1:z:0\sequential_39_bidirectional_39_backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource^sequential_39_bidirectional_39_backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource]sequential_39_bidirectional_39_backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *N
bodyFRD
Bsequential_39_bidirectional_39_backward_lstm_39_while_body_4929170*N
condFRD
Bsequential_39_bidirectional_39_backward_lstm_39_while_cond_4929169*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 27
5sequential_39/bidirectional_39/backward_lstm_39/while
`sequential_39/bidirectional_39/backward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2b
`sequential_39/bidirectional_39/backward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeБ
Rsequential_39/bidirectional_39/backward_lstm_39/TensorArrayV2Stack/TensorListStackTensorListStack>sequential_39/bidirectional_39/backward_lstm_39/while:output:3isequential_39/bidirectional_39/backward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02T
Rsequential_39/bidirectional_39/backward_lstm_39/TensorArrayV2Stack/TensorListStackс
Esequential_39/bidirectional_39/backward_lstm_39/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2G
Esequential_39/bidirectional_39/backward_lstm_39/strided_slice_3/stackм
Gsequential_39/bidirectional_39/backward_lstm_39/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2I
Gsequential_39/bidirectional_39/backward_lstm_39/strided_slice_3/stack_1м
Gsequential_39/bidirectional_39/backward_lstm_39/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_39/bidirectional_39/backward_lstm_39/strided_slice_3/stack_2К
?sequential_39/bidirectional_39/backward_lstm_39/strided_slice_3StridedSlice[sequential_39/bidirectional_39/backward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0Nsequential_39/bidirectional_39/backward_lstm_39/strided_slice_3/stack:output:0Psequential_39/bidirectional_39/backward_lstm_39/strided_slice_3/stack_1:output:0Psequential_39/bidirectional_39/backward_lstm_39/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2A
?sequential_39/bidirectional_39/backward_lstm_39/strided_slice_3й
@sequential_39/bidirectional_39/backward_lstm_39/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2B
@sequential_39/bidirectional_39/backward_lstm_39/transpose_1/permю
;sequential_39/bidirectional_39/backward_lstm_39/transpose_1	Transpose[sequential_39/bidirectional_39/backward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0Isequential_39/bidirectional_39/backward_lstm_39/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22=
;sequential_39/bidirectional_39/backward_lstm_39/transpose_1Ц
7sequential_39/bidirectional_39/backward_lstm_39/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    29
7sequential_39/bidirectional_39/backward_lstm_39/runtime
*sequential_39/bidirectional_39/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2,
*sequential_39/bidirectional_39/concat/axisн
%sequential_39/bidirectional_39/concatConcatV2Gsequential_39/bidirectional_39/forward_lstm_39/strided_slice_3:output:0Hsequential_39/bidirectional_39/backward_lstm_39/strided_slice_3:output:03sequential_39/bidirectional_39/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2'
%sequential_39/bidirectional_39/concatв
,sequential_39/dense_39/MatMul/ReadVariableOpReadVariableOp5sequential_39_dense_39_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,sequential_39/dense_39/MatMul/ReadVariableOpр
sequential_39/dense_39/MatMulMatMul.sequential_39/bidirectional_39/concat:output:04sequential_39/dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_39/dense_39/MatMulб
-sequential_39/dense_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_39_dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_39/dense_39/BiasAdd/ReadVariableOpн
sequential_39/dense_39/BiasAddBiasAdd'sequential_39/dense_39/MatMul:product:05sequential_39/dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
sequential_39/dense_39/BiasAddІ
sequential_39/dense_39/SigmoidSigmoid'sequential_39/dense_39/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
sequential_39/dense_39/Sigmoid}
IdentityIdentity"sequential_39/dense_39/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityЃ
NoOpNoOpU^sequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpT^sequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpV^sequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp6^sequential_39/bidirectional_39/backward_lstm_39/whileT^sequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpS^sequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOpU^sequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp5^sequential_39/bidirectional_39/forward_lstm_39/while.^sequential_39/dense_39/BiasAdd/ReadVariableOp-^sequential_39/dense_39/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2Ќ
Tsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpTsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp2Њ
Ssequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpSsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp2Ў
Usequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOpUsequential_39/bidirectional_39/backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp2n
5sequential_39/bidirectional_39/backward_lstm_39/while5sequential_39/bidirectional_39/backward_lstm_39/while2Њ
Ssequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpSsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp2Ј
Rsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOpRsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp2Ќ
Tsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpTsequential_39/bidirectional_39/forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp2l
4sequential_39/bidirectional_39/forward_lstm_39/while4sequential_39/bidirectional_39/forward_lstm_39/while2^
-sequential_39/dense_39/BiasAdd/ReadVariableOp-sequential_39/dense_39/BiasAdd/ReadVariableOp2\
,sequential_39/dense_39/MatMul/ReadVariableOp,sequential_39/dense_39/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameargs_0:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameargs_0
ЛИ
п
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_4931710

inputs
inputs_1	O
<forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource:	ШQ
>forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource:	2ШL
=forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource:	ШP
=backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource:	ШR
?backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource:	2ШM
>backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource:	Ш
identityЂ5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpЂ4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpЂ6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOpЂbackward_lstm_39/whileЂ4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpЂ3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOpЂ5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpЂforward_lstm_39/while
$forward_lstm_39/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$forward_lstm_39/RaggedToTensor/zeros
$forward_lstm_39/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2&
$forward_lstm_39/RaggedToTensor/Const
3forward_lstm_39/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-forward_lstm_39/RaggedToTensor/Const:output:0inputs-forward_lstm_39/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3forward_lstm_39/RaggedToTensor/RaggedTensorToTensorТ
:forward_lstm_39/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:forward_lstm_39/RaggedNestedRowLengths/strided_slice/stackЦ
<forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1Ц
<forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2Є
4forward_lstm_39/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cforward_lstm_39/RaggedNestedRowLengths/strided_slice/stack:output:0Eforward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1:output:0Eforward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask26
4forward_lstm_39/RaggedNestedRowLengths/strided_sliceЦ
<forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stackг
>forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2@
>forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1Ъ
>forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2А
6forward_lstm_39/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Eforward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gforward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gforward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask28
6forward_lstm_39/RaggedNestedRowLengths/strided_slice_1
*forward_lstm_39/RaggedNestedRowLengths/subSub=forward_lstm_39/RaggedNestedRowLengths/strided_slice:output:0?forward_lstm_39/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2,
*forward_lstm_39/RaggedNestedRowLengths/subЁ
forward_lstm_39/CastCast.forward_lstm_39/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_39/Cast
forward_lstm_39/ShapeShape<forward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_39/Shape
#forward_lstm_39/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_39/strided_slice/stack
%forward_lstm_39/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_39/strided_slice/stack_1
%forward_lstm_39/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_39/strided_slice/stack_2Т
forward_lstm_39/strided_sliceStridedSliceforward_lstm_39/Shape:output:0,forward_lstm_39/strided_slice/stack:output:0.forward_lstm_39/strided_slice/stack_1:output:0.forward_lstm_39/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_39/strided_slice|
forward_lstm_39/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_39/zeros/mul/yЌ
forward_lstm_39/zeros/mulMul&forward_lstm_39/strided_slice:output:0$forward_lstm_39/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros/mul
forward_lstm_39/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_39/zeros/Less/yЇ
forward_lstm_39/zeros/LessLessforward_lstm_39/zeros/mul:z:0%forward_lstm_39/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros/Less
forward_lstm_39/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_39/zeros/packed/1У
forward_lstm_39/zeros/packedPack&forward_lstm_39/strided_slice:output:0'forward_lstm_39/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_39/zeros/packed
forward_lstm_39/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_39/zeros/ConstЕ
forward_lstm_39/zerosFill%forward_lstm_39/zeros/packed:output:0$forward_lstm_39/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/zeros
forward_lstm_39/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_39/zeros_1/mul/yВ
forward_lstm_39/zeros_1/mulMul&forward_lstm_39/strided_slice:output:0&forward_lstm_39/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros_1/mul
forward_lstm_39/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
forward_lstm_39/zeros_1/Less/yЏ
forward_lstm_39/zeros_1/LessLessforward_lstm_39/zeros_1/mul:z:0'forward_lstm_39/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros_1/Less
 forward_lstm_39/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_39/zeros_1/packed/1Щ
forward_lstm_39/zeros_1/packedPack&forward_lstm_39/strided_slice:output:0)forward_lstm_39/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_39/zeros_1/packed
forward_lstm_39/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_39/zeros_1/ConstН
forward_lstm_39/zeros_1Fill'forward_lstm_39/zeros_1/packed:output:0&forward_lstm_39/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/zeros_1
forward_lstm_39/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_39/transpose/permщ
forward_lstm_39/transpose	Transpose<forward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0'forward_lstm_39/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_39/transpose
forward_lstm_39/Shape_1Shapeforward_lstm_39/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_39/Shape_1
%forward_lstm_39/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_39/strided_slice_1/stack
'forward_lstm_39/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_1/stack_1
'forward_lstm_39/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_1/stack_2Ю
forward_lstm_39/strided_slice_1StridedSlice forward_lstm_39/Shape_1:output:0.forward_lstm_39/strided_slice_1/stack:output:00forward_lstm_39/strided_slice_1/stack_1:output:00forward_lstm_39/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_39/strided_slice_1Ѕ
+forward_lstm_39/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+forward_lstm_39/TensorArrayV2/element_shapeђ
forward_lstm_39/TensorArrayV2TensorListReserve4forward_lstm_39/TensorArrayV2/element_shape:output:0(forward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_39/TensorArrayV2п
Eforward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2G
Eforward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
7forward_lstm_39/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_39/transpose:y:0Nforward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_39/TensorArrayUnstack/TensorListFromTensor
%forward_lstm_39/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_39/strided_slice_2/stack
'forward_lstm_39/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_2/stack_1
'forward_lstm_39/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_2/stack_2м
forward_lstm_39/strided_slice_2StridedSliceforward_lstm_39/transpose:y:0.forward_lstm_39/strided_slice_2/stack:output:00forward_lstm_39/strided_slice_2/stack_1:output:00forward_lstm_39/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2!
forward_lstm_39/strided_slice_2ш
3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp<forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype025
3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp№
$forward_lstm_39/lstm_cell_118/MatMulMatMul(forward_lstm_39/strided_slice_2:output:0;forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$forward_lstm_39/lstm_cell_118/MatMulю
5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp>forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype027
5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpь
&forward_lstm_39/lstm_cell_118/MatMul_1MatMulforward_lstm_39/zeros:output:0=forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_39/lstm_cell_118/MatMul_1ф
!forward_lstm_39/lstm_cell_118/addAddV2.forward_lstm_39/lstm_cell_118/MatMul:product:00forward_lstm_39/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2#
!forward_lstm_39/lstm_cell_118/addч
4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp=forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype026
4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpё
%forward_lstm_39/lstm_cell_118/BiasAddBiasAdd%forward_lstm_39/lstm_cell_118/add:z:0<forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_39/lstm_cell_118/BiasAdd 
-forward_lstm_39/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-forward_lstm_39/lstm_cell_118/split/split_dimЗ
#forward_lstm_39/lstm_cell_118/splitSplit6forward_lstm_39/lstm_cell_118/split/split_dim:output:0.forward_lstm_39/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2%
#forward_lstm_39/lstm_cell_118/splitЙ
%forward_lstm_39/lstm_cell_118/SigmoidSigmoid,forward_lstm_39/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_39/lstm_cell_118/SigmoidН
'forward_lstm_39/lstm_cell_118/Sigmoid_1Sigmoid,forward_lstm_39/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_39/lstm_cell_118/Sigmoid_1Ю
!forward_lstm_39/lstm_cell_118/mulMul+forward_lstm_39/lstm_cell_118/Sigmoid_1:y:0 forward_lstm_39/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_39/lstm_cell_118/mulА
"forward_lstm_39/lstm_cell_118/ReluRelu,forward_lstm_39/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_39/lstm_cell_118/Reluр
#forward_lstm_39/lstm_cell_118/mul_1Mul)forward_lstm_39/lstm_cell_118/Sigmoid:y:00forward_lstm_39/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_39/lstm_cell_118/mul_1е
#forward_lstm_39/lstm_cell_118/add_1AddV2%forward_lstm_39/lstm_cell_118/mul:z:0'forward_lstm_39/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_39/lstm_cell_118/add_1Н
'forward_lstm_39/lstm_cell_118/Sigmoid_2Sigmoid,forward_lstm_39/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_39/lstm_cell_118/Sigmoid_2Џ
$forward_lstm_39/lstm_cell_118/Relu_1Relu'forward_lstm_39/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_39/lstm_cell_118/Relu_1ф
#forward_lstm_39/lstm_cell_118/mul_2Mul+forward_lstm_39/lstm_cell_118/Sigmoid_2:y:02forward_lstm_39/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_39/lstm_cell_118/mul_2Џ
-forward_lstm_39/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2/
-forward_lstm_39/TensorArrayV2_1/element_shapeј
forward_lstm_39/TensorArrayV2_1TensorListReserve6forward_lstm_39/TensorArrayV2_1/element_shape:output:0(forward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_39/TensorArrayV2_1n
forward_lstm_39/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_39/time 
forward_lstm_39/zeros_like	ZerosLike'forward_lstm_39/lstm_cell_118/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/zeros_like
(forward_lstm_39/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(forward_lstm_39/while/maximum_iterations
"forward_lstm_39/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_39/while/loop_counter	
forward_lstm_39/whileWhile+forward_lstm_39/while/loop_counter:output:01forward_lstm_39/while/maximum_iterations:output:0forward_lstm_39/time:output:0(forward_lstm_39/TensorArrayV2_1:handle:0forward_lstm_39/zeros_like:y:0forward_lstm_39/zeros:output:0 forward_lstm_39/zeros_1:output:0(forward_lstm_39/strided_slice_1:output:0Gforward_lstm_39/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_39/Cast:y:0<forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource>forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource=forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *.
body&R$
"forward_lstm_39_while_body_4931434*.
cond&R$
"forward_lstm_39_while_cond_4931433*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_39/whileе
@forward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2B
@forward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeБ
2forward_lstm_39/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_39/while:output:3Iforward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype024
2forward_lstm_39/TensorArrayV2Stack/TensorListStackЁ
%forward_lstm_39/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2'
%forward_lstm_39/strided_slice_3/stack
'forward_lstm_39/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_39/strided_slice_3/stack_1
'forward_lstm_39/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_3/stack_2њ
forward_lstm_39/strided_slice_3StridedSlice;forward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_39/strided_slice_3/stack:output:00forward_lstm_39/strided_slice_3/stack_1:output:00forward_lstm_39/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2!
forward_lstm_39/strided_slice_3
 forward_lstm_39/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_39/transpose_1/permю
forward_lstm_39/transpose_1	Transpose;forward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_39/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_39/transpose_1
forward_lstm_39/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_39/runtime
%backward_lstm_39/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%backward_lstm_39/RaggedToTensor/zeros
%backward_lstm_39/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2'
%backward_lstm_39/RaggedToTensor/Const
4backward_lstm_39/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.backward_lstm_39/RaggedToTensor/Const:output:0inputs.backward_lstm_39/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4backward_lstm_39/RaggedToTensor/RaggedTensorToTensorФ
;backward_lstm_39/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;backward_lstm_39/RaggedNestedRowLengths/strided_slice/stackШ
=backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1Ш
=backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2Љ
5backward_lstm_39/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dbackward_lstm_39/RaggedNestedRowLengths/strided_slice/stack:output:0Fbackward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fbackward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask27
5backward_lstm_39/RaggedNestedRowLengths/strided_sliceШ
=backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stackе
?backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2A
?backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1Ь
?backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2Е
7backward_lstm_39/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fbackward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hbackward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hbackward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask29
7backward_lstm_39/RaggedNestedRowLengths/strided_slice_1
+backward_lstm_39/RaggedNestedRowLengths/subSub>backward_lstm_39/RaggedNestedRowLengths/strided_slice:output:0@backward_lstm_39/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2-
+backward_lstm_39/RaggedNestedRowLengths/subЄ
backward_lstm_39/CastCast/backward_lstm_39/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_39/Cast
backward_lstm_39/ShapeShape=backward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_39/Shape
$backward_lstm_39/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_39/strided_slice/stack
&backward_lstm_39/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_39/strided_slice/stack_1
&backward_lstm_39/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_39/strided_slice/stack_2Ш
backward_lstm_39/strided_sliceStridedSlicebackward_lstm_39/Shape:output:0-backward_lstm_39/strided_slice/stack:output:0/backward_lstm_39/strided_slice/stack_1:output:0/backward_lstm_39/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_39/strided_slice~
backward_lstm_39/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_39/zeros/mul/yА
backward_lstm_39/zeros/mulMul'backward_lstm_39/strided_slice:output:0%backward_lstm_39/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros/mul
backward_lstm_39/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
backward_lstm_39/zeros/Less/yЋ
backward_lstm_39/zeros/LessLessbackward_lstm_39/zeros/mul:z:0&backward_lstm_39/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros/Less
backward_lstm_39/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_39/zeros/packed/1Ч
backward_lstm_39/zeros/packedPack'backward_lstm_39/strided_slice:output:0(backward_lstm_39/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_39/zeros/packed
backward_lstm_39/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_39/zeros/ConstЙ
backward_lstm_39/zerosFill&backward_lstm_39/zeros/packed:output:0%backward_lstm_39/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/zeros
backward_lstm_39/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_39/zeros_1/mul/yЖ
backward_lstm_39/zeros_1/mulMul'backward_lstm_39/strided_slice:output:0'backward_lstm_39/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros_1/mul
backward_lstm_39/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
backward_lstm_39/zeros_1/Less/yГ
backward_lstm_39/zeros_1/LessLess backward_lstm_39/zeros_1/mul:z:0(backward_lstm_39/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros_1/Less
!backward_lstm_39/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_39/zeros_1/packed/1Э
backward_lstm_39/zeros_1/packedPack'backward_lstm_39/strided_slice:output:0*backward_lstm_39/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_39/zeros_1/packed
backward_lstm_39/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_39/zeros_1/ConstС
backward_lstm_39/zeros_1Fill(backward_lstm_39/zeros_1/packed:output:0'backward_lstm_39/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/zeros_1
backward_lstm_39/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_39/transpose/permэ
backward_lstm_39/transpose	Transpose=backward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0(backward_lstm_39/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_39/transpose
backward_lstm_39/Shape_1Shapebackward_lstm_39/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_39/Shape_1
&backward_lstm_39/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_39/strided_slice_1/stack
(backward_lstm_39/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_1/stack_1
(backward_lstm_39/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_1/stack_2д
 backward_lstm_39/strided_slice_1StridedSlice!backward_lstm_39/Shape_1:output:0/backward_lstm_39/strided_slice_1/stack:output:01backward_lstm_39/strided_slice_1/stack_1:output:01backward_lstm_39/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_39/strided_slice_1Ї
,backward_lstm_39/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,backward_lstm_39/TensorArrayV2/element_shapeі
backward_lstm_39/TensorArrayV2TensorListReserve5backward_lstm_39/TensorArrayV2/element_shape:output:0)backward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_39/TensorArrayV2
backward_lstm_39/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_39/ReverseV2/axisЮ
backward_lstm_39/ReverseV2	ReverseV2backward_lstm_39/transpose:y:0(backward_lstm_39/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_39/ReverseV2с
Fbackward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fbackward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeС
8backward_lstm_39/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_39/ReverseV2:output:0Obackward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_39/TensorArrayUnstack/TensorListFromTensor
&backward_lstm_39/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_39/strided_slice_2/stack
(backward_lstm_39/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_2/stack_1
(backward_lstm_39/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_2/stack_2т
 backward_lstm_39/strided_slice_2StridedSlicebackward_lstm_39/transpose:y:0/backward_lstm_39/strided_slice_2/stack:output:01backward_lstm_39/strided_slice_2/stack_1:output:01backward_lstm_39/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 backward_lstm_39/strided_slice_2ы
4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpReadVariableOp=backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpє
%backward_lstm_39/lstm_cell_119/MatMulMatMul)backward_lstm_39/strided_slice_2:output:0<backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%backward_lstm_39/lstm_cell_119/MatMulё
6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOp?backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp№
'backward_lstm_39/lstm_cell_119/MatMul_1MatMulbackward_lstm_39/zeros:output:0>backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_39/lstm_cell_119/MatMul_1ш
"backward_lstm_39/lstm_cell_119/addAddV2/backward_lstm_39/lstm_cell_119/MatMul:product:01backward_lstm_39/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"backward_lstm_39/lstm_cell_119/addъ
5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOp>backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpѕ
&backward_lstm_39/lstm_cell_119/BiasAddBiasAdd&backward_lstm_39/lstm_cell_119/add:z:0=backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_39/lstm_cell_119/BiasAddЂ
.backward_lstm_39/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.backward_lstm_39/lstm_cell_119/split/split_dimЛ
$backward_lstm_39/lstm_cell_119/splitSplit7backward_lstm_39/lstm_cell_119/split/split_dim:output:0/backward_lstm_39/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$backward_lstm_39/lstm_cell_119/splitМ
&backward_lstm_39/lstm_cell_119/SigmoidSigmoid-backward_lstm_39/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_39/lstm_cell_119/SigmoidР
(backward_lstm_39/lstm_cell_119/Sigmoid_1Sigmoid-backward_lstm_39/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_39/lstm_cell_119/Sigmoid_1в
"backward_lstm_39/lstm_cell_119/mulMul,backward_lstm_39/lstm_cell_119/Sigmoid_1:y:0!backward_lstm_39/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_39/lstm_cell_119/mulГ
#backward_lstm_39/lstm_cell_119/ReluRelu-backward_lstm_39/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_39/lstm_cell_119/Reluф
$backward_lstm_39/lstm_cell_119/mul_1Mul*backward_lstm_39/lstm_cell_119/Sigmoid:y:01backward_lstm_39/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_39/lstm_cell_119/mul_1й
$backward_lstm_39/lstm_cell_119/add_1AddV2&backward_lstm_39/lstm_cell_119/mul:z:0(backward_lstm_39/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_39/lstm_cell_119/add_1Р
(backward_lstm_39/lstm_cell_119/Sigmoid_2Sigmoid-backward_lstm_39/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_39/lstm_cell_119/Sigmoid_2В
%backward_lstm_39/lstm_cell_119/Relu_1Relu(backward_lstm_39/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_39/lstm_cell_119/Relu_1ш
$backward_lstm_39/lstm_cell_119/mul_2Mul,backward_lstm_39/lstm_cell_119/Sigmoid_2:y:03backward_lstm_39/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_39/lstm_cell_119/mul_2Б
.backward_lstm_39/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.backward_lstm_39/TensorArrayV2_1/element_shapeќ
 backward_lstm_39/TensorArrayV2_1TensorListReserve7backward_lstm_39/TensorArrayV2_1/element_shape:output:0)backward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_39/TensorArrayV2_1p
backward_lstm_39/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_39/time
&backward_lstm_39/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_lstm_39/Max/reduction_indices 
backward_lstm_39/MaxMaxbackward_lstm_39/Cast:y:0/backward_lstm_39/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/Maxr
backward_lstm_39/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_39/sub/y
backward_lstm_39/subSubbackward_lstm_39/Max:output:0backward_lstm_39/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/sub
backward_lstm_39/Sub_1Subbackward_lstm_39/sub:z:0backward_lstm_39/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_39/Sub_1Ѓ
backward_lstm_39/zeros_like	ZerosLike(backward_lstm_39/lstm_cell_119/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/zeros_likeЁ
)backward_lstm_39/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)backward_lstm_39/while/maximum_iterations
#backward_lstm_39/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_39/while/loop_counter	
backward_lstm_39/whileWhile,backward_lstm_39/while/loop_counter:output:02backward_lstm_39/while/maximum_iterations:output:0backward_lstm_39/time:output:0)backward_lstm_39/TensorArrayV2_1:handle:0backward_lstm_39/zeros_like:y:0backward_lstm_39/zeros:output:0!backward_lstm_39/zeros_1:output:0)backward_lstm_39/strided_slice_1:output:0Hbackward_lstm_39/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_39/Sub_1:z:0=backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource?backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource>backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( */
body'R%
#backward_lstm_39_while_body_4931613*/
cond'R%
#backward_lstm_39_while_cond_4931612*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_39/whileз
Abackward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Abackward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3backward_lstm_39/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_39/while:output:3Jbackward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3backward_lstm_39/TensorArrayV2Stack/TensorListStackЃ
&backward_lstm_39/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&backward_lstm_39/strided_slice_3/stack
(backward_lstm_39/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_39/strided_slice_3/stack_1
(backward_lstm_39/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_3/stack_2
 backward_lstm_39/strided_slice_3StridedSlice<backward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_39/strided_slice_3/stack:output:01backward_lstm_39/strided_slice_3/stack_1:output:01backward_lstm_39/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 backward_lstm_39/strided_slice_3
!backward_lstm_39/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_39/transpose_1/permђ
backward_lstm_39/transpose_1	Transpose<backward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_39/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_39/transpose_1
backward_lstm_39/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_39/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisТ
concatConcatV2(forward_lstm_39/strided_slice_3:output:0)backward_lstm_39/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

IdentityЬ
NoOpNoOp6^backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp5^backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp7^backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp^backward_lstm_39/while5^forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp4^forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp6^forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp^forward_lstm_39/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2n
5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp2l
4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp2p
6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp20
backward_lstm_39/whilebackward_lstm_39/while2l
4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp2j
3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp2n
5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp2.
forward_lstm_39/whileforward_lstm_39/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѓ

*__inference_dense_39_layer_call_fn_4933729

inputs
unknown:d
	unknown_0:
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_39_layer_call_and_return_conditional_losses_49317352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
ј?
к
while_body_4934153
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_118_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_118_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_118_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_118_matmul_readvariableop_resource:	ШG
4while_lstm_cell_118_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_118_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_118/BiasAdd/ReadVariableOpЂ)while/lstm_cell_118/MatMul/ReadVariableOpЂ+while/lstm_cell_118/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeм
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЬ
)while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_118/MatMul/ReadVariableOpк
while/lstm_cell_118/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/MatMulв
+while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_118/MatMul_1/ReadVariableOpУ
while/lstm_cell_118/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/MatMul_1М
while/lstm_cell_118/addAddV2$while/lstm_cell_118/MatMul:product:0&while/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/addЫ
*while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_118/BiasAdd/ReadVariableOpЩ
while/lstm_cell_118/BiasAddBiasAddwhile/lstm_cell_118/add:z:02while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/BiasAdd
#while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_118/split/split_dim
while/lstm_cell_118/splitSplit,while/lstm_cell_118/split/split_dim:output:0$while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_118/split
while/lstm_cell_118/SigmoidSigmoid"while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Sigmoid
while/lstm_cell_118/Sigmoid_1Sigmoid"while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Sigmoid_1Ѓ
while/lstm_cell_118/mulMul!while/lstm_cell_118/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/mul
while/lstm_cell_118/ReluRelu"while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/ReluИ
while/lstm_cell_118/mul_1Mulwhile/lstm_cell_118/Sigmoid:y:0&while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/mul_1­
while/lstm_cell_118/add_1AddV2while/lstm_cell_118/mul:z:0while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/add_1
while/lstm_cell_118/Sigmoid_2Sigmoid"while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Sigmoid_2
while/lstm_cell_118/Relu_1Reluwhile/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Relu_1М
while/lstm_cell_118/mul_2Mul!while/lstm_cell_118/Sigmoid_2:y:0(while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_118/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_118/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_118/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_118/BiasAdd/ReadVariableOp*^while/lstm_cell_118/MatMul/ReadVariableOp,^while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_118_biasadd_readvariableop_resource5while_lstm_cell_118_biasadd_readvariableop_resource_0"n
4while_lstm_cell_118_matmul_1_readvariableop_resource6while_lstm_cell_118_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_118_matmul_readvariableop_resource4while_lstm_cell_118_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_118/BiasAdd/ReadVariableOp*while/lstm_cell_118/BiasAdd/ReadVariableOp2V
)while/lstm_cell_118/MatMul/ReadVariableOp)while/lstm_cell_118/MatMul/ReadVariableOp2Z
+while/lstm_cell_118/MatMul_1/ReadVariableOp+while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
в
С
2__inference_backward_lstm_39_layer_call_fn_4934410
inputs_0
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_49302762
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
к
Ш
while_cond_4934806
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_4934806___redundant_placeholder05
1while_while_cond_4934806___redundant_placeholder15
1while_while_cond_4934806___redundant_placeholder25
1while_while_cond_4934806___redundant_placeholder3
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
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
§

J__inference_lstm_cell_119_layer_call_and_return_conditional_losses_4935240

inputs
states_0
states_11
matmul_readvariableop_resource:	Ш3
 matmul_1_readvariableop_resource:	2Ш.
biasadd_readvariableop_resource:	Ш
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2
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
?:џџџџџџџџџ:џџџџџџџџџ2:џџџџџџџџџ2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/1
Ш

Ь
%__inference_signature_wrapper_4932330

args_0
args_0_1	
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
	unknown_2:	Ш
	unknown_3:	2Ш
	unknown_4:	Ш
	unknown_5:d
	unknown_6:
identityЂStatefulPartitionedCallЋ
StatefulPartitionedCallStatefulPartitionedCallargs_0args_0_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_49292742
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameargs_0:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
args_0_1
єd
Н
#backward_lstm_39_while_body_4931613>
:backward_lstm_39_while_backward_lstm_39_while_loop_counterD
@backward_lstm_39_while_backward_lstm_39_while_maximum_iterations&
"backward_lstm_39_while_placeholder(
$backward_lstm_39_while_placeholder_1(
$backward_lstm_39_while_placeholder_2(
$backward_lstm_39_while_placeholder_3(
$backward_lstm_39_while_placeholder_4=
9backward_lstm_39_while_backward_lstm_39_strided_slice_1_0y
ubackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_08
4backward_lstm_39_while_less_backward_lstm_39_sub_1_0X
Ebackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0:	ШZ
Gbackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0:	2ШU
Fbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0:	Ш#
backward_lstm_39_while_identity%
!backward_lstm_39_while_identity_1%
!backward_lstm_39_while_identity_2%
!backward_lstm_39_while_identity_3%
!backward_lstm_39_while_identity_4%
!backward_lstm_39_while_identity_5%
!backward_lstm_39_while_identity_6;
7backward_lstm_39_while_backward_lstm_39_strided_slice_1w
sbackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor6
2backward_lstm_39_while_less_backward_lstm_39_sub_1V
Cbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource:	ШX
Ebackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource:	2ШS
Dbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource:	ШЂ;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOpЂ:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOpЂ<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOpх
Hbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2J
Hbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
:backward_lstm_39/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_39_while_placeholderQbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02<
:backward_lstm_39/while/TensorArrayV2Read/TensorListGetItemЪ
backward_lstm_39/while/LessLess4backward_lstm_39_while_less_backward_lstm_39_sub_1_0"backward_lstm_39_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_39/while/Lessџ
:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOpReadVariableOpEbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp
+backward_lstm_39/while/lstm_cell_119/MatMulMatMulAbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem:item:0Bbackward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+backward_lstm_39/while/lstm_cell_119/MatMul
<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOpGbackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp
-backward_lstm_39/while/lstm_cell_119/MatMul_1MatMul$backward_lstm_39_while_placeholder_3Dbackward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_39/while/lstm_cell_119/MatMul_1
(backward_lstm_39/while/lstm_cell_119/addAddV25backward_lstm_39/while/lstm_cell_119/MatMul:product:07backward_lstm_39/while/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_39/while/lstm_cell_119/addў
;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOpFbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp
,backward_lstm_39/while/lstm_cell_119/BiasAddBiasAdd,backward_lstm_39/while/lstm_cell_119/add:z:0Cbackward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_39/while/lstm_cell_119/BiasAddЎ
4backward_lstm_39/while/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4backward_lstm_39/while/lstm_cell_119/split/split_dimг
*backward_lstm_39/while/lstm_cell_119/splitSplit=backward_lstm_39/while/lstm_cell_119/split/split_dim:output:05backward_lstm_39/while/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*backward_lstm_39/while/lstm_cell_119/splitЮ
,backward_lstm_39/while/lstm_cell_119/SigmoidSigmoid3backward_lstm_39/while/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_39/while/lstm_cell_119/Sigmoidв
.backward_lstm_39/while/lstm_cell_119/Sigmoid_1Sigmoid3backward_lstm_39/while/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.backward_lstm_39/while/lstm_cell_119/Sigmoid_1ч
(backward_lstm_39/while/lstm_cell_119/mulMul2backward_lstm_39/while/lstm_cell_119/Sigmoid_1:y:0$backward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_39/while/lstm_cell_119/mulХ
)backward_lstm_39/while/lstm_cell_119/ReluRelu3backward_lstm_39/while/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_39/while/lstm_cell_119/Reluќ
*backward_lstm_39/while/lstm_cell_119/mul_1Mul0backward_lstm_39/while/lstm_cell_119/Sigmoid:y:07backward_lstm_39/while/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_39/while/lstm_cell_119/mul_1ё
*backward_lstm_39/while/lstm_cell_119/add_1AddV2,backward_lstm_39/while/lstm_cell_119/mul:z:0.backward_lstm_39/while/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_39/while/lstm_cell_119/add_1в
.backward_lstm_39/while/lstm_cell_119/Sigmoid_2Sigmoid3backward_lstm_39/while/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.backward_lstm_39/while/lstm_cell_119/Sigmoid_2Ф
+backward_lstm_39/while/lstm_cell_119/Relu_1Relu.backward_lstm_39/while/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_39/while/lstm_cell_119/Relu_1
*backward_lstm_39/while/lstm_cell_119/mul_2Mul2backward_lstm_39/while/lstm_cell_119/Sigmoid_2:y:09backward_lstm_39/while/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_39/while/lstm_cell_119/mul_2ё
backward_lstm_39/while/SelectSelectbackward_lstm_39/while/Less:z:0.backward_lstm_39/while/lstm_cell_119/mul_2:z:0$backward_lstm_39_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/while/Selectѕ
backward_lstm_39/while/Select_1Selectbackward_lstm_39/while/Less:z:0.backward_lstm_39/while/lstm_cell_119/mul_2:z:0$backward_lstm_39_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22!
backward_lstm_39/while/Select_1ѕ
backward_lstm_39/while/Select_2Selectbackward_lstm_39/while/Less:z:0.backward_lstm_39/while/lstm_cell_119/add_1:z:0$backward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22!
backward_lstm_39/while/Select_2Ў
;backward_lstm_39/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_39_while_placeholder_1"backward_lstm_39_while_placeholder&backward_lstm_39/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_39/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_39/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_39/while/add/y­
backward_lstm_39/while/addAddV2"backward_lstm_39_while_placeholder%backward_lstm_39/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/while/add
backward_lstm_39/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_39/while/add_1/yЫ
backward_lstm_39/while/add_1AddV2:backward_lstm_39_while_backward_lstm_39_while_loop_counter'backward_lstm_39/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/while/add_1Џ
backward_lstm_39/while/IdentityIdentity backward_lstm_39/while/add_1:z:0^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_39/while/Identityг
!backward_lstm_39/while/Identity_1Identity@backward_lstm_39_while_backward_lstm_39_while_maximum_iterations^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_39/while/Identity_1Б
!backward_lstm_39/while/Identity_2Identitybackward_lstm_39/while/add:z:0^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_39/while/Identity_2о
!backward_lstm_39/while/Identity_3IdentityKbackward_lstm_39/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_39/while/Identity_3Ъ
!backward_lstm_39/while/Identity_4Identity&backward_lstm_39/while/Select:output:0^backward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_39/while/Identity_4Ь
!backward_lstm_39/while/Identity_5Identity(backward_lstm_39/while/Select_1:output:0^backward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_39/while/Identity_5Ь
!backward_lstm_39/while/Identity_6Identity(backward_lstm_39/while/Select_2:output:0^backward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_39/while/Identity_6Ж
backward_lstm_39/while/NoOpNoOp<^backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp;^backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp=^backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_39/while/NoOp"t
7backward_lstm_39_while_backward_lstm_39_strided_slice_19backward_lstm_39_while_backward_lstm_39_strided_slice_1_0"K
backward_lstm_39_while_identity(backward_lstm_39/while/Identity:output:0"O
!backward_lstm_39_while_identity_1*backward_lstm_39/while/Identity_1:output:0"O
!backward_lstm_39_while_identity_2*backward_lstm_39/while/Identity_2:output:0"O
!backward_lstm_39_while_identity_3*backward_lstm_39/while/Identity_3:output:0"O
!backward_lstm_39_while_identity_4*backward_lstm_39/while/Identity_4:output:0"O
!backward_lstm_39_while_identity_5*backward_lstm_39/while/Identity_5:output:0"O
!backward_lstm_39_while_identity_6*backward_lstm_39/while/Identity_6:output:0"j
2backward_lstm_39_while_less_backward_lstm_39_sub_14backward_lstm_39_while_less_backward_lstm_39_sub_1_0"
Dbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resourceFbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0"
Ebackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resourceGbackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0"
Cbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resourceEbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0"ь
sbackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensorubackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2z
;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp2x
:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp2|
<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:џџџџџџџџџ
ј?
к
while_body_4930615
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_118_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_118_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_118_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_118_matmul_readvariableop_resource:	ШG
4while_lstm_cell_118_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_118_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_118/BiasAdd/ReadVariableOpЂ)while/lstm_cell_118/MatMul/ReadVariableOpЂ+while/lstm_cell_118/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeм
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЬ
)while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_118/MatMul/ReadVariableOpк
while/lstm_cell_118/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/MatMulв
+while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_118/MatMul_1/ReadVariableOpУ
while/lstm_cell_118/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/MatMul_1М
while/lstm_cell_118/addAddV2$while/lstm_cell_118/MatMul:product:0&while/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/addЫ
*while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_118/BiasAdd/ReadVariableOpЩ
while/lstm_cell_118/BiasAddBiasAddwhile/lstm_cell_118/add:z:02while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/BiasAdd
#while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_118/split/split_dim
while/lstm_cell_118/splitSplit,while/lstm_cell_118/split/split_dim:output:0$while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_118/split
while/lstm_cell_118/SigmoidSigmoid"while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Sigmoid
while/lstm_cell_118/Sigmoid_1Sigmoid"while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Sigmoid_1Ѓ
while/lstm_cell_118/mulMul!while/lstm_cell_118/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/mul
while/lstm_cell_118/ReluRelu"while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/ReluИ
while/lstm_cell_118/mul_1Mulwhile/lstm_cell_118/Sigmoid:y:0&while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/mul_1­
while/lstm_cell_118/add_1AddV2while/lstm_cell_118/mul:z:0while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/add_1
while/lstm_cell_118/Sigmoid_2Sigmoid"while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Sigmoid_2
while/lstm_cell_118/Relu_1Reluwhile/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Relu_1М
while/lstm_cell_118/mul_2Mul!while/lstm_cell_118/Sigmoid_2:y:0(while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_118/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_118/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_118/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_118/BiasAdd/ReadVariableOp*^while/lstm_cell_118/MatMul/ReadVariableOp,^while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_118_biasadd_readvariableop_resource5while_lstm_cell_118_biasadd_readvariableop_resource_0"n
4while_lstm_cell_118_matmul_1_readvariableop_resource6while_lstm_cell_118_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_118_matmul_readvariableop_resource4while_lstm_cell_118_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_118/BiasAdd/ReadVariableOp*while/lstm_cell_118/BiasAdd/ReadVariableOp2V
)while/lstm_cell_118/MatMul/ReadVariableOp)while/lstm_cell_118/MatMul/ReadVariableOp2Z
+while/lstm_cell_118/MatMul_1/ReadVariableOp+while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
єd
Н
#backward_lstm_39_while_body_4933265>
:backward_lstm_39_while_backward_lstm_39_while_loop_counterD
@backward_lstm_39_while_backward_lstm_39_while_maximum_iterations&
"backward_lstm_39_while_placeholder(
$backward_lstm_39_while_placeholder_1(
$backward_lstm_39_while_placeholder_2(
$backward_lstm_39_while_placeholder_3(
$backward_lstm_39_while_placeholder_4=
9backward_lstm_39_while_backward_lstm_39_strided_slice_1_0y
ubackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_08
4backward_lstm_39_while_less_backward_lstm_39_sub_1_0X
Ebackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0:	ШZ
Gbackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0:	2ШU
Fbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0:	Ш#
backward_lstm_39_while_identity%
!backward_lstm_39_while_identity_1%
!backward_lstm_39_while_identity_2%
!backward_lstm_39_while_identity_3%
!backward_lstm_39_while_identity_4%
!backward_lstm_39_while_identity_5%
!backward_lstm_39_while_identity_6;
7backward_lstm_39_while_backward_lstm_39_strided_slice_1w
sbackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor6
2backward_lstm_39_while_less_backward_lstm_39_sub_1V
Cbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource:	ШX
Ebackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource:	2ШS
Dbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource:	ШЂ;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOpЂ:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOpЂ<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOpх
Hbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2J
Hbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
:backward_lstm_39/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_39_while_placeholderQbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02<
:backward_lstm_39/while/TensorArrayV2Read/TensorListGetItemЪ
backward_lstm_39/while/LessLess4backward_lstm_39_while_less_backward_lstm_39_sub_1_0"backward_lstm_39_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_39/while/Lessџ
:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOpReadVariableOpEbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp
+backward_lstm_39/while/lstm_cell_119/MatMulMatMulAbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem:item:0Bbackward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+backward_lstm_39/while/lstm_cell_119/MatMul
<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOpGbackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp
-backward_lstm_39/while/lstm_cell_119/MatMul_1MatMul$backward_lstm_39_while_placeholder_3Dbackward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_39/while/lstm_cell_119/MatMul_1
(backward_lstm_39/while/lstm_cell_119/addAddV25backward_lstm_39/while/lstm_cell_119/MatMul:product:07backward_lstm_39/while/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_39/while/lstm_cell_119/addў
;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOpFbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp
,backward_lstm_39/while/lstm_cell_119/BiasAddBiasAdd,backward_lstm_39/while/lstm_cell_119/add:z:0Cbackward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_39/while/lstm_cell_119/BiasAddЎ
4backward_lstm_39/while/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4backward_lstm_39/while/lstm_cell_119/split/split_dimг
*backward_lstm_39/while/lstm_cell_119/splitSplit=backward_lstm_39/while/lstm_cell_119/split/split_dim:output:05backward_lstm_39/while/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*backward_lstm_39/while/lstm_cell_119/splitЮ
,backward_lstm_39/while/lstm_cell_119/SigmoidSigmoid3backward_lstm_39/while/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_39/while/lstm_cell_119/Sigmoidв
.backward_lstm_39/while/lstm_cell_119/Sigmoid_1Sigmoid3backward_lstm_39/while/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.backward_lstm_39/while/lstm_cell_119/Sigmoid_1ч
(backward_lstm_39/while/lstm_cell_119/mulMul2backward_lstm_39/while/lstm_cell_119/Sigmoid_1:y:0$backward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_39/while/lstm_cell_119/mulХ
)backward_lstm_39/while/lstm_cell_119/ReluRelu3backward_lstm_39/while/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_39/while/lstm_cell_119/Reluќ
*backward_lstm_39/while/lstm_cell_119/mul_1Mul0backward_lstm_39/while/lstm_cell_119/Sigmoid:y:07backward_lstm_39/while/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_39/while/lstm_cell_119/mul_1ё
*backward_lstm_39/while/lstm_cell_119/add_1AddV2,backward_lstm_39/while/lstm_cell_119/mul:z:0.backward_lstm_39/while/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_39/while/lstm_cell_119/add_1в
.backward_lstm_39/while/lstm_cell_119/Sigmoid_2Sigmoid3backward_lstm_39/while/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.backward_lstm_39/while/lstm_cell_119/Sigmoid_2Ф
+backward_lstm_39/while/lstm_cell_119/Relu_1Relu.backward_lstm_39/while/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_39/while/lstm_cell_119/Relu_1
*backward_lstm_39/while/lstm_cell_119/mul_2Mul2backward_lstm_39/while/lstm_cell_119/Sigmoid_2:y:09backward_lstm_39/while/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_39/while/lstm_cell_119/mul_2ё
backward_lstm_39/while/SelectSelectbackward_lstm_39/while/Less:z:0.backward_lstm_39/while/lstm_cell_119/mul_2:z:0$backward_lstm_39_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/while/Selectѕ
backward_lstm_39/while/Select_1Selectbackward_lstm_39/while/Less:z:0.backward_lstm_39/while/lstm_cell_119/mul_2:z:0$backward_lstm_39_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22!
backward_lstm_39/while/Select_1ѕ
backward_lstm_39/while/Select_2Selectbackward_lstm_39/while/Less:z:0.backward_lstm_39/while/lstm_cell_119/add_1:z:0$backward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22!
backward_lstm_39/while/Select_2Ў
;backward_lstm_39/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_39_while_placeholder_1"backward_lstm_39_while_placeholder&backward_lstm_39/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_39/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_39/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_39/while/add/y­
backward_lstm_39/while/addAddV2"backward_lstm_39_while_placeholder%backward_lstm_39/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/while/add
backward_lstm_39/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_39/while/add_1/yЫ
backward_lstm_39/while/add_1AddV2:backward_lstm_39_while_backward_lstm_39_while_loop_counter'backward_lstm_39/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/while/add_1Џ
backward_lstm_39/while/IdentityIdentity backward_lstm_39/while/add_1:z:0^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_39/while/Identityг
!backward_lstm_39/while/Identity_1Identity@backward_lstm_39_while_backward_lstm_39_while_maximum_iterations^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_39/while/Identity_1Б
!backward_lstm_39/while/Identity_2Identitybackward_lstm_39/while/add:z:0^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_39/while/Identity_2о
!backward_lstm_39/while/Identity_3IdentityKbackward_lstm_39/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_39/while/Identity_3Ъ
!backward_lstm_39/while/Identity_4Identity&backward_lstm_39/while/Select:output:0^backward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_39/while/Identity_4Ь
!backward_lstm_39/while/Identity_5Identity(backward_lstm_39/while/Select_1:output:0^backward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_39/while/Identity_5Ь
!backward_lstm_39/while/Identity_6Identity(backward_lstm_39/while/Select_2:output:0^backward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_39/while/Identity_6Ж
backward_lstm_39/while/NoOpNoOp<^backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp;^backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp=^backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_39/while/NoOp"t
7backward_lstm_39_while_backward_lstm_39_strided_slice_19backward_lstm_39_while_backward_lstm_39_strided_slice_1_0"K
backward_lstm_39_while_identity(backward_lstm_39/while/Identity:output:0"O
!backward_lstm_39_while_identity_1*backward_lstm_39/while/Identity_1:output:0"O
!backward_lstm_39_while_identity_2*backward_lstm_39/while/Identity_2:output:0"O
!backward_lstm_39_while_identity_3*backward_lstm_39/while/Identity_3:output:0"O
!backward_lstm_39_while_identity_4*backward_lstm_39/while/Identity_4:output:0"O
!backward_lstm_39_while_identity_5*backward_lstm_39/while/Identity_5:output:0"O
!backward_lstm_39_while_identity_6*backward_lstm_39/while/Identity_6:output:0"j
2backward_lstm_39_while_less_backward_lstm_39_sub_14backward_lstm_39_while_less_backward_lstm_39_sub_1_0"
Dbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resourceFbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0"
Ebackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resourceGbackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0"
Cbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resourceEbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0"ь
sbackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensorubackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2z
;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp2x
:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp2|
<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:џџџџџџџџџ
ў\
Ќ
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_4934086
inputs_0?
,lstm_cell_118_matmul_readvariableop_resource:	ШA
.lstm_cell_118_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_118_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_118/BiasAdd/ReadVariableOpЂ#lstm_cell_118/MatMul/ReadVariableOpЂ%lstm_cell_118/MatMul_1/ReadVariableOpЂwhileF
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1
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
:џџџџџџџџџ22
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
B :ш2
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
zeros_1/packed/1
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
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
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
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2И
#lstm_cell_118/MatMul/ReadVariableOpReadVariableOp,lstm_cell_118_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_118/MatMul/ReadVariableOpА
lstm_cell_118/MatMulMatMulstrided_slice_2:output:0+lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/MatMulО
%lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_118/MatMul_1/ReadVariableOpЌ
lstm_cell_118/MatMul_1MatMulzeros:output:0-lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/MatMul_1Є
lstm_cell_118/addAddV2lstm_cell_118/MatMul:product:0 lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/addЗ
$lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_118/BiasAdd/ReadVariableOpБ
lstm_cell_118/BiasAddBiasAddlstm_cell_118/add:z:0,lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/BiasAdd
lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_118/split/split_dimї
lstm_cell_118/splitSplit&lstm_cell_118/split/split_dim:output:0lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_118/split
lstm_cell_118/SigmoidSigmoidlstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Sigmoid
lstm_cell_118/Sigmoid_1Sigmoidlstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Sigmoid_1
lstm_cell_118/mulMullstm_cell_118/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/mul
lstm_cell_118/ReluRelulstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Relu 
lstm_cell_118/mul_1Mullstm_cell_118/Sigmoid:y:0 lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/mul_1
lstm_cell_118/add_1AddV2lstm_cell_118/mul:z:0lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/add_1
lstm_cell_118/Sigmoid_2Sigmoidlstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Sigmoid_2
lstm_cell_118/Relu_1Relulstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Relu_1Є
lstm_cell_118/mul_2Mullstm_cell_118/Sigmoid_2:y:0"lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_118_matmul_readvariableop_resource.lstm_cell_118_matmul_1_readvariableop_resource-lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_4934002*
condR
while_cond_4934001*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
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
:џџџџџџџџџ22

IdentityЫ
NoOpNoOp%^lstm_cell_118/BiasAdd/ReadVariableOp$^lstm_cell_118/MatMul/ReadVariableOp&^lstm_cell_118/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_118/BiasAdd/ReadVariableOp$lstm_cell_118/BiasAdd/ReadVariableOp2J
#lstm_cell_118/MatMul/ReadVariableOp#lstm_cell_118/MatMul/ReadVariableOp2N
%lstm_cell_118/MatMul_1/ReadVariableOp%lstm_cell_118/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
К
ј
/__inference_lstm_cell_118_layer_call_fn_4935061

inputs
states_0
states_1
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identity

identity_1

identity_2ЂStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_118_layer_call_and_return_conditional_losses_49293492
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

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
?:џџџџџџџџџ:џџџџџџџџџ2:џџџџџџџџџ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/1
Ђ_
Ћ
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_4931051

inputs?
,lstm_cell_119_matmul_readvariableop_resource:	ШA
.lstm_cell_119_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_119_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_119/BiasAdd/ReadVariableOpЂ#lstm_cell_119/MatMul/ReadVariableOpЂ%lstm_cell_119/MatMul_1/ReadVariableOpЂwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1
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
:џџџџџџџџџ22
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
B :ш2
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
zeros_1/packed/1
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
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
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
ReverseV2/axis
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
	ReverseV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ27
5TensorArrayUnstack/TensorListFromTensor/element_shape§
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
strided_slice_2/stack_2
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2
strided_slice_2И
#lstm_cell_119/MatMul/ReadVariableOpReadVariableOp,lstm_cell_119_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_119/MatMul/ReadVariableOpА
lstm_cell_119/MatMulMatMulstrided_slice_2:output:0+lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/MatMulО
%lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_119_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_119/MatMul_1/ReadVariableOpЌ
lstm_cell_119/MatMul_1MatMulzeros:output:0-lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/MatMul_1Є
lstm_cell_119/addAddV2lstm_cell_119/MatMul:product:0 lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/addЗ
$lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_119_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_119/BiasAdd/ReadVariableOpБ
lstm_cell_119/BiasAddBiasAddlstm_cell_119/add:z:0,lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/BiasAdd
lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_119/split/split_dimї
lstm_cell_119/splitSplit&lstm_cell_119/split/split_dim:output:0lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_119/split
lstm_cell_119/SigmoidSigmoidlstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Sigmoid
lstm_cell_119/Sigmoid_1Sigmoidlstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Sigmoid_1
lstm_cell_119/mulMullstm_cell_119/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/mul
lstm_cell_119/ReluRelulstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Relu 
lstm_cell_119/mul_1Mullstm_cell_119/Sigmoid:y:0 lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/mul_1
lstm_cell_119/add_1AddV2lstm_cell_119/mul:z:0lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/add_1
lstm_cell_119/Sigmoid_2Sigmoidlstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Sigmoid_2
lstm_cell_119/Relu_1Relulstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Relu_1Є
lstm_cell_119/mul_2Mullstm_cell_119/Sigmoid_2:y:0"lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_119_matmul_readvariableop_resource.lstm_cell_119_matmul_1_readvariableop_resource-lstm_cell_119_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_4930967*
condR
while_cond_4930966*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
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
:џџџџџџџџџ22

IdentityЫ
NoOpNoOp%^lstm_cell_119/BiasAdd/ReadVariableOp$^lstm_cell_119/MatMul/ReadVariableOp&^lstm_cell_119/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_119/BiasAdd/ReadVariableOp$lstm_cell_119/BiasAdd/ReadVariableOp2J
#lstm_cell_119/MatMul/ReadVariableOp#lstm_cell_119/MatMul/ReadVariableOp2N
%lstm_cell_119/MatMul_1/ReadVariableOp%lstm_cell_119/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Б
К
Asequential_39_bidirectional_39_forward_lstm_39_while_body_4928991z
vsequential_39_bidirectional_39_forward_lstm_39_while_sequential_39_bidirectional_39_forward_lstm_39_while_loop_counter
|sequential_39_bidirectional_39_forward_lstm_39_while_sequential_39_bidirectional_39_forward_lstm_39_while_maximum_iterationsD
@sequential_39_bidirectional_39_forward_lstm_39_while_placeholderF
Bsequential_39_bidirectional_39_forward_lstm_39_while_placeholder_1F
Bsequential_39_bidirectional_39_forward_lstm_39_while_placeholder_2F
Bsequential_39_bidirectional_39_forward_lstm_39_while_placeholder_3F
Bsequential_39_bidirectional_39_forward_lstm_39_while_placeholder_4y
usequential_39_bidirectional_39_forward_lstm_39_while_sequential_39_bidirectional_39_forward_lstm_39_strided_slice_1_0Ж
Бsequential_39_bidirectional_39_forward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_sequential_39_bidirectional_39_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0v
rsequential_39_bidirectional_39_forward_lstm_39_while_greater_sequential_39_bidirectional_39_forward_lstm_39_cast_0v
csequential_39_bidirectional_39_forward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0:	Шx
esequential_39_bidirectional_39_forward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0:	2Шs
dsequential_39_bidirectional_39_forward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0:	ШA
=sequential_39_bidirectional_39_forward_lstm_39_while_identityC
?sequential_39_bidirectional_39_forward_lstm_39_while_identity_1C
?sequential_39_bidirectional_39_forward_lstm_39_while_identity_2C
?sequential_39_bidirectional_39_forward_lstm_39_while_identity_3C
?sequential_39_bidirectional_39_forward_lstm_39_while_identity_4C
?sequential_39_bidirectional_39_forward_lstm_39_while_identity_5C
?sequential_39_bidirectional_39_forward_lstm_39_while_identity_6w
ssequential_39_bidirectional_39_forward_lstm_39_while_sequential_39_bidirectional_39_forward_lstm_39_strided_slice_1Д
Џsequential_39_bidirectional_39_forward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_sequential_39_bidirectional_39_forward_lstm_39_tensorarrayunstack_tensorlistfromtensort
psequential_39_bidirectional_39_forward_lstm_39_while_greater_sequential_39_bidirectional_39_forward_lstm_39_castt
asequential_39_bidirectional_39_forward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource:	Шv
csequential_39_bidirectional_39_forward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource:	2Шq
bsequential_39_bidirectional_39_forward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource:	ШЂYsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOpЂXsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOpЂZsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOpЁ
fsequential_39/bidirectional_39/forward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2h
fsequential_39/bidirectional_39/forward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeю
Xsequential_39/bidirectional_39/forward_lstm_39/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemБsequential_39_bidirectional_39_forward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_sequential_39_bidirectional_39_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0@sequential_39_bidirectional_39_forward_lstm_39_while_placeholderosequential_39/bidirectional_39/forward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02Z
Xsequential_39/bidirectional_39/forward_lstm_39/while/TensorArrayV2Read/TensorListGetItemы
<sequential_39/bidirectional_39/forward_lstm_39/while/GreaterGreaterrsequential_39_bidirectional_39_forward_lstm_39_while_greater_sequential_39_bidirectional_39_forward_lstm_39_cast_0@sequential_39_bidirectional_39_forward_lstm_39_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2>
<sequential_39/bidirectional_39/forward_lstm_39/while/Greaterй
Xsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOpcsequential_39_bidirectional_39_forward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02Z
Xsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp
Isequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMulMatMul_sequential_39/bidirectional_39/forward_lstm_39/while/TensorArrayV2Read/TensorListGetItem:item:0`sequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2K
Isequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMulп
Zsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOpesequential_39_bidirectional_39_forward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02\
Zsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOpџ
Ksequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMul_1MatMulBsequential_39_bidirectional_39_forward_lstm_39_while_placeholder_3bsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2M
Ksequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMul_1ј
Fsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/addAddV2Ssequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMul:product:0Usequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2H
Fsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/addи
Ysequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOpdsequential_39_bidirectional_39_forward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02[
Ysequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp
Jsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/BiasAddBiasAddJsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/add:z:0asequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2L
Jsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/BiasAddъ
Rsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2T
Rsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/split/split_dimЫ
Hsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/splitSplit[sequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/split/split_dim:output:0Ssequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2J
Hsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/splitЈ
Jsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/SigmoidSigmoidQsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22L
Jsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/SigmoidЌ
Lsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/Sigmoid_1SigmoidQsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22N
Lsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/Sigmoid_1п
Fsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/mulMulPsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/Sigmoid_1:y:0Bsequential_39_bidirectional_39_forward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22H
Fsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/mul
Gsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/ReluReluQsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22I
Gsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/Reluє
Hsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/mul_1MulNsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/Sigmoid:y:0Usequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22J
Hsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/mul_1щ
Hsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/add_1AddV2Jsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/mul:z:0Lsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22J
Hsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/add_1Ќ
Lsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/Sigmoid_2SigmoidQsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22N
Lsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/Sigmoid_2
Isequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/Relu_1ReluLsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22K
Isequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/Relu_1ј
Hsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/mul_2MulPsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/Sigmoid_2:y:0Wsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22J
Hsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/mul_2
;sequential_39/bidirectional_39/forward_lstm_39/while/SelectSelect@sequential_39/bidirectional_39/forward_lstm_39/while/Greater:z:0Lsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/mul_2:z:0Bsequential_39_bidirectional_39_forward_lstm_39_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22=
;sequential_39/bidirectional_39/forward_lstm_39/while/Select
=sequential_39/bidirectional_39/forward_lstm_39/while/Select_1Select@sequential_39/bidirectional_39/forward_lstm_39/while/Greater:z:0Lsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/mul_2:z:0Bsequential_39_bidirectional_39_forward_lstm_39_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22?
=sequential_39/bidirectional_39/forward_lstm_39/while/Select_1
=sequential_39/bidirectional_39/forward_lstm_39/while/Select_2Select@sequential_39/bidirectional_39/forward_lstm_39/while/Greater:z:0Lsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/add_1:z:0Bsequential_39_bidirectional_39_forward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22?
=sequential_39/bidirectional_39/forward_lstm_39/while/Select_2Ф
Ysequential_39/bidirectional_39/forward_lstm_39/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemBsequential_39_bidirectional_39_forward_lstm_39_while_placeholder_1@sequential_39_bidirectional_39_forward_lstm_39_while_placeholderDsequential_39/bidirectional_39/forward_lstm_39/while/Select:output:0*
_output_shapes
: *
element_dtype02[
Ysequential_39/bidirectional_39/forward_lstm_39/while/TensorArrayV2Write/TensorListSetItemК
:sequential_39/bidirectional_39/forward_lstm_39/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_39/bidirectional_39/forward_lstm_39/while/add/yЅ
8sequential_39/bidirectional_39/forward_lstm_39/while/addAddV2@sequential_39_bidirectional_39_forward_lstm_39_while_placeholderCsequential_39/bidirectional_39/forward_lstm_39/while/add/y:output:0*
T0*
_output_shapes
: 2:
8sequential_39/bidirectional_39/forward_lstm_39/while/addО
<sequential_39/bidirectional_39/forward_lstm_39/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2>
<sequential_39/bidirectional_39/forward_lstm_39/while/add_1/yс
:sequential_39/bidirectional_39/forward_lstm_39/while/add_1AddV2vsequential_39_bidirectional_39_forward_lstm_39_while_sequential_39_bidirectional_39_forward_lstm_39_while_loop_counterEsequential_39/bidirectional_39/forward_lstm_39/while/add_1/y:output:0*
T0*
_output_shapes
: 2<
:sequential_39/bidirectional_39/forward_lstm_39/while/add_1Ї
=sequential_39/bidirectional_39/forward_lstm_39/while/IdentityIdentity>sequential_39/bidirectional_39/forward_lstm_39/while/add_1:z:0:^sequential_39/bidirectional_39/forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2?
=sequential_39/bidirectional_39/forward_lstm_39/while/Identityщ
?sequential_39/bidirectional_39/forward_lstm_39/while/Identity_1Identity|sequential_39_bidirectional_39_forward_lstm_39_while_sequential_39_bidirectional_39_forward_lstm_39_while_maximum_iterations:^sequential_39/bidirectional_39/forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2A
?sequential_39/bidirectional_39/forward_lstm_39/while/Identity_1Љ
?sequential_39/bidirectional_39/forward_lstm_39/while/Identity_2Identity<sequential_39/bidirectional_39/forward_lstm_39/while/add:z:0:^sequential_39/bidirectional_39/forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2A
?sequential_39/bidirectional_39/forward_lstm_39/while/Identity_2ж
?sequential_39/bidirectional_39/forward_lstm_39/while/Identity_3Identityisequential_39/bidirectional_39/forward_lstm_39/while/TensorArrayV2Write/TensorListSetItem:output_handle:0:^sequential_39/bidirectional_39/forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2A
?sequential_39/bidirectional_39/forward_lstm_39/while/Identity_3Т
?sequential_39/bidirectional_39/forward_lstm_39/while/Identity_4IdentityDsequential_39/bidirectional_39/forward_lstm_39/while/Select:output:0:^sequential_39/bidirectional_39/forward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22A
?sequential_39/bidirectional_39/forward_lstm_39/while/Identity_4Ф
?sequential_39/bidirectional_39/forward_lstm_39/while/Identity_5IdentityFsequential_39/bidirectional_39/forward_lstm_39/while/Select_1:output:0:^sequential_39/bidirectional_39/forward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22A
?sequential_39/bidirectional_39/forward_lstm_39/while/Identity_5Ф
?sequential_39/bidirectional_39/forward_lstm_39/while/Identity_6IdentityFsequential_39/bidirectional_39/forward_lstm_39/while/Select_2:output:0:^sequential_39/bidirectional_39/forward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22A
?sequential_39/bidirectional_39/forward_lstm_39/while/Identity_6Ь
9sequential_39/bidirectional_39/forward_lstm_39/while/NoOpNoOpZ^sequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOpY^sequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp[^sequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2;
9sequential_39/bidirectional_39/forward_lstm_39/while/NoOp"ц
psequential_39_bidirectional_39_forward_lstm_39_while_greater_sequential_39_bidirectional_39_forward_lstm_39_castrsequential_39_bidirectional_39_forward_lstm_39_while_greater_sequential_39_bidirectional_39_forward_lstm_39_cast_0"
=sequential_39_bidirectional_39_forward_lstm_39_while_identityFsequential_39/bidirectional_39/forward_lstm_39/while/Identity:output:0"
?sequential_39_bidirectional_39_forward_lstm_39_while_identity_1Hsequential_39/bidirectional_39/forward_lstm_39/while/Identity_1:output:0"
?sequential_39_bidirectional_39_forward_lstm_39_while_identity_2Hsequential_39/bidirectional_39/forward_lstm_39/while/Identity_2:output:0"
?sequential_39_bidirectional_39_forward_lstm_39_while_identity_3Hsequential_39/bidirectional_39/forward_lstm_39/while/Identity_3:output:0"
?sequential_39_bidirectional_39_forward_lstm_39_while_identity_4Hsequential_39/bidirectional_39/forward_lstm_39/while/Identity_4:output:0"
?sequential_39_bidirectional_39_forward_lstm_39_while_identity_5Hsequential_39/bidirectional_39/forward_lstm_39/while/Identity_5:output:0"
?sequential_39_bidirectional_39_forward_lstm_39_while_identity_6Hsequential_39/bidirectional_39/forward_lstm_39/while/Identity_6:output:0"Ъ
bsequential_39_bidirectional_39_forward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resourcedsequential_39_bidirectional_39_forward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0"Ь
csequential_39_bidirectional_39_forward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resourceesequential_39_bidirectional_39_forward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0"Ш
asequential_39_bidirectional_39_forward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resourcecsequential_39_bidirectional_39_forward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0"ь
ssequential_39_bidirectional_39_forward_lstm_39_while_sequential_39_bidirectional_39_forward_lstm_39_strided_slice_1usequential_39_bidirectional_39_forward_lstm_39_while_sequential_39_bidirectional_39_forward_lstm_39_strided_slice_1_0"ц
Џsequential_39_bidirectional_39_forward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_sequential_39_bidirectional_39_forward_lstm_39_tensorarrayunstack_tensorlistfromtensorБsequential_39_bidirectional_39_forward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_sequential_39_bidirectional_39_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2Ж
Ysequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOpYsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp2Д
Xsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOpXsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp2И
Zsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOpZsequential_39/bidirectional_39/forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:џџџџџџџџџ
]
Њ
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_4930699

inputs?
,lstm_cell_118_matmul_readvariableop_resource:	ШA
.lstm_cell_118_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_118_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_118/BiasAdd/ReadVariableOpЂ#lstm_cell_118/MatMul/ReadVariableOpЂ%lstm_cell_118/MatMul_1/ReadVariableOpЂwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1
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
:џџџџџџџџџ22
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
B :ш2
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
zeros_1/packed/1
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
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
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
strided_slice_2/stack_2
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2
strided_slice_2И
#lstm_cell_118/MatMul/ReadVariableOpReadVariableOp,lstm_cell_118_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_118/MatMul/ReadVariableOpА
lstm_cell_118/MatMulMatMulstrided_slice_2:output:0+lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/MatMulО
%lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_118/MatMul_1/ReadVariableOpЌ
lstm_cell_118/MatMul_1MatMulzeros:output:0-lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/MatMul_1Є
lstm_cell_118/addAddV2lstm_cell_118/MatMul:product:0 lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/addЗ
$lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_118/BiasAdd/ReadVariableOpБ
lstm_cell_118/BiasAddBiasAddlstm_cell_118/add:z:0,lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/BiasAdd
lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_118/split/split_dimї
lstm_cell_118/splitSplit&lstm_cell_118/split/split_dim:output:0lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_118/split
lstm_cell_118/SigmoidSigmoidlstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Sigmoid
lstm_cell_118/Sigmoid_1Sigmoidlstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Sigmoid_1
lstm_cell_118/mulMullstm_cell_118/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/mul
lstm_cell_118/ReluRelulstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Relu 
lstm_cell_118/mul_1Mullstm_cell_118/Sigmoid:y:0 lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/mul_1
lstm_cell_118/add_1AddV2lstm_cell_118/mul:z:0lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/add_1
lstm_cell_118/Sigmoid_2Sigmoidlstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Sigmoid_2
lstm_cell_118/Relu_1Relulstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Relu_1Є
lstm_cell_118/mul_2Mullstm_cell_118/Sigmoid_2:y:0"lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_118_matmul_readvariableop_resource.lstm_cell_118_matmul_1_readvariableop_resource-lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_4930615*
condR
while_cond_4930614*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
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
:џџџџџџџџџ22

IdentityЫ
NoOpNoOp%^lstm_cell_118/BiasAdd/ReadVariableOp$^lstm_cell_118/MatMul/ReadVariableOp&^lstm_cell_118/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_118/BiasAdd/ReadVariableOp$lstm_cell_118/BiasAdd/ReadVariableOp2J
#lstm_cell_118/MatMul/ReadVariableOp#lstm_cell_118/MatMul/ReadVariableOp2N
%lstm_cell_118/MatMul_1/ReadVariableOp%lstm_cell_118/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

і
E__inference_dense_39_layer_call_and_return_conditional_losses_4931735

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
ч	

2__inference_bidirectional_39_layer_call_fn_4932364
inputs_0
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
	unknown_2:	Ш
	unknown_3:	2Ш
	unknown_4:	Ш
identityЂStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_49312722
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Њ

Ѓ
2__inference_bidirectional_39_layer_call_fn_4932400

inputs
inputs_1	
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
	unknown_2:	Ш
	unknown_3:	2Ш
	unknown_4:	Ш
identityЂStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_49321502
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
М

"forward_lstm_39_while_cond_4932768<
8forward_lstm_39_while_forward_lstm_39_while_loop_counterB
>forward_lstm_39_while_forward_lstm_39_while_maximum_iterations%
!forward_lstm_39_while_placeholder'
#forward_lstm_39_while_placeholder_1'
#forward_lstm_39_while_placeholder_2'
#forward_lstm_39_while_placeholder_3>
:forward_lstm_39_while_less_forward_lstm_39_strided_slice_1U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4932768___redundant_placeholder0U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4932768___redundant_placeholder1U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4932768___redundant_placeholder2U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4932768___redundant_placeholder3"
forward_lstm_39_while_identity
Р
forward_lstm_39/while/LessLess!forward_lstm_39_while_placeholder:forward_lstm_39_while_less_forward_lstm_39_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_39/while/Less
forward_lstm_39/while/IdentityIdentityforward_lstm_39/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_39/while/Identity"I
forward_lstm_39_while_identity'forward_lstm_39/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:

і
E__inference_dense_39_layer_call_and_return_conditional_losses_4933740

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
ЛИ
п
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_4933720

inputs
inputs_1	O
<forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource:	ШQ
>forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource:	2ШL
=forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource:	ШP
=backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource:	ШR
?backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource:	2ШM
>backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource:	Ш
identityЂ5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpЂ4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpЂ6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOpЂbackward_lstm_39/whileЂ4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpЂ3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOpЂ5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpЂforward_lstm_39/while
$forward_lstm_39/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$forward_lstm_39/RaggedToTensor/zeros
$forward_lstm_39/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2&
$forward_lstm_39/RaggedToTensor/Const
3forward_lstm_39/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-forward_lstm_39/RaggedToTensor/Const:output:0inputs-forward_lstm_39/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3forward_lstm_39/RaggedToTensor/RaggedTensorToTensorТ
:forward_lstm_39/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:forward_lstm_39/RaggedNestedRowLengths/strided_slice/stackЦ
<forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1Ц
<forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2Є
4forward_lstm_39/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cforward_lstm_39/RaggedNestedRowLengths/strided_slice/stack:output:0Eforward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1:output:0Eforward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask26
4forward_lstm_39/RaggedNestedRowLengths/strided_sliceЦ
<forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stackг
>forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2@
>forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1Ъ
>forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2А
6forward_lstm_39/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Eforward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gforward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gforward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask28
6forward_lstm_39/RaggedNestedRowLengths/strided_slice_1
*forward_lstm_39/RaggedNestedRowLengths/subSub=forward_lstm_39/RaggedNestedRowLengths/strided_slice:output:0?forward_lstm_39/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2,
*forward_lstm_39/RaggedNestedRowLengths/subЁ
forward_lstm_39/CastCast.forward_lstm_39/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_39/Cast
forward_lstm_39/ShapeShape<forward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_39/Shape
#forward_lstm_39/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_39/strided_slice/stack
%forward_lstm_39/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_39/strided_slice/stack_1
%forward_lstm_39/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_39/strided_slice/stack_2Т
forward_lstm_39/strided_sliceStridedSliceforward_lstm_39/Shape:output:0,forward_lstm_39/strided_slice/stack:output:0.forward_lstm_39/strided_slice/stack_1:output:0.forward_lstm_39/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_39/strided_slice|
forward_lstm_39/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_39/zeros/mul/yЌ
forward_lstm_39/zeros/mulMul&forward_lstm_39/strided_slice:output:0$forward_lstm_39/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros/mul
forward_lstm_39/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_39/zeros/Less/yЇ
forward_lstm_39/zeros/LessLessforward_lstm_39/zeros/mul:z:0%forward_lstm_39/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros/Less
forward_lstm_39/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_39/zeros/packed/1У
forward_lstm_39/zeros/packedPack&forward_lstm_39/strided_slice:output:0'forward_lstm_39/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_39/zeros/packed
forward_lstm_39/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_39/zeros/ConstЕ
forward_lstm_39/zerosFill%forward_lstm_39/zeros/packed:output:0$forward_lstm_39/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/zeros
forward_lstm_39/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_39/zeros_1/mul/yВ
forward_lstm_39/zeros_1/mulMul&forward_lstm_39/strided_slice:output:0&forward_lstm_39/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros_1/mul
forward_lstm_39/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
forward_lstm_39/zeros_1/Less/yЏ
forward_lstm_39/zeros_1/LessLessforward_lstm_39/zeros_1/mul:z:0'forward_lstm_39/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros_1/Less
 forward_lstm_39/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_39/zeros_1/packed/1Щ
forward_lstm_39/zeros_1/packedPack&forward_lstm_39/strided_slice:output:0)forward_lstm_39/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_39/zeros_1/packed
forward_lstm_39/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_39/zeros_1/ConstН
forward_lstm_39/zeros_1Fill'forward_lstm_39/zeros_1/packed:output:0&forward_lstm_39/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/zeros_1
forward_lstm_39/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_39/transpose/permщ
forward_lstm_39/transpose	Transpose<forward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0'forward_lstm_39/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_39/transpose
forward_lstm_39/Shape_1Shapeforward_lstm_39/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_39/Shape_1
%forward_lstm_39/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_39/strided_slice_1/stack
'forward_lstm_39/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_1/stack_1
'forward_lstm_39/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_1/stack_2Ю
forward_lstm_39/strided_slice_1StridedSlice forward_lstm_39/Shape_1:output:0.forward_lstm_39/strided_slice_1/stack:output:00forward_lstm_39/strided_slice_1/stack_1:output:00forward_lstm_39/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_39/strided_slice_1Ѕ
+forward_lstm_39/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+forward_lstm_39/TensorArrayV2/element_shapeђ
forward_lstm_39/TensorArrayV2TensorListReserve4forward_lstm_39/TensorArrayV2/element_shape:output:0(forward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_39/TensorArrayV2п
Eforward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2G
Eforward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
7forward_lstm_39/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_39/transpose:y:0Nforward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_39/TensorArrayUnstack/TensorListFromTensor
%forward_lstm_39/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_39/strided_slice_2/stack
'forward_lstm_39/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_2/stack_1
'forward_lstm_39/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_2/stack_2м
forward_lstm_39/strided_slice_2StridedSliceforward_lstm_39/transpose:y:0.forward_lstm_39/strided_slice_2/stack:output:00forward_lstm_39/strided_slice_2/stack_1:output:00forward_lstm_39/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2!
forward_lstm_39/strided_slice_2ш
3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp<forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype025
3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp№
$forward_lstm_39/lstm_cell_118/MatMulMatMul(forward_lstm_39/strided_slice_2:output:0;forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$forward_lstm_39/lstm_cell_118/MatMulю
5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp>forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype027
5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpь
&forward_lstm_39/lstm_cell_118/MatMul_1MatMulforward_lstm_39/zeros:output:0=forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_39/lstm_cell_118/MatMul_1ф
!forward_lstm_39/lstm_cell_118/addAddV2.forward_lstm_39/lstm_cell_118/MatMul:product:00forward_lstm_39/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2#
!forward_lstm_39/lstm_cell_118/addч
4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp=forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype026
4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpё
%forward_lstm_39/lstm_cell_118/BiasAddBiasAdd%forward_lstm_39/lstm_cell_118/add:z:0<forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_39/lstm_cell_118/BiasAdd 
-forward_lstm_39/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-forward_lstm_39/lstm_cell_118/split/split_dimЗ
#forward_lstm_39/lstm_cell_118/splitSplit6forward_lstm_39/lstm_cell_118/split/split_dim:output:0.forward_lstm_39/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2%
#forward_lstm_39/lstm_cell_118/splitЙ
%forward_lstm_39/lstm_cell_118/SigmoidSigmoid,forward_lstm_39/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_39/lstm_cell_118/SigmoidН
'forward_lstm_39/lstm_cell_118/Sigmoid_1Sigmoid,forward_lstm_39/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_39/lstm_cell_118/Sigmoid_1Ю
!forward_lstm_39/lstm_cell_118/mulMul+forward_lstm_39/lstm_cell_118/Sigmoid_1:y:0 forward_lstm_39/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_39/lstm_cell_118/mulА
"forward_lstm_39/lstm_cell_118/ReluRelu,forward_lstm_39/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_39/lstm_cell_118/Reluр
#forward_lstm_39/lstm_cell_118/mul_1Mul)forward_lstm_39/lstm_cell_118/Sigmoid:y:00forward_lstm_39/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_39/lstm_cell_118/mul_1е
#forward_lstm_39/lstm_cell_118/add_1AddV2%forward_lstm_39/lstm_cell_118/mul:z:0'forward_lstm_39/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_39/lstm_cell_118/add_1Н
'forward_lstm_39/lstm_cell_118/Sigmoid_2Sigmoid,forward_lstm_39/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_39/lstm_cell_118/Sigmoid_2Џ
$forward_lstm_39/lstm_cell_118/Relu_1Relu'forward_lstm_39/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_39/lstm_cell_118/Relu_1ф
#forward_lstm_39/lstm_cell_118/mul_2Mul+forward_lstm_39/lstm_cell_118/Sigmoid_2:y:02forward_lstm_39/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_39/lstm_cell_118/mul_2Џ
-forward_lstm_39/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2/
-forward_lstm_39/TensorArrayV2_1/element_shapeј
forward_lstm_39/TensorArrayV2_1TensorListReserve6forward_lstm_39/TensorArrayV2_1/element_shape:output:0(forward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_39/TensorArrayV2_1n
forward_lstm_39/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_39/time 
forward_lstm_39/zeros_like	ZerosLike'forward_lstm_39/lstm_cell_118/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/zeros_like
(forward_lstm_39/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(forward_lstm_39/while/maximum_iterations
"forward_lstm_39/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_39/while/loop_counter	
forward_lstm_39/whileWhile+forward_lstm_39/while/loop_counter:output:01forward_lstm_39/while/maximum_iterations:output:0forward_lstm_39/time:output:0(forward_lstm_39/TensorArrayV2_1:handle:0forward_lstm_39/zeros_like:y:0forward_lstm_39/zeros:output:0 forward_lstm_39/zeros_1:output:0(forward_lstm_39/strided_slice_1:output:0Gforward_lstm_39/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_39/Cast:y:0<forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource>forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource=forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *.
body&R$
"forward_lstm_39_while_body_4933444*.
cond&R$
"forward_lstm_39_while_cond_4933443*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_39/whileе
@forward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2B
@forward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeБ
2forward_lstm_39/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_39/while:output:3Iforward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype024
2forward_lstm_39/TensorArrayV2Stack/TensorListStackЁ
%forward_lstm_39/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2'
%forward_lstm_39/strided_slice_3/stack
'forward_lstm_39/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_39/strided_slice_3/stack_1
'forward_lstm_39/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_3/stack_2њ
forward_lstm_39/strided_slice_3StridedSlice;forward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_39/strided_slice_3/stack:output:00forward_lstm_39/strided_slice_3/stack_1:output:00forward_lstm_39/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2!
forward_lstm_39/strided_slice_3
 forward_lstm_39/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_39/transpose_1/permю
forward_lstm_39/transpose_1	Transpose;forward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_39/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_39/transpose_1
forward_lstm_39/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_39/runtime
%backward_lstm_39/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%backward_lstm_39/RaggedToTensor/zeros
%backward_lstm_39/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2'
%backward_lstm_39/RaggedToTensor/Const
4backward_lstm_39/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.backward_lstm_39/RaggedToTensor/Const:output:0inputs.backward_lstm_39/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4backward_lstm_39/RaggedToTensor/RaggedTensorToTensorФ
;backward_lstm_39/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;backward_lstm_39/RaggedNestedRowLengths/strided_slice/stackШ
=backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1Ш
=backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2Љ
5backward_lstm_39/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dbackward_lstm_39/RaggedNestedRowLengths/strided_slice/stack:output:0Fbackward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fbackward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask27
5backward_lstm_39/RaggedNestedRowLengths/strided_sliceШ
=backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stackе
?backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2A
?backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1Ь
?backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2Е
7backward_lstm_39/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fbackward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hbackward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hbackward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask29
7backward_lstm_39/RaggedNestedRowLengths/strided_slice_1
+backward_lstm_39/RaggedNestedRowLengths/subSub>backward_lstm_39/RaggedNestedRowLengths/strided_slice:output:0@backward_lstm_39/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2-
+backward_lstm_39/RaggedNestedRowLengths/subЄ
backward_lstm_39/CastCast/backward_lstm_39/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_39/Cast
backward_lstm_39/ShapeShape=backward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_39/Shape
$backward_lstm_39/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_39/strided_slice/stack
&backward_lstm_39/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_39/strided_slice/stack_1
&backward_lstm_39/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_39/strided_slice/stack_2Ш
backward_lstm_39/strided_sliceStridedSlicebackward_lstm_39/Shape:output:0-backward_lstm_39/strided_slice/stack:output:0/backward_lstm_39/strided_slice/stack_1:output:0/backward_lstm_39/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_39/strided_slice~
backward_lstm_39/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_39/zeros/mul/yА
backward_lstm_39/zeros/mulMul'backward_lstm_39/strided_slice:output:0%backward_lstm_39/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros/mul
backward_lstm_39/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
backward_lstm_39/zeros/Less/yЋ
backward_lstm_39/zeros/LessLessbackward_lstm_39/zeros/mul:z:0&backward_lstm_39/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros/Less
backward_lstm_39/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_39/zeros/packed/1Ч
backward_lstm_39/zeros/packedPack'backward_lstm_39/strided_slice:output:0(backward_lstm_39/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_39/zeros/packed
backward_lstm_39/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_39/zeros/ConstЙ
backward_lstm_39/zerosFill&backward_lstm_39/zeros/packed:output:0%backward_lstm_39/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/zeros
backward_lstm_39/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_39/zeros_1/mul/yЖ
backward_lstm_39/zeros_1/mulMul'backward_lstm_39/strided_slice:output:0'backward_lstm_39/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros_1/mul
backward_lstm_39/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
backward_lstm_39/zeros_1/Less/yГ
backward_lstm_39/zeros_1/LessLess backward_lstm_39/zeros_1/mul:z:0(backward_lstm_39/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros_1/Less
!backward_lstm_39/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_39/zeros_1/packed/1Э
backward_lstm_39/zeros_1/packedPack'backward_lstm_39/strided_slice:output:0*backward_lstm_39/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_39/zeros_1/packed
backward_lstm_39/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_39/zeros_1/ConstС
backward_lstm_39/zeros_1Fill(backward_lstm_39/zeros_1/packed:output:0'backward_lstm_39/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/zeros_1
backward_lstm_39/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_39/transpose/permэ
backward_lstm_39/transpose	Transpose=backward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0(backward_lstm_39/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_39/transpose
backward_lstm_39/Shape_1Shapebackward_lstm_39/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_39/Shape_1
&backward_lstm_39/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_39/strided_slice_1/stack
(backward_lstm_39/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_1/stack_1
(backward_lstm_39/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_1/stack_2д
 backward_lstm_39/strided_slice_1StridedSlice!backward_lstm_39/Shape_1:output:0/backward_lstm_39/strided_slice_1/stack:output:01backward_lstm_39/strided_slice_1/stack_1:output:01backward_lstm_39/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_39/strided_slice_1Ї
,backward_lstm_39/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,backward_lstm_39/TensorArrayV2/element_shapeі
backward_lstm_39/TensorArrayV2TensorListReserve5backward_lstm_39/TensorArrayV2/element_shape:output:0)backward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_39/TensorArrayV2
backward_lstm_39/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_39/ReverseV2/axisЮ
backward_lstm_39/ReverseV2	ReverseV2backward_lstm_39/transpose:y:0(backward_lstm_39/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_39/ReverseV2с
Fbackward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fbackward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeС
8backward_lstm_39/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_39/ReverseV2:output:0Obackward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_39/TensorArrayUnstack/TensorListFromTensor
&backward_lstm_39/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_39/strided_slice_2/stack
(backward_lstm_39/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_2/stack_1
(backward_lstm_39/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_2/stack_2т
 backward_lstm_39/strided_slice_2StridedSlicebackward_lstm_39/transpose:y:0/backward_lstm_39/strided_slice_2/stack:output:01backward_lstm_39/strided_slice_2/stack_1:output:01backward_lstm_39/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 backward_lstm_39/strided_slice_2ы
4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpReadVariableOp=backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpє
%backward_lstm_39/lstm_cell_119/MatMulMatMul)backward_lstm_39/strided_slice_2:output:0<backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%backward_lstm_39/lstm_cell_119/MatMulё
6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOp?backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp№
'backward_lstm_39/lstm_cell_119/MatMul_1MatMulbackward_lstm_39/zeros:output:0>backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_39/lstm_cell_119/MatMul_1ш
"backward_lstm_39/lstm_cell_119/addAddV2/backward_lstm_39/lstm_cell_119/MatMul:product:01backward_lstm_39/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"backward_lstm_39/lstm_cell_119/addъ
5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOp>backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpѕ
&backward_lstm_39/lstm_cell_119/BiasAddBiasAdd&backward_lstm_39/lstm_cell_119/add:z:0=backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_39/lstm_cell_119/BiasAddЂ
.backward_lstm_39/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.backward_lstm_39/lstm_cell_119/split/split_dimЛ
$backward_lstm_39/lstm_cell_119/splitSplit7backward_lstm_39/lstm_cell_119/split/split_dim:output:0/backward_lstm_39/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$backward_lstm_39/lstm_cell_119/splitМ
&backward_lstm_39/lstm_cell_119/SigmoidSigmoid-backward_lstm_39/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_39/lstm_cell_119/SigmoidР
(backward_lstm_39/lstm_cell_119/Sigmoid_1Sigmoid-backward_lstm_39/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_39/lstm_cell_119/Sigmoid_1в
"backward_lstm_39/lstm_cell_119/mulMul,backward_lstm_39/lstm_cell_119/Sigmoid_1:y:0!backward_lstm_39/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_39/lstm_cell_119/mulГ
#backward_lstm_39/lstm_cell_119/ReluRelu-backward_lstm_39/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_39/lstm_cell_119/Reluф
$backward_lstm_39/lstm_cell_119/mul_1Mul*backward_lstm_39/lstm_cell_119/Sigmoid:y:01backward_lstm_39/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_39/lstm_cell_119/mul_1й
$backward_lstm_39/lstm_cell_119/add_1AddV2&backward_lstm_39/lstm_cell_119/mul:z:0(backward_lstm_39/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_39/lstm_cell_119/add_1Р
(backward_lstm_39/lstm_cell_119/Sigmoid_2Sigmoid-backward_lstm_39/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_39/lstm_cell_119/Sigmoid_2В
%backward_lstm_39/lstm_cell_119/Relu_1Relu(backward_lstm_39/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_39/lstm_cell_119/Relu_1ш
$backward_lstm_39/lstm_cell_119/mul_2Mul,backward_lstm_39/lstm_cell_119/Sigmoid_2:y:03backward_lstm_39/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_39/lstm_cell_119/mul_2Б
.backward_lstm_39/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.backward_lstm_39/TensorArrayV2_1/element_shapeќ
 backward_lstm_39/TensorArrayV2_1TensorListReserve7backward_lstm_39/TensorArrayV2_1/element_shape:output:0)backward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_39/TensorArrayV2_1p
backward_lstm_39/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_39/time
&backward_lstm_39/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_lstm_39/Max/reduction_indices 
backward_lstm_39/MaxMaxbackward_lstm_39/Cast:y:0/backward_lstm_39/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/Maxr
backward_lstm_39/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_39/sub/y
backward_lstm_39/subSubbackward_lstm_39/Max:output:0backward_lstm_39/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/sub
backward_lstm_39/Sub_1Subbackward_lstm_39/sub:z:0backward_lstm_39/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_39/Sub_1Ѓ
backward_lstm_39/zeros_like	ZerosLike(backward_lstm_39/lstm_cell_119/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/zeros_likeЁ
)backward_lstm_39/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)backward_lstm_39/while/maximum_iterations
#backward_lstm_39/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_39/while/loop_counter	
backward_lstm_39/whileWhile,backward_lstm_39/while/loop_counter:output:02backward_lstm_39/while/maximum_iterations:output:0backward_lstm_39/time:output:0)backward_lstm_39/TensorArrayV2_1:handle:0backward_lstm_39/zeros_like:y:0backward_lstm_39/zeros:output:0!backward_lstm_39/zeros_1:output:0)backward_lstm_39/strided_slice_1:output:0Hbackward_lstm_39/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_39/Sub_1:z:0=backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource?backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource>backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( */
body'R%
#backward_lstm_39_while_body_4933623*/
cond'R%
#backward_lstm_39_while_cond_4933622*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_39/whileз
Abackward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Abackward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3backward_lstm_39/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_39/while:output:3Jbackward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3backward_lstm_39/TensorArrayV2Stack/TensorListStackЃ
&backward_lstm_39/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&backward_lstm_39/strided_slice_3/stack
(backward_lstm_39/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_39/strided_slice_3/stack_1
(backward_lstm_39/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_3/stack_2
 backward_lstm_39/strided_slice_3StridedSlice<backward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_39/strided_slice_3/stack:output:01backward_lstm_39/strided_slice_3/stack_1:output:01backward_lstm_39/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 backward_lstm_39/strided_slice_3
!backward_lstm_39/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_39/transpose_1/permђ
backward_lstm_39/transpose_1	Transpose<backward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_39/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_39/transpose_1
backward_lstm_39/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_39/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisТ
concatConcatV2(forward_lstm_39/strided_slice_3:output:0)backward_lstm_39/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

IdentityЬ
NoOpNoOp6^backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp5^backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp7^backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp^backward_lstm_39/while5^forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp4^forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp6^forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp^forward_lstm_39/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2n
5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp2l
4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp2p
6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp20
backward_lstm_39/whilebackward_lstm_39/while2l
4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp2j
3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp2n
5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp2.
forward_lstm_39/whileforward_lstm_39/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
сV
к
"forward_lstm_39_while_body_4932769<
8forward_lstm_39_while_forward_lstm_39_while_loop_counterB
>forward_lstm_39_while_forward_lstm_39_while_maximum_iterations%
!forward_lstm_39_while_placeholder'
#forward_lstm_39_while_placeholder_1'
#forward_lstm_39_while_placeholder_2'
#forward_lstm_39_while_placeholder_3;
7forward_lstm_39_while_forward_lstm_39_strided_slice_1_0w
sforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0W
Dforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0:	ШY
Fforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0:	2ШT
Eforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0:	Ш"
forward_lstm_39_while_identity$
 forward_lstm_39_while_identity_1$
 forward_lstm_39_while_identity_2$
 forward_lstm_39_while_identity_3$
 forward_lstm_39_while_identity_4$
 forward_lstm_39_while_identity_59
5forward_lstm_39_while_forward_lstm_39_strided_slice_1u
qforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensorU
Bforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource:	ШW
Dforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource:	2ШR
Cforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource:	ШЂ:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOpЂ9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOpЂ;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOpу
Gforward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2I
Gforward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeМ
9forward_lstm_39/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_39_while_placeholderPforward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02;
9forward_lstm_39/while/TensorArrayV2Read/TensorListGetItemќ
9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOpDforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02;
9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp
*forward_lstm_39/while/lstm_cell_118/MatMulMatMul@forward_lstm_39/while/TensorArrayV2Read/TensorListGetItem:item:0Aforward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2,
*forward_lstm_39/while/lstm_cell_118/MatMul
;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOpFforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02=
;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp
,forward_lstm_39/while/lstm_cell_118/MatMul_1MatMul#forward_lstm_39_while_placeholder_2Cforward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_39/while/lstm_cell_118/MatMul_1ќ
'forward_lstm_39/while/lstm_cell_118/addAddV24forward_lstm_39/while/lstm_cell_118/MatMul:product:06forward_lstm_39/while/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_39/while/lstm_cell_118/addћ
:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOpEforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02<
:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp
+forward_lstm_39/while/lstm_cell_118/BiasAddBiasAdd+forward_lstm_39/while/lstm_cell_118/add:z:0Bforward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_39/while/lstm_cell_118/BiasAddЌ
3forward_lstm_39/while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3forward_lstm_39/while/lstm_cell_118/split/split_dimЯ
)forward_lstm_39/while/lstm_cell_118/splitSplit<forward_lstm_39/while/lstm_cell_118/split/split_dim:output:04forward_lstm_39/while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2+
)forward_lstm_39/while/lstm_cell_118/splitЫ
+forward_lstm_39/while/lstm_cell_118/SigmoidSigmoid2forward_lstm_39/while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_39/while/lstm_cell_118/SigmoidЯ
-forward_lstm_39/while/lstm_cell_118/Sigmoid_1Sigmoid2forward_lstm_39/while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22/
-forward_lstm_39/while/lstm_cell_118/Sigmoid_1у
'forward_lstm_39/while/lstm_cell_118/mulMul1forward_lstm_39/while/lstm_cell_118/Sigmoid_1:y:0#forward_lstm_39_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_39/while/lstm_cell_118/mulТ
(forward_lstm_39/while/lstm_cell_118/ReluRelu2forward_lstm_39/while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_39/while/lstm_cell_118/Reluј
)forward_lstm_39/while/lstm_cell_118/mul_1Mul/forward_lstm_39/while/lstm_cell_118/Sigmoid:y:06forward_lstm_39/while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_39/while/lstm_cell_118/mul_1э
)forward_lstm_39/while/lstm_cell_118/add_1AddV2+forward_lstm_39/while/lstm_cell_118/mul:z:0-forward_lstm_39/while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_39/while/lstm_cell_118/add_1Я
-forward_lstm_39/while/lstm_cell_118/Sigmoid_2Sigmoid2forward_lstm_39/while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22/
-forward_lstm_39/while/lstm_cell_118/Sigmoid_2С
*forward_lstm_39/while/lstm_cell_118/Relu_1Relu-forward_lstm_39/while/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_39/while/lstm_cell_118/Relu_1ќ
)forward_lstm_39/while/lstm_cell_118/mul_2Mul1forward_lstm_39/while/lstm_cell_118/Sigmoid_2:y:08forward_lstm_39/while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_39/while/lstm_cell_118/mul_2Б
:forward_lstm_39/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_39_while_placeholder_1!forward_lstm_39_while_placeholder-forward_lstm_39/while/lstm_cell_118/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_39/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_39/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_39/while/add/yЉ
forward_lstm_39/while/addAddV2!forward_lstm_39_while_placeholder$forward_lstm_39/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/while/add
forward_lstm_39/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_39/while/add_1/yЦ
forward_lstm_39/while/add_1AddV28forward_lstm_39_while_forward_lstm_39_while_loop_counter&forward_lstm_39/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/while/add_1Ћ
forward_lstm_39/while/IdentityIdentityforward_lstm_39/while/add_1:z:0^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_39/while/IdentityЮ
 forward_lstm_39/while/Identity_1Identity>forward_lstm_39_while_forward_lstm_39_while_maximum_iterations^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_39/while/Identity_1­
 forward_lstm_39/while/Identity_2Identityforward_lstm_39/while/add:z:0^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_39/while/Identity_2к
 forward_lstm_39/while/Identity_3IdentityJforward_lstm_39/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_39/while/Identity_3Ю
 forward_lstm_39/while/Identity_4Identity-forward_lstm_39/while/lstm_cell_118/mul_2:z:0^forward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_39/while/Identity_4Ю
 forward_lstm_39/while/Identity_5Identity-forward_lstm_39/while/lstm_cell_118/add_1:z:0^forward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_39/while/Identity_5Б
forward_lstm_39/while/NoOpNoOp;^forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp:^forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp<^forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_39/while/NoOp"p
5forward_lstm_39_while_forward_lstm_39_strided_slice_17forward_lstm_39_while_forward_lstm_39_strided_slice_1_0"I
forward_lstm_39_while_identity'forward_lstm_39/while/Identity:output:0"M
 forward_lstm_39_while_identity_1)forward_lstm_39/while/Identity_1:output:0"M
 forward_lstm_39_while_identity_2)forward_lstm_39/while/Identity_2:output:0"M
 forward_lstm_39_while_identity_3)forward_lstm_39/while/Identity_3:output:0"M
 forward_lstm_39_while_identity_4)forward_lstm_39/while/Identity_4:output:0"M
 forward_lstm_39_while_identity_5)forward_lstm_39/while/Identity_5:output:0"
Cforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resourceEforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0"
Dforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resourceFforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0"
Bforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resourceDforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0"ш
qforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensorsforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2x
:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp2v
9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp2z
;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
ј?
к
while_body_4934960
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_119_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_119_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_119_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_119_matmul_readvariableop_resource:	ШG
4while_lstm_cell_119_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_119_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_119/BiasAdd/ReadVariableOpЂ)while/lstm_cell_119/MatMul/ReadVariableOpЂ+while/lstm_cell_119/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeм
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЬ
)while/lstm_cell_119/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_119_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_119/MatMul/ReadVariableOpк
while/lstm_cell_119/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/MatMulв
+while/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_119_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_119/MatMul_1/ReadVariableOpУ
while/lstm_cell_119/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/MatMul_1М
while/lstm_cell_119/addAddV2$while/lstm_cell_119/MatMul:product:0&while/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/addЫ
*while/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_119_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_119/BiasAdd/ReadVariableOpЩ
while/lstm_cell_119/BiasAddBiasAddwhile/lstm_cell_119/add:z:02while/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/BiasAdd
#while/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_119/split/split_dim
while/lstm_cell_119/splitSplit,while/lstm_cell_119/split/split_dim:output:0$while/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_119/split
while/lstm_cell_119/SigmoidSigmoid"while/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Sigmoid
while/lstm_cell_119/Sigmoid_1Sigmoid"while/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Sigmoid_1Ѓ
while/lstm_cell_119/mulMul!while/lstm_cell_119/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/mul
while/lstm_cell_119/ReluRelu"while/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/ReluИ
while/lstm_cell_119/mul_1Mulwhile/lstm_cell_119/Sigmoid:y:0&while/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/mul_1­
while/lstm_cell_119/add_1AddV2while/lstm_cell_119/mul:z:0while/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/add_1
while/lstm_cell_119/Sigmoid_2Sigmoid"while/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Sigmoid_2
while/lstm_cell_119/Relu_1Reluwhile/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Relu_1М
while/lstm_cell_119/mul_2Mul!while/lstm_cell_119/Sigmoid_2:y:0(while/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_119/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_119/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_119/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_119/BiasAdd/ReadVariableOp*^while/lstm_cell_119/MatMul/ReadVariableOp,^while/lstm_cell_119/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_119_biasadd_readvariableop_resource5while_lstm_cell_119_biasadd_readvariableop_resource_0"n
4while_lstm_cell_119_matmul_1_readvariableop_resource6while_lstm_cell_119_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_119_matmul_readvariableop_resource4while_lstm_cell_119_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_119/BiasAdd/ReadVariableOp*while/lstm_cell_119/BiasAdd/ReadVariableOp2V
)while/lstm_cell_119/MatMul/ReadVariableOp)while/lstm_cell_119/MatMul/ReadVariableOp2Z
+while/lstm_cell_119/MatMul_1/ReadVariableOp+while/lstm_cell_119/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Ђ

"forward_lstm_39_while_cond_4933085<
8forward_lstm_39_while_forward_lstm_39_while_loop_counterB
>forward_lstm_39_while_forward_lstm_39_while_maximum_iterations%
!forward_lstm_39_while_placeholder'
#forward_lstm_39_while_placeholder_1'
#forward_lstm_39_while_placeholder_2'
#forward_lstm_39_while_placeholder_3'
#forward_lstm_39_while_placeholder_4>
:forward_lstm_39_while_less_forward_lstm_39_strided_slice_1U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4933085___redundant_placeholder0U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4933085___redundant_placeholder1U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4933085___redundant_placeholder2U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4933085___redundant_placeholder3U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4933085___redundant_placeholder4"
forward_lstm_39_while_identity
Р
forward_lstm_39/while/LessLess!forward_lstm_39_while_placeholder:forward_lstm_39_while_less_forward_lstm_39_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_39/while/Less
forward_lstm_39/while/IdentityIdentityforward_lstm_39/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_39/while/Identity"I
forward_lstm_39_while_identity'forward_lstm_39/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: :::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
к
Ш
while_cond_4933850
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_4933850___redundant_placeholder05
1while_while_cond_4933850___redundant_placeholder15
1while_while_cond_4933850___redundant_placeholder25
1while_while_cond_4933850___redundant_placeholder3
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
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
я?
к
while_body_4934501
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_119_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_119_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_119_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_119_matmul_readvariableop_resource:	ШG
4while_lstm_cell_119_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_119_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_119/BiasAdd/ReadVariableOpЂ)while/lstm_cell_119/MatMul/ReadVariableOpЂ+while/lstm_cell_119/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЬ
)while/lstm_cell_119/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_119_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_119/MatMul/ReadVariableOpк
while/lstm_cell_119/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/MatMulв
+while/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_119_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_119/MatMul_1/ReadVariableOpУ
while/lstm_cell_119/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/MatMul_1М
while/lstm_cell_119/addAddV2$while/lstm_cell_119/MatMul:product:0&while/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/addЫ
*while/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_119_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_119/BiasAdd/ReadVariableOpЩ
while/lstm_cell_119/BiasAddBiasAddwhile/lstm_cell_119/add:z:02while/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/BiasAdd
#while/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_119/split/split_dim
while/lstm_cell_119/splitSplit,while/lstm_cell_119/split/split_dim:output:0$while/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_119/split
while/lstm_cell_119/SigmoidSigmoid"while/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Sigmoid
while/lstm_cell_119/Sigmoid_1Sigmoid"while/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Sigmoid_1Ѓ
while/lstm_cell_119/mulMul!while/lstm_cell_119/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/mul
while/lstm_cell_119/ReluRelu"while/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/ReluИ
while/lstm_cell_119/mul_1Mulwhile/lstm_cell_119/Sigmoid:y:0&while/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/mul_1­
while/lstm_cell_119/add_1AddV2while/lstm_cell_119/mul:z:0while/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/add_1
while/lstm_cell_119/Sigmoid_2Sigmoid"while/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Sigmoid_2
while/lstm_cell_119/Relu_1Reluwhile/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Relu_1М
while/lstm_cell_119/mul_2Mul!while/lstm_cell_119/Sigmoid_2:y:0(while/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_119/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_119/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_119/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_119/BiasAdd/ReadVariableOp*^while/lstm_cell_119/MatMul/ReadVariableOp,^while/lstm_cell_119/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_119_biasadd_readvariableop_resource5while_lstm_cell_119_biasadd_readvariableop_resource_0"n
4while_lstm_cell_119_matmul_1_readvariableop_resource6while_lstm_cell_119_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_119_matmul_readvariableop_resource4while_lstm_cell_119_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_119/BiasAdd/ReadVariableOp*while/lstm_cell_119/BiasAdd/ReadVariableOp2V
)while/lstm_cell_119/MatMul/ReadVariableOp)while/lstm_cell_119/MatMul/ReadVariableOp2Z
+while/lstm_cell_119/MatMul_1/ReadVariableOp+while/lstm_cell_119/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Њ

Ѓ
2__inference_bidirectional_39_layer_call_fn_4932382

inputs
inputs_1	
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
	unknown_2:	Ш
	unknown_3:	2Ш
	unknown_4:	Ш
identityЂStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_49317102
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ў\
Ќ
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_4933935
inputs_0?
,lstm_cell_118_matmul_readvariableop_resource:	ШA
.lstm_cell_118_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_118_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_118/BiasAdd/ReadVariableOpЂ#lstm_cell_118/MatMul/ReadVariableOpЂ%lstm_cell_118/MatMul_1/ReadVariableOpЂwhileF
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1
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
:џџџџџџџџџ22
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
B :ш2
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
zeros_1/packed/1
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
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
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
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2И
#lstm_cell_118/MatMul/ReadVariableOpReadVariableOp,lstm_cell_118_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_118/MatMul/ReadVariableOpА
lstm_cell_118/MatMulMatMulstrided_slice_2:output:0+lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/MatMulО
%lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_118/MatMul_1/ReadVariableOpЌ
lstm_cell_118/MatMul_1MatMulzeros:output:0-lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/MatMul_1Є
lstm_cell_118/addAddV2lstm_cell_118/MatMul:product:0 lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/addЗ
$lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_118/BiasAdd/ReadVariableOpБ
lstm_cell_118/BiasAddBiasAddlstm_cell_118/add:z:0,lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/BiasAdd
lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_118/split/split_dimї
lstm_cell_118/splitSplit&lstm_cell_118/split/split_dim:output:0lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_118/split
lstm_cell_118/SigmoidSigmoidlstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Sigmoid
lstm_cell_118/Sigmoid_1Sigmoidlstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Sigmoid_1
lstm_cell_118/mulMullstm_cell_118/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/mul
lstm_cell_118/ReluRelulstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Relu 
lstm_cell_118/mul_1Mullstm_cell_118/Sigmoid:y:0 lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/mul_1
lstm_cell_118/add_1AddV2lstm_cell_118/mul:z:0lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/add_1
lstm_cell_118/Sigmoid_2Sigmoidlstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Sigmoid_2
lstm_cell_118/Relu_1Relulstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Relu_1Є
lstm_cell_118/mul_2Mullstm_cell_118/Sigmoid_2:y:0"lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_118_matmul_readvariableop_resource.lstm_cell_118_matmul_1_readvariableop_resource-lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_4933851*
condR
while_cond_4933850*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
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
:џџџџџџџџџ22

IdentityЫ
NoOpNoOp%^lstm_cell_118/BiasAdd/ReadVariableOp$^lstm_cell_118/MatMul/ReadVariableOp&^lstm_cell_118/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_118/BiasAdd/ReadVariableOp$lstm_cell_118/BiasAdd/ReadVariableOp2J
#lstm_cell_118/MatMul/ReadVariableOp#lstm_cell_118/MatMul/ReadVariableOp2N
%lstm_cell_118/MatMul_1/ReadVariableOp%lstm_cell_118/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
У

#backward_lstm_39_while_cond_4931612>
:backward_lstm_39_while_backward_lstm_39_while_loop_counterD
@backward_lstm_39_while_backward_lstm_39_while_maximum_iterations&
"backward_lstm_39_while_placeholder(
$backward_lstm_39_while_placeholder_1(
$backward_lstm_39_while_placeholder_2(
$backward_lstm_39_while_placeholder_3(
$backward_lstm_39_while_placeholder_4@
<backward_lstm_39_while_less_backward_lstm_39_strided_slice_1W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4931612___redundant_placeholder0W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4931612___redundant_placeholder1W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4931612___redundant_placeholder2W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4931612___redundant_placeholder3W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4931612___redundant_placeholder4#
backward_lstm_39_while_identity
Х
backward_lstm_39/while/LessLess"backward_lstm_39_while_placeholder<backward_lstm_39_while_less_backward_lstm_39_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_39/while/Less
backward_lstm_39/while/IdentityIdentitybackward_lstm_39/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_39/while/Identity"K
backward_lstm_39_while_identity(backward_lstm_39/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: :::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
ЛИ
п
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_4933362

inputs
inputs_1	O
<forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource:	ШQ
>forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource:	2ШL
=forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource:	ШP
=backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource:	ШR
?backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource:	2ШM
>backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource:	Ш
identityЂ5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpЂ4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpЂ6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOpЂbackward_lstm_39/whileЂ4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpЂ3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOpЂ5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpЂforward_lstm_39/while
$forward_lstm_39/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$forward_lstm_39/RaggedToTensor/zeros
$forward_lstm_39/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2&
$forward_lstm_39/RaggedToTensor/Const
3forward_lstm_39/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-forward_lstm_39/RaggedToTensor/Const:output:0inputs-forward_lstm_39/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3forward_lstm_39/RaggedToTensor/RaggedTensorToTensorТ
:forward_lstm_39/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:forward_lstm_39/RaggedNestedRowLengths/strided_slice/stackЦ
<forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1Ц
<forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<forward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2Є
4forward_lstm_39/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cforward_lstm_39/RaggedNestedRowLengths/strided_slice/stack:output:0Eforward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1:output:0Eforward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask26
4forward_lstm_39/RaggedNestedRowLengths/strided_sliceЦ
<forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stackг
>forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2@
>forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1Ъ
>forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>forward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2А
6forward_lstm_39/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Eforward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gforward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gforward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask28
6forward_lstm_39/RaggedNestedRowLengths/strided_slice_1
*forward_lstm_39/RaggedNestedRowLengths/subSub=forward_lstm_39/RaggedNestedRowLengths/strided_slice:output:0?forward_lstm_39/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2,
*forward_lstm_39/RaggedNestedRowLengths/subЁ
forward_lstm_39/CastCast.forward_lstm_39/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_39/Cast
forward_lstm_39/ShapeShape<forward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_39/Shape
#forward_lstm_39/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_39/strided_slice/stack
%forward_lstm_39/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_39/strided_slice/stack_1
%forward_lstm_39/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_39/strided_slice/stack_2Т
forward_lstm_39/strided_sliceStridedSliceforward_lstm_39/Shape:output:0,forward_lstm_39/strided_slice/stack:output:0.forward_lstm_39/strided_slice/stack_1:output:0.forward_lstm_39/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_39/strided_slice|
forward_lstm_39/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_39/zeros/mul/yЌ
forward_lstm_39/zeros/mulMul&forward_lstm_39/strided_slice:output:0$forward_lstm_39/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros/mul
forward_lstm_39/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_39/zeros/Less/yЇ
forward_lstm_39/zeros/LessLessforward_lstm_39/zeros/mul:z:0%forward_lstm_39/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros/Less
forward_lstm_39/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_39/zeros/packed/1У
forward_lstm_39/zeros/packedPack&forward_lstm_39/strided_slice:output:0'forward_lstm_39/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_39/zeros/packed
forward_lstm_39/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_39/zeros/ConstЕ
forward_lstm_39/zerosFill%forward_lstm_39/zeros/packed:output:0$forward_lstm_39/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/zeros
forward_lstm_39/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_39/zeros_1/mul/yВ
forward_lstm_39/zeros_1/mulMul&forward_lstm_39/strided_slice:output:0&forward_lstm_39/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros_1/mul
forward_lstm_39/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
forward_lstm_39/zeros_1/Less/yЏ
forward_lstm_39/zeros_1/LessLessforward_lstm_39/zeros_1/mul:z:0'forward_lstm_39/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros_1/Less
 forward_lstm_39/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_39/zeros_1/packed/1Щ
forward_lstm_39/zeros_1/packedPack&forward_lstm_39/strided_slice:output:0)forward_lstm_39/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_39/zeros_1/packed
forward_lstm_39/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_39/zeros_1/ConstН
forward_lstm_39/zeros_1Fill'forward_lstm_39/zeros_1/packed:output:0&forward_lstm_39/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/zeros_1
forward_lstm_39/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_39/transpose/permщ
forward_lstm_39/transpose	Transpose<forward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0'forward_lstm_39/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_39/transpose
forward_lstm_39/Shape_1Shapeforward_lstm_39/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_39/Shape_1
%forward_lstm_39/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_39/strided_slice_1/stack
'forward_lstm_39/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_1/stack_1
'forward_lstm_39/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_1/stack_2Ю
forward_lstm_39/strided_slice_1StridedSlice forward_lstm_39/Shape_1:output:0.forward_lstm_39/strided_slice_1/stack:output:00forward_lstm_39/strided_slice_1/stack_1:output:00forward_lstm_39/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_39/strided_slice_1Ѕ
+forward_lstm_39/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+forward_lstm_39/TensorArrayV2/element_shapeђ
forward_lstm_39/TensorArrayV2TensorListReserve4forward_lstm_39/TensorArrayV2/element_shape:output:0(forward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_39/TensorArrayV2п
Eforward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2G
Eforward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
7forward_lstm_39/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_39/transpose:y:0Nforward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_39/TensorArrayUnstack/TensorListFromTensor
%forward_lstm_39/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_39/strided_slice_2/stack
'forward_lstm_39/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_2/stack_1
'forward_lstm_39/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_2/stack_2м
forward_lstm_39/strided_slice_2StridedSliceforward_lstm_39/transpose:y:0.forward_lstm_39/strided_slice_2/stack:output:00forward_lstm_39/strided_slice_2/stack_1:output:00forward_lstm_39/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2!
forward_lstm_39/strided_slice_2ш
3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp<forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype025
3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp№
$forward_lstm_39/lstm_cell_118/MatMulMatMul(forward_lstm_39/strided_slice_2:output:0;forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$forward_lstm_39/lstm_cell_118/MatMulю
5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp>forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype027
5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpь
&forward_lstm_39/lstm_cell_118/MatMul_1MatMulforward_lstm_39/zeros:output:0=forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_39/lstm_cell_118/MatMul_1ф
!forward_lstm_39/lstm_cell_118/addAddV2.forward_lstm_39/lstm_cell_118/MatMul:product:00forward_lstm_39/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2#
!forward_lstm_39/lstm_cell_118/addч
4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp=forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype026
4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpё
%forward_lstm_39/lstm_cell_118/BiasAddBiasAdd%forward_lstm_39/lstm_cell_118/add:z:0<forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_39/lstm_cell_118/BiasAdd 
-forward_lstm_39/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-forward_lstm_39/lstm_cell_118/split/split_dimЗ
#forward_lstm_39/lstm_cell_118/splitSplit6forward_lstm_39/lstm_cell_118/split/split_dim:output:0.forward_lstm_39/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2%
#forward_lstm_39/lstm_cell_118/splitЙ
%forward_lstm_39/lstm_cell_118/SigmoidSigmoid,forward_lstm_39/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_39/lstm_cell_118/SigmoidН
'forward_lstm_39/lstm_cell_118/Sigmoid_1Sigmoid,forward_lstm_39/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_39/lstm_cell_118/Sigmoid_1Ю
!forward_lstm_39/lstm_cell_118/mulMul+forward_lstm_39/lstm_cell_118/Sigmoid_1:y:0 forward_lstm_39/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_39/lstm_cell_118/mulА
"forward_lstm_39/lstm_cell_118/ReluRelu,forward_lstm_39/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_39/lstm_cell_118/Reluр
#forward_lstm_39/lstm_cell_118/mul_1Mul)forward_lstm_39/lstm_cell_118/Sigmoid:y:00forward_lstm_39/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_39/lstm_cell_118/mul_1е
#forward_lstm_39/lstm_cell_118/add_1AddV2%forward_lstm_39/lstm_cell_118/mul:z:0'forward_lstm_39/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_39/lstm_cell_118/add_1Н
'forward_lstm_39/lstm_cell_118/Sigmoid_2Sigmoid,forward_lstm_39/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_39/lstm_cell_118/Sigmoid_2Џ
$forward_lstm_39/lstm_cell_118/Relu_1Relu'forward_lstm_39/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_39/lstm_cell_118/Relu_1ф
#forward_lstm_39/lstm_cell_118/mul_2Mul+forward_lstm_39/lstm_cell_118/Sigmoid_2:y:02forward_lstm_39/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_39/lstm_cell_118/mul_2Џ
-forward_lstm_39/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2/
-forward_lstm_39/TensorArrayV2_1/element_shapeј
forward_lstm_39/TensorArrayV2_1TensorListReserve6forward_lstm_39/TensorArrayV2_1/element_shape:output:0(forward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_39/TensorArrayV2_1n
forward_lstm_39/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_39/time 
forward_lstm_39/zeros_like	ZerosLike'forward_lstm_39/lstm_cell_118/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/zeros_like
(forward_lstm_39/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(forward_lstm_39/while/maximum_iterations
"forward_lstm_39/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_39/while/loop_counter	
forward_lstm_39/whileWhile+forward_lstm_39/while/loop_counter:output:01forward_lstm_39/while/maximum_iterations:output:0forward_lstm_39/time:output:0(forward_lstm_39/TensorArrayV2_1:handle:0forward_lstm_39/zeros_like:y:0forward_lstm_39/zeros:output:0 forward_lstm_39/zeros_1:output:0(forward_lstm_39/strided_slice_1:output:0Gforward_lstm_39/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_39/Cast:y:0<forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource>forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource=forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *.
body&R$
"forward_lstm_39_while_body_4933086*.
cond&R$
"forward_lstm_39_while_cond_4933085*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_39/whileе
@forward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2B
@forward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeБ
2forward_lstm_39/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_39/while:output:3Iforward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype024
2forward_lstm_39/TensorArrayV2Stack/TensorListStackЁ
%forward_lstm_39/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2'
%forward_lstm_39/strided_slice_3/stack
'forward_lstm_39/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_39/strided_slice_3/stack_1
'forward_lstm_39/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_3/stack_2њ
forward_lstm_39/strided_slice_3StridedSlice;forward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_39/strided_slice_3/stack:output:00forward_lstm_39/strided_slice_3/stack_1:output:00forward_lstm_39/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2!
forward_lstm_39/strided_slice_3
 forward_lstm_39/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_39/transpose_1/permю
forward_lstm_39/transpose_1	Transpose;forward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_39/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_39/transpose_1
forward_lstm_39/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_39/runtime
%backward_lstm_39/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%backward_lstm_39/RaggedToTensor/zeros
%backward_lstm_39/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2'
%backward_lstm_39/RaggedToTensor/Const
4backward_lstm_39/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.backward_lstm_39/RaggedToTensor/Const:output:0inputs.backward_lstm_39/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4backward_lstm_39/RaggedToTensor/RaggedTensorToTensorФ
;backward_lstm_39/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;backward_lstm_39/RaggedNestedRowLengths/strided_slice/stackШ
=backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1Ш
=backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=backward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2Љ
5backward_lstm_39/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dbackward_lstm_39/RaggedNestedRowLengths/strided_slice/stack:output:0Fbackward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fbackward_lstm_39/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask27
5backward_lstm_39/RaggedNestedRowLengths/strided_sliceШ
=backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stackе
?backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2A
?backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1Ь
?backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?backward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2Е
7backward_lstm_39/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fbackward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hbackward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hbackward_lstm_39/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask29
7backward_lstm_39/RaggedNestedRowLengths/strided_slice_1
+backward_lstm_39/RaggedNestedRowLengths/subSub>backward_lstm_39/RaggedNestedRowLengths/strided_slice:output:0@backward_lstm_39/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2-
+backward_lstm_39/RaggedNestedRowLengths/subЄ
backward_lstm_39/CastCast/backward_lstm_39/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_39/Cast
backward_lstm_39/ShapeShape=backward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_39/Shape
$backward_lstm_39/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_39/strided_slice/stack
&backward_lstm_39/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_39/strided_slice/stack_1
&backward_lstm_39/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_39/strided_slice/stack_2Ш
backward_lstm_39/strided_sliceStridedSlicebackward_lstm_39/Shape:output:0-backward_lstm_39/strided_slice/stack:output:0/backward_lstm_39/strided_slice/stack_1:output:0/backward_lstm_39/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_39/strided_slice~
backward_lstm_39/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_39/zeros/mul/yА
backward_lstm_39/zeros/mulMul'backward_lstm_39/strided_slice:output:0%backward_lstm_39/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros/mul
backward_lstm_39/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
backward_lstm_39/zeros/Less/yЋ
backward_lstm_39/zeros/LessLessbackward_lstm_39/zeros/mul:z:0&backward_lstm_39/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros/Less
backward_lstm_39/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_39/zeros/packed/1Ч
backward_lstm_39/zeros/packedPack'backward_lstm_39/strided_slice:output:0(backward_lstm_39/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_39/zeros/packed
backward_lstm_39/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_39/zeros/ConstЙ
backward_lstm_39/zerosFill&backward_lstm_39/zeros/packed:output:0%backward_lstm_39/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/zeros
backward_lstm_39/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_39/zeros_1/mul/yЖ
backward_lstm_39/zeros_1/mulMul'backward_lstm_39/strided_slice:output:0'backward_lstm_39/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros_1/mul
backward_lstm_39/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
backward_lstm_39/zeros_1/Less/yГ
backward_lstm_39/zeros_1/LessLess backward_lstm_39/zeros_1/mul:z:0(backward_lstm_39/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros_1/Less
!backward_lstm_39/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_39/zeros_1/packed/1Э
backward_lstm_39/zeros_1/packedPack'backward_lstm_39/strided_slice:output:0*backward_lstm_39/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_39/zeros_1/packed
backward_lstm_39/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_39/zeros_1/ConstС
backward_lstm_39/zeros_1Fill(backward_lstm_39/zeros_1/packed:output:0'backward_lstm_39/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/zeros_1
backward_lstm_39/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_39/transpose/permэ
backward_lstm_39/transpose	Transpose=backward_lstm_39/RaggedToTensor/RaggedTensorToTensor:result:0(backward_lstm_39/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_39/transpose
backward_lstm_39/Shape_1Shapebackward_lstm_39/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_39/Shape_1
&backward_lstm_39/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_39/strided_slice_1/stack
(backward_lstm_39/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_1/stack_1
(backward_lstm_39/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_1/stack_2д
 backward_lstm_39/strided_slice_1StridedSlice!backward_lstm_39/Shape_1:output:0/backward_lstm_39/strided_slice_1/stack:output:01backward_lstm_39/strided_slice_1/stack_1:output:01backward_lstm_39/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_39/strided_slice_1Ї
,backward_lstm_39/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,backward_lstm_39/TensorArrayV2/element_shapeі
backward_lstm_39/TensorArrayV2TensorListReserve5backward_lstm_39/TensorArrayV2/element_shape:output:0)backward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_39/TensorArrayV2
backward_lstm_39/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_39/ReverseV2/axisЮ
backward_lstm_39/ReverseV2	ReverseV2backward_lstm_39/transpose:y:0(backward_lstm_39/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_39/ReverseV2с
Fbackward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fbackward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeС
8backward_lstm_39/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_39/ReverseV2:output:0Obackward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_39/TensorArrayUnstack/TensorListFromTensor
&backward_lstm_39/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_39/strided_slice_2/stack
(backward_lstm_39/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_2/stack_1
(backward_lstm_39/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_2/stack_2т
 backward_lstm_39/strided_slice_2StridedSlicebackward_lstm_39/transpose:y:0/backward_lstm_39/strided_slice_2/stack:output:01backward_lstm_39/strided_slice_2/stack_1:output:01backward_lstm_39/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 backward_lstm_39/strided_slice_2ы
4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpReadVariableOp=backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpє
%backward_lstm_39/lstm_cell_119/MatMulMatMul)backward_lstm_39/strided_slice_2:output:0<backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%backward_lstm_39/lstm_cell_119/MatMulё
6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOp?backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp№
'backward_lstm_39/lstm_cell_119/MatMul_1MatMulbackward_lstm_39/zeros:output:0>backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_39/lstm_cell_119/MatMul_1ш
"backward_lstm_39/lstm_cell_119/addAddV2/backward_lstm_39/lstm_cell_119/MatMul:product:01backward_lstm_39/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"backward_lstm_39/lstm_cell_119/addъ
5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOp>backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpѕ
&backward_lstm_39/lstm_cell_119/BiasAddBiasAdd&backward_lstm_39/lstm_cell_119/add:z:0=backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_39/lstm_cell_119/BiasAddЂ
.backward_lstm_39/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.backward_lstm_39/lstm_cell_119/split/split_dimЛ
$backward_lstm_39/lstm_cell_119/splitSplit7backward_lstm_39/lstm_cell_119/split/split_dim:output:0/backward_lstm_39/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$backward_lstm_39/lstm_cell_119/splitМ
&backward_lstm_39/lstm_cell_119/SigmoidSigmoid-backward_lstm_39/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_39/lstm_cell_119/SigmoidР
(backward_lstm_39/lstm_cell_119/Sigmoid_1Sigmoid-backward_lstm_39/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_39/lstm_cell_119/Sigmoid_1в
"backward_lstm_39/lstm_cell_119/mulMul,backward_lstm_39/lstm_cell_119/Sigmoid_1:y:0!backward_lstm_39/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_39/lstm_cell_119/mulГ
#backward_lstm_39/lstm_cell_119/ReluRelu-backward_lstm_39/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_39/lstm_cell_119/Reluф
$backward_lstm_39/lstm_cell_119/mul_1Mul*backward_lstm_39/lstm_cell_119/Sigmoid:y:01backward_lstm_39/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_39/lstm_cell_119/mul_1й
$backward_lstm_39/lstm_cell_119/add_1AddV2&backward_lstm_39/lstm_cell_119/mul:z:0(backward_lstm_39/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_39/lstm_cell_119/add_1Р
(backward_lstm_39/lstm_cell_119/Sigmoid_2Sigmoid-backward_lstm_39/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_39/lstm_cell_119/Sigmoid_2В
%backward_lstm_39/lstm_cell_119/Relu_1Relu(backward_lstm_39/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_39/lstm_cell_119/Relu_1ш
$backward_lstm_39/lstm_cell_119/mul_2Mul,backward_lstm_39/lstm_cell_119/Sigmoid_2:y:03backward_lstm_39/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_39/lstm_cell_119/mul_2Б
.backward_lstm_39/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.backward_lstm_39/TensorArrayV2_1/element_shapeќ
 backward_lstm_39/TensorArrayV2_1TensorListReserve7backward_lstm_39/TensorArrayV2_1/element_shape:output:0)backward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_39/TensorArrayV2_1p
backward_lstm_39/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_39/time
&backward_lstm_39/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_lstm_39/Max/reduction_indices 
backward_lstm_39/MaxMaxbackward_lstm_39/Cast:y:0/backward_lstm_39/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/Maxr
backward_lstm_39/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_39/sub/y
backward_lstm_39/subSubbackward_lstm_39/Max:output:0backward_lstm_39/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/sub
backward_lstm_39/Sub_1Subbackward_lstm_39/sub:z:0backward_lstm_39/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_39/Sub_1Ѓ
backward_lstm_39/zeros_like	ZerosLike(backward_lstm_39/lstm_cell_119/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/zeros_likeЁ
)backward_lstm_39/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)backward_lstm_39/while/maximum_iterations
#backward_lstm_39/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_39/while/loop_counter	
backward_lstm_39/whileWhile,backward_lstm_39/while/loop_counter:output:02backward_lstm_39/while/maximum_iterations:output:0backward_lstm_39/time:output:0)backward_lstm_39/TensorArrayV2_1:handle:0backward_lstm_39/zeros_like:y:0backward_lstm_39/zeros:output:0!backward_lstm_39/zeros_1:output:0)backward_lstm_39/strided_slice_1:output:0Hbackward_lstm_39/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_39/Sub_1:z:0=backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource?backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource>backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( */
body'R%
#backward_lstm_39_while_body_4933265*/
cond'R%
#backward_lstm_39_while_cond_4933264*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_39/whileз
Abackward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Abackward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3backward_lstm_39/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_39/while:output:3Jbackward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3backward_lstm_39/TensorArrayV2Stack/TensorListStackЃ
&backward_lstm_39/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&backward_lstm_39/strided_slice_3/stack
(backward_lstm_39/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_39/strided_slice_3/stack_1
(backward_lstm_39/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_3/stack_2
 backward_lstm_39/strided_slice_3StridedSlice<backward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_39/strided_slice_3/stack:output:01backward_lstm_39/strided_slice_3/stack_1:output:01backward_lstm_39/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 backward_lstm_39/strided_slice_3
!backward_lstm_39/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_39/transpose_1/permђ
backward_lstm_39/transpose_1	Transpose<backward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_39/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_39/transpose_1
backward_lstm_39/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_39/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisТ
concatConcatV2(forward_lstm_39/strided_slice_3:output:0)backward_lstm_39/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

IdentityЬ
NoOpNoOp6^backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp5^backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp7^backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp^backward_lstm_39/while5^forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp4^forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp6^forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp^forward_lstm_39/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2n
5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp2l
4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp2p
6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp20
backward_lstm_39/whilebackward_lstm_39/while2l
4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp2j
3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp2n
5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp2.
forward_lstm_39/whileforward_lstm_39/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
к

#backward_lstm_39_while_cond_4932615>
:backward_lstm_39_while_backward_lstm_39_while_loop_counterD
@backward_lstm_39_while_backward_lstm_39_while_maximum_iterations&
"backward_lstm_39_while_placeholder(
$backward_lstm_39_while_placeholder_1(
$backward_lstm_39_while_placeholder_2(
$backward_lstm_39_while_placeholder_3@
<backward_lstm_39_while_less_backward_lstm_39_strided_slice_1W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4932615___redundant_placeholder0W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4932615___redundant_placeholder1W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4932615___redundant_placeholder2W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4932615___redundant_placeholder3#
backward_lstm_39_while_identity
Х
backward_lstm_39/while/LessLess"backward_lstm_39_while_placeholder<backward_lstm_39_while_less_backward_lstm_39_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_39/while/Less
backward_lstm_39/while/IdentityIdentitybackward_lstm_39/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_39/while/Identity"K
backward_lstm_39_while_identity(backward_lstm_39/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
ц
Ф
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_4930870

inputs*
forward_lstm_39_4930700:	Ш*
forward_lstm_39_4930702:	2Ш&
forward_lstm_39_4930704:	Ш+
backward_lstm_39_4930860:	Ш+
backward_lstm_39_4930862:	2Ш'
backward_lstm_39_4930864:	Ш
identityЂ(backward_lstm_39/StatefulPartitionedCallЂ'forward_lstm_39/StatefulPartitionedCallе
'forward_lstm_39/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_39_4930700forward_lstm_39_4930702forward_lstm_39_4930704*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_49306992)
'forward_lstm_39/StatefulPartitionedCallл
(backward_lstm_39/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_39_4930860backward_lstm_39_4930862backward_lstm_39_4930864*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_49308592*
(backward_lstm_39/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisв
concatConcatV20forward_lstm_39/StatefulPartitionedCall:output:01backward_lstm_39/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

IdentityЃ
NoOpNoOp)^backward_lstm_39/StatefulPartitionedCall(^forward_lstm_39/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2T
(backward_lstm_39/StatefulPartitionedCall(backward_lstm_39/StatefulPartitionedCall2R
'forward_lstm_39/StatefulPartitionedCall'forward_lstm_39/StatefulPartitionedCall:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
]
Њ
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_4934237

inputs?
,lstm_cell_118_matmul_readvariableop_resource:	ШA
.lstm_cell_118_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_118_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_118/BiasAdd/ReadVariableOpЂ#lstm_cell_118/MatMul/ReadVariableOpЂ%lstm_cell_118/MatMul_1/ReadVariableOpЂwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1
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
:џџџџџџџџџ22
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
B :ш2
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
zeros_1/packed/1
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
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
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
strided_slice_2/stack_2
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2
strided_slice_2И
#lstm_cell_118/MatMul/ReadVariableOpReadVariableOp,lstm_cell_118_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_118/MatMul/ReadVariableOpА
lstm_cell_118/MatMulMatMulstrided_slice_2:output:0+lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/MatMulО
%lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_118/MatMul_1/ReadVariableOpЌ
lstm_cell_118/MatMul_1MatMulzeros:output:0-lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/MatMul_1Є
lstm_cell_118/addAddV2lstm_cell_118/MatMul:product:0 lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/addЗ
$lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_118/BiasAdd/ReadVariableOpБ
lstm_cell_118/BiasAddBiasAddlstm_cell_118/add:z:0,lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/BiasAdd
lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_118/split/split_dimї
lstm_cell_118/splitSplit&lstm_cell_118/split/split_dim:output:0lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_118/split
lstm_cell_118/SigmoidSigmoidlstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Sigmoid
lstm_cell_118/Sigmoid_1Sigmoidlstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Sigmoid_1
lstm_cell_118/mulMullstm_cell_118/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/mul
lstm_cell_118/ReluRelulstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Relu 
lstm_cell_118/mul_1Mullstm_cell_118/Sigmoid:y:0 lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/mul_1
lstm_cell_118/add_1AddV2lstm_cell_118/mul:z:0lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/add_1
lstm_cell_118/Sigmoid_2Sigmoidlstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Sigmoid_2
lstm_cell_118/Relu_1Relulstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Relu_1Є
lstm_cell_118/mul_2Mullstm_cell_118/Sigmoid_2:y:0"lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_118_matmul_readvariableop_resource.lstm_cell_118_matmul_1_readvariableop_resource-lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_4934153*
condR
while_cond_4934152*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
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
:џџџџџџџџџ22

IdentityЫ
NoOpNoOp%^lstm_cell_118/BiasAdd/ReadVariableOp$^lstm_cell_118/MatMul/ReadVariableOp&^lstm_cell_118/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_118/BiasAdd/ReadVariableOp$lstm_cell_118/BiasAdd/ReadVariableOp2J
#lstm_cell_118/MatMul/ReadVariableOp#lstm_cell_118/MatMul/ReadVariableOp2N
%lstm_cell_118/MatMul_1/ReadVariableOp%lstm_cell_118/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
&
ё
while_body_4929995
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_119_4930019_0:	Ш0
while_lstm_cell_119_4930021_0:	2Ш,
while_lstm_cell_119_4930023_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_119_4930019:	Ш.
while_lstm_cell_119_4930021:	2Ш*
while_lstm_cell_119_4930023:	ШЂ+while/lstm_cell_119/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemы
+while/lstm_cell_119/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_119_4930019_0while_lstm_cell_119_4930021_0while_lstm_cell_119_4930023_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_119_layer_call_and_return_conditional_losses_49299812-
+while/lstm_cell_119/StatefulPartitionedCallј
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_119/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Ѕ
while/Identity_4Identity4while/lstm_cell_119/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѕ
while/Identity_5Identity4while/lstm_cell_119/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp,^while/lstm_cell_119/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_119_4930019while_lstm_cell_119_4930019_0"<
while_lstm_cell_119_4930021while_lstm_cell_119_4930021_0"<
while_lstm_cell_119_4930023while_lstm_cell_119_4930023_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2Z
+while/lstm_cell_119/StatefulPartitionedCall+while/lstm_cell_119/StatefulPartitionedCall: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
к
Ш
while_cond_4934303
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_4934303___redundant_placeholder05
1while_while_cond_4934303___redundant_placeholder15
1while_while_cond_4934303___redundant_placeholder25
1while_while_cond_4934303___redundant_placeholder3
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
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
У

#backward_lstm_39_while_cond_4933622>
:backward_lstm_39_while_backward_lstm_39_while_loop_counterD
@backward_lstm_39_while_backward_lstm_39_while_maximum_iterations&
"backward_lstm_39_while_placeholder(
$backward_lstm_39_while_placeholder_1(
$backward_lstm_39_while_placeholder_2(
$backward_lstm_39_while_placeholder_3(
$backward_lstm_39_while_placeholder_4@
<backward_lstm_39_while_less_backward_lstm_39_strided_slice_1W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4933622___redundant_placeholder0W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4933622___redundant_placeholder1W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4933622___redundant_placeholder2W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4933622___redundant_placeholder3W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4933622___redundant_placeholder4#
backward_lstm_39_while_identity
Х
backward_lstm_39/while/LessLess"backward_lstm_39_while_placeholder<backward_lstm_39_while_less_backward_lstm_39_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_39/while/Less
backward_lstm_39/while/IdentityIdentitybackward_lstm_39/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_39/while/Identity"K
backward_lstm_39_while_identity(backward_lstm_39/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: :::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
ј?
к
while_body_4931140
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_118_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_118_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_118_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_118_matmul_readvariableop_resource:	ШG
4while_lstm_cell_118_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_118_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_118/BiasAdd/ReadVariableOpЂ)while/lstm_cell_118/MatMul/ReadVariableOpЂ+while/lstm_cell_118/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeм
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЬ
)while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_118/MatMul/ReadVariableOpк
while/lstm_cell_118/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/MatMulв
+while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_118/MatMul_1/ReadVariableOpУ
while/lstm_cell_118/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/MatMul_1М
while/lstm_cell_118/addAddV2$while/lstm_cell_118/MatMul:product:0&while/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/addЫ
*while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_118/BiasAdd/ReadVariableOpЩ
while/lstm_cell_118/BiasAddBiasAddwhile/lstm_cell_118/add:z:02while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/BiasAdd
#while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_118/split/split_dim
while/lstm_cell_118/splitSplit,while/lstm_cell_118/split/split_dim:output:0$while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_118/split
while/lstm_cell_118/SigmoidSigmoid"while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Sigmoid
while/lstm_cell_118/Sigmoid_1Sigmoid"while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Sigmoid_1Ѓ
while/lstm_cell_118/mulMul!while/lstm_cell_118/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/mul
while/lstm_cell_118/ReluRelu"while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/ReluИ
while/lstm_cell_118/mul_1Mulwhile/lstm_cell_118/Sigmoid:y:0&while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/mul_1­
while/lstm_cell_118/add_1AddV2while/lstm_cell_118/mul:z:0while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/add_1
while/lstm_cell_118/Sigmoid_2Sigmoid"while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Sigmoid_2
while/lstm_cell_118/Relu_1Reluwhile/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Relu_1М
while/lstm_cell_118/mul_2Mul!while/lstm_cell_118/Sigmoid_2:y:0(while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_118/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_118/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_118/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_118/BiasAdd/ReadVariableOp*^while/lstm_cell_118/MatMul/ReadVariableOp,^while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_118_biasadd_readvariableop_resource5while_lstm_cell_118_biasadd_readvariableop_resource_0"n
4while_lstm_cell_118_matmul_1_readvariableop_resource6while_lstm_cell_118_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_118_matmul_readvariableop_resource4while_lstm_cell_118_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_118/BiasAdd/ReadVariableOp*while/lstm_cell_118/BiasAdd/ReadVariableOp2V
)while/lstm_cell_118/MatMul/ReadVariableOp)while/lstm_cell_118/MatMul/ReadVariableOp2Z
+while/lstm_cell_118/MatMul_1/ReadVariableOp+while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
о
П
2__inference_backward_lstm_39_layer_call_fn_4934421

inputs
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_49308592
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
X
њ
#backward_lstm_39_while_body_4932616>
:backward_lstm_39_while_backward_lstm_39_while_loop_counterD
@backward_lstm_39_while_backward_lstm_39_while_maximum_iterations&
"backward_lstm_39_while_placeholder(
$backward_lstm_39_while_placeholder_1(
$backward_lstm_39_while_placeholder_2(
$backward_lstm_39_while_placeholder_3=
9backward_lstm_39_while_backward_lstm_39_strided_slice_1_0y
ubackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0X
Ebackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0:	ШZ
Gbackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0:	2ШU
Fbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0:	Ш#
backward_lstm_39_while_identity%
!backward_lstm_39_while_identity_1%
!backward_lstm_39_while_identity_2%
!backward_lstm_39_while_identity_3%
!backward_lstm_39_while_identity_4%
!backward_lstm_39_while_identity_5;
7backward_lstm_39_while_backward_lstm_39_strided_slice_1w
sbackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensorV
Cbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource:	ШX
Ebackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource:	2ШS
Dbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource:	ШЂ;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOpЂ:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOpЂ<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOpх
Hbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2J
Hbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeТ
:backward_lstm_39/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_39_while_placeholderQbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02<
:backward_lstm_39/while/TensorArrayV2Read/TensorListGetItemџ
:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOpReadVariableOpEbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp
+backward_lstm_39/while/lstm_cell_119/MatMulMatMulAbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem:item:0Bbackward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+backward_lstm_39/while/lstm_cell_119/MatMul
<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOpGbackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp
-backward_lstm_39/while/lstm_cell_119/MatMul_1MatMul$backward_lstm_39_while_placeholder_2Dbackward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_39/while/lstm_cell_119/MatMul_1
(backward_lstm_39/while/lstm_cell_119/addAddV25backward_lstm_39/while/lstm_cell_119/MatMul:product:07backward_lstm_39/while/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_39/while/lstm_cell_119/addў
;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOpFbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp
,backward_lstm_39/while/lstm_cell_119/BiasAddBiasAdd,backward_lstm_39/while/lstm_cell_119/add:z:0Cbackward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_39/while/lstm_cell_119/BiasAddЎ
4backward_lstm_39/while/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4backward_lstm_39/while/lstm_cell_119/split/split_dimг
*backward_lstm_39/while/lstm_cell_119/splitSplit=backward_lstm_39/while/lstm_cell_119/split/split_dim:output:05backward_lstm_39/while/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*backward_lstm_39/while/lstm_cell_119/splitЮ
,backward_lstm_39/while/lstm_cell_119/SigmoidSigmoid3backward_lstm_39/while/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_39/while/lstm_cell_119/Sigmoidв
.backward_lstm_39/while/lstm_cell_119/Sigmoid_1Sigmoid3backward_lstm_39/while/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.backward_lstm_39/while/lstm_cell_119/Sigmoid_1ч
(backward_lstm_39/while/lstm_cell_119/mulMul2backward_lstm_39/while/lstm_cell_119/Sigmoid_1:y:0$backward_lstm_39_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_39/while/lstm_cell_119/mulХ
)backward_lstm_39/while/lstm_cell_119/ReluRelu3backward_lstm_39/while/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_39/while/lstm_cell_119/Reluќ
*backward_lstm_39/while/lstm_cell_119/mul_1Mul0backward_lstm_39/while/lstm_cell_119/Sigmoid:y:07backward_lstm_39/while/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_39/while/lstm_cell_119/mul_1ё
*backward_lstm_39/while/lstm_cell_119/add_1AddV2,backward_lstm_39/while/lstm_cell_119/mul:z:0.backward_lstm_39/while/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_39/while/lstm_cell_119/add_1в
.backward_lstm_39/while/lstm_cell_119/Sigmoid_2Sigmoid3backward_lstm_39/while/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.backward_lstm_39/while/lstm_cell_119/Sigmoid_2Ф
+backward_lstm_39/while/lstm_cell_119/Relu_1Relu.backward_lstm_39/while/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_39/while/lstm_cell_119/Relu_1
*backward_lstm_39/while/lstm_cell_119/mul_2Mul2backward_lstm_39/while/lstm_cell_119/Sigmoid_2:y:09backward_lstm_39/while/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_39/while/lstm_cell_119/mul_2Ж
;backward_lstm_39/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_39_while_placeholder_1"backward_lstm_39_while_placeholder.backward_lstm_39/while/lstm_cell_119/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_39/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_39/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_39/while/add/y­
backward_lstm_39/while/addAddV2"backward_lstm_39_while_placeholder%backward_lstm_39/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/while/add
backward_lstm_39/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_39/while/add_1/yЫ
backward_lstm_39/while/add_1AddV2:backward_lstm_39_while_backward_lstm_39_while_loop_counter'backward_lstm_39/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/while/add_1Џ
backward_lstm_39/while/IdentityIdentity backward_lstm_39/while/add_1:z:0^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_39/while/Identityг
!backward_lstm_39/while/Identity_1Identity@backward_lstm_39_while_backward_lstm_39_while_maximum_iterations^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_39/while/Identity_1Б
!backward_lstm_39/while/Identity_2Identitybackward_lstm_39/while/add:z:0^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_39/while/Identity_2о
!backward_lstm_39/while/Identity_3IdentityKbackward_lstm_39/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_39/while/Identity_3в
!backward_lstm_39/while/Identity_4Identity.backward_lstm_39/while/lstm_cell_119/mul_2:z:0^backward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_39/while/Identity_4в
!backward_lstm_39/while/Identity_5Identity.backward_lstm_39/while/lstm_cell_119/add_1:z:0^backward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_39/while/Identity_5Ж
backward_lstm_39/while/NoOpNoOp<^backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp;^backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp=^backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_39/while/NoOp"t
7backward_lstm_39_while_backward_lstm_39_strided_slice_19backward_lstm_39_while_backward_lstm_39_strided_slice_1_0"K
backward_lstm_39_while_identity(backward_lstm_39/while/Identity:output:0"O
!backward_lstm_39_while_identity_1*backward_lstm_39/while/Identity_1:output:0"O
!backward_lstm_39_while_identity_2*backward_lstm_39/while/Identity_2:output:0"O
!backward_lstm_39_while_identity_3*backward_lstm_39/while/Identity_3:output:0"O
!backward_lstm_39_while_identity_4*backward_lstm_39/while/Identity_4:output:0"O
!backward_lstm_39_while_identity_5*backward_lstm_39/while/Identity_5:output:0"
Dbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resourceFbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0"
Ebackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resourceGbackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0"
Cbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resourceEbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0"ь
sbackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensorubackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2z
;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp2x
:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp2|
<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
к
Ш
while_cond_4934959
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_4934959___redundant_placeholder05
1while_while_cond_4934959___redundant_placeholder15
1while_while_cond_4934959___redundant_placeholder25
1while_while_cond_4934959___redundant_placeholder3
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
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Жc

"forward_lstm_39_while_body_4931434<
8forward_lstm_39_while_forward_lstm_39_while_loop_counterB
>forward_lstm_39_while_forward_lstm_39_while_maximum_iterations%
!forward_lstm_39_while_placeholder'
#forward_lstm_39_while_placeholder_1'
#forward_lstm_39_while_placeholder_2'
#forward_lstm_39_while_placeholder_3'
#forward_lstm_39_while_placeholder_4;
7forward_lstm_39_while_forward_lstm_39_strided_slice_1_0w
sforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_08
4forward_lstm_39_while_greater_forward_lstm_39_cast_0W
Dforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0:	ШY
Fforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0:	2ШT
Eforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0:	Ш"
forward_lstm_39_while_identity$
 forward_lstm_39_while_identity_1$
 forward_lstm_39_while_identity_2$
 forward_lstm_39_while_identity_3$
 forward_lstm_39_while_identity_4$
 forward_lstm_39_while_identity_5$
 forward_lstm_39_while_identity_69
5forward_lstm_39_while_forward_lstm_39_strided_slice_1u
qforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor6
2forward_lstm_39_while_greater_forward_lstm_39_castU
Bforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource:	ШW
Dforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource:	2ШR
Cforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource:	ШЂ:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOpЂ9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOpЂ;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOpу
Gforward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gforward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
9forward_lstm_39/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_39_while_placeholderPforward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02;
9forward_lstm_39/while/TensorArrayV2Read/TensorListGetItemа
forward_lstm_39/while/GreaterGreater4forward_lstm_39_while_greater_forward_lstm_39_cast_0!forward_lstm_39_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_39/while/Greaterќ
9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOpDforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02;
9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp
*forward_lstm_39/while/lstm_cell_118/MatMulMatMul@forward_lstm_39/while/TensorArrayV2Read/TensorListGetItem:item:0Aforward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2,
*forward_lstm_39/while/lstm_cell_118/MatMul
;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOpFforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02=
;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp
,forward_lstm_39/while/lstm_cell_118/MatMul_1MatMul#forward_lstm_39_while_placeholder_3Cforward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_39/while/lstm_cell_118/MatMul_1ќ
'forward_lstm_39/while/lstm_cell_118/addAddV24forward_lstm_39/while/lstm_cell_118/MatMul:product:06forward_lstm_39/while/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_39/while/lstm_cell_118/addћ
:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOpEforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02<
:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp
+forward_lstm_39/while/lstm_cell_118/BiasAddBiasAdd+forward_lstm_39/while/lstm_cell_118/add:z:0Bforward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_39/while/lstm_cell_118/BiasAddЌ
3forward_lstm_39/while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3forward_lstm_39/while/lstm_cell_118/split/split_dimЯ
)forward_lstm_39/while/lstm_cell_118/splitSplit<forward_lstm_39/while/lstm_cell_118/split/split_dim:output:04forward_lstm_39/while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2+
)forward_lstm_39/while/lstm_cell_118/splitЫ
+forward_lstm_39/while/lstm_cell_118/SigmoidSigmoid2forward_lstm_39/while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_39/while/lstm_cell_118/SigmoidЯ
-forward_lstm_39/while/lstm_cell_118/Sigmoid_1Sigmoid2forward_lstm_39/while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22/
-forward_lstm_39/while/lstm_cell_118/Sigmoid_1у
'forward_lstm_39/while/lstm_cell_118/mulMul1forward_lstm_39/while/lstm_cell_118/Sigmoid_1:y:0#forward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_39/while/lstm_cell_118/mulТ
(forward_lstm_39/while/lstm_cell_118/ReluRelu2forward_lstm_39/while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_39/while/lstm_cell_118/Reluј
)forward_lstm_39/while/lstm_cell_118/mul_1Mul/forward_lstm_39/while/lstm_cell_118/Sigmoid:y:06forward_lstm_39/while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_39/while/lstm_cell_118/mul_1э
)forward_lstm_39/while/lstm_cell_118/add_1AddV2+forward_lstm_39/while/lstm_cell_118/mul:z:0-forward_lstm_39/while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_39/while/lstm_cell_118/add_1Я
-forward_lstm_39/while/lstm_cell_118/Sigmoid_2Sigmoid2forward_lstm_39/while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22/
-forward_lstm_39/while/lstm_cell_118/Sigmoid_2С
*forward_lstm_39/while/lstm_cell_118/Relu_1Relu-forward_lstm_39/while/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_39/while/lstm_cell_118/Relu_1ќ
)forward_lstm_39/while/lstm_cell_118/mul_2Mul1forward_lstm_39/while/lstm_cell_118/Sigmoid_2:y:08forward_lstm_39/while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_39/while/lstm_cell_118/mul_2я
forward_lstm_39/while/SelectSelect!forward_lstm_39/while/Greater:z:0-forward_lstm_39/while/lstm_cell_118/mul_2:z:0#forward_lstm_39_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/while/Selectѓ
forward_lstm_39/while/Select_1Select!forward_lstm_39/while/Greater:z:0-forward_lstm_39/while/lstm_cell_118/mul_2:z:0#forward_lstm_39_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22 
forward_lstm_39/while/Select_1ѓ
forward_lstm_39/while/Select_2Select!forward_lstm_39/while/Greater:z:0-forward_lstm_39/while/lstm_cell_118/add_1:z:0#forward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22 
forward_lstm_39/while/Select_2Љ
:forward_lstm_39/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_39_while_placeholder_1!forward_lstm_39_while_placeholder%forward_lstm_39/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_39/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_39/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_39/while/add/yЉ
forward_lstm_39/while/addAddV2!forward_lstm_39_while_placeholder$forward_lstm_39/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/while/add
forward_lstm_39/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_39/while/add_1/yЦ
forward_lstm_39/while/add_1AddV28forward_lstm_39_while_forward_lstm_39_while_loop_counter&forward_lstm_39/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/while/add_1Ћ
forward_lstm_39/while/IdentityIdentityforward_lstm_39/while/add_1:z:0^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_39/while/IdentityЮ
 forward_lstm_39/while/Identity_1Identity>forward_lstm_39_while_forward_lstm_39_while_maximum_iterations^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_39/while/Identity_1­
 forward_lstm_39/while/Identity_2Identityforward_lstm_39/while/add:z:0^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_39/while/Identity_2к
 forward_lstm_39/while/Identity_3IdentityJforward_lstm_39/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_39/while/Identity_3Ц
 forward_lstm_39/while/Identity_4Identity%forward_lstm_39/while/Select:output:0^forward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_39/while/Identity_4Ш
 forward_lstm_39/while/Identity_5Identity'forward_lstm_39/while/Select_1:output:0^forward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_39/while/Identity_5Ш
 forward_lstm_39/while/Identity_6Identity'forward_lstm_39/while/Select_2:output:0^forward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_39/while/Identity_6Б
forward_lstm_39/while/NoOpNoOp;^forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp:^forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp<^forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_39/while/NoOp"p
5forward_lstm_39_while_forward_lstm_39_strided_slice_17forward_lstm_39_while_forward_lstm_39_strided_slice_1_0"j
2forward_lstm_39_while_greater_forward_lstm_39_cast4forward_lstm_39_while_greater_forward_lstm_39_cast_0"I
forward_lstm_39_while_identity'forward_lstm_39/while/Identity:output:0"M
 forward_lstm_39_while_identity_1)forward_lstm_39/while/Identity_1:output:0"M
 forward_lstm_39_while_identity_2)forward_lstm_39/while/Identity_2:output:0"M
 forward_lstm_39_while_identity_3)forward_lstm_39/while/Identity_3:output:0"M
 forward_lstm_39_while_identity_4)forward_lstm_39/while/Identity_4:output:0"M
 forward_lstm_39_while_identity_5)forward_lstm_39/while/Identity_5:output:0"M
 forward_lstm_39_while_identity_6)forward_lstm_39/while/Identity_6:output:0"
Cforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resourceEforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0"
Dforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resourceFforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0"
Bforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resourceDforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0"ш
qforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensorsforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2x
:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp2v
9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp2z
;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:џџџџџџџџџ
ј

ж
/__inference_sequential_39_layer_call_fn_4932254

inputs
inputs_1	
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
	unknown_2:	Ш
	unknown_3:	2Ш
	unknown_4:	Ш
	unknown_5:d
	unknown_6:
identityЂStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_39_layer_call_and_return_conditional_losses_49322132
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
М

"forward_lstm_39_while_cond_4932466<
8forward_lstm_39_while_forward_lstm_39_while_loop_counterB
>forward_lstm_39_while_forward_lstm_39_while_maximum_iterations%
!forward_lstm_39_while_placeholder'
#forward_lstm_39_while_placeholder_1'
#forward_lstm_39_while_placeholder_2'
#forward_lstm_39_while_placeholder_3>
:forward_lstm_39_while_less_forward_lstm_39_strided_slice_1U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4932466___redundant_placeholder0U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4932466___redundant_placeholder1U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4932466___redundant_placeholder2U
Qforward_lstm_39_while_forward_lstm_39_while_cond_4932466___redundant_placeholder3"
forward_lstm_39_while_identity
Р
forward_lstm_39/while/LessLess!forward_lstm_39_while_placeholder:forward_lstm_39_while_less_forward_lstm_39_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_39/while/Less
forward_lstm_39/while/IdentityIdentityforward_lstm_39/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_39/while/Identity"I
forward_lstm_39_while_identity'forward_lstm_39/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
]
Њ
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_4931224

inputs?
,lstm_cell_118_matmul_readvariableop_resource:	ШA
.lstm_cell_118_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_118_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_118/BiasAdd/ReadVariableOpЂ#lstm_cell_118/MatMul/ReadVariableOpЂ%lstm_cell_118/MatMul_1/ReadVariableOpЂwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1
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
:џџџџџџџџџ22
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
B :ш2
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
zeros_1/packed/1
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
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
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
strided_slice_2/stack_2
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2
strided_slice_2И
#lstm_cell_118/MatMul/ReadVariableOpReadVariableOp,lstm_cell_118_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_118/MatMul/ReadVariableOpА
lstm_cell_118/MatMulMatMulstrided_slice_2:output:0+lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/MatMulО
%lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_118/MatMul_1/ReadVariableOpЌ
lstm_cell_118/MatMul_1MatMulzeros:output:0-lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/MatMul_1Є
lstm_cell_118/addAddV2lstm_cell_118/MatMul:product:0 lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/addЗ
$lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_118/BiasAdd/ReadVariableOpБ
lstm_cell_118/BiasAddBiasAddlstm_cell_118/add:z:0,lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_118/BiasAdd
lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_118/split/split_dimї
lstm_cell_118/splitSplit&lstm_cell_118/split/split_dim:output:0lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_118/split
lstm_cell_118/SigmoidSigmoidlstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Sigmoid
lstm_cell_118/Sigmoid_1Sigmoidlstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Sigmoid_1
lstm_cell_118/mulMullstm_cell_118/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/mul
lstm_cell_118/ReluRelulstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Relu 
lstm_cell_118/mul_1Mullstm_cell_118/Sigmoid:y:0 lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/mul_1
lstm_cell_118/add_1AddV2lstm_cell_118/mul:z:0lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/add_1
lstm_cell_118/Sigmoid_2Sigmoidlstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Sigmoid_2
lstm_cell_118/Relu_1Relulstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/Relu_1Є
lstm_cell_118/mul_2Mullstm_cell_118/Sigmoid_2:y:0"lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_118/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_118_matmul_readvariableop_resource.lstm_cell_118_matmul_1_readvariableop_resource-lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_4931140*
condR
while_cond_4931139*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
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
:џџџџџџџџџ22

IdentityЫ
NoOpNoOp%^lstm_cell_118/BiasAdd/ReadVariableOp$^lstm_cell_118/MatMul/ReadVariableOp&^lstm_cell_118/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_118/BiasAdd/ReadVariableOp$lstm_cell_118/BiasAdd/ReadVariableOp2J
#lstm_cell_118/MatMul/ReadVariableOp#lstm_cell_118/MatMul/ReadVariableOp2N
%lstm_cell_118/MatMul_1/ReadVariableOp%lstm_cell_118/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
к
Ш
while_cond_4929362
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_4929362___redundant_placeholder05
1while_while_cond_4929362___redundant_placeholder15
1while_while_cond_4929362___redundant_placeholder25
1while_while_cond_4929362___redundant_placeholder3
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
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
ЧH

M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_4930064

inputs(
lstm_cell_119_4929982:	Ш(
lstm_cell_119_4929984:	2Ш$
lstm_cell_119_4929986:	Ш
identityЂ%lstm_cell_119/StatefulPartitionedCallЂwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1
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
:џџџџџџџџџ22
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
B :ш2
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
zeros_1/packed/1
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
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
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
ReverseV2/axis
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	ReverseV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shape§
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
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Ї
%lstm_cell_119/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_119_4929982lstm_cell_119_4929984lstm_cell_119_4929986*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_119_layer_call_and_return_conditional_losses_49299812'
%lstm_cell_119/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterШ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_119_4929982lstm_cell_119_4929984lstm_cell_119_4929986*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_4929995*
condR
while_cond_4929994*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
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
:џџџџџџџџџ22

Identity~
NoOpNoOp&^lstm_cell_119/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2N
%lstm_cell_119/StatefulPartitionedCall%lstm_cell_119/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
гљ
г
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_4933004
inputs_0O
<forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource:	ШQ
>forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource:	2ШL
=forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource:	ШP
=backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource:	ШR
?backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource:	2ШM
>backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource:	Ш
identityЂ5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpЂ4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpЂ6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOpЂbackward_lstm_39/whileЂ4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpЂ3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOpЂ5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpЂforward_lstm_39/whilef
forward_lstm_39/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_39/Shape
#forward_lstm_39/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_39/strided_slice/stack
%forward_lstm_39/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_39/strided_slice/stack_1
%forward_lstm_39/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_39/strided_slice/stack_2Т
forward_lstm_39/strided_sliceStridedSliceforward_lstm_39/Shape:output:0,forward_lstm_39/strided_slice/stack:output:0.forward_lstm_39/strided_slice/stack_1:output:0.forward_lstm_39/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_39/strided_slice|
forward_lstm_39/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_39/zeros/mul/yЌ
forward_lstm_39/zeros/mulMul&forward_lstm_39/strided_slice:output:0$forward_lstm_39/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros/mul
forward_lstm_39/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_39/zeros/Less/yЇ
forward_lstm_39/zeros/LessLessforward_lstm_39/zeros/mul:z:0%forward_lstm_39/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros/Less
forward_lstm_39/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_39/zeros/packed/1У
forward_lstm_39/zeros/packedPack&forward_lstm_39/strided_slice:output:0'forward_lstm_39/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_39/zeros/packed
forward_lstm_39/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_39/zeros/ConstЕ
forward_lstm_39/zerosFill%forward_lstm_39/zeros/packed:output:0$forward_lstm_39/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/zeros
forward_lstm_39/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_39/zeros_1/mul/yВ
forward_lstm_39/zeros_1/mulMul&forward_lstm_39/strided_slice:output:0&forward_lstm_39/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros_1/mul
forward_lstm_39/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
forward_lstm_39/zeros_1/Less/yЏ
forward_lstm_39/zeros_1/LessLessforward_lstm_39/zeros_1/mul:z:0'forward_lstm_39/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/zeros_1/Less
 forward_lstm_39/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_39/zeros_1/packed/1Щ
forward_lstm_39/zeros_1/packedPack&forward_lstm_39/strided_slice:output:0)forward_lstm_39/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_39/zeros_1/packed
forward_lstm_39/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_39/zeros_1/ConstН
forward_lstm_39/zeros_1Fill'forward_lstm_39/zeros_1/packed:output:0&forward_lstm_39/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/zeros_1
forward_lstm_39/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_39/transpose/permО
forward_lstm_39/transpose	Transposeinputs_0'forward_lstm_39/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
forward_lstm_39/transpose
forward_lstm_39/Shape_1Shapeforward_lstm_39/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_39/Shape_1
%forward_lstm_39/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_39/strided_slice_1/stack
'forward_lstm_39/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_1/stack_1
'forward_lstm_39/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_1/stack_2Ю
forward_lstm_39/strided_slice_1StridedSlice forward_lstm_39/Shape_1:output:0.forward_lstm_39/strided_slice_1/stack:output:00forward_lstm_39/strided_slice_1/stack_1:output:00forward_lstm_39/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_39/strided_slice_1Ѕ
+forward_lstm_39/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+forward_lstm_39/TensorArrayV2/element_shapeђ
forward_lstm_39/TensorArrayV2TensorListReserve4forward_lstm_39/TensorArrayV2/element_shape:output:0(forward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_39/TensorArrayV2п
Eforward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2G
Eforward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
7forward_lstm_39/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_39/transpose:y:0Nforward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_39/TensorArrayUnstack/TensorListFromTensor
%forward_lstm_39/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_39/strided_slice_2/stack
'forward_lstm_39/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_2/stack_1
'forward_lstm_39/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_2/stack_2х
forward_lstm_39/strided_slice_2StridedSliceforward_lstm_39/transpose:y:0.forward_lstm_39/strided_slice_2/stack:output:00forward_lstm_39/strided_slice_2/stack_1:output:00forward_lstm_39/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2!
forward_lstm_39/strided_slice_2ш
3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp<forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype025
3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp№
$forward_lstm_39/lstm_cell_118/MatMulMatMul(forward_lstm_39/strided_slice_2:output:0;forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$forward_lstm_39/lstm_cell_118/MatMulю
5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp>forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype027
5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOpь
&forward_lstm_39/lstm_cell_118/MatMul_1MatMulforward_lstm_39/zeros:output:0=forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_39/lstm_cell_118/MatMul_1ф
!forward_lstm_39/lstm_cell_118/addAddV2.forward_lstm_39/lstm_cell_118/MatMul:product:00forward_lstm_39/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2#
!forward_lstm_39/lstm_cell_118/addч
4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp=forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype026
4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOpё
%forward_lstm_39/lstm_cell_118/BiasAddBiasAdd%forward_lstm_39/lstm_cell_118/add:z:0<forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_39/lstm_cell_118/BiasAdd 
-forward_lstm_39/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-forward_lstm_39/lstm_cell_118/split/split_dimЗ
#forward_lstm_39/lstm_cell_118/splitSplit6forward_lstm_39/lstm_cell_118/split/split_dim:output:0.forward_lstm_39/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2%
#forward_lstm_39/lstm_cell_118/splitЙ
%forward_lstm_39/lstm_cell_118/SigmoidSigmoid,forward_lstm_39/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_39/lstm_cell_118/SigmoidН
'forward_lstm_39/lstm_cell_118/Sigmoid_1Sigmoid,forward_lstm_39/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_39/lstm_cell_118/Sigmoid_1Ю
!forward_lstm_39/lstm_cell_118/mulMul+forward_lstm_39/lstm_cell_118/Sigmoid_1:y:0 forward_lstm_39/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_39/lstm_cell_118/mulА
"forward_lstm_39/lstm_cell_118/ReluRelu,forward_lstm_39/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_39/lstm_cell_118/Reluр
#forward_lstm_39/lstm_cell_118/mul_1Mul)forward_lstm_39/lstm_cell_118/Sigmoid:y:00forward_lstm_39/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_39/lstm_cell_118/mul_1е
#forward_lstm_39/lstm_cell_118/add_1AddV2%forward_lstm_39/lstm_cell_118/mul:z:0'forward_lstm_39/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_39/lstm_cell_118/add_1Н
'forward_lstm_39/lstm_cell_118/Sigmoid_2Sigmoid,forward_lstm_39/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_39/lstm_cell_118/Sigmoid_2Џ
$forward_lstm_39/lstm_cell_118/Relu_1Relu'forward_lstm_39/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_39/lstm_cell_118/Relu_1ф
#forward_lstm_39/lstm_cell_118/mul_2Mul+forward_lstm_39/lstm_cell_118/Sigmoid_2:y:02forward_lstm_39/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_39/lstm_cell_118/mul_2Џ
-forward_lstm_39/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2/
-forward_lstm_39/TensorArrayV2_1/element_shapeј
forward_lstm_39/TensorArrayV2_1TensorListReserve6forward_lstm_39/TensorArrayV2_1/element_shape:output:0(forward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_39/TensorArrayV2_1n
forward_lstm_39/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_39/time
(forward_lstm_39/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(forward_lstm_39/while/maximum_iterations
"forward_lstm_39/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_39/while/loop_counter
forward_lstm_39/whileWhile+forward_lstm_39/while/loop_counter:output:01forward_lstm_39/while/maximum_iterations:output:0forward_lstm_39/time:output:0(forward_lstm_39/TensorArrayV2_1:handle:0forward_lstm_39/zeros:output:0 forward_lstm_39/zeros_1:output:0(forward_lstm_39/strided_slice_1:output:0Gforward_lstm_39/TensorArrayUnstack/TensorListFromTensor:output_handle:0<forward_lstm_39_lstm_cell_118_matmul_readvariableop_resource>forward_lstm_39_lstm_cell_118_matmul_1_readvariableop_resource=forward_lstm_39_lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *.
body&R$
"forward_lstm_39_while_body_4932769*.
cond&R$
"forward_lstm_39_while_cond_4932768*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
forward_lstm_39/whileе
@forward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2B
@forward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeБ
2forward_lstm_39/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_39/while:output:3Iforward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype024
2forward_lstm_39/TensorArrayV2Stack/TensorListStackЁ
%forward_lstm_39/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2'
%forward_lstm_39/strided_slice_3/stack
'forward_lstm_39/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_39/strided_slice_3/stack_1
'forward_lstm_39/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_39/strided_slice_3/stack_2њ
forward_lstm_39/strided_slice_3StridedSlice;forward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_39/strided_slice_3/stack:output:00forward_lstm_39/strided_slice_3/stack_1:output:00forward_lstm_39/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2!
forward_lstm_39/strided_slice_3
 forward_lstm_39/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_39/transpose_1/permю
forward_lstm_39/transpose_1	Transpose;forward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_39/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_39/transpose_1
forward_lstm_39/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_39/runtimeh
backward_lstm_39/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_39/Shape
$backward_lstm_39/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_39/strided_slice/stack
&backward_lstm_39/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_39/strided_slice/stack_1
&backward_lstm_39/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_39/strided_slice/stack_2Ш
backward_lstm_39/strided_sliceStridedSlicebackward_lstm_39/Shape:output:0-backward_lstm_39/strided_slice/stack:output:0/backward_lstm_39/strided_slice/stack_1:output:0/backward_lstm_39/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_39/strided_slice~
backward_lstm_39/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_39/zeros/mul/yА
backward_lstm_39/zeros/mulMul'backward_lstm_39/strided_slice:output:0%backward_lstm_39/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros/mul
backward_lstm_39/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
backward_lstm_39/zeros/Less/yЋ
backward_lstm_39/zeros/LessLessbackward_lstm_39/zeros/mul:z:0&backward_lstm_39/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros/Less
backward_lstm_39/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_39/zeros/packed/1Ч
backward_lstm_39/zeros/packedPack'backward_lstm_39/strided_slice:output:0(backward_lstm_39/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_39/zeros/packed
backward_lstm_39/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_39/zeros/ConstЙ
backward_lstm_39/zerosFill&backward_lstm_39/zeros/packed:output:0%backward_lstm_39/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/zeros
backward_lstm_39/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_39/zeros_1/mul/yЖ
backward_lstm_39/zeros_1/mulMul'backward_lstm_39/strided_slice:output:0'backward_lstm_39/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros_1/mul
backward_lstm_39/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
backward_lstm_39/zeros_1/Less/yГ
backward_lstm_39/zeros_1/LessLess backward_lstm_39/zeros_1/mul:z:0(backward_lstm_39/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/zeros_1/Less
!backward_lstm_39/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_39/zeros_1/packed/1Э
backward_lstm_39/zeros_1/packedPack'backward_lstm_39/strided_slice:output:0*backward_lstm_39/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_39/zeros_1/packed
backward_lstm_39/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_39/zeros_1/ConstС
backward_lstm_39/zeros_1Fill(backward_lstm_39/zeros_1/packed:output:0'backward_lstm_39/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/zeros_1
backward_lstm_39/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_39/transpose/permС
backward_lstm_39/transpose	Transposeinputs_0(backward_lstm_39/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_39/transpose
backward_lstm_39/Shape_1Shapebackward_lstm_39/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_39/Shape_1
&backward_lstm_39/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_39/strided_slice_1/stack
(backward_lstm_39/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_1/stack_1
(backward_lstm_39/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_1/stack_2д
 backward_lstm_39/strided_slice_1StridedSlice!backward_lstm_39/Shape_1:output:0/backward_lstm_39/strided_slice_1/stack:output:01backward_lstm_39/strided_slice_1/stack_1:output:01backward_lstm_39/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_39/strided_slice_1Ї
,backward_lstm_39/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,backward_lstm_39/TensorArrayV2/element_shapeі
backward_lstm_39/TensorArrayV2TensorListReserve5backward_lstm_39/TensorArrayV2/element_shape:output:0)backward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_39/TensorArrayV2
backward_lstm_39/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_39/ReverseV2/axisз
backward_lstm_39/ReverseV2	ReverseV2backward_lstm_39/transpose:y:0(backward_lstm_39/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_39/ReverseV2с
Fbackward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2H
Fbackward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shapeС
8backward_lstm_39/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_39/ReverseV2:output:0Obackward_lstm_39/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_39/TensorArrayUnstack/TensorListFromTensor
&backward_lstm_39/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_39/strided_slice_2/stack
(backward_lstm_39/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_2/stack_1
(backward_lstm_39/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_2/stack_2ы
 backward_lstm_39/strided_slice_2StridedSlicebackward_lstm_39/transpose:y:0/backward_lstm_39/strided_slice_2/stack:output:01backward_lstm_39/strided_slice_2/stack_1:output:01backward_lstm_39/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2"
 backward_lstm_39/strided_slice_2ы
4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpReadVariableOp=backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOpє
%backward_lstm_39/lstm_cell_119/MatMulMatMul)backward_lstm_39/strided_slice_2:output:0<backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%backward_lstm_39/lstm_cell_119/MatMulё
6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOp?backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp№
'backward_lstm_39/lstm_cell_119/MatMul_1MatMulbackward_lstm_39/zeros:output:0>backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_39/lstm_cell_119/MatMul_1ш
"backward_lstm_39/lstm_cell_119/addAddV2/backward_lstm_39/lstm_cell_119/MatMul:product:01backward_lstm_39/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"backward_lstm_39/lstm_cell_119/addъ
5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOp>backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOpѕ
&backward_lstm_39/lstm_cell_119/BiasAddBiasAdd&backward_lstm_39/lstm_cell_119/add:z:0=backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_39/lstm_cell_119/BiasAddЂ
.backward_lstm_39/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.backward_lstm_39/lstm_cell_119/split/split_dimЛ
$backward_lstm_39/lstm_cell_119/splitSplit7backward_lstm_39/lstm_cell_119/split/split_dim:output:0/backward_lstm_39/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$backward_lstm_39/lstm_cell_119/splitМ
&backward_lstm_39/lstm_cell_119/SigmoidSigmoid-backward_lstm_39/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_39/lstm_cell_119/SigmoidР
(backward_lstm_39/lstm_cell_119/Sigmoid_1Sigmoid-backward_lstm_39/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_39/lstm_cell_119/Sigmoid_1в
"backward_lstm_39/lstm_cell_119/mulMul,backward_lstm_39/lstm_cell_119/Sigmoid_1:y:0!backward_lstm_39/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_39/lstm_cell_119/mulГ
#backward_lstm_39/lstm_cell_119/ReluRelu-backward_lstm_39/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_39/lstm_cell_119/Reluф
$backward_lstm_39/lstm_cell_119/mul_1Mul*backward_lstm_39/lstm_cell_119/Sigmoid:y:01backward_lstm_39/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_39/lstm_cell_119/mul_1й
$backward_lstm_39/lstm_cell_119/add_1AddV2&backward_lstm_39/lstm_cell_119/mul:z:0(backward_lstm_39/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_39/lstm_cell_119/add_1Р
(backward_lstm_39/lstm_cell_119/Sigmoid_2Sigmoid-backward_lstm_39/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_39/lstm_cell_119/Sigmoid_2В
%backward_lstm_39/lstm_cell_119/Relu_1Relu(backward_lstm_39/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_39/lstm_cell_119/Relu_1ш
$backward_lstm_39/lstm_cell_119/mul_2Mul,backward_lstm_39/lstm_cell_119/Sigmoid_2:y:03backward_lstm_39/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_39/lstm_cell_119/mul_2Б
.backward_lstm_39/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.backward_lstm_39/TensorArrayV2_1/element_shapeќ
 backward_lstm_39/TensorArrayV2_1TensorListReserve7backward_lstm_39/TensorArrayV2_1/element_shape:output:0)backward_lstm_39/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_39/TensorArrayV2_1p
backward_lstm_39/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_39/timeЁ
)backward_lstm_39/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)backward_lstm_39/while/maximum_iterations
#backward_lstm_39/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_39/while/loop_counter
backward_lstm_39/whileWhile,backward_lstm_39/while/loop_counter:output:02backward_lstm_39/while/maximum_iterations:output:0backward_lstm_39/time:output:0)backward_lstm_39/TensorArrayV2_1:handle:0backward_lstm_39/zeros:output:0!backward_lstm_39/zeros_1:output:0)backward_lstm_39/strided_slice_1:output:0Hbackward_lstm_39/TensorArrayUnstack/TensorListFromTensor:output_handle:0=backward_lstm_39_lstm_cell_119_matmul_readvariableop_resource?backward_lstm_39_lstm_cell_119_matmul_1_readvariableop_resource>backward_lstm_39_lstm_cell_119_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( */
body'R%
#backward_lstm_39_while_body_4932918*/
cond'R%
#backward_lstm_39_while_cond_4932917*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
backward_lstm_39/whileз
Abackward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Abackward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3backward_lstm_39/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_39/while:output:3Jbackward_lstm_39/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3backward_lstm_39/TensorArrayV2Stack/TensorListStackЃ
&backward_lstm_39/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&backward_lstm_39/strided_slice_3/stack
(backward_lstm_39/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_39/strided_slice_3/stack_1
(backward_lstm_39/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_39/strided_slice_3/stack_2
 backward_lstm_39/strided_slice_3StridedSlice<backward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_39/strided_slice_3/stack:output:01backward_lstm_39/strided_slice_3/stack_1:output:01backward_lstm_39/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 backward_lstm_39/strided_slice_3
!backward_lstm_39/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_39/transpose_1/permђ
backward_lstm_39/transpose_1	Transpose<backward_lstm_39/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_39/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_39/transpose_1
backward_lstm_39/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_39/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisТ
concatConcatV2(forward_lstm_39/strided_slice_3:output:0)backward_lstm_39/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

IdentityЬ
NoOpNoOp6^backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp5^backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp7^backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp^backward_lstm_39/while5^forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp4^forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp6^forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp^forward_lstm_39/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2n
5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp5backward_lstm_39/lstm_cell_119/BiasAdd/ReadVariableOp2l
4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp4backward_lstm_39/lstm_cell_119/MatMul/ReadVariableOp2p
6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp6backward_lstm_39/lstm_cell_119/MatMul_1/ReadVariableOp20
backward_lstm_39/whilebackward_lstm_39/while2l
4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp4forward_lstm_39/lstm_cell_118/BiasAdd/ReadVariableOp2j
3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp3forward_lstm_39/lstm_cell_118/MatMul/ReadVariableOp2n
5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp5forward_lstm_39/lstm_cell_118/MatMul_1/ReadVariableOp2.
forward_lstm_39/whileforward_lstm_39/while:g c
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
К
ј
/__inference_lstm_cell_119_layer_call_fn_4935159

inputs
states_0
states_1
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identity

identity_1

identity_2ЂStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_119_layer_call_and_return_conditional_losses_49299812
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

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
?:џџџџџџџџџ:џџџџџџџџџ2:џџџџџџџџџ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/1
я?
к
while_body_4933851
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_118_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_118_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_118_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_118_matmul_readvariableop_resource:	ШG
4while_lstm_cell_118_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_118_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_118/BiasAdd/ReadVariableOpЂ)while/lstm_cell_118/MatMul/ReadVariableOpЂ+while/lstm_cell_118/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЬ
)while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_118/MatMul/ReadVariableOpк
while/lstm_cell_118/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/MatMulв
+while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_118/MatMul_1/ReadVariableOpУ
while/lstm_cell_118/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/MatMul_1М
while/lstm_cell_118/addAddV2$while/lstm_cell_118/MatMul:product:0&while/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/addЫ
*while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_118/BiasAdd/ReadVariableOpЩ
while/lstm_cell_118/BiasAddBiasAddwhile/lstm_cell_118/add:z:02while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/BiasAdd
#while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_118/split/split_dim
while/lstm_cell_118/splitSplit,while/lstm_cell_118/split/split_dim:output:0$while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_118/split
while/lstm_cell_118/SigmoidSigmoid"while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Sigmoid
while/lstm_cell_118/Sigmoid_1Sigmoid"while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Sigmoid_1Ѓ
while/lstm_cell_118/mulMul!while/lstm_cell_118/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/mul
while/lstm_cell_118/ReluRelu"while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/ReluИ
while/lstm_cell_118/mul_1Mulwhile/lstm_cell_118/Sigmoid:y:0&while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/mul_1­
while/lstm_cell_118/add_1AddV2while/lstm_cell_118/mul:z:0while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/add_1
while/lstm_cell_118/Sigmoid_2Sigmoid"while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Sigmoid_2
while/lstm_cell_118/Relu_1Reluwhile/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Relu_1М
while/lstm_cell_118/mul_2Mul!while/lstm_cell_118/Sigmoid_2:y:0(while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_118/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_118/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_118/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_118/BiasAdd/ReadVariableOp*^while/lstm_cell_118/MatMul/ReadVariableOp,^while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_118_biasadd_readvariableop_resource5while_lstm_cell_118_biasadd_readvariableop_resource_0"n
4while_lstm_cell_118_matmul_1_readvariableop_resource6while_lstm_cell_118_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_118_matmul_readvariableop_resource4while_lstm_cell_118_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_118/BiasAdd/ReadVariableOp*while/lstm_cell_118/BiasAdd/ReadVariableOp2V
)while/lstm_cell_118/MatMul/ReadVariableOp)while/lstm_cell_118/MatMul/ReadVariableOp2Z
+while/lstm_cell_118/MatMul_1/ReadVariableOp+while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 


J__inference_sequential_39_layer_call_and_return_conditional_losses_4931742

inputs
inputs_1	+
bidirectional_39_4931711:	Ш+
bidirectional_39_4931713:	2Ш'
bidirectional_39_4931715:	Ш+
bidirectional_39_4931717:	Ш+
bidirectional_39_4931719:	2Ш'
bidirectional_39_4931721:	Ш"
dense_39_4931736:d
dense_39_4931738:
identityЂ(bidirectional_39/StatefulPartitionedCallЂ dense_39/StatefulPartitionedCallК
(bidirectional_39/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_39_4931711bidirectional_39_4931713bidirectional_39_4931715bidirectional_39_4931717bidirectional_39_4931719bidirectional_39_4931721*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_49317102*
(bidirectional_39/StatefulPartitionedCallТ
 dense_39/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_39/StatefulPartitionedCall:output:0dense_39_4931736dense_39_4931738*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_39_layer_call_and_return_conditional_losses_49317352"
 dense_39/StatefulPartitionedCall
IdentityIdentity)dense_39/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp)^bidirectional_39/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2T
(bidirectional_39/StatefulPartitionedCall(bidirectional_39/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ј?
к
while_body_4930967
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_119_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_119_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_119_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_119_matmul_readvariableop_resource:	ШG
4while_lstm_cell_119_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_119_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_119/BiasAdd/ReadVariableOpЂ)while/lstm_cell_119/MatMul/ReadVariableOpЂ+while/lstm_cell_119/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeм
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЬ
)while/lstm_cell_119/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_119_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_119/MatMul/ReadVariableOpк
while/lstm_cell_119/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/MatMulв
+while/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_119_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_119/MatMul_1/ReadVariableOpУ
while/lstm_cell_119/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/MatMul_1М
while/lstm_cell_119/addAddV2$while/lstm_cell_119/MatMul:product:0&while/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/addЫ
*while/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_119_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_119/BiasAdd/ReadVariableOpЩ
while/lstm_cell_119/BiasAddBiasAddwhile/lstm_cell_119/add:z:02while/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/BiasAdd
#while/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_119/split/split_dim
while/lstm_cell_119/splitSplit,while/lstm_cell_119/split/split_dim:output:0$while/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_119/split
while/lstm_cell_119/SigmoidSigmoid"while/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Sigmoid
while/lstm_cell_119/Sigmoid_1Sigmoid"while/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Sigmoid_1Ѓ
while/lstm_cell_119/mulMul!while/lstm_cell_119/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/mul
while/lstm_cell_119/ReluRelu"while/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/ReluИ
while/lstm_cell_119/mul_1Mulwhile/lstm_cell_119/Sigmoid:y:0&while/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/mul_1­
while/lstm_cell_119/add_1AddV2while/lstm_cell_119/mul:z:0while/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/add_1
while/lstm_cell_119/Sigmoid_2Sigmoid"while/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Sigmoid_2
while/lstm_cell_119/Relu_1Reluwhile/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Relu_1М
while/lstm_cell_119/mul_2Mul!while/lstm_cell_119/Sigmoid_2:y:0(while/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_119/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_119/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_119/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_119/BiasAdd/ReadVariableOp*^while/lstm_cell_119/MatMul/ReadVariableOp,^while/lstm_cell_119/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_119_biasadd_readvariableop_resource5while_lstm_cell_119_biasadd_readvariableop_resource_0"n
4while_lstm_cell_119_matmul_1_readvariableop_resource6while_lstm_cell_119_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_119_matmul_readvariableop_resource4while_lstm_cell_119_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_119/BiasAdd/ReadVariableOp*while/lstm_cell_119/BiasAdd/ReadVariableOp2V
)while/lstm_cell_119/MatMul/ReadVariableOp)while/lstm_cell_119/MatMul/ReadVariableOp2Z
+while/lstm_cell_119/MatMul_1/ReadVariableOp+while/lstm_cell_119/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
к
Ш
while_cond_4934001
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_4934001___redundant_placeholder05
1while_while_cond_4934001___redundant_placeholder15
1while_while_cond_4934001___redundant_placeholder25
1while_while_cond_4934001___redundant_placeholder3
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
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
к
Ш
while_cond_4934653
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_4934653___redundant_placeholder05
1while_while_cond_4934653___redundant_placeholder15
1while_while_cond_4934653___redundant_placeholder25
1while_while_cond_4934653___redundant_placeholder3
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
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
сV
к
"forward_lstm_39_while_body_4932467<
8forward_lstm_39_while_forward_lstm_39_while_loop_counterB
>forward_lstm_39_while_forward_lstm_39_while_maximum_iterations%
!forward_lstm_39_while_placeholder'
#forward_lstm_39_while_placeholder_1'
#forward_lstm_39_while_placeholder_2'
#forward_lstm_39_while_placeholder_3;
7forward_lstm_39_while_forward_lstm_39_strided_slice_1_0w
sforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0W
Dforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0:	ШY
Fforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0:	2ШT
Eforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0:	Ш"
forward_lstm_39_while_identity$
 forward_lstm_39_while_identity_1$
 forward_lstm_39_while_identity_2$
 forward_lstm_39_while_identity_3$
 forward_lstm_39_while_identity_4$
 forward_lstm_39_while_identity_59
5forward_lstm_39_while_forward_lstm_39_strided_slice_1u
qforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensorU
Bforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource:	ШW
Dforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource:	2ШR
Cforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource:	ШЂ:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOpЂ9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOpЂ;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOpу
Gforward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2I
Gforward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeМ
9forward_lstm_39/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_39_while_placeholderPforward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02;
9forward_lstm_39/while/TensorArrayV2Read/TensorListGetItemќ
9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOpDforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02;
9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp
*forward_lstm_39/while/lstm_cell_118/MatMulMatMul@forward_lstm_39/while/TensorArrayV2Read/TensorListGetItem:item:0Aforward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2,
*forward_lstm_39/while/lstm_cell_118/MatMul
;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOpFforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02=
;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp
,forward_lstm_39/while/lstm_cell_118/MatMul_1MatMul#forward_lstm_39_while_placeholder_2Cforward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_39/while/lstm_cell_118/MatMul_1ќ
'forward_lstm_39/while/lstm_cell_118/addAddV24forward_lstm_39/while/lstm_cell_118/MatMul:product:06forward_lstm_39/while/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_39/while/lstm_cell_118/addћ
:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOpEforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02<
:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp
+forward_lstm_39/while/lstm_cell_118/BiasAddBiasAdd+forward_lstm_39/while/lstm_cell_118/add:z:0Bforward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_39/while/lstm_cell_118/BiasAddЌ
3forward_lstm_39/while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3forward_lstm_39/while/lstm_cell_118/split/split_dimЯ
)forward_lstm_39/while/lstm_cell_118/splitSplit<forward_lstm_39/while/lstm_cell_118/split/split_dim:output:04forward_lstm_39/while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2+
)forward_lstm_39/while/lstm_cell_118/splitЫ
+forward_lstm_39/while/lstm_cell_118/SigmoidSigmoid2forward_lstm_39/while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_39/while/lstm_cell_118/SigmoidЯ
-forward_lstm_39/while/lstm_cell_118/Sigmoid_1Sigmoid2forward_lstm_39/while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22/
-forward_lstm_39/while/lstm_cell_118/Sigmoid_1у
'forward_lstm_39/while/lstm_cell_118/mulMul1forward_lstm_39/while/lstm_cell_118/Sigmoid_1:y:0#forward_lstm_39_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_39/while/lstm_cell_118/mulТ
(forward_lstm_39/while/lstm_cell_118/ReluRelu2forward_lstm_39/while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_39/while/lstm_cell_118/Reluј
)forward_lstm_39/while/lstm_cell_118/mul_1Mul/forward_lstm_39/while/lstm_cell_118/Sigmoid:y:06forward_lstm_39/while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_39/while/lstm_cell_118/mul_1э
)forward_lstm_39/while/lstm_cell_118/add_1AddV2+forward_lstm_39/while/lstm_cell_118/mul:z:0-forward_lstm_39/while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_39/while/lstm_cell_118/add_1Я
-forward_lstm_39/while/lstm_cell_118/Sigmoid_2Sigmoid2forward_lstm_39/while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22/
-forward_lstm_39/while/lstm_cell_118/Sigmoid_2С
*forward_lstm_39/while/lstm_cell_118/Relu_1Relu-forward_lstm_39/while/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_39/while/lstm_cell_118/Relu_1ќ
)forward_lstm_39/while/lstm_cell_118/mul_2Mul1forward_lstm_39/while/lstm_cell_118/Sigmoid_2:y:08forward_lstm_39/while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_39/while/lstm_cell_118/mul_2Б
:forward_lstm_39/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_39_while_placeholder_1!forward_lstm_39_while_placeholder-forward_lstm_39/while/lstm_cell_118/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_39/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_39/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_39/while/add/yЉ
forward_lstm_39/while/addAddV2!forward_lstm_39_while_placeholder$forward_lstm_39/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/while/add
forward_lstm_39/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_39/while/add_1/yЦ
forward_lstm_39/while/add_1AddV28forward_lstm_39_while_forward_lstm_39_while_loop_counter&forward_lstm_39/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/while/add_1Ћ
forward_lstm_39/while/IdentityIdentityforward_lstm_39/while/add_1:z:0^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_39/while/IdentityЮ
 forward_lstm_39/while/Identity_1Identity>forward_lstm_39_while_forward_lstm_39_while_maximum_iterations^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_39/while/Identity_1­
 forward_lstm_39/while/Identity_2Identityforward_lstm_39/while/add:z:0^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_39/while/Identity_2к
 forward_lstm_39/while/Identity_3IdentityJforward_lstm_39/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_39/while/Identity_3Ю
 forward_lstm_39/while/Identity_4Identity-forward_lstm_39/while/lstm_cell_118/mul_2:z:0^forward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_39/while/Identity_4Ю
 forward_lstm_39/while/Identity_5Identity-forward_lstm_39/while/lstm_cell_118/add_1:z:0^forward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_39/while/Identity_5Б
forward_lstm_39/while/NoOpNoOp;^forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp:^forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp<^forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_39/while/NoOp"p
5forward_lstm_39_while_forward_lstm_39_strided_slice_17forward_lstm_39_while_forward_lstm_39_strided_slice_1_0"I
forward_lstm_39_while_identity'forward_lstm_39/while/Identity:output:0"M
 forward_lstm_39_while_identity_1)forward_lstm_39/while/Identity_1:output:0"M
 forward_lstm_39_while_identity_2)forward_lstm_39/while/Identity_2:output:0"M
 forward_lstm_39_while_identity_3)forward_lstm_39/while/Identity_3:output:0"M
 forward_lstm_39_while_identity_4)forward_lstm_39/while/Identity_4:output:0"M
 forward_lstm_39_while_identity_5)forward_lstm_39/while/Identity_5:output:0"
Cforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resourceEforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0"
Dforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resourceFforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0"
Bforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resourceDforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0"ш
qforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensorsforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2x
:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp2v
9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp2z
;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
§^
­
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_4934585
inputs_0?
,lstm_cell_119_matmul_readvariableop_resource:	ШA
.lstm_cell_119_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_119_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_119/BiasAdd/ReadVariableOpЂ#lstm_cell_119/MatMul/ReadVariableOpЂ%lstm_cell_119/MatMul_1/ReadVariableOpЂwhileF
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1
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
:џџџџџџџџџ22
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
B :ш2
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
zeros_1/packed/1
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
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
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
ReverseV2/axis
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	ReverseV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shape§
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
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2И
#lstm_cell_119/MatMul/ReadVariableOpReadVariableOp,lstm_cell_119_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_119/MatMul/ReadVariableOpА
lstm_cell_119/MatMulMatMulstrided_slice_2:output:0+lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/MatMulО
%lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_119_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_119/MatMul_1/ReadVariableOpЌ
lstm_cell_119/MatMul_1MatMulzeros:output:0-lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/MatMul_1Є
lstm_cell_119/addAddV2lstm_cell_119/MatMul:product:0 lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/addЗ
$lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_119_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_119/BiasAdd/ReadVariableOpБ
lstm_cell_119/BiasAddBiasAddlstm_cell_119/add:z:0,lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_119/BiasAdd
lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_119/split/split_dimї
lstm_cell_119/splitSplit&lstm_cell_119/split/split_dim:output:0lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_119/split
lstm_cell_119/SigmoidSigmoidlstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Sigmoid
lstm_cell_119/Sigmoid_1Sigmoidlstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Sigmoid_1
lstm_cell_119/mulMullstm_cell_119/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/mul
lstm_cell_119/ReluRelulstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Relu 
lstm_cell_119/mul_1Mullstm_cell_119/Sigmoid:y:0 lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/mul_1
lstm_cell_119/add_1AddV2lstm_cell_119/mul:z:0lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/add_1
lstm_cell_119/Sigmoid_2Sigmoidlstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Sigmoid_2
lstm_cell_119/Relu_1Relulstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/Relu_1Є
lstm_cell_119/mul_2Mullstm_cell_119/Sigmoid_2:y:0"lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_119/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_119_matmul_readvariableop_resource.lstm_cell_119_matmul_1_readvariableop_resource-lstm_cell_119_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_4934501*
condR
while_cond_4934500*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
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
:џџџџџџџџџ22

IdentityЫ
NoOpNoOp%^lstm_cell_119/BiasAdd/ReadVariableOp$^lstm_cell_119/MatMul/ReadVariableOp&^lstm_cell_119/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_119/BiasAdd/ReadVariableOp$lstm_cell_119/BiasAdd/ReadVariableOp2J
#lstm_cell_119/MatMul/ReadVariableOp#lstm_cell_119/MatMul/ReadVariableOp2N
%lstm_cell_119/MatMul_1/ReadVariableOp%lstm_cell_119/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Жc

"forward_lstm_39_while_body_4933444<
8forward_lstm_39_while_forward_lstm_39_while_loop_counterB
>forward_lstm_39_while_forward_lstm_39_while_maximum_iterations%
!forward_lstm_39_while_placeholder'
#forward_lstm_39_while_placeholder_1'
#forward_lstm_39_while_placeholder_2'
#forward_lstm_39_while_placeholder_3'
#forward_lstm_39_while_placeholder_4;
7forward_lstm_39_while_forward_lstm_39_strided_slice_1_0w
sforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_08
4forward_lstm_39_while_greater_forward_lstm_39_cast_0W
Dforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0:	ШY
Fforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0:	2ШT
Eforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0:	Ш"
forward_lstm_39_while_identity$
 forward_lstm_39_while_identity_1$
 forward_lstm_39_while_identity_2$
 forward_lstm_39_while_identity_3$
 forward_lstm_39_while_identity_4$
 forward_lstm_39_while_identity_5$
 forward_lstm_39_while_identity_69
5forward_lstm_39_while_forward_lstm_39_strided_slice_1u
qforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor6
2forward_lstm_39_while_greater_forward_lstm_39_castU
Bforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource:	ШW
Dforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource:	2ШR
Cforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource:	ШЂ:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOpЂ9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOpЂ;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOpу
Gforward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gforward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
9forward_lstm_39/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_39_while_placeholderPforward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02;
9forward_lstm_39/while/TensorArrayV2Read/TensorListGetItemа
forward_lstm_39/while/GreaterGreater4forward_lstm_39_while_greater_forward_lstm_39_cast_0!forward_lstm_39_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_39/while/Greaterќ
9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOpDforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02;
9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp
*forward_lstm_39/while/lstm_cell_118/MatMulMatMul@forward_lstm_39/while/TensorArrayV2Read/TensorListGetItem:item:0Aforward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2,
*forward_lstm_39/while/lstm_cell_118/MatMul
;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOpFforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02=
;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp
,forward_lstm_39/while/lstm_cell_118/MatMul_1MatMul#forward_lstm_39_while_placeholder_3Cforward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_39/while/lstm_cell_118/MatMul_1ќ
'forward_lstm_39/while/lstm_cell_118/addAddV24forward_lstm_39/while/lstm_cell_118/MatMul:product:06forward_lstm_39/while/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_39/while/lstm_cell_118/addћ
:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOpEforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02<
:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp
+forward_lstm_39/while/lstm_cell_118/BiasAddBiasAdd+forward_lstm_39/while/lstm_cell_118/add:z:0Bforward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_39/while/lstm_cell_118/BiasAddЌ
3forward_lstm_39/while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3forward_lstm_39/while/lstm_cell_118/split/split_dimЯ
)forward_lstm_39/while/lstm_cell_118/splitSplit<forward_lstm_39/while/lstm_cell_118/split/split_dim:output:04forward_lstm_39/while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2+
)forward_lstm_39/while/lstm_cell_118/splitЫ
+forward_lstm_39/while/lstm_cell_118/SigmoidSigmoid2forward_lstm_39/while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_39/while/lstm_cell_118/SigmoidЯ
-forward_lstm_39/while/lstm_cell_118/Sigmoid_1Sigmoid2forward_lstm_39/while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22/
-forward_lstm_39/while/lstm_cell_118/Sigmoid_1у
'forward_lstm_39/while/lstm_cell_118/mulMul1forward_lstm_39/while/lstm_cell_118/Sigmoid_1:y:0#forward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_39/while/lstm_cell_118/mulТ
(forward_lstm_39/while/lstm_cell_118/ReluRelu2forward_lstm_39/while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_39/while/lstm_cell_118/Reluј
)forward_lstm_39/while/lstm_cell_118/mul_1Mul/forward_lstm_39/while/lstm_cell_118/Sigmoid:y:06forward_lstm_39/while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_39/while/lstm_cell_118/mul_1э
)forward_lstm_39/while/lstm_cell_118/add_1AddV2+forward_lstm_39/while/lstm_cell_118/mul:z:0-forward_lstm_39/while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_39/while/lstm_cell_118/add_1Я
-forward_lstm_39/while/lstm_cell_118/Sigmoid_2Sigmoid2forward_lstm_39/while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22/
-forward_lstm_39/while/lstm_cell_118/Sigmoid_2С
*forward_lstm_39/while/lstm_cell_118/Relu_1Relu-forward_lstm_39/while/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_39/while/lstm_cell_118/Relu_1ќ
)forward_lstm_39/while/lstm_cell_118/mul_2Mul1forward_lstm_39/while/lstm_cell_118/Sigmoid_2:y:08forward_lstm_39/while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_39/while/lstm_cell_118/mul_2я
forward_lstm_39/while/SelectSelect!forward_lstm_39/while/Greater:z:0-forward_lstm_39/while/lstm_cell_118/mul_2:z:0#forward_lstm_39_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/while/Selectѓ
forward_lstm_39/while/Select_1Select!forward_lstm_39/while/Greater:z:0-forward_lstm_39/while/lstm_cell_118/mul_2:z:0#forward_lstm_39_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22 
forward_lstm_39/while/Select_1ѓ
forward_lstm_39/while/Select_2Select!forward_lstm_39/while/Greater:z:0-forward_lstm_39/while/lstm_cell_118/add_1:z:0#forward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22 
forward_lstm_39/while/Select_2Љ
:forward_lstm_39/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_39_while_placeholder_1!forward_lstm_39_while_placeholder%forward_lstm_39/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_39/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_39/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_39/while/add/yЉ
forward_lstm_39/while/addAddV2!forward_lstm_39_while_placeholder$forward_lstm_39/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/while/add
forward_lstm_39/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_39/while/add_1/yЦ
forward_lstm_39/while/add_1AddV28forward_lstm_39_while_forward_lstm_39_while_loop_counter&forward_lstm_39/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/while/add_1Ћ
forward_lstm_39/while/IdentityIdentityforward_lstm_39/while/add_1:z:0^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_39/while/IdentityЮ
 forward_lstm_39/while/Identity_1Identity>forward_lstm_39_while_forward_lstm_39_while_maximum_iterations^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_39/while/Identity_1­
 forward_lstm_39/while/Identity_2Identityforward_lstm_39/while/add:z:0^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_39/while/Identity_2к
 forward_lstm_39/while/Identity_3IdentityJforward_lstm_39/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_39/while/Identity_3Ц
 forward_lstm_39/while/Identity_4Identity%forward_lstm_39/while/Select:output:0^forward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_39/while/Identity_4Ш
 forward_lstm_39/while/Identity_5Identity'forward_lstm_39/while/Select_1:output:0^forward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_39/while/Identity_5Ш
 forward_lstm_39/while/Identity_6Identity'forward_lstm_39/while/Select_2:output:0^forward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_39/while/Identity_6Б
forward_lstm_39/while/NoOpNoOp;^forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp:^forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp<^forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_39/while/NoOp"p
5forward_lstm_39_while_forward_lstm_39_strided_slice_17forward_lstm_39_while_forward_lstm_39_strided_slice_1_0"j
2forward_lstm_39_while_greater_forward_lstm_39_cast4forward_lstm_39_while_greater_forward_lstm_39_cast_0"I
forward_lstm_39_while_identity'forward_lstm_39/while/Identity:output:0"M
 forward_lstm_39_while_identity_1)forward_lstm_39/while/Identity_1:output:0"M
 forward_lstm_39_while_identity_2)forward_lstm_39/while/Identity_2:output:0"M
 forward_lstm_39_while_identity_3)forward_lstm_39/while/Identity_3:output:0"M
 forward_lstm_39_while_identity_4)forward_lstm_39/while/Identity_4:output:0"M
 forward_lstm_39_while_identity_5)forward_lstm_39/while/Identity_5:output:0"M
 forward_lstm_39_while_identity_6)forward_lstm_39/while/Identity_6:output:0"
Cforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resourceEforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0"
Dforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resourceFforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0"
Bforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resourceDforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0"ш
qforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensorsforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2x
:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp2v
9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp2z
;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:џџџџџџџџџ
Жc

"forward_lstm_39_while_body_4931874<
8forward_lstm_39_while_forward_lstm_39_while_loop_counterB
>forward_lstm_39_while_forward_lstm_39_while_maximum_iterations%
!forward_lstm_39_while_placeholder'
#forward_lstm_39_while_placeholder_1'
#forward_lstm_39_while_placeholder_2'
#forward_lstm_39_while_placeholder_3'
#forward_lstm_39_while_placeholder_4;
7forward_lstm_39_while_forward_lstm_39_strided_slice_1_0w
sforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_08
4forward_lstm_39_while_greater_forward_lstm_39_cast_0W
Dforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0:	ШY
Fforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0:	2ШT
Eforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0:	Ш"
forward_lstm_39_while_identity$
 forward_lstm_39_while_identity_1$
 forward_lstm_39_while_identity_2$
 forward_lstm_39_while_identity_3$
 forward_lstm_39_while_identity_4$
 forward_lstm_39_while_identity_5$
 forward_lstm_39_while_identity_69
5forward_lstm_39_while_forward_lstm_39_strided_slice_1u
qforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor6
2forward_lstm_39_while_greater_forward_lstm_39_castU
Bforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource:	ШW
Dforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource:	2ШR
Cforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource:	ШЂ:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOpЂ9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOpЂ;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOpу
Gforward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gforward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
9forward_lstm_39/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_39_while_placeholderPforward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02;
9forward_lstm_39/while/TensorArrayV2Read/TensorListGetItemа
forward_lstm_39/while/GreaterGreater4forward_lstm_39_while_greater_forward_lstm_39_cast_0!forward_lstm_39_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_39/while/Greaterќ
9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOpDforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02;
9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp
*forward_lstm_39/while/lstm_cell_118/MatMulMatMul@forward_lstm_39/while/TensorArrayV2Read/TensorListGetItem:item:0Aforward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2,
*forward_lstm_39/while/lstm_cell_118/MatMul
;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOpFforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02=
;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp
,forward_lstm_39/while/lstm_cell_118/MatMul_1MatMul#forward_lstm_39_while_placeholder_3Cforward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_39/while/lstm_cell_118/MatMul_1ќ
'forward_lstm_39/while/lstm_cell_118/addAddV24forward_lstm_39/while/lstm_cell_118/MatMul:product:06forward_lstm_39/while/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_39/while/lstm_cell_118/addћ
:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOpEforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02<
:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp
+forward_lstm_39/while/lstm_cell_118/BiasAddBiasAdd+forward_lstm_39/while/lstm_cell_118/add:z:0Bforward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_39/while/lstm_cell_118/BiasAddЌ
3forward_lstm_39/while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3forward_lstm_39/while/lstm_cell_118/split/split_dimЯ
)forward_lstm_39/while/lstm_cell_118/splitSplit<forward_lstm_39/while/lstm_cell_118/split/split_dim:output:04forward_lstm_39/while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2+
)forward_lstm_39/while/lstm_cell_118/splitЫ
+forward_lstm_39/while/lstm_cell_118/SigmoidSigmoid2forward_lstm_39/while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_39/while/lstm_cell_118/SigmoidЯ
-forward_lstm_39/while/lstm_cell_118/Sigmoid_1Sigmoid2forward_lstm_39/while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22/
-forward_lstm_39/while/lstm_cell_118/Sigmoid_1у
'forward_lstm_39/while/lstm_cell_118/mulMul1forward_lstm_39/while/lstm_cell_118/Sigmoid_1:y:0#forward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_39/while/lstm_cell_118/mulТ
(forward_lstm_39/while/lstm_cell_118/ReluRelu2forward_lstm_39/while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_39/while/lstm_cell_118/Reluј
)forward_lstm_39/while/lstm_cell_118/mul_1Mul/forward_lstm_39/while/lstm_cell_118/Sigmoid:y:06forward_lstm_39/while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_39/while/lstm_cell_118/mul_1э
)forward_lstm_39/while/lstm_cell_118/add_1AddV2+forward_lstm_39/while/lstm_cell_118/mul:z:0-forward_lstm_39/while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_39/while/lstm_cell_118/add_1Я
-forward_lstm_39/while/lstm_cell_118/Sigmoid_2Sigmoid2forward_lstm_39/while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22/
-forward_lstm_39/while/lstm_cell_118/Sigmoid_2С
*forward_lstm_39/while/lstm_cell_118/Relu_1Relu-forward_lstm_39/while/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_39/while/lstm_cell_118/Relu_1ќ
)forward_lstm_39/while/lstm_cell_118/mul_2Mul1forward_lstm_39/while/lstm_cell_118/Sigmoid_2:y:08forward_lstm_39/while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_39/while/lstm_cell_118/mul_2я
forward_lstm_39/while/SelectSelect!forward_lstm_39/while/Greater:z:0-forward_lstm_39/while/lstm_cell_118/mul_2:z:0#forward_lstm_39_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_39/while/Selectѓ
forward_lstm_39/while/Select_1Select!forward_lstm_39/while/Greater:z:0-forward_lstm_39/while/lstm_cell_118/mul_2:z:0#forward_lstm_39_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22 
forward_lstm_39/while/Select_1ѓ
forward_lstm_39/while/Select_2Select!forward_lstm_39/while/Greater:z:0-forward_lstm_39/while/lstm_cell_118/add_1:z:0#forward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22 
forward_lstm_39/while/Select_2Љ
:forward_lstm_39/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_39_while_placeholder_1!forward_lstm_39_while_placeholder%forward_lstm_39/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_39/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_39/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_39/while/add/yЉ
forward_lstm_39/while/addAddV2!forward_lstm_39_while_placeholder$forward_lstm_39/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/while/add
forward_lstm_39/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_39/while/add_1/yЦ
forward_lstm_39/while/add_1AddV28forward_lstm_39_while_forward_lstm_39_while_loop_counter&forward_lstm_39/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_39/while/add_1Ћ
forward_lstm_39/while/IdentityIdentityforward_lstm_39/while/add_1:z:0^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_39/while/IdentityЮ
 forward_lstm_39/while/Identity_1Identity>forward_lstm_39_while_forward_lstm_39_while_maximum_iterations^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_39/while/Identity_1­
 forward_lstm_39/while/Identity_2Identityforward_lstm_39/while/add:z:0^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_39/while/Identity_2к
 forward_lstm_39/while/Identity_3IdentityJforward_lstm_39/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_39/while/Identity_3Ц
 forward_lstm_39/while/Identity_4Identity%forward_lstm_39/while/Select:output:0^forward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_39/while/Identity_4Ш
 forward_lstm_39/while/Identity_5Identity'forward_lstm_39/while/Select_1:output:0^forward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_39/while/Identity_5Ш
 forward_lstm_39/while/Identity_6Identity'forward_lstm_39/while/Select_2:output:0^forward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_39/while/Identity_6Б
forward_lstm_39/while/NoOpNoOp;^forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp:^forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp<^forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_39/while/NoOp"p
5forward_lstm_39_while_forward_lstm_39_strided_slice_17forward_lstm_39_while_forward_lstm_39_strided_slice_1_0"j
2forward_lstm_39_while_greater_forward_lstm_39_cast4forward_lstm_39_while_greater_forward_lstm_39_cast_0"I
forward_lstm_39_while_identity'forward_lstm_39/while/Identity:output:0"M
 forward_lstm_39_while_identity_1)forward_lstm_39/while/Identity_1:output:0"M
 forward_lstm_39_while_identity_2)forward_lstm_39/while/Identity_2:output:0"M
 forward_lstm_39_while_identity_3)forward_lstm_39/while/Identity_3:output:0"M
 forward_lstm_39_while_identity_4)forward_lstm_39/while/Identity_4:output:0"M
 forward_lstm_39_while_identity_5)forward_lstm_39/while/Identity_5:output:0"M
 forward_lstm_39_while_identity_6)forward_lstm_39/while/Identity_6:output:0"
Cforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resourceEforward_lstm_39_while_lstm_cell_118_biasadd_readvariableop_resource_0"
Dforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resourceFforward_lstm_39_while_lstm_cell_118_matmul_1_readvariableop_resource_0"
Bforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resourceDforward_lstm_39_while_lstm_cell_118_matmul_readvariableop_resource_0"ш
qforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensorsforward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2x
:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp:forward_lstm_39/while/lstm_cell_118/BiasAdd/ReadVariableOp2v
9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp9forward_lstm_39/while/lstm_cell_118/MatMul/ReadVariableOp2z
;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp;forward_lstm_39/while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:џџџџџџџџџ
к
Ш
while_cond_4929994
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_4929994___redundant_placeholder05
1while_while_cond_4929994___redundant_placeholder15
1while_while_cond_4929994___redundant_placeholder25
1while_while_cond_4929994___redundant_placeholder3
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
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
к
Ш
while_cond_4931139
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_4931139___redundant_placeholder05
1while_while_cond_4931139___redundant_placeholder15
1while_while_cond_4931139___redundant_placeholder25
1while_while_cond_4931139___redundant_placeholder3
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
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
§

J__inference_lstm_cell_118_layer_call_and_return_conditional_losses_4935142

inputs
states_0
states_11
matmul_readvariableop_resource:	Ш3
 matmul_1_readvariableop_resource:	2Ш.
biasadd_readvariableop_resource:	Ш
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2
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
?:џџџџџџџџџ:џџџџџџџџџ2:џџџџџџџџџ2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/1
я?
к
while_body_4934002
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_118_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_118_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_118_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_118_matmul_readvariableop_resource:	ШG
4while_lstm_cell_118_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_118_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_118/BiasAdd/ReadVariableOpЂ)while/lstm_cell_118/MatMul/ReadVariableOpЂ+while/lstm_cell_118/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЬ
)while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_118/MatMul/ReadVariableOpк
while/lstm_cell_118/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/MatMulв
+while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_118/MatMul_1/ReadVariableOpУ
while/lstm_cell_118/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/MatMul_1М
while/lstm_cell_118/addAddV2$while/lstm_cell_118/MatMul:product:0&while/lstm_cell_118/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/addЫ
*while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_118/BiasAdd/ReadVariableOpЩ
while/lstm_cell_118/BiasAddBiasAddwhile/lstm_cell_118/add:z:02while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_118/BiasAdd
#while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_118/split/split_dim
while/lstm_cell_118/splitSplit,while/lstm_cell_118/split/split_dim:output:0$while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_118/split
while/lstm_cell_118/SigmoidSigmoid"while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Sigmoid
while/lstm_cell_118/Sigmoid_1Sigmoid"while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Sigmoid_1Ѓ
while/lstm_cell_118/mulMul!while/lstm_cell_118/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/mul
while/lstm_cell_118/ReluRelu"while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/ReluИ
while/lstm_cell_118/mul_1Mulwhile/lstm_cell_118/Sigmoid:y:0&while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/mul_1­
while/lstm_cell_118/add_1AddV2while/lstm_cell_118/mul:z:0while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/add_1
while/lstm_cell_118/Sigmoid_2Sigmoid"while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Sigmoid_2
while/lstm_cell_118/Relu_1Reluwhile/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/Relu_1М
while/lstm_cell_118/mul_2Mul!while/lstm_cell_118/Sigmoid_2:y:0(while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_118/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_118/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_118/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_118/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_118/BiasAdd/ReadVariableOp*^while/lstm_cell_118/MatMul/ReadVariableOp,^while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_118_biasadd_readvariableop_resource5while_lstm_cell_118_biasadd_readvariableop_resource_0"n
4while_lstm_cell_118_matmul_1_readvariableop_resource6while_lstm_cell_118_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_118_matmul_readvariableop_resource4while_lstm_cell_118_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_118/BiasAdd/ReadVariableOp*while/lstm_cell_118/BiasAdd/ReadVariableOp2V
)while/lstm_cell_118/MatMul/ReadVariableOp)while/lstm_cell_118/MatMul/ReadVariableOp2Z
+while/lstm_cell_118/MatMul_1/ReadVariableOp+while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
к

#backward_lstm_39_while_cond_4932917>
:backward_lstm_39_while_backward_lstm_39_while_loop_counterD
@backward_lstm_39_while_backward_lstm_39_while_maximum_iterations&
"backward_lstm_39_while_placeholder(
$backward_lstm_39_while_placeholder_1(
$backward_lstm_39_while_placeholder_2(
$backward_lstm_39_while_placeholder_3@
<backward_lstm_39_while_less_backward_lstm_39_strided_slice_1W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4932917___redundant_placeholder0W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4932917___redundant_placeholder1W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4932917___redundant_placeholder2W
Sbackward_lstm_39_while_backward_lstm_39_while_cond_4932917___redundant_placeholder3#
backward_lstm_39_while_identity
Х
backward_lstm_39/while/LessLess"backward_lstm_39_while_placeholder<backward_lstm_39_while_less_backward_lstm_39_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_39/while/Less
backward_lstm_39/while/IdentityIdentitybackward_lstm_39/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_39/while/Identity"K
backward_lstm_39_while_identity(backward_lstm_39/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Оb
­
 __inference__traced_save_4935381
file_prefix.
*savev2_dense_39_kernel_read_readvariableop,
(savev2_dense_39_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopT
Psavev2_bidirectional_39_forward_lstm_39_lstm_cell_118_kernel_read_readvariableop^
Zsavev2_bidirectional_39_forward_lstm_39_lstm_cell_118_recurrent_kernel_read_readvariableopR
Nsavev2_bidirectional_39_forward_lstm_39_lstm_cell_118_bias_read_readvariableopU
Qsavev2_bidirectional_39_backward_lstm_39_lstm_cell_119_kernel_read_readvariableop_
[savev2_bidirectional_39_backward_lstm_39_lstm_cell_119_recurrent_kernel_read_readvariableopS
Osavev2_bidirectional_39_backward_lstm_39_lstm_cell_119_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_39_kernel_m_read_readvariableop3
/savev2_adam_dense_39_bias_m_read_readvariableop[
Wsavev2_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_kernel_m_read_readvariableope
asavev2_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_recurrent_kernel_m_read_readvariableopY
Usavev2_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_bias_m_read_readvariableop\
Xsavev2_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_kernel_m_read_readvariableopf
bsavev2_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_recurrent_kernel_m_read_readvariableopZ
Vsavev2_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_bias_m_read_readvariableop5
1savev2_adam_dense_39_kernel_v_read_readvariableop3
/savev2_adam_dense_39_bias_v_read_readvariableop[
Wsavev2_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_kernel_v_read_readvariableope
asavev2_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_recurrent_kernel_v_read_readvariableopY
Usavev2_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_bias_v_read_readvariableop\
Xsavev2_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_kernel_v_read_readvariableopf
bsavev2_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_recurrent_kernel_v_read_readvariableopZ
Vsavev2_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_bias_v_read_readvariableop8
4savev2_adam_dense_39_kernel_vhat_read_readvariableop6
2savev2_adam_dense_39_bias_vhat_read_readvariableop^
Zsavev2_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_kernel_vhat_read_readvariableoph
dsavev2_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_recurrent_kernel_vhat_read_readvariableop\
Xsavev2_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_bias_vhat_read_readvariableop_
[savev2_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_kernel_vhat_read_readvariableopi
esavev2_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_recurrent_kernel_vhat_read_readvariableop]
Ysavev2_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_bias_vhat_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
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
Const_1
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЂ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*Д
valueЊBЇ(B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesи
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_39_kernel_read_readvariableop(savev2_dense_39_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopPsavev2_bidirectional_39_forward_lstm_39_lstm_cell_118_kernel_read_readvariableopZsavev2_bidirectional_39_forward_lstm_39_lstm_cell_118_recurrent_kernel_read_readvariableopNsavev2_bidirectional_39_forward_lstm_39_lstm_cell_118_bias_read_readvariableopQsavev2_bidirectional_39_backward_lstm_39_lstm_cell_119_kernel_read_readvariableop[savev2_bidirectional_39_backward_lstm_39_lstm_cell_119_recurrent_kernel_read_readvariableopOsavev2_bidirectional_39_backward_lstm_39_lstm_cell_119_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_39_kernel_m_read_readvariableop/savev2_adam_dense_39_bias_m_read_readvariableopWsavev2_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_kernel_m_read_readvariableopasavev2_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_recurrent_kernel_m_read_readvariableopUsavev2_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_bias_m_read_readvariableopXsavev2_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_kernel_m_read_readvariableopbsavev2_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_recurrent_kernel_m_read_readvariableopVsavev2_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_bias_m_read_readvariableop1savev2_adam_dense_39_kernel_v_read_readvariableop/savev2_adam_dense_39_bias_v_read_readvariableopWsavev2_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_kernel_v_read_readvariableopasavev2_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_recurrent_kernel_v_read_readvariableopUsavev2_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_bias_v_read_readvariableopXsavev2_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_kernel_v_read_readvariableopbsavev2_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_recurrent_kernel_v_read_readvariableopVsavev2_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_bias_v_read_readvariableop4savev2_adam_dense_39_kernel_vhat_read_readvariableop2savev2_adam_dense_39_bias_vhat_read_readvariableopZsavev2_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_kernel_vhat_read_readvariableopdsavev2_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_recurrent_kernel_vhat_read_readvariableopXsavev2_adam_bidirectional_39_forward_lstm_39_lstm_cell_118_bias_vhat_read_readvariableop[savev2_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_kernel_vhat_read_readvariableopesavev2_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_recurrent_kernel_vhat_read_readvariableopYsavev2_adam_bidirectional_39_backward_lstm_39_lstm_cell_119_bias_vhat_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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

identity_1Identity_1:output:0*Я
_input_shapesН
К: :d:: : : : : :	Ш:	2Ш:Ш:	Ш:	2Ш:Ш: : :d::	Ш:	2Ш:Ш:	Ш:	2Ш:Ш:d::	Ш:	2Ш:Ш:	Ш:	2Ш:Ш:d::	Ш:	2Ш:Ш:	Ш:	2Ш:Ш: 2(
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
:	Ш:%	!

_output_shapes
:	2Ш:!


_output_shapes	
:Ш:%!

_output_shapes
:	Ш:%!

_output_shapes
:	2Ш:!

_output_shapes	
:Ш:
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
:	Ш:%!

_output_shapes
:	2Ш:!

_output_shapes	
:Ш:%!

_output_shapes
:	Ш:%!

_output_shapes
:	2Ш:!

_output_shapes	
:Ш:$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	Ш:%!

_output_shapes
:	2Ш:!

_output_shapes	
:Ш:%!

_output_shapes
:	Ш:%!

_output_shapes
:	2Ш:!

_output_shapes	
:Ш:$  

_output_shapes

:d: !

_output_shapes
::%"!

_output_shapes
:	Ш:%#!

_output_shapes
:	2Ш:!$

_output_shapes	
:Ш:%%!

_output_shapes
:	Ш:%&!

_output_shapes
:	2Ш:!'

_output_shapes	
:Ш:(

_output_shapes
: 
ј?
к
while_body_4930775
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_119_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_119_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_119_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_119_matmul_readvariableop_resource:	ШG
4while_lstm_cell_119_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_119_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_119/BiasAdd/ReadVariableOpЂ)while/lstm_cell_119/MatMul/ReadVariableOpЂ+while/lstm_cell_119/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeм
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЬ
)while/lstm_cell_119/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_119_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_119/MatMul/ReadVariableOpк
while/lstm_cell_119/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/MatMulв
+while/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_119_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_119/MatMul_1/ReadVariableOpУ
while/lstm_cell_119/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/MatMul_1М
while/lstm_cell_119/addAddV2$while/lstm_cell_119/MatMul:product:0&while/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/addЫ
*while/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_119_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_119/BiasAdd/ReadVariableOpЩ
while/lstm_cell_119/BiasAddBiasAddwhile/lstm_cell_119/add:z:02while/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_119/BiasAdd
#while/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_119/split/split_dim
while/lstm_cell_119/splitSplit,while/lstm_cell_119/split/split_dim:output:0$while/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_119/split
while/lstm_cell_119/SigmoidSigmoid"while/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Sigmoid
while/lstm_cell_119/Sigmoid_1Sigmoid"while/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Sigmoid_1Ѓ
while/lstm_cell_119/mulMul!while/lstm_cell_119/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/mul
while/lstm_cell_119/ReluRelu"while/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/ReluИ
while/lstm_cell_119/mul_1Mulwhile/lstm_cell_119/Sigmoid:y:0&while/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/mul_1­
while/lstm_cell_119/add_1AddV2while/lstm_cell_119/mul:z:0while/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/add_1
while/lstm_cell_119/Sigmoid_2Sigmoid"while/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Sigmoid_2
while/lstm_cell_119/Relu_1Reluwhile/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/Relu_1М
while/lstm_cell_119/mul_2Mul!while/lstm_cell_119/Sigmoid_2:y:0(while/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_119/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_119/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_119/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_119/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_119/BiasAdd/ReadVariableOp*^while/lstm_cell_119/MatMul/ReadVariableOp,^while/lstm_cell_119/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_119_biasadd_readvariableop_resource5while_lstm_cell_119_biasadd_readvariableop_resource_0"n
4while_lstm_cell_119_matmul_1_readvariableop_resource6while_lstm_cell_119_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_119_matmul_readvariableop_resource4while_lstm_cell_119_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_119/BiasAdd/ReadVariableOp*while/lstm_cell_119/BiasAdd/ReadVariableOp2V
)while/lstm_cell_119/MatMul/ReadVariableOp)while/lstm_cell_119/MatMul/ReadVariableOp2Z
+while/lstm_cell_119/MatMul_1/ReadVariableOp+while/lstm_cell_119/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
ѕ

J__inference_lstm_cell_119_layer_call_and_return_conditional_losses_4930127

inputs

states
states_11
matmul_readvariableop_resource:	Ш3
 matmul_1_readvariableop_resource:	2Ш.
biasadd_readvariableop_resource:	Ш
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2
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
?:џџџџџџџџџ:џџџџџџџџџ2:џџџџџџџџџ2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates
§

J__inference_lstm_cell_118_layer_call_and_return_conditional_losses_4935110

inputs
states_0
states_11
matmul_readvariableop_resource:	Ш3
 matmul_1_readvariableop_resource:	2Ш.
biasadd_readvariableop_resource:	Ш
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2
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
?:џџџџџџџџџ:џџџџџџџџџ2:џџџџџџџџџ2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/1
ч	

2__inference_bidirectional_39_layer_call_fn_4932347
inputs_0
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
	unknown_2:	Ш
	unknown_3:	2Ш
	unknown_4:	Ш
identityЂStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_49308702
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
м
О
1__inference_forward_lstm_39_layer_call_fn_4933773

inputs
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_49306992
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
§

J__inference_lstm_cell_119_layer_call_and_return_conditional_losses_4935208

inputs
states_0
states_11
matmul_readvariableop_resource:	Ш3
 matmul_1_readvariableop_resource:	2Ш.
biasadd_readvariableop_resource:	Ш
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2
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
?:џџџџџџџџџ:џџџџџџџџџ2:џџџџџџџџџ2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/1
X
њ
#backward_lstm_39_while_body_4932918>
:backward_lstm_39_while_backward_lstm_39_while_loop_counterD
@backward_lstm_39_while_backward_lstm_39_while_maximum_iterations&
"backward_lstm_39_while_placeholder(
$backward_lstm_39_while_placeholder_1(
$backward_lstm_39_while_placeholder_2(
$backward_lstm_39_while_placeholder_3=
9backward_lstm_39_while_backward_lstm_39_strided_slice_1_0y
ubackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0X
Ebackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0:	ШZ
Gbackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0:	2ШU
Fbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0:	Ш#
backward_lstm_39_while_identity%
!backward_lstm_39_while_identity_1%
!backward_lstm_39_while_identity_2%
!backward_lstm_39_while_identity_3%
!backward_lstm_39_while_identity_4%
!backward_lstm_39_while_identity_5;
7backward_lstm_39_while_backward_lstm_39_strided_slice_1w
sbackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensorV
Cbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource:	ШX
Ebackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource:	2ШS
Dbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource:	ШЂ;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOpЂ:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOpЂ<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOpх
Hbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2J
Hbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeТ
:backward_lstm_39/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_39_while_placeholderQbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02<
:backward_lstm_39/while/TensorArrayV2Read/TensorListGetItemџ
:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOpReadVariableOpEbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp
+backward_lstm_39/while/lstm_cell_119/MatMulMatMulAbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem:item:0Bbackward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+backward_lstm_39/while/lstm_cell_119/MatMul
<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOpGbackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp
-backward_lstm_39/while/lstm_cell_119/MatMul_1MatMul$backward_lstm_39_while_placeholder_2Dbackward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_39/while/lstm_cell_119/MatMul_1
(backward_lstm_39/while/lstm_cell_119/addAddV25backward_lstm_39/while/lstm_cell_119/MatMul:product:07backward_lstm_39/while/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_39/while/lstm_cell_119/addў
;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOpFbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp
,backward_lstm_39/while/lstm_cell_119/BiasAddBiasAdd,backward_lstm_39/while/lstm_cell_119/add:z:0Cbackward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_39/while/lstm_cell_119/BiasAddЎ
4backward_lstm_39/while/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4backward_lstm_39/while/lstm_cell_119/split/split_dimг
*backward_lstm_39/while/lstm_cell_119/splitSplit=backward_lstm_39/while/lstm_cell_119/split/split_dim:output:05backward_lstm_39/while/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*backward_lstm_39/while/lstm_cell_119/splitЮ
,backward_lstm_39/while/lstm_cell_119/SigmoidSigmoid3backward_lstm_39/while/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_39/while/lstm_cell_119/Sigmoidв
.backward_lstm_39/while/lstm_cell_119/Sigmoid_1Sigmoid3backward_lstm_39/while/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.backward_lstm_39/while/lstm_cell_119/Sigmoid_1ч
(backward_lstm_39/while/lstm_cell_119/mulMul2backward_lstm_39/while/lstm_cell_119/Sigmoid_1:y:0$backward_lstm_39_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_39/while/lstm_cell_119/mulХ
)backward_lstm_39/while/lstm_cell_119/ReluRelu3backward_lstm_39/while/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_39/while/lstm_cell_119/Reluќ
*backward_lstm_39/while/lstm_cell_119/mul_1Mul0backward_lstm_39/while/lstm_cell_119/Sigmoid:y:07backward_lstm_39/while/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_39/while/lstm_cell_119/mul_1ё
*backward_lstm_39/while/lstm_cell_119/add_1AddV2,backward_lstm_39/while/lstm_cell_119/mul:z:0.backward_lstm_39/while/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_39/while/lstm_cell_119/add_1в
.backward_lstm_39/while/lstm_cell_119/Sigmoid_2Sigmoid3backward_lstm_39/while/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.backward_lstm_39/while/lstm_cell_119/Sigmoid_2Ф
+backward_lstm_39/while/lstm_cell_119/Relu_1Relu.backward_lstm_39/while/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_39/while/lstm_cell_119/Relu_1
*backward_lstm_39/while/lstm_cell_119/mul_2Mul2backward_lstm_39/while/lstm_cell_119/Sigmoid_2:y:09backward_lstm_39/while/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_39/while/lstm_cell_119/mul_2Ж
;backward_lstm_39/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_39_while_placeholder_1"backward_lstm_39_while_placeholder.backward_lstm_39/while/lstm_cell_119/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_39/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_39/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_39/while/add/y­
backward_lstm_39/while/addAddV2"backward_lstm_39_while_placeholder%backward_lstm_39/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/while/add
backward_lstm_39/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_39/while/add_1/yЫ
backward_lstm_39/while/add_1AddV2:backward_lstm_39_while_backward_lstm_39_while_loop_counter'backward_lstm_39/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/while/add_1Џ
backward_lstm_39/while/IdentityIdentity backward_lstm_39/while/add_1:z:0^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_39/while/Identityг
!backward_lstm_39/while/Identity_1Identity@backward_lstm_39_while_backward_lstm_39_while_maximum_iterations^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_39/while/Identity_1Б
!backward_lstm_39/while/Identity_2Identitybackward_lstm_39/while/add:z:0^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_39/while/Identity_2о
!backward_lstm_39/while/Identity_3IdentityKbackward_lstm_39/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_39/while/Identity_3в
!backward_lstm_39/while/Identity_4Identity.backward_lstm_39/while/lstm_cell_119/mul_2:z:0^backward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_39/while/Identity_4в
!backward_lstm_39/while/Identity_5Identity.backward_lstm_39/while/lstm_cell_119/add_1:z:0^backward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_39/while/Identity_5Ж
backward_lstm_39/while/NoOpNoOp<^backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp;^backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp=^backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_39/while/NoOp"t
7backward_lstm_39_while_backward_lstm_39_strided_slice_19backward_lstm_39_while_backward_lstm_39_strided_slice_1_0"K
backward_lstm_39_while_identity(backward_lstm_39/while/Identity:output:0"O
!backward_lstm_39_while_identity_1*backward_lstm_39/while/Identity_1:output:0"O
!backward_lstm_39_while_identity_2*backward_lstm_39/while/Identity_2:output:0"O
!backward_lstm_39_while_identity_3*backward_lstm_39/while/Identity_3:output:0"O
!backward_lstm_39_while_identity_4*backward_lstm_39/while/Identity_4:output:0"O
!backward_lstm_39_while_identity_5*backward_lstm_39/while/Identity_5:output:0"
Dbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resourceFbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0"
Ebackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resourceGbackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0"
Cbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resourceEbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0"ь
sbackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensorubackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2z
;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp2x
:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp2|
<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
о
П
2__inference_backward_lstm_39_layer_call_fn_4934432

inputs
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_49310512
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
єd
Н
#backward_lstm_39_while_body_4932053>
:backward_lstm_39_while_backward_lstm_39_while_loop_counterD
@backward_lstm_39_while_backward_lstm_39_while_maximum_iterations&
"backward_lstm_39_while_placeholder(
$backward_lstm_39_while_placeholder_1(
$backward_lstm_39_while_placeholder_2(
$backward_lstm_39_while_placeholder_3(
$backward_lstm_39_while_placeholder_4=
9backward_lstm_39_while_backward_lstm_39_strided_slice_1_0y
ubackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_08
4backward_lstm_39_while_less_backward_lstm_39_sub_1_0X
Ebackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0:	ШZ
Gbackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0:	2ШU
Fbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0:	Ш#
backward_lstm_39_while_identity%
!backward_lstm_39_while_identity_1%
!backward_lstm_39_while_identity_2%
!backward_lstm_39_while_identity_3%
!backward_lstm_39_while_identity_4%
!backward_lstm_39_while_identity_5%
!backward_lstm_39_while_identity_6;
7backward_lstm_39_while_backward_lstm_39_strided_slice_1w
sbackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor6
2backward_lstm_39_while_less_backward_lstm_39_sub_1V
Cbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource:	ШX
Ebackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource:	2ШS
Dbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource:	ШЂ;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOpЂ:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOpЂ<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOpх
Hbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2J
Hbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
:backward_lstm_39/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_39_while_placeholderQbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02<
:backward_lstm_39/while/TensorArrayV2Read/TensorListGetItemЪ
backward_lstm_39/while/LessLess4backward_lstm_39_while_less_backward_lstm_39_sub_1_0"backward_lstm_39_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_39/while/Lessџ
:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOpReadVariableOpEbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp
+backward_lstm_39/while/lstm_cell_119/MatMulMatMulAbackward_lstm_39/while/TensorArrayV2Read/TensorListGetItem:item:0Bbackward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+backward_lstm_39/while/lstm_cell_119/MatMul
<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOpReadVariableOpGbackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp
-backward_lstm_39/while/lstm_cell_119/MatMul_1MatMul$backward_lstm_39_while_placeholder_3Dbackward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_39/while/lstm_cell_119/MatMul_1
(backward_lstm_39/while/lstm_cell_119/addAddV25backward_lstm_39/while/lstm_cell_119/MatMul:product:07backward_lstm_39/while/lstm_cell_119/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_39/while/lstm_cell_119/addў
;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOpReadVariableOpFbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp
,backward_lstm_39/while/lstm_cell_119/BiasAddBiasAdd,backward_lstm_39/while/lstm_cell_119/add:z:0Cbackward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_39/while/lstm_cell_119/BiasAddЎ
4backward_lstm_39/while/lstm_cell_119/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4backward_lstm_39/while/lstm_cell_119/split/split_dimг
*backward_lstm_39/while/lstm_cell_119/splitSplit=backward_lstm_39/while/lstm_cell_119/split/split_dim:output:05backward_lstm_39/while/lstm_cell_119/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*backward_lstm_39/while/lstm_cell_119/splitЮ
,backward_lstm_39/while/lstm_cell_119/SigmoidSigmoid3backward_lstm_39/while/lstm_cell_119/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_39/while/lstm_cell_119/Sigmoidв
.backward_lstm_39/while/lstm_cell_119/Sigmoid_1Sigmoid3backward_lstm_39/while/lstm_cell_119/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.backward_lstm_39/while/lstm_cell_119/Sigmoid_1ч
(backward_lstm_39/while/lstm_cell_119/mulMul2backward_lstm_39/while/lstm_cell_119/Sigmoid_1:y:0$backward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_39/while/lstm_cell_119/mulХ
)backward_lstm_39/while/lstm_cell_119/ReluRelu3backward_lstm_39/while/lstm_cell_119/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_39/while/lstm_cell_119/Reluќ
*backward_lstm_39/while/lstm_cell_119/mul_1Mul0backward_lstm_39/while/lstm_cell_119/Sigmoid:y:07backward_lstm_39/while/lstm_cell_119/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_39/while/lstm_cell_119/mul_1ё
*backward_lstm_39/while/lstm_cell_119/add_1AddV2,backward_lstm_39/while/lstm_cell_119/mul:z:0.backward_lstm_39/while/lstm_cell_119/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_39/while/lstm_cell_119/add_1в
.backward_lstm_39/while/lstm_cell_119/Sigmoid_2Sigmoid3backward_lstm_39/while/lstm_cell_119/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.backward_lstm_39/while/lstm_cell_119/Sigmoid_2Ф
+backward_lstm_39/while/lstm_cell_119/Relu_1Relu.backward_lstm_39/while/lstm_cell_119/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_39/while/lstm_cell_119/Relu_1
*backward_lstm_39/while/lstm_cell_119/mul_2Mul2backward_lstm_39/while/lstm_cell_119/Sigmoid_2:y:09backward_lstm_39/while/lstm_cell_119/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_39/while/lstm_cell_119/mul_2ё
backward_lstm_39/while/SelectSelectbackward_lstm_39/while/Less:z:0.backward_lstm_39/while/lstm_cell_119/mul_2:z:0$backward_lstm_39_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_39/while/Selectѕ
backward_lstm_39/while/Select_1Selectbackward_lstm_39/while/Less:z:0.backward_lstm_39/while/lstm_cell_119/mul_2:z:0$backward_lstm_39_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22!
backward_lstm_39/while/Select_1ѕ
backward_lstm_39/while/Select_2Selectbackward_lstm_39/while/Less:z:0.backward_lstm_39/while/lstm_cell_119/add_1:z:0$backward_lstm_39_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22!
backward_lstm_39/while/Select_2Ў
;backward_lstm_39/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_39_while_placeholder_1"backward_lstm_39_while_placeholder&backward_lstm_39/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_39/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_39/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_39/while/add/y­
backward_lstm_39/while/addAddV2"backward_lstm_39_while_placeholder%backward_lstm_39/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/while/add
backward_lstm_39/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_39/while/add_1/yЫ
backward_lstm_39/while/add_1AddV2:backward_lstm_39_while_backward_lstm_39_while_loop_counter'backward_lstm_39/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_39/while/add_1Џ
backward_lstm_39/while/IdentityIdentity backward_lstm_39/while/add_1:z:0^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_39/while/Identityг
!backward_lstm_39/while/Identity_1Identity@backward_lstm_39_while_backward_lstm_39_while_maximum_iterations^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_39/while/Identity_1Б
!backward_lstm_39/while/Identity_2Identitybackward_lstm_39/while/add:z:0^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_39/while/Identity_2о
!backward_lstm_39/while/Identity_3IdentityKbackward_lstm_39/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_39/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_39/while/Identity_3Ъ
!backward_lstm_39/while/Identity_4Identity&backward_lstm_39/while/Select:output:0^backward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_39/while/Identity_4Ь
!backward_lstm_39/while/Identity_5Identity(backward_lstm_39/while/Select_1:output:0^backward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_39/while/Identity_5Ь
!backward_lstm_39/while/Identity_6Identity(backward_lstm_39/while/Select_2:output:0^backward_lstm_39/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_39/while/Identity_6Ж
backward_lstm_39/while/NoOpNoOp<^backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp;^backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp=^backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_39/while/NoOp"t
7backward_lstm_39_while_backward_lstm_39_strided_slice_19backward_lstm_39_while_backward_lstm_39_strided_slice_1_0"K
backward_lstm_39_while_identity(backward_lstm_39/while/Identity:output:0"O
!backward_lstm_39_while_identity_1*backward_lstm_39/while/Identity_1:output:0"O
!backward_lstm_39_while_identity_2*backward_lstm_39/while/Identity_2:output:0"O
!backward_lstm_39_while_identity_3*backward_lstm_39/while/Identity_3:output:0"O
!backward_lstm_39_while_identity_4*backward_lstm_39/while/Identity_4:output:0"O
!backward_lstm_39_while_identity_5*backward_lstm_39/while/Identity_5:output:0"O
!backward_lstm_39_while_identity_6*backward_lstm_39/while/Identity_6:output:0"j
2backward_lstm_39_while_less_backward_lstm_39_sub_14backward_lstm_39_while_less_backward_lstm_39_sub_1_0"
Dbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resourceFbackward_lstm_39_while_lstm_cell_119_biasadd_readvariableop_resource_0"
Ebackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resourceGbackward_lstm_39_while_lstm_cell_119_matmul_1_readvariableop_resource_0"
Cbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resourceEbackward_lstm_39_while_lstm_cell_119_matmul_readvariableop_resource_0"ь
sbackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensorubackward_lstm_39_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_39_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2z
;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp;backward_lstm_39/while/lstm_cell_119/BiasAdd/ReadVariableOp2x
:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp:backward_lstm_39/while/lstm_cell_119/MatMul/ReadVariableOp2|
<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp<backward_lstm_39/while/lstm_cell_119/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:џџџџџџџџџ
ѕ

J__inference_lstm_cell_118_layer_call_and_return_conditional_losses_4929349

inputs

states
states_11
matmul_readvariableop_resource:	Ш3
 matmul_1_readvariableop_resource:	2Ш.
biasadd_readvariableop_resource:	Ш
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2
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
?:џџџџџџџџџ:џџџџџџџџџ2:џџџџџџџџџ2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates
К
ј
/__inference_lstm_cell_119_layer_call_fn_4935176

inputs
states_0
states_1
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identity

identity_1

identity_2ЂStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_119_layer_call_and_return_conditional_losses_49301272
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

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
?:џџџџџџџџџ:џџџџџџџџџ2:џџџџџџџџџ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/1
ѕ

J__inference_lstm_cell_119_layer_call_and_return_conditional_losses_4929981

inputs

states
states_11
matmul_readvariableop_resource:	Ш3
 matmul_1_readvariableop_resource:	2Ш.
biasadd_readvariableop_resource:	Ш
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2
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
?:џџџџџџџџџ:џџџџџџџџџ2:џџџџџџџџџ2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates
в
С
2__inference_backward_lstm_39_layer_call_fn_4934399
inputs_0
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_49300642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
ј

ж
/__inference_sequential_39_layer_call_fn_4931761

inputs
inputs_1	
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
	unknown_2:	Ш
	unknown_3:	2Ш
	unknown_4:	Ш
	unknown_5:d
	unknown_6:
identityЂStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_39_layer_call_and_return_conditional_losses_49317422
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ю
ѓ
Bsequential_39_bidirectional_39_backward_lstm_39_while_cond_4929169|
xsequential_39_bidirectional_39_backward_lstm_39_while_sequential_39_bidirectional_39_backward_lstm_39_while_loop_counter
~sequential_39_bidirectional_39_backward_lstm_39_while_sequential_39_bidirectional_39_backward_lstm_39_while_maximum_iterationsE
Asequential_39_bidirectional_39_backward_lstm_39_while_placeholderG
Csequential_39_bidirectional_39_backward_lstm_39_while_placeholder_1G
Csequential_39_bidirectional_39_backward_lstm_39_while_placeholder_2G
Csequential_39_bidirectional_39_backward_lstm_39_while_placeholder_3G
Csequential_39_bidirectional_39_backward_lstm_39_while_placeholder_4~
zsequential_39_bidirectional_39_backward_lstm_39_while_less_sequential_39_bidirectional_39_backward_lstm_39_strided_slice_1
sequential_39_bidirectional_39_backward_lstm_39_while_sequential_39_bidirectional_39_backward_lstm_39_while_cond_4929169___redundant_placeholder0
sequential_39_bidirectional_39_backward_lstm_39_while_sequential_39_bidirectional_39_backward_lstm_39_while_cond_4929169___redundant_placeholder1
sequential_39_bidirectional_39_backward_lstm_39_while_sequential_39_bidirectional_39_backward_lstm_39_while_cond_4929169___redundant_placeholder2
sequential_39_bidirectional_39_backward_lstm_39_while_sequential_39_bidirectional_39_backward_lstm_39_while_cond_4929169___redundant_placeholder3
sequential_39_bidirectional_39_backward_lstm_39_while_sequential_39_bidirectional_39_backward_lstm_39_while_cond_4929169___redundant_placeholder4B
>sequential_39_bidirectional_39_backward_lstm_39_while_identity
р
:sequential_39/bidirectional_39/backward_lstm_39/while/LessLessAsequential_39_bidirectional_39_backward_lstm_39_while_placeholderzsequential_39_bidirectional_39_backward_lstm_39_while_less_sequential_39_bidirectional_39_backward_lstm_39_strided_slice_1*
T0*
_output_shapes
: 2<
:sequential_39/bidirectional_39/backward_lstm_39/while/Lessэ
>sequential_39/bidirectional_39/backward_lstm_39/while/IdentityIdentity>sequential_39/bidirectional_39/backward_lstm_39/while/Less:z:0*
T0
*
_output_shapes
: 2@
>sequential_39/bidirectional_39/backward_lstm_39/while/Identity"
>sequential_39_bidirectional_39_backward_lstm_39_while_identityGsequential_39/bidirectional_39/backward_lstm_39/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: :::::: 
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
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
К
ј
/__inference_lstm_cell_118_layer_call_fn_4935078

inputs
states_0
states_1
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identity

identity_1

identity_2ЂStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_118_layer_call_and_return_conditional_losses_49294952
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22

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
?:џџџџџџџџџ:џџџџџџџџџ2:џџџџџџџџџ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/1"ЈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ф
serving_defaultа
9
args_0/
serving_default_args_0:0џџџџџџџџџ
9
args_0_1-
serving_default_args_0_1:0	џџџџџџџџџ<
dense_390
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:К
Д
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
Ь
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
Л

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
У
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
Ъ
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
Х
%cell
&
state_spec
'regularization_losses
(	variables
)trainable_variables
*	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
Х
+cell
,
state_spec
-regularization_losses
.	variables
/trainable_variables
0	keras_api
__call__
+&call_and_return_all_conditional_losses"
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
­
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
!:d2dense_39/kernel
:2dense_39/bias
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
­
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
H:F	Ш25bidirectional_39/forward_lstm_39/lstm_cell_118/kernel
R:P	2Ш2?bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel
B:@Ш23bidirectional_39/forward_lstm_39/lstm_cell_118/bias
I:G	Ш26bidirectional_39/backward_lstm_39/lstm_cell_119/kernel
S:Q	2Ш2@bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel
C:AШ24bidirectional_39/backward_lstm_39/lstm_cell_119/bias
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
у
<
state_size

kernel
recurrent_kernel
bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
__call__
+&call_and_return_all_conditional_losses"
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
М
Alayer_metrics
Bnon_trainable_variables

Cstates
'regularization_losses
(	variables
Dmetrics
)trainable_variables
Elayer_regularization_losses

Flayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
у
G
state_size

kernel
recurrent_kernel
bias
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
__call__
+&call_and_return_all_conditional_losses"
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
М
Llayer_metrics
Mnon_trainable_variables

Nstates
-regularization_losses
.	variables
Ometrics
/trainable_variables
Player_regularization_losses

Qlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
А
Vlayer_metrics
Wnon_trainable_variables
=regularization_losses
>	variables
Xmetrics
?trainable_variables
Ylayer_regularization_losses

Zlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
А
[layer_metrics
\non_trainable_variables
Hregularization_losses
I	variables
]metrics
Jtrainable_variables
^layer_regularization_losses

_layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
&:$d2Adam/dense_39/kernel/m
 :2Adam/dense_39/bias/m
M:K	Ш2<Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/m
W:U	2Ш2FAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/m
G:EШ2:Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/m
N:L	Ш2=Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/m
X:V	2Ш2GAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/m
H:FШ2;Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/m
&:$d2Adam/dense_39/kernel/v
 :2Adam/dense_39/bias/v
M:K	Ш2<Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/v
W:U	2Ш2FAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/v
G:EШ2:Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/v
N:L	Ш2=Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/v
X:V	2Ш2GAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/v
H:FШ2;Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/v
):'d2Adam/dense_39/kernel/vhat
#:!2Adam/dense_39/bias/vhat
P:N	Ш2?Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/kernel/vhat
Z:X	2Ш2IAdam/bidirectional_39/forward_lstm_39/lstm_cell_118/recurrent_kernel/vhat
J:HШ2=Adam/bidirectional_39/forward_lstm_39/lstm_cell_118/bias/vhat
Q:O	Ш2@Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/kernel/vhat
[:Y	2Ш2JAdam/bidirectional_39/backward_lstm_39/lstm_cell_119/recurrent_kernel/vhat
K:IШ2>Adam/bidirectional_39/backward_lstm_39/lstm_cell_119/bias/vhat
Ј2Ѕ
/__inference_sequential_39_layer_call_fn_4931761
/__inference_sequential_39_layer_call_fn_4932254Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
жBг
"__inference__wrapped_model_4929274args_0args_0_1"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
о2л
J__inference_sequential_39_layer_call_and_return_conditional_losses_4932277
J__inference_sequential_39_layer_call_and_return_conditional_losses_4932300Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
М2Й
2__inference_bidirectional_39_layer_call_fn_4932347
2__inference_bidirectional_39_layer_call_fn_4932364
2__inference_bidirectional_39_layer_call_fn_4932382
2__inference_bidirectional_39_layer_call_fn_4932400ц
нВй
FullArgSpecO
argsGD
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
defaults
p 

 

 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2Ѕ
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_4932702
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_4933004
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_4933362
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_4933720ц
нВй
FullArgSpecO
argsGD
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
defaults
p 

 

 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
д2б
*__inference_dense_39_layer_call_fn_4933729Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_39_layer_call_and_return_conditional_losses_4933740Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
гBа
%__inference_signature_wrapper_4932330args_0args_0_1"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ї2Є
1__inference_forward_lstm_39_layer_call_fn_4933751
1__inference_forward_lstm_39_layer_call_fn_4933762
1__inference_forward_lstm_39_layer_call_fn_4933773
1__inference_forward_lstm_39_layer_call_fn_4933784е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_4933935
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_4934086
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_4934237
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_4934388е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ћ2Ј
2__inference_backward_lstm_39_layer_call_fn_4934399
2__inference_backward_lstm_39_layer_call_fn_4934410
2__inference_backward_lstm_39_layer_call_fn_4934421
2__inference_backward_lstm_39_layer_call_fn_4934432е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_4934585
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_4934738
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_4934891
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_4935044е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
І2Ѓ
/__inference_lstm_cell_118_layer_call_fn_4935061
/__inference_lstm_cell_118_layer_call_fn_4935078О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
м2й
J__inference_lstm_cell_118_layer_call_and_return_conditional_losses_4935110
J__inference_lstm_cell_118_layer_call_and_return_conditional_losses_4935142О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
І2Ѓ
/__inference_lstm_cell_119_layer_call_fn_4935159
/__inference_lstm_cell_119_layer_call_fn_4935176О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
м2й
J__inference_lstm_cell_119_layer_call_and_return_conditional_losses_4935208
J__inference_lstm_cell_119_layer_call_and_return_conditional_losses_4935240О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 Ф
"__inference__wrapped_model_4929274\ЂY
RЂO
MJ4Ђ1
!њџџџџџџџџџџџџџџџџџџ

`
	RaggedTensorSpec
Њ "3Њ0
.
dense_39"
dense_39џџџџџџџџџЮ
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_4934585}OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ2
 Ю
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_4934738}OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџ2
 а
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_4934891QЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ2
 а
M__inference_backward_lstm_39_layer_call_and_return_conditional_losses_4935044QЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџ2
 І
2__inference_backward_lstm_39_layer_call_fn_4934399pOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2І
2__inference_backward_lstm_39_layer_call_fn_4934410pOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2Ј
2__inference_backward_lstm_39_layer_call_fn_4934421rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2Ј
2__inference_backward_lstm_39_layer_call_fn_4934432rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2п
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_4932702\ЂY
RЂO
=:
85
inputs/0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 

 

 
Њ "%Ђ"

0џџџџџџџџџd
 п
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_4933004\ЂY
RЂO
=:
85
inputs/0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 

 

 
Њ "%Ђ"

0џџџџџџџџџd
 я
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_4933362lЂi
bЂ_
MJ4Ђ1
!њџџџџџџџџџџџџџџџџџџ

`
	RaggedTensorSpec
p 

 

 

 
Њ "%Ђ"

0џџџџџџџџџd
 я
M__inference_bidirectional_39_layer_call_and_return_conditional_losses_4933720lЂi
bЂ_
MJ4Ђ1
!њџџџџџџџџџџџџџџџџџџ

`
	RaggedTensorSpec
p

 

 

 
Њ "%Ђ"

0џџџџџџџџџd
 З
2__inference_bidirectional_39_layer_call_fn_4932347\ЂY
RЂO
=:
85
inputs/0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 

 

 
Њ "џџџџџџџџџdЗ
2__inference_bidirectional_39_layer_call_fn_4932364\ЂY
RЂO
=:
85
inputs/0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 

 

 
Њ "џџџџџџџџџdЧ
2__inference_bidirectional_39_layer_call_fn_4932382lЂi
bЂ_
MJ4Ђ1
!њџџџџџџџџџџџџџџџџџџ

`
	RaggedTensorSpec
p 

 

 

 
Њ "џџџџџџџџџdЧ
2__inference_bidirectional_39_layer_call_fn_4932400lЂi
bЂ_
MJ4Ђ1
!њџџџџџџџџџџџџџџџџџџ

`
	RaggedTensorSpec
p

 

 

 
Њ "џџџџџџџџџdЅ
E__inference_dense_39_layer_call_and_return_conditional_losses_4933740\/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџ
 }
*__inference_dense_39_layer_call_fn_4933729O/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџЭ
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_4933935}OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ2
 Э
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_4934086}OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџ2
 Я
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_4934237QЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ2
 Я
L__inference_forward_lstm_39_layer_call_and_return_conditional_losses_4934388QЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџ2
 Ѕ
1__inference_forward_lstm_39_layer_call_fn_4933751pOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2Ѕ
1__inference_forward_lstm_39_layer_call_fn_4933762pOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2Ї
1__inference_forward_lstm_39_layer_call_fn_4933773rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2Ї
1__inference_forward_lstm_39_layer_call_fn_4933784rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2Ь
J__inference_lstm_cell_118_layer_call_and_return_conditional_losses_4935110§Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ2
"
states/1џџџџџџџџџ2
p 
Њ "sЂp
iЂf

0/0џџџџџџџџџ2
EB

0/1/0џџџџџџџџџ2

0/1/1џџџџџџџџџ2
 Ь
J__inference_lstm_cell_118_layer_call_and_return_conditional_losses_4935142§Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ2
"
states/1џџџџџџџџџ2
p
Њ "sЂp
iЂf

0/0џџџџџџџџџ2
EB

0/1/0џџџџџџџџџ2

0/1/1џџџџџџџџџ2
 Ё
/__inference_lstm_cell_118_layer_call_fn_4935061эЂ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ2
"
states/1џџџџџџџџџ2
p 
Њ "cЂ`

0џџџџџџџџџ2
A>

1/0џџџџџџџџџ2

1/1џџџџџџџџџ2Ё
/__inference_lstm_cell_118_layer_call_fn_4935078эЂ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ2
"
states/1џџџџџџџџџ2
p
Њ "cЂ`

0џџџџџџџџџ2
A>

1/0џџџџџџџџџ2

1/1џџџџџџџџџ2Ь
J__inference_lstm_cell_119_layer_call_and_return_conditional_losses_4935208§Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ2
"
states/1џџџџџџџџџ2
p 
Њ "sЂp
iЂf

0/0џџџџџџџџџ2
EB

0/1/0џџџџџџџџџ2

0/1/1џџџџџџџџџ2
 Ь
J__inference_lstm_cell_119_layer_call_and_return_conditional_losses_4935240§Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ2
"
states/1џџџџџџџџџ2
p
Њ "sЂp
iЂf

0/0џџџџџџџџџ2
EB

0/1/0џџџџџџџџџ2

0/1/1џџџџџџџџџ2
 Ё
/__inference_lstm_cell_119_layer_call_fn_4935159эЂ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ2
"
states/1џџџџџџџџџ2
p 
Њ "cЂ`

0џџџџџџџџџ2
A>

1/0џџџџџџџџџ2

1/1џџџџџџџџџ2Ё
/__inference_lstm_cell_119_layer_call_fn_4935176эЂ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ2
"
states/1џџџџџџџџџ2
p
Њ "cЂ`

0џџџџџџџџџ2
A>

1/0џџџџџџџџџ2

1/1џџџџџџџџџ2ц
J__inference_sequential_39_layer_call_and_return_conditional_losses_4932277dЂa
ZЂW
MJ4Ђ1
!њџџџџџџџџџџџџџџџџџџ

`
	RaggedTensorSpec
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 ц
J__inference_sequential_39_layer_call_and_return_conditional_losses_4932300dЂa
ZЂW
MJ4Ђ1
!њџџџџџџџџџџџџџџџџџџ

`
	RaggedTensorSpec
p

 
Њ "%Ђ"

0џџџџџџџџџ
 О
/__inference_sequential_39_layer_call_fn_4931761dЂa
ZЂW
MJ4Ђ1
!њџџџџџџџџџџџџџџџџџџ

`
	RaggedTensorSpec
p 

 
Њ "џџџџџџџџџО
/__inference_sequential_39_layer_call_fn_4932254dЂa
ZЂW
MJ4Ђ1
!њџџџџџџџџџџџџџџџџџџ

`
	RaggedTensorSpec
p

 
Њ "џџџџџџџџџа
%__inference_signature_wrapper_4932330ІeЂb
Ђ 
[ЊX
*
args_0 
args_0џџџџџџџџџ
*
args_0_1
args_0_1џџџџџџџџџ	"3Њ0
.
dense_39"
dense_39џџџџџџџџџ