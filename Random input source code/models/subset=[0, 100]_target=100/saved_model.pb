ЃЕ<
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
Ttype"serve*2.6.02v2.6.0-rc2-32-g919f693420e8з:
|
dense_133/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*!
shared_namedense_133/kernel
u
$dense_133/kernel/Read/ReadVariableOpReadVariableOpdense_133/kernel*
_output_shapes

:d*
dtype0
t
dense_133/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_133/bias
m
"dense_133/bias/Read/ReadVariableOpReadVariableOpdense_133/bias*
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
Ы
7bidirectional_133/forward_lstm_133/lstm_cell_400/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*H
shared_name97bidirectional_133/forward_lstm_133/lstm_cell_400/kernel
Ф
Kbidirectional_133/forward_lstm_133/lstm_cell_400/kernel/Read/ReadVariableOpReadVariableOp7bidirectional_133/forward_lstm_133/lstm_cell_400/kernel*
_output_shapes
:	Ш*
dtype0
п
Abidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*R
shared_nameCAbidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel
и
Ubidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/Read/ReadVariableOpReadVariableOpAbidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel*
_output_shapes
:	2Ш*
dtype0
У
5bidirectional_133/forward_lstm_133/lstm_cell_400/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*F
shared_name75bidirectional_133/forward_lstm_133/lstm_cell_400/bias
М
Ibidirectional_133/forward_lstm_133/lstm_cell_400/bias/Read/ReadVariableOpReadVariableOp5bidirectional_133/forward_lstm_133/lstm_cell_400/bias*
_output_shapes	
:Ш*
dtype0
Э
8bidirectional_133/backward_lstm_133/lstm_cell_401/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*I
shared_name:8bidirectional_133/backward_lstm_133/lstm_cell_401/kernel
Ц
Lbidirectional_133/backward_lstm_133/lstm_cell_401/kernel/Read/ReadVariableOpReadVariableOp8bidirectional_133/backward_lstm_133/lstm_cell_401/kernel*
_output_shapes
:	Ш*
dtype0
с
Bbidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*S
shared_nameDBbidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel
к
Vbidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/Read/ReadVariableOpReadVariableOpBbidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel*
_output_shapes
:	2Ш*
dtype0
Х
6bidirectional_133/backward_lstm_133/lstm_cell_401/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*G
shared_name86bidirectional_133/backward_lstm_133/lstm_cell_401/bias
О
Jbidirectional_133/backward_lstm_133/lstm_cell_401/bias/Read/ReadVariableOpReadVariableOp6bidirectional_133/backward_lstm_133/lstm_cell_401/bias*
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

Adam/dense_133/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_133/kernel/m

+Adam/dense_133/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_133/kernel/m*
_output_shapes

:d*
dtype0

Adam/dense_133/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_133/bias/m
{
)Adam/dense_133/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_133/bias/m*
_output_shapes
:*
dtype0
й
>Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*O
shared_name@>Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/m
в
RAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/m/Read/ReadVariableOpReadVariableOp>Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/m*
_output_shapes
:	Ш*
dtype0
э
HAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*Y
shared_nameJHAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/m
ц
\Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpHAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/m*
_output_shapes
:	2Ш*
dtype0
б
<Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*M
shared_name><Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/m
Ъ
PAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/m/Read/ReadVariableOpReadVariableOp<Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/m*
_output_shapes	
:Ш*
dtype0
л
?Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*P
shared_nameA?Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/m
д
SAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/m/Read/ReadVariableOpReadVariableOp?Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/m*
_output_shapes
:	Ш*
dtype0
я
IAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*Z
shared_nameKIAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/m
ш
]Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpIAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/m*
_output_shapes
:	2Ш*
dtype0
г
=Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*N
shared_name?=Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/m
Ь
QAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/m/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/m*
_output_shapes	
:Ш*
dtype0

Adam/dense_133/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_133/kernel/v

+Adam/dense_133/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_133/kernel/v*
_output_shapes

:d*
dtype0

Adam/dense_133/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_133/bias/v
{
)Adam/dense_133/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_133/bias/v*
_output_shapes
:*
dtype0
й
>Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*O
shared_name@>Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/v
в
RAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/v/Read/ReadVariableOpReadVariableOp>Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/v*
_output_shapes
:	Ш*
dtype0
э
HAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*Y
shared_nameJHAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/v
ц
\Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpHAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/v*
_output_shapes
:	2Ш*
dtype0
б
<Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*M
shared_name><Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/v
Ъ
PAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/v/Read/ReadVariableOpReadVariableOp<Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/v*
_output_shapes	
:Ш*
dtype0
л
?Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*P
shared_nameA?Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/v
д
SAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/v/Read/ReadVariableOpReadVariableOp?Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/v*
_output_shapes
:	Ш*
dtype0
я
IAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*Z
shared_nameKIAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/v
ш
]Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpIAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/v*
_output_shapes
:	2Ш*
dtype0
г
=Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*N
shared_name?=Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/v
Ь
QAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/v/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/v*
_output_shapes	
:Ш*
dtype0

Adam/dense_133/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*+
shared_nameAdam/dense_133/kernel/vhat

.Adam/dense_133/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_133/kernel/vhat*
_output_shapes

:d*
dtype0

Adam/dense_133/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/dense_133/bias/vhat

,Adam/dense_133/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_133/bias/vhat*
_output_shapes
:*
dtype0
п
AAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*R
shared_nameCAAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/vhat
и
UAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/vhat/Read/ReadVariableOpReadVariableOpAAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/vhat*
_output_shapes
:	Ш*
dtype0
ѓ
KAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*\
shared_nameMKAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/vhat
ь
_Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpKAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/vhat*
_output_shapes
:	2Ш*
dtype0
з
?Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*P
shared_nameA?Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/vhat
а
SAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/vhat/Read/ReadVariableOpReadVariableOp?Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/vhat*
_output_shapes	
:Ш*
dtype0
с
BAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*S
shared_nameDBAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/vhat
к
VAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/vhat/Read/ReadVariableOpReadVariableOpBAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/vhat*
_output_shapes
:	Ш*
dtype0
ѕ
LAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*]
shared_nameNLAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/vhat
ю
`Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpLAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/vhat*
_output_shapes
:	2Ш*
dtype0
й
@Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*Q
shared_nameB@Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/vhat
в
TAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/vhat/Read/ReadVariableOpReadVariableOp@Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/vhat*
_output_shapes	
:Ш*
dtype0

NoOpNoOp
A
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*е@
valueЫ@BШ@ BС@
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
\Z
VARIABLE_VALUEdense_133/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_133/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
sq
VARIABLE_VALUE7bidirectional_133/forward_lstm_133/lstm_cell_400/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAbidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5bidirectional_133/forward_lstm_133/lstm_cell_400/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE8bidirectional_133/backward_lstm_133/lstm_cell_401/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEBbidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6bidirectional_133/backward_lstm_133/lstm_cell_401/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
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
}
VARIABLE_VALUEAdam/dense_133/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_133/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ё
VARIABLE_VALUEHAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ђ
VARIABLE_VALUEIAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_133/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_133/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ё
VARIABLE_VALUEHAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ђ
VARIABLE_VALUEIAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_133/kernel/vhatUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_133/bias/vhatSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/vhatEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ЇЄ
VARIABLE_VALUEKAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/vhatEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/vhatEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEBAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/vhatEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ЈЅ
VARIABLE_VALUELAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/vhatEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE@Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/vhatEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
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
щ
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_0_17bidirectional_133/forward_lstm_133/lstm_cell_400/kernelAbidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel5bidirectional_133/forward_lstm_133/lstm_cell_400/bias8bidirectional_133/backward_lstm_133/lstm_cell_401/kernelBbidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel6bidirectional_133/backward_lstm_133/lstm_cell_401/biasdense_133/kerneldense_133/bias*
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
GPU 2J 8 */
f*R(
&__inference_signature_wrapper_16887324
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_133/kernel/Read/ReadVariableOp"dense_133/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpKbidirectional_133/forward_lstm_133/lstm_cell_400/kernel/Read/ReadVariableOpUbidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/Read/ReadVariableOpIbidirectional_133/forward_lstm_133/lstm_cell_400/bias/Read/ReadVariableOpLbidirectional_133/backward_lstm_133/lstm_cell_401/kernel/Read/ReadVariableOpVbidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/Read/ReadVariableOpJbidirectional_133/backward_lstm_133/lstm_cell_401/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_133/kernel/m/Read/ReadVariableOp)Adam/dense_133/bias/m/Read/ReadVariableOpRAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/m/Read/ReadVariableOp\Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/m/Read/ReadVariableOpPAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/m/Read/ReadVariableOpSAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/m/Read/ReadVariableOp]Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/m/Read/ReadVariableOpQAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/m/Read/ReadVariableOp+Adam/dense_133/kernel/v/Read/ReadVariableOp)Adam/dense_133/bias/v/Read/ReadVariableOpRAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/v/Read/ReadVariableOp\Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/v/Read/ReadVariableOpPAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/v/Read/ReadVariableOpSAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/v/Read/ReadVariableOp]Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/v/Read/ReadVariableOpQAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/v/Read/ReadVariableOp.Adam/dense_133/kernel/vhat/Read/ReadVariableOp,Adam/dense_133/bias/vhat/Read/ReadVariableOpUAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/vhat/Read/ReadVariableOp_Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/vhat/Read/ReadVariableOpSAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/vhat/Read/ReadVariableOpVAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/vhat/Read/ReadVariableOp`Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/vhat/Read/ReadVariableOpTAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/vhat/Read/ReadVariableOpConst*4
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
GPU 2J 8 **
f%R#
!__inference__traced_save_16890375
ў
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_133/kerneldense_133/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate7bidirectional_133/forward_lstm_133/lstm_cell_400/kernelAbidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel5bidirectional_133/forward_lstm_133/lstm_cell_400/bias8bidirectional_133/backward_lstm_133/lstm_cell_401/kernelBbidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel6bidirectional_133/backward_lstm_133/lstm_cell_401/biastotalcountAdam/dense_133/kernel/mAdam/dense_133/bias/m>Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/mHAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/m<Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/m?Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/mIAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/m=Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/mAdam/dense_133/kernel/vAdam/dense_133/bias/v>Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/vHAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/v<Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/v?Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/vIAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/v=Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/vAdam/dense_133/kernel/vhatAdam/dense_133/bias/vhatAAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/vhatKAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/vhat?Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/vhatBAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/vhatLAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/vhat@Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/vhat*3
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
GPU 2J 8 *-
f(R&
$__inference__traced_restore_16890502Њї8
п
Э
while_cond_16885768
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16885768___redundant_placeholder06
2while_while_cond_16885768___redundant_placeholder16
2while_while_cond_16885768___redundant_placeholder26
2while_while_cond_16885768___redundant_placeholder3
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
п
Э
while_cond_16889800
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16889800___redundant_placeholder06
2while_while_cond_16889800___redundant_placeholder16
2while_while_cond_16889800___redundant_placeholder26
2while_while_cond_16889800___redundant_placeholder3
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
п
Э
while_cond_16886133
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16886133___redundant_placeholder06
2while_while_cond_16886133___redundant_placeholder16
2while_while_cond_16886133___redundant_placeholder26
2while_while_cond_16886133___redundant_placeholder3
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
Н
я
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_16886704

inputs
inputs_1	P
=forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource:	ШR
?forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource:	2ШM
>forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource:	ШQ
>backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource:	ШS
@backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource:	2ШN
?backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource:	Ш
identityЂ6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpЂ5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpЂ7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpЂbackward_lstm_133/whileЂ5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpЂ4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpЂ6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOpЂforward_lstm_133/while
%forward_lstm_133/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_133/RaggedToTensor/zeros
%forward_lstm_133/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2'
%forward_lstm_133/RaggedToTensor/Const
4forward_lstm_133/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_133/RaggedToTensor/Const:output:0inputs.forward_lstm_133/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_133/RaggedToTensor/RaggedTensorToTensorФ
;forward_lstm_133/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_133/RaggedNestedRowLengths/strided_slice/stackШ
=forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1Ш
=forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2Љ
5forward_lstm_133/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_133/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask27
5forward_lstm_133/RaggedNestedRowLengths/strided_sliceШ
=forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stackе
?forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2A
?forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1Ь
?forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2Е
7forward_lstm_133/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask29
7forward_lstm_133/RaggedNestedRowLengths/strided_slice_1
+forward_lstm_133/RaggedNestedRowLengths/subSub>forward_lstm_133/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_133/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2-
+forward_lstm_133/RaggedNestedRowLengths/subЄ
forward_lstm_133/CastCast/forward_lstm_133/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_133/Cast
forward_lstm_133/ShapeShape=forward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_133/Shape
$forward_lstm_133/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_133/strided_slice/stack
&forward_lstm_133/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_133/strided_slice/stack_1
&forward_lstm_133/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_133/strided_slice/stack_2Ш
forward_lstm_133/strided_sliceStridedSliceforward_lstm_133/Shape:output:0-forward_lstm_133/strided_slice/stack:output:0/forward_lstm_133/strided_slice/stack_1:output:0/forward_lstm_133/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_133/strided_slice~
forward_lstm_133/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_133/zeros/mul/yА
forward_lstm_133/zeros/mulMul'forward_lstm_133/strided_slice:output:0%forward_lstm_133/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros/mul
forward_lstm_133/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_133/zeros/Less/yЋ
forward_lstm_133/zeros/LessLessforward_lstm_133/zeros/mul:z:0&forward_lstm_133/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros/Less
forward_lstm_133/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_133/zeros/packed/1Ч
forward_lstm_133/zeros/packedPack'forward_lstm_133/strided_slice:output:0(forward_lstm_133/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_133/zeros/packed
forward_lstm_133/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_133/zeros/ConstЙ
forward_lstm_133/zerosFill&forward_lstm_133/zeros/packed:output:0%forward_lstm_133/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/zeros
forward_lstm_133/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_133/zeros_1/mul/yЖ
forward_lstm_133/zeros_1/mulMul'forward_lstm_133/strided_slice:output:0'forward_lstm_133/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros_1/mul
forward_lstm_133/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
forward_lstm_133/zeros_1/Less/yГ
forward_lstm_133/zeros_1/LessLess forward_lstm_133/zeros_1/mul:z:0(forward_lstm_133/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros_1/Less
!forward_lstm_133/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_133/zeros_1/packed/1Э
forward_lstm_133/zeros_1/packedPack'forward_lstm_133/strided_slice:output:0*forward_lstm_133/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_133/zeros_1/packed
forward_lstm_133/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_133/zeros_1/ConstС
forward_lstm_133/zeros_1Fill(forward_lstm_133/zeros_1/packed:output:0'forward_lstm_133/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/zeros_1
forward_lstm_133/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_133/transpose/permэ
forward_lstm_133/transpose	Transpose=forward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_133/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_133/transpose
forward_lstm_133/Shape_1Shapeforward_lstm_133/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_133/Shape_1
&forward_lstm_133/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_133/strided_slice_1/stack
(forward_lstm_133/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_1/stack_1
(forward_lstm_133/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_1/stack_2д
 forward_lstm_133/strided_slice_1StridedSlice!forward_lstm_133/Shape_1:output:0/forward_lstm_133/strided_slice_1/stack:output:01forward_lstm_133/strided_slice_1/stack_1:output:01forward_lstm_133/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_133/strided_slice_1Ї
,forward_lstm_133/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,forward_lstm_133/TensorArrayV2/element_shapeі
forward_lstm_133/TensorArrayV2TensorListReserve5forward_lstm_133/TensorArrayV2/element_shape:output:0)forward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_133/TensorArrayV2с
Fforward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fforward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8forward_lstm_133/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_133/transpose:y:0Oforward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_133/TensorArrayUnstack/TensorListFromTensor
&forward_lstm_133/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_133/strided_slice_2/stack
(forward_lstm_133/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_2/stack_1
(forward_lstm_133/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_2/stack_2т
 forward_lstm_133/strided_slice_2StridedSliceforward_lstm_133/transpose:y:0/forward_lstm_133/strided_slice_2/stack:output:01forward_lstm_133/strided_slice_2/stack_1:output:01forward_lstm_133/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 forward_lstm_133/strided_slice_2ы
4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpReadVariableOp=forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpє
%forward_lstm_133/lstm_cell_400/MatMulMatMul)forward_lstm_133/strided_slice_2:output:0<forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_133/lstm_cell_400/MatMulё
6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOp?forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp№
'forward_lstm_133/lstm_cell_400/MatMul_1MatMulforward_lstm_133/zeros:output:0>forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_133/lstm_cell_400/MatMul_1ш
"forward_lstm_133/lstm_cell_400/addAddV2/forward_lstm_133/lstm_cell_400/MatMul:product:01forward_lstm_133/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_133/lstm_cell_400/addъ
5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOp>forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpѕ
&forward_lstm_133/lstm_cell_400/BiasAddBiasAdd&forward_lstm_133/lstm_cell_400/add:z:0=forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_133/lstm_cell_400/BiasAddЂ
.forward_lstm_133/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.forward_lstm_133/lstm_cell_400/split/split_dimЛ
$forward_lstm_133/lstm_cell_400/splitSplit7forward_lstm_133/lstm_cell_400/split/split_dim:output:0/forward_lstm_133/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$forward_lstm_133/lstm_cell_400/splitМ
&forward_lstm_133/lstm_cell_400/SigmoidSigmoid-forward_lstm_133/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_133/lstm_cell_400/SigmoidР
(forward_lstm_133/lstm_cell_400/Sigmoid_1Sigmoid-forward_lstm_133/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_133/lstm_cell_400/Sigmoid_1в
"forward_lstm_133/lstm_cell_400/mulMul,forward_lstm_133/lstm_cell_400/Sigmoid_1:y:0!forward_lstm_133/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_133/lstm_cell_400/mulГ
#forward_lstm_133/lstm_cell_400/ReluRelu-forward_lstm_133/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_133/lstm_cell_400/Reluф
$forward_lstm_133/lstm_cell_400/mul_1Mul*forward_lstm_133/lstm_cell_400/Sigmoid:y:01forward_lstm_133/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_133/lstm_cell_400/mul_1й
$forward_lstm_133/lstm_cell_400/add_1AddV2&forward_lstm_133/lstm_cell_400/mul:z:0(forward_lstm_133/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_133/lstm_cell_400/add_1Р
(forward_lstm_133/lstm_cell_400/Sigmoid_2Sigmoid-forward_lstm_133/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_133/lstm_cell_400/Sigmoid_2В
%forward_lstm_133/lstm_cell_400/Relu_1Relu(forward_lstm_133/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_133/lstm_cell_400/Relu_1ш
$forward_lstm_133/lstm_cell_400/mul_2Mul,forward_lstm_133/lstm_cell_400/Sigmoid_2:y:03forward_lstm_133/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_133/lstm_cell_400/mul_2Б
.forward_lstm_133/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.forward_lstm_133/TensorArrayV2_1/element_shapeќ
 forward_lstm_133/TensorArrayV2_1TensorListReserve7forward_lstm_133/TensorArrayV2_1/element_shape:output:0)forward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_133/TensorArrayV2_1p
forward_lstm_133/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_133/timeЃ
forward_lstm_133/zeros_like	ZerosLike(forward_lstm_133/lstm_cell_400/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/zeros_likeЁ
)forward_lstm_133/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)forward_lstm_133/while/maximum_iterations
#forward_lstm_133/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_133/while/loop_counter	
forward_lstm_133/whileWhile,forward_lstm_133/while/loop_counter:output:02forward_lstm_133/while/maximum_iterations:output:0forward_lstm_133/time:output:0)forward_lstm_133/TensorArrayV2_1:handle:0forward_lstm_133/zeros_like:y:0forward_lstm_133/zeros:output:0!forward_lstm_133/zeros_1:output:0)forward_lstm_133/strided_slice_1:output:0Hforward_lstm_133/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_133/Cast:y:0=forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource?forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource>forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *0
body(R&
$forward_lstm_133_while_body_16886428*0
cond(R&
$forward_lstm_133_while_cond_16886427*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_133/whileз
Aforward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Aforward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3forward_lstm_133/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_133/while:output:3Jforward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3forward_lstm_133/TensorArrayV2Stack/TensorListStackЃ
&forward_lstm_133/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&forward_lstm_133/strided_slice_3/stack
(forward_lstm_133/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_133/strided_slice_3/stack_1
(forward_lstm_133/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_3/stack_2
 forward_lstm_133/strided_slice_3StridedSlice<forward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_133/strided_slice_3/stack:output:01forward_lstm_133/strided_slice_3/stack_1:output:01forward_lstm_133/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 forward_lstm_133/strided_slice_3
!forward_lstm_133/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_133/transpose_1/permђ
forward_lstm_133/transpose_1	Transpose<forward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_133/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_133/transpose_1
forward_lstm_133/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_133/runtime
&backward_lstm_133/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_133/RaggedToTensor/zeros
&backward_lstm_133/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2(
&backward_lstm_133/RaggedToTensor/Const
5backward_lstm_133/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_133/RaggedToTensor/Const:output:0inputs/backward_lstm_133/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_133/RaggedToTensor/RaggedTensorToTensorЦ
<backward_lstm_133/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_133/RaggedNestedRowLengths/strided_slice/stackЪ
>backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1Ъ
>backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2Ў
6backward_lstm_133/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_133/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask28
6backward_lstm_133/RaggedNestedRowLengths/strided_sliceЪ
>backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stackз
@backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2B
@backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1Ю
@backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2К
8backward_lstm_133/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2:
8backward_lstm_133/RaggedNestedRowLengths/strided_slice_1
,backward_lstm_133/RaggedNestedRowLengths/subSub?backward_lstm_133/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_133/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2.
,backward_lstm_133/RaggedNestedRowLengths/subЇ
backward_lstm_133/CastCast0backward_lstm_133/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_133/Cast 
backward_lstm_133/ShapeShape>backward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_133/Shape
%backward_lstm_133/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_133/strided_slice/stack
'backward_lstm_133/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_133/strided_slice/stack_1
'backward_lstm_133/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_133/strided_slice/stack_2Ю
backward_lstm_133/strided_sliceStridedSlice backward_lstm_133/Shape:output:0.backward_lstm_133/strided_slice/stack:output:00backward_lstm_133/strided_slice/stack_1:output:00backward_lstm_133/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_133/strided_slice
backward_lstm_133/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_133/zeros/mul/yД
backward_lstm_133/zeros/mulMul(backward_lstm_133/strided_slice:output:0&backward_lstm_133/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/zeros/mul
backward_lstm_133/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_133/zeros/Less/yЏ
backward_lstm_133/zeros/LessLessbackward_lstm_133/zeros/mul:z:0'backward_lstm_133/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/zeros/Less
 backward_lstm_133/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_133/zeros/packed/1Ы
backward_lstm_133/zeros/packedPack(backward_lstm_133/strided_slice:output:0)backward_lstm_133/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_133/zeros/packed
backward_lstm_133/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_133/zeros/ConstН
backward_lstm_133/zerosFill'backward_lstm_133/zeros/packed:output:0&backward_lstm_133/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_133/zeros
backward_lstm_133/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_133/zeros_1/mul/yК
backward_lstm_133/zeros_1/mulMul(backward_lstm_133/strided_slice:output:0(backward_lstm_133/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/zeros_1/mul
 backward_lstm_133/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 backward_lstm_133/zeros_1/Less/yЗ
backward_lstm_133/zeros_1/LessLess!backward_lstm_133/zeros_1/mul:z:0)backward_lstm_133/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_133/zeros_1/Less
"backward_lstm_133/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_133/zeros_1/packed/1б
 backward_lstm_133/zeros_1/packedPack(backward_lstm_133/strided_slice:output:0+backward_lstm_133/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_133/zeros_1/packed
backward_lstm_133/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_133/zeros_1/ConstХ
backward_lstm_133/zeros_1Fill)backward_lstm_133/zeros_1/packed:output:0(backward_lstm_133/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_133/zeros_1
 backward_lstm_133/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_133/transpose/permё
backward_lstm_133/transpose	Transpose>backward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_133/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_133/transpose
backward_lstm_133/Shape_1Shapebackward_lstm_133/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_133/Shape_1
'backward_lstm_133/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_133/strided_slice_1/stack 
)backward_lstm_133/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_1/stack_1 
)backward_lstm_133/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_1/stack_2к
!backward_lstm_133/strided_slice_1StridedSlice"backward_lstm_133/Shape_1:output:00backward_lstm_133/strided_slice_1/stack:output:02backward_lstm_133/strided_slice_1/stack_1:output:02backward_lstm_133/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_133/strided_slice_1Љ
-backward_lstm_133/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2/
-backward_lstm_133/TensorArrayV2/element_shapeњ
backward_lstm_133/TensorArrayV2TensorListReserve6backward_lstm_133/TensorArrayV2/element_shape:output:0*backward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_133/TensorArrayV2
 backward_lstm_133/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_133/ReverseV2/axisв
backward_lstm_133/ReverseV2	ReverseV2backward_lstm_133/transpose:y:0)backward_lstm_133/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_133/ReverseV2у
Gbackward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gbackward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeХ
9backward_lstm_133/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_133/ReverseV2:output:0Pbackward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_133/TensorArrayUnstack/TensorListFromTensor
'backward_lstm_133/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_133/strided_slice_2/stack 
)backward_lstm_133/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_2/stack_1 
)backward_lstm_133/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_2/stack_2ш
!backward_lstm_133/strided_slice_2StridedSlicebackward_lstm_133/transpose:y:00backward_lstm_133/strided_slice_2/stack:output:02backward_lstm_133/strided_slice_2/stack_1:output:02backward_lstm_133/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2#
!backward_lstm_133/strided_slice_2ю
5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpReadVariableOp>backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype027
5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpј
&backward_lstm_133/lstm_cell_401/MatMulMatMul*backward_lstm_133/strided_slice_2:output:0=backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_133/lstm_cell_401/MatMulє
7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOp@backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype029
7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpє
(backward_lstm_133/lstm_cell_401/MatMul_1MatMul backward_lstm_133/zeros:output:0?backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_133/lstm_cell_401/MatMul_1ь
#backward_lstm_133/lstm_cell_401/addAddV20backward_lstm_133/lstm_cell_401/MatMul:product:02backward_lstm_133/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_133/lstm_cell_401/addэ
6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOp?backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype028
6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpљ
'backward_lstm_133/lstm_cell_401/BiasAddBiasAdd'backward_lstm_133/lstm_cell_401/add:z:0>backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_133/lstm_cell_401/BiasAddЄ
/backward_lstm_133/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/backward_lstm_133/lstm_cell_401/split/split_dimП
%backward_lstm_133/lstm_cell_401/splitSplit8backward_lstm_133/lstm_cell_401/split/split_dim:output:00backward_lstm_133/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2'
%backward_lstm_133/lstm_cell_401/splitП
'backward_lstm_133/lstm_cell_401/SigmoidSigmoid.backward_lstm_133/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_133/lstm_cell_401/SigmoidУ
)backward_lstm_133/lstm_cell_401/Sigmoid_1Sigmoid.backward_lstm_133/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_133/lstm_cell_401/Sigmoid_1ж
#backward_lstm_133/lstm_cell_401/mulMul-backward_lstm_133/lstm_cell_401/Sigmoid_1:y:0"backward_lstm_133/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_133/lstm_cell_401/mulЖ
$backward_lstm_133/lstm_cell_401/ReluRelu.backward_lstm_133/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_133/lstm_cell_401/Reluш
%backward_lstm_133/lstm_cell_401/mul_1Mul+backward_lstm_133/lstm_cell_401/Sigmoid:y:02backward_lstm_133/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_133/lstm_cell_401/mul_1н
%backward_lstm_133/lstm_cell_401/add_1AddV2'backward_lstm_133/lstm_cell_401/mul:z:0)backward_lstm_133/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_133/lstm_cell_401/add_1У
)backward_lstm_133/lstm_cell_401/Sigmoid_2Sigmoid.backward_lstm_133/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_133/lstm_cell_401/Sigmoid_2Е
&backward_lstm_133/lstm_cell_401/Relu_1Relu)backward_lstm_133/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_133/lstm_cell_401/Relu_1ь
%backward_lstm_133/lstm_cell_401/mul_2Mul-backward_lstm_133/lstm_cell_401/Sigmoid_2:y:04backward_lstm_133/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_133/lstm_cell_401/mul_2Г
/backward_lstm_133/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   21
/backward_lstm_133/TensorArrayV2_1/element_shape
!backward_lstm_133/TensorArrayV2_1TensorListReserve8backward_lstm_133/TensorArrayV2_1/element_shape:output:0*backward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_133/TensorArrayV2_1r
backward_lstm_133/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_133/time
'backward_lstm_133/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_133/Max/reduction_indicesЄ
backward_lstm_133/MaxMaxbackward_lstm_133/Cast:y:00backward_lstm_133/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/Maxt
backward_lstm_133/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_133/sub/y
backward_lstm_133/subSubbackward_lstm_133/Max:output:0 backward_lstm_133/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/sub
backward_lstm_133/Sub_1Subbackward_lstm_133/sub:z:0backward_lstm_133/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_133/Sub_1І
backward_lstm_133/zeros_like	ZerosLike)backward_lstm_133/lstm_cell_401/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_133/zeros_likeЃ
*backward_lstm_133/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*backward_lstm_133/while/maximum_iterations
$backward_lstm_133/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_133/while/loop_counterЅ	
backward_lstm_133/whileWhile-backward_lstm_133/while/loop_counter:output:03backward_lstm_133/while/maximum_iterations:output:0backward_lstm_133/time:output:0*backward_lstm_133/TensorArrayV2_1:handle:0 backward_lstm_133/zeros_like:y:0 backward_lstm_133/zeros:output:0"backward_lstm_133/zeros_1:output:0*backward_lstm_133/strided_slice_1:output:0Ibackward_lstm_133/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_133/Sub_1:z:0>backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource@backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource?backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *1
body)R'
%backward_lstm_133_while_body_16886607*1
cond)R'
%backward_lstm_133_while_cond_16886606*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_133/whileй
Bbackward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2D
Bbackward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeЙ
4backward_lstm_133/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_133/while:output:3Kbackward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype026
4backward_lstm_133/TensorArrayV2Stack/TensorListStackЅ
'backward_lstm_133/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2)
'backward_lstm_133/strided_slice_3/stack 
)backward_lstm_133/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_133/strided_slice_3/stack_1 
)backward_lstm_133/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_3/stack_2
!backward_lstm_133/strided_slice_3StridedSlice=backward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_133/strided_slice_3/stack:output:02backward_lstm_133/strided_slice_3/stack_1:output:02backward_lstm_133/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2#
!backward_lstm_133/strided_slice_3
"backward_lstm_133/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_133/transpose_1/permі
backward_lstm_133/transpose_1	Transpose=backward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_133/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_133/transpose_1
backward_lstm_133/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_133/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisФ
concatConcatV2)forward_lstm_133/strided_slice_3:output:0*backward_lstm_133/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

Identityд
NoOpNoOp7^backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp6^backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp8^backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp^backward_lstm_133/while6^forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp5^forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp7^forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp^forward_lstm_133/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2p
6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp2n
5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp2r
7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp22
backward_lstm_133/whilebackward_lstm_133/while2n
5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp2l
4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp2p
6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp20
forward_lstm_133/whileforward_lstm_133/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
љ?
л
while_body_16885769
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_401_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_401_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_401_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_401_matmul_readvariableop_resource:	ШG
4while_lstm_cell_401_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_401_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_401/BiasAdd/ReadVariableOpЂ)while/lstm_cell_401/MatMul/ReadVariableOpЂ+while/lstm_cell_401/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_401/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_401_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_401/MatMul/ReadVariableOpк
while/lstm_cell_401/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/MatMulв
+while/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_401_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_401/MatMul_1/ReadVariableOpУ
while/lstm_cell_401/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/MatMul_1М
while/lstm_cell_401/addAddV2$while/lstm_cell_401/MatMul:product:0&while/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/addЫ
*while/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_401_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_401/BiasAdd/ReadVariableOpЩ
while/lstm_cell_401/BiasAddBiasAddwhile/lstm_cell_401/add:z:02while/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/BiasAdd
#while/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_401/split/split_dim
while/lstm_cell_401/splitSplit,while/lstm_cell_401/split/split_dim:output:0$while/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_401/split
while/lstm_cell_401/SigmoidSigmoid"while/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Sigmoid
while/lstm_cell_401/Sigmoid_1Sigmoid"while/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Sigmoid_1Ѓ
while/lstm_cell_401/mulMul!while/lstm_cell_401/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/mul
while/lstm_cell_401/ReluRelu"while/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/ReluИ
while/lstm_cell_401/mul_1Mulwhile/lstm_cell_401/Sigmoid:y:0&while/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/mul_1­
while/lstm_cell_401/add_1AddV2while/lstm_cell_401/mul:z:0while/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/add_1
while/lstm_cell_401/Sigmoid_2Sigmoid"while/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Sigmoid_2
while/lstm_cell_401/Relu_1Reluwhile/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Relu_1М
while/lstm_cell_401/mul_2Mul!while/lstm_cell_401/Sigmoid_2:y:0(while/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_401/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_401/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_401/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_401/BiasAdd/ReadVariableOp*^while/lstm_cell_401/MatMul/ReadVariableOp,^while/lstm_cell_401/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_401_biasadd_readvariableop_resource5while_lstm_cell_401_biasadd_readvariableop_resource_0"n
4while_lstm_cell_401_matmul_1_readvariableop_resource6while_lstm_cell_401_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_401_matmul_readvariableop_resource4while_lstm_cell_401_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_401/BiasAdd/ReadVariableOp*while/lstm_cell_401/BiasAdd/ReadVariableOp2V
)while/lstm_cell_401/MatMul/ReadVariableOp)while/lstm_cell_401/MatMul/ReadVariableOp2Z
+while/lstm_cell_401/MatMul_1/ReadVariableOp+while/lstm_cell_401/MatMul_1/ReadVariableOp: 
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
_
Џ
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_16889579
inputs_0?
,lstm_cell_401_matmul_readvariableop_resource:	ШA
.lstm_cell_401_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_401_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_401/BiasAdd/ReadVariableOpЂ#lstm_cell_401/MatMul/ReadVariableOpЂ%lstm_cell_401/MatMul_1/ReadVariableOpЂwhileF
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
#lstm_cell_401/MatMul/ReadVariableOpReadVariableOp,lstm_cell_401_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_401/MatMul/ReadVariableOpА
lstm_cell_401/MatMulMatMulstrided_slice_2:output:0+lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/MatMulО
%lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_401_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_401/MatMul_1/ReadVariableOpЌ
lstm_cell_401/MatMul_1MatMulzeros:output:0-lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/MatMul_1Є
lstm_cell_401/addAddV2lstm_cell_401/MatMul:product:0 lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/addЗ
$lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_401_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_401/BiasAdd/ReadVariableOpБ
lstm_cell_401/BiasAddBiasAddlstm_cell_401/add:z:0,lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/BiasAdd
lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_401/split/split_dimї
lstm_cell_401/splitSplit&lstm_cell_401/split/split_dim:output:0lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_401/split
lstm_cell_401/SigmoidSigmoidlstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Sigmoid
lstm_cell_401/Sigmoid_1Sigmoidlstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Sigmoid_1
lstm_cell_401/mulMullstm_cell_401/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/mul
lstm_cell_401/ReluRelulstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Relu 
lstm_cell_401/mul_1Mullstm_cell_401/Sigmoid:y:0 lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/mul_1
lstm_cell_401/add_1AddV2lstm_cell_401/mul:z:0lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/add_1
lstm_cell_401/Sigmoid_2Sigmoidlstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Sigmoid_2
lstm_cell_401/Relu_1Relulstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Relu_1Є
lstm_cell_401/mul_2Mullstm_cell_401/Sigmoid_2:y:0"lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_401_matmul_readvariableop_resource.lstm_cell_401_matmul_1_readvariableop_resource-lstm_cell_401_biasadd_readvariableop_resource*
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
bodyR
while_body_16889495*
condR
while_cond_16889494*K
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
NoOpNoOp%^lstm_cell_401/BiasAdd/ReadVariableOp$^lstm_cell_401/MatMul/ReadVariableOp&^lstm_cell_401/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_401/BiasAdd/ReadVariableOp$lstm_cell_401/BiasAdd/ReadVariableOp2J
#lstm_cell_401/MatMul/ReadVariableOp#lstm_cell_401/MatMul/ReadVariableOp2N
%lstm_cell_401/MatMul_1/ReadVariableOp%lstm_cell_401/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
п
Э
while_cond_16885608
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16885608___redundant_placeholder06
2while_while_cond_16885608___redundant_placeholder16
2while_while_cond_16885608___redundant_placeholder26
2while_while_cond_16885608___redundant_placeholder3
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
еH

O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_16885270

inputs)
lstm_cell_401_16885188:	Ш)
lstm_cell_401_16885190:	2Ш%
lstm_cell_401_16885192:	Ш
identityЂ%lstm_cell_401/StatefulPartitionedCallЂwhileD
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
strided_slice_2Ћ
%lstm_cell_401/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_401_16885188lstm_cell_401_16885190lstm_cell_401_16885192*
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
GPU 2J 8 *T
fORM
K__inference_lstm_cell_401_layer_call_and_return_conditional_losses_168851212'
%lstm_cell_401/StatefulPartitionedCall
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
while/loop_counterЭ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_401_16885188lstm_cell_401_16885190lstm_cell_401_16885192*
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
bodyR
while_body_16885201*
condR
while_cond_16885200*K
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
NoOpNoOp&^lstm_cell_401/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2N
%lstm_cell_401/StatefulPartitionedCall%lstm_cell_401/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
]
Ќ
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_16886218

inputs?
,lstm_cell_400_matmul_readvariableop_resource:	ШA
.lstm_cell_400_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_400_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_400/BiasAdd/ReadVariableOpЂ#lstm_cell_400/MatMul/ReadVariableOpЂ%lstm_cell_400/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_400/MatMul/ReadVariableOpReadVariableOp,lstm_cell_400_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_400/MatMul/ReadVariableOpА
lstm_cell_400/MatMulMatMulstrided_slice_2:output:0+lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/MatMulО
%lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_400_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_400/MatMul_1/ReadVariableOpЌ
lstm_cell_400/MatMul_1MatMulzeros:output:0-lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/MatMul_1Є
lstm_cell_400/addAddV2lstm_cell_400/MatMul:product:0 lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/addЗ
$lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_400_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_400/BiasAdd/ReadVariableOpБ
lstm_cell_400/BiasAddBiasAddlstm_cell_400/add:z:0,lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/BiasAdd
lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_400/split/split_dimї
lstm_cell_400/splitSplit&lstm_cell_400/split/split_dim:output:0lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_400/split
lstm_cell_400/SigmoidSigmoidlstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Sigmoid
lstm_cell_400/Sigmoid_1Sigmoidlstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Sigmoid_1
lstm_cell_400/mulMullstm_cell_400/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/mul
lstm_cell_400/ReluRelulstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Relu 
lstm_cell_400/mul_1Mullstm_cell_400/Sigmoid:y:0 lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/mul_1
lstm_cell_400/add_1AddV2lstm_cell_400/mul:z:0lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/add_1
lstm_cell_400/Sigmoid_2Sigmoidlstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Sigmoid_2
lstm_cell_400/Relu_1Relulstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Relu_1Є
lstm_cell_400/mul_2Mullstm_cell_400/Sigmoid_2:y:0"lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_400_matmul_readvariableop_resource.lstm_cell_400_matmul_1_readvariableop_resource-lstm_cell_400_biasadd_readvariableop_resource*
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
bodyR
while_body_16886134*
condR
while_cond_16886133*K
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
NoOpNoOp%^lstm_cell_400/BiasAdd/ReadVariableOp$^lstm_cell_400/MatMul/ReadVariableOp&^lstm_cell_400/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_400/BiasAdd/ReadVariableOp$lstm_cell_400/BiasAdd/ReadVariableOp2J
#lstm_cell_400/MatMul/ReadVariableOp#lstm_cell_400/MatMul/ReadVariableOp2N
%lstm_cell_400/MatMul_1/ReadVariableOp%lstm_cell_400/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
§
Е
%backward_lstm_133_while_cond_16887609@
<backward_lstm_133_while_backward_lstm_133_while_loop_counterF
Bbackward_lstm_133_while_backward_lstm_133_while_maximum_iterations'
#backward_lstm_133_while_placeholder)
%backward_lstm_133_while_placeholder_1)
%backward_lstm_133_while_placeholder_2)
%backward_lstm_133_while_placeholder_3B
>backward_lstm_133_while_less_backward_lstm_133_strided_slice_1Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16887609___redundant_placeholder0Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16887609___redundant_placeholder1Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16887609___redundant_placeholder2Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16887609___redundant_placeholder3$
 backward_lstm_133_while_identity
Ъ
backward_lstm_133/while/LessLess#backward_lstm_133_while_placeholder>backward_lstm_133_while_less_backward_lstm_133_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_133/while/Less
 backward_lstm_133/while/IdentityIdentity backward_lstm_133/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_133/while/Identity"M
 backward_lstm_133_while_identity)backward_lstm_133/while/Identity:output:0*(
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
М
љ
0__inference_lstm_cell_401_layer_call_fn_16890153

inputs
states_0
states_1
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identity

identity_1

identity_2ЂStatefulPartitionedCallЦ
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
GPU 2J 8 *T
fORM
K__inference_lstm_cell_401_layer_call_and_return_conditional_losses_168849752
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
№?
л
while_body_16889495
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_401_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_401_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_401_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_401_matmul_readvariableop_resource:	ШG
4while_lstm_cell_401_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_401_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_401/BiasAdd/ReadVariableOpЂ)while/lstm_cell_401/MatMul/ReadVariableOpЂ+while/lstm_cell_401/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_401/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_401_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_401/MatMul/ReadVariableOpк
while/lstm_cell_401/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/MatMulв
+while/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_401_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_401/MatMul_1/ReadVariableOpУ
while/lstm_cell_401/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/MatMul_1М
while/lstm_cell_401/addAddV2$while/lstm_cell_401/MatMul:product:0&while/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/addЫ
*while/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_401_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_401/BiasAdd/ReadVariableOpЩ
while/lstm_cell_401/BiasAddBiasAddwhile/lstm_cell_401/add:z:02while/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/BiasAdd
#while/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_401/split/split_dim
while/lstm_cell_401/splitSplit,while/lstm_cell_401/split/split_dim:output:0$while/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_401/split
while/lstm_cell_401/SigmoidSigmoid"while/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Sigmoid
while/lstm_cell_401/Sigmoid_1Sigmoid"while/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Sigmoid_1Ѓ
while/lstm_cell_401/mulMul!while/lstm_cell_401/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/mul
while/lstm_cell_401/ReluRelu"while/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/ReluИ
while/lstm_cell_401/mul_1Mulwhile/lstm_cell_401/Sigmoid:y:0&while/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/mul_1­
while/lstm_cell_401/add_1AddV2while/lstm_cell_401/mul:z:0while/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/add_1
while/lstm_cell_401/Sigmoid_2Sigmoid"while/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Sigmoid_2
while/lstm_cell_401/Relu_1Reluwhile/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Relu_1М
while/lstm_cell_401/mul_2Mul!while/lstm_cell_401/Sigmoid_2:y:0(while/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_401/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_401/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_401/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_401/BiasAdd/ReadVariableOp*^while/lstm_cell_401/MatMul/ReadVariableOp,^while/lstm_cell_401/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_401_biasadd_readvariableop_resource5while_lstm_cell_401_biasadd_readvariableop_resource_0"n
4while_lstm_cell_401_matmul_1_readvariableop_resource6while_lstm_cell_401_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_401_matmul_readvariableop_resource4while_lstm_cell_401_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_401/BiasAdd/ReadVariableOp*while/lstm_cell_401/BiasAdd/ReadVariableOp2V
)while/lstm_cell_401/MatMul/ReadVariableOp)while/lstm_cell_401/MatMul/ReadVariableOp2Z
+while/lstm_cell_401/MatMul_1/ReadVariableOp+while/lstm_cell_401/MatMul_1/ReadVariableOp: 
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
М
љ
0__inference_lstm_cell_400_layer_call_fn_16890072

inputs
states_0
states_1
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identity

identity_1

identity_2ЂStatefulPartitionedCallЦ
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
GPU 2J 8 *T
fORM
K__inference_lstm_cell_400_layer_call_and_return_conditional_losses_168844892
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

ј
G__inference_dense_133_layer_call_and_return_conditional_losses_16888734

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
І_
­
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_16886045

inputs?
,lstm_cell_401_matmul_readvariableop_resource:	ШA
.lstm_cell_401_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_401_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_401/BiasAdd/ReadVariableOpЂ#lstm_cell_401/MatMul/ReadVariableOpЂ%lstm_cell_401/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_401/MatMul/ReadVariableOpReadVariableOp,lstm_cell_401_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_401/MatMul/ReadVariableOpА
lstm_cell_401/MatMulMatMulstrided_slice_2:output:0+lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/MatMulО
%lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_401_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_401/MatMul_1/ReadVariableOpЌ
lstm_cell_401/MatMul_1MatMulzeros:output:0-lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/MatMul_1Є
lstm_cell_401/addAddV2lstm_cell_401/MatMul:product:0 lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/addЗ
$lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_401_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_401/BiasAdd/ReadVariableOpБ
lstm_cell_401/BiasAddBiasAddlstm_cell_401/add:z:0,lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/BiasAdd
lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_401/split/split_dimї
lstm_cell_401/splitSplit&lstm_cell_401/split/split_dim:output:0lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_401/split
lstm_cell_401/SigmoidSigmoidlstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Sigmoid
lstm_cell_401/Sigmoid_1Sigmoidlstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Sigmoid_1
lstm_cell_401/mulMullstm_cell_401/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/mul
lstm_cell_401/ReluRelulstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Relu 
lstm_cell_401/mul_1Mullstm_cell_401/Sigmoid:y:0 lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/mul_1
lstm_cell_401/add_1AddV2lstm_cell_401/mul:z:0lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/add_1
lstm_cell_401/Sigmoid_2Sigmoidlstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Sigmoid_2
lstm_cell_401/Relu_1Relulstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Relu_1Є
lstm_cell_401/mul_2Mullstm_cell_401/Sigmoid_2:y:0"lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_401_matmul_readvariableop_resource.lstm_cell_401_matmul_1_readvariableop_resource-lstm_cell_401_biasadd_readvariableop_resource*
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
bodyR
while_body_16885961*
condR
while_cond_16885960*K
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
NoOpNoOp%^lstm_cell_401/BiasAdd/ReadVariableOp$^lstm_cell_401/MatMul/ReadVariableOp&^lstm_cell_401/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_401/BiasAdd/ReadVariableOp$lstm_cell_401/BiasAdd/ReadVariableOp2J
#lstm_cell_401/MatMul/ReadVariableOp#lstm_cell_401/MatMul/ReadVariableOp2N
%lstm_cell_401/MatMul_1/ReadVariableOp%lstm_cell_401/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ќ

и
1__inference_sequential_133_layer_call_fn_16886755

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
identityЂStatefulPartitionedCallе
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
GPU 2J 8 *U
fPRN
L__inference_sequential_133_layer_call_and_return_conditional_losses_168867362
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

Њ
Esequential_133_bidirectional_133_forward_lstm_133_while_cond_16883984
|sequential_133_bidirectional_133_forward_lstm_133_while_sequential_133_bidirectional_133_forward_lstm_133_while_loop_counter
sequential_133_bidirectional_133_forward_lstm_133_while_sequential_133_bidirectional_133_forward_lstm_133_while_maximum_iterationsG
Csequential_133_bidirectional_133_forward_lstm_133_while_placeholderI
Esequential_133_bidirectional_133_forward_lstm_133_while_placeholder_1I
Esequential_133_bidirectional_133_forward_lstm_133_while_placeholder_2I
Esequential_133_bidirectional_133_forward_lstm_133_while_placeholder_3I
Esequential_133_bidirectional_133_forward_lstm_133_while_placeholder_4
~sequential_133_bidirectional_133_forward_lstm_133_while_less_sequential_133_bidirectional_133_forward_lstm_133_strided_slice_1
sequential_133_bidirectional_133_forward_lstm_133_while_sequential_133_bidirectional_133_forward_lstm_133_while_cond_16883984___redundant_placeholder0
sequential_133_bidirectional_133_forward_lstm_133_while_sequential_133_bidirectional_133_forward_lstm_133_while_cond_16883984___redundant_placeholder1
sequential_133_bidirectional_133_forward_lstm_133_while_sequential_133_bidirectional_133_forward_lstm_133_while_cond_16883984___redundant_placeholder2
sequential_133_bidirectional_133_forward_lstm_133_while_sequential_133_bidirectional_133_forward_lstm_133_while_cond_16883984___redundant_placeholder3
sequential_133_bidirectional_133_forward_lstm_133_while_sequential_133_bidirectional_133_forward_lstm_133_while_cond_16883984___redundant_placeholder4D
@sequential_133_bidirectional_133_forward_lstm_133_while_identity
ъ
<sequential_133/bidirectional_133/forward_lstm_133/while/LessLessCsequential_133_bidirectional_133_forward_lstm_133_while_placeholder~sequential_133_bidirectional_133_forward_lstm_133_while_less_sequential_133_bidirectional_133_forward_lstm_133_strided_slice_1*
T0*
_output_shapes
: 2>
<sequential_133/bidirectional_133/forward_lstm_133/while/Lessѓ
@sequential_133/bidirectional_133/forward_lstm_133/while/IdentityIdentity@sequential_133/bidirectional_133/forward_lstm_133/while/Less:z:0*
T0
*
_output_shapes
: 2B
@sequential_133/bidirectional_133/forward_lstm_133/while/Identity"
@sequential_133_bidirectional_133_forward_lstm_133_while_identityIsequential_133/bidirectional_133/forward_lstm_133/while/Identity:output:0*(
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
&
ј
while_body_16884989
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_401_16885013_0:	Ш1
while_lstm_cell_401_16885015_0:	2Ш-
while_lstm_cell_401_16885017_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_401_16885013:	Ш/
while_lstm_cell_401_16885015:	2Ш+
while_lstm_cell_401_16885017:	ШЂ+while/lstm_cell_401/StatefulPartitionedCallУ
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
)while/TensorArrayV2Read/TensorListGetItemя
+while/lstm_cell_401/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_401_16885013_0while_lstm_cell_401_16885015_0while_lstm_cell_401_16885017_0*
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
GPU 2J 8 *T
fORM
K__inference_lstm_cell_401_layer_call_and_return_conditional_losses_168849752-
+while/lstm_cell_401/StatefulPartitionedCallј
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_401/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_401/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѕ
while/Identity_5Identity4while/lstm_cell_401/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp,^while/lstm_cell_401/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_401_16885013while_lstm_cell_401_16885013_0">
while_lstm_cell_401_16885015while_lstm_cell_401_16885015_0">
while_lstm_cell_401_16885017while_lstm_cell_401_16885017_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2Z
+while/lstm_cell_401/StatefulPartitionedCall+while/lstm_cell_401/StatefulPartitionedCall: 
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
ЯY

%backward_lstm_133_while_body_16887610@
<backward_lstm_133_while_backward_lstm_133_while_loop_counterF
Bbackward_lstm_133_while_backward_lstm_133_while_maximum_iterations'
#backward_lstm_133_while_placeholder)
%backward_lstm_133_while_placeholder_1)
%backward_lstm_133_while_placeholder_2)
%backward_lstm_133_while_placeholder_3?
;backward_lstm_133_while_backward_lstm_133_strided_slice_1_0{
wbackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0Y
Fbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0:	Ш[
Hbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0:	2ШV
Gbackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0:	Ш$
 backward_lstm_133_while_identity&
"backward_lstm_133_while_identity_1&
"backward_lstm_133_while_identity_2&
"backward_lstm_133_while_identity_3&
"backward_lstm_133_while_identity_4&
"backward_lstm_133_while_identity_5=
9backward_lstm_133_while_backward_lstm_133_strided_slice_1y
ubackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensorW
Dbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource:	ШY
Fbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource:	2ШT
Ebackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource:	ШЂ<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOpЂ;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpЂ=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOpч
Ibackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2K
Ibackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeШ
;backward_lstm_133/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_133_while_placeholderRbackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02=
;backward_lstm_133/while/TensorArrayV2Read/TensorListGetItem
;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpReadVariableOpFbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02=
;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpЂ
,backward_lstm_133/while/lstm_cell_401/MatMulMatMulBbackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem:item:0Cbackward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_133/while/lstm_cell_401/MatMul
=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOpHbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02?
=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp
.backward_lstm_133/while/lstm_cell_401/MatMul_1MatMul%backward_lstm_133_while_placeholder_2Ebackward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ20
.backward_lstm_133/while/lstm_cell_401/MatMul_1
)backward_lstm_133/while/lstm_cell_401/addAddV26backward_lstm_133/while/lstm_cell_401/MatMul:product:08backward_lstm_133/while/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_133/while/lstm_cell_401/add
<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOpGbackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02>
<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp
-backward_lstm_133/while/lstm_cell_401/BiasAddBiasAdd-backward_lstm_133/while/lstm_cell_401/add:z:0Dbackward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_133/while/lstm_cell_401/BiasAddА
5backward_lstm_133/while/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5backward_lstm_133/while/lstm_cell_401/split/split_dimз
+backward_lstm_133/while/lstm_cell_401/splitSplit>backward_lstm_133/while/lstm_cell_401/split/split_dim:output:06backward_lstm_133/while/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2-
+backward_lstm_133/while/lstm_cell_401/splitб
-backward_lstm_133/while/lstm_cell_401/SigmoidSigmoid4backward_lstm_133/while/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22/
-backward_lstm_133/while/lstm_cell_401/Sigmoidе
/backward_lstm_133/while/lstm_cell_401/Sigmoid_1Sigmoid4backward_lstm_133/while/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_133/while/lstm_cell_401/Sigmoid_1ы
)backward_lstm_133/while/lstm_cell_401/mulMul3backward_lstm_133/while/lstm_cell_401/Sigmoid_1:y:0%backward_lstm_133_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_133/while/lstm_cell_401/mulШ
*backward_lstm_133/while/lstm_cell_401/ReluRelu4backward_lstm_133/while/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_133/while/lstm_cell_401/Relu
+backward_lstm_133/while/lstm_cell_401/mul_1Mul1backward_lstm_133/while/lstm_cell_401/Sigmoid:y:08backward_lstm_133/while/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_133/while/lstm_cell_401/mul_1ѕ
+backward_lstm_133/while/lstm_cell_401/add_1AddV2-backward_lstm_133/while/lstm_cell_401/mul:z:0/backward_lstm_133/while/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_133/while/lstm_cell_401/add_1е
/backward_lstm_133/while/lstm_cell_401/Sigmoid_2Sigmoid4backward_lstm_133/while/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_133/while/lstm_cell_401/Sigmoid_2Ч
,backward_lstm_133/while/lstm_cell_401/Relu_1Relu/backward_lstm_133/while/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_133/while/lstm_cell_401/Relu_1
+backward_lstm_133/while/lstm_cell_401/mul_2Mul3backward_lstm_133/while/lstm_cell_401/Sigmoid_2:y:0:backward_lstm_133/while/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_133/while/lstm_cell_401/mul_2Л
<backward_lstm_133/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_133_while_placeholder_1#backward_lstm_133_while_placeholder/backward_lstm_133/while/lstm_cell_401/mul_2:z:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_133/while/TensorArrayV2Write/TensorListSetItem
backward_lstm_133/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_133/while/add/yБ
backward_lstm_133/while/addAddV2#backward_lstm_133_while_placeholder&backward_lstm_133/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/while/add
backward_lstm_133/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_133/while/add_1/yа
backward_lstm_133/while/add_1AddV2<backward_lstm_133_while_backward_lstm_133_while_loop_counter(backward_lstm_133/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/while/add_1Г
 backward_lstm_133/while/IdentityIdentity!backward_lstm_133/while/add_1:z:0^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_133/while/Identityи
"backward_lstm_133/while/Identity_1IdentityBbackward_lstm_133_while_backward_lstm_133_while_maximum_iterations^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_133/while/Identity_1Е
"backward_lstm_133/while/Identity_2Identitybackward_lstm_133/while/add:z:0^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_133/while/Identity_2т
"backward_lstm_133/while/Identity_3IdentityLbackward_lstm_133/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_133/while/Identity_3ж
"backward_lstm_133/while/Identity_4Identity/backward_lstm_133/while/lstm_cell_401/mul_2:z:0^backward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_133/while/Identity_4ж
"backward_lstm_133/while/Identity_5Identity/backward_lstm_133/while/lstm_cell_401/add_1:z:0^backward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_133/while/Identity_5Л
backward_lstm_133/while/NoOpNoOp=^backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp<^backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp>^backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_133/while/NoOp"x
9backward_lstm_133_while_backward_lstm_133_strided_slice_1;backward_lstm_133_while_backward_lstm_133_strided_slice_1_0"M
 backward_lstm_133_while_identity)backward_lstm_133/while/Identity:output:0"Q
"backward_lstm_133_while_identity_1+backward_lstm_133/while/Identity_1:output:0"Q
"backward_lstm_133_while_identity_2+backward_lstm_133/while/Identity_2:output:0"Q
"backward_lstm_133_while_identity_3+backward_lstm_133/while/Identity_3:output:0"Q
"backward_lstm_133_while_identity_4+backward_lstm_133/while/Identity_4:output:0"Q
"backward_lstm_133_while_identity_5+backward_lstm_133/while/Identity_5:output:0"
Ebackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resourceGbackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0"
Fbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resourceHbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0"
Dbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resourceFbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0"№
ubackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2|
<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp2z
;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp2~
=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp: 
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
ж
У
4__inference_backward_lstm_133_layer_call_fn_16889404
inputs_0
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
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
GPU 2J 8 *X
fSRQ
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_168852702
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
]
Ќ
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_16885693

inputs?
,lstm_cell_400_matmul_readvariableop_resource:	ШA
.lstm_cell_400_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_400_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_400/BiasAdd/ReadVariableOpЂ#lstm_cell_400/MatMul/ReadVariableOpЂ%lstm_cell_400/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_400/MatMul/ReadVariableOpReadVariableOp,lstm_cell_400_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_400/MatMul/ReadVariableOpА
lstm_cell_400/MatMulMatMulstrided_slice_2:output:0+lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/MatMulО
%lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_400_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_400/MatMul_1/ReadVariableOpЌ
lstm_cell_400/MatMul_1MatMulzeros:output:0-lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/MatMul_1Є
lstm_cell_400/addAddV2lstm_cell_400/MatMul:product:0 lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/addЗ
$lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_400_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_400/BiasAdd/ReadVariableOpБ
lstm_cell_400/BiasAddBiasAddlstm_cell_400/add:z:0,lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/BiasAdd
lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_400/split/split_dimї
lstm_cell_400/splitSplit&lstm_cell_400/split/split_dim:output:0lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_400/split
lstm_cell_400/SigmoidSigmoidlstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Sigmoid
lstm_cell_400/Sigmoid_1Sigmoidlstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Sigmoid_1
lstm_cell_400/mulMullstm_cell_400/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/mul
lstm_cell_400/ReluRelulstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Relu 
lstm_cell_400/mul_1Mullstm_cell_400/Sigmoid:y:0 lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/mul_1
lstm_cell_400/add_1AddV2lstm_cell_400/mul:z:0lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/add_1
lstm_cell_400/Sigmoid_2Sigmoidlstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Sigmoid_2
lstm_cell_400/Relu_1Relulstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Relu_1Є
lstm_cell_400/mul_2Mullstm_cell_400/Sigmoid_2:y:0"lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_400_matmul_readvariableop_resource.lstm_cell_400_matmul_1_readvariableop_resource-lstm_cell_400_biasadd_readvariableop_resource*
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
bodyR
while_body_16885609*
condR
while_cond_16885608*K
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
NoOpNoOp%^lstm_cell_400/BiasAdd/ReadVariableOp$^lstm_cell_400/MatMul/ReadVariableOp&^lstm_cell_400/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_400/BiasAdd/ReadVariableOp$lstm_cell_400/BiasAdd/ReadVariableOp2J
#lstm_cell_400/MatMul/ReadVariableOp#lstm_cell_400/MatMul/ReadVariableOp2N
%lstm_cell_400/MatMul_1/ReadVariableOp%lstm_cell_400/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Н
я
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_16888714

inputs
inputs_1	P
=forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource:	ШR
?forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource:	2ШM
>forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource:	ШQ
>backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource:	ШS
@backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource:	2ШN
?backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource:	Ш
identityЂ6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpЂ5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpЂ7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpЂbackward_lstm_133/whileЂ5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpЂ4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpЂ6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOpЂforward_lstm_133/while
%forward_lstm_133/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_133/RaggedToTensor/zeros
%forward_lstm_133/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2'
%forward_lstm_133/RaggedToTensor/Const
4forward_lstm_133/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_133/RaggedToTensor/Const:output:0inputs.forward_lstm_133/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_133/RaggedToTensor/RaggedTensorToTensorФ
;forward_lstm_133/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_133/RaggedNestedRowLengths/strided_slice/stackШ
=forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1Ш
=forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2Љ
5forward_lstm_133/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_133/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask27
5forward_lstm_133/RaggedNestedRowLengths/strided_sliceШ
=forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stackе
?forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2A
?forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1Ь
?forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2Е
7forward_lstm_133/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask29
7forward_lstm_133/RaggedNestedRowLengths/strided_slice_1
+forward_lstm_133/RaggedNestedRowLengths/subSub>forward_lstm_133/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_133/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2-
+forward_lstm_133/RaggedNestedRowLengths/subЄ
forward_lstm_133/CastCast/forward_lstm_133/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_133/Cast
forward_lstm_133/ShapeShape=forward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_133/Shape
$forward_lstm_133/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_133/strided_slice/stack
&forward_lstm_133/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_133/strided_slice/stack_1
&forward_lstm_133/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_133/strided_slice/stack_2Ш
forward_lstm_133/strided_sliceStridedSliceforward_lstm_133/Shape:output:0-forward_lstm_133/strided_slice/stack:output:0/forward_lstm_133/strided_slice/stack_1:output:0/forward_lstm_133/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_133/strided_slice~
forward_lstm_133/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_133/zeros/mul/yА
forward_lstm_133/zeros/mulMul'forward_lstm_133/strided_slice:output:0%forward_lstm_133/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros/mul
forward_lstm_133/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_133/zeros/Less/yЋ
forward_lstm_133/zeros/LessLessforward_lstm_133/zeros/mul:z:0&forward_lstm_133/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros/Less
forward_lstm_133/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_133/zeros/packed/1Ч
forward_lstm_133/zeros/packedPack'forward_lstm_133/strided_slice:output:0(forward_lstm_133/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_133/zeros/packed
forward_lstm_133/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_133/zeros/ConstЙ
forward_lstm_133/zerosFill&forward_lstm_133/zeros/packed:output:0%forward_lstm_133/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/zeros
forward_lstm_133/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_133/zeros_1/mul/yЖ
forward_lstm_133/zeros_1/mulMul'forward_lstm_133/strided_slice:output:0'forward_lstm_133/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros_1/mul
forward_lstm_133/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
forward_lstm_133/zeros_1/Less/yГ
forward_lstm_133/zeros_1/LessLess forward_lstm_133/zeros_1/mul:z:0(forward_lstm_133/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros_1/Less
!forward_lstm_133/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_133/zeros_1/packed/1Э
forward_lstm_133/zeros_1/packedPack'forward_lstm_133/strided_slice:output:0*forward_lstm_133/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_133/zeros_1/packed
forward_lstm_133/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_133/zeros_1/ConstС
forward_lstm_133/zeros_1Fill(forward_lstm_133/zeros_1/packed:output:0'forward_lstm_133/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/zeros_1
forward_lstm_133/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_133/transpose/permэ
forward_lstm_133/transpose	Transpose=forward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_133/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_133/transpose
forward_lstm_133/Shape_1Shapeforward_lstm_133/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_133/Shape_1
&forward_lstm_133/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_133/strided_slice_1/stack
(forward_lstm_133/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_1/stack_1
(forward_lstm_133/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_1/stack_2д
 forward_lstm_133/strided_slice_1StridedSlice!forward_lstm_133/Shape_1:output:0/forward_lstm_133/strided_slice_1/stack:output:01forward_lstm_133/strided_slice_1/stack_1:output:01forward_lstm_133/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_133/strided_slice_1Ї
,forward_lstm_133/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,forward_lstm_133/TensorArrayV2/element_shapeі
forward_lstm_133/TensorArrayV2TensorListReserve5forward_lstm_133/TensorArrayV2/element_shape:output:0)forward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_133/TensorArrayV2с
Fforward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fforward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8forward_lstm_133/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_133/transpose:y:0Oforward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_133/TensorArrayUnstack/TensorListFromTensor
&forward_lstm_133/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_133/strided_slice_2/stack
(forward_lstm_133/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_2/stack_1
(forward_lstm_133/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_2/stack_2т
 forward_lstm_133/strided_slice_2StridedSliceforward_lstm_133/transpose:y:0/forward_lstm_133/strided_slice_2/stack:output:01forward_lstm_133/strided_slice_2/stack_1:output:01forward_lstm_133/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 forward_lstm_133/strided_slice_2ы
4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpReadVariableOp=forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpє
%forward_lstm_133/lstm_cell_400/MatMulMatMul)forward_lstm_133/strided_slice_2:output:0<forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_133/lstm_cell_400/MatMulё
6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOp?forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp№
'forward_lstm_133/lstm_cell_400/MatMul_1MatMulforward_lstm_133/zeros:output:0>forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_133/lstm_cell_400/MatMul_1ш
"forward_lstm_133/lstm_cell_400/addAddV2/forward_lstm_133/lstm_cell_400/MatMul:product:01forward_lstm_133/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_133/lstm_cell_400/addъ
5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOp>forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpѕ
&forward_lstm_133/lstm_cell_400/BiasAddBiasAdd&forward_lstm_133/lstm_cell_400/add:z:0=forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_133/lstm_cell_400/BiasAddЂ
.forward_lstm_133/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.forward_lstm_133/lstm_cell_400/split/split_dimЛ
$forward_lstm_133/lstm_cell_400/splitSplit7forward_lstm_133/lstm_cell_400/split/split_dim:output:0/forward_lstm_133/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$forward_lstm_133/lstm_cell_400/splitМ
&forward_lstm_133/lstm_cell_400/SigmoidSigmoid-forward_lstm_133/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_133/lstm_cell_400/SigmoidР
(forward_lstm_133/lstm_cell_400/Sigmoid_1Sigmoid-forward_lstm_133/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_133/lstm_cell_400/Sigmoid_1в
"forward_lstm_133/lstm_cell_400/mulMul,forward_lstm_133/lstm_cell_400/Sigmoid_1:y:0!forward_lstm_133/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_133/lstm_cell_400/mulГ
#forward_lstm_133/lstm_cell_400/ReluRelu-forward_lstm_133/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_133/lstm_cell_400/Reluф
$forward_lstm_133/lstm_cell_400/mul_1Mul*forward_lstm_133/lstm_cell_400/Sigmoid:y:01forward_lstm_133/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_133/lstm_cell_400/mul_1й
$forward_lstm_133/lstm_cell_400/add_1AddV2&forward_lstm_133/lstm_cell_400/mul:z:0(forward_lstm_133/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_133/lstm_cell_400/add_1Р
(forward_lstm_133/lstm_cell_400/Sigmoid_2Sigmoid-forward_lstm_133/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_133/lstm_cell_400/Sigmoid_2В
%forward_lstm_133/lstm_cell_400/Relu_1Relu(forward_lstm_133/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_133/lstm_cell_400/Relu_1ш
$forward_lstm_133/lstm_cell_400/mul_2Mul,forward_lstm_133/lstm_cell_400/Sigmoid_2:y:03forward_lstm_133/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_133/lstm_cell_400/mul_2Б
.forward_lstm_133/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.forward_lstm_133/TensorArrayV2_1/element_shapeќ
 forward_lstm_133/TensorArrayV2_1TensorListReserve7forward_lstm_133/TensorArrayV2_1/element_shape:output:0)forward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_133/TensorArrayV2_1p
forward_lstm_133/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_133/timeЃ
forward_lstm_133/zeros_like	ZerosLike(forward_lstm_133/lstm_cell_400/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/zeros_likeЁ
)forward_lstm_133/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)forward_lstm_133/while/maximum_iterations
#forward_lstm_133/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_133/while/loop_counter	
forward_lstm_133/whileWhile,forward_lstm_133/while/loop_counter:output:02forward_lstm_133/while/maximum_iterations:output:0forward_lstm_133/time:output:0)forward_lstm_133/TensorArrayV2_1:handle:0forward_lstm_133/zeros_like:y:0forward_lstm_133/zeros:output:0!forward_lstm_133/zeros_1:output:0)forward_lstm_133/strided_slice_1:output:0Hforward_lstm_133/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_133/Cast:y:0=forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource?forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource>forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *0
body(R&
$forward_lstm_133_while_body_16888438*0
cond(R&
$forward_lstm_133_while_cond_16888437*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_133/whileз
Aforward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Aforward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3forward_lstm_133/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_133/while:output:3Jforward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3forward_lstm_133/TensorArrayV2Stack/TensorListStackЃ
&forward_lstm_133/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&forward_lstm_133/strided_slice_3/stack
(forward_lstm_133/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_133/strided_slice_3/stack_1
(forward_lstm_133/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_3/stack_2
 forward_lstm_133/strided_slice_3StridedSlice<forward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_133/strided_slice_3/stack:output:01forward_lstm_133/strided_slice_3/stack_1:output:01forward_lstm_133/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 forward_lstm_133/strided_slice_3
!forward_lstm_133/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_133/transpose_1/permђ
forward_lstm_133/transpose_1	Transpose<forward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_133/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_133/transpose_1
forward_lstm_133/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_133/runtime
&backward_lstm_133/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_133/RaggedToTensor/zeros
&backward_lstm_133/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2(
&backward_lstm_133/RaggedToTensor/Const
5backward_lstm_133/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_133/RaggedToTensor/Const:output:0inputs/backward_lstm_133/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_133/RaggedToTensor/RaggedTensorToTensorЦ
<backward_lstm_133/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_133/RaggedNestedRowLengths/strided_slice/stackЪ
>backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1Ъ
>backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2Ў
6backward_lstm_133/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_133/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask28
6backward_lstm_133/RaggedNestedRowLengths/strided_sliceЪ
>backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stackз
@backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2B
@backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1Ю
@backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2К
8backward_lstm_133/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2:
8backward_lstm_133/RaggedNestedRowLengths/strided_slice_1
,backward_lstm_133/RaggedNestedRowLengths/subSub?backward_lstm_133/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_133/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2.
,backward_lstm_133/RaggedNestedRowLengths/subЇ
backward_lstm_133/CastCast0backward_lstm_133/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_133/Cast 
backward_lstm_133/ShapeShape>backward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_133/Shape
%backward_lstm_133/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_133/strided_slice/stack
'backward_lstm_133/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_133/strided_slice/stack_1
'backward_lstm_133/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_133/strided_slice/stack_2Ю
backward_lstm_133/strided_sliceStridedSlice backward_lstm_133/Shape:output:0.backward_lstm_133/strided_slice/stack:output:00backward_lstm_133/strided_slice/stack_1:output:00backward_lstm_133/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_133/strided_slice
backward_lstm_133/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_133/zeros/mul/yД
backward_lstm_133/zeros/mulMul(backward_lstm_133/strided_slice:output:0&backward_lstm_133/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/zeros/mul
backward_lstm_133/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_133/zeros/Less/yЏ
backward_lstm_133/zeros/LessLessbackward_lstm_133/zeros/mul:z:0'backward_lstm_133/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/zeros/Less
 backward_lstm_133/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_133/zeros/packed/1Ы
backward_lstm_133/zeros/packedPack(backward_lstm_133/strided_slice:output:0)backward_lstm_133/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_133/zeros/packed
backward_lstm_133/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_133/zeros/ConstН
backward_lstm_133/zerosFill'backward_lstm_133/zeros/packed:output:0&backward_lstm_133/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_133/zeros
backward_lstm_133/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_133/zeros_1/mul/yК
backward_lstm_133/zeros_1/mulMul(backward_lstm_133/strided_slice:output:0(backward_lstm_133/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/zeros_1/mul
 backward_lstm_133/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 backward_lstm_133/zeros_1/Less/yЗ
backward_lstm_133/zeros_1/LessLess!backward_lstm_133/zeros_1/mul:z:0)backward_lstm_133/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_133/zeros_1/Less
"backward_lstm_133/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_133/zeros_1/packed/1б
 backward_lstm_133/zeros_1/packedPack(backward_lstm_133/strided_slice:output:0+backward_lstm_133/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_133/zeros_1/packed
backward_lstm_133/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_133/zeros_1/ConstХ
backward_lstm_133/zeros_1Fill)backward_lstm_133/zeros_1/packed:output:0(backward_lstm_133/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_133/zeros_1
 backward_lstm_133/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_133/transpose/permё
backward_lstm_133/transpose	Transpose>backward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_133/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_133/transpose
backward_lstm_133/Shape_1Shapebackward_lstm_133/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_133/Shape_1
'backward_lstm_133/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_133/strided_slice_1/stack 
)backward_lstm_133/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_1/stack_1 
)backward_lstm_133/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_1/stack_2к
!backward_lstm_133/strided_slice_1StridedSlice"backward_lstm_133/Shape_1:output:00backward_lstm_133/strided_slice_1/stack:output:02backward_lstm_133/strided_slice_1/stack_1:output:02backward_lstm_133/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_133/strided_slice_1Љ
-backward_lstm_133/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2/
-backward_lstm_133/TensorArrayV2/element_shapeњ
backward_lstm_133/TensorArrayV2TensorListReserve6backward_lstm_133/TensorArrayV2/element_shape:output:0*backward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_133/TensorArrayV2
 backward_lstm_133/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_133/ReverseV2/axisв
backward_lstm_133/ReverseV2	ReverseV2backward_lstm_133/transpose:y:0)backward_lstm_133/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_133/ReverseV2у
Gbackward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gbackward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeХ
9backward_lstm_133/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_133/ReverseV2:output:0Pbackward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_133/TensorArrayUnstack/TensorListFromTensor
'backward_lstm_133/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_133/strided_slice_2/stack 
)backward_lstm_133/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_2/stack_1 
)backward_lstm_133/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_2/stack_2ш
!backward_lstm_133/strided_slice_2StridedSlicebackward_lstm_133/transpose:y:00backward_lstm_133/strided_slice_2/stack:output:02backward_lstm_133/strided_slice_2/stack_1:output:02backward_lstm_133/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2#
!backward_lstm_133/strided_slice_2ю
5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpReadVariableOp>backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype027
5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpј
&backward_lstm_133/lstm_cell_401/MatMulMatMul*backward_lstm_133/strided_slice_2:output:0=backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_133/lstm_cell_401/MatMulє
7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOp@backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype029
7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpє
(backward_lstm_133/lstm_cell_401/MatMul_1MatMul backward_lstm_133/zeros:output:0?backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_133/lstm_cell_401/MatMul_1ь
#backward_lstm_133/lstm_cell_401/addAddV20backward_lstm_133/lstm_cell_401/MatMul:product:02backward_lstm_133/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_133/lstm_cell_401/addэ
6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOp?backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype028
6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpљ
'backward_lstm_133/lstm_cell_401/BiasAddBiasAdd'backward_lstm_133/lstm_cell_401/add:z:0>backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_133/lstm_cell_401/BiasAddЄ
/backward_lstm_133/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/backward_lstm_133/lstm_cell_401/split/split_dimП
%backward_lstm_133/lstm_cell_401/splitSplit8backward_lstm_133/lstm_cell_401/split/split_dim:output:00backward_lstm_133/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2'
%backward_lstm_133/lstm_cell_401/splitП
'backward_lstm_133/lstm_cell_401/SigmoidSigmoid.backward_lstm_133/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_133/lstm_cell_401/SigmoidУ
)backward_lstm_133/lstm_cell_401/Sigmoid_1Sigmoid.backward_lstm_133/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_133/lstm_cell_401/Sigmoid_1ж
#backward_lstm_133/lstm_cell_401/mulMul-backward_lstm_133/lstm_cell_401/Sigmoid_1:y:0"backward_lstm_133/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_133/lstm_cell_401/mulЖ
$backward_lstm_133/lstm_cell_401/ReluRelu.backward_lstm_133/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_133/lstm_cell_401/Reluш
%backward_lstm_133/lstm_cell_401/mul_1Mul+backward_lstm_133/lstm_cell_401/Sigmoid:y:02backward_lstm_133/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_133/lstm_cell_401/mul_1н
%backward_lstm_133/lstm_cell_401/add_1AddV2'backward_lstm_133/lstm_cell_401/mul:z:0)backward_lstm_133/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_133/lstm_cell_401/add_1У
)backward_lstm_133/lstm_cell_401/Sigmoid_2Sigmoid.backward_lstm_133/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_133/lstm_cell_401/Sigmoid_2Е
&backward_lstm_133/lstm_cell_401/Relu_1Relu)backward_lstm_133/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_133/lstm_cell_401/Relu_1ь
%backward_lstm_133/lstm_cell_401/mul_2Mul-backward_lstm_133/lstm_cell_401/Sigmoid_2:y:04backward_lstm_133/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_133/lstm_cell_401/mul_2Г
/backward_lstm_133/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   21
/backward_lstm_133/TensorArrayV2_1/element_shape
!backward_lstm_133/TensorArrayV2_1TensorListReserve8backward_lstm_133/TensorArrayV2_1/element_shape:output:0*backward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_133/TensorArrayV2_1r
backward_lstm_133/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_133/time
'backward_lstm_133/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_133/Max/reduction_indicesЄ
backward_lstm_133/MaxMaxbackward_lstm_133/Cast:y:00backward_lstm_133/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/Maxt
backward_lstm_133/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_133/sub/y
backward_lstm_133/subSubbackward_lstm_133/Max:output:0 backward_lstm_133/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/sub
backward_lstm_133/Sub_1Subbackward_lstm_133/sub:z:0backward_lstm_133/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_133/Sub_1І
backward_lstm_133/zeros_like	ZerosLike)backward_lstm_133/lstm_cell_401/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_133/zeros_likeЃ
*backward_lstm_133/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*backward_lstm_133/while/maximum_iterations
$backward_lstm_133/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_133/while/loop_counterЅ	
backward_lstm_133/whileWhile-backward_lstm_133/while/loop_counter:output:03backward_lstm_133/while/maximum_iterations:output:0backward_lstm_133/time:output:0*backward_lstm_133/TensorArrayV2_1:handle:0 backward_lstm_133/zeros_like:y:0 backward_lstm_133/zeros:output:0"backward_lstm_133/zeros_1:output:0*backward_lstm_133/strided_slice_1:output:0Ibackward_lstm_133/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_133/Sub_1:z:0>backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource@backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource?backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *1
body)R'
%backward_lstm_133_while_body_16888617*1
cond)R'
%backward_lstm_133_while_cond_16888616*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_133/whileй
Bbackward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2D
Bbackward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeЙ
4backward_lstm_133/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_133/while:output:3Kbackward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype026
4backward_lstm_133/TensorArrayV2Stack/TensorListStackЅ
'backward_lstm_133/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2)
'backward_lstm_133/strided_slice_3/stack 
)backward_lstm_133/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_133/strided_slice_3/stack_1 
)backward_lstm_133/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_3/stack_2
!backward_lstm_133/strided_slice_3StridedSlice=backward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_133/strided_slice_3/stack:output:02backward_lstm_133/strided_slice_3/stack_1:output:02backward_lstm_133/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2#
!backward_lstm_133/strided_slice_3
"backward_lstm_133/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_133/transpose_1/permі
backward_lstm_133/transpose_1	Transpose=backward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_133/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_133/transpose_1
backward_lstm_133/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_133/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisФ
concatConcatV2)forward_lstm_133/strided_slice_3:output:0*backward_lstm_133/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

Identityд
NoOpNoOp7^backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp6^backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp8^backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp^backward_lstm_133/while6^forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp5^forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp7^forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp^forward_lstm_133/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2p
6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp2n
5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp2r
7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp22
backward_lstm_133/whilebackward_lstm_133/while2n
5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp2l
4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp2p
6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp20
forward_lstm_133/whileforward_lstm_133/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ї

,__inference_dense_133_layer_call_fn_16888723

inputs
unknown:d
	unknown_0:
identityЂStatefulPartitionedCallї
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
GPU 2J 8 *P
fKRI
G__inference_dense_133_layer_call_and_return_conditional_losses_168867292
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
ў

K__inference_lstm_cell_400_layer_call_and_return_conditional_losses_16890136

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
і

K__inference_lstm_cell_401_layer_call_and_return_conditional_losses_16884975

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
љ?
л
while_body_16885609
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_400_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_400_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_400_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_400_matmul_readvariableop_resource:	ШG
4while_lstm_cell_400_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_400_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_400/BiasAdd/ReadVariableOpЂ)while/lstm_cell_400/MatMul/ReadVariableOpЂ+while/lstm_cell_400/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_400/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_400_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_400/MatMul/ReadVariableOpк
while/lstm_cell_400/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/MatMulв
+while/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_400_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_400/MatMul_1/ReadVariableOpУ
while/lstm_cell_400/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/MatMul_1М
while/lstm_cell_400/addAddV2$while/lstm_cell_400/MatMul:product:0&while/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/addЫ
*while/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_400_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_400/BiasAdd/ReadVariableOpЩ
while/lstm_cell_400/BiasAddBiasAddwhile/lstm_cell_400/add:z:02while/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/BiasAdd
#while/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_400/split/split_dim
while/lstm_cell_400/splitSplit,while/lstm_cell_400/split/split_dim:output:0$while/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_400/split
while/lstm_cell_400/SigmoidSigmoid"while/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Sigmoid
while/lstm_cell_400/Sigmoid_1Sigmoid"while/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Sigmoid_1Ѓ
while/lstm_cell_400/mulMul!while/lstm_cell_400/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/mul
while/lstm_cell_400/ReluRelu"while/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/ReluИ
while/lstm_cell_400/mul_1Mulwhile/lstm_cell_400/Sigmoid:y:0&while/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/mul_1­
while/lstm_cell_400/add_1AddV2while/lstm_cell_400/mul:z:0while/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/add_1
while/lstm_cell_400/Sigmoid_2Sigmoid"while/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Sigmoid_2
while/lstm_cell_400/Relu_1Reluwhile/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Relu_1М
while/lstm_cell_400/mul_2Mul!while/lstm_cell_400/Sigmoid_2:y:0(while/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_400/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_400/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_400/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_400/BiasAdd/ReadVariableOp*^while/lstm_cell_400/MatMul/ReadVariableOp,^while/lstm_cell_400/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_400_biasadd_readvariableop_resource5while_lstm_cell_400_biasadd_readvariableop_resource_0"n
4while_lstm_cell_400_matmul_1_readvariableop_resource6while_lstm_cell_400_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_400_matmul_readvariableop_resource4while_lstm_cell_400_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_400/BiasAdd/ReadVariableOp*while/lstm_cell_400/BiasAdd/ReadVariableOp2V
)while/lstm_cell_400/MatMul/ReadVariableOp)while/lstm_cell_400/MatMul/ReadVariableOp2Z
+while/lstm_cell_400/MatMul_1/ReadVariableOp+while/lstm_cell_400/MatMul_1/ReadVariableOp: 
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
љ?
л
while_body_16889147
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_400_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_400_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_400_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_400_matmul_readvariableop_resource:	ШG
4while_lstm_cell_400_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_400_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_400/BiasAdd/ReadVariableOpЂ)while/lstm_cell_400/MatMul/ReadVariableOpЂ+while/lstm_cell_400/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_400/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_400_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_400/MatMul/ReadVariableOpк
while/lstm_cell_400/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/MatMulв
+while/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_400_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_400/MatMul_1/ReadVariableOpУ
while/lstm_cell_400/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/MatMul_1М
while/lstm_cell_400/addAddV2$while/lstm_cell_400/MatMul:product:0&while/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/addЫ
*while/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_400_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_400/BiasAdd/ReadVariableOpЩ
while/lstm_cell_400/BiasAddBiasAddwhile/lstm_cell_400/add:z:02while/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/BiasAdd
#while/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_400/split/split_dim
while/lstm_cell_400/splitSplit,while/lstm_cell_400/split/split_dim:output:0$while/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_400/split
while/lstm_cell_400/SigmoidSigmoid"while/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Sigmoid
while/lstm_cell_400/Sigmoid_1Sigmoid"while/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Sigmoid_1Ѓ
while/lstm_cell_400/mulMul!while/lstm_cell_400/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/mul
while/lstm_cell_400/ReluRelu"while/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/ReluИ
while/lstm_cell_400/mul_1Mulwhile/lstm_cell_400/Sigmoid:y:0&while/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/mul_1­
while/lstm_cell_400/add_1AddV2while/lstm_cell_400/mul:z:0while/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/add_1
while/lstm_cell_400/Sigmoid_2Sigmoid"while/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Sigmoid_2
while/lstm_cell_400/Relu_1Reluwhile/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Relu_1М
while/lstm_cell_400/mul_2Mul!while/lstm_cell_400/Sigmoid_2:y:0(while/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_400/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_400/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_400/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_400/BiasAdd/ReadVariableOp*^while/lstm_cell_400/MatMul/ReadVariableOp,^while/lstm_cell_400/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_400_biasadd_readvariableop_resource5while_lstm_cell_400_biasadd_readvariableop_resource_0"n
4while_lstm_cell_400_matmul_1_readvariableop_resource6while_lstm_cell_400_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_400_matmul_readvariableop_resource4while_lstm_cell_400_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_400/BiasAdd/ReadVariableOp*while/lstm_cell_400/BiasAdd/ReadVariableOp2V
)while/lstm_cell_400/MatMul/ReadVariableOp)while/lstm_cell_400/MatMul/ReadVariableOp2Z
+while/lstm_cell_400/MatMul_1/ReadVariableOp+while/lstm_cell_400/MatMul_1/ReadVariableOp: 
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
ў

K__inference_lstm_cell_400_layer_call_and_return_conditional_losses_16890104

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
ъ
М
%backward_lstm_133_while_cond_16887046@
<backward_lstm_133_while_backward_lstm_133_while_loop_counterF
Bbackward_lstm_133_while_backward_lstm_133_while_maximum_iterations'
#backward_lstm_133_while_placeholder)
%backward_lstm_133_while_placeholder_1)
%backward_lstm_133_while_placeholder_2)
%backward_lstm_133_while_placeholder_3)
%backward_lstm_133_while_placeholder_4B
>backward_lstm_133_while_less_backward_lstm_133_strided_slice_1Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16887046___redundant_placeholder0Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16887046___redundant_placeholder1Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16887046___redundant_placeholder2Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16887046___redundant_placeholder3Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16887046___redundant_placeholder4$
 backward_lstm_133_while_identity
Ъ
backward_lstm_133/while/LessLess#backward_lstm_133_while_placeholder>backward_lstm_133_while_less_backward_lstm_133_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_133/while/Less
 backward_lstm_133/while/IdentityIdentity backward_lstm_133/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_133/while/Identity"M
 backward_lstm_133_while_identity)backward_lstm_133/while/Identity:output:0*(
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
Ф§
у
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_16887998
inputs_0P
=forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource:	ШR
?forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource:	2ШM
>forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource:	ШQ
>backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource:	ШS
@backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource:	2ШN
?backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource:	Ш
identityЂ6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpЂ5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpЂ7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpЂbackward_lstm_133/whileЂ5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpЂ4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpЂ6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOpЂforward_lstm_133/whileh
forward_lstm_133/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_133/Shape
$forward_lstm_133/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_133/strided_slice/stack
&forward_lstm_133/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_133/strided_slice/stack_1
&forward_lstm_133/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_133/strided_slice/stack_2Ш
forward_lstm_133/strided_sliceStridedSliceforward_lstm_133/Shape:output:0-forward_lstm_133/strided_slice/stack:output:0/forward_lstm_133/strided_slice/stack_1:output:0/forward_lstm_133/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_133/strided_slice~
forward_lstm_133/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_133/zeros/mul/yА
forward_lstm_133/zeros/mulMul'forward_lstm_133/strided_slice:output:0%forward_lstm_133/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros/mul
forward_lstm_133/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_133/zeros/Less/yЋ
forward_lstm_133/zeros/LessLessforward_lstm_133/zeros/mul:z:0&forward_lstm_133/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros/Less
forward_lstm_133/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_133/zeros/packed/1Ч
forward_lstm_133/zeros/packedPack'forward_lstm_133/strided_slice:output:0(forward_lstm_133/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_133/zeros/packed
forward_lstm_133/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_133/zeros/ConstЙ
forward_lstm_133/zerosFill&forward_lstm_133/zeros/packed:output:0%forward_lstm_133/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/zeros
forward_lstm_133/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_133/zeros_1/mul/yЖ
forward_lstm_133/zeros_1/mulMul'forward_lstm_133/strided_slice:output:0'forward_lstm_133/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros_1/mul
forward_lstm_133/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
forward_lstm_133/zeros_1/Less/yГ
forward_lstm_133/zeros_1/LessLess forward_lstm_133/zeros_1/mul:z:0(forward_lstm_133/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros_1/Less
!forward_lstm_133/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_133/zeros_1/packed/1Э
forward_lstm_133/zeros_1/packedPack'forward_lstm_133/strided_slice:output:0*forward_lstm_133/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_133/zeros_1/packed
forward_lstm_133/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_133/zeros_1/ConstС
forward_lstm_133/zeros_1Fill(forward_lstm_133/zeros_1/packed:output:0'forward_lstm_133/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/zeros_1
forward_lstm_133/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_133/transpose/permС
forward_lstm_133/transpose	Transposeinputs_0(forward_lstm_133/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
forward_lstm_133/transpose
forward_lstm_133/Shape_1Shapeforward_lstm_133/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_133/Shape_1
&forward_lstm_133/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_133/strided_slice_1/stack
(forward_lstm_133/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_1/stack_1
(forward_lstm_133/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_1/stack_2д
 forward_lstm_133/strided_slice_1StridedSlice!forward_lstm_133/Shape_1:output:0/forward_lstm_133/strided_slice_1/stack:output:01forward_lstm_133/strided_slice_1/stack_1:output:01forward_lstm_133/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_133/strided_slice_1Ї
,forward_lstm_133/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,forward_lstm_133/TensorArrayV2/element_shapeі
forward_lstm_133/TensorArrayV2TensorListReserve5forward_lstm_133/TensorArrayV2/element_shape:output:0)forward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_133/TensorArrayV2с
Fforward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2H
Fforward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8forward_lstm_133/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_133/transpose:y:0Oforward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_133/TensorArrayUnstack/TensorListFromTensor
&forward_lstm_133/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_133/strided_slice_2/stack
(forward_lstm_133/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_2/stack_1
(forward_lstm_133/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_2/stack_2ы
 forward_lstm_133/strided_slice_2StridedSliceforward_lstm_133/transpose:y:0/forward_lstm_133/strided_slice_2/stack:output:01forward_lstm_133/strided_slice_2/stack_1:output:01forward_lstm_133/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2"
 forward_lstm_133/strided_slice_2ы
4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpReadVariableOp=forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpє
%forward_lstm_133/lstm_cell_400/MatMulMatMul)forward_lstm_133/strided_slice_2:output:0<forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_133/lstm_cell_400/MatMulё
6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOp?forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp№
'forward_lstm_133/lstm_cell_400/MatMul_1MatMulforward_lstm_133/zeros:output:0>forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_133/lstm_cell_400/MatMul_1ш
"forward_lstm_133/lstm_cell_400/addAddV2/forward_lstm_133/lstm_cell_400/MatMul:product:01forward_lstm_133/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_133/lstm_cell_400/addъ
5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOp>forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpѕ
&forward_lstm_133/lstm_cell_400/BiasAddBiasAdd&forward_lstm_133/lstm_cell_400/add:z:0=forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_133/lstm_cell_400/BiasAddЂ
.forward_lstm_133/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.forward_lstm_133/lstm_cell_400/split/split_dimЛ
$forward_lstm_133/lstm_cell_400/splitSplit7forward_lstm_133/lstm_cell_400/split/split_dim:output:0/forward_lstm_133/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$forward_lstm_133/lstm_cell_400/splitМ
&forward_lstm_133/lstm_cell_400/SigmoidSigmoid-forward_lstm_133/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_133/lstm_cell_400/SigmoidР
(forward_lstm_133/lstm_cell_400/Sigmoid_1Sigmoid-forward_lstm_133/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_133/lstm_cell_400/Sigmoid_1в
"forward_lstm_133/lstm_cell_400/mulMul,forward_lstm_133/lstm_cell_400/Sigmoid_1:y:0!forward_lstm_133/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_133/lstm_cell_400/mulГ
#forward_lstm_133/lstm_cell_400/ReluRelu-forward_lstm_133/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_133/lstm_cell_400/Reluф
$forward_lstm_133/lstm_cell_400/mul_1Mul*forward_lstm_133/lstm_cell_400/Sigmoid:y:01forward_lstm_133/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_133/lstm_cell_400/mul_1й
$forward_lstm_133/lstm_cell_400/add_1AddV2&forward_lstm_133/lstm_cell_400/mul:z:0(forward_lstm_133/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_133/lstm_cell_400/add_1Р
(forward_lstm_133/lstm_cell_400/Sigmoid_2Sigmoid-forward_lstm_133/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_133/lstm_cell_400/Sigmoid_2В
%forward_lstm_133/lstm_cell_400/Relu_1Relu(forward_lstm_133/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_133/lstm_cell_400/Relu_1ш
$forward_lstm_133/lstm_cell_400/mul_2Mul,forward_lstm_133/lstm_cell_400/Sigmoid_2:y:03forward_lstm_133/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_133/lstm_cell_400/mul_2Б
.forward_lstm_133/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.forward_lstm_133/TensorArrayV2_1/element_shapeќ
 forward_lstm_133/TensorArrayV2_1TensorListReserve7forward_lstm_133/TensorArrayV2_1/element_shape:output:0)forward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_133/TensorArrayV2_1p
forward_lstm_133/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_133/timeЁ
)forward_lstm_133/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)forward_lstm_133/while/maximum_iterations
#forward_lstm_133/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_133/while/loop_counter
forward_lstm_133/whileWhile,forward_lstm_133/while/loop_counter:output:02forward_lstm_133/while/maximum_iterations:output:0forward_lstm_133/time:output:0)forward_lstm_133/TensorArrayV2_1:handle:0forward_lstm_133/zeros:output:0!forward_lstm_133/zeros_1:output:0)forward_lstm_133/strided_slice_1:output:0Hforward_lstm_133/TensorArrayUnstack/TensorListFromTensor:output_handle:0=forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource?forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource>forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *0
body(R&
$forward_lstm_133_while_body_16887763*0
cond(R&
$forward_lstm_133_while_cond_16887762*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
forward_lstm_133/whileз
Aforward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Aforward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3forward_lstm_133/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_133/while:output:3Jforward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3forward_lstm_133/TensorArrayV2Stack/TensorListStackЃ
&forward_lstm_133/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&forward_lstm_133/strided_slice_3/stack
(forward_lstm_133/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_133/strided_slice_3/stack_1
(forward_lstm_133/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_3/stack_2
 forward_lstm_133/strided_slice_3StridedSlice<forward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_133/strided_slice_3/stack:output:01forward_lstm_133/strided_slice_3/stack_1:output:01forward_lstm_133/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 forward_lstm_133/strided_slice_3
!forward_lstm_133/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_133/transpose_1/permђ
forward_lstm_133/transpose_1	Transpose<forward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_133/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_133/transpose_1
forward_lstm_133/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_133/runtimej
backward_lstm_133/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_133/Shape
%backward_lstm_133/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_133/strided_slice/stack
'backward_lstm_133/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_133/strided_slice/stack_1
'backward_lstm_133/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_133/strided_slice/stack_2Ю
backward_lstm_133/strided_sliceStridedSlice backward_lstm_133/Shape:output:0.backward_lstm_133/strided_slice/stack:output:00backward_lstm_133/strided_slice/stack_1:output:00backward_lstm_133/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_133/strided_slice
backward_lstm_133/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_133/zeros/mul/yД
backward_lstm_133/zeros/mulMul(backward_lstm_133/strided_slice:output:0&backward_lstm_133/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/zeros/mul
backward_lstm_133/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_133/zeros/Less/yЏ
backward_lstm_133/zeros/LessLessbackward_lstm_133/zeros/mul:z:0'backward_lstm_133/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/zeros/Less
 backward_lstm_133/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_133/zeros/packed/1Ы
backward_lstm_133/zeros/packedPack(backward_lstm_133/strided_slice:output:0)backward_lstm_133/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_133/zeros/packed
backward_lstm_133/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_133/zeros/ConstН
backward_lstm_133/zerosFill'backward_lstm_133/zeros/packed:output:0&backward_lstm_133/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_133/zeros
backward_lstm_133/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_133/zeros_1/mul/yК
backward_lstm_133/zeros_1/mulMul(backward_lstm_133/strided_slice:output:0(backward_lstm_133/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/zeros_1/mul
 backward_lstm_133/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 backward_lstm_133/zeros_1/Less/yЗ
backward_lstm_133/zeros_1/LessLess!backward_lstm_133/zeros_1/mul:z:0)backward_lstm_133/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_133/zeros_1/Less
"backward_lstm_133/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_133/zeros_1/packed/1б
 backward_lstm_133/zeros_1/packedPack(backward_lstm_133/strided_slice:output:0+backward_lstm_133/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_133/zeros_1/packed
backward_lstm_133/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_133/zeros_1/ConstХ
backward_lstm_133/zeros_1Fill)backward_lstm_133/zeros_1/packed:output:0(backward_lstm_133/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_133/zeros_1
 backward_lstm_133/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_133/transpose/permФ
backward_lstm_133/transpose	Transposeinputs_0)backward_lstm_133/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_133/transpose
backward_lstm_133/Shape_1Shapebackward_lstm_133/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_133/Shape_1
'backward_lstm_133/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_133/strided_slice_1/stack 
)backward_lstm_133/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_1/stack_1 
)backward_lstm_133/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_1/stack_2к
!backward_lstm_133/strided_slice_1StridedSlice"backward_lstm_133/Shape_1:output:00backward_lstm_133/strided_slice_1/stack:output:02backward_lstm_133/strided_slice_1/stack_1:output:02backward_lstm_133/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_133/strided_slice_1Љ
-backward_lstm_133/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2/
-backward_lstm_133/TensorArrayV2/element_shapeњ
backward_lstm_133/TensorArrayV2TensorListReserve6backward_lstm_133/TensorArrayV2/element_shape:output:0*backward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_133/TensorArrayV2
 backward_lstm_133/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_133/ReverseV2/axisл
backward_lstm_133/ReverseV2	ReverseV2backward_lstm_133/transpose:y:0)backward_lstm_133/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_133/ReverseV2у
Gbackward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2I
Gbackward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeХ
9backward_lstm_133/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_133/ReverseV2:output:0Pbackward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_133/TensorArrayUnstack/TensorListFromTensor
'backward_lstm_133/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_133/strided_slice_2/stack 
)backward_lstm_133/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_2/stack_1 
)backward_lstm_133/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_2/stack_2ё
!backward_lstm_133/strided_slice_2StridedSlicebackward_lstm_133/transpose:y:00backward_lstm_133/strided_slice_2/stack:output:02backward_lstm_133/strided_slice_2/stack_1:output:02backward_lstm_133/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2#
!backward_lstm_133/strided_slice_2ю
5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpReadVariableOp>backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype027
5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpј
&backward_lstm_133/lstm_cell_401/MatMulMatMul*backward_lstm_133/strided_slice_2:output:0=backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_133/lstm_cell_401/MatMulє
7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOp@backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype029
7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpє
(backward_lstm_133/lstm_cell_401/MatMul_1MatMul backward_lstm_133/zeros:output:0?backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_133/lstm_cell_401/MatMul_1ь
#backward_lstm_133/lstm_cell_401/addAddV20backward_lstm_133/lstm_cell_401/MatMul:product:02backward_lstm_133/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_133/lstm_cell_401/addэ
6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOp?backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype028
6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpљ
'backward_lstm_133/lstm_cell_401/BiasAddBiasAdd'backward_lstm_133/lstm_cell_401/add:z:0>backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_133/lstm_cell_401/BiasAddЄ
/backward_lstm_133/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/backward_lstm_133/lstm_cell_401/split/split_dimП
%backward_lstm_133/lstm_cell_401/splitSplit8backward_lstm_133/lstm_cell_401/split/split_dim:output:00backward_lstm_133/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2'
%backward_lstm_133/lstm_cell_401/splitП
'backward_lstm_133/lstm_cell_401/SigmoidSigmoid.backward_lstm_133/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_133/lstm_cell_401/SigmoidУ
)backward_lstm_133/lstm_cell_401/Sigmoid_1Sigmoid.backward_lstm_133/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_133/lstm_cell_401/Sigmoid_1ж
#backward_lstm_133/lstm_cell_401/mulMul-backward_lstm_133/lstm_cell_401/Sigmoid_1:y:0"backward_lstm_133/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_133/lstm_cell_401/mulЖ
$backward_lstm_133/lstm_cell_401/ReluRelu.backward_lstm_133/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_133/lstm_cell_401/Reluш
%backward_lstm_133/lstm_cell_401/mul_1Mul+backward_lstm_133/lstm_cell_401/Sigmoid:y:02backward_lstm_133/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_133/lstm_cell_401/mul_1н
%backward_lstm_133/lstm_cell_401/add_1AddV2'backward_lstm_133/lstm_cell_401/mul:z:0)backward_lstm_133/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_133/lstm_cell_401/add_1У
)backward_lstm_133/lstm_cell_401/Sigmoid_2Sigmoid.backward_lstm_133/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_133/lstm_cell_401/Sigmoid_2Е
&backward_lstm_133/lstm_cell_401/Relu_1Relu)backward_lstm_133/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_133/lstm_cell_401/Relu_1ь
%backward_lstm_133/lstm_cell_401/mul_2Mul-backward_lstm_133/lstm_cell_401/Sigmoid_2:y:04backward_lstm_133/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_133/lstm_cell_401/mul_2Г
/backward_lstm_133/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   21
/backward_lstm_133/TensorArrayV2_1/element_shape
!backward_lstm_133/TensorArrayV2_1TensorListReserve8backward_lstm_133/TensorArrayV2_1/element_shape:output:0*backward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_133/TensorArrayV2_1r
backward_lstm_133/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_133/timeЃ
*backward_lstm_133/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*backward_lstm_133/while/maximum_iterations
$backward_lstm_133/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_133/while/loop_counter 
backward_lstm_133/whileWhile-backward_lstm_133/while/loop_counter:output:03backward_lstm_133/while/maximum_iterations:output:0backward_lstm_133/time:output:0*backward_lstm_133/TensorArrayV2_1:handle:0 backward_lstm_133/zeros:output:0"backward_lstm_133/zeros_1:output:0*backward_lstm_133/strided_slice_1:output:0Ibackward_lstm_133/TensorArrayUnstack/TensorListFromTensor:output_handle:0>backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource@backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource?backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *1
body)R'
%backward_lstm_133_while_body_16887912*1
cond)R'
%backward_lstm_133_while_cond_16887911*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
backward_lstm_133/whileй
Bbackward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2D
Bbackward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeЙ
4backward_lstm_133/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_133/while:output:3Kbackward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype026
4backward_lstm_133/TensorArrayV2Stack/TensorListStackЅ
'backward_lstm_133/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2)
'backward_lstm_133/strided_slice_3/stack 
)backward_lstm_133/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_133/strided_slice_3/stack_1 
)backward_lstm_133/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_3/stack_2
!backward_lstm_133/strided_slice_3StridedSlice=backward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_133/strided_slice_3/stack:output:02backward_lstm_133/strided_slice_3/stack_1:output:02backward_lstm_133/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2#
!backward_lstm_133/strided_slice_3
"backward_lstm_133/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_133/transpose_1/permі
backward_lstm_133/transpose_1	Transpose=backward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_133/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_133/transpose_1
backward_lstm_133/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_133/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisФ
concatConcatV2)forward_lstm_133/strided_slice_3:output:0*backward_lstm_133/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

Identityд
NoOpNoOp7^backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp6^backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp8^backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp^backward_lstm_133/while6^forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp5^forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp7^forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp^forward_lstm_133/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2p
6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp2n
5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp2r
7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp22
backward_lstm_133/whilebackward_lstm_133/while2n
5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp2l
4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp2p
6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp20
forward_lstm_133/whileforward_lstm_133/while:g c
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
І_
­
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_16890038

inputs?
,lstm_cell_401_matmul_readvariableop_resource:	ШA
.lstm_cell_401_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_401_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_401/BiasAdd/ReadVariableOpЂ#lstm_cell_401/MatMul/ReadVariableOpЂ%lstm_cell_401/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_401/MatMul/ReadVariableOpReadVariableOp,lstm_cell_401_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_401/MatMul/ReadVariableOpА
lstm_cell_401/MatMulMatMulstrided_slice_2:output:0+lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/MatMulО
%lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_401_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_401/MatMul_1/ReadVariableOpЌ
lstm_cell_401/MatMul_1MatMulzeros:output:0-lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/MatMul_1Є
lstm_cell_401/addAddV2lstm_cell_401/MatMul:product:0 lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/addЗ
$lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_401_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_401/BiasAdd/ReadVariableOpБ
lstm_cell_401/BiasAddBiasAddlstm_cell_401/add:z:0,lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/BiasAdd
lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_401/split/split_dimї
lstm_cell_401/splitSplit&lstm_cell_401/split/split_dim:output:0lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_401/split
lstm_cell_401/SigmoidSigmoidlstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Sigmoid
lstm_cell_401/Sigmoid_1Sigmoidlstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Sigmoid_1
lstm_cell_401/mulMullstm_cell_401/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/mul
lstm_cell_401/ReluRelulstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Relu 
lstm_cell_401/mul_1Mullstm_cell_401/Sigmoid:y:0 lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/mul_1
lstm_cell_401/add_1AddV2lstm_cell_401/mul:z:0lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/add_1
lstm_cell_401/Sigmoid_2Sigmoidlstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Sigmoid_2
lstm_cell_401/Relu_1Relulstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Relu_1Є
lstm_cell_401/mul_2Mullstm_cell_401/Sigmoid_2:y:0"lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_401_matmul_readvariableop_resource.lstm_cell_401_matmul_1_readvariableop_resource-lstm_cell_401_biasadd_readvariableop_resource*
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
bodyR
while_body_16889954*
condR
while_cond_16889953*K
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
NoOpNoOp%^lstm_cell_401/BiasAdd/ReadVariableOp$^lstm_cell_401/MatMul/ReadVariableOp&^lstm_cell_401/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_401/BiasAdd/ReadVariableOp$lstm_cell_401/BiasAdd/ReadVariableOp2J
#lstm_cell_401/MatMul/ReadVariableOp#lstm_cell_401/MatMul/ReadVariableOp2N
%lstm_cell_401/MatMul_1/ReadVariableOp%lstm_cell_401/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
п
Э
while_cond_16889297
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16889297___redundant_placeholder06
2while_while_cond_16889297___redundant_placeholder16
2while_while_cond_16889297___redundant_placeholder26
2while_while_cond_16889297___redundant_placeholder3
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
д
Т
3__inference_forward_lstm_133_layer_call_fn_16888745
inputs_0
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
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
GPU 2J 8 *W
fRRP
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_168844262
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
ъ
М
%backward_lstm_133_while_cond_16888616@
<backward_lstm_133_while_backward_lstm_133_while_loop_counterF
Bbackward_lstm_133_while_backward_lstm_133_while_maximum_iterations'
#backward_lstm_133_while_placeholder)
%backward_lstm_133_while_placeholder_1)
%backward_lstm_133_while_placeholder_2)
%backward_lstm_133_while_placeholder_3)
%backward_lstm_133_while_placeholder_4B
>backward_lstm_133_while_less_backward_lstm_133_strided_slice_1Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16888616___redundant_placeholder0Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16888616___redundant_placeholder1Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16888616___redundant_placeholder2Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16888616___redundant_placeholder3Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16888616___redundant_placeholder4$
 backward_lstm_133_while_identity
Ъ
backward_lstm_133/while/LessLess#backward_lstm_133_while_placeholder>backward_lstm_133_while_less_backward_lstm_133_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_133/while/Less
 backward_lstm_133/while/IdentityIdentity backward_lstm_133/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_133/while/Identity"M
 backward_lstm_133_while_identity)backward_lstm_133/while/Identity:output:0*(
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
Щ
Ѕ
$forward_lstm_133_while_cond_16888079>
:forward_lstm_133_while_forward_lstm_133_while_loop_counterD
@forward_lstm_133_while_forward_lstm_133_while_maximum_iterations&
"forward_lstm_133_while_placeholder(
$forward_lstm_133_while_placeholder_1(
$forward_lstm_133_while_placeholder_2(
$forward_lstm_133_while_placeholder_3(
$forward_lstm_133_while_placeholder_4@
<forward_lstm_133_while_less_forward_lstm_133_strided_slice_1X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16888079___redundant_placeholder0X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16888079___redundant_placeholder1X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16888079___redundant_placeholder2X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16888079___redundant_placeholder3X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16888079___redundant_placeholder4#
forward_lstm_133_while_identity
Х
forward_lstm_133/while/LessLess"forward_lstm_133_while_placeholder<forward_lstm_133_while_less_forward_lstm_133_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_133/while/Less
forward_lstm_133/while/IdentityIdentityforward_lstm_133/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_133/while/Identity"K
forward_lstm_133_while_identity(forward_lstm_133/while/Identity:output:0*(
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
Щ
Ѕ
$forward_lstm_133_while_cond_16888437>
:forward_lstm_133_while_forward_lstm_133_while_loop_counterD
@forward_lstm_133_while_forward_lstm_133_while_maximum_iterations&
"forward_lstm_133_while_placeholder(
$forward_lstm_133_while_placeholder_1(
$forward_lstm_133_while_placeholder_2(
$forward_lstm_133_while_placeholder_3(
$forward_lstm_133_while_placeholder_4@
<forward_lstm_133_while_less_forward_lstm_133_strided_slice_1X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16888437___redundant_placeholder0X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16888437___redundant_placeholder1X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16888437___redundant_placeholder2X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16888437___redundant_placeholder3X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16888437___redundant_placeholder4#
forward_lstm_133_while_identity
Х
forward_lstm_133/while/LessLess"forward_lstm_133_while_placeholder<forward_lstm_133_while_less_forward_lstm_133_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_133/while/Less
forward_lstm_133/while/IdentityIdentityforward_lstm_133/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_133/while/Identity"K
forward_lstm_133_while_identity(forward_lstm_133/while/Identity:output:0*(
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
&
ј
while_body_16885201
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_401_16885225_0:	Ш1
while_lstm_cell_401_16885227_0:	2Ш-
while_lstm_cell_401_16885229_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_401_16885225:	Ш/
while_lstm_cell_401_16885227:	2Ш+
while_lstm_cell_401_16885229:	ШЂ+while/lstm_cell_401/StatefulPartitionedCallУ
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
)while/TensorArrayV2Read/TensorListGetItemя
+while/lstm_cell_401/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_401_16885225_0while_lstm_cell_401_16885227_0while_lstm_cell_401_16885229_0*
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
GPU 2J 8 *T
fORM
K__inference_lstm_cell_401_layer_call_and_return_conditional_losses_168851212-
+while/lstm_cell_401/StatefulPartitionedCallј
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_401/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_401/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѕ
while/Identity_5Identity4while/lstm_cell_401/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp,^while/lstm_cell_401/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_401_16885225while_lstm_cell_401_16885225_0">
while_lstm_cell_401_16885227while_lstm_cell_401_16885227_0">
while_lstm_cell_401_16885229while_lstm_cell_401_16885229_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2Z
+while/lstm_cell_401/StatefulPartitionedCall+while/lstm_cell_401/StatefulPartitionedCall: 
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
X
ћ
$forward_lstm_133_while_body_16887763>
:forward_lstm_133_while_forward_lstm_133_while_loop_counterD
@forward_lstm_133_while_forward_lstm_133_while_maximum_iterations&
"forward_lstm_133_while_placeholder(
$forward_lstm_133_while_placeholder_1(
$forward_lstm_133_while_placeholder_2(
$forward_lstm_133_while_placeholder_3=
9forward_lstm_133_while_forward_lstm_133_strided_slice_1_0y
uforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0X
Eforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0:	ШZ
Gforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0:	2ШU
Fforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0:	Ш#
forward_lstm_133_while_identity%
!forward_lstm_133_while_identity_1%
!forward_lstm_133_while_identity_2%
!forward_lstm_133_while_identity_3%
!forward_lstm_133_while_identity_4%
!forward_lstm_133_while_identity_5;
7forward_lstm_133_while_forward_lstm_133_strided_slice_1w
sforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensorV
Cforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource:	ШX
Eforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource:	2ШS
Dforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource:	ШЂ;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOpЂ:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOpЂ<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOpх
Hforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2J
Hforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeТ
:forward_lstm_133/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_133_while_placeholderQforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02<
:forward_lstm_133/while/TensorArrayV2Read/TensorListGetItemџ
:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOpReadVariableOpEforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp
+forward_lstm_133/while/lstm_cell_400/MatMulMatMulAforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem:item:0Bforward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_133/while/lstm_cell_400/MatMul
<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOpGforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp
-forward_lstm_133/while/lstm_cell_400/MatMul_1MatMul$forward_lstm_133_while_placeholder_2Dforward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-forward_lstm_133/while/lstm_cell_400/MatMul_1
(forward_lstm_133/while/lstm_cell_400/addAddV25forward_lstm_133/while/lstm_cell_400/MatMul:product:07forward_lstm_133/while/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_133/while/lstm_cell_400/addў
;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOpFforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp
,forward_lstm_133/while/lstm_cell_400/BiasAddBiasAdd,forward_lstm_133/while/lstm_cell_400/add:z:0Cforward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_133/while/lstm_cell_400/BiasAddЎ
4forward_lstm_133/while/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4forward_lstm_133/while/lstm_cell_400/split/split_dimг
*forward_lstm_133/while/lstm_cell_400/splitSplit=forward_lstm_133/while/lstm_cell_400/split/split_dim:output:05forward_lstm_133/while/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*forward_lstm_133/while/lstm_cell_400/splitЮ
,forward_lstm_133/while/lstm_cell_400/SigmoidSigmoid3forward_lstm_133/while/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,forward_lstm_133/while/lstm_cell_400/Sigmoidв
.forward_lstm_133/while/lstm_cell_400/Sigmoid_1Sigmoid3forward_lstm_133/while/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_133/while/lstm_cell_400/Sigmoid_1ч
(forward_lstm_133/while/lstm_cell_400/mulMul2forward_lstm_133/while/lstm_cell_400/Sigmoid_1:y:0$forward_lstm_133_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_133/while/lstm_cell_400/mulХ
)forward_lstm_133/while/lstm_cell_400/ReluRelu3forward_lstm_133/while/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_133/while/lstm_cell_400/Reluќ
*forward_lstm_133/while/lstm_cell_400/mul_1Mul0forward_lstm_133/while/lstm_cell_400/Sigmoid:y:07forward_lstm_133/while/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_133/while/lstm_cell_400/mul_1ё
*forward_lstm_133/while/lstm_cell_400/add_1AddV2,forward_lstm_133/while/lstm_cell_400/mul:z:0.forward_lstm_133/while/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_133/while/lstm_cell_400/add_1в
.forward_lstm_133/while/lstm_cell_400/Sigmoid_2Sigmoid3forward_lstm_133/while/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_133/while/lstm_cell_400/Sigmoid_2Ф
+forward_lstm_133/while/lstm_cell_400/Relu_1Relu.forward_lstm_133/while/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_133/while/lstm_cell_400/Relu_1
*forward_lstm_133/while/lstm_cell_400/mul_2Mul2forward_lstm_133/while/lstm_cell_400/Sigmoid_2:y:09forward_lstm_133/while/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_133/while/lstm_cell_400/mul_2Ж
;forward_lstm_133/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_133_while_placeholder_1"forward_lstm_133_while_placeholder.forward_lstm_133/while/lstm_cell_400/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_133/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_133/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_133/while/add/y­
forward_lstm_133/while/addAddV2"forward_lstm_133_while_placeholder%forward_lstm_133/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/while/add
forward_lstm_133/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_133/while/add_1/yЫ
forward_lstm_133/while/add_1AddV2:forward_lstm_133_while_forward_lstm_133_while_loop_counter'forward_lstm_133/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/while/add_1Џ
forward_lstm_133/while/IdentityIdentity forward_lstm_133/while/add_1:z:0^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_133/while/Identityг
!forward_lstm_133/while/Identity_1Identity@forward_lstm_133_while_forward_lstm_133_while_maximum_iterations^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_133/while/Identity_1Б
!forward_lstm_133/while/Identity_2Identityforward_lstm_133/while/add:z:0^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_133/while/Identity_2о
!forward_lstm_133/while/Identity_3IdentityKforward_lstm_133/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_133/while/Identity_3в
!forward_lstm_133/while/Identity_4Identity.forward_lstm_133/while/lstm_cell_400/mul_2:z:0^forward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_133/while/Identity_4в
!forward_lstm_133/while/Identity_5Identity.forward_lstm_133/while/lstm_cell_400/add_1:z:0^forward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_133/while/Identity_5Ж
forward_lstm_133/while/NoOpNoOp<^forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp;^forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp=^forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_133/while/NoOp"t
7forward_lstm_133_while_forward_lstm_133_strided_slice_19forward_lstm_133_while_forward_lstm_133_strided_slice_1_0"K
forward_lstm_133_while_identity(forward_lstm_133/while/Identity:output:0"O
!forward_lstm_133_while_identity_1*forward_lstm_133/while/Identity_1:output:0"O
!forward_lstm_133_while_identity_2*forward_lstm_133/while/Identity_2:output:0"O
!forward_lstm_133_while_identity_3*forward_lstm_133/while/Identity_3:output:0"O
!forward_lstm_133_while_identity_4*forward_lstm_133/while/Identity_4:output:0"O
!forward_lstm_133_while_identity_5*forward_lstm_133/while/Identity_5:output:0"
Dforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resourceFforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0"
Eforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resourceGforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0"
Cforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resourceEforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0"ь
sforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensoruforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2z
;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp2x
:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp2|
<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp: 
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
e
Т
$forward_lstm_133_while_body_16886428>
:forward_lstm_133_while_forward_lstm_133_while_loop_counterD
@forward_lstm_133_while_forward_lstm_133_while_maximum_iterations&
"forward_lstm_133_while_placeholder(
$forward_lstm_133_while_placeholder_1(
$forward_lstm_133_while_placeholder_2(
$forward_lstm_133_while_placeholder_3(
$forward_lstm_133_while_placeholder_4=
9forward_lstm_133_while_forward_lstm_133_strided_slice_1_0y
uforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_133_while_greater_forward_lstm_133_cast_0X
Eforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0:	ШZ
Gforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0:	2ШU
Fforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0:	Ш#
forward_lstm_133_while_identity%
!forward_lstm_133_while_identity_1%
!forward_lstm_133_while_identity_2%
!forward_lstm_133_while_identity_3%
!forward_lstm_133_while_identity_4%
!forward_lstm_133_while_identity_5%
!forward_lstm_133_while_identity_6;
7forward_lstm_133_while_forward_lstm_133_strided_slice_1w
sforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_133_while_greater_forward_lstm_133_castV
Cforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource:	ШX
Eforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource:	2ШS
Dforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource:	ШЂ;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOpЂ:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOpЂ<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOpх
Hforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2J
Hforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
:forward_lstm_133/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_133_while_placeholderQforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02<
:forward_lstm_133/while/TensorArrayV2Read/TensorListGetItemе
forward_lstm_133/while/GreaterGreater6forward_lstm_133_while_greater_forward_lstm_133_cast_0"forward_lstm_133_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2 
forward_lstm_133/while/Greaterџ
:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOpReadVariableOpEforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp
+forward_lstm_133/while/lstm_cell_400/MatMulMatMulAforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem:item:0Bforward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_133/while/lstm_cell_400/MatMul
<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOpGforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp
-forward_lstm_133/while/lstm_cell_400/MatMul_1MatMul$forward_lstm_133_while_placeholder_3Dforward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-forward_lstm_133/while/lstm_cell_400/MatMul_1
(forward_lstm_133/while/lstm_cell_400/addAddV25forward_lstm_133/while/lstm_cell_400/MatMul:product:07forward_lstm_133/while/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_133/while/lstm_cell_400/addў
;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOpFforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp
,forward_lstm_133/while/lstm_cell_400/BiasAddBiasAdd,forward_lstm_133/while/lstm_cell_400/add:z:0Cforward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_133/while/lstm_cell_400/BiasAddЎ
4forward_lstm_133/while/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4forward_lstm_133/while/lstm_cell_400/split/split_dimг
*forward_lstm_133/while/lstm_cell_400/splitSplit=forward_lstm_133/while/lstm_cell_400/split/split_dim:output:05forward_lstm_133/while/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*forward_lstm_133/while/lstm_cell_400/splitЮ
,forward_lstm_133/while/lstm_cell_400/SigmoidSigmoid3forward_lstm_133/while/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,forward_lstm_133/while/lstm_cell_400/Sigmoidв
.forward_lstm_133/while/lstm_cell_400/Sigmoid_1Sigmoid3forward_lstm_133/while/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_133/while/lstm_cell_400/Sigmoid_1ч
(forward_lstm_133/while/lstm_cell_400/mulMul2forward_lstm_133/while/lstm_cell_400/Sigmoid_1:y:0$forward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_133/while/lstm_cell_400/mulХ
)forward_lstm_133/while/lstm_cell_400/ReluRelu3forward_lstm_133/while/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_133/while/lstm_cell_400/Reluќ
*forward_lstm_133/while/lstm_cell_400/mul_1Mul0forward_lstm_133/while/lstm_cell_400/Sigmoid:y:07forward_lstm_133/while/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_133/while/lstm_cell_400/mul_1ё
*forward_lstm_133/while/lstm_cell_400/add_1AddV2,forward_lstm_133/while/lstm_cell_400/mul:z:0.forward_lstm_133/while/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_133/while/lstm_cell_400/add_1в
.forward_lstm_133/while/lstm_cell_400/Sigmoid_2Sigmoid3forward_lstm_133/while/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_133/while/lstm_cell_400/Sigmoid_2Ф
+forward_lstm_133/while/lstm_cell_400/Relu_1Relu.forward_lstm_133/while/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_133/while/lstm_cell_400/Relu_1
*forward_lstm_133/while/lstm_cell_400/mul_2Mul2forward_lstm_133/while/lstm_cell_400/Sigmoid_2:y:09forward_lstm_133/while/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_133/while/lstm_cell_400/mul_2є
forward_lstm_133/while/SelectSelect"forward_lstm_133/while/Greater:z:0.forward_lstm_133/while/lstm_cell_400/mul_2:z:0$forward_lstm_133_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/while/Selectј
forward_lstm_133/while/Select_1Select"forward_lstm_133/while/Greater:z:0.forward_lstm_133/while/lstm_cell_400/mul_2:z:0$forward_lstm_133_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_133/while/Select_1ј
forward_lstm_133/while/Select_2Select"forward_lstm_133/while/Greater:z:0.forward_lstm_133/while/lstm_cell_400/add_1:z:0$forward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_133/while/Select_2Ў
;forward_lstm_133/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_133_while_placeholder_1"forward_lstm_133_while_placeholder&forward_lstm_133/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_133/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_133/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_133/while/add/y­
forward_lstm_133/while/addAddV2"forward_lstm_133_while_placeholder%forward_lstm_133/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/while/add
forward_lstm_133/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_133/while/add_1/yЫ
forward_lstm_133/while/add_1AddV2:forward_lstm_133_while_forward_lstm_133_while_loop_counter'forward_lstm_133/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/while/add_1Џ
forward_lstm_133/while/IdentityIdentity forward_lstm_133/while/add_1:z:0^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_133/while/Identityг
!forward_lstm_133/while/Identity_1Identity@forward_lstm_133_while_forward_lstm_133_while_maximum_iterations^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_133/while/Identity_1Б
!forward_lstm_133/while/Identity_2Identityforward_lstm_133/while/add:z:0^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_133/while/Identity_2о
!forward_lstm_133/while/Identity_3IdentityKforward_lstm_133/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_133/while/Identity_3Ъ
!forward_lstm_133/while/Identity_4Identity&forward_lstm_133/while/Select:output:0^forward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_133/while/Identity_4Ь
!forward_lstm_133/while/Identity_5Identity(forward_lstm_133/while/Select_1:output:0^forward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_133/while/Identity_5Ь
!forward_lstm_133/while/Identity_6Identity(forward_lstm_133/while/Select_2:output:0^forward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_133/while/Identity_6Ж
forward_lstm_133/while/NoOpNoOp<^forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp;^forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp=^forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_133/while/NoOp"t
7forward_lstm_133_while_forward_lstm_133_strided_slice_19forward_lstm_133_while_forward_lstm_133_strided_slice_1_0"n
4forward_lstm_133_while_greater_forward_lstm_133_cast6forward_lstm_133_while_greater_forward_lstm_133_cast_0"K
forward_lstm_133_while_identity(forward_lstm_133/while/Identity:output:0"O
!forward_lstm_133_while_identity_1*forward_lstm_133/while/Identity_1:output:0"O
!forward_lstm_133_while_identity_2*forward_lstm_133/while/Identity_2:output:0"O
!forward_lstm_133_while_identity_3*forward_lstm_133/while/Identity_3:output:0"O
!forward_lstm_133_while_identity_4*forward_lstm_133/while/Identity_4:output:0"O
!forward_lstm_133_while_identity_5*forward_lstm_133/while/Identity_5:output:0"O
!forward_lstm_133_while_identity_6*forward_lstm_133/while/Identity_6:output:0"
Dforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resourceFforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0"
Eforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resourceGforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0"
Cforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resourceEforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0"ь
sforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensoruforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2z
;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp2x
:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp2|
<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp: 
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
У
Џ
Esequential_133_bidirectional_133_forward_lstm_133_while_body_16883985
|sequential_133_bidirectional_133_forward_lstm_133_while_sequential_133_bidirectional_133_forward_lstm_133_while_loop_counter
sequential_133_bidirectional_133_forward_lstm_133_while_sequential_133_bidirectional_133_forward_lstm_133_while_maximum_iterationsG
Csequential_133_bidirectional_133_forward_lstm_133_while_placeholderI
Esequential_133_bidirectional_133_forward_lstm_133_while_placeholder_1I
Esequential_133_bidirectional_133_forward_lstm_133_while_placeholder_2I
Esequential_133_bidirectional_133_forward_lstm_133_while_placeholder_3I
Esequential_133_bidirectional_133_forward_lstm_133_while_placeholder_4
{sequential_133_bidirectional_133_forward_lstm_133_while_sequential_133_bidirectional_133_forward_lstm_133_strided_slice_1_0М
Зsequential_133_bidirectional_133_forward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_sequential_133_bidirectional_133_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0|
xsequential_133_bidirectional_133_forward_lstm_133_while_greater_sequential_133_bidirectional_133_forward_lstm_133_cast_0y
fsequential_133_bidirectional_133_forward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0:	Ш{
hsequential_133_bidirectional_133_forward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0:	2Шv
gsequential_133_bidirectional_133_forward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0:	ШD
@sequential_133_bidirectional_133_forward_lstm_133_while_identityF
Bsequential_133_bidirectional_133_forward_lstm_133_while_identity_1F
Bsequential_133_bidirectional_133_forward_lstm_133_while_identity_2F
Bsequential_133_bidirectional_133_forward_lstm_133_while_identity_3F
Bsequential_133_bidirectional_133_forward_lstm_133_while_identity_4F
Bsequential_133_bidirectional_133_forward_lstm_133_while_identity_5F
Bsequential_133_bidirectional_133_forward_lstm_133_while_identity_6}
ysequential_133_bidirectional_133_forward_lstm_133_while_sequential_133_bidirectional_133_forward_lstm_133_strided_slice_1К
Еsequential_133_bidirectional_133_forward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_sequential_133_bidirectional_133_forward_lstm_133_tensorarrayunstack_tensorlistfromtensorz
vsequential_133_bidirectional_133_forward_lstm_133_while_greater_sequential_133_bidirectional_133_forward_lstm_133_castw
dsequential_133_bidirectional_133_forward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource:	Шy
fsequential_133_bidirectional_133_forward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource:	2Шt
esequential_133_bidirectional_133_forward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource:	ШЂ\sequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOpЂ[sequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOpЂ]sequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOpЇ
isequential_133/bidirectional_133/forward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2k
isequential_133/bidirectional_133/forward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shape
[sequential_133/bidirectional_133/forward_lstm_133/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЗsequential_133_bidirectional_133_forward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_sequential_133_bidirectional_133_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0Csequential_133_bidirectional_133_forward_lstm_133_while_placeholderrsequential_133/bidirectional_133/forward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02]
[sequential_133/bidirectional_133/forward_lstm_133/while/TensorArrayV2Read/TensorListGetItemњ
?sequential_133/bidirectional_133/forward_lstm_133/while/GreaterGreaterxsequential_133_bidirectional_133_forward_lstm_133_while_greater_sequential_133_bidirectional_133_forward_lstm_133_cast_0Csequential_133_bidirectional_133_forward_lstm_133_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2A
?sequential_133/bidirectional_133/forward_lstm_133/while/Greaterт
[sequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOpReadVariableOpfsequential_133_bidirectional_133_forward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02]
[sequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOpЂ
Lsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMulMatMulbsequential_133/bidirectional_133/forward_lstm_133/while/TensorArrayV2Read/TensorListGetItem:item:0csequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2N
Lsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMulш
]sequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOphsequential_133_bidirectional_133_forward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02_
]sequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp
Nsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMul_1MatMulEsequential_133_bidirectional_133_forward_lstm_133_while_placeholder_3esequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2P
Nsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMul_1
Isequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/addAddV2Vsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMul:product:0Xsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2K
Isequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/addс
\sequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOpgsequential_133_bidirectional_133_forward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02^
\sequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp
Msequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/BiasAddBiasAddMsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/add:z:0dsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2O
Msequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/BiasAdd№
Usequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2W
Usequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/split/split_dimз
Ksequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/splitSplit^sequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/split/split_dim:output:0Vsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2M
Ksequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/splitБ
Msequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/SigmoidSigmoidTsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22O
Msequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/SigmoidЕ
Osequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/Sigmoid_1SigmoidTsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22Q
Osequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/Sigmoid_1ы
Isequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/mulMulSsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/Sigmoid_1:y:0Esequential_133_bidirectional_133_forward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22K
Isequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/mulЈ
Jsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/ReluReluTsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22L
Jsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/Relu
Ksequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/mul_1MulQsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/Sigmoid:y:0Xsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22M
Ksequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/mul_1ѕ
Ksequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/add_1AddV2Msequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/mul:z:0Osequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22M
Ksequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/add_1Е
Osequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/Sigmoid_2SigmoidTsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22Q
Osequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/Sigmoid_2Ї
Lsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/Relu_1ReluOsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22N
Lsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/Relu_1
Ksequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/mul_2MulSsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/Sigmoid_2:y:0Zsequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22M
Ksequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/mul_2
>sequential_133/bidirectional_133/forward_lstm_133/while/SelectSelectCsequential_133/bidirectional_133/forward_lstm_133/while/Greater:z:0Osequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/mul_2:z:0Esequential_133_bidirectional_133_forward_lstm_133_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22@
>sequential_133/bidirectional_133/forward_lstm_133/while/Select
@sequential_133/bidirectional_133/forward_lstm_133/while/Select_1SelectCsequential_133/bidirectional_133/forward_lstm_133/while/Greater:z:0Osequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/mul_2:z:0Esequential_133_bidirectional_133_forward_lstm_133_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22B
@sequential_133/bidirectional_133/forward_lstm_133/while/Select_1
@sequential_133/bidirectional_133/forward_lstm_133/while/Select_2SelectCsequential_133/bidirectional_133/forward_lstm_133/while/Greater:z:0Osequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/add_1:z:0Esequential_133_bidirectional_133_forward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22B
@sequential_133/bidirectional_133/forward_lstm_133/while/Select_2г
\sequential_133/bidirectional_133/forward_lstm_133/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemEsequential_133_bidirectional_133_forward_lstm_133_while_placeholder_1Csequential_133_bidirectional_133_forward_lstm_133_while_placeholderGsequential_133/bidirectional_133/forward_lstm_133/while/Select:output:0*
_output_shapes
: *
element_dtype02^
\sequential_133/bidirectional_133/forward_lstm_133/while/TensorArrayV2Write/TensorListSetItemР
=sequential_133/bidirectional_133/forward_lstm_133/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2?
=sequential_133/bidirectional_133/forward_lstm_133/while/add/yБ
;sequential_133/bidirectional_133/forward_lstm_133/while/addAddV2Csequential_133_bidirectional_133_forward_lstm_133_while_placeholderFsequential_133/bidirectional_133/forward_lstm_133/while/add/y:output:0*
T0*
_output_shapes
: 2=
;sequential_133/bidirectional_133/forward_lstm_133/while/addФ
?sequential_133/bidirectional_133/forward_lstm_133/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2A
?sequential_133/bidirectional_133/forward_lstm_133/while/add_1/y№
=sequential_133/bidirectional_133/forward_lstm_133/while/add_1AddV2|sequential_133_bidirectional_133_forward_lstm_133_while_sequential_133_bidirectional_133_forward_lstm_133_while_loop_counterHsequential_133/bidirectional_133/forward_lstm_133/while/add_1/y:output:0*
T0*
_output_shapes
: 2?
=sequential_133/bidirectional_133/forward_lstm_133/while/add_1Г
@sequential_133/bidirectional_133/forward_lstm_133/while/IdentityIdentityAsequential_133/bidirectional_133/forward_lstm_133/while/add_1:z:0=^sequential_133/bidirectional_133/forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2B
@sequential_133/bidirectional_133/forward_lstm_133/while/Identityљ
Bsequential_133/bidirectional_133/forward_lstm_133/while/Identity_1Identitysequential_133_bidirectional_133_forward_lstm_133_while_sequential_133_bidirectional_133_forward_lstm_133_while_maximum_iterations=^sequential_133/bidirectional_133/forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2D
Bsequential_133/bidirectional_133/forward_lstm_133/while/Identity_1Е
Bsequential_133/bidirectional_133/forward_lstm_133/while/Identity_2Identity?sequential_133/bidirectional_133/forward_lstm_133/while/add:z:0=^sequential_133/bidirectional_133/forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2D
Bsequential_133/bidirectional_133/forward_lstm_133/while/Identity_2т
Bsequential_133/bidirectional_133/forward_lstm_133/while/Identity_3Identitylsequential_133/bidirectional_133/forward_lstm_133/while/TensorArrayV2Write/TensorListSetItem:output_handle:0=^sequential_133/bidirectional_133/forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2D
Bsequential_133/bidirectional_133/forward_lstm_133/while/Identity_3Ю
Bsequential_133/bidirectional_133/forward_lstm_133/while/Identity_4IdentityGsequential_133/bidirectional_133/forward_lstm_133/while/Select:output:0=^sequential_133/bidirectional_133/forward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22D
Bsequential_133/bidirectional_133/forward_lstm_133/while/Identity_4а
Bsequential_133/bidirectional_133/forward_lstm_133/while/Identity_5IdentityIsequential_133/bidirectional_133/forward_lstm_133/while/Select_1:output:0=^sequential_133/bidirectional_133/forward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22D
Bsequential_133/bidirectional_133/forward_lstm_133/while/Identity_5а
Bsequential_133/bidirectional_133/forward_lstm_133/while/Identity_6IdentityIsequential_133/bidirectional_133/forward_lstm_133/while/Select_2:output:0=^sequential_133/bidirectional_133/forward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22D
Bsequential_133/bidirectional_133/forward_lstm_133/while/Identity_6л
<sequential_133/bidirectional_133/forward_lstm_133/while/NoOpNoOp]^sequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp\^sequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp^^sequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2>
<sequential_133/bidirectional_133/forward_lstm_133/while/NoOp"ђ
vsequential_133_bidirectional_133_forward_lstm_133_while_greater_sequential_133_bidirectional_133_forward_lstm_133_castxsequential_133_bidirectional_133_forward_lstm_133_while_greater_sequential_133_bidirectional_133_forward_lstm_133_cast_0"
@sequential_133_bidirectional_133_forward_lstm_133_while_identityIsequential_133/bidirectional_133/forward_lstm_133/while/Identity:output:0"
Bsequential_133_bidirectional_133_forward_lstm_133_while_identity_1Ksequential_133/bidirectional_133/forward_lstm_133/while/Identity_1:output:0"
Bsequential_133_bidirectional_133_forward_lstm_133_while_identity_2Ksequential_133/bidirectional_133/forward_lstm_133/while/Identity_2:output:0"
Bsequential_133_bidirectional_133_forward_lstm_133_while_identity_3Ksequential_133/bidirectional_133/forward_lstm_133/while/Identity_3:output:0"
Bsequential_133_bidirectional_133_forward_lstm_133_while_identity_4Ksequential_133/bidirectional_133/forward_lstm_133/while/Identity_4:output:0"
Bsequential_133_bidirectional_133_forward_lstm_133_while_identity_5Ksequential_133/bidirectional_133/forward_lstm_133/while/Identity_5:output:0"
Bsequential_133_bidirectional_133_forward_lstm_133_while_identity_6Ksequential_133/bidirectional_133/forward_lstm_133/while/Identity_6:output:0"а
esequential_133_bidirectional_133_forward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resourcegsequential_133_bidirectional_133_forward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0"в
fsequential_133_bidirectional_133_forward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resourcehsequential_133_bidirectional_133_forward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0"Ю
dsequential_133_bidirectional_133_forward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resourcefsequential_133_bidirectional_133_forward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0"ј
ysequential_133_bidirectional_133_forward_lstm_133_while_sequential_133_bidirectional_133_forward_lstm_133_strided_slice_1{sequential_133_bidirectional_133_forward_lstm_133_while_sequential_133_bidirectional_133_forward_lstm_133_strided_slice_1_0"ђ
Еsequential_133_bidirectional_133_forward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_sequential_133_bidirectional_133_forward_lstm_133_tensorarrayunstack_tensorlistfromtensorЗsequential_133_bidirectional_133_forward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_sequential_133_bidirectional_133_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2М
\sequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp\sequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp2К
[sequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp[sequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp2О
]sequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp]sequential_133/bidirectional_133/forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp: 
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

ј
G__inference_dense_133_layer_call_and_return_conditional_losses_16886729

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
п
Ё
$forward_lstm_133_while_cond_16887762>
:forward_lstm_133_while_forward_lstm_133_while_loop_counterD
@forward_lstm_133_while_forward_lstm_133_while_maximum_iterations&
"forward_lstm_133_while_placeholder(
$forward_lstm_133_while_placeholder_1(
$forward_lstm_133_while_placeholder_2(
$forward_lstm_133_while_placeholder_3@
<forward_lstm_133_while_less_forward_lstm_133_strided_slice_1X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16887762___redundant_placeholder0X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16887762___redundant_placeholder1X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16887762___redundant_placeholder2X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16887762___redundant_placeholder3#
forward_lstm_133_while_identity
Х
forward_lstm_133/while/LessLess"forward_lstm_133_while_placeholder<forward_lstm_133_while_less_forward_lstm_133_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_133/while/Less
forward_lstm_133/while/IdentityIdentityforward_lstm_133/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_133/while/Identity"K
forward_lstm_133_while_identity(forward_lstm_133/while/Identity:output:0*(
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

д
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_16886266

inputs,
forward_lstm_133_16886249:	Ш,
forward_lstm_133_16886251:	2Ш(
forward_lstm_133_16886253:	Ш-
backward_lstm_133_16886256:	Ш-
backward_lstm_133_16886258:	2Ш)
backward_lstm_133_16886260:	Ш
identityЂ)backward_lstm_133/StatefulPartitionedCallЂ(forward_lstm_133/StatefulPartitionedCallп
(forward_lstm_133/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_133_16886249forward_lstm_133_16886251forward_lstm_133_16886253*
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
GPU 2J 8 *W
fRRP
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_168862182*
(forward_lstm_133/StatefulPartitionedCallх
)backward_lstm_133/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_133_16886256backward_lstm_133_16886258backward_lstm_133_16886260*
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
GPU 2J 8 *X
fSRQ
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_168860452+
)backward_lstm_133/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisд
concatConcatV21forward_lstm_133/StatefulPartitionedCall:output:02backward_lstm_133/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

IdentityЅ
NoOpNoOp*^backward_lstm_133/StatefulPartitionedCall)^forward_lstm_133/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2V
)backward_lstm_133/StatefulPartitionedCall)backward_lstm_133/StatefulPartitionedCall2T
(forward_lstm_133/StatefulPartitionedCall(forward_lstm_133/StatefulPartitionedCall:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
еH

O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_16885058

inputs)
lstm_cell_401_16884976:	Ш)
lstm_cell_401_16884978:	2Ш%
lstm_cell_401_16884980:	Ш
identityЂ%lstm_cell_401/StatefulPartitionedCallЂwhileD
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
strided_slice_2Ћ
%lstm_cell_401/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_401_16884976lstm_cell_401_16884978lstm_cell_401_16884980*
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
GPU 2J 8 *T
fORM
K__inference_lstm_cell_401_layer_call_and_return_conditional_losses_168849752'
%lstm_cell_401/StatefulPartitionedCall
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
while/loop_counterЭ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_401_16884976lstm_cell_401_16884978lstm_cell_401_16884980*
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
bodyR
while_body_16884989*
condR
while_cond_16884988*K
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
NoOpNoOp&^lstm_cell_401/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2N
%lstm_cell_401/StatefulPartitionedCall%lstm_cell_401/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
И
Ѓ
L__inference_sequential_133_layer_call_and_return_conditional_losses_16886736

inputs
inputs_1	-
bidirectional_133_16886705:	Ш-
bidirectional_133_16886707:	2Ш)
bidirectional_133_16886709:	Ш-
bidirectional_133_16886711:	Ш-
bidirectional_133_16886713:	2Ш)
bidirectional_133_16886715:	Ш$
dense_133_16886730:d 
dense_133_16886732:
identityЂ)bidirectional_133/StatefulPartitionedCallЂ!dense_133/StatefulPartitionedCallЪ
)bidirectional_133/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_133_16886705bidirectional_133_16886707bidirectional_133_16886709bidirectional_133_16886711bidirectional_133_16886713bidirectional_133_16886715*
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
GPU 2J 8 *X
fSRQ
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_168867042+
)bidirectional_133/StatefulPartitionedCallЫ
!dense_133/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_133/StatefulPartitionedCall:output:0dense_133_16886730dense_133_16886732*
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
GPU 2J 8 *P
fKRI
G__inference_dense_133_layer_call_and_return_conditional_losses_168867292#
!dense_133/StatefulPartitionedCall
IdentityIdentity*dense_133/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp*^bidirectional_133/StatefulPartitionedCall"^dense_133/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2V
)bidirectional_133/StatefulPartitionedCall)bidirectional_133/StatefulPartitionedCall2F
!dense_133/StatefulPartitionedCall!dense_133/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
т
С
4__inference_backward_lstm_133_layer_call_fn_16889426

inputs
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
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
GPU 2J 8 *X
fSRQ
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_168860452
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
І_
­
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_16885853

inputs?
,lstm_cell_401_matmul_readvariableop_resource:	ШA
.lstm_cell_401_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_401_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_401/BiasAdd/ReadVariableOpЂ#lstm_cell_401/MatMul/ReadVariableOpЂ%lstm_cell_401/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_401/MatMul/ReadVariableOpReadVariableOp,lstm_cell_401_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_401/MatMul/ReadVariableOpА
lstm_cell_401/MatMulMatMulstrided_slice_2:output:0+lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/MatMulО
%lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_401_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_401/MatMul_1/ReadVariableOpЌ
lstm_cell_401/MatMul_1MatMulzeros:output:0-lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/MatMul_1Є
lstm_cell_401/addAddV2lstm_cell_401/MatMul:product:0 lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/addЗ
$lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_401_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_401/BiasAdd/ReadVariableOpБ
lstm_cell_401/BiasAddBiasAddlstm_cell_401/add:z:0,lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/BiasAdd
lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_401/split/split_dimї
lstm_cell_401/splitSplit&lstm_cell_401/split/split_dim:output:0lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_401/split
lstm_cell_401/SigmoidSigmoidlstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Sigmoid
lstm_cell_401/Sigmoid_1Sigmoidlstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Sigmoid_1
lstm_cell_401/mulMullstm_cell_401/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/mul
lstm_cell_401/ReluRelulstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Relu 
lstm_cell_401/mul_1Mullstm_cell_401/Sigmoid:y:0 lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/mul_1
lstm_cell_401/add_1AddV2lstm_cell_401/mul:z:0lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/add_1
lstm_cell_401/Sigmoid_2Sigmoidlstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Sigmoid_2
lstm_cell_401/Relu_1Relulstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Relu_1Є
lstm_cell_401/mul_2Mullstm_cell_401/Sigmoid_2:y:0"lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_401_matmul_readvariableop_resource.lstm_cell_401_matmul_1_readvariableop_resource-lstm_cell_401_biasadd_readvariableop_resource*
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
bodyR
while_body_16885769*
condR
while_cond_16885768*K
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
NoOpNoOp%^lstm_cell_401/BiasAdd/ReadVariableOp$^lstm_cell_401/MatMul/ReadVariableOp&^lstm_cell_401/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_401/BiasAdd/ReadVariableOp$lstm_cell_401/BiasAdd/ReadVariableOp2J
#lstm_cell_401/MatMul/ReadVariableOp#lstm_cell_401/MatMul/ReadVariableOp2N
%lstm_cell_401/MatMul_1/ReadVariableOp%lstm_cell_401/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Щ
Ѕ
$forward_lstm_133_while_cond_16886427>
:forward_lstm_133_while_forward_lstm_133_while_loop_counterD
@forward_lstm_133_while_forward_lstm_133_while_maximum_iterations&
"forward_lstm_133_while_placeholder(
$forward_lstm_133_while_placeholder_1(
$forward_lstm_133_while_placeholder_2(
$forward_lstm_133_while_placeholder_3(
$forward_lstm_133_while_placeholder_4@
<forward_lstm_133_while_less_forward_lstm_133_strided_slice_1X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16886427___redundant_placeholder0X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16886427___redundant_placeholder1X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16886427___redundant_placeholder2X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16886427___redundant_placeholder3X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16886427___redundant_placeholder4#
forward_lstm_133_while_identity
Х
forward_lstm_133/while/LessLess"forward_lstm_133_while_placeholder<forward_lstm_133_while_less_forward_lstm_133_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_133/while/Less
forward_lstm_133/while/IdentityIdentityforward_lstm_133/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_133/while/Identity"K
forward_lstm_133_while_identity(forward_lstm_133/while/Identity:output:0*(
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
e
Т
$forward_lstm_133_while_body_16888438>
:forward_lstm_133_while_forward_lstm_133_while_loop_counterD
@forward_lstm_133_while_forward_lstm_133_while_maximum_iterations&
"forward_lstm_133_while_placeholder(
$forward_lstm_133_while_placeholder_1(
$forward_lstm_133_while_placeholder_2(
$forward_lstm_133_while_placeholder_3(
$forward_lstm_133_while_placeholder_4=
9forward_lstm_133_while_forward_lstm_133_strided_slice_1_0y
uforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_133_while_greater_forward_lstm_133_cast_0X
Eforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0:	ШZ
Gforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0:	2ШU
Fforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0:	Ш#
forward_lstm_133_while_identity%
!forward_lstm_133_while_identity_1%
!forward_lstm_133_while_identity_2%
!forward_lstm_133_while_identity_3%
!forward_lstm_133_while_identity_4%
!forward_lstm_133_while_identity_5%
!forward_lstm_133_while_identity_6;
7forward_lstm_133_while_forward_lstm_133_strided_slice_1w
sforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_133_while_greater_forward_lstm_133_castV
Cforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource:	ШX
Eforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource:	2ШS
Dforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource:	ШЂ;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOpЂ:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOpЂ<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOpх
Hforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2J
Hforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
:forward_lstm_133/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_133_while_placeholderQforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02<
:forward_lstm_133/while/TensorArrayV2Read/TensorListGetItemе
forward_lstm_133/while/GreaterGreater6forward_lstm_133_while_greater_forward_lstm_133_cast_0"forward_lstm_133_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2 
forward_lstm_133/while/Greaterџ
:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOpReadVariableOpEforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp
+forward_lstm_133/while/lstm_cell_400/MatMulMatMulAforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem:item:0Bforward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_133/while/lstm_cell_400/MatMul
<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOpGforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp
-forward_lstm_133/while/lstm_cell_400/MatMul_1MatMul$forward_lstm_133_while_placeholder_3Dforward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-forward_lstm_133/while/lstm_cell_400/MatMul_1
(forward_lstm_133/while/lstm_cell_400/addAddV25forward_lstm_133/while/lstm_cell_400/MatMul:product:07forward_lstm_133/while/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_133/while/lstm_cell_400/addў
;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOpFforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp
,forward_lstm_133/while/lstm_cell_400/BiasAddBiasAdd,forward_lstm_133/while/lstm_cell_400/add:z:0Cforward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_133/while/lstm_cell_400/BiasAddЎ
4forward_lstm_133/while/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4forward_lstm_133/while/lstm_cell_400/split/split_dimг
*forward_lstm_133/while/lstm_cell_400/splitSplit=forward_lstm_133/while/lstm_cell_400/split/split_dim:output:05forward_lstm_133/while/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*forward_lstm_133/while/lstm_cell_400/splitЮ
,forward_lstm_133/while/lstm_cell_400/SigmoidSigmoid3forward_lstm_133/while/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,forward_lstm_133/while/lstm_cell_400/Sigmoidв
.forward_lstm_133/while/lstm_cell_400/Sigmoid_1Sigmoid3forward_lstm_133/while/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_133/while/lstm_cell_400/Sigmoid_1ч
(forward_lstm_133/while/lstm_cell_400/mulMul2forward_lstm_133/while/lstm_cell_400/Sigmoid_1:y:0$forward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_133/while/lstm_cell_400/mulХ
)forward_lstm_133/while/lstm_cell_400/ReluRelu3forward_lstm_133/while/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_133/while/lstm_cell_400/Reluќ
*forward_lstm_133/while/lstm_cell_400/mul_1Mul0forward_lstm_133/while/lstm_cell_400/Sigmoid:y:07forward_lstm_133/while/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_133/while/lstm_cell_400/mul_1ё
*forward_lstm_133/while/lstm_cell_400/add_1AddV2,forward_lstm_133/while/lstm_cell_400/mul:z:0.forward_lstm_133/while/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_133/while/lstm_cell_400/add_1в
.forward_lstm_133/while/lstm_cell_400/Sigmoid_2Sigmoid3forward_lstm_133/while/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_133/while/lstm_cell_400/Sigmoid_2Ф
+forward_lstm_133/while/lstm_cell_400/Relu_1Relu.forward_lstm_133/while/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_133/while/lstm_cell_400/Relu_1
*forward_lstm_133/while/lstm_cell_400/mul_2Mul2forward_lstm_133/while/lstm_cell_400/Sigmoid_2:y:09forward_lstm_133/while/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_133/while/lstm_cell_400/mul_2є
forward_lstm_133/while/SelectSelect"forward_lstm_133/while/Greater:z:0.forward_lstm_133/while/lstm_cell_400/mul_2:z:0$forward_lstm_133_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/while/Selectј
forward_lstm_133/while/Select_1Select"forward_lstm_133/while/Greater:z:0.forward_lstm_133/while/lstm_cell_400/mul_2:z:0$forward_lstm_133_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_133/while/Select_1ј
forward_lstm_133/while/Select_2Select"forward_lstm_133/while/Greater:z:0.forward_lstm_133/while/lstm_cell_400/add_1:z:0$forward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_133/while/Select_2Ў
;forward_lstm_133/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_133_while_placeholder_1"forward_lstm_133_while_placeholder&forward_lstm_133/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_133/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_133/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_133/while/add/y­
forward_lstm_133/while/addAddV2"forward_lstm_133_while_placeholder%forward_lstm_133/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/while/add
forward_lstm_133/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_133/while/add_1/yЫ
forward_lstm_133/while/add_1AddV2:forward_lstm_133_while_forward_lstm_133_while_loop_counter'forward_lstm_133/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/while/add_1Џ
forward_lstm_133/while/IdentityIdentity forward_lstm_133/while/add_1:z:0^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_133/while/Identityг
!forward_lstm_133/while/Identity_1Identity@forward_lstm_133_while_forward_lstm_133_while_maximum_iterations^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_133/while/Identity_1Б
!forward_lstm_133/while/Identity_2Identityforward_lstm_133/while/add:z:0^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_133/while/Identity_2о
!forward_lstm_133/while/Identity_3IdentityKforward_lstm_133/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_133/while/Identity_3Ъ
!forward_lstm_133/while/Identity_4Identity&forward_lstm_133/while/Select:output:0^forward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_133/while/Identity_4Ь
!forward_lstm_133/while/Identity_5Identity(forward_lstm_133/while/Select_1:output:0^forward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_133/while/Identity_5Ь
!forward_lstm_133/while/Identity_6Identity(forward_lstm_133/while/Select_2:output:0^forward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_133/while/Identity_6Ж
forward_lstm_133/while/NoOpNoOp<^forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp;^forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp=^forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_133/while/NoOp"t
7forward_lstm_133_while_forward_lstm_133_strided_slice_19forward_lstm_133_while_forward_lstm_133_strided_slice_1_0"n
4forward_lstm_133_while_greater_forward_lstm_133_cast6forward_lstm_133_while_greater_forward_lstm_133_cast_0"K
forward_lstm_133_while_identity(forward_lstm_133/while/Identity:output:0"O
!forward_lstm_133_while_identity_1*forward_lstm_133/while/Identity_1:output:0"O
!forward_lstm_133_while_identity_2*forward_lstm_133/while/Identity_2:output:0"O
!forward_lstm_133_while_identity_3*forward_lstm_133/while/Identity_3:output:0"O
!forward_lstm_133_while_identity_4*forward_lstm_133/while/Identity_4:output:0"O
!forward_lstm_133_while_identity_5*forward_lstm_133/while/Identity_5:output:0"O
!forward_lstm_133_while_identity_6*forward_lstm_133/while/Identity_6:output:0"
Dforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resourceFforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0"
Eforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resourceGforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0"
Cforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resourceEforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0"ь
sforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensoruforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2z
;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp2x
:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp2|
<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp: 
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
п
Э
while_cond_16884356
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16884356___redundant_placeholder06
2while_while_cond_16884356___redundant_placeholder16
2while_while_cond_16884356___redundant_placeholder26
2while_while_cond_16884356___redundant_placeholder3
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
р
Р
3__inference_forward_lstm_133_layer_call_fn_16888778

inputs
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
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
GPU 2J 8 *W
fRRP
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_168862182
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
И
Ѓ
L__inference_sequential_133_layer_call_and_return_conditional_losses_16887207

inputs
inputs_1	-
bidirectional_133_16887188:	Ш-
bidirectional_133_16887190:	2Ш)
bidirectional_133_16887192:	Ш-
bidirectional_133_16887194:	Ш-
bidirectional_133_16887196:	2Ш)
bidirectional_133_16887198:	Ш$
dense_133_16887201:d 
dense_133_16887203:
identityЂ)bidirectional_133/StatefulPartitionedCallЂ!dense_133/StatefulPartitionedCallЪ
)bidirectional_133/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_133_16887188bidirectional_133_16887190bidirectional_133_16887192bidirectional_133_16887194bidirectional_133_16887196bidirectional_133_16887198*
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
GPU 2J 8 *X
fSRQ
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_168871442+
)bidirectional_133/StatefulPartitionedCallЫ
!dense_133/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_133/StatefulPartitionedCall:output:0dense_133_16887201dense_133_16887203*
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
GPU 2J 8 *P
fKRI
G__inference_dense_133_layer_call_and_return_conditional_losses_168867292#
!dense_133/StatefulPartitionedCall
IdentityIdentity*dense_133/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp*^bidirectional_133/StatefulPartitionedCall"^dense_133/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2V
)bidirectional_133/StatefulPartitionedCall)bidirectional_133/StatefulPartitionedCall2F
!dense_133/StatefulPartitionedCall!dense_133/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ў

Ѕ
4__inference_bidirectional_133_layer_call_fn_16887376

inputs
inputs_1	
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
	unknown_2:	Ш
	unknown_3:	2Ш
	unknown_4:	Ш
identityЂStatefulPartitionedCallО
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
GPU 2J 8 *X
fSRQ
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_168867042
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
e
Т
$forward_lstm_133_while_body_16888080>
:forward_lstm_133_while_forward_lstm_133_while_loop_counterD
@forward_lstm_133_while_forward_lstm_133_while_maximum_iterations&
"forward_lstm_133_while_placeholder(
$forward_lstm_133_while_placeholder_1(
$forward_lstm_133_while_placeholder_2(
$forward_lstm_133_while_placeholder_3(
$forward_lstm_133_while_placeholder_4=
9forward_lstm_133_while_forward_lstm_133_strided_slice_1_0y
uforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_133_while_greater_forward_lstm_133_cast_0X
Eforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0:	ШZ
Gforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0:	2ШU
Fforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0:	Ш#
forward_lstm_133_while_identity%
!forward_lstm_133_while_identity_1%
!forward_lstm_133_while_identity_2%
!forward_lstm_133_while_identity_3%
!forward_lstm_133_while_identity_4%
!forward_lstm_133_while_identity_5%
!forward_lstm_133_while_identity_6;
7forward_lstm_133_while_forward_lstm_133_strided_slice_1w
sforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_133_while_greater_forward_lstm_133_castV
Cforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource:	ШX
Eforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource:	2ШS
Dforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource:	ШЂ;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOpЂ:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOpЂ<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOpх
Hforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2J
Hforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
:forward_lstm_133/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_133_while_placeholderQforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02<
:forward_lstm_133/while/TensorArrayV2Read/TensorListGetItemе
forward_lstm_133/while/GreaterGreater6forward_lstm_133_while_greater_forward_lstm_133_cast_0"forward_lstm_133_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2 
forward_lstm_133/while/Greaterџ
:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOpReadVariableOpEforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp
+forward_lstm_133/while/lstm_cell_400/MatMulMatMulAforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem:item:0Bforward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_133/while/lstm_cell_400/MatMul
<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOpGforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp
-forward_lstm_133/while/lstm_cell_400/MatMul_1MatMul$forward_lstm_133_while_placeholder_3Dforward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-forward_lstm_133/while/lstm_cell_400/MatMul_1
(forward_lstm_133/while/lstm_cell_400/addAddV25forward_lstm_133/while/lstm_cell_400/MatMul:product:07forward_lstm_133/while/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_133/while/lstm_cell_400/addў
;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOpFforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp
,forward_lstm_133/while/lstm_cell_400/BiasAddBiasAdd,forward_lstm_133/while/lstm_cell_400/add:z:0Cforward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_133/while/lstm_cell_400/BiasAddЎ
4forward_lstm_133/while/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4forward_lstm_133/while/lstm_cell_400/split/split_dimг
*forward_lstm_133/while/lstm_cell_400/splitSplit=forward_lstm_133/while/lstm_cell_400/split/split_dim:output:05forward_lstm_133/while/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*forward_lstm_133/while/lstm_cell_400/splitЮ
,forward_lstm_133/while/lstm_cell_400/SigmoidSigmoid3forward_lstm_133/while/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,forward_lstm_133/while/lstm_cell_400/Sigmoidв
.forward_lstm_133/while/lstm_cell_400/Sigmoid_1Sigmoid3forward_lstm_133/while/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_133/while/lstm_cell_400/Sigmoid_1ч
(forward_lstm_133/while/lstm_cell_400/mulMul2forward_lstm_133/while/lstm_cell_400/Sigmoid_1:y:0$forward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_133/while/lstm_cell_400/mulХ
)forward_lstm_133/while/lstm_cell_400/ReluRelu3forward_lstm_133/while/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_133/while/lstm_cell_400/Reluќ
*forward_lstm_133/while/lstm_cell_400/mul_1Mul0forward_lstm_133/while/lstm_cell_400/Sigmoid:y:07forward_lstm_133/while/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_133/while/lstm_cell_400/mul_1ё
*forward_lstm_133/while/lstm_cell_400/add_1AddV2,forward_lstm_133/while/lstm_cell_400/mul:z:0.forward_lstm_133/while/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_133/while/lstm_cell_400/add_1в
.forward_lstm_133/while/lstm_cell_400/Sigmoid_2Sigmoid3forward_lstm_133/while/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_133/while/lstm_cell_400/Sigmoid_2Ф
+forward_lstm_133/while/lstm_cell_400/Relu_1Relu.forward_lstm_133/while/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_133/while/lstm_cell_400/Relu_1
*forward_lstm_133/while/lstm_cell_400/mul_2Mul2forward_lstm_133/while/lstm_cell_400/Sigmoid_2:y:09forward_lstm_133/while/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_133/while/lstm_cell_400/mul_2є
forward_lstm_133/while/SelectSelect"forward_lstm_133/while/Greater:z:0.forward_lstm_133/while/lstm_cell_400/mul_2:z:0$forward_lstm_133_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/while/Selectј
forward_lstm_133/while/Select_1Select"forward_lstm_133/while/Greater:z:0.forward_lstm_133/while/lstm_cell_400/mul_2:z:0$forward_lstm_133_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_133/while/Select_1ј
forward_lstm_133/while/Select_2Select"forward_lstm_133/while/Greater:z:0.forward_lstm_133/while/lstm_cell_400/add_1:z:0$forward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_133/while/Select_2Ў
;forward_lstm_133/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_133_while_placeholder_1"forward_lstm_133_while_placeholder&forward_lstm_133/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_133/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_133/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_133/while/add/y­
forward_lstm_133/while/addAddV2"forward_lstm_133_while_placeholder%forward_lstm_133/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/while/add
forward_lstm_133/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_133/while/add_1/yЫ
forward_lstm_133/while/add_1AddV2:forward_lstm_133_while_forward_lstm_133_while_loop_counter'forward_lstm_133/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/while/add_1Џ
forward_lstm_133/while/IdentityIdentity forward_lstm_133/while/add_1:z:0^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_133/while/Identityг
!forward_lstm_133/while/Identity_1Identity@forward_lstm_133_while_forward_lstm_133_while_maximum_iterations^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_133/while/Identity_1Б
!forward_lstm_133/while/Identity_2Identityforward_lstm_133/while/add:z:0^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_133/while/Identity_2о
!forward_lstm_133/while/Identity_3IdentityKforward_lstm_133/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_133/while/Identity_3Ъ
!forward_lstm_133/while/Identity_4Identity&forward_lstm_133/while/Select:output:0^forward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_133/while/Identity_4Ь
!forward_lstm_133/while/Identity_5Identity(forward_lstm_133/while/Select_1:output:0^forward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_133/while/Identity_5Ь
!forward_lstm_133/while/Identity_6Identity(forward_lstm_133/while/Select_2:output:0^forward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_133/while/Identity_6Ж
forward_lstm_133/while/NoOpNoOp<^forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp;^forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp=^forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_133/while/NoOp"t
7forward_lstm_133_while_forward_lstm_133_strided_slice_19forward_lstm_133_while_forward_lstm_133_strided_slice_1_0"n
4forward_lstm_133_while_greater_forward_lstm_133_cast6forward_lstm_133_while_greater_forward_lstm_133_cast_0"K
forward_lstm_133_while_identity(forward_lstm_133/while/Identity:output:0"O
!forward_lstm_133_while_identity_1*forward_lstm_133/while/Identity_1:output:0"O
!forward_lstm_133_while_identity_2*forward_lstm_133/while/Identity_2:output:0"O
!forward_lstm_133_while_identity_3*forward_lstm_133/while/Identity_3:output:0"O
!forward_lstm_133_while_identity_4*forward_lstm_133/while/Identity_4:output:0"O
!forward_lstm_133_while_identity_5*forward_lstm_133/while/Identity_5:output:0"O
!forward_lstm_133_while_identity_6*forward_lstm_133/while/Identity_6:output:0"
Dforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resourceFforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0"
Eforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resourceGforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0"
Cforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resourceEforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0"ь
sforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensoruforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2z
;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp2x
:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp2|
<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp: 
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
љ?
л
while_body_16885961
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_401_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_401_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_401_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_401_matmul_readvariableop_resource:	ШG
4while_lstm_cell_401_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_401_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_401/BiasAdd/ReadVariableOpЂ)while/lstm_cell_401/MatMul/ReadVariableOpЂ+while/lstm_cell_401/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_401/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_401_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_401/MatMul/ReadVariableOpк
while/lstm_cell_401/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/MatMulв
+while/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_401_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_401/MatMul_1/ReadVariableOpУ
while/lstm_cell_401/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/MatMul_1М
while/lstm_cell_401/addAddV2$while/lstm_cell_401/MatMul:product:0&while/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/addЫ
*while/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_401_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_401/BiasAdd/ReadVariableOpЩ
while/lstm_cell_401/BiasAddBiasAddwhile/lstm_cell_401/add:z:02while/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/BiasAdd
#while/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_401/split/split_dim
while/lstm_cell_401/splitSplit,while/lstm_cell_401/split/split_dim:output:0$while/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_401/split
while/lstm_cell_401/SigmoidSigmoid"while/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Sigmoid
while/lstm_cell_401/Sigmoid_1Sigmoid"while/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Sigmoid_1Ѓ
while/lstm_cell_401/mulMul!while/lstm_cell_401/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/mul
while/lstm_cell_401/ReluRelu"while/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/ReluИ
while/lstm_cell_401/mul_1Mulwhile/lstm_cell_401/Sigmoid:y:0&while/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/mul_1­
while/lstm_cell_401/add_1AddV2while/lstm_cell_401/mul:z:0while/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/add_1
while/lstm_cell_401/Sigmoid_2Sigmoid"while/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Sigmoid_2
while/lstm_cell_401/Relu_1Reluwhile/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Relu_1М
while/lstm_cell_401/mul_2Mul!while/lstm_cell_401/Sigmoid_2:y:0(while/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_401/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_401/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_401/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_401/BiasAdd/ReadVariableOp*^while/lstm_cell_401/MatMul/ReadVariableOp,^while/lstm_cell_401/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_401_biasadd_readvariableop_resource5while_lstm_cell_401_biasadd_readvariableop_resource_0"n
4while_lstm_cell_401_matmul_1_readvariableop_resource6while_lstm_cell_401_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_401_matmul_readvariableop_resource4while_lstm_cell_401_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_401/BiasAdd/ReadVariableOp*while/lstm_cell_401/BiasAdd/ReadVariableOp2V
)while/lstm_cell_401/MatMul/ReadVariableOp)while/lstm_cell_401/MatMul/ReadVariableOp2Z
+while/lstm_cell_401/MatMul_1/ReadVariableOp+while/lstm_cell_401/MatMul_1/ReadVariableOp: 
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
М
љ
0__inference_lstm_cell_400_layer_call_fn_16890055

inputs
states_0
states_1
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identity

identity_1

identity_2ЂStatefulPartitionedCallЦ
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
GPU 2J 8 *T
fORM
K__inference_lstm_cell_400_layer_call_and_return_conditional_losses_168843432
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
љ?
л
while_body_16889298
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_400_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_400_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_400_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_400_matmul_readvariableop_resource:	ШG
4while_lstm_cell_400_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_400_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_400/BiasAdd/ReadVariableOpЂ)while/lstm_cell_400/MatMul/ReadVariableOpЂ+while/lstm_cell_400/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_400/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_400_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_400/MatMul/ReadVariableOpк
while/lstm_cell_400/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/MatMulв
+while/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_400_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_400/MatMul_1/ReadVariableOpУ
while/lstm_cell_400/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/MatMul_1М
while/lstm_cell_400/addAddV2$while/lstm_cell_400/MatMul:product:0&while/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/addЫ
*while/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_400_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_400/BiasAdd/ReadVariableOpЩ
while/lstm_cell_400/BiasAddBiasAddwhile/lstm_cell_400/add:z:02while/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/BiasAdd
#while/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_400/split/split_dim
while/lstm_cell_400/splitSplit,while/lstm_cell_400/split/split_dim:output:0$while/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_400/split
while/lstm_cell_400/SigmoidSigmoid"while/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Sigmoid
while/lstm_cell_400/Sigmoid_1Sigmoid"while/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Sigmoid_1Ѓ
while/lstm_cell_400/mulMul!while/lstm_cell_400/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/mul
while/lstm_cell_400/ReluRelu"while/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/ReluИ
while/lstm_cell_400/mul_1Mulwhile/lstm_cell_400/Sigmoid:y:0&while/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/mul_1­
while/lstm_cell_400/add_1AddV2while/lstm_cell_400/mul:z:0while/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/add_1
while/lstm_cell_400/Sigmoid_2Sigmoid"while/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Sigmoid_2
while/lstm_cell_400/Relu_1Reluwhile/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Relu_1М
while/lstm_cell_400/mul_2Mul!while/lstm_cell_400/Sigmoid_2:y:0(while/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_400/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_400/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_400/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_400/BiasAdd/ReadVariableOp*^while/lstm_cell_400/MatMul/ReadVariableOp,^while/lstm_cell_400/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_400_biasadd_readvariableop_resource5while_lstm_cell_400_biasadd_readvariableop_resource_0"n
4while_lstm_cell_400_matmul_1_readvariableop_resource6while_lstm_cell_400_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_400_matmul_readvariableop_resource4while_lstm_cell_400_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_400/BiasAdd/ReadVariableOp*while/lstm_cell_400/BiasAdd/ReadVariableOp2V
)while/lstm_cell_400/MatMul/ReadVariableOp)while/lstm_cell_400/MatMul/ReadVariableOp2Z
+while/lstm_cell_400/MatMul_1/ReadVariableOp+while/lstm_cell_400/MatMul_1/ReadVariableOp: 
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
жF

N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_16884426

inputs)
lstm_cell_400_16884344:	Ш)
lstm_cell_400_16884346:	2Ш%
lstm_cell_400_16884348:	Ш
identityЂ%lstm_cell_400/StatefulPartitionedCallЂwhileD
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
strided_slice_2Ћ
%lstm_cell_400/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_400_16884344lstm_cell_400_16884346lstm_cell_400_16884348*
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
GPU 2J 8 *T
fORM
K__inference_lstm_cell_400_layer_call_and_return_conditional_losses_168843432'
%lstm_cell_400/StatefulPartitionedCall
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
while/loop_counterЭ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_400_16884344lstm_cell_400_16884346lstm_cell_400_16884348*
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
bodyR
while_body_16884357*
condR
while_cond_16884356*K
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
NoOpNoOp&^lstm_cell_400/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2N
%lstm_cell_400/StatefulPartitionedCall%lstm_cell_400/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
п
Э
while_cond_16889494
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16889494___redundant_placeholder06
2while_while_cond_16889494___redundant_placeholder16
2while_while_cond_16889494___redundant_placeholder26
2while_while_cond_16889494___redundant_placeholder3
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
жF

N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_16884636

inputs)
lstm_cell_400_16884554:	Ш)
lstm_cell_400_16884556:	2Ш%
lstm_cell_400_16884558:	Ш
identityЂ%lstm_cell_400/StatefulPartitionedCallЂwhileD
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
strided_slice_2Ћ
%lstm_cell_400/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_400_16884554lstm_cell_400_16884556lstm_cell_400_16884558*
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
GPU 2J 8 *T
fORM
K__inference_lstm_cell_400_layer_call_and_return_conditional_losses_168844892'
%lstm_cell_400/StatefulPartitionedCall
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
while/loop_counterЭ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_400_16884554lstm_cell_400_16884556lstm_cell_400_16884558*
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
bodyR
while_body_16884567*
condR
while_cond_16884566*K
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
NoOpNoOp&^lstm_cell_400/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2N
%lstm_cell_400/StatefulPartitionedCall%lstm_cell_400/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
e
Т
$forward_lstm_133_while_body_16886868>
:forward_lstm_133_while_forward_lstm_133_while_loop_counterD
@forward_lstm_133_while_forward_lstm_133_while_maximum_iterations&
"forward_lstm_133_while_placeholder(
$forward_lstm_133_while_placeholder_1(
$forward_lstm_133_while_placeholder_2(
$forward_lstm_133_while_placeholder_3(
$forward_lstm_133_while_placeholder_4=
9forward_lstm_133_while_forward_lstm_133_strided_slice_1_0y
uforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_133_while_greater_forward_lstm_133_cast_0X
Eforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0:	ШZ
Gforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0:	2ШU
Fforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0:	Ш#
forward_lstm_133_while_identity%
!forward_lstm_133_while_identity_1%
!forward_lstm_133_while_identity_2%
!forward_lstm_133_while_identity_3%
!forward_lstm_133_while_identity_4%
!forward_lstm_133_while_identity_5%
!forward_lstm_133_while_identity_6;
7forward_lstm_133_while_forward_lstm_133_strided_slice_1w
sforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_133_while_greater_forward_lstm_133_castV
Cforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource:	ШX
Eforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource:	2ШS
Dforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource:	ШЂ;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOpЂ:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOpЂ<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOpх
Hforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2J
Hforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
:forward_lstm_133/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_133_while_placeholderQforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02<
:forward_lstm_133/while/TensorArrayV2Read/TensorListGetItemе
forward_lstm_133/while/GreaterGreater6forward_lstm_133_while_greater_forward_lstm_133_cast_0"forward_lstm_133_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2 
forward_lstm_133/while/Greaterџ
:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOpReadVariableOpEforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp
+forward_lstm_133/while/lstm_cell_400/MatMulMatMulAforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem:item:0Bforward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_133/while/lstm_cell_400/MatMul
<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOpGforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp
-forward_lstm_133/while/lstm_cell_400/MatMul_1MatMul$forward_lstm_133_while_placeholder_3Dforward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-forward_lstm_133/while/lstm_cell_400/MatMul_1
(forward_lstm_133/while/lstm_cell_400/addAddV25forward_lstm_133/while/lstm_cell_400/MatMul:product:07forward_lstm_133/while/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_133/while/lstm_cell_400/addў
;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOpFforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp
,forward_lstm_133/while/lstm_cell_400/BiasAddBiasAdd,forward_lstm_133/while/lstm_cell_400/add:z:0Cforward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_133/while/lstm_cell_400/BiasAddЎ
4forward_lstm_133/while/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4forward_lstm_133/while/lstm_cell_400/split/split_dimг
*forward_lstm_133/while/lstm_cell_400/splitSplit=forward_lstm_133/while/lstm_cell_400/split/split_dim:output:05forward_lstm_133/while/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*forward_lstm_133/while/lstm_cell_400/splitЮ
,forward_lstm_133/while/lstm_cell_400/SigmoidSigmoid3forward_lstm_133/while/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,forward_lstm_133/while/lstm_cell_400/Sigmoidв
.forward_lstm_133/while/lstm_cell_400/Sigmoid_1Sigmoid3forward_lstm_133/while/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_133/while/lstm_cell_400/Sigmoid_1ч
(forward_lstm_133/while/lstm_cell_400/mulMul2forward_lstm_133/while/lstm_cell_400/Sigmoid_1:y:0$forward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_133/while/lstm_cell_400/mulХ
)forward_lstm_133/while/lstm_cell_400/ReluRelu3forward_lstm_133/while/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_133/while/lstm_cell_400/Reluќ
*forward_lstm_133/while/lstm_cell_400/mul_1Mul0forward_lstm_133/while/lstm_cell_400/Sigmoid:y:07forward_lstm_133/while/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_133/while/lstm_cell_400/mul_1ё
*forward_lstm_133/while/lstm_cell_400/add_1AddV2,forward_lstm_133/while/lstm_cell_400/mul:z:0.forward_lstm_133/while/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_133/while/lstm_cell_400/add_1в
.forward_lstm_133/while/lstm_cell_400/Sigmoid_2Sigmoid3forward_lstm_133/while/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_133/while/lstm_cell_400/Sigmoid_2Ф
+forward_lstm_133/while/lstm_cell_400/Relu_1Relu.forward_lstm_133/while/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_133/while/lstm_cell_400/Relu_1
*forward_lstm_133/while/lstm_cell_400/mul_2Mul2forward_lstm_133/while/lstm_cell_400/Sigmoid_2:y:09forward_lstm_133/while/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_133/while/lstm_cell_400/mul_2є
forward_lstm_133/while/SelectSelect"forward_lstm_133/while/Greater:z:0.forward_lstm_133/while/lstm_cell_400/mul_2:z:0$forward_lstm_133_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/while/Selectј
forward_lstm_133/while/Select_1Select"forward_lstm_133/while/Greater:z:0.forward_lstm_133/while/lstm_cell_400/mul_2:z:0$forward_lstm_133_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_133/while/Select_1ј
forward_lstm_133/while/Select_2Select"forward_lstm_133/while/Greater:z:0.forward_lstm_133/while/lstm_cell_400/add_1:z:0$forward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_133/while/Select_2Ў
;forward_lstm_133/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_133_while_placeholder_1"forward_lstm_133_while_placeholder&forward_lstm_133/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_133/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_133/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_133/while/add/y­
forward_lstm_133/while/addAddV2"forward_lstm_133_while_placeholder%forward_lstm_133/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/while/add
forward_lstm_133/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_133/while/add_1/yЫ
forward_lstm_133/while/add_1AddV2:forward_lstm_133_while_forward_lstm_133_while_loop_counter'forward_lstm_133/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/while/add_1Џ
forward_lstm_133/while/IdentityIdentity forward_lstm_133/while/add_1:z:0^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_133/while/Identityг
!forward_lstm_133/while/Identity_1Identity@forward_lstm_133_while_forward_lstm_133_while_maximum_iterations^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_133/while/Identity_1Б
!forward_lstm_133/while/Identity_2Identityforward_lstm_133/while/add:z:0^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_133/while/Identity_2о
!forward_lstm_133/while/Identity_3IdentityKforward_lstm_133/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_133/while/Identity_3Ъ
!forward_lstm_133/while/Identity_4Identity&forward_lstm_133/while/Select:output:0^forward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_133/while/Identity_4Ь
!forward_lstm_133/while/Identity_5Identity(forward_lstm_133/while/Select_1:output:0^forward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_133/while/Identity_5Ь
!forward_lstm_133/while/Identity_6Identity(forward_lstm_133/while/Select_2:output:0^forward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_133/while/Identity_6Ж
forward_lstm_133/while/NoOpNoOp<^forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp;^forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp=^forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_133/while/NoOp"t
7forward_lstm_133_while_forward_lstm_133_strided_slice_19forward_lstm_133_while_forward_lstm_133_strided_slice_1_0"n
4forward_lstm_133_while_greater_forward_lstm_133_cast6forward_lstm_133_while_greater_forward_lstm_133_cast_0"K
forward_lstm_133_while_identity(forward_lstm_133/while/Identity:output:0"O
!forward_lstm_133_while_identity_1*forward_lstm_133/while/Identity_1:output:0"O
!forward_lstm_133_while_identity_2*forward_lstm_133/while/Identity_2:output:0"O
!forward_lstm_133_while_identity_3*forward_lstm_133/while/Identity_3:output:0"O
!forward_lstm_133_while_identity_4*forward_lstm_133/while/Identity_4:output:0"O
!forward_lstm_133_while_identity_5*forward_lstm_133/while/Identity_5:output:0"O
!forward_lstm_133_while_identity_6*forward_lstm_133/while/Identity_6:output:0"
Dforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resourceFforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0"
Eforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resourceGforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0"
Cforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resourceEforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0"ь
sforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensoruforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2z
;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp2x
:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp2|
<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp: 
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
№?
л
while_body_16888845
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_400_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_400_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_400_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_400_matmul_readvariableop_resource:	ШG
4while_lstm_cell_400_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_400_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_400/BiasAdd/ReadVariableOpЂ)while/lstm_cell_400/MatMul/ReadVariableOpЂ+while/lstm_cell_400/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_400/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_400_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_400/MatMul/ReadVariableOpк
while/lstm_cell_400/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/MatMulв
+while/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_400_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_400/MatMul_1/ReadVariableOpУ
while/lstm_cell_400/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/MatMul_1М
while/lstm_cell_400/addAddV2$while/lstm_cell_400/MatMul:product:0&while/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/addЫ
*while/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_400_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_400/BiasAdd/ReadVariableOpЩ
while/lstm_cell_400/BiasAddBiasAddwhile/lstm_cell_400/add:z:02while/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/BiasAdd
#while/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_400/split/split_dim
while/lstm_cell_400/splitSplit,while/lstm_cell_400/split/split_dim:output:0$while/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_400/split
while/lstm_cell_400/SigmoidSigmoid"while/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Sigmoid
while/lstm_cell_400/Sigmoid_1Sigmoid"while/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Sigmoid_1Ѓ
while/lstm_cell_400/mulMul!while/lstm_cell_400/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/mul
while/lstm_cell_400/ReluRelu"while/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/ReluИ
while/lstm_cell_400/mul_1Mulwhile/lstm_cell_400/Sigmoid:y:0&while/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/mul_1­
while/lstm_cell_400/add_1AddV2while/lstm_cell_400/mul:z:0while/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/add_1
while/lstm_cell_400/Sigmoid_2Sigmoid"while/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Sigmoid_2
while/lstm_cell_400/Relu_1Reluwhile/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Relu_1М
while/lstm_cell_400/mul_2Mul!while/lstm_cell_400/Sigmoid_2:y:0(while/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_400/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_400/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_400/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_400/BiasAdd/ReadVariableOp*^while/lstm_cell_400/MatMul/ReadVariableOp,^while/lstm_cell_400/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_400_biasadd_readvariableop_resource5while_lstm_cell_400_biasadd_readvariableop_resource_0"n
4while_lstm_cell_400_matmul_1_readvariableop_resource6while_lstm_cell_400_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_400_matmul_readvariableop_resource4while_lstm_cell_400_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_400/BiasAdd/ReadVariableOp*while/lstm_cell_400/BiasAdd/ReadVariableOp2V
)while/lstm_cell_400/MatMul/ReadVariableOp)while/lstm_cell_400/MatMul/ReadVariableOp2Z
+while/lstm_cell_400/MatMul_1/ReadVariableOp+while/lstm_cell_400/MatMul_1/ReadVariableOp: 
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
п
Э
while_cond_16888844
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16888844___redundant_placeholder06
2while_while_cond_16888844___redundant_placeholder16
2while_while_cond_16888844___redundant_placeholder26
2while_while_cond_16888844___redundant_placeholder3
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
зу

#__inference__wrapped_model_16884268

args_0
args_0_1	q
^sequential_133_bidirectional_133_forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource:	Шs
`sequential_133_bidirectional_133_forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource:	2Шn
_sequential_133_bidirectional_133_forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource:	Шr
_sequential_133_bidirectional_133_backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource:	Шt
asequential_133_bidirectional_133_backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource:	2Шo
`sequential_133_bidirectional_133_backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource:	ШI
7sequential_133_dense_133_matmul_readvariableop_resource:dF
8sequential_133_dense_133_biasadd_readvariableop_resource:
identityЂWsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpЂVsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpЂXsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpЂ8sequential_133/bidirectional_133/backward_lstm_133/whileЂVsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpЂUsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpЂWsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOpЂ7sequential_133/bidirectional_133/forward_lstm_133/whileЂ/sequential_133/dense_133/BiasAdd/ReadVariableOpЂ.sequential_133/dense_133/MatMul/ReadVariableOpй
Fsequential_133/bidirectional_133/forward_lstm_133/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2H
Fsequential_133/bidirectional_133/forward_lstm_133/RaggedToTensor/zerosл
Fsequential_133/bidirectional_133/forward_lstm_133/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2H
Fsequential_133/bidirectional_133/forward_lstm_133/RaggedToTensor/Const
Usequential_133/bidirectional_133/forward_lstm_133/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorOsequential_133/bidirectional_133/forward_lstm_133/RaggedToTensor/Const:output:0args_0Osequential_133/bidirectional_133/forward_lstm_133/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2W
Usequential_133/bidirectional_133/forward_lstm_133/RaggedToTensor/RaggedTensorToTensor
\sequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2^
\sequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack
^sequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2`
^sequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1
^sequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2`
^sequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2Ю
Vsequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1esequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack:output:0gsequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1:output:0gsequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask2X
Vsequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice
^sequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2`
^sequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack
`sequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2b
`sequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1
`sequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2b
`sequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2к
Xsequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1gsequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack:output:0isequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0isequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2Z
Xsequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice_1
Lsequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/subSub_sequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice:output:0asequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2N
Lsequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/sub
6sequential_133/bidirectional_133/forward_lstm_133/CastCastPsequential_133/bidirectional_133/forward_lstm_133/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ28
6sequential_133/bidirectional_133/forward_lstm_133/Cast
7sequential_133/bidirectional_133/forward_lstm_133/ShapeShape^sequential_133/bidirectional_133/forward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:29
7sequential_133/bidirectional_133/forward_lstm_133/Shapeи
Esequential_133/bidirectional_133/forward_lstm_133/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_133/bidirectional_133/forward_lstm_133/strided_slice/stackм
Gsequential_133/bidirectional_133/forward_lstm_133/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_133/bidirectional_133/forward_lstm_133/strided_slice/stack_1м
Gsequential_133/bidirectional_133/forward_lstm_133/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_133/bidirectional_133/forward_lstm_133/strided_slice/stack_2
?sequential_133/bidirectional_133/forward_lstm_133/strided_sliceStridedSlice@sequential_133/bidirectional_133/forward_lstm_133/Shape:output:0Nsequential_133/bidirectional_133/forward_lstm_133/strided_slice/stack:output:0Psequential_133/bidirectional_133/forward_lstm_133/strided_slice/stack_1:output:0Psequential_133/bidirectional_133/forward_lstm_133/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?sequential_133/bidirectional_133/forward_lstm_133/strided_sliceР
=sequential_133/bidirectional_133/forward_lstm_133/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22?
=sequential_133/bidirectional_133/forward_lstm_133/zeros/mul/yД
;sequential_133/bidirectional_133/forward_lstm_133/zeros/mulMulHsequential_133/bidirectional_133/forward_lstm_133/strided_slice:output:0Fsequential_133/bidirectional_133/forward_lstm_133/zeros/mul/y:output:0*
T0*
_output_shapes
: 2=
;sequential_133/bidirectional_133/forward_lstm_133/zeros/mulУ
>sequential_133/bidirectional_133/forward_lstm_133/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2@
>sequential_133/bidirectional_133/forward_lstm_133/zeros/Less/yЏ
<sequential_133/bidirectional_133/forward_lstm_133/zeros/LessLess?sequential_133/bidirectional_133/forward_lstm_133/zeros/mul:z:0Gsequential_133/bidirectional_133/forward_lstm_133/zeros/Less/y:output:0*
T0*
_output_shapes
: 2>
<sequential_133/bidirectional_133/forward_lstm_133/zeros/LessЦ
@sequential_133/bidirectional_133/forward_lstm_133/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22B
@sequential_133/bidirectional_133/forward_lstm_133/zeros/packed/1Ы
>sequential_133/bidirectional_133/forward_lstm_133/zeros/packedPackHsequential_133/bidirectional_133/forward_lstm_133/strided_slice:output:0Isequential_133/bidirectional_133/forward_lstm_133/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2@
>sequential_133/bidirectional_133/forward_lstm_133/zeros/packedЧ
=sequential_133/bidirectional_133/forward_lstm_133/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2?
=sequential_133/bidirectional_133/forward_lstm_133/zeros/ConstН
7sequential_133/bidirectional_133/forward_lstm_133/zerosFillGsequential_133/bidirectional_133/forward_lstm_133/zeros/packed:output:0Fsequential_133/bidirectional_133/forward_lstm_133/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ229
7sequential_133/bidirectional_133/forward_lstm_133/zerosФ
?sequential_133/bidirectional_133/forward_lstm_133/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22A
?sequential_133/bidirectional_133/forward_lstm_133/zeros_1/mul/yК
=sequential_133/bidirectional_133/forward_lstm_133/zeros_1/mulMulHsequential_133/bidirectional_133/forward_lstm_133/strided_slice:output:0Hsequential_133/bidirectional_133/forward_lstm_133/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2?
=sequential_133/bidirectional_133/forward_lstm_133/zeros_1/mulЧ
@sequential_133/bidirectional_133/forward_lstm_133/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2B
@sequential_133/bidirectional_133/forward_lstm_133/zeros_1/Less/yЗ
>sequential_133/bidirectional_133/forward_lstm_133/zeros_1/LessLessAsequential_133/bidirectional_133/forward_lstm_133/zeros_1/mul:z:0Isequential_133/bidirectional_133/forward_lstm_133/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2@
>sequential_133/bidirectional_133/forward_lstm_133/zeros_1/LessЪ
Bsequential_133/bidirectional_133/forward_lstm_133/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22D
Bsequential_133/bidirectional_133/forward_lstm_133/zeros_1/packed/1б
@sequential_133/bidirectional_133/forward_lstm_133/zeros_1/packedPackHsequential_133/bidirectional_133/forward_lstm_133/strided_slice:output:0Ksequential_133/bidirectional_133/forward_lstm_133/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2B
@sequential_133/bidirectional_133/forward_lstm_133/zeros_1/packedЫ
?sequential_133/bidirectional_133/forward_lstm_133/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2A
?sequential_133/bidirectional_133/forward_lstm_133/zeros_1/ConstХ
9sequential_133/bidirectional_133/forward_lstm_133/zeros_1FillIsequential_133/bidirectional_133/forward_lstm_133/zeros_1/packed:output:0Hsequential_133/bidirectional_133/forward_lstm_133/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22;
9sequential_133/bidirectional_133/forward_lstm_133/zeros_1й
@sequential_133/bidirectional_133/forward_lstm_133/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2B
@sequential_133/bidirectional_133/forward_lstm_133/transpose/permё
;sequential_133/bidirectional_133/forward_lstm_133/transpose	Transpose^sequential_133/bidirectional_133/forward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0Isequential_133/bidirectional_133/forward_lstm_133/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2=
;sequential_133/bidirectional_133/forward_lstm_133/transposeх
9sequential_133/bidirectional_133/forward_lstm_133/Shape_1Shape?sequential_133/bidirectional_133/forward_lstm_133/transpose:y:0*
T0*
_output_shapes
:2;
9sequential_133/bidirectional_133/forward_lstm_133/Shape_1м
Gsequential_133/bidirectional_133/forward_lstm_133/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2I
Gsequential_133/bidirectional_133/forward_lstm_133/strided_slice_1/stackр
Isequential_133/bidirectional_133/forward_lstm_133/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_133/bidirectional_133/forward_lstm_133/strided_slice_1/stack_1р
Isequential_133/bidirectional_133/forward_lstm_133/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_133/bidirectional_133/forward_lstm_133/strided_slice_1/stack_2
Asequential_133/bidirectional_133/forward_lstm_133/strided_slice_1StridedSliceBsequential_133/bidirectional_133/forward_lstm_133/Shape_1:output:0Psequential_133/bidirectional_133/forward_lstm_133/strided_slice_1/stack:output:0Rsequential_133/bidirectional_133/forward_lstm_133/strided_slice_1/stack_1:output:0Rsequential_133/bidirectional_133/forward_lstm_133/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2C
Asequential_133/bidirectional_133/forward_lstm_133/strided_slice_1щ
Msequential_133/bidirectional_133/forward_lstm_133/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2O
Msequential_133/bidirectional_133/forward_lstm_133/TensorArrayV2/element_shapeњ
?sequential_133/bidirectional_133/forward_lstm_133/TensorArrayV2TensorListReserveVsequential_133/bidirectional_133/forward_lstm_133/TensorArrayV2/element_shape:output:0Jsequential_133/bidirectional_133/forward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?sequential_133/bidirectional_133/forward_lstm_133/TensorArrayV2Ѓ
gsequential_133/bidirectional_133/forward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2i
gsequential_133/bidirectional_133/forward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeР
Ysequential_133/bidirectional_133/forward_lstm_133/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor?sequential_133/bidirectional_133/forward_lstm_133/transpose:y:0psequential_133/bidirectional_133/forward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02[
Ysequential_133/bidirectional_133/forward_lstm_133/TensorArrayUnstack/TensorListFromTensorм
Gsequential_133/bidirectional_133/forward_lstm_133/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2I
Gsequential_133/bidirectional_133/forward_lstm_133/strided_slice_2/stackр
Isequential_133/bidirectional_133/forward_lstm_133/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_133/bidirectional_133/forward_lstm_133/strided_slice_2/stack_1р
Isequential_133/bidirectional_133/forward_lstm_133/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_133/bidirectional_133/forward_lstm_133/strided_slice_2/stack_2Ј
Asequential_133/bidirectional_133/forward_lstm_133/strided_slice_2StridedSlice?sequential_133/bidirectional_133/forward_lstm_133/transpose:y:0Psequential_133/bidirectional_133/forward_lstm_133/strided_slice_2/stack:output:0Rsequential_133/bidirectional_133/forward_lstm_133/strided_slice_2/stack_1:output:0Rsequential_133/bidirectional_133/forward_lstm_133/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2C
Asequential_133/bidirectional_133/forward_lstm_133/strided_slice_2Ю
Usequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpReadVariableOp^sequential_133_bidirectional_133_forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02W
Usequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpј
Fsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMulMatMulJsequential_133/bidirectional_133/forward_lstm_133/strided_slice_2:output:0]sequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2H
Fsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMulд
Wsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOp`sequential_133_bidirectional_133_forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02Y
Wsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOpє
Hsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMul_1MatMul@sequential_133/bidirectional_133/forward_lstm_133/zeros:output:0_sequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2J
Hsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMul_1ь
Csequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/addAddV2Psequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMul:product:0Rsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2E
Csequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/addЭ
Vsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOp_sequential_133_bidirectional_133_forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02X
Vsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpљ
Gsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/BiasAddBiasAddGsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/add:z:0^sequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2I
Gsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/BiasAddф
Osequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2Q
Osequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/split/split_dimП
Esequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/splitSplitXsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/split/split_dim:output:0Psequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2G
Esequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/split
Gsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/SigmoidSigmoidNsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22I
Gsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/SigmoidЃ
Isequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/Sigmoid_1SigmoidNsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22K
Isequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/Sigmoid_1ж
Csequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/mulMulMsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/Sigmoid_1:y:0Bsequential_133/bidirectional_133/forward_lstm_133/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/mul
Dsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/ReluReluNsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22F
Dsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/Reluш
Esequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/mul_1MulKsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/Sigmoid:y:0Rsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22G
Esequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/mul_1н
Esequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/add_1AddV2Gsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/mul:z:0Isequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22G
Esequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/add_1Ѓ
Isequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/Sigmoid_2SigmoidNsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22K
Isequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/Sigmoid_2
Fsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/Relu_1ReluIsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22H
Fsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/Relu_1ь
Esequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/mul_2MulMsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/Sigmoid_2:y:0Tsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22G
Esequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/mul_2ѓ
Osequential_133/bidirectional_133/forward_lstm_133/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2Q
Osequential_133/bidirectional_133/forward_lstm_133/TensorArrayV2_1/element_shape
Asequential_133/bidirectional_133/forward_lstm_133/TensorArrayV2_1TensorListReserveXsequential_133/bidirectional_133/forward_lstm_133/TensorArrayV2_1/element_shape:output:0Jsequential_133/bidirectional_133/forward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02C
Asequential_133/bidirectional_133/forward_lstm_133/TensorArrayV2_1В
6sequential_133/bidirectional_133/forward_lstm_133/timeConst*
_output_shapes
: *
dtype0*
value	B : 28
6sequential_133/bidirectional_133/forward_lstm_133/time
<sequential_133/bidirectional_133/forward_lstm_133/zeros_like	ZerosLikeIsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22>
<sequential_133/bidirectional_133/forward_lstm_133/zeros_likeу
Jsequential_133/bidirectional_133/forward_lstm_133/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2L
Jsequential_133/bidirectional_133/forward_lstm_133/while/maximum_iterationsЮ
Dsequential_133/bidirectional_133/forward_lstm_133/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dsequential_133/bidirectional_133/forward_lstm_133/while/loop_counterФ
7sequential_133/bidirectional_133/forward_lstm_133/whileWhileMsequential_133/bidirectional_133/forward_lstm_133/while/loop_counter:output:0Ssequential_133/bidirectional_133/forward_lstm_133/while/maximum_iterations:output:0?sequential_133/bidirectional_133/forward_lstm_133/time:output:0Jsequential_133/bidirectional_133/forward_lstm_133/TensorArrayV2_1:handle:0@sequential_133/bidirectional_133/forward_lstm_133/zeros_like:y:0@sequential_133/bidirectional_133/forward_lstm_133/zeros:output:0Bsequential_133/bidirectional_133/forward_lstm_133/zeros_1:output:0Jsequential_133/bidirectional_133/forward_lstm_133/strided_slice_1:output:0isequential_133/bidirectional_133/forward_lstm_133/TensorArrayUnstack/TensorListFromTensor:output_handle:0:sequential_133/bidirectional_133/forward_lstm_133/Cast:y:0^sequential_133_bidirectional_133_forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource`sequential_133_bidirectional_133_forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource_sequential_133_bidirectional_133_forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *Q
bodyIRG
Esequential_133_bidirectional_133_forward_lstm_133_while_body_16883985*Q
condIRG
Esequential_133_bidirectional_133_forward_lstm_133_while_cond_16883984*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 29
7sequential_133/bidirectional_133/forward_lstm_133/while
bsequential_133/bidirectional_133/forward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2d
bsequential_133/bidirectional_133/forward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeЙ
Tsequential_133/bidirectional_133/forward_lstm_133/TensorArrayV2Stack/TensorListStackTensorListStack@sequential_133/bidirectional_133/forward_lstm_133/while:output:3ksequential_133/bidirectional_133/forward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02V
Tsequential_133/bidirectional_133/forward_lstm_133/TensorArrayV2Stack/TensorListStackх
Gsequential_133/bidirectional_133/forward_lstm_133/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2I
Gsequential_133/bidirectional_133/forward_lstm_133/strided_slice_3/stackр
Isequential_133/bidirectional_133/forward_lstm_133/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2K
Isequential_133/bidirectional_133/forward_lstm_133/strided_slice_3/stack_1р
Isequential_133/bidirectional_133/forward_lstm_133/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_133/bidirectional_133/forward_lstm_133/strided_slice_3/stack_2Ц
Asequential_133/bidirectional_133/forward_lstm_133/strided_slice_3StridedSlice]sequential_133/bidirectional_133/forward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0Psequential_133/bidirectional_133/forward_lstm_133/strided_slice_3/stack:output:0Rsequential_133/bidirectional_133/forward_lstm_133/strided_slice_3/stack_1:output:0Rsequential_133/bidirectional_133/forward_lstm_133/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2C
Asequential_133/bidirectional_133/forward_lstm_133/strided_slice_3н
Bsequential_133/bidirectional_133/forward_lstm_133/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2D
Bsequential_133/bidirectional_133/forward_lstm_133/transpose_1/permі
=sequential_133/bidirectional_133/forward_lstm_133/transpose_1	Transpose]sequential_133/bidirectional_133/forward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0Ksequential_133/bidirectional_133/forward_lstm_133/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22?
=sequential_133/bidirectional_133/forward_lstm_133/transpose_1Ъ
9sequential_133/bidirectional_133/forward_lstm_133/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2;
9sequential_133/bidirectional_133/forward_lstm_133/runtimeл
Gsequential_133/bidirectional_133/backward_lstm_133/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2I
Gsequential_133/bidirectional_133/backward_lstm_133/RaggedToTensor/zerosн
Gsequential_133/bidirectional_133/backward_lstm_133/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2I
Gsequential_133/bidirectional_133/backward_lstm_133/RaggedToTensor/ConstЁ
Vsequential_133/bidirectional_133/backward_lstm_133/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorPsequential_133/bidirectional_133/backward_lstm_133/RaggedToTensor/Const:output:0args_0Psequential_133/bidirectional_133/backward_lstm_133/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2X
Vsequential_133/bidirectional_133/backward_lstm_133/RaggedToTensor/RaggedTensorToTensor
]sequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2_
]sequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack
_sequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2a
_sequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1
_sequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2a
_sequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2г
Wsequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1fsequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack:output:0hsequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1:output:0hsequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask2Y
Wsequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice
_sequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2a
_sequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack
asequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2c
asequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1
asequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2c
asequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2п
Ysequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1hsequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack:output:0jsequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0jsequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2[
Ysequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice_1
Msequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/subSub`sequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice:output:0bsequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2O
Msequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/sub
7sequential_133/bidirectional_133/backward_lstm_133/CastCastQsequential_133/bidirectional_133/backward_lstm_133/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ29
7sequential_133/bidirectional_133/backward_lstm_133/Cast
8sequential_133/bidirectional_133/backward_lstm_133/ShapeShape_sequential_133/bidirectional_133/backward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2:
8sequential_133/bidirectional_133/backward_lstm_133/Shapeк
Fsequential_133/bidirectional_133/backward_lstm_133/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fsequential_133/bidirectional_133/backward_lstm_133/strided_slice/stackо
Hsequential_133/bidirectional_133/backward_lstm_133/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_133/bidirectional_133/backward_lstm_133/strided_slice/stack_1о
Hsequential_133/bidirectional_133/backward_lstm_133/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_133/bidirectional_133/backward_lstm_133/strided_slice/stack_2
@sequential_133/bidirectional_133/backward_lstm_133/strided_sliceStridedSliceAsequential_133/bidirectional_133/backward_lstm_133/Shape:output:0Osequential_133/bidirectional_133/backward_lstm_133/strided_slice/stack:output:0Qsequential_133/bidirectional_133/backward_lstm_133/strided_slice/stack_1:output:0Qsequential_133/bidirectional_133/backward_lstm_133/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2B
@sequential_133/bidirectional_133/backward_lstm_133/strided_sliceТ
>sequential_133/bidirectional_133/backward_lstm_133/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22@
>sequential_133/bidirectional_133/backward_lstm_133/zeros/mul/yИ
<sequential_133/bidirectional_133/backward_lstm_133/zeros/mulMulIsequential_133/bidirectional_133/backward_lstm_133/strided_slice:output:0Gsequential_133/bidirectional_133/backward_lstm_133/zeros/mul/y:output:0*
T0*
_output_shapes
: 2>
<sequential_133/bidirectional_133/backward_lstm_133/zeros/mulХ
?sequential_133/bidirectional_133/backward_lstm_133/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2A
?sequential_133/bidirectional_133/backward_lstm_133/zeros/Less/yГ
=sequential_133/bidirectional_133/backward_lstm_133/zeros/LessLess@sequential_133/bidirectional_133/backward_lstm_133/zeros/mul:z:0Hsequential_133/bidirectional_133/backward_lstm_133/zeros/Less/y:output:0*
T0*
_output_shapes
: 2?
=sequential_133/bidirectional_133/backward_lstm_133/zeros/LessШ
Asequential_133/bidirectional_133/backward_lstm_133/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22C
Asequential_133/bidirectional_133/backward_lstm_133/zeros/packed/1Я
?sequential_133/bidirectional_133/backward_lstm_133/zeros/packedPackIsequential_133/bidirectional_133/backward_lstm_133/strided_slice:output:0Jsequential_133/bidirectional_133/backward_lstm_133/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2A
?sequential_133/bidirectional_133/backward_lstm_133/zeros/packedЩ
>sequential_133/bidirectional_133/backward_lstm_133/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2@
>sequential_133/bidirectional_133/backward_lstm_133/zeros/ConstС
8sequential_133/bidirectional_133/backward_lstm_133/zerosFillHsequential_133/bidirectional_133/backward_lstm_133/zeros/packed:output:0Gsequential_133/bidirectional_133/backward_lstm_133/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22:
8sequential_133/bidirectional_133/backward_lstm_133/zerosЦ
@sequential_133/bidirectional_133/backward_lstm_133/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22B
@sequential_133/bidirectional_133/backward_lstm_133/zeros_1/mul/yО
>sequential_133/bidirectional_133/backward_lstm_133/zeros_1/mulMulIsequential_133/bidirectional_133/backward_lstm_133/strided_slice:output:0Isequential_133/bidirectional_133/backward_lstm_133/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2@
>sequential_133/bidirectional_133/backward_lstm_133/zeros_1/mulЩ
Asequential_133/bidirectional_133/backward_lstm_133/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2C
Asequential_133/bidirectional_133/backward_lstm_133/zeros_1/Less/yЛ
?sequential_133/bidirectional_133/backward_lstm_133/zeros_1/LessLessBsequential_133/bidirectional_133/backward_lstm_133/zeros_1/mul:z:0Jsequential_133/bidirectional_133/backward_lstm_133/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2A
?sequential_133/bidirectional_133/backward_lstm_133/zeros_1/LessЬ
Csequential_133/bidirectional_133/backward_lstm_133/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22E
Csequential_133/bidirectional_133/backward_lstm_133/zeros_1/packed/1е
Asequential_133/bidirectional_133/backward_lstm_133/zeros_1/packedPackIsequential_133/bidirectional_133/backward_lstm_133/strided_slice:output:0Lsequential_133/bidirectional_133/backward_lstm_133/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2C
Asequential_133/bidirectional_133/backward_lstm_133/zeros_1/packedЭ
@sequential_133/bidirectional_133/backward_lstm_133/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2B
@sequential_133/bidirectional_133/backward_lstm_133/zeros_1/ConstЩ
:sequential_133/bidirectional_133/backward_lstm_133/zeros_1FillJsequential_133/bidirectional_133/backward_lstm_133/zeros_1/packed:output:0Isequential_133/bidirectional_133/backward_lstm_133/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22<
:sequential_133/bidirectional_133/backward_lstm_133/zeros_1л
Asequential_133/bidirectional_133/backward_lstm_133/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2C
Asequential_133/bidirectional_133/backward_lstm_133/transpose/permѕ
<sequential_133/bidirectional_133/backward_lstm_133/transpose	Transpose_sequential_133/bidirectional_133/backward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0Jsequential_133/bidirectional_133/backward_lstm_133/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2>
<sequential_133/bidirectional_133/backward_lstm_133/transposeш
:sequential_133/bidirectional_133/backward_lstm_133/Shape_1Shape@sequential_133/bidirectional_133/backward_lstm_133/transpose:y:0*
T0*
_output_shapes
:2<
:sequential_133/bidirectional_133/backward_lstm_133/Shape_1о
Hsequential_133/bidirectional_133/backward_lstm_133/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_133/bidirectional_133/backward_lstm_133/strided_slice_1/stackт
Jsequential_133/bidirectional_133/backward_lstm_133/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_133/bidirectional_133/backward_lstm_133/strided_slice_1/stack_1т
Jsequential_133/bidirectional_133/backward_lstm_133/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_133/bidirectional_133/backward_lstm_133/strided_slice_1/stack_2 
Bsequential_133/bidirectional_133/backward_lstm_133/strided_slice_1StridedSliceCsequential_133/bidirectional_133/backward_lstm_133/Shape_1:output:0Qsequential_133/bidirectional_133/backward_lstm_133/strided_slice_1/stack:output:0Ssequential_133/bidirectional_133/backward_lstm_133/strided_slice_1/stack_1:output:0Ssequential_133/bidirectional_133/backward_lstm_133/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2D
Bsequential_133/bidirectional_133/backward_lstm_133/strided_slice_1ы
Nsequential_133/bidirectional_133/backward_lstm_133/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2P
Nsequential_133/bidirectional_133/backward_lstm_133/TensorArrayV2/element_shapeў
@sequential_133/bidirectional_133/backward_lstm_133/TensorArrayV2TensorListReserveWsequential_133/bidirectional_133/backward_lstm_133/TensorArrayV2/element_shape:output:0Ksequential_133/bidirectional_133/backward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02B
@sequential_133/bidirectional_133/backward_lstm_133/TensorArrayV2а
Asequential_133/bidirectional_133/backward_lstm_133/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential_133/bidirectional_133/backward_lstm_133/ReverseV2/axisж
<sequential_133/bidirectional_133/backward_lstm_133/ReverseV2	ReverseV2@sequential_133/bidirectional_133/backward_lstm_133/transpose:y:0Jsequential_133/bidirectional_133/backward_lstm_133/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2>
<sequential_133/bidirectional_133/backward_lstm_133/ReverseV2Ѕ
hsequential_133/bidirectional_133/backward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2j
hsequential_133/bidirectional_133/backward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeЩ
Zsequential_133/bidirectional_133/backward_lstm_133/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorEsequential_133/bidirectional_133/backward_lstm_133/ReverseV2:output:0qsequential_133/bidirectional_133/backward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02\
Zsequential_133/bidirectional_133/backward_lstm_133/TensorArrayUnstack/TensorListFromTensorо
Hsequential_133/bidirectional_133/backward_lstm_133/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_133/bidirectional_133/backward_lstm_133/strided_slice_2/stackт
Jsequential_133/bidirectional_133/backward_lstm_133/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_133/bidirectional_133/backward_lstm_133/strided_slice_2/stack_1т
Jsequential_133/bidirectional_133/backward_lstm_133/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_133/bidirectional_133/backward_lstm_133/strided_slice_2/stack_2Ў
Bsequential_133/bidirectional_133/backward_lstm_133/strided_slice_2StridedSlice@sequential_133/bidirectional_133/backward_lstm_133/transpose:y:0Qsequential_133/bidirectional_133/backward_lstm_133/strided_slice_2/stack:output:0Ssequential_133/bidirectional_133/backward_lstm_133/strided_slice_2/stack_1:output:0Ssequential_133/bidirectional_133/backward_lstm_133/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2D
Bsequential_133/bidirectional_133/backward_lstm_133/strided_slice_2б
Vsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpReadVariableOp_sequential_133_bidirectional_133_backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02X
Vsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpќ
Gsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMulMatMulKsequential_133/bidirectional_133/backward_lstm_133/strided_slice_2:output:0^sequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2I
Gsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMulз
Xsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOpasequential_133_bidirectional_133_backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02Z
Xsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpј
Isequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMul_1MatMulAsequential_133/bidirectional_133/backward_lstm_133/zeros:output:0`sequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2K
Isequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMul_1№
Dsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/addAddV2Qsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMul:product:0Ssequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2F
Dsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/addа
Wsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOp`sequential_133_bidirectional_133_backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02Y
Wsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp§
Hsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/BiasAddBiasAddHsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/add:z:0_sequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2J
Hsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/BiasAddц
Psequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2R
Psequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/split/split_dimУ
Fsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/splitSplitYsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/split/split_dim:output:0Qsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2H
Fsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/splitЂ
Hsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/SigmoidSigmoidOsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22J
Hsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/SigmoidІ
Jsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/Sigmoid_1SigmoidOsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22L
Jsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/Sigmoid_1к
Dsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/mulMulNsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/Sigmoid_1:y:0Csequential_133/bidirectional_133/backward_lstm_133/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22F
Dsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/mul
Esequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/ReluReluOsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22G
Esequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/Reluь
Fsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/mul_1MulLsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/Sigmoid:y:0Ssequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22H
Fsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/mul_1с
Fsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/add_1AddV2Hsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/mul:z:0Jsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22H
Fsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/add_1І
Jsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/Sigmoid_2SigmoidOsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22L
Jsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/Sigmoid_2
Gsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/Relu_1ReluJsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22I
Gsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/Relu_1№
Fsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/mul_2MulNsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/Sigmoid_2:y:0Usequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22H
Fsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/mul_2ѕ
Psequential_133/bidirectional_133/backward_lstm_133/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2R
Psequential_133/bidirectional_133/backward_lstm_133/TensorArrayV2_1/element_shape
Bsequential_133/bidirectional_133/backward_lstm_133/TensorArrayV2_1TensorListReserveYsequential_133/bidirectional_133/backward_lstm_133/TensorArrayV2_1/element_shape:output:0Ksequential_133/bidirectional_133/backward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02D
Bsequential_133/bidirectional_133/backward_lstm_133/TensorArrayV2_1Д
7sequential_133/bidirectional_133/backward_lstm_133/timeConst*
_output_shapes
: *
dtype0*
value	B : 29
7sequential_133/bidirectional_133/backward_lstm_133/timeж
Hsequential_133/bidirectional_133/backward_lstm_133/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hsequential_133/bidirectional_133/backward_lstm_133/Max/reduction_indicesЈ
6sequential_133/bidirectional_133/backward_lstm_133/MaxMax;sequential_133/bidirectional_133/backward_lstm_133/Cast:y:0Qsequential_133/bidirectional_133/backward_lstm_133/Max/reduction_indices:output:0*
T0*
_output_shapes
: 28
6sequential_133/bidirectional_133/backward_lstm_133/MaxЖ
8sequential_133/bidirectional_133/backward_lstm_133/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_133/bidirectional_133/backward_lstm_133/sub/y
6sequential_133/bidirectional_133/backward_lstm_133/subSub?sequential_133/bidirectional_133/backward_lstm_133/Max:output:0Asequential_133/bidirectional_133/backward_lstm_133/sub/y:output:0*
T0*
_output_shapes
: 28
6sequential_133/bidirectional_133/backward_lstm_133/subЂ
8sequential_133/bidirectional_133/backward_lstm_133/Sub_1Sub:sequential_133/bidirectional_133/backward_lstm_133/sub:z:0;sequential_133/bidirectional_133/backward_lstm_133/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2:
8sequential_133/bidirectional_133/backward_lstm_133/Sub_1
=sequential_133/bidirectional_133/backward_lstm_133/zeros_like	ZerosLikeJsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22?
=sequential_133/bidirectional_133/backward_lstm_133/zeros_likeх
Ksequential_133/bidirectional_133/backward_lstm_133/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2M
Ksequential_133/bidirectional_133/backward_lstm_133/while/maximum_iterationsа
Esequential_133/bidirectional_133/backward_lstm_133/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2G
Esequential_133/bidirectional_133/backward_lstm_133/while/loop_counterж
8sequential_133/bidirectional_133/backward_lstm_133/whileWhileNsequential_133/bidirectional_133/backward_lstm_133/while/loop_counter:output:0Tsequential_133/bidirectional_133/backward_lstm_133/while/maximum_iterations:output:0@sequential_133/bidirectional_133/backward_lstm_133/time:output:0Ksequential_133/bidirectional_133/backward_lstm_133/TensorArrayV2_1:handle:0Asequential_133/bidirectional_133/backward_lstm_133/zeros_like:y:0Asequential_133/bidirectional_133/backward_lstm_133/zeros:output:0Csequential_133/bidirectional_133/backward_lstm_133/zeros_1:output:0Ksequential_133/bidirectional_133/backward_lstm_133/strided_slice_1:output:0jsequential_133/bidirectional_133/backward_lstm_133/TensorArrayUnstack/TensorListFromTensor:output_handle:0<sequential_133/bidirectional_133/backward_lstm_133/Sub_1:z:0_sequential_133_bidirectional_133_backward_lstm_133_lstm_cell_401_matmul_readvariableop_resourceasequential_133_bidirectional_133_backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource`sequential_133_bidirectional_133_backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *R
bodyJRH
Fsequential_133_bidirectional_133_backward_lstm_133_while_body_16884164*R
condJRH
Fsequential_133_bidirectional_133_backward_lstm_133_while_cond_16884163*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2:
8sequential_133/bidirectional_133/backward_lstm_133/while
csequential_133/bidirectional_133/backward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2e
csequential_133/bidirectional_133/backward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeН
Usequential_133/bidirectional_133/backward_lstm_133/TensorArrayV2Stack/TensorListStackTensorListStackAsequential_133/bidirectional_133/backward_lstm_133/while:output:3lsequential_133/bidirectional_133/backward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02W
Usequential_133/bidirectional_133/backward_lstm_133/TensorArrayV2Stack/TensorListStackч
Hsequential_133/bidirectional_133/backward_lstm_133/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2J
Hsequential_133/bidirectional_133/backward_lstm_133/strided_slice_3/stackт
Jsequential_133/bidirectional_133/backward_lstm_133/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jsequential_133/bidirectional_133/backward_lstm_133/strided_slice_3/stack_1т
Jsequential_133/bidirectional_133/backward_lstm_133/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_133/bidirectional_133/backward_lstm_133/strided_slice_3/stack_2Ь
Bsequential_133/bidirectional_133/backward_lstm_133/strided_slice_3StridedSlice^sequential_133/bidirectional_133/backward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0Qsequential_133/bidirectional_133/backward_lstm_133/strided_slice_3/stack:output:0Ssequential_133/bidirectional_133/backward_lstm_133/strided_slice_3/stack_1:output:0Ssequential_133/bidirectional_133/backward_lstm_133/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2D
Bsequential_133/bidirectional_133/backward_lstm_133/strided_slice_3п
Csequential_133/bidirectional_133/backward_lstm_133/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2E
Csequential_133/bidirectional_133/backward_lstm_133/transpose_1/permњ
>sequential_133/bidirectional_133/backward_lstm_133/transpose_1	Transpose^sequential_133/bidirectional_133/backward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0Lsequential_133/bidirectional_133/backward_lstm_133/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22@
>sequential_133/bidirectional_133/backward_lstm_133/transpose_1Ь
:sequential_133/bidirectional_133/backward_lstm_133/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2<
:sequential_133/bidirectional_133/backward_lstm_133/runtime
,sequential_133/bidirectional_133/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_133/bidirectional_133/concat/axisщ
'sequential_133/bidirectional_133/concatConcatV2Jsequential_133/bidirectional_133/forward_lstm_133/strided_slice_3:output:0Ksequential_133/bidirectional_133/backward_lstm_133/strided_slice_3:output:05sequential_133/bidirectional_133/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2)
'sequential_133/bidirectional_133/concatи
.sequential_133/dense_133/MatMul/ReadVariableOpReadVariableOp7sequential_133_dense_133_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_133/dense_133/MatMul/ReadVariableOpш
sequential_133/dense_133/MatMulMatMul0sequential_133/bidirectional_133/concat:output:06sequential_133/dense_133/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
sequential_133/dense_133/MatMulз
/sequential_133/dense_133/BiasAdd/ReadVariableOpReadVariableOp8sequential_133_dense_133_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_133/dense_133/BiasAdd/ReadVariableOpх
 sequential_133/dense_133/BiasAddBiasAdd)sequential_133/dense_133/MatMul:product:07sequential_133/dense_133/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 sequential_133/dense_133/BiasAddЌ
 sequential_133/dense_133/SigmoidSigmoid)sequential_133/dense_133/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 sequential_133/dense_133/Sigmoid
IdentityIdentity$sequential_133/dense_133/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityП
NoOpNoOpX^sequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpW^sequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpY^sequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp9^sequential_133/bidirectional_133/backward_lstm_133/whileW^sequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpV^sequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpX^sequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp8^sequential_133/bidirectional_133/forward_lstm_133/while0^sequential_133/dense_133/BiasAdd/ReadVariableOp/^sequential_133/dense_133/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2В
Wsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpWsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp2А
Vsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpVsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp2Д
Xsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpXsequential_133/bidirectional_133/backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp2t
8sequential_133/bidirectional_133/backward_lstm_133/while8sequential_133/bidirectional_133/backward_lstm_133/while2А
Vsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpVsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp2Ў
Usequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpUsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp2В
Wsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOpWsequential_133/bidirectional_133/forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp2r
7sequential_133/bidirectional_133/forward_lstm_133/while7sequential_133/bidirectional_133/forward_lstm_133/while2b
/sequential_133/dense_133/BiasAdd/ReadVariableOp/sequential_133/dense_133/BiasAdd/ReadVariableOp2`
.sequential_133/dense_133/MatMul/ReadVariableOp.sequential_133/dense_133/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameargs_0:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameargs_0
Ф§
у
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_16887696
inputs_0P
=forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource:	ШR
?forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource:	2ШM
>forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource:	ШQ
>backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource:	ШS
@backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource:	2ШN
?backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource:	Ш
identityЂ6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpЂ5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpЂ7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpЂbackward_lstm_133/whileЂ5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpЂ4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpЂ6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOpЂforward_lstm_133/whileh
forward_lstm_133/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_133/Shape
$forward_lstm_133/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_133/strided_slice/stack
&forward_lstm_133/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_133/strided_slice/stack_1
&forward_lstm_133/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_133/strided_slice/stack_2Ш
forward_lstm_133/strided_sliceStridedSliceforward_lstm_133/Shape:output:0-forward_lstm_133/strided_slice/stack:output:0/forward_lstm_133/strided_slice/stack_1:output:0/forward_lstm_133/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_133/strided_slice~
forward_lstm_133/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_133/zeros/mul/yА
forward_lstm_133/zeros/mulMul'forward_lstm_133/strided_slice:output:0%forward_lstm_133/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros/mul
forward_lstm_133/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_133/zeros/Less/yЋ
forward_lstm_133/zeros/LessLessforward_lstm_133/zeros/mul:z:0&forward_lstm_133/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros/Less
forward_lstm_133/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_133/zeros/packed/1Ч
forward_lstm_133/zeros/packedPack'forward_lstm_133/strided_slice:output:0(forward_lstm_133/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_133/zeros/packed
forward_lstm_133/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_133/zeros/ConstЙ
forward_lstm_133/zerosFill&forward_lstm_133/zeros/packed:output:0%forward_lstm_133/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/zeros
forward_lstm_133/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_133/zeros_1/mul/yЖ
forward_lstm_133/zeros_1/mulMul'forward_lstm_133/strided_slice:output:0'forward_lstm_133/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros_1/mul
forward_lstm_133/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
forward_lstm_133/zeros_1/Less/yГ
forward_lstm_133/zeros_1/LessLess forward_lstm_133/zeros_1/mul:z:0(forward_lstm_133/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros_1/Less
!forward_lstm_133/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_133/zeros_1/packed/1Э
forward_lstm_133/zeros_1/packedPack'forward_lstm_133/strided_slice:output:0*forward_lstm_133/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_133/zeros_1/packed
forward_lstm_133/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_133/zeros_1/ConstС
forward_lstm_133/zeros_1Fill(forward_lstm_133/zeros_1/packed:output:0'forward_lstm_133/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/zeros_1
forward_lstm_133/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_133/transpose/permС
forward_lstm_133/transpose	Transposeinputs_0(forward_lstm_133/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
forward_lstm_133/transpose
forward_lstm_133/Shape_1Shapeforward_lstm_133/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_133/Shape_1
&forward_lstm_133/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_133/strided_slice_1/stack
(forward_lstm_133/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_1/stack_1
(forward_lstm_133/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_1/stack_2д
 forward_lstm_133/strided_slice_1StridedSlice!forward_lstm_133/Shape_1:output:0/forward_lstm_133/strided_slice_1/stack:output:01forward_lstm_133/strided_slice_1/stack_1:output:01forward_lstm_133/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_133/strided_slice_1Ї
,forward_lstm_133/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,forward_lstm_133/TensorArrayV2/element_shapeі
forward_lstm_133/TensorArrayV2TensorListReserve5forward_lstm_133/TensorArrayV2/element_shape:output:0)forward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_133/TensorArrayV2с
Fforward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2H
Fforward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8forward_lstm_133/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_133/transpose:y:0Oforward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_133/TensorArrayUnstack/TensorListFromTensor
&forward_lstm_133/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_133/strided_slice_2/stack
(forward_lstm_133/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_2/stack_1
(forward_lstm_133/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_2/stack_2ы
 forward_lstm_133/strided_slice_2StridedSliceforward_lstm_133/transpose:y:0/forward_lstm_133/strided_slice_2/stack:output:01forward_lstm_133/strided_slice_2/stack_1:output:01forward_lstm_133/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2"
 forward_lstm_133/strided_slice_2ы
4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpReadVariableOp=forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpє
%forward_lstm_133/lstm_cell_400/MatMulMatMul)forward_lstm_133/strided_slice_2:output:0<forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_133/lstm_cell_400/MatMulё
6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOp?forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp№
'forward_lstm_133/lstm_cell_400/MatMul_1MatMulforward_lstm_133/zeros:output:0>forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_133/lstm_cell_400/MatMul_1ш
"forward_lstm_133/lstm_cell_400/addAddV2/forward_lstm_133/lstm_cell_400/MatMul:product:01forward_lstm_133/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_133/lstm_cell_400/addъ
5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOp>forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpѕ
&forward_lstm_133/lstm_cell_400/BiasAddBiasAdd&forward_lstm_133/lstm_cell_400/add:z:0=forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_133/lstm_cell_400/BiasAddЂ
.forward_lstm_133/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.forward_lstm_133/lstm_cell_400/split/split_dimЛ
$forward_lstm_133/lstm_cell_400/splitSplit7forward_lstm_133/lstm_cell_400/split/split_dim:output:0/forward_lstm_133/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$forward_lstm_133/lstm_cell_400/splitМ
&forward_lstm_133/lstm_cell_400/SigmoidSigmoid-forward_lstm_133/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_133/lstm_cell_400/SigmoidР
(forward_lstm_133/lstm_cell_400/Sigmoid_1Sigmoid-forward_lstm_133/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_133/lstm_cell_400/Sigmoid_1в
"forward_lstm_133/lstm_cell_400/mulMul,forward_lstm_133/lstm_cell_400/Sigmoid_1:y:0!forward_lstm_133/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_133/lstm_cell_400/mulГ
#forward_lstm_133/lstm_cell_400/ReluRelu-forward_lstm_133/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_133/lstm_cell_400/Reluф
$forward_lstm_133/lstm_cell_400/mul_1Mul*forward_lstm_133/lstm_cell_400/Sigmoid:y:01forward_lstm_133/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_133/lstm_cell_400/mul_1й
$forward_lstm_133/lstm_cell_400/add_1AddV2&forward_lstm_133/lstm_cell_400/mul:z:0(forward_lstm_133/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_133/lstm_cell_400/add_1Р
(forward_lstm_133/lstm_cell_400/Sigmoid_2Sigmoid-forward_lstm_133/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_133/lstm_cell_400/Sigmoid_2В
%forward_lstm_133/lstm_cell_400/Relu_1Relu(forward_lstm_133/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_133/lstm_cell_400/Relu_1ш
$forward_lstm_133/lstm_cell_400/mul_2Mul,forward_lstm_133/lstm_cell_400/Sigmoid_2:y:03forward_lstm_133/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_133/lstm_cell_400/mul_2Б
.forward_lstm_133/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.forward_lstm_133/TensorArrayV2_1/element_shapeќ
 forward_lstm_133/TensorArrayV2_1TensorListReserve7forward_lstm_133/TensorArrayV2_1/element_shape:output:0)forward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_133/TensorArrayV2_1p
forward_lstm_133/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_133/timeЁ
)forward_lstm_133/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)forward_lstm_133/while/maximum_iterations
#forward_lstm_133/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_133/while/loop_counter
forward_lstm_133/whileWhile,forward_lstm_133/while/loop_counter:output:02forward_lstm_133/while/maximum_iterations:output:0forward_lstm_133/time:output:0)forward_lstm_133/TensorArrayV2_1:handle:0forward_lstm_133/zeros:output:0!forward_lstm_133/zeros_1:output:0)forward_lstm_133/strided_slice_1:output:0Hforward_lstm_133/TensorArrayUnstack/TensorListFromTensor:output_handle:0=forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource?forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource>forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *0
body(R&
$forward_lstm_133_while_body_16887461*0
cond(R&
$forward_lstm_133_while_cond_16887460*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
forward_lstm_133/whileз
Aforward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Aforward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3forward_lstm_133/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_133/while:output:3Jforward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3forward_lstm_133/TensorArrayV2Stack/TensorListStackЃ
&forward_lstm_133/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&forward_lstm_133/strided_slice_3/stack
(forward_lstm_133/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_133/strided_slice_3/stack_1
(forward_lstm_133/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_3/stack_2
 forward_lstm_133/strided_slice_3StridedSlice<forward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_133/strided_slice_3/stack:output:01forward_lstm_133/strided_slice_3/stack_1:output:01forward_lstm_133/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 forward_lstm_133/strided_slice_3
!forward_lstm_133/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_133/transpose_1/permђ
forward_lstm_133/transpose_1	Transpose<forward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_133/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_133/transpose_1
forward_lstm_133/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_133/runtimej
backward_lstm_133/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_133/Shape
%backward_lstm_133/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_133/strided_slice/stack
'backward_lstm_133/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_133/strided_slice/stack_1
'backward_lstm_133/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_133/strided_slice/stack_2Ю
backward_lstm_133/strided_sliceStridedSlice backward_lstm_133/Shape:output:0.backward_lstm_133/strided_slice/stack:output:00backward_lstm_133/strided_slice/stack_1:output:00backward_lstm_133/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_133/strided_slice
backward_lstm_133/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_133/zeros/mul/yД
backward_lstm_133/zeros/mulMul(backward_lstm_133/strided_slice:output:0&backward_lstm_133/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/zeros/mul
backward_lstm_133/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_133/zeros/Less/yЏ
backward_lstm_133/zeros/LessLessbackward_lstm_133/zeros/mul:z:0'backward_lstm_133/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/zeros/Less
 backward_lstm_133/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_133/zeros/packed/1Ы
backward_lstm_133/zeros/packedPack(backward_lstm_133/strided_slice:output:0)backward_lstm_133/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_133/zeros/packed
backward_lstm_133/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_133/zeros/ConstН
backward_lstm_133/zerosFill'backward_lstm_133/zeros/packed:output:0&backward_lstm_133/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_133/zeros
backward_lstm_133/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_133/zeros_1/mul/yК
backward_lstm_133/zeros_1/mulMul(backward_lstm_133/strided_slice:output:0(backward_lstm_133/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/zeros_1/mul
 backward_lstm_133/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 backward_lstm_133/zeros_1/Less/yЗ
backward_lstm_133/zeros_1/LessLess!backward_lstm_133/zeros_1/mul:z:0)backward_lstm_133/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_133/zeros_1/Less
"backward_lstm_133/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_133/zeros_1/packed/1б
 backward_lstm_133/zeros_1/packedPack(backward_lstm_133/strided_slice:output:0+backward_lstm_133/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_133/zeros_1/packed
backward_lstm_133/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_133/zeros_1/ConstХ
backward_lstm_133/zeros_1Fill)backward_lstm_133/zeros_1/packed:output:0(backward_lstm_133/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_133/zeros_1
 backward_lstm_133/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_133/transpose/permФ
backward_lstm_133/transpose	Transposeinputs_0)backward_lstm_133/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_133/transpose
backward_lstm_133/Shape_1Shapebackward_lstm_133/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_133/Shape_1
'backward_lstm_133/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_133/strided_slice_1/stack 
)backward_lstm_133/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_1/stack_1 
)backward_lstm_133/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_1/stack_2к
!backward_lstm_133/strided_slice_1StridedSlice"backward_lstm_133/Shape_1:output:00backward_lstm_133/strided_slice_1/stack:output:02backward_lstm_133/strided_slice_1/stack_1:output:02backward_lstm_133/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_133/strided_slice_1Љ
-backward_lstm_133/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2/
-backward_lstm_133/TensorArrayV2/element_shapeњ
backward_lstm_133/TensorArrayV2TensorListReserve6backward_lstm_133/TensorArrayV2/element_shape:output:0*backward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_133/TensorArrayV2
 backward_lstm_133/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_133/ReverseV2/axisл
backward_lstm_133/ReverseV2	ReverseV2backward_lstm_133/transpose:y:0)backward_lstm_133/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_133/ReverseV2у
Gbackward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2I
Gbackward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeХ
9backward_lstm_133/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_133/ReverseV2:output:0Pbackward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_133/TensorArrayUnstack/TensorListFromTensor
'backward_lstm_133/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_133/strided_slice_2/stack 
)backward_lstm_133/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_2/stack_1 
)backward_lstm_133/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_2/stack_2ё
!backward_lstm_133/strided_slice_2StridedSlicebackward_lstm_133/transpose:y:00backward_lstm_133/strided_slice_2/stack:output:02backward_lstm_133/strided_slice_2/stack_1:output:02backward_lstm_133/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2#
!backward_lstm_133/strided_slice_2ю
5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpReadVariableOp>backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype027
5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpј
&backward_lstm_133/lstm_cell_401/MatMulMatMul*backward_lstm_133/strided_slice_2:output:0=backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_133/lstm_cell_401/MatMulє
7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOp@backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype029
7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpє
(backward_lstm_133/lstm_cell_401/MatMul_1MatMul backward_lstm_133/zeros:output:0?backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_133/lstm_cell_401/MatMul_1ь
#backward_lstm_133/lstm_cell_401/addAddV20backward_lstm_133/lstm_cell_401/MatMul:product:02backward_lstm_133/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_133/lstm_cell_401/addэ
6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOp?backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype028
6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpљ
'backward_lstm_133/lstm_cell_401/BiasAddBiasAdd'backward_lstm_133/lstm_cell_401/add:z:0>backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_133/lstm_cell_401/BiasAddЄ
/backward_lstm_133/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/backward_lstm_133/lstm_cell_401/split/split_dimП
%backward_lstm_133/lstm_cell_401/splitSplit8backward_lstm_133/lstm_cell_401/split/split_dim:output:00backward_lstm_133/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2'
%backward_lstm_133/lstm_cell_401/splitП
'backward_lstm_133/lstm_cell_401/SigmoidSigmoid.backward_lstm_133/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_133/lstm_cell_401/SigmoidУ
)backward_lstm_133/lstm_cell_401/Sigmoid_1Sigmoid.backward_lstm_133/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_133/lstm_cell_401/Sigmoid_1ж
#backward_lstm_133/lstm_cell_401/mulMul-backward_lstm_133/lstm_cell_401/Sigmoid_1:y:0"backward_lstm_133/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_133/lstm_cell_401/mulЖ
$backward_lstm_133/lstm_cell_401/ReluRelu.backward_lstm_133/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_133/lstm_cell_401/Reluш
%backward_lstm_133/lstm_cell_401/mul_1Mul+backward_lstm_133/lstm_cell_401/Sigmoid:y:02backward_lstm_133/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_133/lstm_cell_401/mul_1н
%backward_lstm_133/lstm_cell_401/add_1AddV2'backward_lstm_133/lstm_cell_401/mul:z:0)backward_lstm_133/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_133/lstm_cell_401/add_1У
)backward_lstm_133/lstm_cell_401/Sigmoid_2Sigmoid.backward_lstm_133/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_133/lstm_cell_401/Sigmoid_2Е
&backward_lstm_133/lstm_cell_401/Relu_1Relu)backward_lstm_133/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_133/lstm_cell_401/Relu_1ь
%backward_lstm_133/lstm_cell_401/mul_2Mul-backward_lstm_133/lstm_cell_401/Sigmoid_2:y:04backward_lstm_133/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_133/lstm_cell_401/mul_2Г
/backward_lstm_133/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   21
/backward_lstm_133/TensorArrayV2_1/element_shape
!backward_lstm_133/TensorArrayV2_1TensorListReserve8backward_lstm_133/TensorArrayV2_1/element_shape:output:0*backward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_133/TensorArrayV2_1r
backward_lstm_133/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_133/timeЃ
*backward_lstm_133/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*backward_lstm_133/while/maximum_iterations
$backward_lstm_133/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_133/while/loop_counter 
backward_lstm_133/whileWhile-backward_lstm_133/while/loop_counter:output:03backward_lstm_133/while/maximum_iterations:output:0backward_lstm_133/time:output:0*backward_lstm_133/TensorArrayV2_1:handle:0 backward_lstm_133/zeros:output:0"backward_lstm_133/zeros_1:output:0*backward_lstm_133/strided_slice_1:output:0Ibackward_lstm_133/TensorArrayUnstack/TensorListFromTensor:output_handle:0>backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource@backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource?backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *1
body)R'
%backward_lstm_133_while_body_16887610*1
cond)R'
%backward_lstm_133_while_cond_16887609*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
backward_lstm_133/whileй
Bbackward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2D
Bbackward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeЙ
4backward_lstm_133/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_133/while:output:3Kbackward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype026
4backward_lstm_133/TensorArrayV2Stack/TensorListStackЅ
'backward_lstm_133/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2)
'backward_lstm_133/strided_slice_3/stack 
)backward_lstm_133/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_133/strided_slice_3/stack_1 
)backward_lstm_133/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_3/stack_2
!backward_lstm_133/strided_slice_3StridedSlice=backward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_133/strided_slice_3/stack:output:02backward_lstm_133/strided_slice_3/stack_1:output:02backward_lstm_133/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2#
!backward_lstm_133/strided_slice_3
"backward_lstm_133/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_133/transpose_1/permі
backward_lstm_133/transpose_1	Transpose=backward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_133/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_133/transpose_1
backward_lstm_133/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_133/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisФ
concatConcatV2)forward_lstm_133/strided_slice_3:output:0*backward_lstm_133/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

Identityд
NoOpNoOp7^backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp6^backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp8^backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp^backward_lstm_133/while6^forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp5^forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp7^forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp^forward_lstm_133/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2p
6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp2n
5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp2r
7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp22
backward_lstm_133/whilebackward_lstm_133/while2n
5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp2l
4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp2p
6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp20
forward_lstm_133/whileforward_lstm_133/while:g c
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
ъ
М
%backward_lstm_133_while_cond_16886606@
<backward_lstm_133_while_backward_lstm_133_while_loop_counterF
Bbackward_lstm_133_while_backward_lstm_133_while_maximum_iterations'
#backward_lstm_133_while_placeholder)
%backward_lstm_133_while_placeholder_1)
%backward_lstm_133_while_placeholder_2)
%backward_lstm_133_while_placeholder_3)
%backward_lstm_133_while_placeholder_4B
>backward_lstm_133_while_less_backward_lstm_133_strided_slice_1Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16886606___redundant_placeholder0Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16886606___redundant_placeholder1Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16886606___redundant_placeholder2Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16886606___redundant_placeholder3Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16886606___redundant_placeholder4$
 backward_lstm_133_while_identity
Ъ
backward_lstm_133/while/LessLess#backward_lstm_133_while_placeholder>backward_lstm_133_while_less_backward_lstm_133_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_133/while/Less
 backward_lstm_133/while/IdentityIdentity backward_lstm_133/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_133/while/Identity"M
 backward_lstm_133_while_identity)backward_lstm_133/while/Identity:output:0*(
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
п
Э
while_cond_16885200
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16885200___redundant_placeholder06
2while_while_cond_16885200___redundant_placeholder16
2while_while_cond_16885200___redundant_placeholder26
2while_while_cond_16885200___redundant_placeholder3
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
&
ј
while_body_16884357
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_400_16884381_0:	Ш1
while_lstm_cell_400_16884383_0:	2Ш-
while_lstm_cell_400_16884385_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_400_16884381:	Ш/
while_lstm_cell_400_16884383:	2Ш+
while_lstm_cell_400_16884385:	ШЂ+while/lstm_cell_400/StatefulPartitionedCallУ
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
)while/TensorArrayV2Read/TensorListGetItemя
+while/lstm_cell_400/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_400_16884381_0while_lstm_cell_400_16884383_0while_lstm_cell_400_16884385_0*
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
GPU 2J 8 *T
fORM
K__inference_lstm_cell_400_layer_call_and_return_conditional_losses_168843432-
+while/lstm_cell_400/StatefulPartitionedCallј
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_400/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_400/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѕ
while/Identity_5Identity4while/lstm_cell_400/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp,^while/lstm_cell_400/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_400_16884381while_lstm_cell_400_16884381_0">
while_lstm_cell_400_16884383while_lstm_cell_400_16884383_0">
while_lstm_cell_400_16884385while_lstm_cell_400_16884385_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2Z
+while/lstm_cell_400/StatefulPartitionedCall+while/lstm_cell_400/StatefulPartitionedCall: 
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
р
Р
3__inference_forward_lstm_133_layer_call_fn_16888767

inputs
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
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
GPU 2J 8 *W
fRRP
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_168856932
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
&
ј
while_body_16884567
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_400_16884591_0:	Ш1
while_lstm_cell_400_16884593_0:	2Ш-
while_lstm_cell_400_16884595_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_400_16884591:	Ш/
while_lstm_cell_400_16884593:	2Ш+
while_lstm_cell_400_16884595:	ШЂ+while/lstm_cell_400/StatefulPartitionedCallУ
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
)while/TensorArrayV2Read/TensorListGetItemя
+while/lstm_cell_400/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_400_16884591_0while_lstm_cell_400_16884593_0while_lstm_cell_400_16884595_0*
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
GPU 2J 8 *T
fORM
K__inference_lstm_cell_400_layer_call_and_return_conditional_losses_168844892-
+while/lstm_cell_400/StatefulPartitionedCallј
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_400/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_400/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѕ
while/Identity_5Identity4while/lstm_cell_400/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp,^while/lstm_cell_400/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_400_16884591while_lstm_cell_400_16884591_0">
while_lstm_cell_400_16884593while_lstm_cell_400_16884593_0">
while_lstm_cell_400_16884595while_lstm_cell_400_16884595_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2Z
+while/lstm_cell_400/StatefulPartitionedCall+while/lstm_cell_400/StatefulPartitionedCall: 
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
]
Ќ
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_16889382

inputs?
,lstm_cell_400_matmul_readvariableop_resource:	ШA
.lstm_cell_400_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_400_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_400/BiasAdd/ReadVariableOpЂ#lstm_cell_400/MatMul/ReadVariableOpЂ%lstm_cell_400/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_400/MatMul/ReadVariableOpReadVariableOp,lstm_cell_400_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_400/MatMul/ReadVariableOpА
lstm_cell_400/MatMulMatMulstrided_slice_2:output:0+lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/MatMulО
%lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_400_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_400/MatMul_1/ReadVariableOpЌ
lstm_cell_400/MatMul_1MatMulzeros:output:0-lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/MatMul_1Є
lstm_cell_400/addAddV2lstm_cell_400/MatMul:product:0 lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/addЗ
$lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_400_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_400/BiasAdd/ReadVariableOpБ
lstm_cell_400/BiasAddBiasAddlstm_cell_400/add:z:0,lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/BiasAdd
lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_400/split/split_dimї
lstm_cell_400/splitSplit&lstm_cell_400/split/split_dim:output:0lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_400/split
lstm_cell_400/SigmoidSigmoidlstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Sigmoid
lstm_cell_400/Sigmoid_1Sigmoidlstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Sigmoid_1
lstm_cell_400/mulMullstm_cell_400/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/mul
lstm_cell_400/ReluRelulstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Relu 
lstm_cell_400/mul_1Mullstm_cell_400/Sigmoid:y:0 lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/mul_1
lstm_cell_400/add_1AddV2lstm_cell_400/mul:z:0lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/add_1
lstm_cell_400/Sigmoid_2Sigmoidlstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Sigmoid_2
lstm_cell_400/Relu_1Relulstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Relu_1Є
lstm_cell_400/mul_2Mullstm_cell_400/Sigmoid_2:y:0"lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_400_matmul_readvariableop_resource.lstm_cell_400_matmul_1_readvariableop_resource-lstm_cell_400_biasadd_readvariableop_resource*
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
bodyR
while_body_16889298*
condR
while_cond_16889297*K
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
NoOpNoOp%^lstm_cell_400/BiasAdd/ReadVariableOp$^lstm_cell_400/MatMul/ReadVariableOp&^lstm_cell_400/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_400/BiasAdd/ReadVariableOp$lstm_cell_400/BiasAdd/ReadVariableOp2J
#lstm_cell_400/MatMul/ReadVariableOp#lstm_cell_400/MatMul/ReadVariableOp2N
%lstm_cell_400/MatMul_1/ReadVariableOp%lstm_cell_400/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
і

K__inference_lstm_cell_401_layer_call_and_return_conditional_losses_16885121

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
п
Ё
$forward_lstm_133_while_cond_16887460>
:forward_lstm_133_while_forward_lstm_133_while_loop_counterD
@forward_lstm_133_while_forward_lstm_133_while_maximum_iterations&
"forward_lstm_133_while_placeholder(
$forward_lstm_133_while_placeholder_1(
$forward_lstm_133_while_placeholder_2(
$forward_lstm_133_while_placeholder_3@
<forward_lstm_133_while_less_forward_lstm_133_strided_slice_1X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16887460___redundant_placeholder0X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16887460___redundant_placeholder1X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16887460___redundant_placeholder2X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16887460___redundant_placeholder3#
forward_lstm_133_while_identity
Х
forward_lstm_133/while/LessLess"forward_lstm_133_while_placeholder<forward_lstm_133_while_less_forward_lstm_133_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_133/while/Less
forward_lstm_133/while/IdentityIdentityforward_lstm_133/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_133/while/Identity"K
forward_lstm_133_while_identity(forward_lstm_133/while/Identity:output:0*(
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
п
Э
while_cond_16884566
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16884566___redundant_placeholder06
2while_while_cond_16884566___redundant_placeholder16
2while_while_cond_16884566___redundant_placeholder26
2while_while_cond_16884566___redundant_placeholder3
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
п
Э
while_cond_16889647
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16889647___redundant_placeholder06
2while_while_cond_16889647___redundant_placeholder16
2while_while_cond_16889647___redundant_placeholder26
2while_while_cond_16889647___redundant_placeholder3
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
т
С
4__inference_backward_lstm_133_layer_call_fn_16889415

inputs
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
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
GPU 2J 8 *X
fSRQ
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_168858532
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
ж
У
4__inference_backward_lstm_133_layer_call_fn_16889393
inputs_0
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
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
GPU 2J 8 *X
fSRQ
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_168850582
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
X
ћ
$forward_lstm_133_while_body_16887461>
:forward_lstm_133_while_forward_lstm_133_while_loop_counterD
@forward_lstm_133_while_forward_lstm_133_while_maximum_iterations&
"forward_lstm_133_while_placeholder(
$forward_lstm_133_while_placeholder_1(
$forward_lstm_133_while_placeholder_2(
$forward_lstm_133_while_placeholder_3=
9forward_lstm_133_while_forward_lstm_133_strided_slice_1_0y
uforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0X
Eforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0:	ШZ
Gforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0:	2ШU
Fforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0:	Ш#
forward_lstm_133_while_identity%
!forward_lstm_133_while_identity_1%
!forward_lstm_133_while_identity_2%
!forward_lstm_133_while_identity_3%
!forward_lstm_133_while_identity_4%
!forward_lstm_133_while_identity_5;
7forward_lstm_133_while_forward_lstm_133_strided_slice_1w
sforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensorV
Cforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource:	ШX
Eforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource:	2ШS
Dforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource:	ШЂ;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOpЂ:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOpЂ<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOpх
Hforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2J
Hforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeТ
:forward_lstm_133/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_133_while_placeholderQforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02<
:forward_lstm_133/while/TensorArrayV2Read/TensorListGetItemџ
:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOpReadVariableOpEforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp
+forward_lstm_133/while/lstm_cell_400/MatMulMatMulAforward_lstm_133/while/TensorArrayV2Read/TensorListGetItem:item:0Bforward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_133/while/lstm_cell_400/MatMul
<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOpGforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp
-forward_lstm_133/while/lstm_cell_400/MatMul_1MatMul$forward_lstm_133_while_placeholder_2Dforward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-forward_lstm_133/while/lstm_cell_400/MatMul_1
(forward_lstm_133/while/lstm_cell_400/addAddV25forward_lstm_133/while/lstm_cell_400/MatMul:product:07forward_lstm_133/while/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_133/while/lstm_cell_400/addў
;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOpFforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp
,forward_lstm_133/while/lstm_cell_400/BiasAddBiasAdd,forward_lstm_133/while/lstm_cell_400/add:z:0Cforward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_133/while/lstm_cell_400/BiasAddЎ
4forward_lstm_133/while/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4forward_lstm_133/while/lstm_cell_400/split/split_dimг
*forward_lstm_133/while/lstm_cell_400/splitSplit=forward_lstm_133/while/lstm_cell_400/split/split_dim:output:05forward_lstm_133/while/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*forward_lstm_133/while/lstm_cell_400/splitЮ
,forward_lstm_133/while/lstm_cell_400/SigmoidSigmoid3forward_lstm_133/while/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,forward_lstm_133/while/lstm_cell_400/Sigmoidв
.forward_lstm_133/while/lstm_cell_400/Sigmoid_1Sigmoid3forward_lstm_133/while/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_133/while/lstm_cell_400/Sigmoid_1ч
(forward_lstm_133/while/lstm_cell_400/mulMul2forward_lstm_133/while/lstm_cell_400/Sigmoid_1:y:0$forward_lstm_133_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_133/while/lstm_cell_400/mulХ
)forward_lstm_133/while/lstm_cell_400/ReluRelu3forward_lstm_133/while/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_133/while/lstm_cell_400/Reluќ
*forward_lstm_133/while/lstm_cell_400/mul_1Mul0forward_lstm_133/while/lstm_cell_400/Sigmoid:y:07forward_lstm_133/while/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_133/while/lstm_cell_400/mul_1ё
*forward_lstm_133/while/lstm_cell_400/add_1AddV2,forward_lstm_133/while/lstm_cell_400/mul:z:0.forward_lstm_133/while/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_133/while/lstm_cell_400/add_1в
.forward_lstm_133/while/lstm_cell_400/Sigmoid_2Sigmoid3forward_lstm_133/while/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_133/while/lstm_cell_400/Sigmoid_2Ф
+forward_lstm_133/while/lstm_cell_400/Relu_1Relu.forward_lstm_133/while/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_133/while/lstm_cell_400/Relu_1
*forward_lstm_133/while/lstm_cell_400/mul_2Mul2forward_lstm_133/while/lstm_cell_400/Sigmoid_2:y:09forward_lstm_133/while/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_133/while/lstm_cell_400/mul_2Ж
;forward_lstm_133/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_133_while_placeholder_1"forward_lstm_133_while_placeholder.forward_lstm_133/while/lstm_cell_400/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_133/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_133/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_133/while/add/y­
forward_lstm_133/while/addAddV2"forward_lstm_133_while_placeholder%forward_lstm_133/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/while/add
forward_lstm_133/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_133/while/add_1/yЫ
forward_lstm_133/while/add_1AddV2:forward_lstm_133_while_forward_lstm_133_while_loop_counter'forward_lstm_133/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/while/add_1Џ
forward_lstm_133/while/IdentityIdentity forward_lstm_133/while/add_1:z:0^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_133/while/Identityг
!forward_lstm_133/while/Identity_1Identity@forward_lstm_133_while_forward_lstm_133_while_maximum_iterations^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_133/while/Identity_1Б
!forward_lstm_133/while/Identity_2Identityforward_lstm_133/while/add:z:0^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_133/while/Identity_2о
!forward_lstm_133/while/Identity_3IdentityKforward_lstm_133/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_133/while/Identity_3в
!forward_lstm_133/while/Identity_4Identity.forward_lstm_133/while/lstm_cell_400/mul_2:z:0^forward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_133/while/Identity_4в
!forward_lstm_133/while/Identity_5Identity.forward_lstm_133/while/lstm_cell_400/add_1:z:0^forward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_133/while/Identity_5Ж
forward_lstm_133/while/NoOpNoOp<^forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp;^forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp=^forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_133/while/NoOp"t
7forward_lstm_133_while_forward_lstm_133_strided_slice_19forward_lstm_133_while_forward_lstm_133_strided_slice_1_0"K
forward_lstm_133_while_identity(forward_lstm_133/while/Identity:output:0"O
!forward_lstm_133_while_identity_1*forward_lstm_133/while/Identity_1:output:0"O
!forward_lstm_133_while_identity_2*forward_lstm_133/while/Identity_2:output:0"O
!forward_lstm_133_while_identity_3*forward_lstm_133/while/Identity_3:output:0"O
!forward_lstm_133_while_identity_4*forward_lstm_133/while/Identity_4:output:0"O
!forward_lstm_133_while_identity_5*forward_lstm_133/while/Identity_5:output:0"
Dforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resourceFforward_lstm_133_while_lstm_cell_400_biasadd_readvariableop_resource_0"
Eforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resourceGforward_lstm_133_while_lstm_cell_400_matmul_1_readvariableop_resource_0"
Cforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resourceEforward_lstm_133_while_lstm_cell_400_matmul_readvariableop_resource_0"ь
sforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensoruforward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2z
;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp;forward_lstm_133/while/lstm_cell_400/BiasAdd/ReadVariableOp2x
:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp:forward_lstm_133/while/lstm_cell_400/MatMul/ReadVariableOp2|
<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp<forward_lstm_133/while/lstm_cell_400/MatMul_1/ReadVariableOp: 
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

в
Fsequential_133_bidirectional_133_backward_lstm_133_while_body_16884164
~sequential_133_bidirectional_133_backward_lstm_133_while_sequential_133_bidirectional_133_backward_lstm_133_while_loop_counter
sequential_133_bidirectional_133_backward_lstm_133_while_sequential_133_bidirectional_133_backward_lstm_133_while_maximum_iterationsH
Dsequential_133_bidirectional_133_backward_lstm_133_while_placeholderJ
Fsequential_133_bidirectional_133_backward_lstm_133_while_placeholder_1J
Fsequential_133_bidirectional_133_backward_lstm_133_while_placeholder_2J
Fsequential_133_bidirectional_133_backward_lstm_133_while_placeholder_3J
Fsequential_133_bidirectional_133_backward_lstm_133_while_placeholder_4
}sequential_133_bidirectional_133_backward_lstm_133_while_sequential_133_bidirectional_133_backward_lstm_133_strided_slice_1_0О
Йsequential_133_bidirectional_133_backward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_sequential_133_bidirectional_133_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0|
xsequential_133_bidirectional_133_backward_lstm_133_while_less_sequential_133_bidirectional_133_backward_lstm_133_sub_1_0z
gsequential_133_bidirectional_133_backward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0:	Ш|
isequential_133_bidirectional_133_backward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0:	2Шw
hsequential_133_bidirectional_133_backward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0:	ШE
Asequential_133_bidirectional_133_backward_lstm_133_while_identityG
Csequential_133_bidirectional_133_backward_lstm_133_while_identity_1G
Csequential_133_bidirectional_133_backward_lstm_133_while_identity_2G
Csequential_133_bidirectional_133_backward_lstm_133_while_identity_3G
Csequential_133_bidirectional_133_backward_lstm_133_while_identity_4G
Csequential_133_bidirectional_133_backward_lstm_133_while_identity_5G
Csequential_133_bidirectional_133_backward_lstm_133_while_identity_6
{sequential_133_bidirectional_133_backward_lstm_133_while_sequential_133_bidirectional_133_backward_lstm_133_strided_slice_1М
Зsequential_133_bidirectional_133_backward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_sequential_133_bidirectional_133_backward_lstm_133_tensorarrayunstack_tensorlistfromtensorz
vsequential_133_bidirectional_133_backward_lstm_133_while_less_sequential_133_bidirectional_133_backward_lstm_133_sub_1x
esequential_133_bidirectional_133_backward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource:	Шz
gsequential_133_bidirectional_133_backward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource:	2Шu
fsequential_133_bidirectional_133_backward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource:	ШЂ]sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOpЂ\sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpЂ^sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOpЉ
jsequential_133/bidirectional_133/backward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2l
jsequential_133/bidirectional_133/backward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shape
\sequential_133/bidirectional_133/backward_lstm_133/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЙsequential_133_bidirectional_133_backward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_sequential_133_bidirectional_133_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0Dsequential_133_bidirectional_133_backward_lstm_133_while_placeholderssequential_133/bidirectional_133/backward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02^
\sequential_133/bidirectional_133/backward_lstm_133/while/TensorArrayV2Read/TensorListGetItemє
=sequential_133/bidirectional_133/backward_lstm_133/while/LessLessxsequential_133_bidirectional_133_backward_lstm_133_while_less_sequential_133_bidirectional_133_backward_lstm_133_sub_1_0Dsequential_133_bidirectional_133_backward_lstm_133_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2?
=sequential_133/bidirectional_133/backward_lstm_133/while/Lessх
\sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpReadVariableOpgsequential_133_bidirectional_133_backward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02^
\sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpІ
Msequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMulMatMulcsequential_133/bidirectional_133/backward_lstm_133/while/TensorArrayV2Read/TensorListGetItem:item:0dsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2O
Msequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMulы
^sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOpisequential_133_bidirectional_133_backward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02`
^sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp
Osequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMul_1MatMulFsequential_133_bidirectional_133_backward_lstm_133_while_placeholder_3fsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2Q
Osequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMul_1
Jsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/addAddV2Wsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMul:product:0Ysequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2L
Jsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/addф
]sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOphsequential_133_bidirectional_133_backward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02_
]sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp
Nsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/BiasAddBiasAddNsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/add:z:0esequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2P
Nsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/BiasAddђ
Vsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2X
Vsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/split/split_dimл
Lsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/splitSplit_sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/split/split_dim:output:0Wsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2N
Lsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/splitД
Nsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/SigmoidSigmoidUsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22P
Nsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/SigmoidИ
Psequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/Sigmoid_1SigmoidUsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22R
Psequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/Sigmoid_1я
Jsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/mulMulTsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/Sigmoid_1:y:0Fsequential_133_bidirectional_133_backward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22L
Jsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/mulЋ
Ksequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/ReluReluUsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22M
Ksequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/Relu
Lsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/mul_1MulRsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/Sigmoid:y:0Ysequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22N
Lsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/mul_1љ
Lsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/add_1AddV2Nsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/mul:z:0Psequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22N
Lsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/add_1И
Psequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/Sigmoid_2SigmoidUsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22R
Psequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/Sigmoid_2Њ
Msequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/Relu_1ReluPsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22O
Msequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/Relu_1
Lsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/mul_2MulTsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/Sigmoid_2:y:0[sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22N
Lsequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/mul_2
?sequential_133/bidirectional_133/backward_lstm_133/while/SelectSelectAsequential_133/bidirectional_133/backward_lstm_133/while/Less:z:0Psequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/mul_2:z:0Fsequential_133_bidirectional_133_backward_lstm_133_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22A
?sequential_133/bidirectional_133/backward_lstm_133/while/Select
Asequential_133/bidirectional_133/backward_lstm_133/while/Select_1SelectAsequential_133/bidirectional_133/backward_lstm_133/while/Less:z:0Psequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/mul_2:z:0Fsequential_133_bidirectional_133_backward_lstm_133_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22C
Asequential_133/bidirectional_133/backward_lstm_133/while/Select_1
Asequential_133/bidirectional_133/backward_lstm_133/while/Select_2SelectAsequential_133/bidirectional_133/backward_lstm_133/while/Less:z:0Psequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/add_1:z:0Fsequential_133_bidirectional_133_backward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22C
Asequential_133/bidirectional_133/backward_lstm_133/while/Select_2и
]sequential_133/bidirectional_133/backward_lstm_133/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemFsequential_133_bidirectional_133_backward_lstm_133_while_placeholder_1Dsequential_133_bidirectional_133_backward_lstm_133_while_placeholderHsequential_133/bidirectional_133/backward_lstm_133/while/Select:output:0*
_output_shapes
: *
element_dtype02_
]sequential_133/bidirectional_133/backward_lstm_133/while/TensorArrayV2Write/TensorListSetItemТ
>sequential_133/bidirectional_133/backward_lstm_133/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_133/bidirectional_133/backward_lstm_133/while/add/yЕ
<sequential_133/bidirectional_133/backward_lstm_133/while/addAddV2Dsequential_133_bidirectional_133_backward_lstm_133_while_placeholderGsequential_133/bidirectional_133/backward_lstm_133/while/add/y:output:0*
T0*
_output_shapes
: 2>
<sequential_133/bidirectional_133/backward_lstm_133/while/addЦ
@sequential_133/bidirectional_133/backward_lstm_133/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2B
@sequential_133/bidirectional_133/backward_lstm_133/while/add_1/yѕ
>sequential_133/bidirectional_133/backward_lstm_133/while/add_1AddV2~sequential_133_bidirectional_133_backward_lstm_133_while_sequential_133_bidirectional_133_backward_lstm_133_while_loop_counterIsequential_133/bidirectional_133/backward_lstm_133/while/add_1/y:output:0*
T0*
_output_shapes
: 2@
>sequential_133/bidirectional_133/backward_lstm_133/while/add_1З
Asequential_133/bidirectional_133/backward_lstm_133/while/IdentityIdentityBsequential_133/bidirectional_133/backward_lstm_133/while/add_1:z:0>^sequential_133/bidirectional_133/backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2C
Asequential_133/bidirectional_133/backward_lstm_133/while/Identityў
Csequential_133/bidirectional_133/backward_lstm_133/while/Identity_1Identitysequential_133_bidirectional_133_backward_lstm_133_while_sequential_133_bidirectional_133_backward_lstm_133_while_maximum_iterations>^sequential_133/bidirectional_133/backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2E
Csequential_133/bidirectional_133/backward_lstm_133/while/Identity_1Й
Csequential_133/bidirectional_133/backward_lstm_133/while/Identity_2Identity@sequential_133/bidirectional_133/backward_lstm_133/while/add:z:0>^sequential_133/bidirectional_133/backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2E
Csequential_133/bidirectional_133/backward_lstm_133/while/Identity_2ц
Csequential_133/bidirectional_133/backward_lstm_133/while/Identity_3Identitymsequential_133/bidirectional_133/backward_lstm_133/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential_133/bidirectional_133/backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2E
Csequential_133/bidirectional_133/backward_lstm_133/while/Identity_3в
Csequential_133/bidirectional_133/backward_lstm_133/while/Identity_4IdentityHsequential_133/bidirectional_133/backward_lstm_133/while/Select:output:0>^sequential_133/bidirectional_133/backward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_133/bidirectional_133/backward_lstm_133/while/Identity_4д
Csequential_133/bidirectional_133/backward_lstm_133/while/Identity_5IdentityJsequential_133/bidirectional_133/backward_lstm_133/while/Select_1:output:0>^sequential_133/bidirectional_133/backward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_133/bidirectional_133/backward_lstm_133/while/Identity_5д
Csequential_133/bidirectional_133/backward_lstm_133/while/Identity_6IdentityJsequential_133/bidirectional_133/backward_lstm_133/while/Select_2:output:0>^sequential_133/bidirectional_133/backward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_133/bidirectional_133/backward_lstm_133/while/Identity_6р
=sequential_133/bidirectional_133/backward_lstm_133/while/NoOpNoOp^^sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp]^sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp_^sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2?
=sequential_133/bidirectional_133/backward_lstm_133/while/NoOp"
Asequential_133_bidirectional_133_backward_lstm_133_while_identityJsequential_133/bidirectional_133/backward_lstm_133/while/Identity:output:0"
Csequential_133_bidirectional_133_backward_lstm_133_while_identity_1Lsequential_133/bidirectional_133/backward_lstm_133/while/Identity_1:output:0"
Csequential_133_bidirectional_133_backward_lstm_133_while_identity_2Lsequential_133/bidirectional_133/backward_lstm_133/while/Identity_2:output:0"
Csequential_133_bidirectional_133_backward_lstm_133_while_identity_3Lsequential_133/bidirectional_133/backward_lstm_133/while/Identity_3:output:0"
Csequential_133_bidirectional_133_backward_lstm_133_while_identity_4Lsequential_133/bidirectional_133/backward_lstm_133/while/Identity_4:output:0"
Csequential_133_bidirectional_133_backward_lstm_133_while_identity_5Lsequential_133/bidirectional_133/backward_lstm_133/while/Identity_5:output:0"
Csequential_133_bidirectional_133_backward_lstm_133_while_identity_6Lsequential_133/bidirectional_133/backward_lstm_133/while/Identity_6:output:0"ђ
vsequential_133_bidirectional_133_backward_lstm_133_while_less_sequential_133_bidirectional_133_backward_lstm_133_sub_1xsequential_133_bidirectional_133_backward_lstm_133_while_less_sequential_133_bidirectional_133_backward_lstm_133_sub_1_0"в
fsequential_133_bidirectional_133_backward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resourcehsequential_133_bidirectional_133_backward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0"д
gsequential_133_bidirectional_133_backward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resourceisequential_133_bidirectional_133_backward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0"а
esequential_133_bidirectional_133_backward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resourcegsequential_133_bidirectional_133_backward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0"ќ
{sequential_133_bidirectional_133_backward_lstm_133_while_sequential_133_bidirectional_133_backward_lstm_133_strided_slice_1}sequential_133_bidirectional_133_backward_lstm_133_while_sequential_133_bidirectional_133_backward_lstm_133_strided_slice_1_0"і
Зsequential_133_bidirectional_133_backward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_sequential_133_bidirectional_133_backward_lstm_133_tensorarrayunstack_tensorlistfromtensorЙsequential_133_bidirectional_133_backward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_sequential_133_bidirectional_133_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2О
]sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp]sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp2М
\sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp\sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp2Р
^sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp^sequential_133/bidirectional_133/backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp: 
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
И
Ѓ
L__inference_sequential_133_layer_call_and_return_conditional_losses_16887294

inputs
inputs_1	-
bidirectional_133_16887275:	Ш-
bidirectional_133_16887277:	2Ш)
bidirectional_133_16887279:	Ш-
bidirectional_133_16887281:	Ш-
bidirectional_133_16887283:	2Ш)
bidirectional_133_16887285:	Ш$
dense_133_16887288:d 
dense_133_16887290:
identityЂ)bidirectional_133/StatefulPartitionedCallЂ!dense_133/StatefulPartitionedCallЪ
)bidirectional_133/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_133_16887275bidirectional_133_16887277bidirectional_133_16887279bidirectional_133_16887281bidirectional_133_16887283bidirectional_133_16887285*
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
GPU 2J 8 *X
fSRQ
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_168871442+
)bidirectional_133/StatefulPartitionedCallЫ
!dense_133/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_133/StatefulPartitionedCall:output:0dense_133_16887288dense_133_16887290*
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
GPU 2J 8 *P
fKRI
G__inference_dense_133_layer_call_and_return_conditional_losses_168867292#
!dense_133/StatefulPartitionedCall
IdentityIdentity*dense_133/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp*^bidirectional_133/StatefulPartitionedCall"^dense_133/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2V
)bidirectional_133/StatefulPartitionedCall)bidirectional_133/StatefulPartitionedCall2F
!dense_133/StatefulPartitionedCall!dense_133/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ
М
%backward_lstm_133_while_cond_16888258@
<backward_lstm_133_while_backward_lstm_133_while_loop_counterF
Bbackward_lstm_133_while_backward_lstm_133_while_maximum_iterations'
#backward_lstm_133_while_placeholder)
%backward_lstm_133_while_placeholder_1)
%backward_lstm_133_while_placeholder_2)
%backward_lstm_133_while_placeholder_3)
%backward_lstm_133_while_placeholder_4B
>backward_lstm_133_while_less_backward_lstm_133_strided_slice_1Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16888258___redundant_placeholder0Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16888258___redundant_placeholder1Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16888258___redundant_placeholder2Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16888258___redundant_placeholder3Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16888258___redundant_placeholder4$
 backward_lstm_133_while_identity
Ъ
backward_lstm_133/while/LessLess#backward_lstm_133_while_placeholder>backward_lstm_133_while_less_backward_lstm_133_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_133/while/Less
 backward_lstm_133/while/IdentityIdentity backward_lstm_133/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_133/while/Identity"M
 backward_lstm_133_while_identity)backward_lstm_133/while/Identity:output:0*(
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
і

K__inference_lstm_cell_400_layer_call_and_return_conditional_losses_16884343

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
№?
л
while_body_16889648
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_401_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_401_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_401_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_401_matmul_readvariableop_resource:	ШG
4while_lstm_cell_401_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_401_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_401/BiasAdd/ReadVariableOpЂ)while/lstm_cell_401/MatMul/ReadVariableOpЂ+while/lstm_cell_401/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_401/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_401_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_401/MatMul/ReadVariableOpк
while/lstm_cell_401/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/MatMulв
+while/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_401_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_401/MatMul_1/ReadVariableOpУ
while/lstm_cell_401/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/MatMul_1М
while/lstm_cell_401/addAddV2$while/lstm_cell_401/MatMul:product:0&while/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/addЫ
*while/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_401_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_401/BiasAdd/ReadVariableOpЩ
while/lstm_cell_401/BiasAddBiasAddwhile/lstm_cell_401/add:z:02while/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/BiasAdd
#while/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_401/split/split_dim
while/lstm_cell_401/splitSplit,while/lstm_cell_401/split/split_dim:output:0$while/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_401/split
while/lstm_cell_401/SigmoidSigmoid"while/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Sigmoid
while/lstm_cell_401/Sigmoid_1Sigmoid"while/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Sigmoid_1Ѓ
while/lstm_cell_401/mulMul!while/lstm_cell_401/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/mul
while/lstm_cell_401/ReluRelu"while/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/ReluИ
while/lstm_cell_401/mul_1Mulwhile/lstm_cell_401/Sigmoid:y:0&while/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/mul_1­
while/lstm_cell_401/add_1AddV2while/lstm_cell_401/mul:z:0while/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/add_1
while/lstm_cell_401/Sigmoid_2Sigmoid"while/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Sigmoid_2
while/lstm_cell_401/Relu_1Reluwhile/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Relu_1М
while/lstm_cell_401/mul_2Mul!while/lstm_cell_401/Sigmoid_2:y:0(while/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_401/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_401/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_401/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_401/BiasAdd/ReadVariableOp*^while/lstm_cell_401/MatMul/ReadVariableOp,^while/lstm_cell_401/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_401_biasadd_readvariableop_resource5while_lstm_cell_401_biasadd_readvariableop_resource_0"n
4while_lstm_cell_401_matmul_1_readvariableop_resource6while_lstm_cell_401_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_401_matmul_readvariableop_resource4while_lstm_cell_401_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_401/BiasAdd/ReadVariableOp*while/lstm_cell_401/BiasAdd/ReadVariableOp2V
)while/lstm_cell_401/MatMul/ReadVariableOp)while/lstm_cell_401/MatMul/ReadVariableOp2Z
+while/lstm_cell_401/MatMul_1/ReadVariableOp+while/lstm_cell_401/MatMul_1/ReadVariableOp: 
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
п
Э
while_cond_16889953
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16889953___redundant_placeholder06
2while_while_cond_16889953___redundant_placeholder16
2while_while_cond_16889953___redundant_placeholder26
2while_while_cond_16889953___redundant_placeholder3
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
љ?
л
while_body_16889954
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_401_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_401_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_401_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_401_matmul_readvariableop_resource:	ШG
4while_lstm_cell_401_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_401_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_401/BiasAdd/ReadVariableOpЂ)while/lstm_cell_401/MatMul/ReadVariableOpЂ+while/lstm_cell_401/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_401/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_401_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_401/MatMul/ReadVariableOpк
while/lstm_cell_401/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/MatMulв
+while/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_401_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_401/MatMul_1/ReadVariableOpУ
while/lstm_cell_401/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/MatMul_1М
while/lstm_cell_401/addAddV2$while/lstm_cell_401/MatMul:product:0&while/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/addЫ
*while/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_401_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_401/BiasAdd/ReadVariableOpЩ
while/lstm_cell_401/BiasAddBiasAddwhile/lstm_cell_401/add:z:02while/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/BiasAdd
#while/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_401/split/split_dim
while/lstm_cell_401/splitSplit,while/lstm_cell_401/split/split_dim:output:0$while/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_401/split
while/lstm_cell_401/SigmoidSigmoid"while/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Sigmoid
while/lstm_cell_401/Sigmoid_1Sigmoid"while/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Sigmoid_1Ѓ
while/lstm_cell_401/mulMul!while/lstm_cell_401/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/mul
while/lstm_cell_401/ReluRelu"while/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/ReluИ
while/lstm_cell_401/mul_1Mulwhile/lstm_cell_401/Sigmoid:y:0&while/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/mul_1­
while/lstm_cell_401/add_1AddV2while/lstm_cell_401/mul:z:0while/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/add_1
while/lstm_cell_401/Sigmoid_2Sigmoid"while/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Sigmoid_2
while/lstm_cell_401/Relu_1Reluwhile/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Relu_1М
while/lstm_cell_401/mul_2Mul!while/lstm_cell_401/Sigmoid_2:y:0(while/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_401/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_401/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_401/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_401/BiasAdd/ReadVariableOp*^while/lstm_cell_401/MatMul/ReadVariableOp,^while/lstm_cell_401/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_401_biasadd_readvariableop_resource5while_lstm_cell_401_biasadd_readvariableop_resource_0"n
4while_lstm_cell_401_matmul_1_readvariableop_resource6while_lstm_cell_401_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_401_matmul_readvariableop_resource4while_lstm_cell_401_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_401/BiasAdd/ReadVariableOp*while/lstm_cell_401/BiasAdd/ReadVariableOp2V
)while/lstm_cell_401/MatMul/ReadVariableOp)while/lstm_cell_401/MatMul/ReadVariableOp2Z
+while/lstm_cell_401/MatMul_1/ReadVariableOp+while/lstm_cell_401/MatMul_1/ReadVariableOp: 
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

д
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_16885864

inputs,
forward_lstm_133_16885694:	Ш,
forward_lstm_133_16885696:	2Ш(
forward_lstm_133_16885698:	Ш-
backward_lstm_133_16885854:	Ш-
backward_lstm_133_16885856:	2Ш)
backward_lstm_133_16885858:	Ш
identityЂ)backward_lstm_133/StatefulPartitionedCallЂ(forward_lstm_133/StatefulPartitionedCallп
(forward_lstm_133/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_133_16885694forward_lstm_133_16885696forward_lstm_133_16885698*
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
GPU 2J 8 *W
fRRP
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_168856932*
(forward_lstm_133/StatefulPartitionedCallх
)backward_lstm_133/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_133_16885854backward_lstm_133_16885856backward_lstm_133_16885858*
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
GPU 2J 8 *X
fSRQ
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_168858532+
)backward_lstm_133/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisд
concatConcatV21forward_lstm_133/StatefulPartitionedCall:output:02backward_lstm_133/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

IdentityЅ
NoOpNoOp*^backward_lstm_133/StatefulPartitionedCall)^forward_lstm_133/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2V
)backward_lstm_133/StatefulPartitionedCall)backward_lstm_133/StatefulPartitionedCall2T
(forward_lstm_133/StatefulPartitionedCall(forward_lstm_133/StatefulPartitionedCall:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
п
Э
while_cond_16888995
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16888995___redundant_placeholder06
2while_while_cond_16888995___redundant_placeholder16
2while_while_cond_16888995___redundant_placeholder26
2while_while_cond_16888995___redundant_placeholder3
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
аf
ф
%backward_lstm_133_while_body_16888617@
<backward_lstm_133_while_backward_lstm_133_while_loop_counterF
Bbackward_lstm_133_while_backward_lstm_133_while_maximum_iterations'
#backward_lstm_133_while_placeholder)
%backward_lstm_133_while_placeholder_1)
%backward_lstm_133_while_placeholder_2)
%backward_lstm_133_while_placeholder_3)
%backward_lstm_133_while_placeholder_4?
;backward_lstm_133_while_backward_lstm_133_strided_slice_1_0{
wbackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_133_while_less_backward_lstm_133_sub_1_0Y
Fbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0:	Ш[
Hbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0:	2ШV
Gbackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0:	Ш$
 backward_lstm_133_while_identity&
"backward_lstm_133_while_identity_1&
"backward_lstm_133_while_identity_2&
"backward_lstm_133_while_identity_3&
"backward_lstm_133_while_identity_4&
"backward_lstm_133_while_identity_5&
"backward_lstm_133_while_identity_6=
9backward_lstm_133_while_backward_lstm_133_strided_slice_1y
ubackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_133_while_less_backward_lstm_133_sub_1W
Dbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource:	ШY
Fbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource:	2ШT
Ebackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource:	ШЂ<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOpЂ;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpЂ=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOpч
Ibackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2K
Ibackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeП
;backward_lstm_133/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_133_while_placeholderRbackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02=
;backward_lstm_133/while/TensorArrayV2Read/TensorListGetItemЯ
backward_lstm_133/while/LessLess6backward_lstm_133_while_less_backward_lstm_133_sub_1_0#backward_lstm_133_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_133/while/Less
;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpReadVariableOpFbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02=
;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpЂ
,backward_lstm_133/while/lstm_cell_401/MatMulMatMulBbackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem:item:0Cbackward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_133/while/lstm_cell_401/MatMul
=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOpHbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02?
=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp
.backward_lstm_133/while/lstm_cell_401/MatMul_1MatMul%backward_lstm_133_while_placeholder_3Ebackward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ20
.backward_lstm_133/while/lstm_cell_401/MatMul_1
)backward_lstm_133/while/lstm_cell_401/addAddV26backward_lstm_133/while/lstm_cell_401/MatMul:product:08backward_lstm_133/while/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_133/while/lstm_cell_401/add
<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOpGbackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02>
<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp
-backward_lstm_133/while/lstm_cell_401/BiasAddBiasAdd-backward_lstm_133/while/lstm_cell_401/add:z:0Dbackward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_133/while/lstm_cell_401/BiasAddА
5backward_lstm_133/while/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5backward_lstm_133/while/lstm_cell_401/split/split_dimз
+backward_lstm_133/while/lstm_cell_401/splitSplit>backward_lstm_133/while/lstm_cell_401/split/split_dim:output:06backward_lstm_133/while/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2-
+backward_lstm_133/while/lstm_cell_401/splitб
-backward_lstm_133/while/lstm_cell_401/SigmoidSigmoid4backward_lstm_133/while/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22/
-backward_lstm_133/while/lstm_cell_401/Sigmoidе
/backward_lstm_133/while/lstm_cell_401/Sigmoid_1Sigmoid4backward_lstm_133/while/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_133/while/lstm_cell_401/Sigmoid_1ы
)backward_lstm_133/while/lstm_cell_401/mulMul3backward_lstm_133/while/lstm_cell_401/Sigmoid_1:y:0%backward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_133/while/lstm_cell_401/mulШ
*backward_lstm_133/while/lstm_cell_401/ReluRelu4backward_lstm_133/while/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_133/while/lstm_cell_401/Relu
+backward_lstm_133/while/lstm_cell_401/mul_1Mul1backward_lstm_133/while/lstm_cell_401/Sigmoid:y:08backward_lstm_133/while/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_133/while/lstm_cell_401/mul_1ѕ
+backward_lstm_133/while/lstm_cell_401/add_1AddV2-backward_lstm_133/while/lstm_cell_401/mul:z:0/backward_lstm_133/while/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_133/while/lstm_cell_401/add_1е
/backward_lstm_133/while/lstm_cell_401/Sigmoid_2Sigmoid4backward_lstm_133/while/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_133/while/lstm_cell_401/Sigmoid_2Ч
,backward_lstm_133/while/lstm_cell_401/Relu_1Relu/backward_lstm_133/while/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_133/while/lstm_cell_401/Relu_1
+backward_lstm_133/while/lstm_cell_401/mul_2Mul3backward_lstm_133/while/lstm_cell_401/Sigmoid_2:y:0:backward_lstm_133/while/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_133/while/lstm_cell_401/mul_2і
backward_lstm_133/while/SelectSelect backward_lstm_133/while/Less:z:0/backward_lstm_133/while/lstm_cell_401/mul_2:z:0%backward_lstm_133_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_133/while/Selectњ
 backward_lstm_133/while/Select_1Select backward_lstm_133/while/Less:z:0/backward_lstm_133/while/lstm_cell_401/mul_2:z:0%backward_lstm_133_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_133/while/Select_1њ
 backward_lstm_133/while/Select_2Select backward_lstm_133/while/Less:z:0/backward_lstm_133/while/lstm_cell_401/add_1:z:0%backward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_133/while/Select_2Г
<backward_lstm_133/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_133_while_placeholder_1#backward_lstm_133_while_placeholder'backward_lstm_133/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_133/while/TensorArrayV2Write/TensorListSetItem
backward_lstm_133/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_133/while/add/yБ
backward_lstm_133/while/addAddV2#backward_lstm_133_while_placeholder&backward_lstm_133/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/while/add
backward_lstm_133/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_133/while/add_1/yа
backward_lstm_133/while/add_1AddV2<backward_lstm_133_while_backward_lstm_133_while_loop_counter(backward_lstm_133/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/while/add_1Г
 backward_lstm_133/while/IdentityIdentity!backward_lstm_133/while/add_1:z:0^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_133/while/Identityи
"backward_lstm_133/while/Identity_1IdentityBbackward_lstm_133_while_backward_lstm_133_while_maximum_iterations^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_133/while/Identity_1Е
"backward_lstm_133/while/Identity_2Identitybackward_lstm_133/while/add:z:0^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_133/while/Identity_2т
"backward_lstm_133/while/Identity_3IdentityLbackward_lstm_133/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_133/while/Identity_3Ю
"backward_lstm_133/while/Identity_4Identity'backward_lstm_133/while/Select:output:0^backward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_133/while/Identity_4а
"backward_lstm_133/while/Identity_5Identity)backward_lstm_133/while/Select_1:output:0^backward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_133/while/Identity_5а
"backward_lstm_133/while/Identity_6Identity)backward_lstm_133/while/Select_2:output:0^backward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_133/while/Identity_6Л
backward_lstm_133/while/NoOpNoOp=^backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp<^backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp>^backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_133/while/NoOp"x
9backward_lstm_133_while_backward_lstm_133_strided_slice_1;backward_lstm_133_while_backward_lstm_133_strided_slice_1_0"M
 backward_lstm_133_while_identity)backward_lstm_133/while/Identity:output:0"Q
"backward_lstm_133_while_identity_1+backward_lstm_133/while/Identity_1:output:0"Q
"backward_lstm_133_while_identity_2+backward_lstm_133/while/Identity_2:output:0"Q
"backward_lstm_133_while_identity_3+backward_lstm_133/while/Identity_3:output:0"Q
"backward_lstm_133_while_identity_4+backward_lstm_133/while/Identity_4:output:0"Q
"backward_lstm_133_while_identity_5+backward_lstm_133/while/Identity_5:output:0"Q
"backward_lstm_133_while_identity_6+backward_lstm_133/while/Identity_6:output:0"n
4backward_lstm_133_while_less_backward_lstm_133_sub_16backward_lstm_133_while_less_backward_lstm_133_sub_1_0"
Ebackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resourceGbackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0"
Fbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resourceHbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0"
Dbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resourceFbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0"№
ubackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2|
<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp2z
;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp2~
=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp: 
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
]
Ў
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_16889080
inputs_0?
,lstm_cell_400_matmul_readvariableop_resource:	ШA
.lstm_cell_400_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_400_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_400/BiasAdd/ReadVariableOpЂ#lstm_cell_400/MatMul/ReadVariableOpЂ%lstm_cell_400/MatMul_1/ReadVariableOpЂwhileF
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
#lstm_cell_400/MatMul/ReadVariableOpReadVariableOp,lstm_cell_400_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_400/MatMul/ReadVariableOpА
lstm_cell_400/MatMulMatMulstrided_slice_2:output:0+lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/MatMulО
%lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_400_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_400/MatMul_1/ReadVariableOpЌ
lstm_cell_400/MatMul_1MatMulzeros:output:0-lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/MatMul_1Є
lstm_cell_400/addAddV2lstm_cell_400/MatMul:product:0 lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/addЗ
$lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_400_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_400/BiasAdd/ReadVariableOpБ
lstm_cell_400/BiasAddBiasAddlstm_cell_400/add:z:0,lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/BiasAdd
lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_400/split/split_dimї
lstm_cell_400/splitSplit&lstm_cell_400/split/split_dim:output:0lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_400/split
lstm_cell_400/SigmoidSigmoidlstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Sigmoid
lstm_cell_400/Sigmoid_1Sigmoidlstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Sigmoid_1
lstm_cell_400/mulMullstm_cell_400/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/mul
lstm_cell_400/ReluRelulstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Relu 
lstm_cell_400/mul_1Mullstm_cell_400/Sigmoid:y:0 lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/mul_1
lstm_cell_400/add_1AddV2lstm_cell_400/mul:z:0lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/add_1
lstm_cell_400/Sigmoid_2Sigmoidlstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Sigmoid_2
lstm_cell_400/Relu_1Relulstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Relu_1Є
lstm_cell_400/mul_2Mullstm_cell_400/Sigmoid_2:y:0"lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_400_matmul_readvariableop_resource.lstm_cell_400_matmul_1_readvariableop_resource-lstm_cell_400_biasadd_readvariableop_resource*
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
bodyR
while_body_16888996*
condR
while_cond_16888995*K
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
NoOpNoOp%^lstm_cell_400/BiasAdd/ReadVariableOp$^lstm_cell_400/MatMul/ReadVariableOp&^lstm_cell_400/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_400/BiasAdd/ReadVariableOp$lstm_cell_400/BiasAdd/ReadVariableOp2J
#lstm_cell_400/MatMul/ReadVariableOp#lstm_cell_400/MatMul/ReadVariableOp2N
%lstm_cell_400/MatMul_1/ReadVariableOp%lstm_cell_400/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
п
Э
while_cond_16889146
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16889146___redundant_placeholder06
2while_while_cond_16889146___redundant_placeholder16
2while_while_cond_16889146___redundant_placeholder26
2while_while_cond_16889146___redundant_placeholder3
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
И
Ѓ
L__inference_sequential_133_layer_call_and_return_conditional_losses_16887271

inputs
inputs_1	-
bidirectional_133_16887252:	Ш-
bidirectional_133_16887254:	2Ш)
bidirectional_133_16887256:	Ш-
bidirectional_133_16887258:	Ш-
bidirectional_133_16887260:	2Ш)
bidirectional_133_16887262:	Ш$
dense_133_16887265:d 
dense_133_16887267:
identityЂ)bidirectional_133/StatefulPartitionedCallЂ!dense_133/StatefulPartitionedCallЪ
)bidirectional_133/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_133_16887252bidirectional_133_16887254bidirectional_133_16887256bidirectional_133_16887258bidirectional_133_16887260bidirectional_133_16887262*
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
GPU 2J 8 *X
fSRQ
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_168867042+
)bidirectional_133/StatefulPartitionedCallЫ
!dense_133/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_133/StatefulPartitionedCall:output:0dense_133_16887265dense_133_16887267*
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
GPU 2J 8 *P
fKRI
G__inference_dense_133_layer_call_and_return_conditional_losses_168867292#
!dense_133/StatefulPartitionedCall
IdentityIdentity*dense_133/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp*^bidirectional_133/StatefulPartitionedCall"^dense_133/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2V
)bidirectional_133/StatefulPartitionedCall)bidirectional_133/StatefulPartitionedCall2F
!dense_133/StatefulPartitionedCall!dense_133/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
_
Џ
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_16889732
inputs_0?
,lstm_cell_401_matmul_readvariableop_resource:	ШA
.lstm_cell_401_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_401_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_401/BiasAdd/ReadVariableOpЂ#lstm_cell_401/MatMul/ReadVariableOpЂ%lstm_cell_401/MatMul_1/ReadVariableOpЂwhileF
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
#lstm_cell_401/MatMul/ReadVariableOpReadVariableOp,lstm_cell_401_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_401/MatMul/ReadVariableOpА
lstm_cell_401/MatMulMatMulstrided_slice_2:output:0+lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/MatMulО
%lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_401_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_401/MatMul_1/ReadVariableOpЌ
lstm_cell_401/MatMul_1MatMulzeros:output:0-lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/MatMul_1Є
lstm_cell_401/addAddV2lstm_cell_401/MatMul:product:0 lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/addЗ
$lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_401_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_401/BiasAdd/ReadVariableOpБ
lstm_cell_401/BiasAddBiasAddlstm_cell_401/add:z:0,lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/BiasAdd
lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_401/split/split_dimї
lstm_cell_401/splitSplit&lstm_cell_401/split/split_dim:output:0lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_401/split
lstm_cell_401/SigmoidSigmoidlstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Sigmoid
lstm_cell_401/Sigmoid_1Sigmoidlstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Sigmoid_1
lstm_cell_401/mulMullstm_cell_401/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/mul
lstm_cell_401/ReluRelulstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Relu 
lstm_cell_401/mul_1Mullstm_cell_401/Sigmoid:y:0 lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/mul_1
lstm_cell_401/add_1AddV2lstm_cell_401/mul:z:0lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/add_1
lstm_cell_401/Sigmoid_2Sigmoidlstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Sigmoid_2
lstm_cell_401/Relu_1Relulstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Relu_1Є
lstm_cell_401/mul_2Mullstm_cell_401/Sigmoid_2:y:0"lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_401_matmul_readvariableop_resource.lstm_cell_401_matmul_1_readvariableop_resource-lstm_cell_401_biasadd_readvariableop_resource*
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
bodyR
while_body_16889648*
condR
while_cond_16889647*K
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
NoOpNoOp%^lstm_cell_401/BiasAdd/ReadVariableOp$^lstm_cell_401/MatMul/ReadVariableOp&^lstm_cell_401/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_401/BiasAdd/ReadVariableOp$lstm_cell_401/BiasAdd/ReadVariableOp2J
#lstm_cell_401/MatMul/ReadVariableOp#lstm_cell_401/MatMul/ReadVariableOp2N
%lstm_cell_401/MatMul_1/ReadVariableOp%lstm_cell_401/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
аf
ф
%backward_lstm_133_while_body_16886607@
<backward_lstm_133_while_backward_lstm_133_while_loop_counterF
Bbackward_lstm_133_while_backward_lstm_133_while_maximum_iterations'
#backward_lstm_133_while_placeholder)
%backward_lstm_133_while_placeholder_1)
%backward_lstm_133_while_placeholder_2)
%backward_lstm_133_while_placeholder_3)
%backward_lstm_133_while_placeholder_4?
;backward_lstm_133_while_backward_lstm_133_strided_slice_1_0{
wbackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_133_while_less_backward_lstm_133_sub_1_0Y
Fbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0:	Ш[
Hbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0:	2ШV
Gbackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0:	Ш$
 backward_lstm_133_while_identity&
"backward_lstm_133_while_identity_1&
"backward_lstm_133_while_identity_2&
"backward_lstm_133_while_identity_3&
"backward_lstm_133_while_identity_4&
"backward_lstm_133_while_identity_5&
"backward_lstm_133_while_identity_6=
9backward_lstm_133_while_backward_lstm_133_strided_slice_1y
ubackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_133_while_less_backward_lstm_133_sub_1W
Dbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource:	ШY
Fbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource:	2ШT
Ebackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource:	ШЂ<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOpЂ;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpЂ=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOpч
Ibackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2K
Ibackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeП
;backward_lstm_133/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_133_while_placeholderRbackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02=
;backward_lstm_133/while/TensorArrayV2Read/TensorListGetItemЯ
backward_lstm_133/while/LessLess6backward_lstm_133_while_less_backward_lstm_133_sub_1_0#backward_lstm_133_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_133/while/Less
;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpReadVariableOpFbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02=
;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpЂ
,backward_lstm_133/while/lstm_cell_401/MatMulMatMulBbackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem:item:0Cbackward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_133/while/lstm_cell_401/MatMul
=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOpHbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02?
=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp
.backward_lstm_133/while/lstm_cell_401/MatMul_1MatMul%backward_lstm_133_while_placeholder_3Ebackward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ20
.backward_lstm_133/while/lstm_cell_401/MatMul_1
)backward_lstm_133/while/lstm_cell_401/addAddV26backward_lstm_133/while/lstm_cell_401/MatMul:product:08backward_lstm_133/while/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_133/while/lstm_cell_401/add
<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOpGbackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02>
<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp
-backward_lstm_133/while/lstm_cell_401/BiasAddBiasAdd-backward_lstm_133/while/lstm_cell_401/add:z:0Dbackward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_133/while/lstm_cell_401/BiasAddА
5backward_lstm_133/while/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5backward_lstm_133/while/lstm_cell_401/split/split_dimз
+backward_lstm_133/while/lstm_cell_401/splitSplit>backward_lstm_133/while/lstm_cell_401/split/split_dim:output:06backward_lstm_133/while/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2-
+backward_lstm_133/while/lstm_cell_401/splitб
-backward_lstm_133/while/lstm_cell_401/SigmoidSigmoid4backward_lstm_133/while/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22/
-backward_lstm_133/while/lstm_cell_401/Sigmoidе
/backward_lstm_133/while/lstm_cell_401/Sigmoid_1Sigmoid4backward_lstm_133/while/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_133/while/lstm_cell_401/Sigmoid_1ы
)backward_lstm_133/while/lstm_cell_401/mulMul3backward_lstm_133/while/lstm_cell_401/Sigmoid_1:y:0%backward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_133/while/lstm_cell_401/mulШ
*backward_lstm_133/while/lstm_cell_401/ReluRelu4backward_lstm_133/while/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_133/while/lstm_cell_401/Relu
+backward_lstm_133/while/lstm_cell_401/mul_1Mul1backward_lstm_133/while/lstm_cell_401/Sigmoid:y:08backward_lstm_133/while/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_133/while/lstm_cell_401/mul_1ѕ
+backward_lstm_133/while/lstm_cell_401/add_1AddV2-backward_lstm_133/while/lstm_cell_401/mul:z:0/backward_lstm_133/while/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_133/while/lstm_cell_401/add_1е
/backward_lstm_133/while/lstm_cell_401/Sigmoid_2Sigmoid4backward_lstm_133/while/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_133/while/lstm_cell_401/Sigmoid_2Ч
,backward_lstm_133/while/lstm_cell_401/Relu_1Relu/backward_lstm_133/while/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_133/while/lstm_cell_401/Relu_1
+backward_lstm_133/while/lstm_cell_401/mul_2Mul3backward_lstm_133/while/lstm_cell_401/Sigmoid_2:y:0:backward_lstm_133/while/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_133/while/lstm_cell_401/mul_2і
backward_lstm_133/while/SelectSelect backward_lstm_133/while/Less:z:0/backward_lstm_133/while/lstm_cell_401/mul_2:z:0%backward_lstm_133_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_133/while/Selectњ
 backward_lstm_133/while/Select_1Select backward_lstm_133/while/Less:z:0/backward_lstm_133/while/lstm_cell_401/mul_2:z:0%backward_lstm_133_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_133/while/Select_1њ
 backward_lstm_133/while/Select_2Select backward_lstm_133/while/Less:z:0/backward_lstm_133/while/lstm_cell_401/add_1:z:0%backward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_133/while/Select_2Г
<backward_lstm_133/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_133_while_placeholder_1#backward_lstm_133_while_placeholder'backward_lstm_133/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_133/while/TensorArrayV2Write/TensorListSetItem
backward_lstm_133/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_133/while/add/yБ
backward_lstm_133/while/addAddV2#backward_lstm_133_while_placeholder&backward_lstm_133/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/while/add
backward_lstm_133/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_133/while/add_1/yа
backward_lstm_133/while/add_1AddV2<backward_lstm_133_while_backward_lstm_133_while_loop_counter(backward_lstm_133/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/while/add_1Г
 backward_lstm_133/while/IdentityIdentity!backward_lstm_133/while/add_1:z:0^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_133/while/Identityи
"backward_lstm_133/while/Identity_1IdentityBbackward_lstm_133_while_backward_lstm_133_while_maximum_iterations^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_133/while/Identity_1Е
"backward_lstm_133/while/Identity_2Identitybackward_lstm_133/while/add:z:0^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_133/while/Identity_2т
"backward_lstm_133/while/Identity_3IdentityLbackward_lstm_133/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_133/while/Identity_3Ю
"backward_lstm_133/while/Identity_4Identity'backward_lstm_133/while/Select:output:0^backward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_133/while/Identity_4а
"backward_lstm_133/while/Identity_5Identity)backward_lstm_133/while/Select_1:output:0^backward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_133/while/Identity_5а
"backward_lstm_133/while/Identity_6Identity)backward_lstm_133/while/Select_2:output:0^backward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_133/while/Identity_6Л
backward_lstm_133/while/NoOpNoOp=^backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp<^backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp>^backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_133/while/NoOp"x
9backward_lstm_133_while_backward_lstm_133_strided_slice_1;backward_lstm_133_while_backward_lstm_133_strided_slice_1_0"M
 backward_lstm_133_while_identity)backward_lstm_133/while/Identity:output:0"Q
"backward_lstm_133_while_identity_1+backward_lstm_133/while/Identity_1:output:0"Q
"backward_lstm_133_while_identity_2+backward_lstm_133/while/Identity_2:output:0"Q
"backward_lstm_133_while_identity_3+backward_lstm_133/while/Identity_3:output:0"Q
"backward_lstm_133_while_identity_4+backward_lstm_133/while/Identity_4:output:0"Q
"backward_lstm_133_while_identity_5+backward_lstm_133/while/Identity_5:output:0"Q
"backward_lstm_133_while_identity_6+backward_lstm_133/while/Identity_6:output:0"n
4backward_lstm_133_while_less_backward_lstm_133_sub_16backward_lstm_133_while_less_backward_lstm_133_sub_1_0"
Ebackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resourceGbackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0"
Fbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resourceHbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0"
Dbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resourceFbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0"№
ubackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2|
<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp2z
;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp2~
=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp: 
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
ў

K__inference_lstm_cell_401_layer_call_and_return_conditional_losses_16890202

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
і

K__inference_lstm_cell_400_layer_call_and_return_conditional_losses_16884489

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
п
Э
while_cond_16884988
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16884988___redundant_placeholder06
2while_while_cond_16884988___redundant_placeholder16
2while_while_cond_16884988___redundant_placeholder26
2while_while_cond_16884988___redundant_placeholder3
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
§
Е
%backward_lstm_133_while_cond_16887911@
<backward_lstm_133_while_backward_lstm_133_while_loop_counterF
Bbackward_lstm_133_while_backward_lstm_133_while_maximum_iterations'
#backward_lstm_133_while_placeholder)
%backward_lstm_133_while_placeholder_1)
%backward_lstm_133_while_placeholder_2)
%backward_lstm_133_while_placeholder_3B
>backward_lstm_133_while_less_backward_lstm_133_strided_slice_1Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16887911___redundant_placeholder0Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16887911___redundant_placeholder1Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16887911___redundant_placeholder2Z
Vbackward_lstm_133_while_backward_lstm_133_while_cond_16887911___redundant_placeholder3$
 backward_lstm_133_while_identity
Ъ
backward_lstm_133/while/LessLess#backward_lstm_133_while_placeholder>backward_lstm_133_while_less_backward_lstm_133_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_133/while/Less
 backward_lstm_133/while/IdentityIdentity backward_lstm_133/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_133/while/Identity"M
 backward_lstm_133_while_identity)backward_lstm_133/while/Identity:output:0*(
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
Џc
ц
!__inference__traced_save_16890375
file_prefix/
+savev2_dense_133_kernel_read_readvariableop-
)savev2_dense_133_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopV
Rsavev2_bidirectional_133_forward_lstm_133_lstm_cell_400_kernel_read_readvariableop`
\savev2_bidirectional_133_forward_lstm_133_lstm_cell_400_recurrent_kernel_read_readvariableopT
Psavev2_bidirectional_133_forward_lstm_133_lstm_cell_400_bias_read_readvariableopW
Ssavev2_bidirectional_133_backward_lstm_133_lstm_cell_401_kernel_read_readvariableopa
]savev2_bidirectional_133_backward_lstm_133_lstm_cell_401_recurrent_kernel_read_readvariableopU
Qsavev2_bidirectional_133_backward_lstm_133_lstm_cell_401_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_133_kernel_m_read_readvariableop4
0savev2_adam_dense_133_bias_m_read_readvariableop]
Ysavev2_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_kernel_m_read_readvariableopg
csavev2_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_recurrent_kernel_m_read_readvariableop[
Wsavev2_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_bias_m_read_readvariableop^
Zsavev2_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_kernel_m_read_readvariableoph
dsavev2_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_recurrent_kernel_m_read_readvariableop\
Xsavev2_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_bias_m_read_readvariableop6
2savev2_adam_dense_133_kernel_v_read_readvariableop4
0savev2_adam_dense_133_bias_v_read_readvariableop]
Ysavev2_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_kernel_v_read_readvariableopg
csavev2_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_recurrent_kernel_v_read_readvariableop[
Wsavev2_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_bias_v_read_readvariableop^
Zsavev2_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_kernel_v_read_readvariableoph
dsavev2_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_recurrent_kernel_v_read_readvariableop\
Xsavev2_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_bias_v_read_readvariableop9
5savev2_adam_dense_133_kernel_vhat_read_readvariableop7
3savev2_adam_dense_133_bias_vhat_read_readvariableop`
\savev2_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_kernel_vhat_read_readvariableopj
fsavev2_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_recurrent_kernel_vhat_read_readvariableop^
Zsavev2_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_bias_vhat_read_readvariableopa
]savev2_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_kernel_vhat_read_readvariableopk
gsavev2_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_recurrent_kernel_vhat_read_readvariableop_
[savev2_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_bias_vhat_read_readvariableop
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
SaveV2/shape_and_slicesН
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_133_kernel_read_readvariableop)savev2_dense_133_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopRsavev2_bidirectional_133_forward_lstm_133_lstm_cell_400_kernel_read_readvariableop\savev2_bidirectional_133_forward_lstm_133_lstm_cell_400_recurrent_kernel_read_readvariableopPsavev2_bidirectional_133_forward_lstm_133_lstm_cell_400_bias_read_readvariableopSsavev2_bidirectional_133_backward_lstm_133_lstm_cell_401_kernel_read_readvariableop]savev2_bidirectional_133_backward_lstm_133_lstm_cell_401_recurrent_kernel_read_readvariableopQsavev2_bidirectional_133_backward_lstm_133_lstm_cell_401_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_133_kernel_m_read_readvariableop0savev2_adam_dense_133_bias_m_read_readvariableopYsavev2_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_kernel_m_read_readvariableopcsavev2_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_recurrent_kernel_m_read_readvariableopWsavev2_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_bias_m_read_readvariableopZsavev2_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_kernel_m_read_readvariableopdsavev2_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_recurrent_kernel_m_read_readvariableopXsavev2_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_bias_m_read_readvariableop2savev2_adam_dense_133_kernel_v_read_readvariableop0savev2_adam_dense_133_bias_v_read_readvariableopYsavev2_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_kernel_v_read_readvariableopcsavev2_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_recurrent_kernel_v_read_readvariableopWsavev2_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_bias_v_read_readvariableopZsavev2_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_kernel_v_read_readvariableopdsavev2_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_recurrent_kernel_v_read_readvariableopXsavev2_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_bias_v_read_readvariableop5savev2_adam_dense_133_kernel_vhat_read_readvariableop3savev2_adam_dense_133_bias_vhat_read_readvariableop\savev2_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_kernel_vhat_read_readvariableopfsavev2_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_recurrent_kernel_vhat_read_readvariableopZsavev2_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_bias_vhat_read_readvariableop]savev2_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_kernel_vhat_read_readvariableopgsavev2_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_recurrent_kernel_vhat_read_readvariableop[savev2_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_bias_vhat_read_readvariableopsavev2_const"/device:CPU:0*
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
]
Ќ
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_16889231

inputs?
,lstm_cell_400_matmul_readvariableop_resource:	ШA
.lstm_cell_400_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_400_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_400/BiasAdd/ReadVariableOpЂ#lstm_cell_400/MatMul/ReadVariableOpЂ%lstm_cell_400/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_400/MatMul/ReadVariableOpReadVariableOp,lstm_cell_400_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_400/MatMul/ReadVariableOpА
lstm_cell_400/MatMulMatMulstrided_slice_2:output:0+lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/MatMulО
%lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_400_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_400/MatMul_1/ReadVariableOpЌ
lstm_cell_400/MatMul_1MatMulzeros:output:0-lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/MatMul_1Є
lstm_cell_400/addAddV2lstm_cell_400/MatMul:product:0 lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/addЗ
$lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_400_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_400/BiasAdd/ReadVariableOpБ
lstm_cell_400/BiasAddBiasAddlstm_cell_400/add:z:0,lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/BiasAdd
lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_400/split/split_dimї
lstm_cell_400/splitSplit&lstm_cell_400/split/split_dim:output:0lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_400/split
lstm_cell_400/SigmoidSigmoidlstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Sigmoid
lstm_cell_400/Sigmoid_1Sigmoidlstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Sigmoid_1
lstm_cell_400/mulMullstm_cell_400/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/mul
lstm_cell_400/ReluRelulstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Relu 
lstm_cell_400/mul_1Mullstm_cell_400/Sigmoid:y:0 lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/mul_1
lstm_cell_400/add_1AddV2lstm_cell_400/mul:z:0lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/add_1
lstm_cell_400/Sigmoid_2Sigmoidlstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Sigmoid_2
lstm_cell_400/Relu_1Relulstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Relu_1Є
lstm_cell_400/mul_2Mullstm_cell_400/Sigmoid_2:y:0"lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_400_matmul_readvariableop_resource.lstm_cell_400_matmul_1_readvariableop_resource-lstm_cell_400_biasadd_readvariableop_resource*
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
bodyR
while_body_16889147*
condR
while_cond_16889146*K
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
NoOpNoOp%^lstm_cell_400/BiasAdd/ReadVariableOp$^lstm_cell_400/MatMul/ReadVariableOp&^lstm_cell_400/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_400/BiasAdd/ReadVariableOp$lstm_cell_400/BiasAdd/ReadVariableOp2J
#lstm_cell_400/MatMul/ReadVariableOp#lstm_cell_400/MatMul/ReadVariableOp2N
%lstm_cell_400/MatMul_1/ReadVariableOp%lstm_cell_400/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Н
я
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_16887144

inputs
inputs_1	P
=forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource:	ШR
?forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource:	2ШM
>forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource:	ШQ
>backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource:	ШS
@backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource:	2ШN
?backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource:	Ш
identityЂ6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpЂ5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpЂ7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpЂbackward_lstm_133/whileЂ5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpЂ4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpЂ6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOpЂforward_lstm_133/while
%forward_lstm_133/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_133/RaggedToTensor/zeros
%forward_lstm_133/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2'
%forward_lstm_133/RaggedToTensor/Const
4forward_lstm_133/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_133/RaggedToTensor/Const:output:0inputs.forward_lstm_133/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_133/RaggedToTensor/RaggedTensorToTensorФ
;forward_lstm_133/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_133/RaggedNestedRowLengths/strided_slice/stackШ
=forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1Ш
=forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2Љ
5forward_lstm_133/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_133/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask27
5forward_lstm_133/RaggedNestedRowLengths/strided_sliceШ
=forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stackе
?forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2A
?forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1Ь
?forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2Е
7forward_lstm_133/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask29
7forward_lstm_133/RaggedNestedRowLengths/strided_slice_1
+forward_lstm_133/RaggedNestedRowLengths/subSub>forward_lstm_133/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_133/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2-
+forward_lstm_133/RaggedNestedRowLengths/subЄ
forward_lstm_133/CastCast/forward_lstm_133/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_133/Cast
forward_lstm_133/ShapeShape=forward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_133/Shape
$forward_lstm_133/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_133/strided_slice/stack
&forward_lstm_133/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_133/strided_slice/stack_1
&forward_lstm_133/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_133/strided_slice/stack_2Ш
forward_lstm_133/strided_sliceStridedSliceforward_lstm_133/Shape:output:0-forward_lstm_133/strided_slice/stack:output:0/forward_lstm_133/strided_slice/stack_1:output:0/forward_lstm_133/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_133/strided_slice~
forward_lstm_133/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_133/zeros/mul/yА
forward_lstm_133/zeros/mulMul'forward_lstm_133/strided_slice:output:0%forward_lstm_133/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros/mul
forward_lstm_133/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_133/zeros/Less/yЋ
forward_lstm_133/zeros/LessLessforward_lstm_133/zeros/mul:z:0&forward_lstm_133/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros/Less
forward_lstm_133/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_133/zeros/packed/1Ч
forward_lstm_133/zeros/packedPack'forward_lstm_133/strided_slice:output:0(forward_lstm_133/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_133/zeros/packed
forward_lstm_133/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_133/zeros/ConstЙ
forward_lstm_133/zerosFill&forward_lstm_133/zeros/packed:output:0%forward_lstm_133/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/zeros
forward_lstm_133/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_133/zeros_1/mul/yЖ
forward_lstm_133/zeros_1/mulMul'forward_lstm_133/strided_slice:output:0'forward_lstm_133/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros_1/mul
forward_lstm_133/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
forward_lstm_133/zeros_1/Less/yГ
forward_lstm_133/zeros_1/LessLess forward_lstm_133/zeros_1/mul:z:0(forward_lstm_133/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros_1/Less
!forward_lstm_133/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_133/zeros_1/packed/1Э
forward_lstm_133/zeros_1/packedPack'forward_lstm_133/strided_slice:output:0*forward_lstm_133/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_133/zeros_1/packed
forward_lstm_133/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_133/zeros_1/ConstС
forward_lstm_133/zeros_1Fill(forward_lstm_133/zeros_1/packed:output:0'forward_lstm_133/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/zeros_1
forward_lstm_133/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_133/transpose/permэ
forward_lstm_133/transpose	Transpose=forward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_133/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_133/transpose
forward_lstm_133/Shape_1Shapeforward_lstm_133/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_133/Shape_1
&forward_lstm_133/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_133/strided_slice_1/stack
(forward_lstm_133/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_1/stack_1
(forward_lstm_133/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_1/stack_2д
 forward_lstm_133/strided_slice_1StridedSlice!forward_lstm_133/Shape_1:output:0/forward_lstm_133/strided_slice_1/stack:output:01forward_lstm_133/strided_slice_1/stack_1:output:01forward_lstm_133/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_133/strided_slice_1Ї
,forward_lstm_133/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,forward_lstm_133/TensorArrayV2/element_shapeі
forward_lstm_133/TensorArrayV2TensorListReserve5forward_lstm_133/TensorArrayV2/element_shape:output:0)forward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_133/TensorArrayV2с
Fforward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fforward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8forward_lstm_133/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_133/transpose:y:0Oforward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_133/TensorArrayUnstack/TensorListFromTensor
&forward_lstm_133/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_133/strided_slice_2/stack
(forward_lstm_133/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_2/stack_1
(forward_lstm_133/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_2/stack_2т
 forward_lstm_133/strided_slice_2StridedSliceforward_lstm_133/transpose:y:0/forward_lstm_133/strided_slice_2/stack:output:01forward_lstm_133/strided_slice_2/stack_1:output:01forward_lstm_133/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 forward_lstm_133/strided_slice_2ы
4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpReadVariableOp=forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpє
%forward_lstm_133/lstm_cell_400/MatMulMatMul)forward_lstm_133/strided_slice_2:output:0<forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_133/lstm_cell_400/MatMulё
6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOp?forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp№
'forward_lstm_133/lstm_cell_400/MatMul_1MatMulforward_lstm_133/zeros:output:0>forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_133/lstm_cell_400/MatMul_1ш
"forward_lstm_133/lstm_cell_400/addAddV2/forward_lstm_133/lstm_cell_400/MatMul:product:01forward_lstm_133/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_133/lstm_cell_400/addъ
5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOp>forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpѕ
&forward_lstm_133/lstm_cell_400/BiasAddBiasAdd&forward_lstm_133/lstm_cell_400/add:z:0=forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_133/lstm_cell_400/BiasAddЂ
.forward_lstm_133/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.forward_lstm_133/lstm_cell_400/split/split_dimЛ
$forward_lstm_133/lstm_cell_400/splitSplit7forward_lstm_133/lstm_cell_400/split/split_dim:output:0/forward_lstm_133/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$forward_lstm_133/lstm_cell_400/splitМ
&forward_lstm_133/lstm_cell_400/SigmoidSigmoid-forward_lstm_133/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_133/lstm_cell_400/SigmoidР
(forward_lstm_133/lstm_cell_400/Sigmoid_1Sigmoid-forward_lstm_133/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_133/lstm_cell_400/Sigmoid_1в
"forward_lstm_133/lstm_cell_400/mulMul,forward_lstm_133/lstm_cell_400/Sigmoid_1:y:0!forward_lstm_133/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_133/lstm_cell_400/mulГ
#forward_lstm_133/lstm_cell_400/ReluRelu-forward_lstm_133/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_133/lstm_cell_400/Reluф
$forward_lstm_133/lstm_cell_400/mul_1Mul*forward_lstm_133/lstm_cell_400/Sigmoid:y:01forward_lstm_133/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_133/lstm_cell_400/mul_1й
$forward_lstm_133/lstm_cell_400/add_1AddV2&forward_lstm_133/lstm_cell_400/mul:z:0(forward_lstm_133/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_133/lstm_cell_400/add_1Р
(forward_lstm_133/lstm_cell_400/Sigmoid_2Sigmoid-forward_lstm_133/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_133/lstm_cell_400/Sigmoid_2В
%forward_lstm_133/lstm_cell_400/Relu_1Relu(forward_lstm_133/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_133/lstm_cell_400/Relu_1ш
$forward_lstm_133/lstm_cell_400/mul_2Mul,forward_lstm_133/lstm_cell_400/Sigmoid_2:y:03forward_lstm_133/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_133/lstm_cell_400/mul_2Б
.forward_lstm_133/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.forward_lstm_133/TensorArrayV2_1/element_shapeќ
 forward_lstm_133/TensorArrayV2_1TensorListReserve7forward_lstm_133/TensorArrayV2_1/element_shape:output:0)forward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_133/TensorArrayV2_1p
forward_lstm_133/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_133/timeЃ
forward_lstm_133/zeros_like	ZerosLike(forward_lstm_133/lstm_cell_400/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/zeros_likeЁ
)forward_lstm_133/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)forward_lstm_133/while/maximum_iterations
#forward_lstm_133/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_133/while/loop_counter	
forward_lstm_133/whileWhile,forward_lstm_133/while/loop_counter:output:02forward_lstm_133/while/maximum_iterations:output:0forward_lstm_133/time:output:0)forward_lstm_133/TensorArrayV2_1:handle:0forward_lstm_133/zeros_like:y:0forward_lstm_133/zeros:output:0!forward_lstm_133/zeros_1:output:0)forward_lstm_133/strided_slice_1:output:0Hforward_lstm_133/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_133/Cast:y:0=forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource?forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource>forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *0
body(R&
$forward_lstm_133_while_body_16886868*0
cond(R&
$forward_lstm_133_while_cond_16886867*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_133/whileз
Aforward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Aforward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3forward_lstm_133/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_133/while:output:3Jforward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3forward_lstm_133/TensorArrayV2Stack/TensorListStackЃ
&forward_lstm_133/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&forward_lstm_133/strided_slice_3/stack
(forward_lstm_133/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_133/strided_slice_3/stack_1
(forward_lstm_133/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_3/stack_2
 forward_lstm_133/strided_slice_3StridedSlice<forward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_133/strided_slice_3/stack:output:01forward_lstm_133/strided_slice_3/stack_1:output:01forward_lstm_133/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 forward_lstm_133/strided_slice_3
!forward_lstm_133/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_133/transpose_1/permђ
forward_lstm_133/transpose_1	Transpose<forward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_133/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_133/transpose_1
forward_lstm_133/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_133/runtime
&backward_lstm_133/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_133/RaggedToTensor/zeros
&backward_lstm_133/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2(
&backward_lstm_133/RaggedToTensor/Const
5backward_lstm_133/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_133/RaggedToTensor/Const:output:0inputs/backward_lstm_133/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_133/RaggedToTensor/RaggedTensorToTensorЦ
<backward_lstm_133/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_133/RaggedNestedRowLengths/strided_slice/stackЪ
>backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1Ъ
>backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2Ў
6backward_lstm_133/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_133/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask28
6backward_lstm_133/RaggedNestedRowLengths/strided_sliceЪ
>backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stackз
@backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2B
@backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1Ю
@backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2К
8backward_lstm_133/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2:
8backward_lstm_133/RaggedNestedRowLengths/strided_slice_1
,backward_lstm_133/RaggedNestedRowLengths/subSub?backward_lstm_133/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_133/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2.
,backward_lstm_133/RaggedNestedRowLengths/subЇ
backward_lstm_133/CastCast0backward_lstm_133/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_133/Cast 
backward_lstm_133/ShapeShape>backward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_133/Shape
%backward_lstm_133/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_133/strided_slice/stack
'backward_lstm_133/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_133/strided_slice/stack_1
'backward_lstm_133/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_133/strided_slice/stack_2Ю
backward_lstm_133/strided_sliceStridedSlice backward_lstm_133/Shape:output:0.backward_lstm_133/strided_slice/stack:output:00backward_lstm_133/strided_slice/stack_1:output:00backward_lstm_133/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_133/strided_slice
backward_lstm_133/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_133/zeros/mul/yД
backward_lstm_133/zeros/mulMul(backward_lstm_133/strided_slice:output:0&backward_lstm_133/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/zeros/mul
backward_lstm_133/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_133/zeros/Less/yЏ
backward_lstm_133/zeros/LessLessbackward_lstm_133/zeros/mul:z:0'backward_lstm_133/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/zeros/Less
 backward_lstm_133/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_133/zeros/packed/1Ы
backward_lstm_133/zeros/packedPack(backward_lstm_133/strided_slice:output:0)backward_lstm_133/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_133/zeros/packed
backward_lstm_133/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_133/zeros/ConstН
backward_lstm_133/zerosFill'backward_lstm_133/zeros/packed:output:0&backward_lstm_133/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_133/zeros
backward_lstm_133/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_133/zeros_1/mul/yК
backward_lstm_133/zeros_1/mulMul(backward_lstm_133/strided_slice:output:0(backward_lstm_133/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/zeros_1/mul
 backward_lstm_133/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 backward_lstm_133/zeros_1/Less/yЗ
backward_lstm_133/zeros_1/LessLess!backward_lstm_133/zeros_1/mul:z:0)backward_lstm_133/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_133/zeros_1/Less
"backward_lstm_133/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_133/zeros_1/packed/1б
 backward_lstm_133/zeros_1/packedPack(backward_lstm_133/strided_slice:output:0+backward_lstm_133/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_133/zeros_1/packed
backward_lstm_133/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_133/zeros_1/ConstХ
backward_lstm_133/zeros_1Fill)backward_lstm_133/zeros_1/packed:output:0(backward_lstm_133/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_133/zeros_1
 backward_lstm_133/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_133/transpose/permё
backward_lstm_133/transpose	Transpose>backward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_133/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_133/transpose
backward_lstm_133/Shape_1Shapebackward_lstm_133/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_133/Shape_1
'backward_lstm_133/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_133/strided_slice_1/stack 
)backward_lstm_133/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_1/stack_1 
)backward_lstm_133/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_1/stack_2к
!backward_lstm_133/strided_slice_1StridedSlice"backward_lstm_133/Shape_1:output:00backward_lstm_133/strided_slice_1/stack:output:02backward_lstm_133/strided_slice_1/stack_1:output:02backward_lstm_133/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_133/strided_slice_1Љ
-backward_lstm_133/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2/
-backward_lstm_133/TensorArrayV2/element_shapeњ
backward_lstm_133/TensorArrayV2TensorListReserve6backward_lstm_133/TensorArrayV2/element_shape:output:0*backward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_133/TensorArrayV2
 backward_lstm_133/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_133/ReverseV2/axisв
backward_lstm_133/ReverseV2	ReverseV2backward_lstm_133/transpose:y:0)backward_lstm_133/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_133/ReverseV2у
Gbackward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gbackward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeХ
9backward_lstm_133/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_133/ReverseV2:output:0Pbackward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_133/TensorArrayUnstack/TensorListFromTensor
'backward_lstm_133/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_133/strided_slice_2/stack 
)backward_lstm_133/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_2/stack_1 
)backward_lstm_133/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_2/stack_2ш
!backward_lstm_133/strided_slice_2StridedSlicebackward_lstm_133/transpose:y:00backward_lstm_133/strided_slice_2/stack:output:02backward_lstm_133/strided_slice_2/stack_1:output:02backward_lstm_133/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2#
!backward_lstm_133/strided_slice_2ю
5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpReadVariableOp>backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype027
5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpј
&backward_lstm_133/lstm_cell_401/MatMulMatMul*backward_lstm_133/strided_slice_2:output:0=backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_133/lstm_cell_401/MatMulє
7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOp@backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype029
7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpє
(backward_lstm_133/lstm_cell_401/MatMul_1MatMul backward_lstm_133/zeros:output:0?backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_133/lstm_cell_401/MatMul_1ь
#backward_lstm_133/lstm_cell_401/addAddV20backward_lstm_133/lstm_cell_401/MatMul:product:02backward_lstm_133/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_133/lstm_cell_401/addэ
6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOp?backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype028
6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpљ
'backward_lstm_133/lstm_cell_401/BiasAddBiasAdd'backward_lstm_133/lstm_cell_401/add:z:0>backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_133/lstm_cell_401/BiasAddЄ
/backward_lstm_133/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/backward_lstm_133/lstm_cell_401/split/split_dimП
%backward_lstm_133/lstm_cell_401/splitSplit8backward_lstm_133/lstm_cell_401/split/split_dim:output:00backward_lstm_133/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2'
%backward_lstm_133/lstm_cell_401/splitП
'backward_lstm_133/lstm_cell_401/SigmoidSigmoid.backward_lstm_133/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_133/lstm_cell_401/SigmoidУ
)backward_lstm_133/lstm_cell_401/Sigmoid_1Sigmoid.backward_lstm_133/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_133/lstm_cell_401/Sigmoid_1ж
#backward_lstm_133/lstm_cell_401/mulMul-backward_lstm_133/lstm_cell_401/Sigmoid_1:y:0"backward_lstm_133/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_133/lstm_cell_401/mulЖ
$backward_lstm_133/lstm_cell_401/ReluRelu.backward_lstm_133/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_133/lstm_cell_401/Reluш
%backward_lstm_133/lstm_cell_401/mul_1Mul+backward_lstm_133/lstm_cell_401/Sigmoid:y:02backward_lstm_133/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_133/lstm_cell_401/mul_1н
%backward_lstm_133/lstm_cell_401/add_1AddV2'backward_lstm_133/lstm_cell_401/mul:z:0)backward_lstm_133/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_133/lstm_cell_401/add_1У
)backward_lstm_133/lstm_cell_401/Sigmoid_2Sigmoid.backward_lstm_133/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_133/lstm_cell_401/Sigmoid_2Е
&backward_lstm_133/lstm_cell_401/Relu_1Relu)backward_lstm_133/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_133/lstm_cell_401/Relu_1ь
%backward_lstm_133/lstm_cell_401/mul_2Mul-backward_lstm_133/lstm_cell_401/Sigmoid_2:y:04backward_lstm_133/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_133/lstm_cell_401/mul_2Г
/backward_lstm_133/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   21
/backward_lstm_133/TensorArrayV2_1/element_shape
!backward_lstm_133/TensorArrayV2_1TensorListReserve8backward_lstm_133/TensorArrayV2_1/element_shape:output:0*backward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_133/TensorArrayV2_1r
backward_lstm_133/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_133/time
'backward_lstm_133/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_133/Max/reduction_indicesЄ
backward_lstm_133/MaxMaxbackward_lstm_133/Cast:y:00backward_lstm_133/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/Maxt
backward_lstm_133/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_133/sub/y
backward_lstm_133/subSubbackward_lstm_133/Max:output:0 backward_lstm_133/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/sub
backward_lstm_133/Sub_1Subbackward_lstm_133/sub:z:0backward_lstm_133/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_133/Sub_1І
backward_lstm_133/zeros_like	ZerosLike)backward_lstm_133/lstm_cell_401/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_133/zeros_likeЃ
*backward_lstm_133/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*backward_lstm_133/while/maximum_iterations
$backward_lstm_133/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_133/while/loop_counterЅ	
backward_lstm_133/whileWhile-backward_lstm_133/while/loop_counter:output:03backward_lstm_133/while/maximum_iterations:output:0backward_lstm_133/time:output:0*backward_lstm_133/TensorArrayV2_1:handle:0 backward_lstm_133/zeros_like:y:0 backward_lstm_133/zeros:output:0"backward_lstm_133/zeros_1:output:0*backward_lstm_133/strided_slice_1:output:0Ibackward_lstm_133/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_133/Sub_1:z:0>backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource@backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource?backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *1
body)R'
%backward_lstm_133_while_body_16887047*1
cond)R'
%backward_lstm_133_while_cond_16887046*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_133/whileй
Bbackward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2D
Bbackward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeЙ
4backward_lstm_133/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_133/while:output:3Kbackward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype026
4backward_lstm_133/TensorArrayV2Stack/TensorListStackЅ
'backward_lstm_133/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2)
'backward_lstm_133/strided_slice_3/stack 
)backward_lstm_133/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_133/strided_slice_3/stack_1 
)backward_lstm_133/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_3/stack_2
!backward_lstm_133/strided_slice_3StridedSlice=backward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_133/strided_slice_3/stack:output:02backward_lstm_133/strided_slice_3/stack_1:output:02backward_lstm_133/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2#
!backward_lstm_133/strided_slice_3
"backward_lstm_133/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_133/transpose_1/permі
backward_lstm_133/transpose_1	Transpose=backward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_133/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_133/transpose_1
backward_lstm_133/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_133/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisФ
concatConcatV2)forward_lstm_133/strided_slice_3:output:0*backward_lstm_133/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

Identityд
NoOpNoOp7^backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp6^backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp8^backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp^backward_lstm_133/while6^forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp5^forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp7^forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp^forward_lstm_133/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2p
6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp2n
5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp2r
7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp22
backward_lstm_133/whilebackward_lstm_133/while2n
5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp2l
4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp2p
6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp20
forward_lstm_133/whileforward_lstm_133/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЯY

%backward_lstm_133_while_body_16887912@
<backward_lstm_133_while_backward_lstm_133_while_loop_counterF
Bbackward_lstm_133_while_backward_lstm_133_while_maximum_iterations'
#backward_lstm_133_while_placeholder)
%backward_lstm_133_while_placeholder_1)
%backward_lstm_133_while_placeholder_2)
%backward_lstm_133_while_placeholder_3?
;backward_lstm_133_while_backward_lstm_133_strided_slice_1_0{
wbackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0Y
Fbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0:	Ш[
Hbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0:	2ШV
Gbackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0:	Ш$
 backward_lstm_133_while_identity&
"backward_lstm_133_while_identity_1&
"backward_lstm_133_while_identity_2&
"backward_lstm_133_while_identity_3&
"backward_lstm_133_while_identity_4&
"backward_lstm_133_while_identity_5=
9backward_lstm_133_while_backward_lstm_133_strided_slice_1y
ubackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensorW
Dbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource:	ШY
Fbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource:	2ШT
Ebackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource:	ШЂ<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOpЂ;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpЂ=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOpч
Ibackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2K
Ibackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeШ
;backward_lstm_133/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_133_while_placeholderRbackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02=
;backward_lstm_133/while/TensorArrayV2Read/TensorListGetItem
;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpReadVariableOpFbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02=
;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpЂ
,backward_lstm_133/while/lstm_cell_401/MatMulMatMulBbackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem:item:0Cbackward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_133/while/lstm_cell_401/MatMul
=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOpHbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02?
=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp
.backward_lstm_133/while/lstm_cell_401/MatMul_1MatMul%backward_lstm_133_while_placeholder_2Ebackward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ20
.backward_lstm_133/while/lstm_cell_401/MatMul_1
)backward_lstm_133/while/lstm_cell_401/addAddV26backward_lstm_133/while/lstm_cell_401/MatMul:product:08backward_lstm_133/while/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_133/while/lstm_cell_401/add
<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOpGbackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02>
<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp
-backward_lstm_133/while/lstm_cell_401/BiasAddBiasAdd-backward_lstm_133/while/lstm_cell_401/add:z:0Dbackward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_133/while/lstm_cell_401/BiasAddА
5backward_lstm_133/while/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5backward_lstm_133/while/lstm_cell_401/split/split_dimз
+backward_lstm_133/while/lstm_cell_401/splitSplit>backward_lstm_133/while/lstm_cell_401/split/split_dim:output:06backward_lstm_133/while/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2-
+backward_lstm_133/while/lstm_cell_401/splitб
-backward_lstm_133/while/lstm_cell_401/SigmoidSigmoid4backward_lstm_133/while/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22/
-backward_lstm_133/while/lstm_cell_401/Sigmoidе
/backward_lstm_133/while/lstm_cell_401/Sigmoid_1Sigmoid4backward_lstm_133/while/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_133/while/lstm_cell_401/Sigmoid_1ы
)backward_lstm_133/while/lstm_cell_401/mulMul3backward_lstm_133/while/lstm_cell_401/Sigmoid_1:y:0%backward_lstm_133_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_133/while/lstm_cell_401/mulШ
*backward_lstm_133/while/lstm_cell_401/ReluRelu4backward_lstm_133/while/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_133/while/lstm_cell_401/Relu
+backward_lstm_133/while/lstm_cell_401/mul_1Mul1backward_lstm_133/while/lstm_cell_401/Sigmoid:y:08backward_lstm_133/while/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_133/while/lstm_cell_401/mul_1ѕ
+backward_lstm_133/while/lstm_cell_401/add_1AddV2-backward_lstm_133/while/lstm_cell_401/mul:z:0/backward_lstm_133/while/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_133/while/lstm_cell_401/add_1е
/backward_lstm_133/while/lstm_cell_401/Sigmoid_2Sigmoid4backward_lstm_133/while/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_133/while/lstm_cell_401/Sigmoid_2Ч
,backward_lstm_133/while/lstm_cell_401/Relu_1Relu/backward_lstm_133/while/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_133/while/lstm_cell_401/Relu_1
+backward_lstm_133/while/lstm_cell_401/mul_2Mul3backward_lstm_133/while/lstm_cell_401/Sigmoid_2:y:0:backward_lstm_133/while/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_133/while/lstm_cell_401/mul_2Л
<backward_lstm_133/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_133_while_placeholder_1#backward_lstm_133_while_placeholder/backward_lstm_133/while/lstm_cell_401/mul_2:z:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_133/while/TensorArrayV2Write/TensorListSetItem
backward_lstm_133/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_133/while/add/yБ
backward_lstm_133/while/addAddV2#backward_lstm_133_while_placeholder&backward_lstm_133/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/while/add
backward_lstm_133/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_133/while/add_1/yа
backward_lstm_133/while/add_1AddV2<backward_lstm_133_while_backward_lstm_133_while_loop_counter(backward_lstm_133/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/while/add_1Г
 backward_lstm_133/while/IdentityIdentity!backward_lstm_133/while/add_1:z:0^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_133/while/Identityи
"backward_lstm_133/while/Identity_1IdentityBbackward_lstm_133_while_backward_lstm_133_while_maximum_iterations^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_133/while/Identity_1Е
"backward_lstm_133/while/Identity_2Identitybackward_lstm_133/while/add:z:0^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_133/while/Identity_2т
"backward_lstm_133/while/Identity_3IdentityLbackward_lstm_133/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_133/while/Identity_3ж
"backward_lstm_133/while/Identity_4Identity/backward_lstm_133/while/lstm_cell_401/mul_2:z:0^backward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_133/while/Identity_4ж
"backward_lstm_133/while/Identity_5Identity/backward_lstm_133/while/lstm_cell_401/add_1:z:0^backward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_133/while/Identity_5Л
backward_lstm_133/while/NoOpNoOp=^backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp<^backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp>^backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_133/while/NoOp"x
9backward_lstm_133_while_backward_lstm_133_strided_slice_1;backward_lstm_133_while_backward_lstm_133_strided_slice_1_0"M
 backward_lstm_133_while_identity)backward_lstm_133/while/Identity:output:0"Q
"backward_lstm_133_while_identity_1+backward_lstm_133/while/Identity_1:output:0"Q
"backward_lstm_133_while_identity_2+backward_lstm_133/while/Identity_2:output:0"Q
"backward_lstm_133_while_identity_3+backward_lstm_133/while/Identity_3:output:0"Q
"backward_lstm_133_while_identity_4+backward_lstm_133/while/Identity_4:output:0"Q
"backward_lstm_133_while_identity_5+backward_lstm_133/while/Identity_5:output:0"
Ebackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resourceGbackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0"
Fbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resourceHbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0"
Dbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resourceFbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0"№
ubackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2|
<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp2z
;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp2~
=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp: 
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
Н
я
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_16888356

inputs
inputs_1	P
=forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource:	ШR
?forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource:	2ШM
>forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource:	ШQ
>backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource:	ШS
@backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource:	2ШN
?backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource:	Ш
identityЂ6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpЂ5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpЂ7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpЂbackward_lstm_133/whileЂ5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpЂ4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpЂ6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOpЂforward_lstm_133/while
%forward_lstm_133/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_133/RaggedToTensor/zeros
%forward_lstm_133/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2'
%forward_lstm_133/RaggedToTensor/Const
4forward_lstm_133/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_133/RaggedToTensor/Const:output:0inputs.forward_lstm_133/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_133/RaggedToTensor/RaggedTensorToTensorФ
;forward_lstm_133/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_133/RaggedNestedRowLengths/strided_slice/stackШ
=forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1Ш
=forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2Љ
5forward_lstm_133/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_133/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask27
5forward_lstm_133/RaggedNestedRowLengths/strided_sliceШ
=forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stackе
?forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2A
?forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1Ь
?forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2Е
7forward_lstm_133/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask29
7forward_lstm_133/RaggedNestedRowLengths/strided_slice_1
+forward_lstm_133/RaggedNestedRowLengths/subSub>forward_lstm_133/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_133/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2-
+forward_lstm_133/RaggedNestedRowLengths/subЄ
forward_lstm_133/CastCast/forward_lstm_133/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_133/Cast
forward_lstm_133/ShapeShape=forward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_133/Shape
$forward_lstm_133/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_133/strided_slice/stack
&forward_lstm_133/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_133/strided_slice/stack_1
&forward_lstm_133/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_133/strided_slice/stack_2Ш
forward_lstm_133/strided_sliceStridedSliceforward_lstm_133/Shape:output:0-forward_lstm_133/strided_slice/stack:output:0/forward_lstm_133/strided_slice/stack_1:output:0/forward_lstm_133/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_133/strided_slice~
forward_lstm_133/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_133/zeros/mul/yА
forward_lstm_133/zeros/mulMul'forward_lstm_133/strided_slice:output:0%forward_lstm_133/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros/mul
forward_lstm_133/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_133/zeros/Less/yЋ
forward_lstm_133/zeros/LessLessforward_lstm_133/zeros/mul:z:0&forward_lstm_133/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros/Less
forward_lstm_133/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_133/zeros/packed/1Ч
forward_lstm_133/zeros/packedPack'forward_lstm_133/strided_slice:output:0(forward_lstm_133/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_133/zeros/packed
forward_lstm_133/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_133/zeros/ConstЙ
forward_lstm_133/zerosFill&forward_lstm_133/zeros/packed:output:0%forward_lstm_133/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/zeros
forward_lstm_133/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_133/zeros_1/mul/yЖ
forward_lstm_133/zeros_1/mulMul'forward_lstm_133/strided_slice:output:0'forward_lstm_133/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros_1/mul
forward_lstm_133/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
forward_lstm_133/zeros_1/Less/yГ
forward_lstm_133/zeros_1/LessLess forward_lstm_133/zeros_1/mul:z:0(forward_lstm_133/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_133/zeros_1/Less
!forward_lstm_133/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_133/zeros_1/packed/1Э
forward_lstm_133/zeros_1/packedPack'forward_lstm_133/strided_slice:output:0*forward_lstm_133/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_133/zeros_1/packed
forward_lstm_133/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_133/zeros_1/ConstС
forward_lstm_133/zeros_1Fill(forward_lstm_133/zeros_1/packed:output:0'forward_lstm_133/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/zeros_1
forward_lstm_133/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_133/transpose/permэ
forward_lstm_133/transpose	Transpose=forward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_133/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_133/transpose
forward_lstm_133/Shape_1Shapeforward_lstm_133/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_133/Shape_1
&forward_lstm_133/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_133/strided_slice_1/stack
(forward_lstm_133/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_1/stack_1
(forward_lstm_133/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_1/stack_2д
 forward_lstm_133/strided_slice_1StridedSlice!forward_lstm_133/Shape_1:output:0/forward_lstm_133/strided_slice_1/stack:output:01forward_lstm_133/strided_slice_1/stack_1:output:01forward_lstm_133/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_133/strided_slice_1Ї
,forward_lstm_133/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,forward_lstm_133/TensorArrayV2/element_shapeі
forward_lstm_133/TensorArrayV2TensorListReserve5forward_lstm_133/TensorArrayV2/element_shape:output:0)forward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_133/TensorArrayV2с
Fforward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fforward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8forward_lstm_133/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_133/transpose:y:0Oforward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_133/TensorArrayUnstack/TensorListFromTensor
&forward_lstm_133/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_133/strided_slice_2/stack
(forward_lstm_133/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_2/stack_1
(forward_lstm_133/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_2/stack_2т
 forward_lstm_133/strided_slice_2StridedSliceforward_lstm_133/transpose:y:0/forward_lstm_133/strided_slice_2/stack:output:01forward_lstm_133/strided_slice_2/stack_1:output:01forward_lstm_133/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 forward_lstm_133/strided_slice_2ы
4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpReadVariableOp=forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOpє
%forward_lstm_133/lstm_cell_400/MatMulMatMul)forward_lstm_133/strided_slice_2:output:0<forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_133/lstm_cell_400/MatMulё
6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOp?forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp№
'forward_lstm_133/lstm_cell_400/MatMul_1MatMulforward_lstm_133/zeros:output:0>forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_133/lstm_cell_400/MatMul_1ш
"forward_lstm_133/lstm_cell_400/addAddV2/forward_lstm_133/lstm_cell_400/MatMul:product:01forward_lstm_133/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_133/lstm_cell_400/addъ
5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOp>forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOpѕ
&forward_lstm_133/lstm_cell_400/BiasAddBiasAdd&forward_lstm_133/lstm_cell_400/add:z:0=forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_133/lstm_cell_400/BiasAddЂ
.forward_lstm_133/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.forward_lstm_133/lstm_cell_400/split/split_dimЛ
$forward_lstm_133/lstm_cell_400/splitSplit7forward_lstm_133/lstm_cell_400/split/split_dim:output:0/forward_lstm_133/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$forward_lstm_133/lstm_cell_400/splitМ
&forward_lstm_133/lstm_cell_400/SigmoidSigmoid-forward_lstm_133/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_133/lstm_cell_400/SigmoidР
(forward_lstm_133/lstm_cell_400/Sigmoid_1Sigmoid-forward_lstm_133/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_133/lstm_cell_400/Sigmoid_1в
"forward_lstm_133/lstm_cell_400/mulMul,forward_lstm_133/lstm_cell_400/Sigmoid_1:y:0!forward_lstm_133/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_133/lstm_cell_400/mulГ
#forward_lstm_133/lstm_cell_400/ReluRelu-forward_lstm_133/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_133/lstm_cell_400/Reluф
$forward_lstm_133/lstm_cell_400/mul_1Mul*forward_lstm_133/lstm_cell_400/Sigmoid:y:01forward_lstm_133/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_133/lstm_cell_400/mul_1й
$forward_lstm_133/lstm_cell_400/add_1AddV2&forward_lstm_133/lstm_cell_400/mul:z:0(forward_lstm_133/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_133/lstm_cell_400/add_1Р
(forward_lstm_133/lstm_cell_400/Sigmoid_2Sigmoid-forward_lstm_133/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_133/lstm_cell_400/Sigmoid_2В
%forward_lstm_133/lstm_cell_400/Relu_1Relu(forward_lstm_133/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_133/lstm_cell_400/Relu_1ш
$forward_lstm_133/lstm_cell_400/mul_2Mul,forward_lstm_133/lstm_cell_400/Sigmoid_2:y:03forward_lstm_133/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_133/lstm_cell_400/mul_2Б
.forward_lstm_133/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.forward_lstm_133/TensorArrayV2_1/element_shapeќ
 forward_lstm_133/TensorArrayV2_1TensorListReserve7forward_lstm_133/TensorArrayV2_1/element_shape:output:0)forward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_133/TensorArrayV2_1p
forward_lstm_133/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_133/timeЃ
forward_lstm_133/zeros_like	ZerosLike(forward_lstm_133/lstm_cell_400/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_133/zeros_likeЁ
)forward_lstm_133/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)forward_lstm_133/while/maximum_iterations
#forward_lstm_133/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_133/while/loop_counter	
forward_lstm_133/whileWhile,forward_lstm_133/while/loop_counter:output:02forward_lstm_133/while/maximum_iterations:output:0forward_lstm_133/time:output:0)forward_lstm_133/TensorArrayV2_1:handle:0forward_lstm_133/zeros_like:y:0forward_lstm_133/zeros:output:0!forward_lstm_133/zeros_1:output:0)forward_lstm_133/strided_slice_1:output:0Hforward_lstm_133/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_133/Cast:y:0=forward_lstm_133_lstm_cell_400_matmul_readvariableop_resource?forward_lstm_133_lstm_cell_400_matmul_1_readvariableop_resource>forward_lstm_133_lstm_cell_400_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *0
body(R&
$forward_lstm_133_while_body_16888080*0
cond(R&
$forward_lstm_133_while_cond_16888079*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_133/whileз
Aforward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Aforward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3forward_lstm_133/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_133/while:output:3Jforward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3forward_lstm_133/TensorArrayV2Stack/TensorListStackЃ
&forward_lstm_133/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&forward_lstm_133/strided_slice_3/stack
(forward_lstm_133/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_133/strided_slice_3/stack_1
(forward_lstm_133/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_133/strided_slice_3/stack_2
 forward_lstm_133/strided_slice_3StridedSlice<forward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_133/strided_slice_3/stack:output:01forward_lstm_133/strided_slice_3/stack_1:output:01forward_lstm_133/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 forward_lstm_133/strided_slice_3
!forward_lstm_133/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_133/transpose_1/permђ
forward_lstm_133/transpose_1	Transpose<forward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_133/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_133/transpose_1
forward_lstm_133/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_133/runtime
&backward_lstm_133/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_133/RaggedToTensor/zeros
&backward_lstm_133/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2(
&backward_lstm_133/RaggedToTensor/Const
5backward_lstm_133/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_133/RaggedToTensor/Const:output:0inputs/backward_lstm_133/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_133/RaggedToTensor/RaggedTensorToTensorЦ
<backward_lstm_133/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_133/RaggedNestedRowLengths/strided_slice/stackЪ
>backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1Ъ
>backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2Ў
6backward_lstm_133/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_133/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_133/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask28
6backward_lstm_133/RaggedNestedRowLengths/strided_sliceЪ
>backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stackз
@backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2B
@backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1Ю
@backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2К
8backward_lstm_133/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_133/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2:
8backward_lstm_133/RaggedNestedRowLengths/strided_slice_1
,backward_lstm_133/RaggedNestedRowLengths/subSub?backward_lstm_133/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_133/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2.
,backward_lstm_133/RaggedNestedRowLengths/subЇ
backward_lstm_133/CastCast0backward_lstm_133/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_133/Cast 
backward_lstm_133/ShapeShape>backward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_133/Shape
%backward_lstm_133/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_133/strided_slice/stack
'backward_lstm_133/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_133/strided_slice/stack_1
'backward_lstm_133/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_133/strided_slice/stack_2Ю
backward_lstm_133/strided_sliceStridedSlice backward_lstm_133/Shape:output:0.backward_lstm_133/strided_slice/stack:output:00backward_lstm_133/strided_slice/stack_1:output:00backward_lstm_133/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_133/strided_slice
backward_lstm_133/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_133/zeros/mul/yД
backward_lstm_133/zeros/mulMul(backward_lstm_133/strided_slice:output:0&backward_lstm_133/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/zeros/mul
backward_lstm_133/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_133/zeros/Less/yЏ
backward_lstm_133/zeros/LessLessbackward_lstm_133/zeros/mul:z:0'backward_lstm_133/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/zeros/Less
 backward_lstm_133/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_133/zeros/packed/1Ы
backward_lstm_133/zeros/packedPack(backward_lstm_133/strided_slice:output:0)backward_lstm_133/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_133/zeros/packed
backward_lstm_133/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_133/zeros/ConstН
backward_lstm_133/zerosFill'backward_lstm_133/zeros/packed:output:0&backward_lstm_133/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_133/zeros
backward_lstm_133/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_133/zeros_1/mul/yК
backward_lstm_133/zeros_1/mulMul(backward_lstm_133/strided_slice:output:0(backward_lstm_133/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/zeros_1/mul
 backward_lstm_133/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 backward_lstm_133/zeros_1/Less/yЗ
backward_lstm_133/zeros_1/LessLess!backward_lstm_133/zeros_1/mul:z:0)backward_lstm_133/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_133/zeros_1/Less
"backward_lstm_133/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_133/zeros_1/packed/1б
 backward_lstm_133/zeros_1/packedPack(backward_lstm_133/strided_slice:output:0+backward_lstm_133/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_133/zeros_1/packed
backward_lstm_133/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_133/zeros_1/ConstХ
backward_lstm_133/zeros_1Fill)backward_lstm_133/zeros_1/packed:output:0(backward_lstm_133/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_133/zeros_1
 backward_lstm_133/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_133/transpose/permё
backward_lstm_133/transpose	Transpose>backward_lstm_133/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_133/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_133/transpose
backward_lstm_133/Shape_1Shapebackward_lstm_133/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_133/Shape_1
'backward_lstm_133/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_133/strided_slice_1/stack 
)backward_lstm_133/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_1/stack_1 
)backward_lstm_133/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_1/stack_2к
!backward_lstm_133/strided_slice_1StridedSlice"backward_lstm_133/Shape_1:output:00backward_lstm_133/strided_slice_1/stack:output:02backward_lstm_133/strided_slice_1/stack_1:output:02backward_lstm_133/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_133/strided_slice_1Љ
-backward_lstm_133/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2/
-backward_lstm_133/TensorArrayV2/element_shapeњ
backward_lstm_133/TensorArrayV2TensorListReserve6backward_lstm_133/TensorArrayV2/element_shape:output:0*backward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_133/TensorArrayV2
 backward_lstm_133/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_133/ReverseV2/axisв
backward_lstm_133/ReverseV2	ReverseV2backward_lstm_133/transpose:y:0)backward_lstm_133/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_133/ReverseV2у
Gbackward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gbackward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shapeХ
9backward_lstm_133/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_133/ReverseV2:output:0Pbackward_lstm_133/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_133/TensorArrayUnstack/TensorListFromTensor
'backward_lstm_133/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_133/strided_slice_2/stack 
)backward_lstm_133/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_2/stack_1 
)backward_lstm_133/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_2/stack_2ш
!backward_lstm_133/strided_slice_2StridedSlicebackward_lstm_133/transpose:y:00backward_lstm_133/strided_slice_2/stack:output:02backward_lstm_133/strided_slice_2/stack_1:output:02backward_lstm_133/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2#
!backward_lstm_133/strided_slice_2ю
5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpReadVariableOp>backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype027
5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOpј
&backward_lstm_133/lstm_cell_401/MatMulMatMul*backward_lstm_133/strided_slice_2:output:0=backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_133/lstm_cell_401/MatMulє
7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOp@backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype029
7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOpє
(backward_lstm_133/lstm_cell_401/MatMul_1MatMul backward_lstm_133/zeros:output:0?backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_133/lstm_cell_401/MatMul_1ь
#backward_lstm_133/lstm_cell_401/addAddV20backward_lstm_133/lstm_cell_401/MatMul:product:02backward_lstm_133/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_133/lstm_cell_401/addэ
6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOp?backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype028
6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOpљ
'backward_lstm_133/lstm_cell_401/BiasAddBiasAdd'backward_lstm_133/lstm_cell_401/add:z:0>backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_133/lstm_cell_401/BiasAddЄ
/backward_lstm_133/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/backward_lstm_133/lstm_cell_401/split/split_dimП
%backward_lstm_133/lstm_cell_401/splitSplit8backward_lstm_133/lstm_cell_401/split/split_dim:output:00backward_lstm_133/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2'
%backward_lstm_133/lstm_cell_401/splitП
'backward_lstm_133/lstm_cell_401/SigmoidSigmoid.backward_lstm_133/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_133/lstm_cell_401/SigmoidУ
)backward_lstm_133/lstm_cell_401/Sigmoid_1Sigmoid.backward_lstm_133/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_133/lstm_cell_401/Sigmoid_1ж
#backward_lstm_133/lstm_cell_401/mulMul-backward_lstm_133/lstm_cell_401/Sigmoid_1:y:0"backward_lstm_133/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_133/lstm_cell_401/mulЖ
$backward_lstm_133/lstm_cell_401/ReluRelu.backward_lstm_133/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_133/lstm_cell_401/Reluш
%backward_lstm_133/lstm_cell_401/mul_1Mul+backward_lstm_133/lstm_cell_401/Sigmoid:y:02backward_lstm_133/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_133/lstm_cell_401/mul_1н
%backward_lstm_133/lstm_cell_401/add_1AddV2'backward_lstm_133/lstm_cell_401/mul:z:0)backward_lstm_133/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_133/lstm_cell_401/add_1У
)backward_lstm_133/lstm_cell_401/Sigmoid_2Sigmoid.backward_lstm_133/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_133/lstm_cell_401/Sigmoid_2Е
&backward_lstm_133/lstm_cell_401/Relu_1Relu)backward_lstm_133/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_133/lstm_cell_401/Relu_1ь
%backward_lstm_133/lstm_cell_401/mul_2Mul-backward_lstm_133/lstm_cell_401/Sigmoid_2:y:04backward_lstm_133/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_133/lstm_cell_401/mul_2Г
/backward_lstm_133/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   21
/backward_lstm_133/TensorArrayV2_1/element_shape
!backward_lstm_133/TensorArrayV2_1TensorListReserve8backward_lstm_133/TensorArrayV2_1/element_shape:output:0*backward_lstm_133/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_133/TensorArrayV2_1r
backward_lstm_133/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_133/time
'backward_lstm_133/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_133/Max/reduction_indicesЄ
backward_lstm_133/MaxMaxbackward_lstm_133/Cast:y:00backward_lstm_133/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/Maxt
backward_lstm_133/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_133/sub/y
backward_lstm_133/subSubbackward_lstm_133/Max:output:0 backward_lstm_133/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/sub
backward_lstm_133/Sub_1Subbackward_lstm_133/sub:z:0backward_lstm_133/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_133/Sub_1І
backward_lstm_133/zeros_like	ZerosLike)backward_lstm_133/lstm_cell_401/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_133/zeros_likeЃ
*backward_lstm_133/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*backward_lstm_133/while/maximum_iterations
$backward_lstm_133/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_133/while/loop_counterЅ	
backward_lstm_133/whileWhile-backward_lstm_133/while/loop_counter:output:03backward_lstm_133/while/maximum_iterations:output:0backward_lstm_133/time:output:0*backward_lstm_133/TensorArrayV2_1:handle:0 backward_lstm_133/zeros_like:y:0 backward_lstm_133/zeros:output:0"backward_lstm_133/zeros_1:output:0*backward_lstm_133/strided_slice_1:output:0Ibackward_lstm_133/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_133/Sub_1:z:0>backward_lstm_133_lstm_cell_401_matmul_readvariableop_resource@backward_lstm_133_lstm_cell_401_matmul_1_readvariableop_resource?backward_lstm_133_lstm_cell_401_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *1
body)R'
%backward_lstm_133_while_body_16888259*1
cond)R'
%backward_lstm_133_while_cond_16888258*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_133/whileй
Bbackward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2D
Bbackward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shapeЙ
4backward_lstm_133/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_133/while:output:3Kbackward_lstm_133/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype026
4backward_lstm_133/TensorArrayV2Stack/TensorListStackЅ
'backward_lstm_133/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2)
'backward_lstm_133/strided_slice_3/stack 
)backward_lstm_133/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_133/strided_slice_3/stack_1 
)backward_lstm_133/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_133/strided_slice_3/stack_2
!backward_lstm_133/strided_slice_3StridedSlice=backward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_133/strided_slice_3/stack:output:02backward_lstm_133/strided_slice_3/stack_1:output:02backward_lstm_133/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2#
!backward_lstm_133/strided_slice_3
"backward_lstm_133/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_133/transpose_1/permі
backward_lstm_133/transpose_1	Transpose=backward_lstm_133/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_133/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_133/transpose_1
backward_lstm_133/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_133/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisФ
concatConcatV2)forward_lstm_133/strided_slice_3:output:0*backward_lstm_133/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

Identityд
NoOpNoOp7^backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp6^backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp8^backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp^backward_lstm_133/while6^forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp5^forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp7^forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp^forward_lstm_133/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2p
6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp6backward_lstm_133/lstm_cell_401/BiasAdd/ReadVariableOp2n
5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp5backward_lstm_133/lstm_cell_401/MatMul/ReadVariableOp2r
7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp7backward_lstm_133/lstm_cell_401/MatMul_1/ReadVariableOp22
backward_lstm_133/whilebackward_lstm_133/while2n
5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp5forward_lstm_133/lstm_cell_400/BiasAdd/ReadVariableOp2l
4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp4forward_lstm_133/lstm_cell_400/MatMul/ReadVariableOp2p
6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp6forward_lstm_133/lstm_cell_400/MatMul_1/ReadVariableOp20
forward_lstm_133/whileforward_lstm_133/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
№?
л
while_body_16888996
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_400_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_400_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_400_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_400_matmul_readvariableop_resource:	ШG
4while_lstm_cell_400_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_400_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_400/BiasAdd/ReadVariableOpЂ)while/lstm_cell_400/MatMul/ReadVariableOpЂ+while/lstm_cell_400/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_400/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_400_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_400/MatMul/ReadVariableOpк
while/lstm_cell_400/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/MatMulв
+while/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_400_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_400/MatMul_1/ReadVariableOpУ
while/lstm_cell_400/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/MatMul_1М
while/lstm_cell_400/addAddV2$while/lstm_cell_400/MatMul:product:0&while/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/addЫ
*while/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_400_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_400/BiasAdd/ReadVariableOpЩ
while/lstm_cell_400/BiasAddBiasAddwhile/lstm_cell_400/add:z:02while/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/BiasAdd
#while/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_400/split/split_dim
while/lstm_cell_400/splitSplit,while/lstm_cell_400/split/split_dim:output:0$while/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_400/split
while/lstm_cell_400/SigmoidSigmoid"while/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Sigmoid
while/lstm_cell_400/Sigmoid_1Sigmoid"while/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Sigmoid_1Ѓ
while/lstm_cell_400/mulMul!while/lstm_cell_400/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/mul
while/lstm_cell_400/ReluRelu"while/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/ReluИ
while/lstm_cell_400/mul_1Mulwhile/lstm_cell_400/Sigmoid:y:0&while/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/mul_1­
while/lstm_cell_400/add_1AddV2while/lstm_cell_400/mul:z:0while/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/add_1
while/lstm_cell_400/Sigmoid_2Sigmoid"while/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Sigmoid_2
while/lstm_cell_400/Relu_1Reluwhile/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Relu_1М
while/lstm_cell_400/mul_2Mul!while/lstm_cell_400/Sigmoid_2:y:0(while/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_400/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_400/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_400/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_400/BiasAdd/ReadVariableOp*^while/lstm_cell_400/MatMul/ReadVariableOp,^while/lstm_cell_400/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_400_biasadd_readvariableop_resource5while_lstm_cell_400_biasadd_readvariableop_resource_0"n
4while_lstm_cell_400_matmul_1_readvariableop_resource6while_lstm_cell_400_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_400_matmul_readvariableop_resource4while_lstm_cell_400_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_400/BiasAdd/ReadVariableOp*while/lstm_cell_400/BiasAdd/ReadVariableOp2V
)while/lstm_cell_400/MatMul/ReadVariableOp)while/lstm_cell_400/MatMul/ReadVariableOp2Z
+while/lstm_cell_400/MatMul_1/ReadVariableOp+while/lstm_cell_400/MatMul_1/ReadVariableOp: 
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
Ъ

Э
&__inference_signature_wrapper_16887324

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
identityЂStatefulPartitionedCallЌ
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
GPU 2J 8 *,
f'R%
#__inference__wrapped_model_168842682
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
љ?
л
while_body_16889801
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_401_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_401_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_401_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_401_matmul_readvariableop_resource:	ШG
4while_lstm_cell_401_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_401_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_401/BiasAdd/ReadVariableOpЂ)while/lstm_cell_401/MatMul/ReadVariableOpЂ+while/lstm_cell_401/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_401/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_401_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_401/MatMul/ReadVariableOpк
while/lstm_cell_401/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/MatMulв
+while/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_401_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_401/MatMul_1/ReadVariableOpУ
while/lstm_cell_401/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/MatMul_1М
while/lstm_cell_401/addAddV2$while/lstm_cell_401/MatMul:product:0&while/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/addЫ
*while/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_401_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_401/BiasAdd/ReadVariableOpЩ
while/lstm_cell_401/BiasAddBiasAddwhile/lstm_cell_401/add:z:02while/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_401/BiasAdd
#while/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_401/split/split_dim
while/lstm_cell_401/splitSplit,while/lstm_cell_401/split/split_dim:output:0$while/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_401/split
while/lstm_cell_401/SigmoidSigmoid"while/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Sigmoid
while/lstm_cell_401/Sigmoid_1Sigmoid"while/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Sigmoid_1Ѓ
while/lstm_cell_401/mulMul!while/lstm_cell_401/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/mul
while/lstm_cell_401/ReluRelu"while/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/ReluИ
while/lstm_cell_401/mul_1Mulwhile/lstm_cell_401/Sigmoid:y:0&while/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/mul_1­
while/lstm_cell_401/add_1AddV2while/lstm_cell_401/mul:z:0while/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/add_1
while/lstm_cell_401/Sigmoid_2Sigmoid"while/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Sigmoid_2
while/lstm_cell_401/Relu_1Reluwhile/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/Relu_1М
while/lstm_cell_401/mul_2Mul!while/lstm_cell_401/Sigmoid_2:y:0(while/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_401/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_401/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_401/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_401/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_401/BiasAdd/ReadVariableOp*^while/lstm_cell_401/MatMul/ReadVariableOp,^while/lstm_cell_401/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_401_biasadd_readvariableop_resource5while_lstm_cell_401_biasadd_readvariableop_resource_0"n
4while_lstm_cell_401_matmul_1_readvariableop_resource6while_lstm_cell_401_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_401_matmul_readvariableop_resource4while_lstm_cell_401_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_401/BiasAdd/ReadVariableOp*while/lstm_cell_401/BiasAdd/ReadVariableOp2V
)while/lstm_cell_401/MatMul/ReadVariableOp)while/lstm_cell_401/MatMul/ReadVariableOp2Z
+while/lstm_cell_401/MatMul_1/ReadVariableOp+while/lstm_cell_401/MatMul_1/ReadVariableOp: 
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
Щ
Ѕ
$forward_lstm_133_while_cond_16886867>
:forward_lstm_133_while_forward_lstm_133_while_loop_counterD
@forward_lstm_133_while_forward_lstm_133_while_maximum_iterations&
"forward_lstm_133_while_placeholder(
$forward_lstm_133_while_placeholder_1(
$forward_lstm_133_while_placeholder_2(
$forward_lstm_133_while_placeholder_3(
$forward_lstm_133_while_placeholder_4@
<forward_lstm_133_while_less_forward_lstm_133_strided_slice_1X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16886867___redundant_placeholder0X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16886867___redundant_placeholder1X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16886867___redundant_placeholder2X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16886867___redundant_placeholder3X
Tforward_lstm_133_while_forward_lstm_133_while_cond_16886867___redundant_placeholder4#
forward_lstm_133_while_identity
Х
forward_lstm_133/while/LessLess"forward_lstm_133_while_placeholder<forward_lstm_133_while_less_forward_lstm_133_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_133/while/Less
forward_lstm_133/while/IdentityIdentityforward_lstm_133/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_133/while/Identity"K
forward_lstm_133_while_identity(forward_lstm_133/while/Identity:output:0*(
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
д
Т
3__inference_forward_lstm_133_layer_call_fn_16888756
inputs_0
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
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
GPU 2J 8 *W
fRRP
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_168846362
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
аf
ф
%backward_lstm_133_while_body_16887047@
<backward_lstm_133_while_backward_lstm_133_while_loop_counterF
Bbackward_lstm_133_while_backward_lstm_133_while_maximum_iterations'
#backward_lstm_133_while_placeholder)
%backward_lstm_133_while_placeholder_1)
%backward_lstm_133_while_placeholder_2)
%backward_lstm_133_while_placeholder_3)
%backward_lstm_133_while_placeholder_4?
;backward_lstm_133_while_backward_lstm_133_strided_slice_1_0{
wbackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_133_while_less_backward_lstm_133_sub_1_0Y
Fbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0:	Ш[
Hbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0:	2ШV
Gbackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0:	Ш$
 backward_lstm_133_while_identity&
"backward_lstm_133_while_identity_1&
"backward_lstm_133_while_identity_2&
"backward_lstm_133_while_identity_3&
"backward_lstm_133_while_identity_4&
"backward_lstm_133_while_identity_5&
"backward_lstm_133_while_identity_6=
9backward_lstm_133_while_backward_lstm_133_strided_slice_1y
ubackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_133_while_less_backward_lstm_133_sub_1W
Dbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource:	ШY
Fbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource:	2ШT
Ebackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource:	ШЂ<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOpЂ;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpЂ=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOpч
Ibackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2K
Ibackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeП
;backward_lstm_133/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_133_while_placeholderRbackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02=
;backward_lstm_133/while/TensorArrayV2Read/TensorListGetItemЯ
backward_lstm_133/while/LessLess6backward_lstm_133_while_less_backward_lstm_133_sub_1_0#backward_lstm_133_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_133/while/Less
;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpReadVariableOpFbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02=
;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpЂ
,backward_lstm_133/while/lstm_cell_401/MatMulMatMulBbackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem:item:0Cbackward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_133/while/lstm_cell_401/MatMul
=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOpHbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02?
=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp
.backward_lstm_133/while/lstm_cell_401/MatMul_1MatMul%backward_lstm_133_while_placeholder_3Ebackward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ20
.backward_lstm_133/while/lstm_cell_401/MatMul_1
)backward_lstm_133/while/lstm_cell_401/addAddV26backward_lstm_133/while/lstm_cell_401/MatMul:product:08backward_lstm_133/while/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_133/while/lstm_cell_401/add
<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOpGbackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02>
<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp
-backward_lstm_133/while/lstm_cell_401/BiasAddBiasAdd-backward_lstm_133/while/lstm_cell_401/add:z:0Dbackward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_133/while/lstm_cell_401/BiasAddА
5backward_lstm_133/while/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5backward_lstm_133/while/lstm_cell_401/split/split_dimз
+backward_lstm_133/while/lstm_cell_401/splitSplit>backward_lstm_133/while/lstm_cell_401/split/split_dim:output:06backward_lstm_133/while/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2-
+backward_lstm_133/while/lstm_cell_401/splitб
-backward_lstm_133/while/lstm_cell_401/SigmoidSigmoid4backward_lstm_133/while/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22/
-backward_lstm_133/while/lstm_cell_401/Sigmoidе
/backward_lstm_133/while/lstm_cell_401/Sigmoid_1Sigmoid4backward_lstm_133/while/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_133/while/lstm_cell_401/Sigmoid_1ы
)backward_lstm_133/while/lstm_cell_401/mulMul3backward_lstm_133/while/lstm_cell_401/Sigmoid_1:y:0%backward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_133/while/lstm_cell_401/mulШ
*backward_lstm_133/while/lstm_cell_401/ReluRelu4backward_lstm_133/while/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_133/while/lstm_cell_401/Relu
+backward_lstm_133/while/lstm_cell_401/mul_1Mul1backward_lstm_133/while/lstm_cell_401/Sigmoid:y:08backward_lstm_133/while/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_133/while/lstm_cell_401/mul_1ѕ
+backward_lstm_133/while/lstm_cell_401/add_1AddV2-backward_lstm_133/while/lstm_cell_401/mul:z:0/backward_lstm_133/while/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_133/while/lstm_cell_401/add_1е
/backward_lstm_133/while/lstm_cell_401/Sigmoid_2Sigmoid4backward_lstm_133/while/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_133/while/lstm_cell_401/Sigmoid_2Ч
,backward_lstm_133/while/lstm_cell_401/Relu_1Relu/backward_lstm_133/while/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_133/while/lstm_cell_401/Relu_1
+backward_lstm_133/while/lstm_cell_401/mul_2Mul3backward_lstm_133/while/lstm_cell_401/Sigmoid_2:y:0:backward_lstm_133/while/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_133/while/lstm_cell_401/mul_2і
backward_lstm_133/while/SelectSelect backward_lstm_133/while/Less:z:0/backward_lstm_133/while/lstm_cell_401/mul_2:z:0%backward_lstm_133_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_133/while/Selectњ
 backward_lstm_133/while/Select_1Select backward_lstm_133/while/Less:z:0/backward_lstm_133/while/lstm_cell_401/mul_2:z:0%backward_lstm_133_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_133/while/Select_1њ
 backward_lstm_133/while/Select_2Select backward_lstm_133/while/Less:z:0/backward_lstm_133/while/lstm_cell_401/add_1:z:0%backward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_133/while/Select_2Г
<backward_lstm_133/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_133_while_placeholder_1#backward_lstm_133_while_placeholder'backward_lstm_133/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_133/while/TensorArrayV2Write/TensorListSetItem
backward_lstm_133/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_133/while/add/yБ
backward_lstm_133/while/addAddV2#backward_lstm_133_while_placeholder&backward_lstm_133/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/while/add
backward_lstm_133/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_133/while/add_1/yа
backward_lstm_133/while/add_1AddV2<backward_lstm_133_while_backward_lstm_133_while_loop_counter(backward_lstm_133/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/while/add_1Г
 backward_lstm_133/while/IdentityIdentity!backward_lstm_133/while/add_1:z:0^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_133/while/Identityи
"backward_lstm_133/while/Identity_1IdentityBbackward_lstm_133_while_backward_lstm_133_while_maximum_iterations^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_133/while/Identity_1Е
"backward_lstm_133/while/Identity_2Identitybackward_lstm_133/while/add:z:0^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_133/while/Identity_2т
"backward_lstm_133/while/Identity_3IdentityLbackward_lstm_133/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_133/while/Identity_3Ю
"backward_lstm_133/while/Identity_4Identity'backward_lstm_133/while/Select:output:0^backward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_133/while/Identity_4а
"backward_lstm_133/while/Identity_5Identity)backward_lstm_133/while/Select_1:output:0^backward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_133/while/Identity_5а
"backward_lstm_133/while/Identity_6Identity)backward_lstm_133/while/Select_2:output:0^backward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_133/while/Identity_6Л
backward_lstm_133/while/NoOpNoOp=^backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp<^backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp>^backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_133/while/NoOp"x
9backward_lstm_133_while_backward_lstm_133_strided_slice_1;backward_lstm_133_while_backward_lstm_133_strided_slice_1_0"M
 backward_lstm_133_while_identity)backward_lstm_133/while/Identity:output:0"Q
"backward_lstm_133_while_identity_1+backward_lstm_133/while/Identity_1:output:0"Q
"backward_lstm_133_while_identity_2+backward_lstm_133/while/Identity_2:output:0"Q
"backward_lstm_133_while_identity_3+backward_lstm_133/while/Identity_3:output:0"Q
"backward_lstm_133_while_identity_4+backward_lstm_133/while/Identity_4:output:0"Q
"backward_lstm_133_while_identity_5+backward_lstm_133/while/Identity_5:output:0"Q
"backward_lstm_133_while_identity_6+backward_lstm_133/while/Identity_6:output:0"n
4backward_lstm_133_while_less_backward_lstm_133_sub_16backward_lstm_133_while_less_backward_lstm_133_sub_1_0"
Ebackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resourceGbackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0"
Fbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resourceHbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0"
Dbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resourceFbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0"№
ubackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2|
<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp2z
;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp2~
=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp: 
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
]
Ў
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_16888929
inputs_0?
,lstm_cell_400_matmul_readvariableop_resource:	ШA
.lstm_cell_400_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_400_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_400/BiasAdd/ReadVariableOpЂ#lstm_cell_400/MatMul/ReadVariableOpЂ%lstm_cell_400/MatMul_1/ReadVariableOpЂwhileF
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
#lstm_cell_400/MatMul/ReadVariableOpReadVariableOp,lstm_cell_400_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_400/MatMul/ReadVariableOpА
lstm_cell_400/MatMulMatMulstrided_slice_2:output:0+lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/MatMulО
%lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_400_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_400/MatMul_1/ReadVariableOpЌ
lstm_cell_400/MatMul_1MatMulzeros:output:0-lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/MatMul_1Є
lstm_cell_400/addAddV2lstm_cell_400/MatMul:product:0 lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/addЗ
$lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_400_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_400/BiasAdd/ReadVariableOpБ
lstm_cell_400/BiasAddBiasAddlstm_cell_400/add:z:0,lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_400/BiasAdd
lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_400/split/split_dimї
lstm_cell_400/splitSplit&lstm_cell_400/split/split_dim:output:0lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_400/split
lstm_cell_400/SigmoidSigmoidlstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Sigmoid
lstm_cell_400/Sigmoid_1Sigmoidlstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Sigmoid_1
lstm_cell_400/mulMullstm_cell_400/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/mul
lstm_cell_400/ReluRelulstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Relu 
lstm_cell_400/mul_1Mullstm_cell_400/Sigmoid:y:0 lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/mul_1
lstm_cell_400/add_1AddV2lstm_cell_400/mul:z:0lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/add_1
lstm_cell_400/Sigmoid_2Sigmoidlstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Sigmoid_2
lstm_cell_400/Relu_1Relulstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/Relu_1Є
lstm_cell_400/mul_2Mullstm_cell_400/Sigmoid_2:y:0"lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_400/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_400_matmul_readvariableop_resource.lstm_cell_400_matmul_1_readvariableop_resource-lstm_cell_400_biasadd_readvariableop_resource*
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
bodyR
while_body_16888845*
condR
while_cond_16888844*K
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
NoOpNoOp%^lstm_cell_400/BiasAdd/ReadVariableOp$^lstm_cell_400/MatMul/ReadVariableOp&^lstm_cell_400/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_400/BiasAdd/ReadVariableOp$lstm_cell_400/BiasAdd/ReadVariableOp2J
#lstm_cell_400/MatMul/ReadVariableOp#lstm_cell_400/MatMul/ReadVariableOp2N
%lstm_cell_400/MatMul_1/ReadVariableOp%lstm_cell_400/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
ЂИ
Ц 
$__inference__traced_restore_16890502
file_prefix3
!assignvariableop_dense_133_kernel:d/
!assignvariableop_1_dense_133_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: ]
Jassignvariableop_7_bidirectional_133_forward_lstm_133_lstm_cell_400_kernel:	Шg
Tassignvariableop_8_bidirectional_133_forward_lstm_133_lstm_cell_400_recurrent_kernel:	2ШW
Hassignvariableop_9_bidirectional_133_forward_lstm_133_lstm_cell_400_bias:	Ш_
Lassignvariableop_10_bidirectional_133_backward_lstm_133_lstm_cell_401_kernel:	Шi
Vassignvariableop_11_bidirectional_133_backward_lstm_133_lstm_cell_401_recurrent_kernel:	2ШY
Jassignvariableop_12_bidirectional_133_backward_lstm_133_lstm_cell_401_bias:	Ш#
assignvariableop_13_total: #
assignvariableop_14_count: =
+assignvariableop_15_adam_dense_133_kernel_m:d7
)assignvariableop_16_adam_dense_133_bias_m:e
Rassignvariableop_17_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_kernel_m:	Шo
\assignvariableop_18_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_recurrent_kernel_m:	2Ш_
Passignvariableop_19_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_bias_m:	Шf
Sassignvariableop_20_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_kernel_m:	Шp
]assignvariableop_21_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_recurrent_kernel_m:	2Ш`
Qassignvariableop_22_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_bias_m:	Ш=
+assignvariableop_23_adam_dense_133_kernel_v:d7
)assignvariableop_24_adam_dense_133_bias_v:e
Rassignvariableop_25_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_kernel_v:	Шo
\assignvariableop_26_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_recurrent_kernel_v:	2Ш_
Passignvariableop_27_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_bias_v:	Шf
Sassignvariableop_28_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_kernel_v:	Шp
]assignvariableop_29_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_recurrent_kernel_v:	2Ш`
Qassignvariableop_30_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_bias_v:	Ш@
.assignvariableop_31_adam_dense_133_kernel_vhat:d:
,assignvariableop_32_adam_dense_133_bias_vhat:h
Uassignvariableop_33_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_kernel_vhat:	Шr
_assignvariableop_34_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_recurrent_kernel_vhat:	2Шb
Sassignvariableop_35_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_bias_vhat:	Шi
Vassignvariableop_36_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_kernel_vhat:	Шs
`assignvariableop_37_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_recurrent_kernel_vhat:	2Шc
Tassignvariableop_38_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_bias_vhat:	Ш
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

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_dense_133_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1І
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_133_biasIdentity_1:output:0"/device:CPU:0*
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

Identity_7Я
AssignVariableOp_7AssignVariableOpJassignvariableop_7_bidirectional_133_forward_lstm_133_lstm_cell_400_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8й
AssignVariableOp_8AssignVariableOpTassignvariableop_8_bidirectional_133_forward_lstm_133_lstm_cell_400_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Э
AssignVariableOp_9AssignVariableOpHassignvariableop_9_bidirectional_133_forward_lstm_133_lstm_cell_400_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10д
AssignVariableOp_10AssignVariableOpLassignvariableop_10_bidirectional_133_backward_lstm_133_lstm_cell_401_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11о
AssignVariableOp_11AssignVariableOpVassignvariableop_11_bidirectional_133_backward_lstm_133_lstm_cell_401_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12в
AssignVariableOp_12AssignVariableOpJassignvariableop_12_bidirectional_133_backward_lstm_133_lstm_cell_401_biasIdentity_12:output:0"/device:CPU:0*
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
Identity_15Г
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_dense_133_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Б
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_dense_133_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17к
AssignVariableOp_17AssignVariableOpRassignvariableop_17_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18ф
AssignVariableOp_18AssignVariableOp\assignvariableop_18_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19и
AssignVariableOp_19AssignVariableOpPassignvariableop_19_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20л
AssignVariableOp_20AssignVariableOpSassignvariableop_20_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21х
AssignVariableOp_21AssignVariableOp]assignvariableop_21_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22й
AssignVariableOp_22AssignVariableOpQassignvariableop_22_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Г
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_133_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Б
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_133_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25к
AssignVariableOp_25AssignVariableOpRassignvariableop_25_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26ф
AssignVariableOp_26AssignVariableOp\assignvariableop_26_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27и
AssignVariableOp_27AssignVariableOpPassignvariableop_27_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28л
AssignVariableOp_28AssignVariableOpSassignvariableop_28_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29х
AssignVariableOp_29AssignVariableOp]assignvariableop_29_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30й
AssignVariableOp_30AssignVariableOpQassignvariableop_30_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ж
AssignVariableOp_31AssignVariableOp.assignvariableop_31_adam_dense_133_kernel_vhatIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Д
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_dense_133_bias_vhatIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33н
AssignVariableOp_33AssignVariableOpUassignvariableop_33_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_kernel_vhatIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34ч
AssignVariableOp_34AssignVariableOp_assignvariableop_34_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_recurrent_kernel_vhatIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35л
AssignVariableOp_35AssignVariableOpSassignvariableop_35_adam_bidirectional_133_forward_lstm_133_lstm_cell_400_bias_vhatIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36о
AssignVariableOp_36AssignVariableOpVassignvariableop_36_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_kernel_vhatIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37ш
AssignVariableOp_37AssignVariableOp`assignvariableop_37_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_recurrent_kernel_vhatIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38м
AssignVariableOp_38AssignVariableOpTassignvariableop_38_adam_bidirectional_133_backward_lstm_133_lstm_cell_401_bias_vhatIdentity_38:output:0"/device:CPU:0*
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
аf
ф
%backward_lstm_133_while_body_16888259@
<backward_lstm_133_while_backward_lstm_133_while_loop_counterF
Bbackward_lstm_133_while_backward_lstm_133_while_maximum_iterations'
#backward_lstm_133_while_placeholder)
%backward_lstm_133_while_placeholder_1)
%backward_lstm_133_while_placeholder_2)
%backward_lstm_133_while_placeholder_3)
%backward_lstm_133_while_placeholder_4?
;backward_lstm_133_while_backward_lstm_133_strided_slice_1_0{
wbackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_133_while_less_backward_lstm_133_sub_1_0Y
Fbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0:	Ш[
Hbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0:	2ШV
Gbackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0:	Ш$
 backward_lstm_133_while_identity&
"backward_lstm_133_while_identity_1&
"backward_lstm_133_while_identity_2&
"backward_lstm_133_while_identity_3&
"backward_lstm_133_while_identity_4&
"backward_lstm_133_while_identity_5&
"backward_lstm_133_while_identity_6=
9backward_lstm_133_while_backward_lstm_133_strided_slice_1y
ubackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_133_while_less_backward_lstm_133_sub_1W
Dbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource:	ШY
Fbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource:	2ШT
Ebackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource:	ШЂ<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOpЂ;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpЂ=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOpч
Ibackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2K
Ibackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shapeП
;backward_lstm_133/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_133_while_placeholderRbackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02=
;backward_lstm_133/while/TensorArrayV2Read/TensorListGetItemЯ
backward_lstm_133/while/LessLess6backward_lstm_133_while_less_backward_lstm_133_sub_1_0#backward_lstm_133_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_133/while/Less
;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpReadVariableOpFbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02=
;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOpЂ
,backward_lstm_133/while/lstm_cell_401/MatMulMatMulBbackward_lstm_133/while/TensorArrayV2Read/TensorListGetItem:item:0Cbackward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_133/while/lstm_cell_401/MatMul
=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOpHbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02?
=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp
.backward_lstm_133/while/lstm_cell_401/MatMul_1MatMul%backward_lstm_133_while_placeholder_3Ebackward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ20
.backward_lstm_133/while/lstm_cell_401/MatMul_1
)backward_lstm_133/while/lstm_cell_401/addAddV26backward_lstm_133/while/lstm_cell_401/MatMul:product:08backward_lstm_133/while/lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_133/while/lstm_cell_401/add
<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOpGbackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02>
<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp
-backward_lstm_133/while/lstm_cell_401/BiasAddBiasAdd-backward_lstm_133/while/lstm_cell_401/add:z:0Dbackward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_133/while/lstm_cell_401/BiasAddА
5backward_lstm_133/while/lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5backward_lstm_133/while/lstm_cell_401/split/split_dimз
+backward_lstm_133/while/lstm_cell_401/splitSplit>backward_lstm_133/while/lstm_cell_401/split/split_dim:output:06backward_lstm_133/while/lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2-
+backward_lstm_133/while/lstm_cell_401/splitб
-backward_lstm_133/while/lstm_cell_401/SigmoidSigmoid4backward_lstm_133/while/lstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22/
-backward_lstm_133/while/lstm_cell_401/Sigmoidе
/backward_lstm_133/while/lstm_cell_401/Sigmoid_1Sigmoid4backward_lstm_133/while/lstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_133/while/lstm_cell_401/Sigmoid_1ы
)backward_lstm_133/while/lstm_cell_401/mulMul3backward_lstm_133/while/lstm_cell_401/Sigmoid_1:y:0%backward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_133/while/lstm_cell_401/mulШ
*backward_lstm_133/while/lstm_cell_401/ReluRelu4backward_lstm_133/while/lstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_133/while/lstm_cell_401/Relu
+backward_lstm_133/while/lstm_cell_401/mul_1Mul1backward_lstm_133/while/lstm_cell_401/Sigmoid:y:08backward_lstm_133/while/lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_133/while/lstm_cell_401/mul_1ѕ
+backward_lstm_133/while/lstm_cell_401/add_1AddV2-backward_lstm_133/while/lstm_cell_401/mul:z:0/backward_lstm_133/while/lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_133/while/lstm_cell_401/add_1е
/backward_lstm_133/while/lstm_cell_401/Sigmoid_2Sigmoid4backward_lstm_133/while/lstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_133/while/lstm_cell_401/Sigmoid_2Ч
,backward_lstm_133/while/lstm_cell_401/Relu_1Relu/backward_lstm_133/while/lstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_133/while/lstm_cell_401/Relu_1
+backward_lstm_133/while/lstm_cell_401/mul_2Mul3backward_lstm_133/while/lstm_cell_401/Sigmoid_2:y:0:backward_lstm_133/while/lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_133/while/lstm_cell_401/mul_2і
backward_lstm_133/while/SelectSelect backward_lstm_133/while/Less:z:0/backward_lstm_133/while/lstm_cell_401/mul_2:z:0%backward_lstm_133_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_133/while/Selectњ
 backward_lstm_133/while/Select_1Select backward_lstm_133/while/Less:z:0/backward_lstm_133/while/lstm_cell_401/mul_2:z:0%backward_lstm_133_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_133/while/Select_1њ
 backward_lstm_133/while/Select_2Select backward_lstm_133/while/Less:z:0/backward_lstm_133/while/lstm_cell_401/add_1:z:0%backward_lstm_133_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_133/while/Select_2Г
<backward_lstm_133/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_133_while_placeholder_1#backward_lstm_133_while_placeholder'backward_lstm_133/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_133/while/TensorArrayV2Write/TensorListSetItem
backward_lstm_133/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_133/while/add/yБ
backward_lstm_133/while/addAddV2#backward_lstm_133_while_placeholder&backward_lstm_133/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/while/add
backward_lstm_133/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_133/while/add_1/yа
backward_lstm_133/while/add_1AddV2<backward_lstm_133_while_backward_lstm_133_while_loop_counter(backward_lstm_133/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_133/while/add_1Г
 backward_lstm_133/while/IdentityIdentity!backward_lstm_133/while/add_1:z:0^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_133/while/Identityи
"backward_lstm_133/while/Identity_1IdentityBbackward_lstm_133_while_backward_lstm_133_while_maximum_iterations^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_133/while/Identity_1Е
"backward_lstm_133/while/Identity_2Identitybackward_lstm_133/while/add:z:0^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_133/while/Identity_2т
"backward_lstm_133/while/Identity_3IdentityLbackward_lstm_133/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_133/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_133/while/Identity_3Ю
"backward_lstm_133/while/Identity_4Identity'backward_lstm_133/while/Select:output:0^backward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_133/while/Identity_4а
"backward_lstm_133/while/Identity_5Identity)backward_lstm_133/while/Select_1:output:0^backward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_133/while/Identity_5а
"backward_lstm_133/while/Identity_6Identity)backward_lstm_133/while/Select_2:output:0^backward_lstm_133/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_133/while/Identity_6Л
backward_lstm_133/while/NoOpNoOp=^backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp<^backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp>^backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_133/while/NoOp"x
9backward_lstm_133_while_backward_lstm_133_strided_slice_1;backward_lstm_133_while_backward_lstm_133_strided_slice_1_0"M
 backward_lstm_133_while_identity)backward_lstm_133/while/Identity:output:0"Q
"backward_lstm_133_while_identity_1+backward_lstm_133/while/Identity_1:output:0"Q
"backward_lstm_133_while_identity_2+backward_lstm_133/while/Identity_2:output:0"Q
"backward_lstm_133_while_identity_3+backward_lstm_133/while/Identity_3:output:0"Q
"backward_lstm_133_while_identity_4+backward_lstm_133/while/Identity_4:output:0"Q
"backward_lstm_133_while_identity_5+backward_lstm_133/while/Identity_5:output:0"Q
"backward_lstm_133_while_identity_6+backward_lstm_133/while/Identity_6:output:0"n
4backward_lstm_133_while_less_backward_lstm_133_sub_16backward_lstm_133_while_less_backward_lstm_133_sub_1_0"
Ebackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resourceGbackward_lstm_133_while_lstm_cell_401_biasadd_readvariableop_resource_0"
Fbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resourceHbackward_lstm_133_while_lstm_cell_401_matmul_1_readvariableop_resource_0"
Dbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resourceFbackward_lstm_133_while_lstm_cell_401_matmul_readvariableop_resource_0"№
ubackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_133_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_133_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2|
<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp<backward_lstm_133/while/lstm_cell_401/BiasAdd/ReadVariableOp2z
;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp;backward_lstm_133/while/lstm_cell_401/MatMul/ReadVariableOp2~
=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp=backward_lstm_133/while/lstm_cell_401/MatMul_1/ReadVariableOp: 
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
І_
­
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_16889885

inputs?
,lstm_cell_401_matmul_readvariableop_resource:	ШA
.lstm_cell_401_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_401_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_401/BiasAdd/ReadVariableOpЂ#lstm_cell_401/MatMul/ReadVariableOpЂ%lstm_cell_401/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_401/MatMul/ReadVariableOpReadVariableOp,lstm_cell_401_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_401/MatMul/ReadVariableOpА
lstm_cell_401/MatMulMatMulstrided_slice_2:output:0+lstm_cell_401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/MatMulО
%lstm_cell_401/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_401_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_401/MatMul_1/ReadVariableOpЌ
lstm_cell_401/MatMul_1MatMulzeros:output:0-lstm_cell_401/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/MatMul_1Є
lstm_cell_401/addAddV2lstm_cell_401/MatMul:product:0 lstm_cell_401/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/addЗ
$lstm_cell_401/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_401_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_401/BiasAdd/ReadVariableOpБ
lstm_cell_401/BiasAddBiasAddlstm_cell_401/add:z:0,lstm_cell_401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_401/BiasAdd
lstm_cell_401/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_401/split/split_dimї
lstm_cell_401/splitSplit&lstm_cell_401/split/split_dim:output:0lstm_cell_401/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_401/split
lstm_cell_401/SigmoidSigmoidlstm_cell_401/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Sigmoid
lstm_cell_401/Sigmoid_1Sigmoidlstm_cell_401/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Sigmoid_1
lstm_cell_401/mulMullstm_cell_401/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/mul
lstm_cell_401/ReluRelulstm_cell_401/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Relu 
lstm_cell_401/mul_1Mullstm_cell_401/Sigmoid:y:0 lstm_cell_401/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/mul_1
lstm_cell_401/add_1AddV2lstm_cell_401/mul:z:0lstm_cell_401/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/add_1
lstm_cell_401/Sigmoid_2Sigmoidlstm_cell_401/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Sigmoid_2
lstm_cell_401/Relu_1Relulstm_cell_401/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/Relu_1Є
lstm_cell_401/mul_2Mullstm_cell_401/Sigmoid_2:y:0"lstm_cell_401/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_401/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_401_matmul_readvariableop_resource.lstm_cell_401_matmul_1_readvariableop_resource-lstm_cell_401_biasadd_readvariableop_resource*
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
bodyR
while_body_16889801*
condR
while_cond_16889800*K
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
NoOpNoOp%^lstm_cell_401/BiasAdd/ReadVariableOp$^lstm_cell_401/MatMul/ReadVariableOp&^lstm_cell_401/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_401/BiasAdd/ReadVariableOp$lstm_cell_401/BiasAdd/ReadVariableOp2J
#lstm_cell_401/MatMul/ReadVariableOp#lstm_cell_401/MatMul/ReadVariableOp2N
%lstm_cell_401/MatMul_1/ReadVariableOp%lstm_cell_401/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
п
Э
while_cond_16885960
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16885960___redundant_placeholder06
2while_while_cond_16885960___redundant_placeholder16
2while_while_cond_16885960___redundant_placeholder26
2while_while_cond_16885960___redundant_placeholder3
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
ќ

и
1__inference_sequential_133_layer_call_fn_16887248

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
identityЂStatefulPartitionedCallе
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
GPU 2J 8 *U
fPRN
L__inference_sequential_133_layer_call_and_return_conditional_losses_168872072
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
ы	

4__inference_bidirectional_133_layer_call_fn_16887341
inputs_0
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
	unknown_2:	Ш
	unknown_3:	2Ш
	unknown_4:	Ш
identityЂStatefulPartitionedCallЕ
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
GPU 2J 8 *X
fSRQ
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_168858642
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
ў

K__inference_lstm_cell_401_layer_call_and_return_conditional_losses_16890234

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
Ў

Ѕ
4__inference_bidirectional_133_layer_call_fn_16887394

inputs
inputs_1	
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
	unknown_2:	Ш
	unknown_3:	2Ш
	unknown_4:	Ш
identityЂStatefulPartitionedCallО
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
GPU 2J 8 *X
fSRQ
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_168871442
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
ы	

4__inference_bidirectional_133_layer_call_fn_16887358
inputs_0
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
	unknown_2:	Ш
	unknown_3:	2Ш
	unknown_4:	Ш
identityЂStatefulPartitionedCallЕ
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
GPU 2J 8 *X
fSRQ
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_168862662
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
М
љ
0__inference_lstm_cell_401_layer_call_fn_16890170

inputs
states_0
states_1
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identity

identity_1

identity_2ЂStatefulPartitionedCallЦ
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
GPU 2J 8 *T
fORM
K__inference_lstm_cell_401_layer_call_and_return_conditional_losses_168851212
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
љ?
л
while_body_16886134
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_400_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_400_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_400_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_400_matmul_readvariableop_resource:	ШG
4while_lstm_cell_400_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_400_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_400/BiasAdd/ReadVariableOpЂ)while/lstm_cell_400/MatMul/ReadVariableOpЂ+while/lstm_cell_400/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_400/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_400_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_400/MatMul/ReadVariableOpк
while/lstm_cell_400/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/MatMulв
+while/lstm_cell_400/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_400_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_400/MatMul_1/ReadVariableOpУ
while/lstm_cell_400/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_400/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/MatMul_1М
while/lstm_cell_400/addAddV2$while/lstm_cell_400/MatMul:product:0&while/lstm_cell_400/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/addЫ
*while/lstm_cell_400/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_400_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_400/BiasAdd/ReadVariableOpЩ
while/lstm_cell_400/BiasAddBiasAddwhile/lstm_cell_400/add:z:02while/lstm_cell_400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_400/BiasAdd
#while/lstm_cell_400/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_400/split/split_dim
while/lstm_cell_400/splitSplit,while/lstm_cell_400/split/split_dim:output:0$while/lstm_cell_400/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_400/split
while/lstm_cell_400/SigmoidSigmoid"while/lstm_cell_400/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Sigmoid
while/lstm_cell_400/Sigmoid_1Sigmoid"while/lstm_cell_400/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Sigmoid_1Ѓ
while/lstm_cell_400/mulMul!while/lstm_cell_400/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/mul
while/lstm_cell_400/ReluRelu"while/lstm_cell_400/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/ReluИ
while/lstm_cell_400/mul_1Mulwhile/lstm_cell_400/Sigmoid:y:0&while/lstm_cell_400/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/mul_1­
while/lstm_cell_400/add_1AddV2while/lstm_cell_400/mul:z:0while/lstm_cell_400/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/add_1
while/lstm_cell_400/Sigmoid_2Sigmoid"while/lstm_cell_400/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Sigmoid_2
while/lstm_cell_400/Relu_1Reluwhile/lstm_cell_400/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/Relu_1М
while/lstm_cell_400/mul_2Mul!while/lstm_cell_400/Sigmoid_2:y:0(while/lstm_cell_400/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_400/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_400/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_400/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_400/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_400/BiasAdd/ReadVariableOp*^while/lstm_cell_400/MatMul/ReadVariableOp,^while/lstm_cell_400/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_400_biasadd_readvariableop_resource5while_lstm_cell_400_biasadd_readvariableop_resource_0"n
4while_lstm_cell_400_matmul_1_readvariableop_resource6while_lstm_cell_400_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_400_matmul_readvariableop_resource4while_lstm_cell_400_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_400/BiasAdd/ReadVariableOp*while/lstm_cell_400/BiasAdd/ReadVariableOp2V
)while/lstm_cell_400/MatMul/ReadVariableOp)while/lstm_cell_400/MatMul/ReadVariableOp2Z
+while/lstm_cell_400/MatMul_1/ReadVariableOp+while/lstm_cell_400/MatMul_1/ReadVariableOp: 
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
М
Т
Fsequential_133_bidirectional_133_backward_lstm_133_while_cond_16884163
~sequential_133_bidirectional_133_backward_lstm_133_while_sequential_133_bidirectional_133_backward_lstm_133_while_loop_counter
sequential_133_bidirectional_133_backward_lstm_133_while_sequential_133_bidirectional_133_backward_lstm_133_while_maximum_iterationsH
Dsequential_133_bidirectional_133_backward_lstm_133_while_placeholderJ
Fsequential_133_bidirectional_133_backward_lstm_133_while_placeholder_1J
Fsequential_133_bidirectional_133_backward_lstm_133_while_placeholder_2J
Fsequential_133_bidirectional_133_backward_lstm_133_while_placeholder_3J
Fsequential_133_bidirectional_133_backward_lstm_133_while_placeholder_4
sequential_133_bidirectional_133_backward_lstm_133_while_less_sequential_133_bidirectional_133_backward_lstm_133_strided_slice_1
sequential_133_bidirectional_133_backward_lstm_133_while_sequential_133_bidirectional_133_backward_lstm_133_while_cond_16884163___redundant_placeholder0
sequential_133_bidirectional_133_backward_lstm_133_while_sequential_133_bidirectional_133_backward_lstm_133_while_cond_16884163___redundant_placeholder1
sequential_133_bidirectional_133_backward_lstm_133_while_sequential_133_bidirectional_133_backward_lstm_133_while_cond_16884163___redundant_placeholder2
sequential_133_bidirectional_133_backward_lstm_133_while_sequential_133_bidirectional_133_backward_lstm_133_while_cond_16884163___redundant_placeholder3
sequential_133_bidirectional_133_backward_lstm_133_while_sequential_133_bidirectional_133_backward_lstm_133_while_cond_16884163___redundant_placeholder4E
Asequential_133_bidirectional_133_backward_lstm_133_while_identity
№
=sequential_133/bidirectional_133/backward_lstm_133/while/LessLessDsequential_133_bidirectional_133_backward_lstm_133_while_placeholdersequential_133_bidirectional_133_backward_lstm_133_while_less_sequential_133_bidirectional_133_backward_lstm_133_strided_slice_1*
T0*
_output_shapes
: 2?
=sequential_133/bidirectional_133/backward_lstm_133/while/Lessі
Asequential_133/bidirectional_133/backward_lstm_133/while/IdentityIdentityAsequential_133/bidirectional_133/backward_lstm_133/while/Less:z:0*
T0
*
_output_shapes
: 2C
Asequential_133/bidirectional_133/backward_lstm_133/while/Identity"
Asequential_133_bidirectional_133_backward_lstm_133_while_identityJsequential_133/bidirectional_133/backward_lstm_133/while/Identity:output:0*(
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
:"ЈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*х
serving_defaultб
9
args_0/
serving_default_args_0:0џџџџџџџџџ
9
args_0_1-
serving_default_args_0_1:0	џџџџџџџџџ=
	dense_1330
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ЩЛ
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
": d2dense_133/kernel
:2dense_133/bias
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
J:H	Ш27bidirectional_133/forward_lstm_133/lstm_cell_400/kernel
T:R	2Ш2Abidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel
D:BШ25bidirectional_133/forward_lstm_133/lstm_cell_400/bias
K:I	Ш28bidirectional_133/backward_lstm_133/lstm_cell_401/kernel
U:S	2Ш2Bbidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel
E:CШ26bidirectional_133/backward_lstm_133/lstm_cell_401/bias
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
':%d2Adam/dense_133/kernel/m
!:2Adam/dense_133/bias/m
O:M	Ш2>Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/m
Y:W	2Ш2HAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/m
I:GШ2<Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/m
P:N	Ш2?Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/m
Z:X	2Ш2IAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/m
J:HШ2=Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/m
':%d2Adam/dense_133/kernel/v
!:2Adam/dense_133/bias/v
O:M	Ш2>Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/v
Y:W	2Ш2HAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/v
I:GШ2<Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/v
P:N	Ш2?Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/v
Z:X	2Ш2IAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/v
J:HШ2=Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/v
*:(d2Adam/dense_133/kernel/vhat
$:"2Adam/dense_133/bias/vhat
R:P	Ш2AAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/kernel/vhat
\:Z	2Ш2KAdam/bidirectional_133/forward_lstm_133/lstm_cell_400/recurrent_kernel/vhat
L:JШ2?Adam/bidirectional_133/forward_lstm_133/lstm_cell_400/bias/vhat
S:Q	Ш2BAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/kernel/vhat
]:[	2Ш2LAdam/bidirectional_133/backward_lstm_133/lstm_cell_401/recurrent_kernel/vhat
M:KШ2@Adam/bidirectional_133/backward_lstm_133/lstm_cell_401/bias/vhat
Ќ2Љ
1__inference_sequential_133_layer_call_fn_16886755
1__inference_sequential_133_layer_call_fn_16887248Р
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
зBд
#__inference__wrapped_model_16884268args_0args_0_1"
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
т2п
L__inference_sequential_133_layer_call_and_return_conditional_losses_16887271
L__inference_sequential_133_layer_call_and_return_conditional_losses_16887294Р
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
Ф2С
4__inference_bidirectional_133_layer_call_fn_16887341
4__inference_bidirectional_133_layer_call_fn_16887358
4__inference_bidirectional_133_layer_call_fn_16887376
4__inference_bidirectional_133_layer_call_fn_16887394ц
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
А2­
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_16887696
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_16887998
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_16888356
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_16888714ц
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
ж2г
,__inference_dense_133_layer_call_fn_16888723Ђ
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
ё2ю
G__inference_dense_133_layer_call_and_return_conditional_losses_16888734Ђ
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
дBб
&__inference_signature_wrapper_16887324args_0args_0_1"
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
Џ2Ќ
3__inference_forward_lstm_133_layer_call_fn_16888745
3__inference_forward_lstm_133_layer_call_fn_16888756
3__inference_forward_lstm_133_layer_call_fn_16888767
3__inference_forward_lstm_133_layer_call_fn_16888778е
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
2
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_16888929
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_16889080
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_16889231
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_16889382е
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
Г2А
4__inference_backward_lstm_133_layer_call_fn_16889393
4__inference_backward_lstm_133_layer_call_fn_16889404
4__inference_backward_lstm_133_layer_call_fn_16889415
4__inference_backward_lstm_133_layer_call_fn_16889426е
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
2
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_16889579
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_16889732
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_16889885
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_16890038е
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
Ј2Ѕ
0__inference_lstm_cell_400_layer_call_fn_16890055
0__inference_lstm_cell_400_layer_call_fn_16890072О
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
о2л
K__inference_lstm_cell_400_layer_call_and_return_conditional_losses_16890104
K__inference_lstm_cell_400_layer_call_and_return_conditional_losses_16890136О
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
Ј2Ѕ
0__inference_lstm_cell_401_layer_call_fn_16890153
0__inference_lstm_cell_401_layer_call_fn_16890170О
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
о2л
K__inference_lstm_cell_401_layer_call_and_return_conditional_losses_16890202
K__inference_lstm_cell_401_layer_call_and_return_conditional_losses_16890234О
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
 Ч
#__inference__wrapped_model_16884268\ЂY
RЂO
MJ4Ђ1
!њџџџџџџџџџџџџџџџџџџ

`
	RaggedTensorSpec
Њ "5Њ2
0
	dense_133# 
	dense_133џџџџџџџџџа
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_16889579}OЂL
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
 а
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_16889732}OЂL
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
 в
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_16889885QЂN
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
 в
O__inference_backward_lstm_133_layer_call_and_return_conditional_losses_16890038QЂN
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
 Ј
4__inference_backward_lstm_133_layer_call_fn_16889393pOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2Ј
4__inference_backward_lstm_133_layer_call_fn_16889404pOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2Њ
4__inference_backward_lstm_133_layer_call_fn_16889415rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2Њ
4__inference_backward_lstm_133_layer_call_fn_16889426rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2с
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_16887696\ЂY
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
 с
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_16887998\ЂY
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
 ё
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_16888356lЂi
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
 ё
O__inference_bidirectional_133_layer_call_and_return_conditional_losses_16888714lЂi
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
 Й
4__inference_bidirectional_133_layer_call_fn_16887341\ЂY
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
Њ "џџџџџџџџџdЙ
4__inference_bidirectional_133_layer_call_fn_16887358\ЂY
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
Њ "џџџџџџџџџdЩ
4__inference_bidirectional_133_layer_call_fn_16887376lЂi
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
Њ "џџџџџџџџџdЩ
4__inference_bidirectional_133_layer_call_fn_16887394lЂi
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
Њ "џџџџџџџџџdЇ
G__inference_dense_133_layer_call_and_return_conditional_losses_16888734\/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџ
 
,__inference_dense_133_layer_call_fn_16888723O/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџЯ
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_16888929}OЂL
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
 Я
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_16889080}OЂL
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
 б
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_16889231QЂN
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
 б
N__inference_forward_lstm_133_layer_call_and_return_conditional_losses_16889382QЂN
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
 Ї
3__inference_forward_lstm_133_layer_call_fn_16888745pOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2Ї
3__inference_forward_lstm_133_layer_call_fn_16888756pOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2Љ
3__inference_forward_lstm_133_layer_call_fn_16888767rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2Љ
3__inference_forward_lstm_133_layer_call_fn_16888778rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2Э
K__inference_lstm_cell_400_layer_call_and_return_conditional_losses_16890104§Ђ}
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
 Э
K__inference_lstm_cell_400_layer_call_and_return_conditional_losses_16890136§Ђ}
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
 Ђ
0__inference_lstm_cell_400_layer_call_fn_16890055эЂ}
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
1/1џџџџџџџџџ2Ђ
0__inference_lstm_cell_400_layer_call_fn_16890072эЂ}
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
1/1џџџџџџџџџ2Э
K__inference_lstm_cell_401_layer_call_and_return_conditional_losses_16890202§Ђ}
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
 Э
K__inference_lstm_cell_401_layer_call_and_return_conditional_losses_16890234§Ђ}
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
 Ђ
0__inference_lstm_cell_401_layer_call_fn_16890153эЂ}
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
1/1џџџџџџџџџ2Ђ
0__inference_lstm_cell_401_layer_call_fn_16890170эЂ}
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
1/1џџџџџџџџџ2ш
L__inference_sequential_133_layer_call_and_return_conditional_losses_16887271dЂa
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
 ш
L__inference_sequential_133_layer_call_and_return_conditional_losses_16887294dЂa
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
 Р
1__inference_sequential_133_layer_call_fn_16886755dЂa
ZЂW
MJ4Ђ1
!њџџџџџџџџџџџџџџџџџџ

`
	RaggedTensorSpec
p 

 
Њ "џџџџџџџџџР
1__inference_sequential_133_layer_call_fn_16887248dЂa
ZЂW
MJ4Ђ1
!њџџџџџџџџџџџџџџџџџџ

`
	RaggedTensorSpec
p

 
Њ "џџџџџџџџџг
&__inference_signature_wrapper_16887324ЈeЂb
Ђ 
[ЊX
*
args_0 
args_0џџџџџџџџџ
*
args_0_1
args_0_1џџџџџџџџџ	"5Њ2
0
	dense_133# 
	dense_133џџџџџџџџџ