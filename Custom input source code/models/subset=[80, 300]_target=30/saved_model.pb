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
dense_150/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*!
shared_namedense_150/kernel
u
$dense_150/kernel/Read/ReadVariableOpReadVariableOpdense_150/kernel*
_output_shapes

:d*
dtype0
t
dense_150/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_150/bias
m
"dense_150/bias/Read/ReadVariableOpReadVariableOpdense_150/bias*
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
7bidirectional_150/forward_lstm_150/lstm_cell_451/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*H
shared_name97bidirectional_150/forward_lstm_150/lstm_cell_451/kernel
Ф
Kbidirectional_150/forward_lstm_150/lstm_cell_451/kernel/Read/ReadVariableOpReadVariableOp7bidirectional_150/forward_lstm_150/lstm_cell_451/kernel*
_output_shapes
:	Ш*
dtype0
п
Abidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*R
shared_nameCAbidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel
и
Ubidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/Read/ReadVariableOpReadVariableOpAbidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel*
_output_shapes
:	2Ш*
dtype0
У
5bidirectional_150/forward_lstm_150/lstm_cell_451/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*F
shared_name75bidirectional_150/forward_lstm_150/lstm_cell_451/bias
М
Ibidirectional_150/forward_lstm_150/lstm_cell_451/bias/Read/ReadVariableOpReadVariableOp5bidirectional_150/forward_lstm_150/lstm_cell_451/bias*
_output_shapes	
:Ш*
dtype0
Э
8bidirectional_150/backward_lstm_150/lstm_cell_452/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*I
shared_name:8bidirectional_150/backward_lstm_150/lstm_cell_452/kernel
Ц
Lbidirectional_150/backward_lstm_150/lstm_cell_452/kernel/Read/ReadVariableOpReadVariableOp8bidirectional_150/backward_lstm_150/lstm_cell_452/kernel*
_output_shapes
:	Ш*
dtype0
с
Bbidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*S
shared_nameDBbidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel
к
Vbidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/Read/ReadVariableOpReadVariableOpBbidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel*
_output_shapes
:	2Ш*
dtype0
Х
6bidirectional_150/backward_lstm_150/lstm_cell_452/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*G
shared_name86bidirectional_150/backward_lstm_150/lstm_cell_452/bias
О
Jbidirectional_150/backward_lstm_150/lstm_cell_452/bias/Read/ReadVariableOpReadVariableOp6bidirectional_150/backward_lstm_150/lstm_cell_452/bias*
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
Adam/dense_150/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_150/kernel/m

+Adam/dense_150/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_150/kernel/m*
_output_shapes

:d*
dtype0

Adam/dense_150/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_150/bias/m
{
)Adam/dense_150/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_150/bias/m*
_output_shapes
:*
dtype0
й
>Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*O
shared_name@>Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/m
в
RAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/m/Read/ReadVariableOpReadVariableOp>Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/m*
_output_shapes
:	Ш*
dtype0
э
HAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*Y
shared_nameJHAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/m
ц
\Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpHAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/m*
_output_shapes
:	2Ш*
dtype0
б
<Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*M
shared_name><Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/m
Ъ
PAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/m/Read/ReadVariableOpReadVariableOp<Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/m*
_output_shapes	
:Ш*
dtype0
л
?Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*P
shared_nameA?Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/m
д
SAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/m/Read/ReadVariableOpReadVariableOp?Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/m*
_output_shapes
:	Ш*
dtype0
я
IAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*Z
shared_nameKIAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/m
ш
]Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpIAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/m*
_output_shapes
:	2Ш*
dtype0
г
=Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*N
shared_name?=Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/m
Ь
QAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/m/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/m*
_output_shapes	
:Ш*
dtype0

Adam/dense_150/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_150/kernel/v

+Adam/dense_150/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_150/kernel/v*
_output_shapes

:d*
dtype0

Adam/dense_150/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_150/bias/v
{
)Adam/dense_150/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_150/bias/v*
_output_shapes
:*
dtype0
й
>Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*O
shared_name@>Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/v
в
RAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/v/Read/ReadVariableOpReadVariableOp>Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/v*
_output_shapes
:	Ш*
dtype0
э
HAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*Y
shared_nameJHAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/v
ц
\Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpHAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/v*
_output_shapes
:	2Ш*
dtype0
б
<Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*M
shared_name><Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/v
Ъ
PAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/v/Read/ReadVariableOpReadVariableOp<Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/v*
_output_shapes	
:Ш*
dtype0
л
?Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*P
shared_nameA?Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/v
д
SAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/v/Read/ReadVariableOpReadVariableOp?Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/v*
_output_shapes
:	Ш*
dtype0
я
IAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*Z
shared_nameKIAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/v
ш
]Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpIAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/v*
_output_shapes
:	2Ш*
dtype0
г
=Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*N
shared_name?=Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/v
Ь
QAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/v/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/v*
_output_shapes	
:Ш*
dtype0

Adam/dense_150/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*+
shared_nameAdam/dense_150/kernel/vhat

.Adam/dense_150/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_150/kernel/vhat*
_output_shapes

:d*
dtype0

Adam/dense_150/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/dense_150/bias/vhat

,Adam/dense_150/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_150/bias/vhat*
_output_shapes
:*
dtype0
п
AAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*R
shared_nameCAAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/vhat
и
UAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/vhat/Read/ReadVariableOpReadVariableOpAAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/vhat*
_output_shapes
:	Ш*
dtype0
ѓ
KAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*\
shared_nameMKAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/vhat
ь
_Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpKAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/vhat*
_output_shapes
:	2Ш*
dtype0
з
?Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*P
shared_nameA?Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/vhat
а
SAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/vhat/Read/ReadVariableOpReadVariableOp?Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/vhat*
_output_shapes	
:Ш*
dtype0
с
BAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*S
shared_nameDBAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/vhat
к
VAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/vhat/Read/ReadVariableOpReadVariableOpBAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/vhat*
_output_shapes
:	Ш*
dtype0
ѕ
LAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*]
shared_nameNLAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/vhat
ю
`Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpLAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/vhat*
_output_shapes
:	2Ш*
dtype0
й
@Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*Q
shared_nameB@Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/vhat
в
TAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/vhat/Read/ReadVariableOpReadVariableOp@Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/vhat*
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
VARIABLE_VALUEdense_150/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_150/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUE7bidirectional_150/forward_lstm_150/lstm_cell_451/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAbidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5bidirectional_150/forward_lstm_150/lstm_cell_451/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE8bidirectional_150/backward_lstm_150/lstm_cell_452/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEBbidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6bidirectional_150/backward_lstm_150/lstm_cell_452/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_150/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_150/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ё
VARIABLE_VALUEHAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ђ
VARIABLE_VALUEIAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_150/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_150/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ё
VARIABLE_VALUEHAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ђ
VARIABLE_VALUEIAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_150/kernel/vhatUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_150/bias/vhatSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/vhatEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ЇЄ
VARIABLE_VALUEKAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/vhatEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/vhatEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEBAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/vhatEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ЈЅ
VARIABLE_VALUELAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/vhatEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE@Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/vhatEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_0_17bidirectional_150/forward_lstm_150/lstm_cell_451/kernelAbidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel5bidirectional_150/forward_lstm_150/lstm_cell_451/bias8bidirectional_150/backward_lstm_150/lstm_cell_452/kernelBbidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel6bidirectional_150/backward_lstm_150/lstm_cell_452/biasdense_150/kerneldense_150/bias*
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
&__inference_signature_wrapper_18420812
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_150/kernel/Read/ReadVariableOp"dense_150/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpKbidirectional_150/forward_lstm_150/lstm_cell_451/kernel/Read/ReadVariableOpUbidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/Read/ReadVariableOpIbidirectional_150/forward_lstm_150/lstm_cell_451/bias/Read/ReadVariableOpLbidirectional_150/backward_lstm_150/lstm_cell_452/kernel/Read/ReadVariableOpVbidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/Read/ReadVariableOpJbidirectional_150/backward_lstm_150/lstm_cell_452/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_150/kernel/m/Read/ReadVariableOp)Adam/dense_150/bias/m/Read/ReadVariableOpRAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/m/Read/ReadVariableOp\Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/m/Read/ReadVariableOpPAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/m/Read/ReadVariableOpSAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/m/Read/ReadVariableOp]Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/m/Read/ReadVariableOpQAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/m/Read/ReadVariableOp+Adam/dense_150/kernel/v/Read/ReadVariableOp)Adam/dense_150/bias/v/Read/ReadVariableOpRAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/v/Read/ReadVariableOp\Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/v/Read/ReadVariableOpPAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/v/Read/ReadVariableOpSAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/v/Read/ReadVariableOp]Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/v/Read/ReadVariableOpQAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/v/Read/ReadVariableOp.Adam/dense_150/kernel/vhat/Read/ReadVariableOp,Adam/dense_150/bias/vhat/Read/ReadVariableOpUAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/vhat/Read/ReadVariableOp_Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/vhat/Read/ReadVariableOpSAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/vhat/Read/ReadVariableOpVAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/vhat/Read/ReadVariableOp`Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/vhat/Read/ReadVariableOpTAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/vhat/Read/ReadVariableOpConst*4
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
!__inference__traced_save_18423863
ў
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_150/kerneldense_150/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate7bidirectional_150/forward_lstm_150/lstm_cell_451/kernelAbidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel5bidirectional_150/forward_lstm_150/lstm_cell_451/bias8bidirectional_150/backward_lstm_150/lstm_cell_452/kernelBbidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel6bidirectional_150/backward_lstm_150/lstm_cell_452/biastotalcountAdam/dense_150/kernel/mAdam/dense_150/bias/m>Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/mHAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/m<Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/m?Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/mIAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/m=Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/mAdam/dense_150/kernel/vAdam/dense_150/bias/v>Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/vHAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/v<Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/v?Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/vIAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/v=Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/vAdam/dense_150/kernel/vhatAdam/dense_150/bias/vhatAAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/vhatKAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/vhat?Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/vhatBAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/vhatLAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/vhat@Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/vhat*3
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
$__inference__traced_restore_18423990Њї8
І_
­
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_18419533

inputs?
,lstm_cell_452_matmul_readvariableop_resource:	ШA
.lstm_cell_452_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_452_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_452/BiasAdd/ReadVariableOpЂ#lstm_cell_452/MatMul/ReadVariableOpЂ%lstm_cell_452/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_452/MatMul/ReadVariableOpReadVariableOp,lstm_cell_452_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_452/MatMul/ReadVariableOpА
lstm_cell_452/MatMulMatMulstrided_slice_2:output:0+lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/MatMulО
%lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_452_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_452/MatMul_1/ReadVariableOpЌ
lstm_cell_452/MatMul_1MatMulzeros:output:0-lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/MatMul_1Є
lstm_cell_452/addAddV2lstm_cell_452/MatMul:product:0 lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/addЗ
$lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_452_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_452/BiasAdd/ReadVariableOpБ
lstm_cell_452/BiasAddBiasAddlstm_cell_452/add:z:0,lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/BiasAdd
lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_452/split/split_dimї
lstm_cell_452/splitSplit&lstm_cell_452/split/split_dim:output:0lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_452/split
lstm_cell_452/SigmoidSigmoidlstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Sigmoid
lstm_cell_452/Sigmoid_1Sigmoidlstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Sigmoid_1
lstm_cell_452/mulMullstm_cell_452/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/mul
lstm_cell_452/ReluRelulstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Relu 
lstm_cell_452/mul_1Mullstm_cell_452/Sigmoid:y:0 lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/mul_1
lstm_cell_452/add_1AddV2lstm_cell_452/mul:z:0lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/add_1
lstm_cell_452/Sigmoid_2Sigmoidlstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Sigmoid_2
lstm_cell_452/Relu_1Relulstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Relu_1Є
lstm_cell_452/mul_2Mullstm_cell_452/Sigmoid_2:y:0"lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_452_matmul_readvariableop_resource.lstm_cell_452_matmul_1_readvariableop_resource-lstm_cell_452_biasadd_readvariableop_resource*
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
while_body_18419449*
condR
while_cond_18419448*K
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
NoOpNoOp%^lstm_cell_452/BiasAdd/ReadVariableOp$^lstm_cell_452/MatMul/ReadVariableOp&^lstm_cell_452/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_452/BiasAdd/ReadVariableOp$lstm_cell_452/BiasAdd/ReadVariableOp2J
#lstm_cell_452/MatMul/ReadVariableOp#lstm_cell_452/MatMul/ReadVariableOp2N
%lstm_cell_452/MatMul_1/ReadVariableOp%lstm_cell_452/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Џc
ц
!__inference__traced_save_18423863
file_prefix/
+savev2_dense_150_kernel_read_readvariableop-
)savev2_dense_150_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopV
Rsavev2_bidirectional_150_forward_lstm_150_lstm_cell_451_kernel_read_readvariableop`
\savev2_bidirectional_150_forward_lstm_150_lstm_cell_451_recurrent_kernel_read_readvariableopT
Psavev2_bidirectional_150_forward_lstm_150_lstm_cell_451_bias_read_readvariableopW
Ssavev2_bidirectional_150_backward_lstm_150_lstm_cell_452_kernel_read_readvariableopa
]savev2_bidirectional_150_backward_lstm_150_lstm_cell_452_recurrent_kernel_read_readvariableopU
Qsavev2_bidirectional_150_backward_lstm_150_lstm_cell_452_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_150_kernel_m_read_readvariableop4
0savev2_adam_dense_150_bias_m_read_readvariableop]
Ysavev2_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_kernel_m_read_readvariableopg
csavev2_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_recurrent_kernel_m_read_readvariableop[
Wsavev2_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_bias_m_read_readvariableop^
Zsavev2_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_kernel_m_read_readvariableoph
dsavev2_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_recurrent_kernel_m_read_readvariableop\
Xsavev2_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_bias_m_read_readvariableop6
2savev2_adam_dense_150_kernel_v_read_readvariableop4
0savev2_adam_dense_150_bias_v_read_readvariableop]
Ysavev2_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_kernel_v_read_readvariableopg
csavev2_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_recurrent_kernel_v_read_readvariableop[
Wsavev2_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_bias_v_read_readvariableop^
Zsavev2_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_kernel_v_read_readvariableoph
dsavev2_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_recurrent_kernel_v_read_readvariableop\
Xsavev2_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_bias_v_read_readvariableop9
5savev2_adam_dense_150_kernel_vhat_read_readvariableop7
3savev2_adam_dense_150_bias_vhat_read_readvariableop`
\savev2_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_kernel_vhat_read_readvariableopj
fsavev2_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_recurrent_kernel_vhat_read_readvariableop^
Zsavev2_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_bias_vhat_read_readvariableopa
]savev2_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_kernel_vhat_read_readvariableopk
gsavev2_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_recurrent_kernel_vhat_read_readvariableop_
[savev2_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_bias_vhat_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_150_kernel_read_readvariableop)savev2_dense_150_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopRsavev2_bidirectional_150_forward_lstm_150_lstm_cell_451_kernel_read_readvariableop\savev2_bidirectional_150_forward_lstm_150_lstm_cell_451_recurrent_kernel_read_readvariableopPsavev2_bidirectional_150_forward_lstm_150_lstm_cell_451_bias_read_readvariableopSsavev2_bidirectional_150_backward_lstm_150_lstm_cell_452_kernel_read_readvariableop]savev2_bidirectional_150_backward_lstm_150_lstm_cell_452_recurrent_kernel_read_readvariableopQsavev2_bidirectional_150_backward_lstm_150_lstm_cell_452_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_150_kernel_m_read_readvariableop0savev2_adam_dense_150_bias_m_read_readvariableopYsavev2_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_kernel_m_read_readvariableopcsavev2_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_recurrent_kernel_m_read_readvariableopWsavev2_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_bias_m_read_readvariableopZsavev2_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_kernel_m_read_readvariableopdsavev2_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_recurrent_kernel_m_read_readvariableopXsavev2_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_bias_m_read_readvariableop2savev2_adam_dense_150_kernel_v_read_readvariableop0savev2_adam_dense_150_bias_v_read_readvariableopYsavev2_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_kernel_v_read_readvariableopcsavev2_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_recurrent_kernel_v_read_readvariableopWsavev2_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_bias_v_read_readvariableopZsavev2_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_kernel_v_read_readvariableopdsavev2_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_recurrent_kernel_v_read_readvariableopXsavev2_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_bias_v_read_readvariableop5savev2_adam_dense_150_kernel_vhat_read_readvariableop3savev2_adam_dense_150_bias_vhat_read_readvariableop\savev2_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_kernel_vhat_read_readvariableopfsavev2_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_recurrent_kernel_vhat_read_readvariableopZsavev2_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_bias_vhat_read_readvariableop]savev2_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_kernel_vhat_read_readvariableopgsavev2_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_recurrent_kernel_vhat_read_readvariableop[savev2_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_bias_vhat_read_readvariableopsavev2_const"/device:CPU:0*
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
т
С
4__inference_backward_lstm_150_layer_call_fn_18422903

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
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_184193412
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
д
Т
3__inference_forward_lstm_150_layer_call_fn_18422233
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
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_184179142
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

в
Fsequential_150_bidirectional_150_backward_lstm_150_while_body_18417652
~sequential_150_bidirectional_150_backward_lstm_150_while_sequential_150_bidirectional_150_backward_lstm_150_while_loop_counter
sequential_150_bidirectional_150_backward_lstm_150_while_sequential_150_bidirectional_150_backward_lstm_150_while_maximum_iterationsH
Dsequential_150_bidirectional_150_backward_lstm_150_while_placeholderJ
Fsequential_150_bidirectional_150_backward_lstm_150_while_placeholder_1J
Fsequential_150_bidirectional_150_backward_lstm_150_while_placeholder_2J
Fsequential_150_bidirectional_150_backward_lstm_150_while_placeholder_3J
Fsequential_150_bidirectional_150_backward_lstm_150_while_placeholder_4
}sequential_150_bidirectional_150_backward_lstm_150_while_sequential_150_bidirectional_150_backward_lstm_150_strided_slice_1_0О
Йsequential_150_bidirectional_150_backward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_sequential_150_bidirectional_150_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0|
xsequential_150_bidirectional_150_backward_lstm_150_while_less_sequential_150_bidirectional_150_backward_lstm_150_sub_1_0z
gsequential_150_bidirectional_150_backward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0:	Ш|
isequential_150_bidirectional_150_backward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0:	2Шw
hsequential_150_bidirectional_150_backward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0:	ШE
Asequential_150_bidirectional_150_backward_lstm_150_while_identityG
Csequential_150_bidirectional_150_backward_lstm_150_while_identity_1G
Csequential_150_bidirectional_150_backward_lstm_150_while_identity_2G
Csequential_150_bidirectional_150_backward_lstm_150_while_identity_3G
Csequential_150_bidirectional_150_backward_lstm_150_while_identity_4G
Csequential_150_bidirectional_150_backward_lstm_150_while_identity_5G
Csequential_150_bidirectional_150_backward_lstm_150_while_identity_6
{sequential_150_bidirectional_150_backward_lstm_150_while_sequential_150_bidirectional_150_backward_lstm_150_strided_slice_1М
Зsequential_150_bidirectional_150_backward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_sequential_150_bidirectional_150_backward_lstm_150_tensorarrayunstack_tensorlistfromtensorz
vsequential_150_bidirectional_150_backward_lstm_150_while_less_sequential_150_bidirectional_150_backward_lstm_150_sub_1x
esequential_150_bidirectional_150_backward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource:	Шz
gsequential_150_bidirectional_150_backward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource:	2Шu
fsequential_150_bidirectional_150_backward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource:	ШЂ]sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOpЂ\sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpЂ^sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOpЉ
jsequential_150/bidirectional_150/backward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2l
jsequential_150/bidirectional_150/backward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shape
\sequential_150/bidirectional_150/backward_lstm_150/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЙsequential_150_bidirectional_150_backward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_sequential_150_bidirectional_150_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0Dsequential_150_bidirectional_150_backward_lstm_150_while_placeholderssequential_150/bidirectional_150/backward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02^
\sequential_150/bidirectional_150/backward_lstm_150/while/TensorArrayV2Read/TensorListGetItemє
=sequential_150/bidirectional_150/backward_lstm_150/while/LessLessxsequential_150_bidirectional_150_backward_lstm_150_while_less_sequential_150_bidirectional_150_backward_lstm_150_sub_1_0Dsequential_150_bidirectional_150_backward_lstm_150_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2?
=sequential_150/bidirectional_150/backward_lstm_150/while/Lessх
\sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpReadVariableOpgsequential_150_bidirectional_150_backward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02^
\sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpІ
Msequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMulMatMulcsequential_150/bidirectional_150/backward_lstm_150/while/TensorArrayV2Read/TensorListGetItem:item:0dsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2O
Msequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMulы
^sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOpisequential_150_bidirectional_150_backward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02`
^sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp
Osequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMul_1MatMulFsequential_150_bidirectional_150_backward_lstm_150_while_placeholder_3fsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2Q
Osequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMul_1
Jsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/addAddV2Wsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMul:product:0Ysequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2L
Jsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/addф
]sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOphsequential_150_bidirectional_150_backward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02_
]sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp
Nsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/BiasAddBiasAddNsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/add:z:0esequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2P
Nsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/BiasAddђ
Vsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2X
Vsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/split/split_dimл
Lsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/splitSplit_sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/split/split_dim:output:0Wsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2N
Lsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/splitД
Nsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/SigmoidSigmoidUsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22P
Nsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/SigmoidИ
Psequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/Sigmoid_1SigmoidUsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22R
Psequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/Sigmoid_1я
Jsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/mulMulTsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/Sigmoid_1:y:0Fsequential_150_bidirectional_150_backward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22L
Jsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/mulЋ
Ksequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/ReluReluUsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22M
Ksequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/Relu
Lsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/mul_1MulRsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/Sigmoid:y:0Ysequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22N
Lsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/mul_1љ
Lsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/add_1AddV2Nsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/mul:z:0Psequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22N
Lsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/add_1И
Psequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/Sigmoid_2SigmoidUsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22R
Psequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/Sigmoid_2Њ
Msequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/Relu_1ReluPsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22O
Msequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/Relu_1
Lsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/mul_2MulTsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/Sigmoid_2:y:0[sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22N
Lsequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/mul_2
?sequential_150/bidirectional_150/backward_lstm_150/while/SelectSelectAsequential_150/bidirectional_150/backward_lstm_150/while/Less:z:0Psequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/mul_2:z:0Fsequential_150_bidirectional_150_backward_lstm_150_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22A
?sequential_150/bidirectional_150/backward_lstm_150/while/Select
Asequential_150/bidirectional_150/backward_lstm_150/while/Select_1SelectAsequential_150/bidirectional_150/backward_lstm_150/while/Less:z:0Psequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/mul_2:z:0Fsequential_150_bidirectional_150_backward_lstm_150_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22C
Asequential_150/bidirectional_150/backward_lstm_150/while/Select_1
Asequential_150/bidirectional_150/backward_lstm_150/while/Select_2SelectAsequential_150/bidirectional_150/backward_lstm_150/while/Less:z:0Psequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/add_1:z:0Fsequential_150_bidirectional_150_backward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22C
Asequential_150/bidirectional_150/backward_lstm_150/while/Select_2и
]sequential_150/bidirectional_150/backward_lstm_150/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemFsequential_150_bidirectional_150_backward_lstm_150_while_placeholder_1Dsequential_150_bidirectional_150_backward_lstm_150_while_placeholderHsequential_150/bidirectional_150/backward_lstm_150/while/Select:output:0*
_output_shapes
: *
element_dtype02_
]sequential_150/bidirectional_150/backward_lstm_150/while/TensorArrayV2Write/TensorListSetItemТ
>sequential_150/bidirectional_150/backward_lstm_150/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_150/bidirectional_150/backward_lstm_150/while/add/yЕ
<sequential_150/bidirectional_150/backward_lstm_150/while/addAddV2Dsequential_150_bidirectional_150_backward_lstm_150_while_placeholderGsequential_150/bidirectional_150/backward_lstm_150/while/add/y:output:0*
T0*
_output_shapes
: 2>
<sequential_150/bidirectional_150/backward_lstm_150/while/addЦ
@sequential_150/bidirectional_150/backward_lstm_150/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2B
@sequential_150/bidirectional_150/backward_lstm_150/while/add_1/yѕ
>sequential_150/bidirectional_150/backward_lstm_150/while/add_1AddV2~sequential_150_bidirectional_150_backward_lstm_150_while_sequential_150_bidirectional_150_backward_lstm_150_while_loop_counterIsequential_150/bidirectional_150/backward_lstm_150/while/add_1/y:output:0*
T0*
_output_shapes
: 2@
>sequential_150/bidirectional_150/backward_lstm_150/while/add_1З
Asequential_150/bidirectional_150/backward_lstm_150/while/IdentityIdentityBsequential_150/bidirectional_150/backward_lstm_150/while/add_1:z:0>^sequential_150/bidirectional_150/backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2C
Asequential_150/bidirectional_150/backward_lstm_150/while/Identityў
Csequential_150/bidirectional_150/backward_lstm_150/while/Identity_1Identitysequential_150_bidirectional_150_backward_lstm_150_while_sequential_150_bidirectional_150_backward_lstm_150_while_maximum_iterations>^sequential_150/bidirectional_150/backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2E
Csequential_150/bidirectional_150/backward_lstm_150/while/Identity_1Й
Csequential_150/bidirectional_150/backward_lstm_150/while/Identity_2Identity@sequential_150/bidirectional_150/backward_lstm_150/while/add:z:0>^sequential_150/bidirectional_150/backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2E
Csequential_150/bidirectional_150/backward_lstm_150/while/Identity_2ц
Csequential_150/bidirectional_150/backward_lstm_150/while/Identity_3Identitymsequential_150/bidirectional_150/backward_lstm_150/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential_150/bidirectional_150/backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2E
Csequential_150/bidirectional_150/backward_lstm_150/while/Identity_3в
Csequential_150/bidirectional_150/backward_lstm_150/while/Identity_4IdentityHsequential_150/bidirectional_150/backward_lstm_150/while/Select:output:0>^sequential_150/bidirectional_150/backward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_150/bidirectional_150/backward_lstm_150/while/Identity_4д
Csequential_150/bidirectional_150/backward_lstm_150/while/Identity_5IdentityJsequential_150/bidirectional_150/backward_lstm_150/while/Select_1:output:0>^sequential_150/bidirectional_150/backward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_150/bidirectional_150/backward_lstm_150/while/Identity_5д
Csequential_150/bidirectional_150/backward_lstm_150/while/Identity_6IdentityJsequential_150/bidirectional_150/backward_lstm_150/while/Select_2:output:0>^sequential_150/bidirectional_150/backward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_150/bidirectional_150/backward_lstm_150/while/Identity_6р
=sequential_150/bidirectional_150/backward_lstm_150/while/NoOpNoOp^^sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp]^sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp_^sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2?
=sequential_150/bidirectional_150/backward_lstm_150/while/NoOp"
Asequential_150_bidirectional_150_backward_lstm_150_while_identityJsequential_150/bidirectional_150/backward_lstm_150/while/Identity:output:0"
Csequential_150_bidirectional_150_backward_lstm_150_while_identity_1Lsequential_150/bidirectional_150/backward_lstm_150/while/Identity_1:output:0"
Csequential_150_bidirectional_150_backward_lstm_150_while_identity_2Lsequential_150/bidirectional_150/backward_lstm_150/while/Identity_2:output:0"
Csequential_150_bidirectional_150_backward_lstm_150_while_identity_3Lsequential_150/bidirectional_150/backward_lstm_150/while/Identity_3:output:0"
Csequential_150_bidirectional_150_backward_lstm_150_while_identity_4Lsequential_150/bidirectional_150/backward_lstm_150/while/Identity_4:output:0"
Csequential_150_bidirectional_150_backward_lstm_150_while_identity_5Lsequential_150/bidirectional_150/backward_lstm_150/while/Identity_5:output:0"
Csequential_150_bidirectional_150_backward_lstm_150_while_identity_6Lsequential_150/bidirectional_150/backward_lstm_150/while/Identity_6:output:0"ђ
vsequential_150_bidirectional_150_backward_lstm_150_while_less_sequential_150_bidirectional_150_backward_lstm_150_sub_1xsequential_150_bidirectional_150_backward_lstm_150_while_less_sequential_150_bidirectional_150_backward_lstm_150_sub_1_0"в
fsequential_150_bidirectional_150_backward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resourcehsequential_150_bidirectional_150_backward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0"д
gsequential_150_bidirectional_150_backward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resourceisequential_150_bidirectional_150_backward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0"а
esequential_150_bidirectional_150_backward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resourcegsequential_150_bidirectional_150_backward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0"ќ
{sequential_150_bidirectional_150_backward_lstm_150_while_sequential_150_bidirectional_150_backward_lstm_150_strided_slice_1}sequential_150_bidirectional_150_backward_lstm_150_while_sequential_150_bidirectional_150_backward_lstm_150_strided_slice_1_0"і
Зsequential_150_bidirectional_150_backward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_sequential_150_bidirectional_150_backward_lstm_150_tensorarrayunstack_tensorlistfromtensorЙsequential_150_bidirectional_150_backward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_sequential_150_bidirectional_150_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2О
]sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp]sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp2М
\sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp\sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp2Р
^sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp^sequential_150/bidirectional_150/backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
т
С
4__inference_backward_lstm_150_layer_call_fn_18422914

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
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_184195332
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
п
Э
while_cond_18419256
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_18419256___redundant_placeholder06
2while_while_cond_18419256___redundant_placeholder16
2while_while_cond_18419256___redundant_placeholder26
2while_while_cond_18419256___redundant_placeholder3
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

Њ
Esequential_150_bidirectional_150_forward_lstm_150_while_cond_18417472
|sequential_150_bidirectional_150_forward_lstm_150_while_sequential_150_bidirectional_150_forward_lstm_150_while_loop_counter
sequential_150_bidirectional_150_forward_lstm_150_while_sequential_150_bidirectional_150_forward_lstm_150_while_maximum_iterationsG
Csequential_150_bidirectional_150_forward_lstm_150_while_placeholderI
Esequential_150_bidirectional_150_forward_lstm_150_while_placeholder_1I
Esequential_150_bidirectional_150_forward_lstm_150_while_placeholder_2I
Esequential_150_bidirectional_150_forward_lstm_150_while_placeholder_3I
Esequential_150_bidirectional_150_forward_lstm_150_while_placeholder_4
~sequential_150_bidirectional_150_forward_lstm_150_while_less_sequential_150_bidirectional_150_forward_lstm_150_strided_slice_1
sequential_150_bidirectional_150_forward_lstm_150_while_sequential_150_bidirectional_150_forward_lstm_150_while_cond_18417472___redundant_placeholder0
sequential_150_bidirectional_150_forward_lstm_150_while_sequential_150_bidirectional_150_forward_lstm_150_while_cond_18417472___redundant_placeholder1
sequential_150_bidirectional_150_forward_lstm_150_while_sequential_150_bidirectional_150_forward_lstm_150_while_cond_18417472___redundant_placeholder2
sequential_150_bidirectional_150_forward_lstm_150_while_sequential_150_bidirectional_150_forward_lstm_150_while_cond_18417472___redundant_placeholder3
sequential_150_bidirectional_150_forward_lstm_150_while_sequential_150_bidirectional_150_forward_lstm_150_while_cond_18417472___redundant_placeholder4D
@sequential_150_bidirectional_150_forward_lstm_150_while_identity
ъ
<sequential_150/bidirectional_150/forward_lstm_150/while/LessLessCsequential_150_bidirectional_150_forward_lstm_150_while_placeholder~sequential_150_bidirectional_150_forward_lstm_150_while_less_sequential_150_bidirectional_150_forward_lstm_150_strided_slice_1*
T0*
_output_shapes
: 2>
<sequential_150/bidirectional_150/forward_lstm_150/while/Lessѓ
@sequential_150/bidirectional_150/forward_lstm_150/while/IdentityIdentity@sequential_150/bidirectional_150/forward_lstm_150/while/Less:z:0*
T0
*
_output_shapes
: 2B
@sequential_150/bidirectional_150/forward_lstm_150/while/Identity"
@sequential_150_bidirectional_150_forward_lstm_150_while_identityIsequential_150/bidirectional_150/forward_lstm_150/while/Identity:output:0*(
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
ў

K__inference_lstm_cell_452_layer_call_and_return_conditional_losses_18423722

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
п
Э
while_cond_18419448
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_18419448___redundant_placeholder06
2while_while_cond_18419448___redundant_placeholder16
2while_while_cond_18419448___redundant_placeholder26
2while_while_cond_18419448___redundant_placeholder3
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
Щ
Ѕ
$forward_lstm_150_while_cond_18420355>
:forward_lstm_150_while_forward_lstm_150_while_loop_counterD
@forward_lstm_150_while_forward_lstm_150_while_maximum_iterations&
"forward_lstm_150_while_placeholder(
$forward_lstm_150_while_placeholder_1(
$forward_lstm_150_while_placeholder_2(
$forward_lstm_150_while_placeholder_3(
$forward_lstm_150_while_placeholder_4@
<forward_lstm_150_while_less_forward_lstm_150_strided_slice_1X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18420355___redundant_placeholder0X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18420355___redundant_placeholder1X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18420355___redundant_placeholder2X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18420355___redundant_placeholder3X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18420355___redundant_placeholder4#
forward_lstm_150_while_identity
Х
forward_lstm_150/while/LessLess"forward_lstm_150_while_placeholder<forward_lstm_150_while_less_forward_lstm_150_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_150/while/Less
forward_lstm_150/while/IdentityIdentityforward_lstm_150/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_150/while/Identity"K
forward_lstm_150_while_identity(forward_lstm_150/while/Identity:output:0*(
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
И
Ѓ
L__inference_sequential_150_layer_call_and_return_conditional_losses_18420224

inputs
inputs_1	-
bidirectional_150_18420193:	Ш-
bidirectional_150_18420195:	2Ш)
bidirectional_150_18420197:	Ш-
bidirectional_150_18420199:	Ш-
bidirectional_150_18420201:	2Ш)
bidirectional_150_18420203:	Ш$
dense_150_18420218:d 
dense_150_18420220:
identityЂ)bidirectional_150/StatefulPartitionedCallЂ!dense_150/StatefulPartitionedCallЪ
)bidirectional_150/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_150_18420193bidirectional_150_18420195bidirectional_150_18420197bidirectional_150_18420199bidirectional_150_18420201bidirectional_150_18420203*
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
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_184201922+
)bidirectional_150/StatefulPartitionedCallЫ
!dense_150/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_150/StatefulPartitionedCall:output:0dense_150_18420218dense_150_18420220*
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
G__inference_dense_150_layer_call_and_return_conditional_losses_184202172#
!dense_150/StatefulPartitionedCall
IdentityIdentity*dense_150/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp*^bidirectional_150/StatefulPartitionedCall"^dense_150/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2V
)bidirectional_150/StatefulPartitionedCall)bidirectional_150/StatefulPartitionedCall2F
!dense_150/StatefulPartitionedCall!dense_150/StatefulPartitionedCall:O K
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
%backward_lstm_150_while_cond_18422104@
<backward_lstm_150_while_backward_lstm_150_while_loop_counterF
Bbackward_lstm_150_while_backward_lstm_150_while_maximum_iterations'
#backward_lstm_150_while_placeholder)
%backward_lstm_150_while_placeholder_1)
%backward_lstm_150_while_placeholder_2)
%backward_lstm_150_while_placeholder_3)
%backward_lstm_150_while_placeholder_4B
>backward_lstm_150_while_less_backward_lstm_150_strided_slice_1Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18422104___redundant_placeholder0Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18422104___redundant_placeholder1Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18422104___redundant_placeholder2Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18422104___redundant_placeholder3Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18422104___redundant_placeholder4$
 backward_lstm_150_while_identity
Ъ
backward_lstm_150/while/LessLess#backward_lstm_150_while_placeholder>backward_lstm_150_while_less_backward_lstm_150_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_150/while/Less
 backward_lstm_150/while/IdentityIdentity backward_lstm_150/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_150/while/Identity"M
 backward_lstm_150_while_identity)backward_lstm_150/while/Identity:output:0*(
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
Н
я
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_18420192

inputs
inputs_1	P
=forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource:	ШR
?forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource:	2ШM
>forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource:	ШQ
>backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource:	ШS
@backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource:	2ШN
?backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource:	Ш
identityЂ6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpЂ5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpЂ7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpЂbackward_lstm_150/whileЂ5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpЂ4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpЂ6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOpЂforward_lstm_150/while
%forward_lstm_150/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_150/RaggedToTensor/zeros
%forward_lstm_150/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2'
%forward_lstm_150/RaggedToTensor/Const
4forward_lstm_150/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_150/RaggedToTensor/Const:output:0inputs.forward_lstm_150/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_150/RaggedToTensor/RaggedTensorToTensorФ
;forward_lstm_150/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_150/RaggedNestedRowLengths/strided_slice/stackШ
=forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1Ш
=forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2Љ
5forward_lstm_150/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_150/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask27
5forward_lstm_150/RaggedNestedRowLengths/strided_sliceШ
=forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stackе
?forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2A
?forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1Ь
?forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2Е
7forward_lstm_150/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask29
7forward_lstm_150/RaggedNestedRowLengths/strided_slice_1
+forward_lstm_150/RaggedNestedRowLengths/subSub>forward_lstm_150/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_150/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2-
+forward_lstm_150/RaggedNestedRowLengths/subЄ
forward_lstm_150/CastCast/forward_lstm_150/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_150/Cast
forward_lstm_150/ShapeShape=forward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_150/Shape
$forward_lstm_150/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_150/strided_slice/stack
&forward_lstm_150/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_150/strided_slice/stack_1
&forward_lstm_150/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_150/strided_slice/stack_2Ш
forward_lstm_150/strided_sliceStridedSliceforward_lstm_150/Shape:output:0-forward_lstm_150/strided_slice/stack:output:0/forward_lstm_150/strided_slice/stack_1:output:0/forward_lstm_150/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_150/strided_slice~
forward_lstm_150/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_150/zeros/mul/yА
forward_lstm_150/zeros/mulMul'forward_lstm_150/strided_slice:output:0%forward_lstm_150/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros/mul
forward_lstm_150/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_150/zeros/Less/yЋ
forward_lstm_150/zeros/LessLessforward_lstm_150/zeros/mul:z:0&forward_lstm_150/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros/Less
forward_lstm_150/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_150/zeros/packed/1Ч
forward_lstm_150/zeros/packedPack'forward_lstm_150/strided_slice:output:0(forward_lstm_150/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_150/zeros/packed
forward_lstm_150/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_150/zeros/ConstЙ
forward_lstm_150/zerosFill&forward_lstm_150/zeros/packed:output:0%forward_lstm_150/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/zeros
forward_lstm_150/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_150/zeros_1/mul/yЖ
forward_lstm_150/zeros_1/mulMul'forward_lstm_150/strided_slice:output:0'forward_lstm_150/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros_1/mul
forward_lstm_150/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
forward_lstm_150/zeros_1/Less/yГ
forward_lstm_150/zeros_1/LessLess forward_lstm_150/zeros_1/mul:z:0(forward_lstm_150/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros_1/Less
!forward_lstm_150/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_150/zeros_1/packed/1Э
forward_lstm_150/zeros_1/packedPack'forward_lstm_150/strided_slice:output:0*forward_lstm_150/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_150/zeros_1/packed
forward_lstm_150/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_150/zeros_1/ConstС
forward_lstm_150/zeros_1Fill(forward_lstm_150/zeros_1/packed:output:0'forward_lstm_150/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/zeros_1
forward_lstm_150/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_150/transpose/permэ
forward_lstm_150/transpose	Transpose=forward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_150/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_150/transpose
forward_lstm_150/Shape_1Shapeforward_lstm_150/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_150/Shape_1
&forward_lstm_150/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_150/strided_slice_1/stack
(forward_lstm_150/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_1/stack_1
(forward_lstm_150/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_1/stack_2д
 forward_lstm_150/strided_slice_1StridedSlice!forward_lstm_150/Shape_1:output:0/forward_lstm_150/strided_slice_1/stack:output:01forward_lstm_150/strided_slice_1/stack_1:output:01forward_lstm_150/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_150/strided_slice_1Ї
,forward_lstm_150/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,forward_lstm_150/TensorArrayV2/element_shapeі
forward_lstm_150/TensorArrayV2TensorListReserve5forward_lstm_150/TensorArrayV2/element_shape:output:0)forward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_150/TensorArrayV2с
Fforward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fforward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8forward_lstm_150/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_150/transpose:y:0Oforward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_150/TensorArrayUnstack/TensorListFromTensor
&forward_lstm_150/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_150/strided_slice_2/stack
(forward_lstm_150/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_2/stack_1
(forward_lstm_150/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_2/stack_2т
 forward_lstm_150/strided_slice_2StridedSliceforward_lstm_150/transpose:y:0/forward_lstm_150/strided_slice_2/stack:output:01forward_lstm_150/strided_slice_2/stack_1:output:01forward_lstm_150/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 forward_lstm_150/strided_slice_2ы
4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpReadVariableOp=forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpє
%forward_lstm_150/lstm_cell_451/MatMulMatMul)forward_lstm_150/strided_slice_2:output:0<forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_150/lstm_cell_451/MatMulё
6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOp?forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp№
'forward_lstm_150/lstm_cell_451/MatMul_1MatMulforward_lstm_150/zeros:output:0>forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_150/lstm_cell_451/MatMul_1ш
"forward_lstm_150/lstm_cell_451/addAddV2/forward_lstm_150/lstm_cell_451/MatMul:product:01forward_lstm_150/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_150/lstm_cell_451/addъ
5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOp>forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpѕ
&forward_lstm_150/lstm_cell_451/BiasAddBiasAdd&forward_lstm_150/lstm_cell_451/add:z:0=forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_150/lstm_cell_451/BiasAddЂ
.forward_lstm_150/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.forward_lstm_150/lstm_cell_451/split/split_dimЛ
$forward_lstm_150/lstm_cell_451/splitSplit7forward_lstm_150/lstm_cell_451/split/split_dim:output:0/forward_lstm_150/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$forward_lstm_150/lstm_cell_451/splitМ
&forward_lstm_150/lstm_cell_451/SigmoidSigmoid-forward_lstm_150/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_150/lstm_cell_451/SigmoidР
(forward_lstm_150/lstm_cell_451/Sigmoid_1Sigmoid-forward_lstm_150/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_150/lstm_cell_451/Sigmoid_1в
"forward_lstm_150/lstm_cell_451/mulMul,forward_lstm_150/lstm_cell_451/Sigmoid_1:y:0!forward_lstm_150/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_150/lstm_cell_451/mulГ
#forward_lstm_150/lstm_cell_451/ReluRelu-forward_lstm_150/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_150/lstm_cell_451/Reluф
$forward_lstm_150/lstm_cell_451/mul_1Mul*forward_lstm_150/lstm_cell_451/Sigmoid:y:01forward_lstm_150/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_150/lstm_cell_451/mul_1й
$forward_lstm_150/lstm_cell_451/add_1AddV2&forward_lstm_150/lstm_cell_451/mul:z:0(forward_lstm_150/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_150/lstm_cell_451/add_1Р
(forward_lstm_150/lstm_cell_451/Sigmoid_2Sigmoid-forward_lstm_150/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_150/lstm_cell_451/Sigmoid_2В
%forward_lstm_150/lstm_cell_451/Relu_1Relu(forward_lstm_150/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_150/lstm_cell_451/Relu_1ш
$forward_lstm_150/lstm_cell_451/mul_2Mul,forward_lstm_150/lstm_cell_451/Sigmoid_2:y:03forward_lstm_150/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_150/lstm_cell_451/mul_2Б
.forward_lstm_150/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.forward_lstm_150/TensorArrayV2_1/element_shapeќ
 forward_lstm_150/TensorArrayV2_1TensorListReserve7forward_lstm_150/TensorArrayV2_1/element_shape:output:0)forward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_150/TensorArrayV2_1p
forward_lstm_150/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_150/timeЃ
forward_lstm_150/zeros_like	ZerosLike(forward_lstm_150/lstm_cell_451/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/zeros_likeЁ
)forward_lstm_150/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)forward_lstm_150/while/maximum_iterations
#forward_lstm_150/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_150/while/loop_counter	
forward_lstm_150/whileWhile,forward_lstm_150/while/loop_counter:output:02forward_lstm_150/while/maximum_iterations:output:0forward_lstm_150/time:output:0)forward_lstm_150/TensorArrayV2_1:handle:0forward_lstm_150/zeros_like:y:0forward_lstm_150/zeros:output:0!forward_lstm_150/zeros_1:output:0)forward_lstm_150/strided_slice_1:output:0Hforward_lstm_150/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_150/Cast:y:0=forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource?forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource>forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource*
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
$forward_lstm_150_while_body_18419916*0
cond(R&
$forward_lstm_150_while_cond_18419915*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_150/whileз
Aforward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Aforward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3forward_lstm_150/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_150/while:output:3Jforward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3forward_lstm_150/TensorArrayV2Stack/TensorListStackЃ
&forward_lstm_150/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&forward_lstm_150/strided_slice_3/stack
(forward_lstm_150/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_150/strided_slice_3/stack_1
(forward_lstm_150/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_3/stack_2
 forward_lstm_150/strided_slice_3StridedSlice<forward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_150/strided_slice_3/stack:output:01forward_lstm_150/strided_slice_3/stack_1:output:01forward_lstm_150/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 forward_lstm_150/strided_slice_3
!forward_lstm_150/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_150/transpose_1/permђ
forward_lstm_150/transpose_1	Transpose<forward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_150/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_150/transpose_1
forward_lstm_150/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_150/runtime
&backward_lstm_150/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_150/RaggedToTensor/zeros
&backward_lstm_150/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2(
&backward_lstm_150/RaggedToTensor/Const
5backward_lstm_150/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_150/RaggedToTensor/Const:output:0inputs/backward_lstm_150/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_150/RaggedToTensor/RaggedTensorToTensorЦ
<backward_lstm_150/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_150/RaggedNestedRowLengths/strided_slice/stackЪ
>backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1Ъ
>backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2Ў
6backward_lstm_150/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_150/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask28
6backward_lstm_150/RaggedNestedRowLengths/strided_sliceЪ
>backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stackз
@backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2B
@backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1Ю
@backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2К
8backward_lstm_150/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2:
8backward_lstm_150/RaggedNestedRowLengths/strided_slice_1
,backward_lstm_150/RaggedNestedRowLengths/subSub?backward_lstm_150/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_150/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2.
,backward_lstm_150/RaggedNestedRowLengths/subЇ
backward_lstm_150/CastCast0backward_lstm_150/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_150/Cast 
backward_lstm_150/ShapeShape>backward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_150/Shape
%backward_lstm_150/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_150/strided_slice/stack
'backward_lstm_150/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_150/strided_slice/stack_1
'backward_lstm_150/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_150/strided_slice/stack_2Ю
backward_lstm_150/strided_sliceStridedSlice backward_lstm_150/Shape:output:0.backward_lstm_150/strided_slice/stack:output:00backward_lstm_150/strided_slice/stack_1:output:00backward_lstm_150/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_150/strided_slice
backward_lstm_150/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_150/zeros/mul/yД
backward_lstm_150/zeros/mulMul(backward_lstm_150/strided_slice:output:0&backward_lstm_150/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/zeros/mul
backward_lstm_150/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_150/zeros/Less/yЏ
backward_lstm_150/zeros/LessLessbackward_lstm_150/zeros/mul:z:0'backward_lstm_150/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/zeros/Less
 backward_lstm_150/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_150/zeros/packed/1Ы
backward_lstm_150/zeros/packedPack(backward_lstm_150/strided_slice:output:0)backward_lstm_150/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_150/zeros/packed
backward_lstm_150/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_150/zeros/ConstН
backward_lstm_150/zerosFill'backward_lstm_150/zeros/packed:output:0&backward_lstm_150/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_150/zeros
backward_lstm_150/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_150/zeros_1/mul/yК
backward_lstm_150/zeros_1/mulMul(backward_lstm_150/strided_slice:output:0(backward_lstm_150/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/zeros_1/mul
 backward_lstm_150/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 backward_lstm_150/zeros_1/Less/yЗ
backward_lstm_150/zeros_1/LessLess!backward_lstm_150/zeros_1/mul:z:0)backward_lstm_150/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_150/zeros_1/Less
"backward_lstm_150/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_150/zeros_1/packed/1б
 backward_lstm_150/zeros_1/packedPack(backward_lstm_150/strided_slice:output:0+backward_lstm_150/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_150/zeros_1/packed
backward_lstm_150/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_150/zeros_1/ConstХ
backward_lstm_150/zeros_1Fill)backward_lstm_150/zeros_1/packed:output:0(backward_lstm_150/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_150/zeros_1
 backward_lstm_150/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_150/transpose/permё
backward_lstm_150/transpose	Transpose>backward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_150/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_150/transpose
backward_lstm_150/Shape_1Shapebackward_lstm_150/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_150/Shape_1
'backward_lstm_150/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_150/strided_slice_1/stack 
)backward_lstm_150/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_1/stack_1 
)backward_lstm_150/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_1/stack_2к
!backward_lstm_150/strided_slice_1StridedSlice"backward_lstm_150/Shape_1:output:00backward_lstm_150/strided_slice_1/stack:output:02backward_lstm_150/strided_slice_1/stack_1:output:02backward_lstm_150/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_150/strided_slice_1Љ
-backward_lstm_150/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2/
-backward_lstm_150/TensorArrayV2/element_shapeњ
backward_lstm_150/TensorArrayV2TensorListReserve6backward_lstm_150/TensorArrayV2/element_shape:output:0*backward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_150/TensorArrayV2
 backward_lstm_150/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_150/ReverseV2/axisв
backward_lstm_150/ReverseV2	ReverseV2backward_lstm_150/transpose:y:0)backward_lstm_150/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_150/ReverseV2у
Gbackward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gbackward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeХ
9backward_lstm_150/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_150/ReverseV2:output:0Pbackward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_150/TensorArrayUnstack/TensorListFromTensor
'backward_lstm_150/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_150/strided_slice_2/stack 
)backward_lstm_150/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_2/stack_1 
)backward_lstm_150/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_2/stack_2ш
!backward_lstm_150/strided_slice_2StridedSlicebackward_lstm_150/transpose:y:00backward_lstm_150/strided_slice_2/stack:output:02backward_lstm_150/strided_slice_2/stack_1:output:02backward_lstm_150/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2#
!backward_lstm_150/strided_slice_2ю
5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpReadVariableOp>backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype027
5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpј
&backward_lstm_150/lstm_cell_452/MatMulMatMul*backward_lstm_150/strided_slice_2:output:0=backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_150/lstm_cell_452/MatMulє
7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOp@backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype029
7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpє
(backward_lstm_150/lstm_cell_452/MatMul_1MatMul backward_lstm_150/zeros:output:0?backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_150/lstm_cell_452/MatMul_1ь
#backward_lstm_150/lstm_cell_452/addAddV20backward_lstm_150/lstm_cell_452/MatMul:product:02backward_lstm_150/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_150/lstm_cell_452/addэ
6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOp?backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype028
6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpљ
'backward_lstm_150/lstm_cell_452/BiasAddBiasAdd'backward_lstm_150/lstm_cell_452/add:z:0>backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_150/lstm_cell_452/BiasAddЄ
/backward_lstm_150/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/backward_lstm_150/lstm_cell_452/split/split_dimП
%backward_lstm_150/lstm_cell_452/splitSplit8backward_lstm_150/lstm_cell_452/split/split_dim:output:00backward_lstm_150/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2'
%backward_lstm_150/lstm_cell_452/splitП
'backward_lstm_150/lstm_cell_452/SigmoidSigmoid.backward_lstm_150/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_150/lstm_cell_452/SigmoidУ
)backward_lstm_150/lstm_cell_452/Sigmoid_1Sigmoid.backward_lstm_150/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_150/lstm_cell_452/Sigmoid_1ж
#backward_lstm_150/lstm_cell_452/mulMul-backward_lstm_150/lstm_cell_452/Sigmoid_1:y:0"backward_lstm_150/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_150/lstm_cell_452/mulЖ
$backward_lstm_150/lstm_cell_452/ReluRelu.backward_lstm_150/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_150/lstm_cell_452/Reluш
%backward_lstm_150/lstm_cell_452/mul_1Mul+backward_lstm_150/lstm_cell_452/Sigmoid:y:02backward_lstm_150/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_150/lstm_cell_452/mul_1н
%backward_lstm_150/lstm_cell_452/add_1AddV2'backward_lstm_150/lstm_cell_452/mul:z:0)backward_lstm_150/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_150/lstm_cell_452/add_1У
)backward_lstm_150/lstm_cell_452/Sigmoid_2Sigmoid.backward_lstm_150/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_150/lstm_cell_452/Sigmoid_2Е
&backward_lstm_150/lstm_cell_452/Relu_1Relu)backward_lstm_150/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_150/lstm_cell_452/Relu_1ь
%backward_lstm_150/lstm_cell_452/mul_2Mul-backward_lstm_150/lstm_cell_452/Sigmoid_2:y:04backward_lstm_150/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_150/lstm_cell_452/mul_2Г
/backward_lstm_150/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   21
/backward_lstm_150/TensorArrayV2_1/element_shape
!backward_lstm_150/TensorArrayV2_1TensorListReserve8backward_lstm_150/TensorArrayV2_1/element_shape:output:0*backward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_150/TensorArrayV2_1r
backward_lstm_150/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_150/time
'backward_lstm_150/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_150/Max/reduction_indicesЄ
backward_lstm_150/MaxMaxbackward_lstm_150/Cast:y:00backward_lstm_150/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/Maxt
backward_lstm_150/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_150/sub/y
backward_lstm_150/subSubbackward_lstm_150/Max:output:0 backward_lstm_150/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/sub
backward_lstm_150/Sub_1Subbackward_lstm_150/sub:z:0backward_lstm_150/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_150/Sub_1І
backward_lstm_150/zeros_like	ZerosLike)backward_lstm_150/lstm_cell_452/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_150/zeros_likeЃ
*backward_lstm_150/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*backward_lstm_150/while/maximum_iterations
$backward_lstm_150/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_150/while/loop_counterЅ	
backward_lstm_150/whileWhile-backward_lstm_150/while/loop_counter:output:03backward_lstm_150/while/maximum_iterations:output:0backward_lstm_150/time:output:0*backward_lstm_150/TensorArrayV2_1:handle:0 backward_lstm_150/zeros_like:y:0 backward_lstm_150/zeros:output:0"backward_lstm_150/zeros_1:output:0*backward_lstm_150/strided_slice_1:output:0Ibackward_lstm_150/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_150/Sub_1:z:0>backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource@backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource?backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource*
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
%backward_lstm_150_while_body_18420095*1
cond)R'
%backward_lstm_150_while_cond_18420094*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_150/whileй
Bbackward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2D
Bbackward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeЙ
4backward_lstm_150/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_150/while:output:3Kbackward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype026
4backward_lstm_150/TensorArrayV2Stack/TensorListStackЅ
'backward_lstm_150/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2)
'backward_lstm_150/strided_slice_3/stack 
)backward_lstm_150/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_150/strided_slice_3/stack_1 
)backward_lstm_150/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_3/stack_2
!backward_lstm_150/strided_slice_3StridedSlice=backward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_150/strided_slice_3/stack:output:02backward_lstm_150/strided_slice_3/stack_1:output:02backward_lstm_150/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2#
!backward_lstm_150/strided_slice_3
"backward_lstm_150/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_150/transpose_1/permі
backward_lstm_150/transpose_1	Transpose=backward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_150/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_150/transpose_1
backward_lstm_150/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_150/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisФ
concatConcatV2)forward_lstm_150/strided_slice_3:output:0*backward_lstm_150/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp7^backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp6^backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp8^backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp^backward_lstm_150/while6^forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp5^forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp7^forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp^forward_lstm_150/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2p
6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp2n
5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp2r
7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp22
backward_lstm_150/whilebackward_lstm_150/while2n
5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp2l
4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp2p
6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp20
forward_lstm_150/whileforward_lstm_150/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
&
ј
while_body_18418477
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_452_18418501_0:	Ш1
while_lstm_cell_452_18418503_0:	2Ш-
while_lstm_cell_452_18418505_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_452_18418501:	Ш/
while_lstm_cell_452_18418503:	2Ш+
while_lstm_cell_452_18418505:	ШЂ+while/lstm_cell_452/StatefulPartitionedCallУ
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
+while/lstm_cell_452/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_452_18418501_0while_lstm_cell_452_18418503_0while_lstm_cell_452_18418505_0*
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
K__inference_lstm_cell_452_layer_call_and_return_conditional_losses_184184632-
+while/lstm_cell_452/StatefulPartitionedCallј
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_452/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_452/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѕ
while/Identity_5Identity4while/lstm_cell_452/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp,^while/lstm_cell_452/StatefulPartitionedCall*"
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
while_lstm_cell_452_18418501while_lstm_cell_452_18418501_0">
while_lstm_cell_452_18418503while_lstm_cell_452_18418503_0">
while_lstm_cell_452_18418505while_lstm_cell_452_18418505_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2Z
+while/lstm_cell_452/StatefulPartitionedCall+while/lstm_cell_452/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
і

K__inference_lstm_cell_451_layer_call_and_return_conditional_losses_18417977

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
&
ј
while_body_18418689
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_452_18418713_0:	Ш1
while_lstm_cell_452_18418715_0:	2Ш-
while_lstm_cell_452_18418717_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_452_18418713:	Ш/
while_lstm_cell_452_18418715:	2Ш+
while_lstm_cell_452_18418717:	ШЂ+while/lstm_cell_452/StatefulPartitionedCallУ
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
+while/lstm_cell_452/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_452_18418713_0while_lstm_cell_452_18418715_0while_lstm_cell_452_18418717_0*
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
K__inference_lstm_cell_452_layer_call_and_return_conditional_losses_184186092-
+while/lstm_cell_452/StatefulPartitionedCallј
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_452/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_452/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѕ
while/Identity_5Identity4while/lstm_cell_452/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp,^while/lstm_cell_452/StatefulPartitionedCall*"
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
while_lstm_cell_452_18418713while_lstm_cell_452_18418713_0">
while_lstm_cell_452_18418715while_lstm_cell_452_18418715_0">
while_lstm_cell_452_18418717while_lstm_cell_452_18418717_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2Z
+while/lstm_cell_452/StatefulPartitionedCall+while/lstm_cell_452/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
while_body_18419622
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_451_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_451_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_451_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_451_matmul_readvariableop_resource:	ШG
4while_lstm_cell_451_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_451_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_451/BiasAdd/ReadVariableOpЂ)while/lstm_cell_451/MatMul/ReadVariableOpЂ+while/lstm_cell_451/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_451/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_451_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_451/MatMul/ReadVariableOpк
while/lstm_cell_451/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/MatMulв
+while/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_451_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_451/MatMul_1/ReadVariableOpУ
while/lstm_cell_451/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/MatMul_1М
while/lstm_cell_451/addAddV2$while/lstm_cell_451/MatMul:product:0&while/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/addЫ
*while/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_451_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_451/BiasAdd/ReadVariableOpЩ
while/lstm_cell_451/BiasAddBiasAddwhile/lstm_cell_451/add:z:02while/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/BiasAdd
#while/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_451/split/split_dim
while/lstm_cell_451/splitSplit,while/lstm_cell_451/split/split_dim:output:0$while/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_451/split
while/lstm_cell_451/SigmoidSigmoid"while/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Sigmoid
while/lstm_cell_451/Sigmoid_1Sigmoid"while/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Sigmoid_1Ѓ
while/lstm_cell_451/mulMul!while/lstm_cell_451/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/mul
while/lstm_cell_451/ReluRelu"while/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/ReluИ
while/lstm_cell_451/mul_1Mulwhile/lstm_cell_451/Sigmoid:y:0&while/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/mul_1­
while/lstm_cell_451/add_1AddV2while/lstm_cell_451/mul:z:0while/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/add_1
while/lstm_cell_451/Sigmoid_2Sigmoid"while/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Sigmoid_2
while/lstm_cell_451/Relu_1Reluwhile/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Relu_1М
while/lstm_cell_451/mul_2Mul!while/lstm_cell_451/Sigmoid_2:y:0(while/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_451/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_451/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_451/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_451/BiasAdd/ReadVariableOp*^while/lstm_cell_451/MatMul/ReadVariableOp,^while/lstm_cell_451/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_451_biasadd_readvariableop_resource5while_lstm_cell_451_biasadd_readvariableop_resource_0"n
4while_lstm_cell_451_matmul_1_readvariableop_resource6while_lstm_cell_451_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_451_matmul_readvariableop_resource4while_lstm_cell_451_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_451/BiasAdd/ReadVariableOp*while/lstm_cell_451/BiasAdd/ReadVariableOp2V
)while/lstm_cell_451/MatMul/ReadVariableOp)while/lstm_cell_451/MatMul/ReadVariableOp2Z
+while/lstm_cell_451/MatMul_1/ReadVariableOp+while/lstm_cell_451/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
4__inference_backward_lstm_150_layer_call_fn_18422881
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
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_184185462
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
М
љ
0__inference_lstm_cell_451_layer_call_fn_18423543

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
K__inference_lstm_cell_451_layer_call_and_return_conditional_losses_184178312
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
§
Е
%backward_lstm_150_while_cond_18421097@
<backward_lstm_150_while_backward_lstm_150_while_loop_counterF
Bbackward_lstm_150_while_backward_lstm_150_while_maximum_iterations'
#backward_lstm_150_while_placeholder)
%backward_lstm_150_while_placeholder_1)
%backward_lstm_150_while_placeholder_2)
%backward_lstm_150_while_placeholder_3B
>backward_lstm_150_while_less_backward_lstm_150_strided_slice_1Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18421097___redundant_placeholder0Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18421097___redundant_placeholder1Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18421097___redundant_placeholder2Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18421097___redundant_placeholder3$
 backward_lstm_150_while_identity
Ъ
backward_lstm_150/while/LessLess#backward_lstm_150_while_placeholder>backward_lstm_150_while_less_backward_lstm_150_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_150/while/Less
 backward_lstm_150/while/IdentityIdentity backward_lstm_150/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_150/while/Identity"M
 backward_lstm_150_while_identity)backward_lstm_150/while/Identity:output:0*(
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
L__inference_sequential_150_layer_call_and_return_conditional_losses_18420759

inputs
inputs_1	-
bidirectional_150_18420740:	Ш-
bidirectional_150_18420742:	2Ш)
bidirectional_150_18420744:	Ш-
bidirectional_150_18420746:	Ш-
bidirectional_150_18420748:	2Ш)
bidirectional_150_18420750:	Ш$
dense_150_18420753:d 
dense_150_18420755:
identityЂ)bidirectional_150/StatefulPartitionedCallЂ!dense_150/StatefulPartitionedCallЪ
)bidirectional_150/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_150_18420740bidirectional_150_18420742bidirectional_150_18420744bidirectional_150_18420746bidirectional_150_18420748bidirectional_150_18420750*
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
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_184201922+
)bidirectional_150/StatefulPartitionedCallЫ
!dense_150/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_150/StatefulPartitionedCall:output:0dense_150_18420753dense_150_18420755*
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
G__inference_dense_150_layer_call_and_return_conditional_losses_184202172#
!dense_150/StatefulPartitionedCall
IdentityIdentity*dense_150/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp*^bidirectional_150/StatefulPartitionedCall"^dense_150/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2V
)bidirectional_150/StatefulPartitionedCall)bidirectional_150/StatefulPartitionedCall2F
!dense_150/StatefulPartitionedCall!dense_150/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
аf
ф
%backward_lstm_150_while_body_18420095@
<backward_lstm_150_while_backward_lstm_150_while_loop_counterF
Bbackward_lstm_150_while_backward_lstm_150_while_maximum_iterations'
#backward_lstm_150_while_placeholder)
%backward_lstm_150_while_placeholder_1)
%backward_lstm_150_while_placeholder_2)
%backward_lstm_150_while_placeholder_3)
%backward_lstm_150_while_placeholder_4?
;backward_lstm_150_while_backward_lstm_150_strided_slice_1_0{
wbackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_150_while_less_backward_lstm_150_sub_1_0Y
Fbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0:	Ш[
Hbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0:	2ШV
Gbackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0:	Ш$
 backward_lstm_150_while_identity&
"backward_lstm_150_while_identity_1&
"backward_lstm_150_while_identity_2&
"backward_lstm_150_while_identity_3&
"backward_lstm_150_while_identity_4&
"backward_lstm_150_while_identity_5&
"backward_lstm_150_while_identity_6=
9backward_lstm_150_while_backward_lstm_150_strided_slice_1y
ubackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_150_while_less_backward_lstm_150_sub_1W
Dbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource:	ШY
Fbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource:	2ШT
Ebackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource:	ШЂ<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOpЂ;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpЂ=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOpч
Ibackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2K
Ibackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeП
;backward_lstm_150/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_150_while_placeholderRbackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02=
;backward_lstm_150/while/TensorArrayV2Read/TensorListGetItemЯ
backward_lstm_150/while/LessLess6backward_lstm_150_while_less_backward_lstm_150_sub_1_0#backward_lstm_150_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_150/while/Less
;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpReadVariableOpFbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02=
;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpЂ
,backward_lstm_150/while/lstm_cell_452/MatMulMatMulBbackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem:item:0Cbackward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_150/while/lstm_cell_452/MatMul
=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOpHbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02?
=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp
.backward_lstm_150/while/lstm_cell_452/MatMul_1MatMul%backward_lstm_150_while_placeholder_3Ebackward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ20
.backward_lstm_150/while/lstm_cell_452/MatMul_1
)backward_lstm_150/while/lstm_cell_452/addAddV26backward_lstm_150/while/lstm_cell_452/MatMul:product:08backward_lstm_150/while/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_150/while/lstm_cell_452/add
<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOpGbackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02>
<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp
-backward_lstm_150/while/lstm_cell_452/BiasAddBiasAdd-backward_lstm_150/while/lstm_cell_452/add:z:0Dbackward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_150/while/lstm_cell_452/BiasAddА
5backward_lstm_150/while/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5backward_lstm_150/while/lstm_cell_452/split/split_dimз
+backward_lstm_150/while/lstm_cell_452/splitSplit>backward_lstm_150/while/lstm_cell_452/split/split_dim:output:06backward_lstm_150/while/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2-
+backward_lstm_150/while/lstm_cell_452/splitб
-backward_lstm_150/while/lstm_cell_452/SigmoidSigmoid4backward_lstm_150/while/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22/
-backward_lstm_150/while/lstm_cell_452/Sigmoidе
/backward_lstm_150/while/lstm_cell_452/Sigmoid_1Sigmoid4backward_lstm_150/while/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_150/while/lstm_cell_452/Sigmoid_1ы
)backward_lstm_150/while/lstm_cell_452/mulMul3backward_lstm_150/while/lstm_cell_452/Sigmoid_1:y:0%backward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_150/while/lstm_cell_452/mulШ
*backward_lstm_150/while/lstm_cell_452/ReluRelu4backward_lstm_150/while/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_150/while/lstm_cell_452/Relu
+backward_lstm_150/while/lstm_cell_452/mul_1Mul1backward_lstm_150/while/lstm_cell_452/Sigmoid:y:08backward_lstm_150/while/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_150/while/lstm_cell_452/mul_1ѕ
+backward_lstm_150/while/lstm_cell_452/add_1AddV2-backward_lstm_150/while/lstm_cell_452/mul:z:0/backward_lstm_150/while/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_150/while/lstm_cell_452/add_1е
/backward_lstm_150/while/lstm_cell_452/Sigmoid_2Sigmoid4backward_lstm_150/while/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_150/while/lstm_cell_452/Sigmoid_2Ч
,backward_lstm_150/while/lstm_cell_452/Relu_1Relu/backward_lstm_150/while/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_150/while/lstm_cell_452/Relu_1
+backward_lstm_150/while/lstm_cell_452/mul_2Mul3backward_lstm_150/while/lstm_cell_452/Sigmoid_2:y:0:backward_lstm_150/while/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_150/while/lstm_cell_452/mul_2і
backward_lstm_150/while/SelectSelect backward_lstm_150/while/Less:z:0/backward_lstm_150/while/lstm_cell_452/mul_2:z:0%backward_lstm_150_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_150/while/Selectњ
 backward_lstm_150/while/Select_1Select backward_lstm_150/while/Less:z:0/backward_lstm_150/while/lstm_cell_452/mul_2:z:0%backward_lstm_150_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_150/while/Select_1њ
 backward_lstm_150/while/Select_2Select backward_lstm_150/while/Less:z:0/backward_lstm_150/while/lstm_cell_452/add_1:z:0%backward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_150/while/Select_2Г
<backward_lstm_150/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_150_while_placeholder_1#backward_lstm_150_while_placeholder'backward_lstm_150/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_150/while/TensorArrayV2Write/TensorListSetItem
backward_lstm_150/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_150/while/add/yБ
backward_lstm_150/while/addAddV2#backward_lstm_150_while_placeholder&backward_lstm_150/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/while/add
backward_lstm_150/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_150/while/add_1/yа
backward_lstm_150/while/add_1AddV2<backward_lstm_150_while_backward_lstm_150_while_loop_counter(backward_lstm_150/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/while/add_1Г
 backward_lstm_150/while/IdentityIdentity!backward_lstm_150/while/add_1:z:0^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_150/while/Identityи
"backward_lstm_150/while/Identity_1IdentityBbackward_lstm_150_while_backward_lstm_150_while_maximum_iterations^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_150/while/Identity_1Е
"backward_lstm_150/while/Identity_2Identitybackward_lstm_150/while/add:z:0^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_150/while/Identity_2т
"backward_lstm_150/while/Identity_3IdentityLbackward_lstm_150/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_150/while/Identity_3Ю
"backward_lstm_150/while/Identity_4Identity'backward_lstm_150/while/Select:output:0^backward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_150/while/Identity_4а
"backward_lstm_150/while/Identity_5Identity)backward_lstm_150/while/Select_1:output:0^backward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_150/while/Identity_5а
"backward_lstm_150/while/Identity_6Identity)backward_lstm_150/while/Select_2:output:0^backward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_150/while/Identity_6Л
backward_lstm_150/while/NoOpNoOp=^backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp<^backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp>^backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_150/while/NoOp"x
9backward_lstm_150_while_backward_lstm_150_strided_slice_1;backward_lstm_150_while_backward_lstm_150_strided_slice_1_0"M
 backward_lstm_150_while_identity)backward_lstm_150/while/Identity:output:0"Q
"backward_lstm_150_while_identity_1+backward_lstm_150/while/Identity_1:output:0"Q
"backward_lstm_150_while_identity_2+backward_lstm_150/while/Identity_2:output:0"Q
"backward_lstm_150_while_identity_3+backward_lstm_150/while/Identity_3:output:0"Q
"backward_lstm_150_while_identity_4+backward_lstm_150/while/Identity_4:output:0"Q
"backward_lstm_150_while_identity_5+backward_lstm_150/while/Identity_5:output:0"Q
"backward_lstm_150_while_identity_6+backward_lstm_150/while/Identity_6:output:0"n
4backward_lstm_150_while_less_backward_lstm_150_sub_16backward_lstm_150_while_less_backward_lstm_150_sub_1_0"
Ebackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resourceGbackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0"
Fbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resourceHbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0"
Dbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resourceFbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0"№
ubackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2|
<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp2z
;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp2~
=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
і

K__inference_lstm_cell_451_layer_call_and_return_conditional_losses_18417831

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
Щ
Ѕ
$forward_lstm_150_while_cond_18421925>
:forward_lstm_150_while_forward_lstm_150_while_loop_counterD
@forward_lstm_150_while_forward_lstm_150_while_maximum_iterations&
"forward_lstm_150_while_placeholder(
$forward_lstm_150_while_placeholder_1(
$forward_lstm_150_while_placeholder_2(
$forward_lstm_150_while_placeholder_3(
$forward_lstm_150_while_placeholder_4@
<forward_lstm_150_while_less_forward_lstm_150_strided_slice_1X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18421925___redundant_placeholder0X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18421925___redundant_placeholder1X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18421925___redundant_placeholder2X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18421925___redundant_placeholder3X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18421925___redundant_placeholder4#
forward_lstm_150_while_identity
Х
forward_lstm_150/while/LessLess"forward_lstm_150_while_placeholder<forward_lstm_150_while_less_forward_lstm_150_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_150/while/Less
forward_lstm_150/while/IdentityIdentityforward_lstm_150/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_150/while/Identity"K
forward_lstm_150_while_identity(forward_lstm_150/while/Identity:output:0*(
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
_
Џ
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_18423220
inputs_0?
,lstm_cell_452_matmul_readvariableop_resource:	ШA
.lstm_cell_452_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_452_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_452/BiasAdd/ReadVariableOpЂ#lstm_cell_452/MatMul/ReadVariableOpЂ%lstm_cell_452/MatMul_1/ReadVariableOpЂwhileF
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
#lstm_cell_452/MatMul/ReadVariableOpReadVariableOp,lstm_cell_452_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_452/MatMul/ReadVariableOpА
lstm_cell_452/MatMulMatMulstrided_slice_2:output:0+lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/MatMulО
%lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_452_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_452/MatMul_1/ReadVariableOpЌ
lstm_cell_452/MatMul_1MatMulzeros:output:0-lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/MatMul_1Є
lstm_cell_452/addAddV2lstm_cell_452/MatMul:product:0 lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/addЗ
$lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_452_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_452/BiasAdd/ReadVariableOpБ
lstm_cell_452/BiasAddBiasAddlstm_cell_452/add:z:0,lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/BiasAdd
lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_452/split/split_dimї
lstm_cell_452/splitSplit&lstm_cell_452/split/split_dim:output:0lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_452/split
lstm_cell_452/SigmoidSigmoidlstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Sigmoid
lstm_cell_452/Sigmoid_1Sigmoidlstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Sigmoid_1
lstm_cell_452/mulMullstm_cell_452/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/mul
lstm_cell_452/ReluRelulstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Relu 
lstm_cell_452/mul_1Mullstm_cell_452/Sigmoid:y:0 lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/mul_1
lstm_cell_452/add_1AddV2lstm_cell_452/mul:z:0lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/add_1
lstm_cell_452/Sigmoid_2Sigmoidlstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Sigmoid_2
lstm_cell_452/Relu_1Relulstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Relu_1Є
lstm_cell_452/mul_2Mullstm_cell_452/Sigmoid_2:y:0"lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_452_matmul_readvariableop_resource.lstm_cell_452_matmul_1_readvariableop_resource-lstm_cell_452_biasadd_readvariableop_resource*
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
while_body_18423136*
condR
while_cond_18423135*K
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
NoOpNoOp%^lstm_cell_452/BiasAdd/ReadVariableOp$^lstm_cell_452/MatMul/ReadVariableOp&^lstm_cell_452/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_452/BiasAdd/ReadVariableOp$lstm_cell_452/BiasAdd/ReadVariableOp2J
#lstm_cell_452/MatMul/ReadVariableOp#lstm_cell_452/MatMul/ReadVariableOp2N
%lstm_cell_452/MatMul_1/ReadVariableOp%lstm_cell_452/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
п
Э
while_cond_18418688
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_18418688___redundant_placeholder06
2while_while_cond_18418688___redundant_placeholder16
2while_while_cond_18418688___redundant_placeholder26
2while_while_cond_18418688___redundant_placeholder3
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
while_body_18419449
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_452_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_452_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_452_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_452_matmul_readvariableop_resource:	ШG
4while_lstm_cell_452_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_452_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_452/BiasAdd/ReadVariableOpЂ)while/lstm_cell_452/MatMul/ReadVariableOpЂ+while/lstm_cell_452/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_452/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_452_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_452/MatMul/ReadVariableOpк
while/lstm_cell_452/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/MatMulв
+while/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_452_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_452/MatMul_1/ReadVariableOpУ
while/lstm_cell_452/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/MatMul_1М
while/lstm_cell_452/addAddV2$while/lstm_cell_452/MatMul:product:0&while/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/addЫ
*while/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_452_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_452/BiasAdd/ReadVariableOpЩ
while/lstm_cell_452/BiasAddBiasAddwhile/lstm_cell_452/add:z:02while/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/BiasAdd
#while/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_452/split/split_dim
while/lstm_cell_452/splitSplit,while/lstm_cell_452/split/split_dim:output:0$while/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_452/split
while/lstm_cell_452/SigmoidSigmoid"while/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Sigmoid
while/lstm_cell_452/Sigmoid_1Sigmoid"while/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Sigmoid_1Ѓ
while/lstm_cell_452/mulMul!while/lstm_cell_452/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/mul
while/lstm_cell_452/ReluRelu"while/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/ReluИ
while/lstm_cell_452/mul_1Mulwhile/lstm_cell_452/Sigmoid:y:0&while/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/mul_1­
while/lstm_cell_452/add_1AddV2while/lstm_cell_452/mul:z:0while/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/add_1
while/lstm_cell_452/Sigmoid_2Sigmoid"while/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Sigmoid_2
while/lstm_cell_452/Relu_1Reluwhile/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Relu_1М
while/lstm_cell_452/mul_2Mul!while/lstm_cell_452/Sigmoid_2:y:0(while/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_452/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_452/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_452/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_452/BiasAdd/ReadVariableOp*^while/lstm_cell_452/MatMul/ReadVariableOp,^while/lstm_cell_452/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_452_biasadd_readvariableop_resource5while_lstm_cell_452_biasadd_readvariableop_resource_0"n
4while_lstm_cell_452_matmul_1_readvariableop_resource6while_lstm_cell_452_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_452_matmul_readvariableop_resource4while_lstm_cell_452_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_452/BiasAdd/ReadVariableOp*while/lstm_cell_452/BiasAdd/ReadVariableOp2V
)while/lstm_cell_452/MatMul/ReadVariableOp)while/lstm_cell_452/MatMul/ReadVariableOp2Z
+while/lstm_cell_452/MatMul_1/ReadVariableOp+while/lstm_cell_452/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
4__inference_backward_lstm_150_layer_call_fn_18422892
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
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_184187582
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
п
Э
while_cond_18423288
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_18423288___redundant_placeholder06
2while_while_cond_18423288___redundant_placeholder16
2while_while_cond_18423288___redundant_placeholder26
2while_while_cond_18423288___redundant_placeholder3
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
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_18418124

inputs)
lstm_cell_451_18418042:	Ш)
lstm_cell_451_18418044:	2Ш%
lstm_cell_451_18418046:	Ш
identityЂ%lstm_cell_451/StatefulPartitionedCallЂwhileD
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
%lstm_cell_451/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_451_18418042lstm_cell_451_18418044lstm_cell_451_18418046*
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
K__inference_lstm_cell_451_layer_call_and_return_conditional_losses_184179772'
%lstm_cell_451/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_451_18418042lstm_cell_451_18418044lstm_cell_451_18418046*
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
while_body_18418055*
condR
while_cond_18418054*K
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
NoOpNoOp&^lstm_cell_451/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2N
%lstm_cell_451/StatefulPartitionedCall%lstm_cell_451/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
У
Џ
Esequential_150_bidirectional_150_forward_lstm_150_while_body_18417473
|sequential_150_bidirectional_150_forward_lstm_150_while_sequential_150_bidirectional_150_forward_lstm_150_while_loop_counter
sequential_150_bidirectional_150_forward_lstm_150_while_sequential_150_bidirectional_150_forward_lstm_150_while_maximum_iterationsG
Csequential_150_bidirectional_150_forward_lstm_150_while_placeholderI
Esequential_150_bidirectional_150_forward_lstm_150_while_placeholder_1I
Esequential_150_bidirectional_150_forward_lstm_150_while_placeholder_2I
Esequential_150_bidirectional_150_forward_lstm_150_while_placeholder_3I
Esequential_150_bidirectional_150_forward_lstm_150_while_placeholder_4
{sequential_150_bidirectional_150_forward_lstm_150_while_sequential_150_bidirectional_150_forward_lstm_150_strided_slice_1_0М
Зsequential_150_bidirectional_150_forward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_sequential_150_bidirectional_150_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0|
xsequential_150_bidirectional_150_forward_lstm_150_while_greater_sequential_150_bidirectional_150_forward_lstm_150_cast_0y
fsequential_150_bidirectional_150_forward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0:	Ш{
hsequential_150_bidirectional_150_forward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0:	2Шv
gsequential_150_bidirectional_150_forward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0:	ШD
@sequential_150_bidirectional_150_forward_lstm_150_while_identityF
Bsequential_150_bidirectional_150_forward_lstm_150_while_identity_1F
Bsequential_150_bidirectional_150_forward_lstm_150_while_identity_2F
Bsequential_150_bidirectional_150_forward_lstm_150_while_identity_3F
Bsequential_150_bidirectional_150_forward_lstm_150_while_identity_4F
Bsequential_150_bidirectional_150_forward_lstm_150_while_identity_5F
Bsequential_150_bidirectional_150_forward_lstm_150_while_identity_6}
ysequential_150_bidirectional_150_forward_lstm_150_while_sequential_150_bidirectional_150_forward_lstm_150_strided_slice_1К
Еsequential_150_bidirectional_150_forward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_sequential_150_bidirectional_150_forward_lstm_150_tensorarrayunstack_tensorlistfromtensorz
vsequential_150_bidirectional_150_forward_lstm_150_while_greater_sequential_150_bidirectional_150_forward_lstm_150_castw
dsequential_150_bidirectional_150_forward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource:	Шy
fsequential_150_bidirectional_150_forward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource:	2Шt
esequential_150_bidirectional_150_forward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource:	ШЂ\sequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOpЂ[sequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOpЂ]sequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOpЇ
isequential_150/bidirectional_150/forward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2k
isequential_150/bidirectional_150/forward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shape
[sequential_150/bidirectional_150/forward_lstm_150/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЗsequential_150_bidirectional_150_forward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_sequential_150_bidirectional_150_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0Csequential_150_bidirectional_150_forward_lstm_150_while_placeholderrsequential_150/bidirectional_150/forward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02]
[sequential_150/bidirectional_150/forward_lstm_150/while/TensorArrayV2Read/TensorListGetItemњ
?sequential_150/bidirectional_150/forward_lstm_150/while/GreaterGreaterxsequential_150_bidirectional_150_forward_lstm_150_while_greater_sequential_150_bidirectional_150_forward_lstm_150_cast_0Csequential_150_bidirectional_150_forward_lstm_150_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2A
?sequential_150/bidirectional_150/forward_lstm_150/while/Greaterт
[sequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOpReadVariableOpfsequential_150_bidirectional_150_forward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02]
[sequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOpЂ
Lsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMulMatMulbsequential_150/bidirectional_150/forward_lstm_150/while/TensorArrayV2Read/TensorListGetItem:item:0csequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2N
Lsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMulш
]sequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOphsequential_150_bidirectional_150_forward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02_
]sequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp
Nsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMul_1MatMulEsequential_150_bidirectional_150_forward_lstm_150_while_placeholder_3esequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2P
Nsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMul_1
Isequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/addAddV2Vsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMul:product:0Xsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2K
Isequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/addс
\sequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOpgsequential_150_bidirectional_150_forward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02^
\sequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp
Msequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/BiasAddBiasAddMsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/add:z:0dsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2O
Msequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/BiasAdd№
Usequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2W
Usequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/split/split_dimз
Ksequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/splitSplit^sequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/split/split_dim:output:0Vsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2M
Ksequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/splitБ
Msequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/SigmoidSigmoidTsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22O
Msequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/SigmoidЕ
Osequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/Sigmoid_1SigmoidTsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22Q
Osequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/Sigmoid_1ы
Isequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/mulMulSsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/Sigmoid_1:y:0Esequential_150_bidirectional_150_forward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22K
Isequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/mulЈ
Jsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/ReluReluTsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22L
Jsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/Relu
Ksequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/mul_1MulQsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/Sigmoid:y:0Xsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22M
Ksequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/mul_1ѕ
Ksequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/add_1AddV2Msequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/mul:z:0Osequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22M
Ksequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/add_1Е
Osequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/Sigmoid_2SigmoidTsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22Q
Osequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/Sigmoid_2Ї
Lsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/Relu_1ReluOsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22N
Lsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/Relu_1
Ksequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/mul_2MulSsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/Sigmoid_2:y:0Zsequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22M
Ksequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/mul_2
>sequential_150/bidirectional_150/forward_lstm_150/while/SelectSelectCsequential_150/bidirectional_150/forward_lstm_150/while/Greater:z:0Osequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/mul_2:z:0Esequential_150_bidirectional_150_forward_lstm_150_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22@
>sequential_150/bidirectional_150/forward_lstm_150/while/Select
@sequential_150/bidirectional_150/forward_lstm_150/while/Select_1SelectCsequential_150/bidirectional_150/forward_lstm_150/while/Greater:z:0Osequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/mul_2:z:0Esequential_150_bidirectional_150_forward_lstm_150_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22B
@sequential_150/bidirectional_150/forward_lstm_150/while/Select_1
@sequential_150/bidirectional_150/forward_lstm_150/while/Select_2SelectCsequential_150/bidirectional_150/forward_lstm_150/while/Greater:z:0Osequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/add_1:z:0Esequential_150_bidirectional_150_forward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22B
@sequential_150/bidirectional_150/forward_lstm_150/while/Select_2г
\sequential_150/bidirectional_150/forward_lstm_150/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemEsequential_150_bidirectional_150_forward_lstm_150_while_placeholder_1Csequential_150_bidirectional_150_forward_lstm_150_while_placeholderGsequential_150/bidirectional_150/forward_lstm_150/while/Select:output:0*
_output_shapes
: *
element_dtype02^
\sequential_150/bidirectional_150/forward_lstm_150/while/TensorArrayV2Write/TensorListSetItemР
=sequential_150/bidirectional_150/forward_lstm_150/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2?
=sequential_150/bidirectional_150/forward_lstm_150/while/add/yБ
;sequential_150/bidirectional_150/forward_lstm_150/while/addAddV2Csequential_150_bidirectional_150_forward_lstm_150_while_placeholderFsequential_150/bidirectional_150/forward_lstm_150/while/add/y:output:0*
T0*
_output_shapes
: 2=
;sequential_150/bidirectional_150/forward_lstm_150/while/addФ
?sequential_150/bidirectional_150/forward_lstm_150/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2A
?sequential_150/bidirectional_150/forward_lstm_150/while/add_1/y№
=sequential_150/bidirectional_150/forward_lstm_150/while/add_1AddV2|sequential_150_bidirectional_150_forward_lstm_150_while_sequential_150_bidirectional_150_forward_lstm_150_while_loop_counterHsequential_150/bidirectional_150/forward_lstm_150/while/add_1/y:output:0*
T0*
_output_shapes
: 2?
=sequential_150/bidirectional_150/forward_lstm_150/while/add_1Г
@sequential_150/bidirectional_150/forward_lstm_150/while/IdentityIdentityAsequential_150/bidirectional_150/forward_lstm_150/while/add_1:z:0=^sequential_150/bidirectional_150/forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2B
@sequential_150/bidirectional_150/forward_lstm_150/while/Identityљ
Bsequential_150/bidirectional_150/forward_lstm_150/while/Identity_1Identitysequential_150_bidirectional_150_forward_lstm_150_while_sequential_150_bidirectional_150_forward_lstm_150_while_maximum_iterations=^sequential_150/bidirectional_150/forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2D
Bsequential_150/bidirectional_150/forward_lstm_150/while/Identity_1Е
Bsequential_150/bidirectional_150/forward_lstm_150/while/Identity_2Identity?sequential_150/bidirectional_150/forward_lstm_150/while/add:z:0=^sequential_150/bidirectional_150/forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2D
Bsequential_150/bidirectional_150/forward_lstm_150/while/Identity_2т
Bsequential_150/bidirectional_150/forward_lstm_150/while/Identity_3Identitylsequential_150/bidirectional_150/forward_lstm_150/while/TensorArrayV2Write/TensorListSetItem:output_handle:0=^sequential_150/bidirectional_150/forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2D
Bsequential_150/bidirectional_150/forward_lstm_150/while/Identity_3Ю
Bsequential_150/bidirectional_150/forward_lstm_150/while/Identity_4IdentityGsequential_150/bidirectional_150/forward_lstm_150/while/Select:output:0=^sequential_150/bidirectional_150/forward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22D
Bsequential_150/bidirectional_150/forward_lstm_150/while/Identity_4а
Bsequential_150/bidirectional_150/forward_lstm_150/while/Identity_5IdentityIsequential_150/bidirectional_150/forward_lstm_150/while/Select_1:output:0=^sequential_150/bidirectional_150/forward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22D
Bsequential_150/bidirectional_150/forward_lstm_150/while/Identity_5а
Bsequential_150/bidirectional_150/forward_lstm_150/while/Identity_6IdentityIsequential_150/bidirectional_150/forward_lstm_150/while/Select_2:output:0=^sequential_150/bidirectional_150/forward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22D
Bsequential_150/bidirectional_150/forward_lstm_150/while/Identity_6л
<sequential_150/bidirectional_150/forward_lstm_150/while/NoOpNoOp]^sequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp\^sequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp^^sequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2>
<sequential_150/bidirectional_150/forward_lstm_150/while/NoOp"ђ
vsequential_150_bidirectional_150_forward_lstm_150_while_greater_sequential_150_bidirectional_150_forward_lstm_150_castxsequential_150_bidirectional_150_forward_lstm_150_while_greater_sequential_150_bidirectional_150_forward_lstm_150_cast_0"
@sequential_150_bidirectional_150_forward_lstm_150_while_identityIsequential_150/bidirectional_150/forward_lstm_150/while/Identity:output:0"
Bsequential_150_bidirectional_150_forward_lstm_150_while_identity_1Ksequential_150/bidirectional_150/forward_lstm_150/while/Identity_1:output:0"
Bsequential_150_bidirectional_150_forward_lstm_150_while_identity_2Ksequential_150/bidirectional_150/forward_lstm_150/while/Identity_2:output:0"
Bsequential_150_bidirectional_150_forward_lstm_150_while_identity_3Ksequential_150/bidirectional_150/forward_lstm_150/while/Identity_3:output:0"
Bsequential_150_bidirectional_150_forward_lstm_150_while_identity_4Ksequential_150/bidirectional_150/forward_lstm_150/while/Identity_4:output:0"
Bsequential_150_bidirectional_150_forward_lstm_150_while_identity_5Ksequential_150/bidirectional_150/forward_lstm_150/while/Identity_5:output:0"
Bsequential_150_bidirectional_150_forward_lstm_150_while_identity_6Ksequential_150/bidirectional_150/forward_lstm_150/while/Identity_6:output:0"а
esequential_150_bidirectional_150_forward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resourcegsequential_150_bidirectional_150_forward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0"в
fsequential_150_bidirectional_150_forward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resourcehsequential_150_bidirectional_150_forward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0"Ю
dsequential_150_bidirectional_150_forward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resourcefsequential_150_bidirectional_150_forward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0"ј
ysequential_150_bidirectional_150_forward_lstm_150_while_sequential_150_bidirectional_150_forward_lstm_150_strided_slice_1{sequential_150_bidirectional_150_forward_lstm_150_while_sequential_150_bidirectional_150_forward_lstm_150_strided_slice_1_0"ђ
Еsequential_150_bidirectional_150_forward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_sequential_150_bidirectional_150_forward_lstm_150_tensorarrayunstack_tensorlistfromtensorЗsequential_150_bidirectional_150_forward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_sequential_150_bidirectional_150_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2М
\sequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp\sequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp2К
[sequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp[sequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp2О
]sequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp]sequential_150/bidirectional_150/forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
L__inference_sequential_150_layer_call_and_return_conditional_losses_18420782

inputs
inputs_1	-
bidirectional_150_18420763:	Ш-
bidirectional_150_18420765:	2Ш)
bidirectional_150_18420767:	Ш-
bidirectional_150_18420769:	Ш-
bidirectional_150_18420771:	2Ш)
bidirectional_150_18420773:	Ш$
dense_150_18420776:d 
dense_150_18420778:
identityЂ)bidirectional_150/StatefulPartitionedCallЂ!dense_150/StatefulPartitionedCallЪ
)bidirectional_150/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_150_18420763bidirectional_150_18420765bidirectional_150_18420767bidirectional_150_18420769bidirectional_150_18420771bidirectional_150_18420773*
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
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_184206322+
)bidirectional_150/StatefulPartitionedCallЫ
!dense_150/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_150/StatefulPartitionedCall:output:0dense_150_18420776dense_150_18420778*
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
G__inference_dense_150_layer_call_and_return_conditional_losses_184202172#
!dense_150/StatefulPartitionedCall
IdentityIdentity*dense_150/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp*^bidirectional_150/StatefulPartitionedCall"^dense_150/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2V
)bidirectional_150/StatefulPartitionedCall)bidirectional_150/StatefulPartitionedCall2F
!dense_150/StatefulPartitionedCall!dense_150/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
]
Ќ
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_18422870

inputs?
,lstm_cell_451_matmul_readvariableop_resource:	ШA
.lstm_cell_451_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_451_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_451/BiasAdd/ReadVariableOpЂ#lstm_cell_451/MatMul/ReadVariableOpЂ%lstm_cell_451/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_451/MatMul/ReadVariableOpReadVariableOp,lstm_cell_451_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_451/MatMul/ReadVariableOpА
lstm_cell_451/MatMulMatMulstrided_slice_2:output:0+lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/MatMulО
%lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_451_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_451/MatMul_1/ReadVariableOpЌ
lstm_cell_451/MatMul_1MatMulzeros:output:0-lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/MatMul_1Є
lstm_cell_451/addAddV2lstm_cell_451/MatMul:product:0 lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/addЗ
$lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_451_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_451/BiasAdd/ReadVariableOpБ
lstm_cell_451/BiasAddBiasAddlstm_cell_451/add:z:0,lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/BiasAdd
lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_451/split/split_dimї
lstm_cell_451/splitSplit&lstm_cell_451/split/split_dim:output:0lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_451/split
lstm_cell_451/SigmoidSigmoidlstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Sigmoid
lstm_cell_451/Sigmoid_1Sigmoidlstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Sigmoid_1
lstm_cell_451/mulMullstm_cell_451/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/mul
lstm_cell_451/ReluRelulstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Relu 
lstm_cell_451/mul_1Mullstm_cell_451/Sigmoid:y:0 lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/mul_1
lstm_cell_451/add_1AddV2lstm_cell_451/mul:z:0lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/add_1
lstm_cell_451/Sigmoid_2Sigmoidlstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Sigmoid_2
lstm_cell_451/Relu_1Relulstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Relu_1Є
lstm_cell_451/mul_2Mullstm_cell_451/Sigmoid_2:y:0"lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_451_matmul_readvariableop_resource.lstm_cell_451_matmul_1_readvariableop_resource-lstm_cell_451_biasadd_readvariableop_resource*
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
while_body_18422786*
condR
while_cond_18422785*K
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
NoOpNoOp%^lstm_cell_451/BiasAdd/ReadVariableOp$^lstm_cell_451/MatMul/ReadVariableOp&^lstm_cell_451/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_451/BiasAdd/ReadVariableOp$lstm_cell_451/BiasAdd/ReadVariableOp2J
#lstm_cell_451/MatMul/ReadVariableOp#lstm_cell_451/MatMul/ReadVariableOp2N
%lstm_cell_451/MatMul_1/ReadVariableOp%lstm_cell_451/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Н
я
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_18420632

inputs
inputs_1	P
=forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource:	ШR
?forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource:	2ШM
>forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource:	ШQ
>backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource:	ШS
@backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource:	2ШN
?backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource:	Ш
identityЂ6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpЂ5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpЂ7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpЂbackward_lstm_150/whileЂ5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpЂ4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpЂ6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOpЂforward_lstm_150/while
%forward_lstm_150/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_150/RaggedToTensor/zeros
%forward_lstm_150/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2'
%forward_lstm_150/RaggedToTensor/Const
4forward_lstm_150/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_150/RaggedToTensor/Const:output:0inputs.forward_lstm_150/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_150/RaggedToTensor/RaggedTensorToTensorФ
;forward_lstm_150/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_150/RaggedNestedRowLengths/strided_slice/stackШ
=forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1Ш
=forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2Љ
5forward_lstm_150/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_150/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask27
5forward_lstm_150/RaggedNestedRowLengths/strided_sliceШ
=forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stackе
?forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2A
?forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1Ь
?forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2Е
7forward_lstm_150/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask29
7forward_lstm_150/RaggedNestedRowLengths/strided_slice_1
+forward_lstm_150/RaggedNestedRowLengths/subSub>forward_lstm_150/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_150/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2-
+forward_lstm_150/RaggedNestedRowLengths/subЄ
forward_lstm_150/CastCast/forward_lstm_150/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_150/Cast
forward_lstm_150/ShapeShape=forward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_150/Shape
$forward_lstm_150/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_150/strided_slice/stack
&forward_lstm_150/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_150/strided_slice/stack_1
&forward_lstm_150/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_150/strided_slice/stack_2Ш
forward_lstm_150/strided_sliceStridedSliceforward_lstm_150/Shape:output:0-forward_lstm_150/strided_slice/stack:output:0/forward_lstm_150/strided_slice/stack_1:output:0/forward_lstm_150/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_150/strided_slice~
forward_lstm_150/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_150/zeros/mul/yА
forward_lstm_150/zeros/mulMul'forward_lstm_150/strided_slice:output:0%forward_lstm_150/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros/mul
forward_lstm_150/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_150/zeros/Less/yЋ
forward_lstm_150/zeros/LessLessforward_lstm_150/zeros/mul:z:0&forward_lstm_150/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros/Less
forward_lstm_150/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_150/zeros/packed/1Ч
forward_lstm_150/zeros/packedPack'forward_lstm_150/strided_slice:output:0(forward_lstm_150/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_150/zeros/packed
forward_lstm_150/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_150/zeros/ConstЙ
forward_lstm_150/zerosFill&forward_lstm_150/zeros/packed:output:0%forward_lstm_150/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/zeros
forward_lstm_150/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_150/zeros_1/mul/yЖ
forward_lstm_150/zeros_1/mulMul'forward_lstm_150/strided_slice:output:0'forward_lstm_150/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros_1/mul
forward_lstm_150/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
forward_lstm_150/zeros_1/Less/yГ
forward_lstm_150/zeros_1/LessLess forward_lstm_150/zeros_1/mul:z:0(forward_lstm_150/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros_1/Less
!forward_lstm_150/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_150/zeros_1/packed/1Э
forward_lstm_150/zeros_1/packedPack'forward_lstm_150/strided_slice:output:0*forward_lstm_150/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_150/zeros_1/packed
forward_lstm_150/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_150/zeros_1/ConstС
forward_lstm_150/zeros_1Fill(forward_lstm_150/zeros_1/packed:output:0'forward_lstm_150/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/zeros_1
forward_lstm_150/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_150/transpose/permэ
forward_lstm_150/transpose	Transpose=forward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_150/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_150/transpose
forward_lstm_150/Shape_1Shapeforward_lstm_150/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_150/Shape_1
&forward_lstm_150/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_150/strided_slice_1/stack
(forward_lstm_150/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_1/stack_1
(forward_lstm_150/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_1/stack_2д
 forward_lstm_150/strided_slice_1StridedSlice!forward_lstm_150/Shape_1:output:0/forward_lstm_150/strided_slice_1/stack:output:01forward_lstm_150/strided_slice_1/stack_1:output:01forward_lstm_150/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_150/strided_slice_1Ї
,forward_lstm_150/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,forward_lstm_150/TensorArrayV2/element_shapeі
forward_lstm_150/TensorArrayV2TensorListReserve5forward_lstm_150/TensorArrayV2/element_shape:output:0)forward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_150/TensorArrayV2с
Fforward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fforward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8forward_lstm_150/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_150/transpose:y:0Oforward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_150/TensorArrayUnstack/TensorListFromTensor
&forward_lstm_150/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_150/strided_slice_2/stack
(forward_lstm_150/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_2/stack_1
(forward_lstm_150/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_2/stack_2т
 forward_lstm_150/strided_slice_2StridedSliceforward_lstm_150/transpose:y:0/forward_lstm_150/strided_slice_2/stack:output:01forward_lstm_150/strided_slice_2/stack_1:output:01forward_lstm_150/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 forward_lstm_150/strided_slice_2ы
4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpReadVariableOp=forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpє
%forward_lstm_150/lstm_cell_451/MatMulMatMul)forward_lstm_150/strided_slice_2:output:0<forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_150/lstm_cell_451/MatMulё
6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOp?forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp№
'forward_lstm_150/lstm_cell_451/MatMul_1MatMulforward_lstm_150/zeros:output:0>forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_150/lstm_cell_451/MatMul_1ш
"forward_lstm_150/lstm_cell_451/addAddV2/forward_lstm_150/lstm_cell_451/MatMul:product:01forward_lstm_150/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_150/lstm_cell_451/addъ
5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOp>forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpѕ
&forward_lstm_150/lstm_cell_451/BiasAddBiasAdd&forward_lstm_150/lstm_cell_451/add:z:0=forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_150/lstm_cell_451/BiasAddЂ
.forward_lstm_150/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.forward_lstm_150/lstm_cell_451/split/split_dimЛ
$forward_lstm_150/lstm_cell_451/splitSplit7forward_lstm_150/lstm_cell_451/split/split_dim:output:0/forward_lstm_150/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$forward_lstm_150/lstm_cell_451/splitМ
&forward_lstm_150/lstm_cell_451/SigmoidSigmoid-forward_lstm_150/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_150/lstm_cell_451/SigmoidР
(forward_lstm_150/lstm_cell_451/Sigmoid_1Sigmoid-forward_lstm_150/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_150/lstm_cell_451/Sigmoid_1в
"forward_lstm_150/lstm_cell_451/mulMul,forward_lstm_150/lstm_cell_451/Sigmoid_1:y:0!forward_lstm_150/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_150/lstm_cell_451/mulГ
#forward_lstm_150/lstm_cell_451/ReluRelu-forward_lstm_150/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_150/lstm_cell_451/Reluф
$forward_lstm_150/lstm_cell_451/mul_1Mul*forward_lstm_150/lstm_cell_451/Sigmoid:y:01forward_lstm_150/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_150/lstm_cell_451/mul_1й
$forward_lstm_150/lstm_cell_451/add_1AddV2&forward_lstm_150/lstm_cell_451/mul:z:0(forward_lstm_150/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_150/lstm_cell_451/add_1Р
(forward_lstm_150/lstm_cell_451/Sigmoid_2Sigmoid-forward_lstm_150/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_150/lstm_cell_451/Sigmoid_2В
%forward_lstm_150/lstm_cell_451/Relu_1Relu(forward_lstm_150/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_150/lstm_cell_451/Relu_1ш
$forward_lstm_150/lstm_cell_451/mul_2Mul,forward_lstm_150/lstm_cell_451/Sigmoid_2:y:03forward_lstm_150/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_150/lstm_cell_451/mul_2Б
.forward_lstm_150/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.forward_lstm_150/TensorArrayV2_1/element_shapeќ
 forward_lstm_150/TensorArrayV2_1TensorListReserve7forward_lstm_150/TensorArrayV2_1/element_shape:output:0)forward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_150/TensorArrayV2_1p
forward_lstm_150/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_150/timeЃ
forward_lstm_150/zeros_like	ZerosLike(forward_lstm_150/lstm_cell_451/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/zeros_likeЁ
)forward_lstm_150/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)forward_lstm_150/while/maximum_iterations
#forward_lstm_150/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_150/while/loop_counter	
forward_lstm_150/whileWhile,forward_lstm_150/while/loop_counter:output:02forward_lstm_150/while/maximum_iterations:output:0forward_lstm_150/time:output:0)forward_lstm_150/TensorArrayV2_1:handle:0forward_lstm_150/zeros_like:y:0forward_lstm_150/zeros:output:0!forward_lstm_150/zeros_1:output:0)forward_lstm_150/strided_slice_1:output:0Hforward_lstm_150/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_150/Cast:y:0=forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource?forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource>forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource*
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
$forward_lstm_150_while_body_18420356*0
cond(R&
$forward_lstm_150_while_cond_18420355*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_150/whileз
Aforward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Aforward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3forward_lstm_150/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_150/while:output:3Jforward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3forward_lstm_150/TensorArrayV2Stack/TensorListStackЃ
&forward_lstm_150/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&forward_lstm_150/strided_slice_3/stack
(forward_lstm_150/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_150/strided_slice_3/stack_1
(forward_lstm_150/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_3/stack_2
 forward_lstm_150/strided_slice_3StridedSlice<forward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_150/strided_slice_3/stack:output:01forward_lstm_150/strided_slice_3/stack_1:output:01forward_lstm_150/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 forward_lstm_150/strided_slice_3
!forward_lstm_150/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_150/transpose_1/permђ
forward_lstm_150/transpose_1	Transpose<forward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_150/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_150/transpose_1
forward_lstm_150/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_150/runtime
&backward_lstm_150/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_150/RaggedToTensor/zeros
&backward_lstm_150/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2(
&backward_lstm_150/RaggedToTensor/Const
5backward_lstm_150/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_150/RaggedToTensor/Const:output:0inputs/backward_lstm_150/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_150/RaggedToTensor/RaggedTensorToTensorЦ
<backward_lstm_150/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_150/RaggedNestedRowLengths/strided_slice/stackЪ
>backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1Ъ
>backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2Ў
6backward_lstm_150/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_150/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask28
6backward_lstm_150/RaggedNestedRowLengths/strided_sliceЪ
>backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stackз
@backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2B
@backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1Ю
@backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2К
8backward_lstm_150/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2:
8backward_lstm_150/RaggedNestedRowLengths/strided_slice_1
,backward_lstm_150/RaggedNestedRowLengths/subSub?backward_lstm_150/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_150/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2.
,backward_lstm_150/RaggedNestedRowLengths/subЇ
backward_lstm_150/CastCast0backward_lstm_150/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_150/Cast 
backward_lstm_150/ShapeShape>backward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_150/Shape
%backward_lstm_150/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_150/strided_slice/stack
'backward_lstm_150/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_150/strided_slice/stack_1
'backward_lstm_150/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_150/strided_slice/stack_2Ю
backward_lstm_150/strided_sliceStridedSlice backward_lstm_150/Shape:output:0.backward_lstm_150/strided_slice/stack:output:00backward_lstm_150/strided_slice/stack_1:output:00backward_lstm_150/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_150/strided_slice
backward_lstm_150/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_150/zeros/mul/yД
backward_lstm_150/zeros/mulMul(backward_lstm_150/strided_slice:output:0&backward_lstm_150/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/zeros/mul
backward_lstm_150/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_150/zeros/Less/yЏ
backward_lstm_150/zeros/LessLessbackward_lstm_150/zeros/mul:z:0'backward_lstm_150/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/zeros/Less
 backward_lstm_150/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_150/zeros/packed/1Ы
backward_lstm_150/zeros/packedPack(backward_lstm_150/strided_slice:output:0)backward_lstm_150/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_150/zeros/packed
backward_lstm_150/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_150/zeros/ConstН
backward_lstm_150/zerosFill'backward_lstm_150/zeros/packed:output:0&backward_lstm_150/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_150/zeros
backward_lstm_150/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_150/zeros_1/mul/yК
backward_lstm_150/zeros_1/mulMul(backward_lstm_150/strided_slice:output:0(backward_lstm_150/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/zeros_1/mul
 backward_lstm_150/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 backward_lstm_150/zeros_1/Less/yЗ
backward_lstm_150/zeros_1/LessLess!backward_lstm_150/zeros_1/mul:z:0)backward_lstm_150/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_150/zeros_1/Less
"backward_lstm_150/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_150/zeros_1/packed/1б
 backward_lstm_150/zeros_1/packedPack(backward_lstm_150/strided_slice:output:0+backward_lstm_150/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_150/zeros_1/packed
backward_lstm_150/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_150/zeros_1/ConstХ
backward_lstm_150/zeros_1Fill)backward_lstm_150/zeros_1/packed:output:0(backward_lstm_150/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_150/zeros_1
 backward_lstm_150/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_150/transpose/permё
backward_lstm_150/transpose	Transpose>backward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_150/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_150/transpose
backward_lstm_150/Shape_1Shapebackward_lstm_150/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_150/Shape_1
'backward_lstm_150/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_150/strided_slice_1/stack 
)backward_lstm_150/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_1/stack_1 
)backward_lstm_150/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_1/stack_2к
!backward_lstm_150/strided_slice_1StridedSlice"backward_lstm_150/Shape_1:output:00backward_lstm_150/strided_slice_1/stack:output:02backward_lstm_150/strided_slice_1/stack_1:output:02backward_lstm_150/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_150/strided_slice_1Љ
-backward_lstm_150/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2/
-backward_lstm_150/TensorArrayV2/element_shapeњ
backward_lstm_150/TensorArrayV2TensorListReserve6backward_lstm_150/TensorArrayV2/element_shape:output:0*backward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_150/TensorArrayV2
 backward_lstm_150/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_150/ReverseV2/axisв
backward_lstm_150/ReverseV2	ReverseV2backward_lstm_150/transpose:y:0)backward_lstm_150/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_150/ReverseV2у
Gbackward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gbackward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeХ
9backward_lstm_150/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_150/ReverseV2:output:0Pbackward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_150/TensorArrayUnstack/TensorListFromTensor
'backward_lstm_150/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_150/strided_slice_2/stack 
)backward_lstm_150/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_2/stack_1 
)backward_lstm_150/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_2/stack_2ш
!backward_lstm_150/strided_slice_2StridedSlicebackward_lstm_150/transpose:y:00backward_lstm_150/strided_slice_2/stack:output:02backward_lstm_150/strided_slice_2/stack_1:output:02backward_lstm_150/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2#
!backward_lstm_150/strided_slice_2ю
5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpReadVariableOp>backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype027
5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpј
&backward_lstm_150/lstm_cell_452/MatMulMatMul*backward_lstm_150/strided_slice_2:output:0=backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_150/lstm_cell_452/MatMulє
7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOp@backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype029
7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpє
(backward_lstm_150/lstm_cell_452/MatMul_1MatMul backward_lstm_150/zeros:output:0?backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_150/lstm_cell_452/MatMul_1ь
#backward_lstm_150/lstm_cell_452/addAddV20backward_lstm_150/lstm_cell_452/MatMul:product:02backward_lstm_150/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_150/lstm_cell_452/addэ
6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOp?backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype028
6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpљ
'backward_lstm_150/lstm_cell_452/BiasAddBiasAdd'backward_lstm_150/lstm_cell_452/add:z:0>backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_150/lstm_cell_452/BiasAddЄ
/backward_lstm_150/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/backward_lstm_150/lstm_cell_452/split/split_dimП
%backward_lstm_150/lstm_cell_452/splitSplit8backward_lstm_150/lstm_cell_452/split/split_dim:output:00backward_lstm_150/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2'
%backward_lstm_150/lstm_cell_452/splitП
'backward_lstm_150/lstm_cell_452/SigmoidSigmoid.backward_lstm_150/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_150/lstm_cell_452/SigmoidУ
)backward_lstm_150/lstm_cell_452/Sigmoid_1Sigmoid.backward_lstm_150/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_150/lstm_cell_452/Sigmoid_1ж
#backward_lstm_150/lstm_cell_452/mulMul-backward_lstm_150/lstm_cell_452/Sigmoid_1:y:0"backward_lstm_150/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_150/lstm_cell_452/mulЖ
$backward_lstm_150/lstm_cell_452/ReluRelu.backward_lstm_150/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_150/lstm_cell_452/Reluш
%backward_lstm_150/lstm_cell_452/mul_1Mul+backward_lstm_150/lstm_cell_452/Sigmoid:y:02backward_lstm_150/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_150/lstm_cell_452/mul_1н
%backward_lstm_150/lstm_cell_452/add_1AddV2'backward_lstm_150/lstm_cell_452/mul:z:0)backward_lstm_150/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_150/lstm_cell_452/add_1У
)backward_lstm_150/lstm_cell_452/Sigmoid_2Sigmoid.backward_lstm_150/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_150/lstm_cell_452/Sigmoid_2Е
&backward_lstm_150/lstm_cell_452/Relu_1Relu)backward_lstm_150/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_150/lstm_cell_452/Relu_1ь
%backward_lstm_150/lstm_cell_452/mul_2Mul-backward_lstm_150/lstm_cell_452/Sigmoid_2:y:04backward_lstm_150/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_150/lstm_cell_452/mul_2Г
/backward_lstm_150/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   21
/backward_lstm_150/TensorArrayV2_1/element_shape
!backward_lstm_150/TensorArrayV2_1TensorListReserve8backward_lstm_150/TensorArrayV2_1/element_shape:output:0*backward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_150/TensorArrayV2_1r
backward_lstm_150/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_150/time
'backward_lstm_150/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_150/Max/reduction_indicesЄ
backward_lstm_150/MaxMaxbackward_lstm_150/Cast:y:00backward_lstm_150/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/Maxt
backward_lstm_150/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_150/sub/y
backward_lstm_150/subSubbackward_lstm_150/Max:output:0 backward_lstm_150/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/sub
backward_lstm_150/Sub_1Subbackward_lstm_150/sub:z:0backward_lstm_150/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_150/Sub_1І
backward_lstm_150/zeros_like	ZerosLike)backward_lstm_150/lstm_cell_452/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_150/zeros_likeЃ
*backward_lstm_150/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*backward_lstm_150/while/maximum_iterations
$backward_lstm_150/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_150/while/loop_counterЅ	
backward_lstm_150/whileWhile-backward_lstm_150/while/loop_counter:output:03backward_lstm_150/while/maximum_iterations:output:0backward_lstm_150/time:output:0*backward_lstm_150/TensorArrayV2_1:handle:0 backward_lstm_150/zeros_like:y:0 backward_lstm_150/zeros:output:0"backward_lstm_150/zeros_1:output:0*backward_lstm_150/strided_slice_1:output:0Ibackward_lstm_150/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_150/Sub_1:z:0>backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource@backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource?backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource*
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
%backward_lstm_150_while_body_18420535*1
cond)R'
%backward_lstm_150_while_cond_18420534*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_150/whileй
Bbackward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2D
Bbackward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeЙ
4backward_lstm_150/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_150/while:output:3Kbackward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype026
4backward_lstm_150/TensorArrayV2Stack/TensorListStackЅ
'backward_lstm_150/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2)
'backward_lstm_150/strided_slice_3/stack 
)backward_lstm_150/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_150/strided_slice_3/stack_1 
)backward_lstm_150/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_3/stack_2
!backward_lstm_150/strided_slice_3StridedSlice=backward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_150/strided_slice_3/stack:output:02backward_lstm_150/strided_slice_3/stack_1:output:02backward_lstm_150/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2#
!backward_lstm_150/strided_slice_3
"backward_lstm_150/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_150/transpose_1/permі
backward_lstm_150/transpose_1	Transpose=backward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_150/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_150/transpose_1
backward_lstm_150/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_150/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisФ
concatConcatV2)forward_lstm_150/strided_slice_3:output:0*backward_lstm_150/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp7^backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp6^backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp8^backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp^backward_lstm_150/while6^forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp5^forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp7^forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp^forward_lstm_150/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2p
6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp2n
5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp2r
7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp22
backward_lstm_150/whilebackward_lstm_150/while2n
5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp2l
4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp2p
6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp20
forward_lstm_150/whileforward_lstm_150/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
п
Э
while_cond_18422982
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_18422982___redundant_placeholder06
2while_while_cond_18422982___redundant_placeholder16
2while_while_cond_18422982___redundant_placeholder26
2while_while_cond_18422982___redundant_placeholder3
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
e
Т
$forward_lstm_150_while_body_18421568>
:forward_lstm_150_while_forward_lstm_150_while_loop_counterD
@forward_lstm_150_while_forward_lstm_150_while_maximum_iterations&
"forward_lstm_150_while_placeholder(
$forward_lstm_150_while_placeholder_1(
$forward_lstm_150_while_placeholder_2(
$forward_lstm_150_while_placeholder_3(
$forward_lstm_150_while_placeholder_4=
9forward_lstm_150_while_forward_lstm_150_strided_slice_1_0y
uforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_150_while_greater_forward_lstm_150_cast_0X
Eforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0:	ШZ
Gforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0:	2ШU
Fforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0:	Ш#
forward_lstm_150_while_identity%
!forward_lstm_150_while_identity_1%
!forward_lstm_150_while_identity_2%
!forward_lstm_150_while_identity_3%
!forward_lstm_150_while_identity_4%
!forward_lstm_150_while_identity_5%
!forward_lstm_150_while_identity_6;
7forward_lstm_150_while_forward_lstm_150_strided_slice_1w
sforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_150_while_greater_forward_lstm_150_castV
Cforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource:	ШX
Eforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource:	2ШS
Dforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource:	ШЂ;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOpЂ:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOpЂ<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOpх
Hforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2J
Hforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
:forward_lstm_150/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_150_while_placeholderQforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02<
:forward_lstm_150/while/TensorArrayV2Read/TensorListGetItemе
forward_lstm_150/while/GreaterGreater6forward_lstm_150_while_greater_forward_lstm_150_cast_0"forward_lstm_150_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2 
forward_lstm_150/while/Greaterџ
:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOpReadVariableOpEforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp
+forward_lstm_150/while/lstm_cell_451/MatMulMatMulAforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem:item:0Bforward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_150/while/lstm_cell_451/MatMul
<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOpGforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp
-forward_lstm_150/while/lstm_cell_451/MatMul_1MatMul$forward_lstm_150_while_placeholder_3Dforward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-forward_lstm_150/while/lstm_cell_451/MatMul_1
(forward_lstm_150/while/lstm_cell_451/addAddV25forward_lstm_150/while/lstm_cell_451/MatMul:product:07forward_lstm_150/while/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_150/while/lstm_cell_451/addў
;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOpFforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp
,forward_lstm_150/while/lstm_cell_451/BiasAddBiasAdd,forward_lstm_150/while/lstm_cell_451/add:z:0Cforward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_150/while/lstm_cell_451/BiasAddЎ
4forward_lstm_150/while/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4forward_lstm_150/while/lstm_cell_451/split/split_dimг
*forward_lstm_150/while/lstm_cell_451/splitSplit=forward_lstm_150/while/lstm_cell_451/split/split_dim:output:05forward_lstm_150/while/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*forward_lstm_150/while/lstm_cell_451/splitЮ
,forward_lstm_150/while/lstm_cell_451/SigmoidSigmoid3forward_lstm_150/while/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,forward_lstm_150/while/lstm_cell_451/Sigmoidв
.forward_lstm_150/while/lstm_cell_451/Sigmoid_1Sigmoid3forward_lstm_150/while/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_150/while/lstm_cell_451/Sigmoid_1ч
(forward_lstm_150/while/lstm_cell_451/mulMul2forward_lstm_150/while/lstm_cell_451/Sigmoid_1:y:0$forward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_150/while/lstm_cell_451/mulХ
)forward_lstm_150/while/lstm_cell_451/ReluRelu3forward_lstm_150/while/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_150/while/lstm_cell_451/Reluќ
*forward_lstm_150/while/lstm_cell_451/mul_1Mul0forward_lstm_150/while/lstm_cell_451/Sigmoid:y:07forward_lstm_150/while/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_150/while/lstm_cell_451/mul_1ё
*forward_lstm_150/while/lstm_cell_451/add_1AddV2,forward_lstm_150/while/lstm_cell_451/mul:z:0.forward_lstm_150/while/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_150/while/lstm_cell_451/add_1в
.forward_lstm_150/while/lstm_cell_451/Sigmoid_2Sigmoid3forward_lstm_150/while/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_150/while/lstm_cell_451/Sigmoid_2Ф
+forward_lstm_150/while/lstm_cell_451/Relu_1Relu.forward_lstm_150/while/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_150/while/lstm_cell_451/Relu_1
*forward_lstm_150/while/lstm_cell_451/mul_2Mul2forward_lstm_150/while/lstm_cell_451/Sigmoid_2:y:09forward_lstm_150/while/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_150/while/lstm_cell_451/mul_2є
forward_lstm_150/while/SelectSelect"forward_lstm_150/while/Greater:z:0.forward_lstm_150/while/lstm_cell_451/mul_2:z:0$forward_lstm_150_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/while/Selectј
forward_lstm_150/while/Select_1Select"forward_lstm_150/while/Greater:z:0.forward_lstm_150/while/lstm_cell_451/mul_2:z:0$forward_lstm_150_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_150/while/Select_1ј
forward_lstm_150/while/Select_2Select"forward_lstm_150/while/Greater:z:0.forward_lstm_150/while/lstm_cell_451/add_1:z:0$forward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_150/while/Select_2Ў
;forward_lstm_150/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_150_while_placeholder_1"forward_lstm_150_while_placeholder&forward_lstm_150/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_150/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_150/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_150/while/add/y­
forward_lstm_150/while/addAddV2"forward_lstm_150_while_placeholder%forward_lstm_150/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/while/add
forward_lstm_150/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_150/while/add_1/yЫ
forward_lstm_150/while/add_1AddV2:forward_lstm_150_while_forward_lstm_150_while_loop_counter'forward_lstm_150/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/while/add_1Џ
forward_lstm_150/while/IdentityIdentity forward_lstm_150/while/add_1:z:0^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_150/while/Identityг
!forward_lstm_150/while/Identity_1Identity@forward_lstm_150_while_forward_lstm_150_while_maximum_iterations^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_150/while/Identity_1Б
!forward_lstm_150/while/Identity_2Identityforward_lstm_150/while/add:z:0^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_150/while/Identity_2о
!forward_lstm_150/while/Identity_3IdentityKforward_lstm_150/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_150/while/Identity_3Ъ
!forward_lstm_150/while/Identity_4Identity&forward_lstm_150/while/Select:output:0^forward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_150/while/Identity_4Ь
!forward_lstm_150/while/Identity_5Identity(forward_lstm_150/while/Select_1:output:0^forward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_150/while/Identity_5Ь
!forward_lstm_150/while/Identity_6Identity(forward_lstm_150/while/Select_2:output:0^forward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_150/while/Identity_6Ж
forward_lstm_150/while/NoOpNoOp<^forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp;^forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp=^forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_150/while/NoOp"t
7forward_lstm_150_while_forward_lstm_150_strided_slice_19forward_lstm_150_while_forward_lstm_150_strided_slice_1_0"n
4forward_lstm_150_while_greater_forward_lstm_150_cast6forward_lstm_150_while_greater_forward_lstm_150_cast_0"K
forward_lstm_150_while_identity(forward_lstm_150/while/Identity:output:0"O
!forward_lstm_150_while_identity_1*forward_lstm_150/while/Identity_1:output:0"O
!forward_lstm_150_while_identity_2*forward_lstm_150/while/Identity_2:output:0"O
!forward_lstm_150_while_identity_3*forward_lstm_150/while/Identity_3:output:0"O
!forward_lstm_150_while_identity_4*forward_lstm_150/while/Identity_4:output:0"O
!forward_lstm_150_while_identity_5*forward_lstm_150/while/Identity_5:output:0"O
!forward_lstm_150_while_identity_6*forward_lstm_150/while/Identity_6:output:0"
Dforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resourceFforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0"
Eforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resourceGforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0"
Cforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resourceEforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0"ь
sforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensoruforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2z
;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp2x
:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp2|
<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_18422417
inputs_0?
,lstm_cell_451_matmul_readvariableop_resource:	ШA
.lstm_cell_451_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_451_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_451/BiasAdd/ReadVariableOpЂ#lstm_cell_451/MatMul/ReadVariableOpЂ%lstm_cell_451/MatMul_1/ReadVariableOpЂwhileF
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
#lstm_cell_451/MatMul/ReadVariableOpReadVariableOp,lstm_cell_451_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_451/MatMul/ReadVariableOpА
lstm_cell_451/MatMulMatMulstrided_slice_2:output:0+lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/MatMulО
%lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_451_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_451/MatMul_1/ReadVariableOpЌ
lstm_cell_451/MatMul_1MatMulzeros:output:0-lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/MatMul_1Є
lstm_cell_451/addAddV2lstm_cell_451/MatMul:product:0 lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/addЗ
$lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_451_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_451/BiasAdd/ReadVariableOpБ
lstm_cell_451/BiasAddBiasAddlstm_cell_451/add:z:0,lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/BiasAdd
lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_451/split/split_dimї
lstm_cell_451/splitSplit&lstm_cell_451/split/split_dim:output:0lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_451/split
lstm_cell_451/SigmoidSigmoidlstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Sigmoid
lstm_cell_451/Sigmoid_1Sigmoidlstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Sigmoid_1
lstm_cell_451/mulMullstm_cell_451/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/mul
lstm_cell_451/ReluRelulstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Relu 
lstm_cell_451/mul_1Mullstm_cell_451/Sigmoid:y:0 lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/mul_1
lstm_cell_451/add_1AddV2lstm_cell_451/mul:z:0lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/add_1
lstm_cell_451/Sigmoid_2Sigmoidlstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Sigmoid_2
lstm_cell_451/Relu_1Relulstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Relu_1Є
lstm_cell_451/mul_2Mullstm_cell_451/Sigmoid_2:y:0"lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_451_matmul_readvariableop_resource.lstm_cell_451_matmul_1_readvariableop_resource-lstm_cell_451_biasadd_readvariableop_resource*
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
while_body_18422333*
condR
while_cond_18422332*K
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
NoOpNoOp%^lstm_cell_451/BiasAdd/ReadVariableOp$^lstm_cell_451/MatMul/ReadVariableOp&^lstm_cell_451/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_451/BiasAdd/ReadVariableOp$lstm_cell_451/BiasAdd/ReadVariableOp2J
#lstm_cell_451/MatMul/ReadVariableOp#lstm_cell_451/MatMul/ReadVariableOp2N
%lstm_cell_451/MatMul_1/ReadVariableOp%lstm_cell_451/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
e
Т
$forward_lstm_150_while_body_18421926>
:forward_lstm_150_while_forward_lstm_150_while_loop_counterD
@forward_lstm_150_while_forward_lstm_150_while_maximum_iterations&
"forward_lstm_150_while_placeholder(
$forward_lstm_150_while_placeholder_1(
$forward_lstm_150_while_placeholder_2(
$forward_lstm_150_while_placeholder_3(
$forward_lstm_150_while_placeholder_4=
9forward_lstm_150_while_forward_lstm_150_strided_slice_1_0y
uforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_150_while_greater_forward_lstm_150_cast_0X
Eforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0:	ШZ
Gforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0:	2ШU
Fforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0:	Ш#
forward_lstm_150_while_identity%
!forward_lstm_150_while_identity_1%
!forward_lstm_150_while_identity_2%
!forward_lstm_150_while_identity_3%
!forward_lstm_150_while_identity_4%
!forward_lstm_150_while_identity_5%
!forward_lstm_150_while_identity_6;
7forward_lstm_150_while_forward_lstm_150_strided_slice_1w
sforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_150_while_greater_forward_lstm_150_castV
Cforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource:	ШX
Eforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource:	2ШS
Dforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource:	ШЂ;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOpЂ:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOpЂ<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOpх
Hforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2J
Hforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
:forward_lstm_150/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_150_while_placeholderQforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02<
:forward_lstm_150/while/TensorArrayV2Read/TensorListGetItemе
forward_lstm_150/while/GreaterGreater6forward_lstm_150_while_greater_forward_lstm_150_cast_0"forward_lstm_150_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2 
forward_lstm_150/while/Greaterџ
:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOpReadVariableOpEforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp
+forward_lstm_150/while/lstm_cell_451/MatMulMatMulAforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem:item:0Bforward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_150/while/lstm_cell_451/MatMul
<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOpGforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp
-forward_lstm_150/while/lstm_cell_451/MatMul_1MatMul$forward_lstm_150_while_placeholder_3Dforward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-forward_lstm_150/while/lstm_cell_451/MatMul_1
(forward_lstm_150/while/lstm_cell_451/addAddV25forward_lstm_150/while/lstm_cell_451/MatMul:product:07forward_lstm_150/while/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_150/while/lstm_cell_451/addў
;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOpFforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp
,forward_lstm_150/while/lstm_cell_451/BiasAddBiasAdd,forward_lstm_150/while/lstm_cell_451/add:z:0Cforward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_150/while/lstm_cell_451/BiasAddЎ
4forward_lstm_150/while/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4forward_lstm_150/while/lstm_cell_451/split/split_dimг
*forward_lstm_150/while/lstm_cell_451/splitSplit=forward_lstm_150/while/lstm_cell_451/split/split_dim:output:05forward_lstm_150/while/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*forward_lstm_150/while/lstm_cell_451/splitЮ
,forward_lstm_150/while/lstm_cell_451/SigmoidSigmoid3forward_lstm_150/while/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,forward_lstm_150/while/lstm_cell_451/Sigmoidв
.forward_lstm_150/while/lstm_cell_451/Sigmoid_1Sigmoid3forward_lstm_150/while/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_150/while/lstm_cell_451/Sigmoid_1ч
(forward_lstm_150/while/lstm_cell_451/mulMul2forward_lstm_150/while/lstm_cell_451/Sigmoid_1:y:0$forward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_150/while/lstm_cell_451/mulХ
)forward_lstm_150/while/lstm_cell_451/ReluRelu3forward_lstm_150/while/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_150/while/lstm_cell_451/Reluќ
*forward_lstm_150/while/lstm_cell_451/mul_1Mul0forward_lstm_150/while/lstm_cell_451/Sigmoid:y:07forward_lstm_150/while/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_150/while/lstm_cell_451/mul_1ё
*forward_lstm_150/while/lstm_cell_451/add_1AddV2,forward_lstm_150/while/lstm_cell_451/mul:z:0.forward_lstm_150/while/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_150/while/lstm_cell_451/add_1в
.forward_lstm_150/while/lstm_cell_451/Sigmoid_2Sigmoid3forward_lstm_150/while/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_150/while/lstm_cell_451/Sigmoid_2Ф
+forward_lstm_150/while/lstm_cell_451/Relu_1Relu.forward_lstm_150/while/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_150/while/lstm_cell_451/Relu_1
*forward_lstm_150/while/lstm_cell_451/mul_2Mul2forward_lstm_150/while/lstm_cell_451/Sigmoid_2:y:09forward_lstm_150/while/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_150/while/lstm_cell_451/mul_2є
forward_lstm_150/while/SelectSelect"forward_lstm_150/while/Greater:z:0.forward_lstm_150/while/lstm_cell_451/mul_2:z:0$forward_lstm_150_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/while/Selectј
forward_lstm_150/while/Select_1Select"forward_lstm_150/while/Greater:z:0.forward_lstm_150/while/lstm_cell_451/mul_2:z:0$forward_lstm_150_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_150/while/Select_1ј
forward_lstm_150/while/Select_2Select"forward_lstm_150/while/Greater:z:0.forward_lstm_150/while/lstm_cell_451/add_1:z:0$forward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_150/while/Select_2Ў
;forward_lstm_150/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_150_while_placeholder_1"forward_lstm_150_while_placeholder&forward_lstm_150/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_150/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_150/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_150/while/add/y­
forward_lstm_150/while/addAddV2"forward_lstm_150_while_placeholder%forward_lstm_150/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/while/add
forward_lstm_150/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_150/while/add_1/yЫ
forward_lstm_150/while/add_1AddV2:forward_lstm_150_while_forward_lstm_150_while_loop_counter'forward_lstm_150/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/while/add_1Џ
forward_lstm_150/while/IdentityIdentity forward_lstm_150/while/add_1:z:0^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_150/while/Identityг
!forward_lstm_150/while/Identity_1Identity@forward_lstm_150_while_forward_lstm_150_while_maximum_iterations^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_150/while/Identity_1Б
!forward_lstm_150/while/Identity_2Identityforward_lstm_150/while/add:z:0^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_150/while/Identity_2о
!forward_lstm_150/while/Identity_3IdentityKforward_lstm_150/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_150/while/Identity_3Ъ
!forward_lstm_150/while/Identity_4Identity&forward_lstm_150/while/Select:output:0^forward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_150/while/Identity_4Ь
!forward_lstm_150/while/Identity_5Identity(forward_lstm_150/while/Select_1:output:0^forward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_150/while/Identity_5Ь
!forward_lstm_150/while/Identity_6Identity(forward_lstm_150/while/Select_2:output:0^forward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_150/while/Identity_6Ж
forward_lstm_150/while/NoOpNoOp<^forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp;^forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp=^forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_150/while/NoOp"t
7forward_lstm_150_while_forward_lstm_150_strided_slice_19forward_lstm_150_while_forward_lstm_150_strided_slice_1_0"n
4forward_lstm_150_while_greater_forward_lstm_150_cast6forward_lstm_150_while_greater_forward_lstm_150_cast_0"K
forward_lstm_150_while_identity(forward_lstm_150/while/Identity:output:0"O
!forward_lstm_150_while_identity_1*forward_lstm_150/while/Identity_1:output:0"O
!forward_lstm_150_while_identity_2*forward_lstm_150/while/Identity_2:output:0"O
!forward_lstm_150_while_identity_3*forward_lstm_150/while/Identity_3:output:0"O
!forward_lstm_150_while_identity_4*forward_lstm_150/while/Identity_4:output:0"O
!forward_lstm_150_while_identity_5*forward_lstm_150/while/Identity_5:output:0"O
!forward_lstm_150_while_identity_6*forward_lstm_150/while/Identity_6:output:0"
Dforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resourceFforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0"
Eforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resourceGforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0"
Cforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resourceEforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0"ь
sforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensoruforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2z
;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp2x
:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp2|
<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
L__inference_sequential_150_layer_call_and_return_conditional_losses_18420695

inputs
inputs_1	-
bidirectional_150_18420676:	Ш-
bidirectional_150_18420678:	2Ш)
bidirectional_150_18420680:	Ш-
bidirectional_150_18420682:	Ш-
bidirectional_150_18420684:	2Ш)
bidirectional_150_18420686:	Ш$
dense_150_18420689:d 
dense_150_18420691:
identityЂ)bidirectional_150/StatefulPartitionedCallЂ!dense_150/StatefulPartitionedCallЪ
)bidirectional_150/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_150_18420676bidirectional_150_18420678bidirectional_150_18420680bidirectional_150_18420682bidirectional_150_18420684bidirectional_150_18420686*
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
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_184206322+
)bidirectional_150/StatefulPartitionedCallЫ
!dense_150/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_150/StatefulPartitionedCall:output:0dense_150_18420689dense_150_18420691*
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
G__inference_dense_150_layer_call_and_return_conditional_losses_184202172#
!dense_150/StatefulPartitionedCall
IdentityIdentity*dense_150/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp*^bidirectional_150/StatefulPartitionedCall"^dense_150/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2V
)bidirectional_150/StatefulPartitionedCall)bidirectional_150/StatefulPartitionedCall2F
!dense_150/StatefulPartitionedCall!dense_150/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
п
Э
while_cond_18422483
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_18422483___redundant_placeholder06
2while_while_cond_18422483___redundant_placeholder16
2while_while_cond_18422483___redundant_placeholder26
2while_while_cond_18422483___redundant_placeholder3
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
п
Ё
$forward_lstm_150_while_cond_18420948>
:forward_lstm_150_while_forward_lstm_150_while_loop_counterD
@forward_lstm_150_while_forward_lstm_150_while_maximum_iterations&
"forward_lstm_150_while_placeholder(
$forward_lstm_150_while_placeholder_1(
$forward_lstm_150_while_placeholder_2(
$forward_lstm_150_while_placeholder_3@
<forward_lstm_150_while_less_forward_lstm_150_strided_slice_1X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18420948___redundant_placeholder0X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18420948___redundant_placeholder1X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18420948___redundant_placeholder2X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18420948___redundant_placeholder3#
forward_lstm_150_while_identity
Х
forward_lstm_150/while/LessLess"forward_lstm_150_while_placeholder<forward_lstm_150_while_less_forward_lstm_150_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_150/while/Less
forward_lstm_150/while/IdentityIdentityforward_lstm_150/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_150/while/Identity"K
forward_lstm_150_while_identity(forward_lstm_150/while/Identity:output:0*(
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
ъ
М
%backward_lstm_150_while_cond_18420094@
<backward_lstm_150_while_backward_lstm_150_while_loop_counterF
Bbackward_lstm_150_while_backward_lstm_150_while_maximum_iterations'
#backward_lstm_150_while_placeholder)
%backward_lstm_150_while_placeholder_1)
%backward_lstm_150_while_placeholder_2)
%backward_lstm_150_while_placeholder_3)
%backward_lstm_150_while_placeholder_4B
>backward_lstm_150_while_less_backward_lstm_150_strided_slice_1Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18420094___redundant_placeholder0Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18420094___redundant_placeholder1Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18420094___redundant_placeholder2Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18420094___redundant_placeholder3Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18420094___redundant_placeholder4$
 backward_lstm_150_while_identity
Ъ
backward_lstm_150/while/LessLess#backward_lstm_150_while_placeholder>backward_lstm_150_while_less_backward_lstm_150_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_150/while/Less
 backward_lstm_150/while/IdentityIdentity backward_lstm_150/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_150/while/Identity"M
 backward_lstm_150_while_identity)backward_lstm_150/while/Identity:output:0*(
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
аf
ф
%backward_lstm_150_while_body_18422105@
<backward_lstm_150_while_backward_lstm_150_while_loop_counterF
Bbackward_lstm_150_while_backward_lstm_150_while_maximum_iterations'
#backward_lstm_150_while_placeholder)
%backward_lstm_150_while_placeholder_1)
%backward_lstm_150_while_placeholder_2)
%backward_lstm_150_while_placeholder_3)
%backward_lstm_150_while_placeholder_4?
;backward_lstm_150_while_backward_lstm_150_strided_slice_1_0{
wbackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_150_while_less_backward_lstm_150_sub_1_0Y
Fbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0:	Ш[
Hbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0:	2ШV
Gbackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0:	Ш$
 backward_lstm_150_while_identity&
"backward_lstm_150_while_identity_1&
"backward_lstm_150_while_identity_2&
"backward_lstm_150_while_identity_3&
"backward_lstm_150_while_identity_4&
"backward_lstm_150_while_identity_5&
"backward_lstm_150_while_identity_6=
9backward_lstm_150_while_backward_lstm_150_strided_slice_1y
ubackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_150_while_less_backward_lstm_150_sub_1W
Dbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource:	ШY
Fbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource:	2ШT
Ebackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource:	ШЂ<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOpЂ;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpЂ=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOpч
Ibackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2K
Ibackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeП
;backward_lstm_150/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_150_while_placeholderRbackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02=
;backward_lstm_150/while/TensorArrayV2Read/TensorListGetItemЯ
backward_lstm_150/while/LessLess6backward_lstm_150_while_less_backward_lstm_150_sub_1_0#backward_lstm_150_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_150/while/Less
;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpReadVariableOpFbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02=
;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpЂ
,backward_lstm_150/while/lstm_cell_452/MatMulMatMulBbackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem:item:0Cbackward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_150/while/lstm_cell_452/MatMul
=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOpHbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02?
=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp
.backward_lstm_150/while/lstm_cell_452/MatMul_1MatMul%backward_lstm_150_while_placeholder_3Ebackward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ20
.backward_lstm_150/while/lstm_cell_452/MatMul_1
)backward_lstm_150/while/lstm_cell_452/addAddV26backward_lstm_150/while/lstm_cell_452/MatMul:product:08backward_lstm_150/while/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_150/while/lstm_cell_452/add
<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOpGbackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02>
<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp
-backward_lstm_150/while/lstm_cell_452/BiasAddBiasAdd-backward_lstm_150/while/lstm_cell_452/add:z:0Dbackward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_150/while/lstm_cell_452/BiasAddА
5backward_lstm_150/while/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5backward_lstm_150/while/lstm_cell_452/split/split_dimз
+backward_lstm_150/while/lstm_cell_452/splitSplit>backward_lstm_150/while/lstm_cell_452/split/split_dim:output:06backward_lstm_150/while/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2-
+backward_lstm_150/while/lstm_cell_452/splitб
-backward_lstm_150/while/lstm_cell_452/SigmoidSigmoid4backward_lstm_150/while/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22/
-backward_lstm_150/while/lstm_cell_452/Sigmoidе
/backward_lstm_150/while/lstm_cell_452/Sigmoid_1Sigmoid4backward_lstm_150/while/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_150/while/lstm_cell_452/Sigmoid_1ы
)backward_lstm_150/while/lstm_cell_452/mulMul3backward_lstm_150/while/lstm_cell_452/Sigmoid_1:y:0%backward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_150/while/lstm_cell_452/mulШ
*backward_lstm_150/while/lstm_cell_452/ReluRelu4backward_lstm_150/while/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_150/while/lstm_cell_452/Relu
+backward_lstm_150/while/lstm_cell_452/mul_1Mul1backward_lstm_150/while/lstm_cell_452/Sigmoid:y:08backward_lstm_150/while/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_150/while/lstm_cell_452/mul_1ѕ
+backward_lstm_150/while/lstm_cell_452/add_1AddV2-backward_lstm_150/while/lstm_cell_452/mul:z:0/backward_lstm_150/while/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_150/while/lstm_cell_452/add_1е
/backward_lstm_150/while/lstm_cell_452/Sigmoid_2Sigmoid4backward_lstm_150/while/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_150/while/lstm_cell_452/Sigmoid_2Ч
,backward_lstm_150/while/lstm_cell_452/Relu_1Relu/backward_lstm_150/while/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_150/while/lstm_cell_452/Relu_1
+backward_lstm_150/while/lstm_cell_452/mul_2Mul3backward_lstm_150/while/lstm_cell_452/Sigmoid_2:y:0:backward_lstm_150/while/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_150/while/lstm_cell_452/mul_2і
backward_lstm_150/while/SelectSelect backward_lstm_150/while/Less:z:0/backward_lstm_150/while/lstm_cell_452/mul_2:z:0%backward_lstm_150_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_150/while/Selectњ
 backward_lstm_150/while/Select_1Select backward_lstm_150/while/Less:z:0/backward_lstm_150/while/lstm_cell_452/mul_2:z:0%backward_lstm_150_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_150/while/Select_1њ
 backward_lstm_150/while/Select_2Select backward_lstm_150/while/Less:z:0/backward_lstm_150/while/lstm_cell_452/add_1:z:0%backward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_150/while/Select_2Г
<backward_lstm_150/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_150_while_placeholder_1#backward_lstm_150_while_placeholder'backward_lstm_150/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_150/while/TensorArrayV2Write/TensorListSetItem
backward_lstm_150/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_150/while/add/yБ
backward_lstm_150/while/addAddV2#backward_lstm_150_while_placeholder&backward_lstm_150/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/while/add
backward_lstm_150/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_150/while/add_1/yа
backward_lstm_150/while/add_1AddV2<backward_lstm_150_while_backward_lstm_150_while_loop_counter(backward_lstm_150/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/while/add_1Г
 backward_lstm_150/while/IdentityIdentity!backward_lstm_150/while/add_1:z:0^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_150/while/Identityи
"backward_lstm_150/while/Identity_1IdentityBbackward_lstm_150_while_backward_lstm_150_while_maximum_iterations^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_150/while/Identity_1Е
"backward_lstm_150/while/Identity_2Identitybackward_lstm_150/while/add:z:0^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_150/while/Identity_2т
"backward_lstm_150/while/Identity_3IdentityLbackward_lstm_150/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_150/while/Identity_3Ю
"backward_lstm_150/while/Identity_4Identity'backward_lstm_150/while/Select:output:0^backward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_150/while/Identity_4а
"backward_lstm_150/while/Identity_5Identity)backward_lstm_150/while/Select_1:output:0^backward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_150/while/Identity_5а
"backward_lstm_150/while/Identity_6Identity)backward_lstm_150/while/Select_2:output:0^backward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_150/while/Identity_6Л
backward_lstm_150/while/NoOpNoOp=^backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp<^backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp>^backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_150/while/NoOp"x
9backward_lstm_150_while_backward_lstm_150_strided_slice_1;backward_lstm_150_while_backward_lstm_150_strided_slice_1_0"M
 backward_lstm_150_while_identity)backward_lstm_150/while/Identity:output:0"Q
"backward_lstm_150_while_identity_1+backward_lstm_150/while/Identity_1:output:0"Q
"backward_lstm_150_while_identity_2+backward_lstm_150/while/Identity_2:output:0"Q
"backward_lstm_150_while_identity_3+backward_lstm_150/while/Identity_3:output:0"Q
"backward_lstm_150_while_identity_4+backward_lstm_150/while/Identity_4:output:0"Q
"backward_lstm_150_while_identity_5+backward_lstm_150/while/Identity_5:output:0"Q
"backward_lstm_150_while_identity_6+backward_lstm_150/while/Identity_6:output:0"n
4backward_lstm_150_while_less_backward_lstm_150_sub_16backward_lstm_150_while_less_backward_lstm_150_sub_1_0"
Ebackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resourceGbackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0"
Fbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resourceHbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0"
Dbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resourceFbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0"№
ubackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2|
<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp2z
;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp2~
=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
зу

#__inference__wrapped_model_18417756

args_0
args_0_1	q
^sequential_150_bidirectional_150_forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource:	Шs
`sequential_150_bidirectional_150_forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource:	2Шn
_sequential_150_bidirectional_150_forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource:	Шr
_sequential_150_bidirectional_150_backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource:	Шt
asequential_150_bidirectional_150_backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource:	2Шo
`sequential_150_bidirectional_150_backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource:	ШI
7sequential_150_dense_150_matmul_readvariableop_resource:dF
8sequential_150_dense_150_biasadd_readvariableop_resource:
identityЂWsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpЂVsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpЂXsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpЂ8sequential_150/bidirectional_150/backward_lstm_150/whileЂVsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpЂUsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpЂWsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOpЂ7sequential_150/bidirectional_150/forward_lstm_150/whileЂ/sequential_150/dense_150/BiasAdd/ReadVariableOpЂ.sequential_150/dense_150/MatMul/ReadVariableOpй
Fsequential_150/bidirectional_150/forward_lstm_150/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2H
Fsequential_150/bidirectional_150/forward_lstm_150/RaggedToTensor/zerosл
Fsequential_150/bidirectional_150/forward_lstm_150/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2H
Fsequential_150/bidirectional_150/forward_lstm_150/RaggedToTensor/Const
Usequential_150/bidirectional_150/forward_lstm_150/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorOsequential_150/bidirectional_150/forward_lstm_150/RaggedToTensor/Const:output:0args_0Osequential_150/bidirectional_150/forward_lstm_150/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2W
Usequential_150/bidirectional_150/forward_lstm_150/RaggedToTensor/RaggedTensorToTensor
\sequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2^
\sequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack
^sequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2`
^sequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1
^sequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2`
^sequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2Ю
Vsequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1esequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack:output:0gsequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1:output:0gsequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask2X
Vsequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice
^sequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2`
^sequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack
`sequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2b
`sequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1
`sequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2b
`sequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2к
Xsequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1gsequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack:output:0isequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0isequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2Z
Xsequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice_1
Lsequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/subSub_sequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice:output:0asequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2N
Lsequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/sub
6sequential_150/bidirectional_150/forward_lstm_150/CastCastPsequential_150/bidirectional_150/forward_lstm_150/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ28
6sequential_150/bidirectional_150/forward_lstm_150/Cast
7sequential_150/bidirectional_150/forward_lstm_150/ShapeShape^sequential_150/bidirectional_150/forward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:29
7sequential_150/bidirectional_150/forward_lstm_150/Shapeи
Esequential_150/bidirectional_150/forward_lstm_150/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_150/bidirectional_150/forward_lstm_150/strided_slice/stackм
Gsequential_150/bidirectional_150/forward_lstm_150/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_150/bidirectional_150/forward_lstm_150/strided_slice/stack_1м
Gsequential_150/bidirectional_150/forward_lstm_150/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_150/bidirectional_150/forward_lstm_150/strided_slice/stack_2
?sequential_150/bidirectional_150/forward_lstm_150/strided_sliceStridedSlice@sequential_150/bidirectional_150/forward_lstm_150/Shape:output:0Nsequential_150/bidirectional_150/forward_lstm_150/strided_slice/stack:output:0Psequential_150/bidirectional_150/forward_lstm_150/strided_slice/stack_1:output:0Psequential_150/bidirectional_150/forward_lstm_150/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?sequential_150/bidirectional_150/forward_lstm_150/strided_sliceР
=sequential_150/bidirectional_150/forward_lstm_150/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22?
=sequential_150/bidirectional_150/forward_lstm_150/zeros/mul/yД
;sequential_150/bidirectional_150/forward_lstm_150/zeros/mulMulHsequential_150/bidirectional_150/forward_lstm_150/strided_slice:output:0Fsequential_150/bidirectional_150/forward_lstm_150/zeros/mul/y:output:0*
T0*
_output_shapes
: 2=
;sequential_150/bidirectional_150/forward_lstm_150/zeros/mulУ
>sequential_150/bidirectional_150/forward_lstm_150/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2@
>sequential_150/bidirectional_150/forward_lstm_150/zeros/Less/yЏ
<sequential_150/bidirectional_150/forward_lstm_150/zeros/LessLess?sequential_150/bidirectional_150/forward_lstm_150/zeros/mul:z:0Gsequential_150/bidirectional_150/forward_lstm_150/zeros/Less/y:output:0*
T0*
_output_shapes
: 2>
<sequential_150/bidirectional_150/forward_lstm_150/zeros/LessЦ
@sequential_150/bidirectional_150/forward_lstm_150/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22B
@sequential_150/bidirectional_150/forward_lstm_150/zeros/packed/1Ы
>sequential_150/bidirectional_150/forward_lstm_150/zeros/packedPackHsequential_150/bidirectional_150/forward_lstm_150/strided_slice:output:0Isequential_150/bidirectional_150/forward_lstm_150/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2@
>sequential_150/bidirectional_150/forward_lstm_150/zeros/packedЧ
=sequential_150/bidirectional_150/forward_lstm_150/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2?
=sequential_150/bidirectional_150/forward_lstm_150/zeros/ConstН
7sequential_150/bidirectional_150/forward_lstm_150/zerosFillGsequential_150/bidirectional_150/forward_lstm_150/zeros/packed:output:0Fsequential_150/bidirectional_150/forward_lstm_150/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ229
7sequential_150/bidirectional_150/forward_lstm_150/zerosФ
?sequential_150/bidirectional_150/forward_lstm_150/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22A
?sequential_150/bidirectional_150/forward_lstm_150/zeros_1/mul/yК
=sequential_150/bidirectional_150/forward_lstm_150/zeros_1/mulMulHsequential_150/bidirectional_150/forward_lstm_150/strided_slice:output:0Hsequential_150/bidirectional_150/forward_lstm_150/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2?
=sequential_150/bidirectional_150/forward_lstm_150/zeros_1/mulЧ
@sequential_150/bidirectional_150/forward_lstm_150/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2B
@sequential_150/bidirectional_150/forward_lstm_150/zeros_1/Less/yЗ
>sequential_150/bidirectional_150/forward_lstm_150/zeros_1/LessLessAsequential_150/bidirectional_150/forward_lstm_150/zeros_1/mul:z:0Isequential_150/bidirectional_150/forward_lstm_150/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2@
>sequential_150/bidirectional_150/forward_lstm_150/zeros_1/LessЪ
Bsequential_150/bidirectional_150/forward_lstm_150/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22D
Bsequential_150/bidirectional_150/forward_lstm_150/zeros_1/packed/1б
@sequential_150/bidirectional_150/forward_lstm_150/zeros_1/packedPackHsequential_150/bidirectional_150/forward_lstm_150/strided_slice:output:0Ksequential_150/bidirectional_150/forward_lstm_150/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2B
@sequential_150/bidirectional_150/forward_lstm_150/zeros_1/packedЫ
?sequential_150/bidirectional_150/forward_lstm_150/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2A
?sequential_150/bidirectional_150/forward_lstm_150/zeros_1/ConstХ
9sequential_150/bidirectional_150/forward_lstm_150/zeros_1FillIsequential_150/bidirectional_150/forward_lstm_150/zeros_1/packed:output:0Hsequential_150/bidirectional_150/forward_lstm_150/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22;
9sequential_150/bidirectional_150/forward_lstm_150/zeros_1й
@sequential_150/bidirectional_150/forward_lstm_150/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2B
@sequential_150/bidirectional_150/forward_lstm_150/transpose/permё
;sequential_150/bidirectional_150/forward_lstm_150/transpose	Transpose^sequential_150/bidirectional_150/forward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0Isequential_150/bidirectional_150/forward_lstm_150/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2=
;sequential_150/bidirectional_150/forward_lstm_150/transposeх
9sequential_150/bidirectional_150/forward_lstm_150/Shape_1Shape?sequential_150/bidirectional_150/forward_lstm_150/transpose:y:0*
T0*
_output_shapes
:2;
9sequential_150/bidirectional_150/forward_lstm_150/Shape_1м
Gsequential_150/bidirectional_150/forward_lstm_150/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2I
Gsequential_150/bidirectional_150/forward_lstm_150/strided_slice_1/stackр
Isequential_150/bidirectional_150/forward_lstm_150/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_150/bidirectional_150/forward_lstm_150/strided_slice_1/stack_1р
Isequential_150/bidirectional_150/forward_lstm_150/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_150/bidirectional_150/forward_lstm_150/strided_slice_1/stack_2
Asequential_150/bidirectional_150/forward_lstm_150/strided_slice_1StridedSliceBsequential_150/bidirectional_150/forward_lstm_150/Shape_1:output:0Psequential_150/bidirectional_150/forward_lstm_150/strided_slice_1/stack:output:0Rsequential_150/bidirectional_150/forward_lstm_150/strided_slice_1/stack_1:output:0Rsequential_150/bidirectional_150/forward_lstm_150/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2C
Asequential_150/bidirectional_150/forward_lstm_150/strided_slice_1щ
Msequential_150/bidirectional_150/forward_lstm_150/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2O
Msequential_150/bidirectional_150/forward_lstm_150/TensorArrayV2/element_shapeњ
?sequential_150/bidirectional_150/forward_lstm_150/TensorArrayV2TensorListReserveVsequential_150/bidirectional_150/forward_lstm_150/TensorArrayV2/element_shape:output:0Jsequential_150/bidirectional_150/forward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?sequential_150/bidirectional_150/forward_lstm_150/TensorArrayV2Ѓ
gsequential_150/bidirectional_150/forward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2i
gsequential_150/bidirectional_150/forward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeР
Ysequential_150/bidirectional_150/forward_lstm_150/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor?sequential_150/bidirectional_150/forward_lstm_150/transpose:y:0psequential_150/bidirectional_150/forward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02[
Ysequential_150/bidirectional_150/forward_lstm_150/TensorArrayUnstack/TensorListFromTensorм
Gsequential_150/bidirectional_150/forward_lstm_150/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2I
Gsequential_150/bidirectional_150/forward_lstm_150/strided_slice_2/stackр
Isequential_150/bidirectional_150/forward_lstm_150/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_150/bidirectional_150/forward_lstm_150/strided_slice_2/stack_1р
Isequential_150/bidirectional_150/forward_lstm_150/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_150/bidirectional_150/forward_lstm_150/strided_slice_2/stack_2Ј
Asequential_150/bidirectional_150/forward_lstm_150/strided_slice_2StridedSlice?sequential_150/bidirectional_150/forward_lstm_150/transpose:y:0Psequential_150/bidirectional_150/forward_lstm_150/strided_slice_2/stack:output:0Rsequential_150/bidirectional_150/forward_lstm_150/strided_slice_2/stack_1:output:0Rsequential_150/bidirectional_150/forward_lstm_150/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2C
Asequential_150/bidirectional_150/forward_lstm_150/strided_slice_2Ю
Usequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpReadVariableOp^sequential_150_bidirectional_150_forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02W
Usequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpј
Fsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMulMatMulJsequential_150/bidirectional_150/forward_lstm_150/strided_slice_2:output:0]sequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2H
Fsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMulд
Wsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOp`sequential_150_bidirectional_150_forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02Y
Wsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOpє
Hsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMul_1MatMul@sequential_150/bidirectional_150/forward_lstm_150/zeros:output:0_sequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2J
Hsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMul_1ь
Csequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/addAddV2Psequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMul:product:0Rsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2E
Csequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/addЭ
Vsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOp_sequential_150_bidirectional_150_forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02X
Vsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpљ
Gsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/BiasAddBiasAddGsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/add:z:0^sequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2I
Gsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/BiasAddф
Osequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2Q
Osequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/split/split_dimП
Esequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/splitSplitXsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/split/split_dim:output:0Psequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2G
Esequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/split
Gsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/SigmoidSigmoidNsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22I
Gsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/SigmoidЃ
Isequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/Sigmoid_1SigmoidNsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22K
Isequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/Sigmoid_1ж
Csequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/mulMulMsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/Sigmoid_1:y:0Bsequential_150/bidirectional_150/forward_lstm_150/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/mul
Dsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/ReluReluNsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22F
Dsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/Reluш
Esequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/mul_1MulKsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/Sigmoid:y:0Rsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22G
Esequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/mul_1н
Esequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/add_1AddV2Gsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/mul:z:0Isequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22G
Esequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/add_1Ѓ
Isequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/Sigmoid_2SigmoidNsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22K
Isequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/Sigmoid_2
Fsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/Relu_1ReluIsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22H
Fsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/Relu_1ь
Esequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/mul_2MulMsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/Sigmoid_2:y:0Tsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22G
Esequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/mul_2ѓ
Osequential_150/bidirectional_150/forward_lstm_150/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2Q
Osequential_150/bidirectional_150/forward_lstm_150/TensorArrayV2_1/element_shape
Asequential_150/bidirectional_150/forward_lstm_150/TensorArrayV2_1TensorListReserveXsequential_150/bidirectional_150/forward_lstm_150/TensorArrayV2_1/element_shape:output:0Jsequential_150/bidirectional_150/forward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02C
Asequential_150/bidirectional_150/forward_lstm_150/TensorArrayV2_1В
6sequential_150/bidirectional_150/forward_lstm_150/timeConst*
_output_shapes
: *
dtype0*
value	B : 28
6sequential_150/bidirectional_150/forward_lstm_150/time
<sequential_150/bidirectional_150/forward_lstm_150/zeros_like	ZerosLikeIsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22>
<sequential_150/bidirectional_150/forward_lstm_150/zeros_likeу
Jsequential_150/bidirectional_150/forward_lstm_150/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2L
Jsequential_150/bidirectional_150/forward_lstm_150/while/maximum_iterationsЮ
Dsequential_150/bidirectional_150/forward_lstm_150/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dsequential_150/bidirectional_150/forward_lstm_150/while/loop_counterФ
7sequential_150/bidirectional_150/forward_lstm_150/whileWhileMsequential_150/bidirectional_150/forward_lstm_150/while/loop_counter:output:0Ssequential_150/bidirectional_150/forward_lstm_150/while/maximum_iterations:output:0?sequential_150/bidirectional_150/forward_lstm_150/time:output:0Jsequential_150/bidirectional_150/forward_lstm_150/TensorArrayV2_1:handle:0@sequential_150/bidirectional_150/forward_lstm_150/zeros_like:y:0@sequential_150/bidirectional_150/forward_lstm_150/zeros:output:0Bsequential_150/bidirectional_150/forward_lstm_150/zeros_1:output:0Jsequential_150/bidirectional_150/forward_lstm_150/strided_slice_1:output:0isequential_150/bidirectional_150/forward_lstm_150/TensorArrayUnstack/TensorListFromTensor:output_handle:0:sequential_150/bidirectional_150/forward_lstm_150/Cast:y:0^sequential_150_bidirectional_150_forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource`sequential_150_bidirectional_150_forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource_sequential_150_bidirectional_150_forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource*
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
Esequential_150_bidirectional_150_forward_lstm_150_while_body_18417473*Q
condIRG
Esequential_150_bidirectional_150_forward_lstm_150_while_cond_18417472*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 29
7sequential_150/bidirectional_150/forward_lstm_150/while
bsequential_150/bidirectional_150/forward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2d
bsequential_150/bidirectional_150/forward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeЙ
Tsequential_150/bidirectional_150/forward_lstm_150/TensorArrayV2Stack/TensorListStackTensorListStack@sequential_150/bidirectional_150/forward_lstm_150/while:output:3ksequential_150/bidirectional_150/forward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02V
Tsequential_150/bidirectional_150/forward_lstm_150/TensorArrayV2Stack/TensorListStackх
Gsequential_150/bidirectional_150/forward_lstm_150/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2I
Gsequential_150/bidirectional_150/forward_lstm_150/strided_slice_3/stackр
Isequential_150/bidirectional_150/forward_lstm_150/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2K
Isequential_150/bidirectional_150/forward_lstm_150/strided_slice_3/stack_1р
Isequential_150/bidirectional_150/forward_lstm_150/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_150/bidirectional_150/forward_lstm_150/strided_slice_3/stack_2Ц
Asequential_150/bidirectional_150/forward_lstm_150/strided_slice_3StridedSlice]sequential_150/bidirectional_150/forward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0Psequential_150/bidirectional_150/forward_lstm_150/strided_slice_3/stack:output:0Rsequential_150/bidirectional_150/forward_lstm_150/strided_slice_3/stack_1:output:0Rsequential_150/bidirectional_150/forward_lstm_150/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2C
Asequential_150/bidirectional_150/forward_lstm_150/strided_slice_3н
Bsequential_150/bidirectional_150/forward_lstm_150/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2D
Bsequential_150/bidirectional_150/forward_lstm_150/transpose_1/permі
=sequential_150/bidirectional_150/forward_lstm_150/transpose_1	Transpose]sequential_150/bidirectional_150/forward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0Ksequential_150/bidirectional_150/forward_lstm_150/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22?
=sequential_150/bidirectional_150/forward_lstm_150/transpose_1Ъ
9sequential_150/bidirectional_150/forward_lstm_150/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2;
9sequential_150/bidirectional_150/forward_lstm_150/runtimeл
Gsequential_150/bidirectional_150/backward_lstm_150/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2I
Gsequential_150/bidirectional_150/backward_lstm_150/RaggedToTensor/zerosн
Gsequential_150/bidirectional_150/backward_lstm_150/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2I
Gsequential_150/bidirectional_150/backward_lstm_150/RaggedToTensor/ConstЁ
Vsequential_150/bidirectional_150/backward_lstm_150/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorPsequential_150/bidirectional_150/backward_lstm_150/RaggedToTensor/Const:output:0args_0Psequential_150/bidirectional_150/backward_lstm_150/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2X
Vsequential_150/bidirectional_150/backward_lstm_150/RaggedToTensor/RaggedTensorToTensor
]sequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2_
]sequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack
_sequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2a
_sequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1
_sequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2a
_sequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2г
Wsequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1fsequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack:output:0hsequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1:output:0hsequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask2Y
Wsequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice
_sequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2a
_sequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack
asequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2c
asequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1
asequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2c
asequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2п
Ysequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1hsequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack:output:0jsequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0jsequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2[
Ysequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice_1
Msequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/subSub`sequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice:output:0bsequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2O
Msequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/sub
7sequential_150/bidirectional_150/backward_lstm_150/CastCastQsequential_150/bidirectional_150/backward_lstm_150/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ29
7sequential_150/bidirectional_150/backward_lstm_150/Cast
8sequential_150/bidirectional_150/backward_lstm_150/ShapeShape_sequential_150/bidirectional_150/backward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2:
8sequential_150/bidirectional_150/backward_lstm_150/Shapeк
Fsequential_150/bidirectional_150/backward_lstm_150/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fsequential_150/bidirectional_150/backward_lstm_150/strided_slice/stackо
Hsequential_150/bidirectional_150/backward_lstm_150/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_150/bidirectional_150/backward_lstm_150/strided_slice/stack_1о
Hsequential_150/bidirectional_150/backward_lstm_150/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_150/bidirectional_150/backward_lstm_150/strided_slice/stack_2
@sequential_150/bidirectional_150/backward_lstm_150/strided_sliceStridedSliceAsequential_150/bidirectional_150/backward_lstm_150/Shape:output:0Osequential_150/bidirectional_150/backward_lstm_150/strided_slice/stack:output:0Qsequential_150/bidirectional_150/backward_lstm_150/strided_slice/stack_1:output:0Qsequential_150/bidirectional_150/backward_lstm_150/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2B
@sequential_150/bidirectional_150/backward_lstm_150/strided_sliceТ
>sequential_150/bidirectional_150/backward_lstm_150/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22@
>sequential_150/bidirectional_150/backward_lstm_150/zeros/mul/yИ
<sequential_150/bidirectional_150/backward_lstm_150/zeros/mulMulIsequential_150/bidirectional_150/backward_lstm_150/strided_slice:output:0Gsequential_150/bidirectional_150/backward_lstm_150/zeros/mul/y:output:0*
T0*
_output_shapes
: 2>
<sequential_150/bidirectional_150/backward_lstm_150/zeros/mulХ
?sequential_150/bidirectional_150/backward_lstm_150/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2A
?sequential_150/bidirectional_150/backward_lstm_150/zeros/Less/yГ
=sequential_150/bidirectional_150/backward_lstm_150/zeros/LessLess@sequential_150/bidirectional_150/backward_lstm_150/zeros/mul:z:0Hsequential_150/bidirectional_150/backward_lstm_150/zeros/Less/y:output:0*
T0*
_output_shapes
: 2?
=sequential_150/bidirectional_150/backward_lstm_150/zeros/LessШ
Asequential_150/bidirectional_150/backward_lstm_150/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22C
Asequential_150/bidirectional_150/backward_lstm_150/zeros/packed/1Я
?sequential_150/bidirectional_150/backward_lstm_150/zeros/packedPackIsequential_150/bidirectional_150/backward_lstm_150/strided_slice:output:0Jsequential_150/bidirectional_150/backward_lstm_150/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2A
?sequential_150/bidirectional_150/backward_lstm_150/zeros/packedЩ
>sequential_150/bidirectional_150/backward_lstm_150/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2@
>sequential_150/bidirectional_150/backward_lstm_150/zeros/ConstС
8sequential_150/bidirectional_150/backward_lstm_150/zerosFillHsequential_150/bidirectional_150/backward_lstm_150/zeros/packed:output:0Gsequential_150/bidirectional_150/backward_lstm_150/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22:
8sequential_150/bidirectional_150/backward_lstm_150/zerosЦ
@sequential_150/bidirectional_150/backward_lstm_150/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22B
@sequential_150/bidirectional_150/backward_lstm_150/zeros_1/mul/yО
>sequential_150/bidirectional_150/backward_lstm_150/zeros_1/mulMulIsequential_150/bidirectional_150/backward_lstm_150/strided_slice:output:0Isequential_150/bidirectional_150/backward_lstm_150/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2@
>sequential_150/bidirectional_150/backward_lstm_150/zeros_1/mulЩ
Asequential_150/bidirectional_150/backward_lstm_150/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2C
Asequential_150/bidirectional_150/backward_lstm_150/zeros_1/Less/yЛ
?sequential_150/bidirectional_150/backward_lstm_150/zeros_1/LessLessBsequential_150/bidirectional_150/backward_lstm_150/zeros_1/mul:z:0Jsequential_150/bidirectional_150/backward_lstm_150/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2A
?sequential_150/bidirectional_150/backward_lstm_150/zeros_1/LessЬ
Csequential_150/bidirectional_150/backward_lstm_150/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22E
Csequential_150/bidirectional_150/backward_lstm_150/zeros_1/packed/1е
Asequential_150/bidirectional_150/backward_lstm_150/zeros_1/packedPackIsequential_150/bidirectional_150/backward_lstm_150/strided_slice:output:0Lsequential_150/bidirectional_150/backward_lstm_150/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2C
Asequential_150/bidirectional_150/backward_lstm_150/zeros_1/packedЭ
@sequential_150/bidirectional_150/backward_lstm_150/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2B
@sequential_150/bidirectional_150/backward_lstm_150/zeros_1/ConstЩ
:sequential_150/bidirectional_150/backward_lstm_150/zeros_1FillJsequential_150/bidirectional_150/backward_lstm_150/zeros_1/packed:output:0Isequential_150/bidirectional_150/backward_lstm_150/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22<
:sequential_150/bidirectional_150/backward_lstm_150/zeros_1л
Asequential_150/bidirectional_150/backward_lstm_150/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2C
Asequential_150/bidirectional_150/backward_lstm_150/transpose/permѕ
<sequential_150/bidirectional_150/backward_lstm_150/transpose	Transpose_sequential_150/bidirectional_150/backward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0Jsequential_150/bidirectional_150/backward_lstm_150/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2>
<sequential_150/bidirectional_150/backward_lstm_150/transposeш
:sequential_150/bidirectional_150/backward_lstm_150/Shape_1Shape@sequential_150/bidirectional_150/backward_lstm_150/transpose:y:0*
T0*
_output_shapes
:2<
:sequential_150/bidirectional_150/backward_lstm_150/Shape_1о
Hsequential_150/bidirectional_150/backward_lstm_150/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_150/bidirectional_150/backward_lstm_150/strided_slice_1/stackт
Jsequential_150/bidirectional_150/backward_lstm_150/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_150/bidirectional_150/backward_lstm_150/strided_slice_1/stack_1т
Jsequential_150/bidirectional_150/backward_lstm_150/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_150/bidirectional_150/backward_lstm_150/strided_slice_1/stack_2 
Bsequential_150/bidirectional_150/backward_lstm_150/strided_slice_1StridedSliceCsequential_150/bidirectional_150/backward_lstm_150/Shape_1:output:0Qsequential_150/bidirectional_150/backward_lstm_150/strided_slice_1/stack:output:0Ssequential_150/bidirectional_150/backward_lstm_150/strided_slice_1/stack_1:output:0Ssequential_150/bidirectional_150/backward_lstm_150/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2D
Bsequential_150/bidirectional_150/backward_lstm_150/strided_slice_1ы
Nsequential_150/bidirectional_150/backward_lstm_150/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2P
Nsequential_150/bidirectional_150/backward_lstm_150/TensorArrayV2/element_shapeў
@sequential_150/bidirectional_150/backward_lstm_150/TensorArrayV2TensorListReserveWsequential_150/bidirectional_150/backward_lstm_150/TensorArrayV2/element_shape:output:0Ksequential_150/bidirectional_150/backward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02B
@sequential_150/bidirectional_150/backward_lstm_150/TensorArrayV2а
Asequential_150/bidirectional_150/backward_lstm_150/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential_150/bidirectional_150/backward_lstm_150/ReverseV2/axisж
<sequential_150/bidirectional_150/backward_lstm_150/ReverseV2	ReverseV2@sequential_150/bidirectional_150/backward_lstm_150/transpose:y:0Jsequential_150/bidirectional_150/backward_lstm_150/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2>
<sequential_150/bidirectional_150/backward_lstm_150/ReverseV2Ѕ
hsequential_150/bidirectional_150/backward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2j
hsequential_150/bidirectional_150/backward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeЩ
Zsequential_150/bidirectional_150/backward_lstm_150/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorEsequential_150/bidirectional_150/backward_lstm_150/ReverseV2:output:0qsequential_150/bidirectional_150/backward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02\
Zsequential_150/bidirectional_150/backward_lstm_150/TensorArrayUnstack/TensorListFromTensorо
Hsequential_150/bidirectional_150/backward_lstm_150/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_150/bidirectional_150/backward_lstm_150/strided_slice_2/stackт
Jsequential_150/bidirectional_150/backward_lstm_150/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_150/bidirectional_150/backward_lstm_150/strided_slice_2/stack_1т
Jsequential_150/bidirectional_150/backward_lstm_150/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_150/bidirectional_150/backward_lstm_150/strided_slice_2/stack_2Ў
Bsequential_150/bidirectional_150/backward_lstm_150/strided_slice_2StridedSlice@sequential_150/bidirectional_150/backward_lstm_150/transpose:y:0Qsequential_150/bidirectional_150/backward_lstm_150/strided_slice_2/stack:output:0Ssequential_150/bidirectional_150/backward_lstm_150/strided_slice_2/stack_1:output:0Ssequential_150/bidirectional_150/backward_lstm_150/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2D
Bsequential_150/bidirectional_150/backward_lstm_150/strided_slice_2б
Vsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpReadVariableOp_sequential_150_bidirectional_150_backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02X
Vsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpќ
Gsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMulMatMulKsequential_150/bidirectional_150/backward_lstm_150/strided_slice_2:output:0^sequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2I
Gsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMulз
Xsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOpasequential_150_bidirectional_150_backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02Z
Xsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpј
Isequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMul_1MatMulAsequential_150/bidirectional_150/backward_lstm_150/zeros:output:0`sequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2K
Isequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMul_1№
Dsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/addAddV2Qsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMul:product:0Ssequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2F
Dsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/addа
Wsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOp`sequential_150_bidirectional_150_backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02Y
Wsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp§
Hsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/BiasAddBiasAddHsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/add:z:0_sequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2J
Hsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/BiasAddц
Psequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2R
Psequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/split/split_dimУ
Fsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/splitSplitYsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/split/split_dim:output:0Qsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2H
Fsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/splitЂ
Hsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/SigmoidSigmoidOsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22J
Hsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/SigmoidІ
Jsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/Sigmoid_1SigmoidOsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22L
Jsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/Sigmoid_1к
Dsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/mulMulNsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/Sigmoid_1:y:0Csequential_150/bidirectional_150/backward_lstm_150/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22F
Dsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/mul
Esequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/ReluReluOsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22G
Esequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/Reluь
Fsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/mul_1MulLsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/Sigmoid:y:0Ssequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22H
Fsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/mul_1с
Fsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/add_1AddV2Hsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/mul:z:0Jsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22H
Fsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/add_1І
Jsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/Sigmoid_2SigmoidOsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22L
Jsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/Sigmoid_2
Gsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/Relu_1ReluJsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22I
Gsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/Relu_1№
Fsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/mul_2MulNsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/Sigmoid_2:y:0Usequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22H
Fsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/mul_2ѕ
Psequential_150/bidirectional_150/backward_lstm_150/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2R
Psequential_150/bidirectional_150/backward_lstm_150/TensorArrayV2_1/element_shape
Bsequential_150/bidirectional_150/backward_lstm_150/TensorArrayV2_1TensorListReserveYsequential_150/bidirectional_150/backward_lstm_150/TensorArrayV2_1/element_shape:output:0Ksequential_150/bidirectional_150/backward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02D
Bsequential_150/bidirectional_150/backward_lstm_150/TensorArrayV2_1Д
7sequential_150/bidirectional_150/backward_lstm_150/timeConst*
_output_shapes
: *
dtype0*
value	B : 29
7sequential_150/bidirectional_150/backward_lstm_150/timeж
Hsequential_150/bidirectional_150/backward_lstm_150/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hsequential_150/bidirectional_150/backward_lstm_150/Max/reduction_indicesЈ
6sequential_150/bidirectional_150/backward_lstm_150/MaxMax;sequential_150/bidirectional_150/backward_lstm_150/Cast:y:0Qsequential_150/bidirectional_150/backward_lstm_150/Max/reduction_indices:output:0*
T0*
_output_shapes
: 28
6sequential_150/bidirectional_150/backward_lstm_150/MaxЖ
8sequential_150/bidirectional_150/backward_lstm_150/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_150/bidirectional_150/backward_lstm_150/sub/y
6sequential_150/bidirectional_150/backward_lstm_150/subSub?sequential_150/bidirectional_150/backward_lstm_150/Max:output:0Asequential_150/bidirectional_150/backward_lstm_150/sub/y:output:0*
T0*
_output_shapes
: 28
6sequential_150/bidirectional_150/backward_lstm_150/subЂ
8sequential_150/bidirectional_150/backward_lstm_150/Sub_1Sub:sequential_150/bidirectional_150/backward_lstm_150/sub:z:0;sequential_150/bidirectional_150/backward_lstm_150/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2:
8sequential_150/bidirectional_150/backward_lstm_150/Sub_1
=sequential_150/bidirectional_150/backward_lstm_150/zeros_like	ZerosLikeJsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22?
=sequential_150/bidirectional_150/backward_lstm_150/zeros_likeх
Ksequential_150/bidirectional_150/backward_lstm_150/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2M
Ksequential_150/bidirectional_150/backward_lstm_150/while/maximum_iterationsа
Esequential_150/bidirectional_150/backward_lstm_150/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2G
Esequential_150/bidirectional_150/backward_lstm_150/while/loop_counterж
8sequential_150/bidirectional_150/backward_lstm_150/whileWhileNsequential_150/bidirectional_150/backward_lstm_150/while/loop_counter:output:0Tsequential_150/bidirectional_150/backward_lstm_150/while/maximum_iterations:output:0@sequential_150/bidirectional_150/backward_lstm_150/time:output:0Ksequential_150/bidirectional_150/backward_lstm_150/TensorArrayV2_1:handle:0Asequential_150/bidirectional_150/backward_lstm_150/zeros_like:y:0Asequential_150/bidirectional_150/backward_lstm_150/zeros:output:0Csequential_150/bidirectional_150/backward_lstm_150/zeros_1:output:0Ksequential_150/bidirectional_150/backward_lstm_150/strided_slice_1:output:0jsequential_150/bidirectional_150/backward_lstm_150/TensorArrayUnstack/TensorListFromTensor:output_handle:0<sequential_150/bidirectional_150/backward_lstm_150/Sub_1:z:0_sequential_150_bidirectional_150_backward_lstm_150_lstm_cell_452_matmul_readvariableop_resourceasequential_150_bidirectional_150_backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource`sequential_150_bidirectional_150_backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource*
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
Fsequential_150_bidirectional_150_backward_lstm_150_while_body_18417652*R
condJRH
Fsequential_150_bidirectional_150_backward_lstm_150_while_cond_18417651*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2:
8sequential_150/bidirectional_150/backward_lstm_150/while
csequential_150/bidirectional_150/backward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2e
csequential_150/bidirectional_150/backward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeН
Usequential_150/bidirectional_150/backward_lstm_150/TensorArrayV2Stack/TensorListStackTensorListStackAsequential_150/bidirectional_150/backward_lstm_150/while:output:3lsequential_150/bidirectional_150/backward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02W
Usequential_150/bidirectional_150/backward_lstm_150/TensorArrayV2Stack/TensorListStackч
Hsequential_150/bidirectional_150/backward_lstm_150/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2J
Hsequential_150/bidirectional_150/backward_lstm_150/strided_slice_3/stackт
Jsequential_150/bidirectional_150/backward_lstm_150/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jsequential_150/bidirectional_150/backward_lstm_150/strided_slice_3/stack_1т
Jsequential_150/bidirectional_150/backward_lstm_150/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_150/bidirectional_150/backward_lstm_150/strided_slice_3/stack_2Ь
Bsequential_150/bidirectional_150/backward_lstm_150/strided_slice_3StridedSlice^sequential_150/bidirectional_150/backward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0Qsequential_150/bidirectional_150/backward_lstm_150/strided_slice_3/stack:output:0Ssequential_150/bidirectional_150/backward_lstm_150/strided_slice_3/stack_1:output:0Ssequential_150/bidirectional_150/backward_lstm_150/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2D
Bsequential_150/bidirectional_150/backward_lstm_150/strided_slice_3п
Csequential_150/bidirectional_150/backward_lstm_150/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2E
Csequential_150/bidirectional_150/backward_lstm_150/transpose_1/permњ
>sequential_150/bidirectional_150/backward_lstm_150/transpose_1	Transpose^sequential_150/bidirectional_150/backward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0Lsequential_150/bidirectional_150/backward_lstm_150/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22@
>sequential_150/bidirectional_150/backward_lstm_150/transpose_1Ь
:sequential_150/bidirectional_150/backward_lstm_150/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2<
:sequential_150/bidirectional_150/backward_lstm_150/runtime
,sequential_150/bidirectional_150/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_150/bidirectional_150/concat/axisщ
'sequential_150/bidirectional_150/concatConcatV2Jsequential_150/bidirectional_150/forward_lstm_150/strided_slice_3:output:0Ksequential_150/bidirectional_150/backward_lstm_150/strided_slice_3:output:05sequential_150/bidirectional_150/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2)
'sequential_150/bidirectional_150/concatи
.sequential_150/dense_150/MatMul/ReadVariableOpReadVariableOp7sequential_150_dense_150_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_150/dense_150/MatMul/ReadVariableOpш
sequential_150/dense_150/MatMulMatMul0sequential_150/bidirectional_150/concat:output:06sequential_150/dense_150/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
sequential_150/dense_150/MatMulз
/sequential_150/dense_150/BiasAdd/ReadVariableOpReadVariableOp8sequential_150_dense_150_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_150/dense_150/BiasAdd/ReadVariableOpх
 sequential_150/dense_150/BiasAddBiasAdd)sequential_150/dense_150/MatMul:product:07sequential_150/dense_150/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 sequential_150/dense_150/BiasAddЌ
 sequential_150/dense_150/SigmoidSigmoid)sequential_150/dense_150/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 sequential_150/dense_150/Sigmoid
IdentityIdentity$sequential_150/dense_150/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityП
NoOpNoOpX^sequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpW^sequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpY^sequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp9^sequential_150/bidirectional_150/backward_lstm_150/whileW^sequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpV^sequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpX^sequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp8^sequential_150/bidirectional_150/forward_lstm_150/while0^sequential_150/dense_150/BiasAdd/ReadVariableOp/^sequential_150/dense_150/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2В
Wsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpWsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp2А
Vsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpVsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp2Д
Xsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpXsequential_150/bidirectional_150/backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp2t
8sequential_150/bidirectional_150/backward_lstm_150/while8sequential_150/bidirectional_150/backward_lstm_150/while2А
Vsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpVsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp2Ў
Usequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpUsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp2В
Wsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOpWsequential_150/bidirectional_150/forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp2r
7sequential_150/bidirectional_150/forward_lstm_150/while7sequential_150/bidirectional_150/forward_lstm_150/while2b
/sequential_150/dense_150/BiasAdd/ReadVariableOp/sequential_150/dense_150/BiasAdd/ReadVariableOp2`
.sequential_150/dense_150/MatMul/ReadVariableOp.sequential_150/dense_150/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameargs_0:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameargs_0
ы	

4__inference_bidirectional_150_layer_call_fn_18420846
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
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_184197542
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
р
Р
3__inference_forward_lstm_150_layer_call_fn_18422255

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
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_184191812
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
ъ
М
%backward_lstm_150_while_cond_18420534@
<backward_lstm_150_while_backward_lstm_150_while_loop_counterF
Bbackward_lstm_150_while_backward_lstm_150_while_maximum_iterations'
#backward_lstm_150_while_placeholder)
%backward_lstm_150_while_placeholder_1)
%backward_lstm_150_while_placeholder_2)
%backward_lstm_150_while_placeholder_3)
%backward_lstm_150_while_placeholder_4B
>backward_lstm_150_while_less_backward_lstm_150_strided_slice_1Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18420534___redundant_placeholder0Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18420534___redundant_placeholder1Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18420534___redundant_placeholder2Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18420534___redundant_placeholder3Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18420534___redundant_placeholder4$
 backward_lstm_150_while_identity
Ъ
backward_lstm_150/while/LessLess#backward_lstm_150_while_placeholder>backward_lstm_150_while_less_backward_lstm_150_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_150/while/Less
 backward_lstm_150/while/IdentityIdentity backward_lstm_150/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_150/while/Identity"M
 backward_lstm_150_while_identity)backward_lstm_150/while/Identity:output:0*(
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
while_cond_18418476
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_18418476___redundant_placeholder06
2while_while_cond_18418476___redundant_placeholder16
2while_while_cond_18418476___redundant_placeholder26
2while_while_cond_18418476___redundant_placeholder3
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
]
Ќ
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_18419706

inputs?
,lstm_cell_451_matmul_readvariableop_resource:	ШA
.lstm_cell_451_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_451_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_451/BiasAdd/ReadVariableOpЂ#lstm_cell_451/MatMul/ReadVariableOpЂ%lstm_cell_451/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_451/MatMul/ReadVariableOpReadVariableOp,lstm_cell_451_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_451/MatMul/ReadVariableOpА
lstm_cell_451/MatMulMatMulstrided_slice_2:output:0+lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/MatMulО
%lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_451_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_451/MatMul_1/ReadVariableOpЌ
lstm_cell_451/MatMul_1MatMulzeros:output:0-lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/MatMul_1Є
lstm_cell_451/addAddV2lstm_cell_451/MatMul:product:0 lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/addЗ
$lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_451_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_451/BiasAdd/ReadVariableOpБ
lstm_cell_451/BiasAddBiasAddlstm_cell_451/add:z:0,lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/BiasAdd
lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_451/split/split_dimї
lstm_cell_451/splitSplit&lstm_cell_451/split/split_dim:output:0lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_451/split
lstm_cell_451/SigmoidSigmoidlstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Sigmoid
lstm_cell_451/Sigmoid_1Sigmoidlstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Sigmoid_1
lstm_cell_451/mulMullstm_cell_451/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/mul
lstm_cell_451/ReluRelulstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Relu 
lstm_cell_451/mul_1Mullstm_cell_451/Sigmoid:y:0 lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/mul_1
lstm_cell_451/add_1AddV2lstm_cell_451/mul:z:0lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/add_1
lstm_cell_451/Sigmoid_2Sigmoidlstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Sigmoid_2
lstm_cell_451/Relu_1Relulstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Relu_1Є
lstm_cell_451/mul_2Mullstm_cell_451/Sigmoid_2:y:0"lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_451_matmul_readvariableop_resource.lstm_cell_451_matmul_1_readvariableop_resource-lstm_cell_451_biasadd_readvariableop_resource*
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
while_body_18419622*
condR
while_cond_18419621*K
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
NoOpNoOp%^lstm_cell_451/BiasAdd/ReadVariableOp$^lstm_cell_451/MatMul/ReadVariableOp&^lstm_cell_451/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_451/BiasAdd/ReadVariableOp$lstm_cell_451/BiasAdd/ReadVariableOp2J
#lstm_cell_451/MatMul/ReadVariableOp#lstm_cell_451/MatMul/ReadVariableOp2N
%lstm_cell_451/MatMul_1/ReadVariableOp%lstm_cell_451/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Н
я
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_18421844

inputs
inputs_1	P
=forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource:	ШR
?forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource:	2ШM
>forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource:	ШQ
>backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource:	ШS
@backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource:	2ШN
?backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource:	Ш
identityЂ6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpЂ5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpЂ7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpЂbackward_lstm_150/whileЂ5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpЂ4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpЂ6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOpЂforward_lstm_150/while
%forward_lstm_150/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_150/RaggedToTensor/zeros
%forward_lstm_150/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2'
%forward_lstm_150/RaggedToTensor/Const
4forward_lstm_150/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_150/RaggedToTensor/Const:output:0inputs.forward_lstm_150/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_150/RaggedToTensor/RaggedTensorToTensorФ
;forward_lstm_150/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_150/RaggedNestedRowLengths/strided_slice/stackШ
=forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1Ш
=forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2Љ
5forward_lstm_150/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_150/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask27
5forward_lstm_150/RaggedNestedRowLengths/strided_sliceШ
=forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stackе
?forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2A
?forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1Ь
?forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2Е
7forward_lstm_150/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask29
7forward_lstm_150/RaggedNestedRowLengths/strided_slice_1
+forward_lstm_150/RaggedNestedRowLengths/subSub>forward_lstm_150/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_150/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2-
+forward_lstm_150/RaggedNestedRowLengths/subЄ
forward_lstm_150/CastCast/forward_lstm_150/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_150/Cast
forward_lstm_150/ShapeShape=forward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_150/Shape
$forward_lstm_150/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_150/strided_slice/stack
&forward_lstm_150/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_150/strided_slice/stack_1
&forward_lstm_150/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_150/strided_slice/stack_2Ш
forward_lstm_150/strided_sliceStridedSliceforward_lstm_150/Shape:output:0-forward_lstm_150/strided_slice/stack:output:0/forward_lstm_150/strided_slice/stack_1:output:0/forward_lstm_150/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_150/strided_slice~
forward_lstm_150/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_150/zeros/mul/yА
forward_lstm_150/zeros/mulMul'forward_lstm_150/strided_slice:output:0%forward_lstm_150/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros/mul
forward_lstm_150/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_150/zeros/Less/yЋ
forward_lstm_150/zeros/LessLessforward_lstm_150/zeros/mul:z:0&forward_lstm_150/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros/Less
forward_lstm_150/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_150/zeros/packed/1Ч
forward_lstm_150/zeros/packedPack'forward_lstm_150/strided_slice:output:0(forward_lstm_150/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_150/zeros/packed
forward_lstm_150/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_150/zeros/ConstЙ
forward_lstm_150/zerosFill&forward_lstm_150/zeros/packed:output:0%forward_lstm_150/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/zeros
forward_lstm_150/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_150/zeros_1/mul/yЖ
forward_lstm_150/zeros_1/mulMul'forward_lstm_150/strided_slice:output:0'forward_lstm_150/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros_1/mul
forward_lstm_150/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
forward_lstm_150/zeros_1/Less/yГ
forward_lstm_150/zeros_1/LessLess forward_lstm_150/zeros_1/mul:z:0(forward_lstm_150/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros_1/Less
!forward_lstm_150/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_150/zeros_1/packed/1Э
forward_lstm_150/zeros_1/packedPack'forward_lstm_150/strided_slice:output:0*forward_lstm_150/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_150/zeros_1/packed
forward_lstm_150/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_150/zeros_1/ConstС
forward_lstm_150/zeros_1Fill(forward_lstm_150/zeros_1/packed:output:0'forward_lstm_150/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/zeros_1
forward_lstm_150/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_150/transpose/permэ
forward_lstm_150/transpose	Transpose=forward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_150/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_150/transpose
forward_lstm_150/Shape_1Shapeforward_lstm_150/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_150/Shape_1
&forward_lstm_150/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_150/strided_slice_1/stack
(forward_lstm_150/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_1/stack_1
(forward_lstm_150/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_1/stack_2д
 forward_lstm_150/strided_slice_1StridedSlice!forward_lstm_150/Shape_1:output:0/forward_lstm_150/strided_slice_1/stack:output:01forward_lstm_150/strided_slice_1/stack_1:output:01forward_lstm_150/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_150/strided_slice_1Ї
,forward_lstm_150/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,forward_lstm_150/TensorArrayV2/element_shapeі
forward_lstm_150/TensorArrayV2TensorListReserve5forward_lstm_150/TensorArrayV2/element_shape:output:0)forward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_150/TensorArrayV2с
Fforward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fforward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8forward_lstm_150/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_150/transpose:y:0Oforward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_150/TensorArrayUnstack/TensorListFromTensor
&forward_lstm_150/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_150/strided_slice_2/stack
(forward_lstm_150/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_2/stack_1
(forward_lstm_150/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_2/stack_2т
 forward_lstm_150/strided_slice_2StridedSliceforward_lstm_150/transpose:y:0/forward_lstm_150/strided_slice_2/stack:output:01forward_lstm_150/strided_slice_2/stack_1:output:01forward_lstm_150/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 forward_lstm_150/strided_slice_2ы
4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpReadVariableOp=forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpє
%forward_lstm_150/lstm_cell_451/MatMulMatMul)forward_lstm_150/strided_slice_2:output:0<forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_150/lstm_cell_451/MatMulё
6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOp?forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp№
'forward_lstm_150/lstm_cell_451/MatMul_1MatMulforward_lstm_150/zeros:output:0>forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_150/lstm_cell_451/MatMul_1ш
"forward_lstm_150/lstm_cell_451/addAddV2/forward_lstm_150/lstm_cell_451/MatMul:product:01forward_lstm_150/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_150/lstm_cell_451/addъ
5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOp>forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpѕ
&forward_lstm_150/lstm_cell_451/BiasAddBiasAdd&forward_lstm_150/lstm_cell_451/add:z:0=forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_150/lstm_cell_451/BiasAddЂ
.forward_lstm_150/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.forward_lstm_150/lstm_cell_451/split/split_dimЛ
$forward_lstm_150/lstm_cell_451/splitSplit7forward_lstm_150/lstm_cell_451/split/split_dim:output:0/forward_lstm_150/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$forward_lstm_150/lstm_cell_451/splitМ
&forward_lstm_150/lstm_cell_451/SigmoidSigmoid-forward_lstm_150/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_150/lstm_cell_451/SigmoidР
(forward_lstm_150/lstm_cell_451/Sigmoid_1Sigmoid-forward_lstm_150/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_150/lstm_cell_451/Sigmoid_1в
"forward_lstm_150/lstm_cell_451/mulMul,forward_lstm_150/lstm_cell_451/Sigmoid_1:y:0!forward_lstm_150/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_150/lstm_cell_451/mulГ
#forward_lstm_150/lstm_cell_451/ReluRelu-forward_lstm_150/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_150/lstm_cell_451/Reluф
$forward_lstm_150/lstm_cell_451/mul_1Mul*forward_lstm_150/lstm_cell_451/Sigmoid:y:01forward_lstm_150/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_150/lstm_cell_451/mul_1й
$forward_lstm_150/lstm_cell_451/add_1AddV2&forward_lstm_150/lstm_cell_451/mul:z:0(forward_lstm_150/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_150/lstm_cell_451/add_1Р
(forward_lstm_150/lstm_cell_451/Sigmoid_2Sigmoid-forward_lstm_150/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_150/lstm_cell_451/Sigmoid_2В
%forward_lstm_150/lstm_cell_451/Relu_1Relu(forward_lstm_150/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_150/lstm_cell_451/Relu_1ш
$forward_lstm_150/lstm_cell_451/mul_2Mul,forward_lstm_150/lstm_cell_451/Sigmoid_2:y:03forward_lstm_150/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_150/lstm_cell_451/mul_2Б
.forward_lstm_150/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.forward_lstm_150/TensorArrayV2_1/element_shapeќ
 forward_lstm_150/TensorArrayV2_1TensorListReserve7forward_lstm_150/TensorArrayV2_1/element_shape:output:0)forward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_150/TensorArrayV2_1p
forward_lstm_150/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_150/timeЃ
forward_lstm_150/zeros_like	ZerosLike(forward_lstm_150/lstm_cell_451/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/zeros_likeЁ
)forward_lstm_150/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)forward_lstm_150/while/maximum_iterations
#forward_lstm_150/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_150/while/loop_counter	
forward_lstm_150/whileWhile,forward_lstm_150/while/loop_counter:output:02forward_lstm_150/while/maximum_iterations:output:0forward_lstm_150/time:output:0)forward_lstm_150/TensorArrayV2_1:handle:0forward_lstm_150/zeros_like:y:0forward_lstm_150/zeros:output:0!forward_lstm_150/zeros_1:output:0)forward_lstm_150/strided_slice_1:output:0Hforward_lstm_150/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_150/Cast:y:0=forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource?forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource>forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource*
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
$forward_lstm_150_while_body_18421568*0
cond(R&
$forward_lstm_150_while_cond_18421567*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_150/whileз
Aforward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Aforward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3forward_lstm_150/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_150/while:output:3Jforward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3forward_lstm_150/TensorArrayV2Stack/TensorListStackЃ
&forward_lstm_150/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&forward_lstm_150/strided_slice_3/stack
(forward_lstm_150/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_150/strided_slice_3/stack_1
(forward_lstm_150/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_3/stack_2
 forward_lstm_150/strided_slice_3StridedSlice<forward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_150/strided_slice_3/stack:output:01forward_lstm_150/strided_slice_3/stack_1:output:01forward_lstm_150/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 forward_lstm_150/strided_slice_3
!forward_lstm_150/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_150/transpose_1/permђ
forward_lstm_150/transpose_1	Transpose<forward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_150/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_150/transpose_1
forward_lstm_150/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_150/runtime
&backward_lstm_150/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_150/RaggedToTensor/zeros
&backward_lstm_150/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2(
&backward_lstm_150/RaggedToTensor/Const
5backward_lstm_150/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_150/RaggedToTensor/Const:output:0inputs/backward_lstm_150/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_150/RaggedToTensor/RaggedTensorToTensorЦ
<backward_lstm_150/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_150/RaggedNestedRowLengths/strided_slice/stackЪ
>backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1Ъ
>backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2Ў
6backward_lstm_150/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_150/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask28
6backward_lstm_150/RaggedNestedRowLengths/strided_sliceЪ
>backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stackз
@backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2B
@backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1Ю
@backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2К
8backward_lstm_150/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2:
8backward_lstm_150/RaggedNestedRowLengths/strided_slice_1
,backward_lstm_150/RaggedNestedRowLengths/subSub?backward_lstm_150/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_150/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2.
,backward_lstm_150/RaggedNestedRowLengths/subЇ
backward_lstm_150/CastCast0backward_lstm_150/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_150/Cast 
backward_lstm_150/ShapeShape>backward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_150/Shape
%backward_lstm_150/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_150/strided_slice/stack
'backward_lstm_150/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_150/strided_slice/stack_1
'backward_lstm_150/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_150/strided_slice/stack_2Ю
backward_lstm_150/strided_sliceStridedSlice backward_lstm_150/Shape:output:0.backward_lstm_150/strided_slice/stack:output:00backward_lstm_150/strided_slice/stack_1:output:00backward_lstm_150/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_150/strided_slice
backward_lstm_150/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_150/zeros/mul/yД
backward_lstm_150/zeros/mulMul(backward_lstm_150/strided_slice:output:0&backward_lstm_150/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/zeros/mul
backward_lstm_150/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_150/zeros/Less/yЏ
backward_lstm_150/zeros/LessLessbackward_lstm_150/zeros/mul:z:0'backward_lstm_150/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/zeros/Less
 backward_lstm_150/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_150/zeros/packed/1Ы
backward_lstm_150/zeros/packedPack(backward_lstm_150/strided_slice:output:0)backward_lstm_150/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_150/zeros/packed
backward_lstm_150/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_150/zeros/ConstН
backward_lstm_150/zerosFill'backward_lstm_150/zeros/packed:output:0&backward_lstm_150/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_150/zeros
backward_lstm_150/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_150/zeros_1/mul/yК
backward_lstm_150/zeros_1/mulMul(backward_lstm_150/strided_slice:output:0(backward_lstm_150/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/zeros_1/mul
 backward_lstm_150/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 backward_lstm_150/zeros_1/Less/yЗ
backward_lstm_150/zeros_1/LessLess!backward_lstm_150/zeros_1/mul:z:0)backward_lstm_150/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_150/zeros_1/Less
"backward_lstm_150/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_150/zeros_1/packed/1б
 backward_lstm_150/zeros_1/packedPack(backward_lstm_150/strided_slice:output:0+backward_lstm_150/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_150/zeros_1/packed
backward_lstm_150/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_150/zeros_1/ConstХ
backward_lstm_150/zeros_1Fill)backward_lstm_150/zeros_1/packed:output:0(backward_lstm_150/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_150/zeros_1
 backward_lstm_150/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_150/transpose/permё
backward_lstm_150/transpose	Transpose>backward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_150/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_150/transpose
backward_lstm_150/Shape_1Shapebackward_lstm_150/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_150/Shape_1
'backward_lstm_150/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_150/strided_slice_1/stack 
)backward_lstm_150/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_1/stack_1 
)backward_lstm_150/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_1/stack_2к
!backward_lstm_150/strided_slice_1StridedSlice"backward_lstm_150/Shape_1:output:00backward_lstm_150/strided_slice_1/stack:output:02backward_lstm_150/strided_slice_1/stack_1:output:02backward_lstm_150/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_150/strided_slice_1Љ
-backward_lstm_150/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2/
-backward_lstm_150/TensorArrayV2/element_shapeњ
backward_lstm_150/TensorArrayV2TensorListReserve6backward_lstm_150/TensorArrayV2/element_shape:output:0*backward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_150/TensorArrayV2
 backward_lstm_150/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_150/ReverseV2/axisв
backward_lstm_150/ReverseV2	ReverseV2backward_lstm_150/transpose:y:0)backward_lstm_150/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_150/ReverseV2у
Gbackward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gbackward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeХ
9backward_lstm_150/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_150/ReverseV2:output:0Pbackward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_150/TensorArrayUnstack/TensorListFromTensor
'backward_lstm_150/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_150/strided_slice_2/stack 
)backward_lstm_150/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_2/stack_1 
)backward_lstm_150/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_2/stack_2ш
!backward_lstm_150/strided_slice_2StridedSlicebackward_lstm_150/transpose:y:00backward_lstm_150/strided_slice_2/stack:output:02backward_lstm_150/strided_slice_2/stack_1:output:02backward_lstm_150/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2#
!backward_lstm_150/strided_slice_2ю
5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpReadVariableOp>backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype027
5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpј
&backward_lstm_150/lstm_cell_452/MatMulMatMul*backward_lstm_150/strided_slice_2:output:0=backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_150/lstm_cell_452/MatMulє
7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOp@backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype029
7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpє
(backward_lstm_150/lstm_cell_452/MatMul_1MatMul backward_lstm_150/zeros:output:0?backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_150/lstm_cell_452/MatMul_1ь
#backward_lstm_150/lstm_cell_452/addAddV20backward_lstm_150/lstm_cell_452/MatMul:product:02backward_lstm_150/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_150/lstm_cell_452/addэ
6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOp?backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype028
6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpљ
'backward_lstm_150/lstm_cell_452/BiasAddBiasAdd'backward_lstm_150/lstm_cell_452/add:z:0>backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_150/lstm_cell_452/BiasAddЄ
/backward_lstm_150/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/backward_lstm_150/lstm_cell_452/split/split_dimП
%backward_lstm_150/lstm_cell_452/splitSplit8backward_lstm_150/lstm_cell_452/split/split_dim:output:00backward_lstm_150/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2'
%backward_lstm_150/lstm_cell_452/splitП
'backward_lstm_150/lstm_cell_452/SigmoidSigmoid.backward_lstm_150/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_150/lstm_cell_452/SigmoidУ
)backward_lstm_150/lstm_cell_452/Sigmoid_1Sigmoid.backward_lstm_150/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_150/lstm_cell_452/Sigmoid_1ж
#backward_lstm_150/lstm_cell_452/mulMul-backward_lstm_150/lstm_cell_452/Sigmoid_1:y:0"backward_lstm_150/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_150/lstm_cell_452/mulЖ
$backward_lstm_150/lstm_cell_452/ReluRelu.backward_lstm_150/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_150/lstm_cell_452/Reluш
%backward_lstm_150/lstm_cell_452/mul_1Mul+backward_lstm_150/lstm_cell_452/Sigmoid:y:02backward_lstm_150/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_150/lstm_cell_452/mul_1н
%backward_lstm_150/lstm_cell_452/add_1AddV2'backward_lstm_150/lstm_cell_452/mul:z:0)backward_lstm_150/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_150/lstm_cell_452/add_1У
)backward_lstm_150/lstm_cell_452/Sigmoid_2Sigmoid.backward_lstm_150/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_150/lstm_cell_452/Sigmoid_2Е
&backward_lstm_150/lstm_cell_452/Relu_1Relu)backward_lstm_150/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_150/lstm_cell_452/Relu_1ь
%backward_lstm_150/lstm_cell_452/mul_2Mul-backward_lstm_150/lstm_cell_452/Sigmoid_2:y:04backward_lstm_150/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_150/lstm_cell_452/mul_2Г
/backward_lstm_150/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   21
/backward_lstm_150/TensorArrayV2_1/element_shape
!backward_lstm_150/TensorArrayV2_1TensorListReserve8backward_lstm_150/TensorArrayV2_1/element_shape:output:0*backward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_150/TensorArrayV2_1r
backward_lstm_150/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_150/time
'backward_lstm_150/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_150/Max/reduction_indicesЄ
backward_lstm_150/MaxMaxbackward_lstm_150/Cast:y:00backward_lstm_150/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/Maxt
backward_lstm_150/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_150/sub/y
backward_lstm_150/subSubbackward_lstm_150/Max:output:0 backward_lstm_150/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/sub
backward_lstm_150/Sub_1Subbackward_lstm_150/sub:z:0backward_lstm_150/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_150/Sub_1І
backward_lstm_150/zeros_like	ZerosLike)backward_lstm_150/lstm_cell_452/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_150/zeros_likeЃ
*backward_lstm_150/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*backward_lstm_150/while/maximum_iterations
$backward_lstm_150/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_150/while/loop_counterЅ	
backward_lstm_150/whileWhile-backward_lstm_150/while/loop_counter:output:03backward_lstm_150/while/maximum_iterations:output:0backward_lstm_150/time:output:0*backward_lstm_150/TensorArrayV2_1:handle:0 backward_lstm_150/zeros_like:y:0 backward_lstm_150/zeros:output:0"backward_lstm_150/zeros_1:output:0*backward_lstm_150/strided_slice_1:output:0Ibackward_lstm_150/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_150/Sub_1:z:0>backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource@backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource?backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource*
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
%backward_lstm_150_while_body_18421747*1
cond)R'
%backward_lstm_150_while_cond_18421746*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_150/whileй
Bbackward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2D
Bbackward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeЙ
4backward_lstm_150/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_150/while:output:3Kbackward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype026
4backward_lstm_150/TensorArrayV2Stack/TensorListStackЅ
'backward_lstm_150/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2)
'backward_lstm_150/strided_slice_3/stack 
)backward_lstm_150/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_150/strided_slice_3/stack_1 
)backward_lstm_150/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_3/stack_2
!backward_lstm_150/strided_slice_3StridedSlice=backward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_150/strided_slice_3/stack:output:02backward_lstm_150/strided_slice_3/stack_1:output:02backward_lstm_150/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2#
!backward_lstm_150/strided_slice_3
"backward_lstm_150/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_150/transpose_1/permі
backward_lstm_150/transpose_1	Transpose=backward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_150/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_150/transpose_1
backward_lstm_150/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_150/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisФ
concatConcatV2)forward_lstm_150/strided_slice_3:output:0*backward_lstm_150/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp7^backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp6^backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp8^backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp^backward_lstm_150/while6^forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp5^forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp7^forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp^forward_lstm_150/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2p
6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp2n
5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp2r
7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp22
backward_lstm_150/whilebackward_lstm_150/while2n
5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp2l
4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp2p
6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp20
forward_lstm_150/whileforward_lstm_150/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф§
у
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_18421184
inputs_0P
=forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource:	ШR
?forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource:	2ШM
>forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource:	ШQ
>backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource:	ШS
@backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource:	2ШN
?backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource:	Ш
identityЂ6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpЂ5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpЂ7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpЂbackward_lstm_150/whileЂ5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpЂ4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpЂ6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOpЂforward_lstm_150/whileh
forward_lstm_150/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_150/Shape
$forward_lstm_150/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_150/strided_slice/stack
&forward_lstm_150/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_150/strided_slice/stack_1
&forward_lstm_150/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_150/strided_slice/stack_2Ш
forward_lstm_150/strided_sliceStridedSliceforward_lstm_150/Shape:output:0-forward_lstm_150/strided_slice/stack:output:0/forward_lstm_150/strided_slice/stack_1:output:0/forward_lstm_150/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_150/strided_slice~
forward_lstm_150/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_150/zeros/mul/yА
forward_lstm_150/zeros/mulMul'forward_lstm_150/strided_slice:output:0%forward_lstm_150/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros/mul
forward_lstm_150/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_150/zeros/Less/yЋ
forward_lstm_150/zeros/LessLessforward_lstm_150/zeros/mul:z:0&forward_lstm_150/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros/Less
forward_lstm_150/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_150/zeros/packed/1Ч
forward_lstm_150/zeros/packedPack'forward_lstm_150/strided_slice:output:0(forward_lstm_150/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_150/zeros/packed
forward_lstm_150/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_150/zeros/ConstЙ
forward_lstm_150/zerosFill&forward_lstm_150/zeros/packed:output:0%forward_lstm_150/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/zeros
forward_lstm_150/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_150/zeros_1/mul/yЖ
forward_lstm_150/zeros_1/mulMul'forward_lstm_150/strided_slice:output:0'forward_lstm_150/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros_1/mul
forward_lstm_150/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
forward_lstm_150/zeros_1/Less/yГ
forward_lstm_150/zeros_1/LessLess forward_lstm_150/zeros_1/mul:z:0(forward_lstm_150/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros_1/Less
!forward_lstm_150/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_150/zeros_1/packed/1Э
forward_lstm_150/zeros_1/packedPack'forward_lstm_150/strided_slice:output:0*forward_lstm_150/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_150/zeros_1/packed
forward_lstm_150/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_150/zeros_1/ConstС
forward_lstm_150/zeros_1Fill(forward_lstm_150/zeros_1/packed:output:0'forward_lstm_150/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/zeros_1
forward_lstm_150/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_150/transpose/permС
forward_lstm_150/transpose	Transposeinputs_0(forward_lstm_150/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
forward_lstm_150/transpose
forward_lstm_150/Shape_1Shapeforward_lstm_150/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_150/Shape_1
&forward_lstm_150/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_150/strided_slice_1/stack
(forward_lstm_150/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_1/stack_1
(forward_lstm_150/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_1/stack_2д
 forward_lstm_150/strided_slice_1StridedSlice!forward_lstm_150/Shape_1:output:0/forward_lstm_150/strided_slice_1/stack:output:01forward_lstm_150/strided_slice_1/stack_1:output:01forward_lstm_150/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_150/strided_slice_1Ї
,forward_lstm_150/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,forward_lstm_150/TensorArrayV2/element_shapeі
forward_lstm_150/TensorArrayV2TensorListReserve5forward_lstm_150/TensorArrayV2/element_shape:output:0)forward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_150/TensorArrayV2с
Fforward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2H
Fforward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8forward_lstm_150/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_150/transpose:y:0Oforward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_150/TensorArrayUnstack/TensorListFromTensor
&forward_lstm_150/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_150/strided_slice_2/stack
(forward_lstm_150/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_2/stack_1
(forward_lstm_150/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_2/stack_2ы
 forward_lstm_150/strided_slice_2StridedSliceforward_lstm_150/transpose:y:0/forward_lstm_150/strided_slice_2/stack:output:01forward_lstm_150/strided_slice_2/stack_1:output:01forward_lstm_150/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2"
 forward_lstm_150/strided_slice_2ы
4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpReadVariableOp=forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpє
%forward_lstm_150/lstm_cell_451/MatMulMatMul)forward_lstm_150/strided_slice_2:output:0<forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_150/lstm_cell_451/MatMulё
6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOp?forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp№
'forward_lstm_150/lstm_cell_451/MatMul_1MatMulforward_lstm_150/zeros:output:0>forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_150/lstm_cell_451/MatMul_1ш
"forward_lstm_150/lstm_cell_451/addAddV2/forward_lstm_150/lstm_cell_451/MatMul:product:01forward_lstm_150/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_150/lstm_cell_451/addъ
5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOp>forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpѕ
&forward_lstm_150/lstm_cell_451/BiasAddBiasAdd&forward_lstm_150/lstm_cell_451/add:z:0=forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_150/lstm_cell_451/BiasAddЂ
.forward_lstm_150/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.forward_lstm_150/lstm_cell_451/split/split_dimЛ
$forward_lstm_150/lstm_cell_451/splitSplit7forward_lstm_150/lstm_cell_451/split/split_dim:output:0/forward_lstm_150/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$forward_lstm_150/lstm_cell_451/splitМ
&forward_lstm_150/lstm_cell_451/SigmoidSigmoid-forward_lstm_150/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_150/lstm_cell_451/SigmoidР
(forward_lstm_150/lstm_cell_451/Sigmoid_1Sigmoid-forward_lstm_150/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_150/lstm_cell_451/Sigmoid_1в
"forward_lstm_150/lstm_cell_451/mulMul,forward_lstm_150/lstm_cell_451/Sigmoid_1:y:0!forward_lstm_150/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_150/lstm_cell_451/mulГ
#forward_lstm_150/lstm_cell_451/ReluRelu-forward_lstm_150/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_150/lstm_cell_451/Reluф
$forward_lstm_150/lstm_cell_451/mul_1Mul*forward_lstm_150/lstm_cell_451/Sigmoid:y:01forward_lstm_150/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_150/lstm_cell_451/mul_1й
$forward_lstm_150/lstm_cell_451/add_1AddV2&forward_lstm_150/lstm_cell_451/mul:z:0(forward_lstm_150/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_150/lstm_cell_451/add_1Р
(forward_lstm_150/lstm_cell_451/Sigmoid_2Sigmoid-forward_lstm_150/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_150/lstm_cell_451/Sigmoid_2В
%forward_lstm_150/lstm_cell_451/Relu_1Relu(forward_lstm_150/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_150/lstm_cell_451/Relu_1ш
$forward_lstm_150/lstm_cell_451/mul_2Mul,forward_lstm_150/lstm_cell_451/Sigmoid_2:y:03forward_lstm_150/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_150/lstm_cell_451/mul_2Б
.forward_lstm_150/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.forward_lstm_150/TensorArrayV2_1/element_shapeќ
 forward_lstm_150/TensorArrayV2_1TensorListReserve7forward_lstm_150/TensorArrayV2_1/element_shape:output:0)forward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_150/TensorArrayV2_1p
forward_lstm_150/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_150/timeЁ
)forward_lstm_150/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)forward_lstm_150/while/maximum_iterations
#forward_lstm_150/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_150/while/loop_counter
forward_lstm_150/whileWhile,forward_lstm_150/while/loop_counter:output:02forward_lstm_150/while/maximum_iterations:output:0forward_lstm_150/time:output:0)forward_lstm_150/TensorArrayV2_1:handle:0forward_lstm_150/zeros:output:0!forward_lstm_150/zeros_1:output:0)forward_lstm_150/strided_slice_1:output:0Hforward_lstm_150/TensorArrayUnstack/TensorListFromTensor:output_handle:0=forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource?forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource>forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource*
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
$forward_lstm_150_while_body_18420949*0
cond(R&
$forward_lstm_150_while_cond_18420948*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
forward_lstm_150/whileз
Aforward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Aforward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3forward_lstm_150/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_150/while:output:3Jforward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3forward_lstm_150/TensorArrayV2Stack/TensorListStackЃ
&forward_lstm_150/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&forward_lstm_150/strided_slice_3/stack
(forward_lstm_150/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_150/strided_slice_3/stack_1
(forward_lstm_150/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_3/stack_2
 forward_lstm_150/strided_slice_3StridedSlice<forward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_150/strided_slice_3/stack:output:01forward_lstm_150/strided_slice_3/stack_1:output:01forward_lstm_150/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 forward_lstm_150/strided_slice_3
!forward_lstm_150/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_150/transpose_1/permђ
forward_lstm_150/transpose_1	Transpose<forward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_150/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_150/transpose_1
forward_lstm_150/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_150/runtimej
backward_lstm_150/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_150/Shape
%backward_lstm_150/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_150/strided_slice/stack
'backward_lstm_150/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_150/strided_slice/stack_1
'backward_lstm_150/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_150/strided_slice/stack_2Ю
backward_lstm_150/strided_sliceStridedSlice backward_lstm_150/Shape:output:0.backward_lstm_150/strided_slice/stack:output:00backward_lstm_150/strided_slice/stack_1:output:00backward_lstm_150/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_150/strided_slice
backward_lstm_150/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_150/zeros/mul/yД
backward_lstm_150/zeros/mulMul(backward_lstm_150/strided_slice:output:0&backward_lstm_150/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/zeros/mul
backward_lstm_150/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_150/zeros/Less/yЏ
backward_lstm_150/zeros/LessLessbackward_lstm_150/zeros/mul:z:0'backward_lstm_150/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/zeros/Less
 backward_lstm_150/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_150/zeros/packed/1Ы
backward_lstm_150/zeros/packedPack(backward_lstm_150/strided_slice:output:0)backward_lstm_150/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_150/zeros/packed
backward_lstm_150/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_150/zeros/ConstН
backward_lstm_150/zerosFill'backward_lstm_150/zeros/packed:output:0&backward_lstm_150/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_150/zeros
backward_lstm_150/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_150/zeros_1/mul/yК
backward_lstm_150/zeros_1/mulMul(backward_lstm_150/strided_slice:output:0(backward_lstm_150/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/zeros_1/mul
 backward_lstm_150/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 backward_lstm_150/zeros_1/Less/yЗ
backward_lstm_150/zeros_1/LessLess!backward_lstm_150/zeros_1/mul:z:0)backward_lstm_150/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_150/zeros_1/Less
"backward_lstm_150/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_150/zeros_1/packed/1б
 backward_lstm_150/zeros_1/packedPack(backward_lstm_150/strided_slice:output:0+backward_lstm_150/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_150/zeros_1/packed
backward_lstm_150/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_150/zeros_1/ConstХ
backward_lstm_150/zeros_1Fill)backward_lstm_150/zeros_1/packed:output:0(backward_lstm_150/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_150/zeros_1
 backward_lstm_150/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_150/transpose/permФ
backward_lstm_150/transpose	Transposeinputs_0)backward_lstm_150/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_150/transpose
backward_lstm_150/Shape_1Shapebackward_lstm_150/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_150/Shape_1
'backward_lstm_150/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_150/strided_slice_1/stack 
)backward_lstm_150/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_1/stack_1 
)backward_lstm_150/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_1/stack_2к
!backward_lstm_150/strided_slice_1StridedSlice"backward_lstm_150/Shape_1:output:00backward_lstm_150/strided_slice_1/stack:output:02backward_lstm_150/strided_slice_1/stack_1:output:02backward_lstm_150/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_150/strided_slice_1Љ
-backward_lstm_150/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2/
-backward_lstm_150/TensorArrayV2/element_shapeњ
backward_lstm_150/TensorArrayV2TensorListReserve6backward_lstm_150/TensorArrayV2/element_shape:output:0*backward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_150/TensorArrayV2
 backward_lstm_150/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_150/ReverseV2/axisл
backward_lstm_150/ReverseV2	ReverseV2backward_lstm_150/transpose:y:0)backward_lstm_150/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_150/ReverseV2у
Gbackward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2I
Gbackward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeХ
9backward_lstm_150/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_150/ReverseV2:output:0Pbackward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_150/TensorArrayUnstack/TensorListFromTensor
'backward_lstm_150/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_150/strided_slice_2/stack 
)backward_lstm_150/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_2/stack_1 
)backward_lstm_150/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_2/stack_2ё
!backward_lstm_150/strided_slice_2StridedSlicebackward_lstm_150/transpose:y:00backward_lstm_150/strided_slice_2/stack:output:02backward_lstm_150/strided_slice_2/stack_1:output:02backward_lstm_150/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2#
!backward_lstm_150/strided_slice_2ю
5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpReadVariableOp>backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype027
5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpј
&backward_lstm_150/lstm_cell_452/MatMulMatMul*backward_lstm_150/strided_slice_2:output:0=backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_150/lstm_cell_452/MatMulє
7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOp@backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype029
7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpє
(backward_lstm_150/lstm_cell_452/MatMul_1MatMul backward_lstm_150/zeros:output:0?backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_150/lstm_cell_452/MatMul_1ь
#backward_lstm_150/lstm_cell_452/addAddV20backward_lstm_150/lstm_cell_452/MatMul:product:02backward_lstm_150/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_150/lstm_cell_452/addэ
6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOp?backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype028
6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpљ
'backward_lstm_150/lstm_cell_452/BiasAddBiasAdd'backward_lstm_150/lstm_cell_452/add:z:0>backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_150/lstm_cell_452/BiasAddЄ
/backward_lstm_150/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/backward_lstm_150/lstm_cell_452/split/split_dimП
%backward_lstm_150/lstm_cell_452/splitSplit8backward_lstm_150/lstm_cell_452/split/split_dim:output:00backward_lstm_150/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2'
%backward_lstm_150/lstm_cell_452/splitП
'backward_lstm_150/lstm_cell_452/SigmoidSigmoid.backward_lstm_150/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_150/lstm_cell_452/SigmoidУ
)backward_lstm_150/lstm_cell_452/Sigmoid_1Sigmoid.backward_lstm_150/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_150/lstm_cell_452/Sigmoid_1ж
#backward_lstm_150/lstm_cell_452/mulMul-backward_lstm_150/lstm_cell_452/Sigmoid_1:y:0"backward_lstm_150/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_150/lstm_cell_452/mulЖ
$backward_lstm_150/lstm_cell_452/ReluRelu.backward_lstm_150/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_150/lstm_cell_452/Reluш
%backward_lstm_150/lstm_cell_452/mul_1Mul+backward_lstm_150/lstm_cell_452/Sigmoid:y:02backward_lstm_150/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_150/lstm_cell_452/mul_1н
%backward_lstm_150/lstm_cell_452/add_1AddV2'backward_lstm_150/lstm_cell_452/mul:z:0)backward_lstm_150/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_150/lstm_cell_452/add_1У
)backward_lstm_150/lstm_cell_452/Sigmoid_2Sigmoid.backward_lstm_150/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_150/lstm_cell_452/Sigmoid_2Е
&backward_lstm_150/lstm_cell_452/Relu_1Relu)backward_lstm_150/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_150/lstm_cell_452/Relu_1ь
%backward_lstm_150/lstm_cell_452/mul_2Mul-backward_lstm_150/lstm_cell_452/Sigmoid_2:y:04backward_lstm_150/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_150/lstm_cell_452/mul_2Г
/backward_lstm_150/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   21
/backward_lstm_150/TensorArrayV2_1/element_shape
!backward_lstm_150/TensorArrayV2_1TensorListReserve8backward_lstm_150/TensorArrayV2_1/element_shape:output:0*backward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_150/TensorArrayV2_1r
backward_lstm_150/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_150/timeЃ
*backward_lstm_150/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*backward_lstm_150/while/maximum_iterations
$backward_lstm_150/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_150/while/loop_counter 
backward_lstm_150/whileWhile-backward_lstm_150/while/loop_counter:output:03backward_lstm_150/while/maximum_iterations:output:0backward_lstm_150/time:output:0*backward_lstm_150/TensorArrayV2_1:handle:0 backward_lstm_150/zeros:output:0"backward_lstm_150/zeros_1:output:0*backward_lstm_150/strided_slice_1:output:0Ibackward_lstm_150/TensorArrayUnstack/TensorListFromTensor:output_handle:0>backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource@backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource?backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource*
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
%backward_lstm_150_while_body_18421098*1
cond)R'
%backward_lstm_150_while_cond_18421097*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
backward_lstm_150/whileй
Bbackward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2D
Bbackward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeЙ
4backward_lstm_150/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_150/while:output:3Kbackward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype026
4backward_lstm_150/TensorArrayV2Stack/TensorListStackЅ
'backward_lstm_150/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2)
'backward_lstm_150/strided_slice_3/stack 
)backward_lstm_150/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_150/strided_slice_3/stack_1 
)backward_lstm_150/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_3/stack_2
!backward_lstm_150/strided_slice_3StridedSlice=backward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_150/strided_slice_3/stack:output:02backward_lstm_150/strided_slice_3/stack_1:output:02backward_lstm_150/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2#
!backward_lstm_150/strided_slice_3
"backward_lstm_150/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_150/transpose_1/permі
backward_lstm_150/transpose_1	Transpose=backward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_150/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_150/transpose_1
backward_lstm_150/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_150/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisФ
concatConcatV2)forward_lstm_150/strided_slice_3:output:0*backward_lstm_150/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp7^backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp6^backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp8^backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp^backward_lstm_150/while6^forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp5^forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp7^forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp^forward_lstm_150/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2p
6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp2n
5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp2r
7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp22
backward_lstm_150/whilebackward_lstm_150/while2n
5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp2l
4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp2p
6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp20
forward_lstm_150/whileforward_lstm_150/while:g c
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
п
Э
while_cond_18419621
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_18419621___redundant_placeholder06
2while_while_cond_18419621___redundant_placeholder16
2while_while_cond_18419621___redundant_placeholder26
2while_while_cond_18419621___redundant_placeholder3
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
X
ћ
$forward_lstm_150_while_body_18420949>
:forward_lstm_150_while_forward_lstm_150_while_loop_counterD
@forward_lstm_150_while_forward_lstm_150_while_maximum_iterations&
"forward_lstm_150_while_placeholder(
$forward_lstm_150_while_placeholder_1(
$forward_lstm_150_while_placeholder_2(
$forward_lstm_150_while_placeholder_3=
9forward_lstm_150_while_forward_lstm_150_strided_slice_1_0y
uforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0X
Eforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0:	ШZ
Gforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0:	2ШU
Fforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0:	Ш#
forward_lstm_150_while_identity%
!forward_lstm_150_while_identity_1%
!forward_lstm_150_while_identity_2%
!forward_lstm_150_while_identity_3%
!forward_lstm_150_while_identity_4%
!forward_lstm_150_while_identity_5;
7forward_lstm_150_while_forward_lstm_150_strided_slice_1w
sforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensorV
Cforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource:	ШX
Eforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource:	2ШS
Dforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource:	ШЂ;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOpЂ:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOpЂ<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOpх
Hforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2J
Hforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeТ
:forward_lstm_150/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_150_while_placeholderQforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02<
:forward_lstm_150/while/TensorArrayV2Read/TensorListGetItemџ
:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOpReadVariableOpEforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp
+forward_lstm_150/while/lstm_cell_451/MatMulMatMulAforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem:item:0Bforward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_150/while/lstm_cell_451/MatMul
<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOpGforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp
-forward_lstm_150/while/lstm_cell_451/MatMul_1MatMul$forward_lstm_150_while_placeholder_2Dforward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-forward_lstm_150/while/lstm_cell_451/MatMul_1
(forward_lstm_150/while/lstm_cell_451/addAddV25forward_lstm_150/while/lstm_cell_451/MatMul:product:07forward_lstm_150/while/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_150/while/lstm_cell_451/addў
;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOpFforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp
,forward_lstm_150/while/lstm_cell_451/BiasAddBiasAdd,forward_lstm_150/while/lstm_cell_451/add:z:0Cforward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_150/while/lstm_cell_451/BiasAddЎ
4forward_lstm_150/while/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4forward_lstm_150/while/lstm_cell_451/split/split_dimг
*forward_lstm_150/while/lstm_cell_451/splitSplit=forward_lstm_150/while/lstm_cell_451/split/split_dim:output:05forward_lstm_150/while/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*forward_lstm_150/while/lstm_cell_451/splitЮ
,forward_lstm_150/while/lstm_cell_451/SigmoidSigmoid3forward_lstm_150/while/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,forward_lstm_150/while/lstm_cell_451/Sigmoidв
.forward_lstm_150/while/lstm_cell_451/Sigmoid_1Sigmoid3forward_lstm_150/while/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_150/while/lstm_cell_451/Sigmoid_1ч
(forward_lstm_150/while/lstm_cell_451/mulMul2forward_lstm_150/while/lstm_cell_451/Sigmoid_1:y:0$forward_lstm_150_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_150/while/lstm_cell_451/mulХ
)forward_lstm_150/while/lstm_cell_451/ReluRelu3forward_lstm_150/while/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_150/while/lstm_cell_451/Reluќ
*forward_lstm_150/while/lstm_cell_451/mul_1Mul0forward_lstm_150/while/lstm_cell_451/Sigmoid:y:07forward_lstm_150/while/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_150/while/lstm_cell_451/mul_1ё
*forward_lstm_150/while/lstm_cell_451/add_1AddV2,forward_lstm_150/while/lstm_cell_451/mul:z:0.forward_lstm_150/while/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_150/while/lstm_cell_451/add_1в
.forward_lstm_150/while/lstm_cell_451/Sigmoid_2Sigmoid3forward_lstm_150/while/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_150/while/lstm_cell_451/Sigmoid_2Ф
+forward_lstm_150/while/lstm_cell_451/Relu_1Relu.forward_lstm_150/while/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_150/while/lstm_cell_451/Relu_1
*forward_lstm_150/while/lstm_cell_451/mul_2Mul2forward_lstm_150/while/lstm_cell_451/Sigmoid_2:y:09forward_lstm_150/while/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_150/while/lstm_cell_451/mul_2Ж
;forward_lstm_150/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_150_while_placeholder_1"forward_lstm_150_while_placeholder.forward_lstm_150/while/lstm_cell_451/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_150/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_150/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_150/while/add/y­
forward_lstm_150/while/addAddV2"forward_lstm_150_while_placeholder%forward_lstm_150/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/while/add
forward_lstm_150/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_150/while/add_1/yЫ
forward_lstm_150/while/add_1AddV2:forward_lstm_150_while_forward_lstm_150_while_loop_counter'forward_lstm_150/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/while/add_1Џ
forward_lstm_150/while/IdentityIdentity forward_lstm_150/while/add_1:z:0^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_150/while/Identityг
!forward_lstm_150/while/Identity_1Identity@forward_lstm_150_while_forward_lstm_150_while_maximum_iterations^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_150/while/Identity_1Б
!forward_lstm_150/while/Identity_2Identityforward_lstm_150/while/add:z:0^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_150/while/Identity_2о
!forward_lstm_150/while/Identity_3IdentityKforward_lstm_150/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_150/while/Identity_3в
!forward_lstm_150/while/Identity_4Identity.forward_lstm_150/while/lstm_cell_451/mul_2:z:0^forward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_150/while/Identity_4в
!forward_lstm_150/while/Identity_5Identity.forward_lstm_150/while/lstm_cell_451/add_1:z:0^forward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_150/while/Identity_5Ж
forward_lstm_150/while/NoOpNoOp<^forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp;^forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp=^forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_150/while/NoOp"t
7forward_lstm_150_while_forward_lstm_150_strided_slice_19forward_lstm_150_while_forward_lstm_150_strided_slice_1_0"K
forward_lstm_150_while_identity(forward_lstm_150/while/Identity:output:0"O
!forward_lstm_150_while_identity_1*forward_lstm_150/while/Identity_1:output:0"O
!forward_lstm_150_while_identity_2*forward_lstm_150/while/Identity_2:output:0"O
!forward_lstm_150_while_identity_3*forward_lstm_150/while/Identity_3:output:0"O
!forward_lstm_150_while_identity_4*forward_lstm_150/while/Identity_4:output:0"O
!forward_lstm_150_while_identity_5*forward_lstm_150/while/Identity_5:output:0"
Dforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resourceFforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0"
Eforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resourceGforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0"
Cforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resourceEforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0"ь
sforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensoruforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2z
;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp2x
:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp2|
<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_body_18419097
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_451_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_451_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_451_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_451_matmul_readvariableop_resource:	ШG
4while_lstm_cell_451_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_451_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_451/BiasAdd/ReadVariableOpЂ)while/lstm_cell_451/MatMul/ReadVariableOpЂ+while/lstm_cell_451/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_451/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_451_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_451/MatMul/ReadVariableOpк
while/lstm_cell_451/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/MatMulв
+while/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_451_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_451/MatMul_1/ReadVariableOpУ
while/lstm_cell_451/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/MatMul_1М
while/lstm_cell_451/addAddV2$while/lstm_cell_451/MatMul:product:0&while/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/addЫ
*while/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_451_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_451/BiasAdd/ReadVariableOpЩ
while/lstm_cell_451/BiasAddBiasAddwhile/lstm_cell_451/add:z:02while/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/BiasAdd
#while/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_451/split/split_dim
while/lstm_cell_451/splitSplit,while/lstm_cell_451/split/split_dim:output:0$while/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_451/split
while/lstm_cell_451/SigmoidSigmoid"while/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Sigmoid
while/lstm_cell_451/Sigmoid_1Sigmoid"while/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Sigmoid_1Ѓ
while/lstm_cell_451/mulMul!while/lstm_cell_451/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/mul
while/lstm_cell_451/ReluRelu"while/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/ReluИ
while/lstm_cell_451/mul_1Mulwhile/lstm_cell_451/Sigmoid:y:0&while/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/mul_1­
while/lstm_cell_451/add_1AddV2while/lstm_cell_451/mul:z:0while/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/add_1
while/lstm_cell_451/Sigmoid_2Sigmoid"while/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Sigmoid_2
while/lstm_cell_451/Relu_1Reluwhile/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Relu_1М
while/lstm_cell_451/mul_2Mul!while/lstm_cell_451/Sigmoid_2:y:0(while/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_451/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_451/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_451/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_451/BiasAdd/ReadVariableOp*^while/lstm_cell_451/MatMul/ReadVariableOp,^while/lstm_cell_451/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_451_biasadd_readvariableop_resource5while_lstm_cell_451_biasadd_readvariableop_resource_0"n
4while_lstm_cell_451_matmul_1_readvariableop_resource6while_lstm_cell_451_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_451_matmul_readvariableop_resource4while_lstm_cell_451_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_451/BiasAdd/ReadVariableOp*while/lstm_cell_451/BiasAdd/ReadVariableOp2V
)while/lstm_cell_451/MatMul/ReadVariableOp)while/lstm_cell_451/MatMul/ReadVariableOp2Z
+while/lstm_cell_451/MatMul_1/ReadVariableOp+while/lstm_cell_451/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_18419352

inputs,
forward_lstm_150_18419182:	Ш,
forward_lstm_150_18419184:	2Ш(
forward_lstm_150_18419186:	Ш-
backward_lstm_150_18419342:	Ш-
backward_lstm_150_18419344:	2Ш)
backward_lstm_150_18419346:	Ш
identityЂ)backward_lstm_150/StatefulPartitionedCallЂ(forward_lstm_150/StatefulPartitionedCallп
(forward_lstm_150/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_150_18419182forward_lstm_150_18419184forward_lstm_150_18419186*
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
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_184191812*
(forward_lstm_150/StatefulPartitionedCallх
)backward_lstm_150/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_150_18419342backward_lstm_150_18419344backward_lstm_150_18419346*
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
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_184193412+
)backward_lstm_150/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisд
concatConcatV21forward_lstm_150/StatefulPartitionedCall:output:02backward_lstm_150/StatefulPartitionedCall:output:0concat/axis:output:0*
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
NoOpNoOp*^backward_lstm_150/StatefulPartitionedCall)^forward_lstm_150/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2V
)backward_lstm_150/StatefulPartitionedCall)backward_lstm_150/StatefulPartitionedCall2T
(forward_lstm_150/StatefulPartitionedCall(forward_lstm_150/StatefulPartitionedCall:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
]
Ќ
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_18422719

inputs?
,lstm_cell_451_matmul_readvariableop_resource:	ШA
.lstm_cell_451_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_451_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_451/BiasAdd/ReadVariableOpЂ#lstm_cell_451/MatMul/ReadVariableOpЂ%lstm_cell_451/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_451/MatMul/ReadVariableOpReadVariableOp,lstm_cell_451_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_451/MatMul/ReadVariableOpА
lstm_cell_451/MatMulMatMulstrided_slice_2:output:0+lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/MatMulО
%lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_451_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_451/MatMul_1/ReadVariableOpЌ
lstm_cell_451/MatMul_1MatMulzeros:output:0-lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/MatMul_1Є
lstm_cell_451/addAddV2lstm_cell_451/MatMul:product:0 lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/addЗ
$lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_451_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_451/BiasAdd/ReadVariableOpБ
lstm_cell_451/BiasAddBiasAddlstm_cell_451/add:z:0,lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/BiasAdd
lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_451/split/split_dimї
lstm_cell_451/splitSplit&lstm_cell_451/split/split_dim:output:0lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_451/split
lstm_cell_451/SigmoidSigmoidlstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Sigmoid
lstm_cell_451/Sigmoid_1Sigmoidlstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Sigmoid_1
lstm_cell_451/mulMullstm_cell_451/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/mul
lstm_cell_451/ReluRelulstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Relu 
lstm_cell_451/mul_1Mullstm_cell_451/Sigmoid:y:0 lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/mul_1
lstm_cell_451/add_1AddV2lstm_cell_451/mul:z:0lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/add_1
lstm_cell_451/Sigmoid_2Sigmoidlstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Sigmoid_2
lstm_cell_451/Relu_1Relulstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Relu_1Є
lstm_cell_451/mul_2Mullstm_cell_451/Sigmoid_2:y:0"lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_451_matmul_readvariableop_resource.lstm_cell_451_matmul_1_readvariableop_resource-lstm_cell_451_biasadd_readvariableop_resource*
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
while_body_18422635*
condR
while_cond_18422634*K
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
NoOpNoOp%^lstm_cell_451/BiasAdd/ReadVariableOp$^lstm_cell_451/MatMul/ReadVariableOp&^lstm_cell_451/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_451/BiasAdd/ReadVariableOp$lstm_cell_451/BiasAdd/ReadVariableOp2J
#lstm_cell_451/MatMul/ReadVariableOp#lstm_cell_451/MatMul/ReadVariableOp2N
%lstm_cell_451/MatMul_1/ReadVariableOp%lstm_cell_451/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
№?
л
while_body_18423136
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_452_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_452_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_452_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_452_matmul_readvariableop_resource:	ШG
4while_lstm_cell_452_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_452_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_452/BiasAdd/ReadVariableOpЂ)while/lstm_cell_452/MatMul/ReadVariableOpЂ+while/lstm_cell_452/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_452/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_452_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_452/MatMul/ReadVariableOpк
while/lstm_cell_452/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/MatMulв
+while/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_452_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_452/MatMul_1/ReadVariableOpУ
while/lstm_cell_452/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/MatMul_1М
while/lstm_cell_452/addAddV2$while/lstm_cell_452/MatMul:product:0&while/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/addЫ
*while/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_452_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_452/BiasAdd/ReadVariableOpЩ
while/lstm_cell_452/BiasAddBiasAddwhile/lstm_cell_452/add:z:02while/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/BiasAdd
#while/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_452/split/split_dim
while/lstm_cell_452/splitSplit,while/lstm_cell_452/split/split_dim:output:0$while/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_452/split
while/lstm_cell_452/SigmoidSigmoid"while/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Sigmoid
while/lstm_cell_452/Sigmoid_1Sigmoid"while/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Sigmoid_1Ѓ
while/lstm_cell_452/mulMul!while/lstm_cell_452/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/mul
while/lstm_cell_452/ReluRelu"while/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/ReluИ
while/lstm_cell_452/mul_1Mulwhile/lstm_cell_452/Sigmoid:y:0&while/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/mul_1­
while/lstm_cell_452/add_1AddV2while/lstm_cell_452/mul:z:0while/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/add_1
while/lstm_cell_452/Sigmoid_2Sigmoid"while/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Sigmoid_2
while/lstm_cell_452/Relu_1Reluwhile/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Relu_1М
while/lstm_cell_452/mul_2Mul!while/lstm_cell_452/Sigmoid_2:y:0(while/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_452/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_452/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_452/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_452/BiasAdd/ReadVariableOp*^while/lstm_cell_452/MatMul/ReadVariableOp,^while/lstm_cell_452/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_452_biasadd_readvariableop_resource5while_lstm_cell_452_biasadd_readvariableop_resource_0"n
4while_lstm_cell_452_matmul_1_readvariableop_resource6while_lstm_cell_452_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_452_matmul_readvariableop_resource4while_lstm_cell_452_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_452/BiasAdd/ReadVariableOp*while/lstm_cell_452/BiasAdd/ReadVariableOp2V
)while/lstm_cell_452/MatMul/ReadVariableOp)while/lstm_cell_452/MatMul/ReadVariableOp2Z
+while/lstm_cell_452/MatMul_1/ReadVariableOp+while/lstm_cell_452/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
K__inference_lstm_cell_451_layer_call_and_return_conditional_losses_18423624

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
І_
­
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_18423526

inputs?
,lstm_cell_452_matmul_readvariableop_resource:	ШA
.lstm_cell_452_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_452_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_452/BiasAdd/ReadVariableOpЂ#lstm_cell_452/MatMul/ReadVariableOpЂ%lstm_cell_452/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_452/MatMul/ReadVariableOpReadVariableOp,lstm_cell_452_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_452/MatMul/ReadVariableOpА
lstm_cell_452/MatMulMatMulstrided_slice_2:output:0+lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/MatMulО
%lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_452_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_452/MatMul_1/ReadVariableOpЌ
lstm_cell_452/MatMul_1MatMulzeros:output:0-lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/MatMul_1Є
lstm_cell_452/addAddV2lstm_cell_452/MatMul:product:0 lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/addЗ
$lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_452_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_452/BiasAdd/ReadVariableOpБ
lstm_cell_452/BiasAddBiasAddlstm_cell_452/add:z:0,lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/BiasAdd
lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_452/split/split_dimї
lstm_cell_452/splitSplit&lstm_cell_452/split/split_dim:output:0lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_452/split
lstm_cell_452/SigmoidSigmoidlstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Sigmoid
lstm_cell_452/Sigmoid_1Sigmoidlstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Sigmoid_1
lstm_cell_452/mulMullstm_cell_452/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/mul
lstm_cell_452/ReluRelulstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Relu 
lstm_cell_452/mul_1Mullstm_cell_452/Sigmoid:y:0 lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/mul_1
lstm_cell_452/add_1AddV2lstm_cell_452/mul:z:0lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/add_1
lstm_cell_452/Sigmoid_2Sigmoidlstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Sigmoid_2
lstm_cell_452/Relu_1Relulstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Relu_1Є
lstm_cell_452/mul_2Mullstm_cell_452/Sigmoid_2:y:0"lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_452_matmul_readvariableop_resource.lstm_cell_452_matmul_1_readvariableop_resource-lstm_cell_452_biasadd_readvariableop_resource*
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
while_body_18423442*
condR
while_cond_18423441*K
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
NoOpNoOp%^lstm_cell_452/BiasAdd/ReadVariableOp$^lstm_cell_452/MatMul/ReadVariableOp&^lstm_cell_452/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_452/BiasAdd/ReadVariableOp$lstm_cell_452/BiasAdd/ReadVariableOp2J
#lstm_cell_452/MatMul/ReadVariableOp#lstm_cell_452/MatMul/ReadVariableOp2N
%lstm_cell_452/MatMul_1/ReadVariableOp%lstm_cell_452/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
љ?
л
while_body_18422786
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_451_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_451_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_451_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_451_matmul_readvariableop_resource:	ШG
4while_lstm_cell_451_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_451_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_451/BiasAdd/ReadVariableOpЂ)while/lstm_cell_451/MatMul/ReadVariableOpЂ+while/lstm_cell_451/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_451/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_451_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_451/MatMul/ReadVariableOpк
while/lstm_cell_451/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/MatMulв
+while/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_451_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_451/MatMul_1/ReadVariableOpУ
while/lstm_cell_451/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/MatMul_1М
while/lstm_cell_451/addAddV2$while/lstm_cell_451/MatMul:product:0&while/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/addЫ
*while/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_451_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_451/BiasAdd/ReadVariableOpЩ
while/lstm_cell_451/BiasAddBiasAddwhile/lstm_cell_451/add:z:02while/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/BiasAdd
#while/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_451/split/split_dim
while/lstm_cell_451/splitSplit,while/lstm_cell_451/split/split_dim:output:0$while/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_451/split
while/lstm_cell_451/SigmoidSigmoid"while/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Sigmoid
while/lstm_cell_451/Sigmoid_1Sigmoid"while/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Sigmoid_1Ѓ
while/lstm_cell_451/mulMul!while/lstm_cell_451/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/mul
while/lstm_cell_451/ReluRelu"while/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/ReluИ
while/lstm_cell_451/mul_1Mulwhile/lstm_cell_451/Sigmoid:y:0&while/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/mul_1­
while/lstm_cell_451/add_1AddV2while/lstm_cell_451/mul:z:0while/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/add_1
while/lstm_cell_451/Sigmoid_2Sigmoid"while/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Sigmoid_2
while/lstm_cell_451/Relu_1Reluwhile/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Relu_1М
while/lstm_cell_451/mul_2Mul!while/lstm_cell_451/Sigmoid_2:y:0(while/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_451/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_451/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_451/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_451/BiasAdd/ReadVariableOp*^while/lstm_cell_451/MatMul/ReadVariableOp,^while/lstm_cell_451/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_451_biasadd_readvariableop_resource5while_lstm_cell_451_biasadd_readvariableop_resource_0"n
4while_lstm_cell_451_matmul_1_readvariableop_resource6while_lstm_cell_451_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_451_matmul_readvariableop_resource4while_lstm_cell_451_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_451/BiasAdd/ReadVariableOp*while/lstm_cell_451/BiasAdd/ReadVariableOp2V
)while/lstm_cell_451/MatMul/ReadVariableOp)while/lstm_cell_451/MatMul/ReadVariableOp2Z
+while/lstm_cell_451/MatMul_1/ReadVariableOp+while/lstm_cell_451/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
K__inference_lstm_cell_451_layer_call_and_return_conditional_losses_18423592

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
ќ

и
1__inference_sequential_150_layer_call_fn_18420736

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
L__inference_sequential_150_layer_call_and_return_conditional_losses_184206952
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

д
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_18419754

inputs,
forward_lstm_150_18419737:	Ш,
forward_lstm_150_18419739:	2Ш(
forward_lstm_150_18419741:	Ш-
backward_lstm_150_18419744:	Ш-
backward_lstm_150_18419746:	2Ш)
backward_lstm_150_18419748:	Ш
identityЂ)backward_lstm_150/StatefulPartitionedCallЂ(forward_lstm_150/StatefulPartitionedCallп
(forward_lstm_150/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_150_18419737forward_lstm_150_18419739forward_lstm_150_18419741*
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
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_184197062*
(forward_lstm_150/StatefulPartitionedCallх
)backward_lstm_150/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_150_18419744backward_lstm_150_18419746backward_lstm_150_18419748*
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
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_184195332+
)backward_lstm_150/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisд
concatConcatV21forward_lstm_150/StatefulPartitionedCall:output:02backward_lstm_150/StatefulPartitionedCall:output:0concat/axis:output:0*
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
NoOpNoOp*^backward_lstm_150/StatefulPartitionedCall)^forward_lstm_150/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2V
)backward_lstm_150/StatefulPartitionedCall)backward_lstm_150/StatefulPartitionedCall2T
(forward_lstm_150/StatefulPartitionedCall(forward_lstm_150/StatefulPartitionedCall:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
М
љ
0__inference_lstm_cell_451_layer_call_fn_18423560

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
K__inference_lstm_cell_451_layer_call_and_return_conditional_losses_184179772
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
р
Р
3__inference_forward_lstm_150_layer_call_fn_18422266

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
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_184197062
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
жF

N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_18417914

inputs)
lstm_cell_451_18417832:	Ш)
lstm_cell_451_18417834:	2Ш%
lstm_cell_451_18417836:	Ш
identityЂ%lstm_cell_451/StatefulPartitionedCallЂwhileD
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
%lstm_cell_451/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_451_18417832lstm_cell_451_18417834lstm_cell_451_18417836*
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
K__inference_lstm_cell_451_layer_call_and_return_conditional_losses_184178312'
%lstm_cell_451/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_451_18417832lstm_cell_451_18417834lstm_cell_451_18417836*
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
while_body_18417845*
condR
while_cond_18417844*K
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
NoOpNoOp&^lstm_cell_451/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2N
%lstm_cell_451/StatefulPartitionedCall%lstm_cell_451/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
]
Ў
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_18422568
inputs_0?
,lstm_cell_451_matmul_readvariableop_resource:	ШA
.lstm_cell_451_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_451_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_451/BiasAdd/ReadVariableOpЂ#lstm_cell_451/MatMul/ReadVariableOpЂ%lstm_cell_451/MatMul_1/ReadVariableOpЂwhileF
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
#lstm_cell_451/MatMul/ReadVariableOpReadVariableOp,lstm_cell_451_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_451/MatMul/ReadVariableOpА
lstm_cell_451/MatMulMatMulstrided_slice_2:output:0+lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/MatMulО
%lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_451_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_451/MatMul_1/ReadVariableOpЌ
lstm_cell_451/MatMul_1MatMulzeros:output:0-lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/MatMul_1Є
lstm_cell_451/addAddV2lstm_cell_451/MatMul:product:0 lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/addЗ
$lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_451_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_451/BiasAdd/ReadVariableOpБ
lstm_cell_451/BiasAddBiasAddlstm_cell_451/add:z:0,lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/BiasAdd
lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_451/split/split_dimї
lstm_cell_451/splitSplit&lstm_cell_451/split/split_dim:output:0lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_451/split
lstm_cell_451/SigmoidSigmoidlstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Sigmoid
lstm_cell_451/Sigmoid_1Sigmoidlstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Sigmoid_1
lstm_cell_451/mulMullstm_cell_451/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/mul
lstm_cell_451/ReluRelulstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Relu 
lstm_cell_451/mul_1Mullstm_cell_451/Sigmoid:y:0 lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/mul_1
lstm_cell_451/add_1AddV2lstm_cell_451/mul:z:0lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/add_1
lstm_cell_451/Sigmoid_2Sigmoidlstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Sigmoid_2
lstm_cell_451/Relu_1Relulstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Relu_1Є
lstm_cell_451/mul_2Mullstm_cell_451/Sigmoid_2:y:0"lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_451_matmul_readvariableop_resource.lstm_cell_451_matmul_1_readvariableop_resource-lstm_cell_451_biasadd_readvariableop_resource*
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
while_body_18422484*
condR
while_cond_18422483*K
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
NoOpNoOp%^lstm_cell_451/BiasAdd/ReadVariableOp$^lstm_cell_451/MatMul/ReadVariableOp&^lstm_cell_451/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_451/BiasAdd/ReadVariableOp$lstm_cell_451/BiasAdd/ReadVariableOp2J
#lstm_cell_451/MatMul/ReadVariableOp#lstm_cell_451/MatMul/ReadVariableOp2N
%lstm_cell_451/MatMul_1/ReadVariableOp%lstm_cell_451/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
X
ћ
$forward_lstm_150_while_body_18421251>
:forward_lstm_150_while_forward_lstm_150_while_loop_counterD
@forward_lstm_150_while_forward_lstm_150_while_maximum_iterations&
"forward_lstm_150_while_placeholder(
$forward_lstm_150_while_placeholder_1(
$forward_lstm_150_while_placeholder_2(
$forward_lstm_150_while_placeholder_3=
9forward_lstm_150_while_forward_lstm_150_strided_slice_1_0y
uforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0X
Eforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0:	ШZ
Gforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0:	2ШU
Fforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0:	Ш#
forward_lstm_150_while_identity%
!forward_lstm_150_while_identity_1%
!forward_lstm_150_while_identity_2%
!forward_lstm_150_while_identity_3%
!forward_lstm_150_while_identity_4%
!forward_lstm_150_while_identity_5;
7forward_lstm_150_while_forward_lstm_150_strided_slice_1w
sforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensorV
Cforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource:	ШX
Eforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource:	2ШS
Dforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource:	ШЂ;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOpЂ:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOpЂ<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOpх
Hforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2J
Hforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeТ
:forward_lstm_150/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_150_while_placeholderQforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02<
:forward_lstm_150/while/TensorArrayV2Read/TensorListGetItemџ
:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOpReadVariableOpEforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp
+forward_lstm_150/while/lstm_cell_451/MatMulMatMulAforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem:item:0Bforward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_150/while/lstm_cell_451/MatMul
<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOpGforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp
-forward_lstm_150/while/lstm_cell_451/MatMul_1MatMul$forward_lstm_150_while_placeholder_2Dforward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-forward_lstm_150/while/lstm_cell_451/MatMul_1
(forward_lstm_150/while/lstm_cell_451/addAddV25forward_lstm_150/while/lstm_cell_451/MatMul:product:07forward_lstm_150/while/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_150/while/lstm_cell_451/addў
;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOpFforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp
,forward_lstm_150/while/lstm_cell_451/BiasAddBiasAdd,forward_lstm_150/while/lstm_cell_451/add:z:0Cforward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_150/while/lstm_cell_451/BiasAddЎ
4forward_lstm_150/while/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4forward_lstm_150/while/lstm_cell_451/split/split_dimг
*forward_lstm_150/while/lstm_cell_451/splitSplit=forward_lstm_150/while/lstm_cell_451/split/split_dim:output:05forward_lstm_150/while/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*forward_lstm_150/while/lstm_cell_451/splitЮ
,forward_lstm_150/while/lstm_cell_451/SigmoidSigmoid3forward_lstm_150/while/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,forward_lstm_150/while/lstm_cell_451/Sigmoidв
.forward_lstm_150/while/lstm_cell_451/Sigmoid_1Sigmoid3forward_lstm_150/while/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_150/while/lstm_cell_451/Sigmoid_1ч
(forward_lstm_150/while/lstm_cell_451/mulMul2forward_lstm_150/while/lstm_cell_451/Sigmoid_1:y:0$forward_lstm_150_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_150/while/lstm_cell_451/mulХ
)forward_lstm_150/while/lstm_cell_451/ReluRelu3forward_lstm_150/while/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_150/while/lstm_cell_451/Reluќ
*forward_lstm_150/while/lstm_cell_451/mul_1Mul0forward_lstm_150/while/lstm_cell_451/Sigmoid:y:07forward_lstm_150/while/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_150/while/lstm_cell_451/mul_1ё
*forward_lstm_150/while/lstm_cell_451/add_1AddV2,forward_lstm_150/while/lstm_cell_451/mul:z:0.forward_lstm_150/while/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_150/while/lstm_cell_451/add_1в
.forward_lstm_150/while/lstm_cell_451/Sigmoid_2Sigmoid3forward_lstm_150/while/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_150/while/lstm_cell_451/Sigmoid_2Ф
+forward_lstm_150/while/lstm_cell_451/Relu_1Relu.forward_lstm_150/while/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_150/while/lstm_cell_451/Relu_1
*forward_lstm_150/while/lstm_cell_451/mul_2Mul2forward_lstm_150/while/lstm_cell_451/Sigmoid_2:y:09forward_lstm_150/while/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_150/while/lstm_cell_451/mul_2Ж
;forward_lstm_150/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_150_while_placeholder_1"forward_lstm_150_while_placeholder.forward_lstm_150/while/lstm_cell_451/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_150/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_150/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_150/while/add/y­
forward_lstm_150/while/addAddV2"forward_lstm_150_while_placeholder%forward_lstm_150/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/while/add
forward_lstm_150/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_150/while/add_1/yЫ
forward_lstm_150/while/add_1AddV2:forward_lstm_150_while_forward_lstm_150_while_loop_counter'forward_lstm_150/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/while/add_1Џ
forward_lstm_150/while/IdentityIdentity forward_lstm_150/while/add_1:z:0^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_150/while/Identityг
!forward_lstm_150/while/Identity_1Identity@forward_lstm_150_while_forward_lstm_150_while_maximum_iterations^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_150/while/Identity_1Б
!forward_lstm_150/while/Identity_2Identityforward_lstm_150/while/add:z:0^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_150/while/Identity_2о
!forward_lstm_150/while/Identity_3IdentityKforward_lstm_150/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_150/while/Identity_3в
!forward_lstm_150/while/Identity_4Identity.forward_lstm_150/while/lstm_cell_451/mul_2:z:0^forward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_150/while/Identity_4в
!forward_lstm_150/while/Identity_5Identity.forward_lstm_150/while/lstm_cell_451/add_1:z:0^forward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_150/while/Identity_5Ж
forward_lstm_150/while/NoOpNoOp<^forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp;^forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp=^forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_150/while/NoOp"t
7forward_lstm_150_while_forward_lstm_150_strided_slice_19forward_lstm_150_while_forward_lstm_150_strided_slice_1_0"K
forward_lstm_150_while_identity(forward_lstm_150/while/Identity:output:0"O
!forward_lstm_150_while_identity_1*forward_lstm_150/while/Identity_1:output:0"O
!forward_lstm_150_while_identity_2*forward_lstm_150/while/Identity_2:output:0"O
!forward_lstm_150_while_identity_3*forward_lstm_150/while/Identity_3:output:0"O
!forward_lstm_150_while_identity_4*forward_lstm_150/while/Identity_4:output:0"O
!forward_lstm_150_while_identity_5*forward_lstm_150/while/Identity_5:output:0"
Dforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resourceFforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0"
Eforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resourceGforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0"
Cforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resourceEforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0"ь
sforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensoruforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2z
;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp2x
:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp2|
<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Ў

Ѕ
4__inference_bidirectional_150_layer_call_fn_18420882

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
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_184206322
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
№?
л
while_body_18422983
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_452_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_452_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_452_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_452_matmul_readvariableop_resource:	ШG
4while_lstm_cell_452_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_452_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_452/BiasAdd/ReadVariableOpЂ)while/lstm_cell_452/MatMul/ReadVariableOpЂ+while/lstm_cell_452/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_452/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_452_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_452/MatMul/ReadVariableOpк
while/lstm_cell_452/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/MatMulв
+while/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_452_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_452/MatMul_1/ReadVariableOpУ
while/lstm_cell_452/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/MatMul_1М
while/lstm_cell_452/addAddV2$while/lstm_cell_452/MatMul:product:0&while/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/addЫ
*while/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_452_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_452/BiasAdd/ReadVariableOpЩ
while/lstm_cell_452/BiasAddBiasAddwhile/lstm_cell_452/add:z:02while/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/BiasAdd
#while/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_452/split/split_dim
while/lstm_cell_452/splitSplit,while/lstm_cell_452/split/split_dim:output:0$while/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_452/split
while/lstm_cell_452/SigmoidSigmoid"while/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Sigmoid
while/lstm_cell_452/Sigmoid_1Sigmoid"while/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Sigmoid_1Ѓ
while/lstm_cell_452/mulMul!while/lstm_cell_452/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/mul
while/lstm_cell_452/ReluRelu"while/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/ReluИ
while/lstm_cell_452/mul_1Mulwhile/lstm_cell_452/Sigmoid:y:0&while/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/mul_1­
while/lstm_cell_452/add_1AddV2while/lstm_cell_452/mul:z:0while/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/add_1
while/lstm_cell_452/Sigmoid_2Sigmoid"while/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Sigmoid_2
while/lstm_cell_452/Relu_1Reluwhile/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Relu_1М
while/lstm_cell_452/mul_2Mul!while/lstm_cell_452/Sigmoid_2:y:0(while/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_452/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_452/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_452/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_452/BiasAdd/ReadVariableOp*^while/lstm_cell_452/MatMul/ReadVariableOp,^while/lstm_cell_452/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_452_biasadd_readvariableop_resource5while_lstm_cell_452_biasadd_readvariableop_resource_0"n
4while_lstm_cell_452_matmul_1_readvariableop_resource6while_lstm_cell_452_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_452_matmul_readvariableop_resource4while_lstm_cell_452_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_452/BiasAdd/ReadVariableOp*while/lstm_cell_452/BiasAdd/ReadVariableOp2V
)while/lstm_cell_452/MatMul/ReadVariableOp)while/lstm_cell_452/MatMul/ReadVariableOp2Z
+while/lstm_cell_452/MatMul_1/ReadVariableOp+while/lstm_cell_452/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
№?
л
while_body_18422333
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_451_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_451_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_451_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_451_matmul_readvariableop_resource:	ШG
4while_lstm_cell_451_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_451_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_451/BiasAdd/ReadVariableOpЂ)while/lstm_cell_451/MatMul/ReadVariableOpЂ+while/lstm_cell_451/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_451/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_451_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_451/MatMul/ReadVariableOpк
while/lstm_cell_451/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/MatMulв
+while/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_451_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_451/MatMul_1/ReadVariableOpУ
while/lstm_cell_451/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/MatMul_1М
while/lstm_cell_451/addAddV2$while/lstm_cell_451/MatMul:product:0&while/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/addЫ
*while/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_451_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_451/BiasAdd/ReadVariableOpЩ
while/lstm_cell_451/BiasAddBiasAddwhile/lstm_cell_451/add:z:02while/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/BiasAdd
#while/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_451/split/split_dim
while/lstm_cell_451/splitSplit,while/lstm_cell_451/split/split_dim:output:0$while/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_451/split
while/lstm_cell_451/SigmoidSigmoid"while/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Sigmoid
while/lstm_cell_451/Sigmoid_1Sigmoid"while/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Sigmoid_1Ѓ
while/lstm_cell_451/mulMul!while/lstm_cell_451/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/mul
while/lstm_cell_451/ReluRelu"while/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/ReluИ
while/lstm_cell_451/mul_1Mulwhile/lstm_cell_451/Sigmoid:y:0&while/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/mul_1­
while/lstm_cell_451/add_1AddV2while/lstm_cell_451/mul:z:0while/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/add_1
while/lstm_cell_451/Sigmoid_2Sigmoid"while/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Sigmoid_2
while/lstm_cell_451/Relu_1Reluwhile/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Relu_1М
while/lstm_cell_451/mul_2Mul!while/lstm_cell_451/Sigmoid_2:y:0(while/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_451/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_451/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_451/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_451/BiasAdd/ReadVariableOp*^while/lstm_cell_451/MatMul/ReadVariableOp,^while/lstm_cell_451/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_451_biasadd_readvariableop_resource5while_lstm_cell_451_biasadd_readvariableop_resource_0"n
4while_lstm_cell_451_matmul_1_readvariableop_resource6while_lstm_cell_451_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_451_matmul_readvariableop_resource4while_lstm_cell_451_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_451/BiasAdd/ReadVariableOp*while/lstm_cell_451/BiasAdd/ReadVariableOp2V
)while/lstm_cell_451/MatMul/ReadVariableOp)while/lstm_cell_451/MatMul/ReadVariableOp2Z
+while/lstm_cell_451/MatMul_1/ReadVariableOp+while/lstm_cell_451/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_cond_18422785
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_18422785___redundant_placeholder06
2while_while_cond_18422785___redundant_placeholder16
2while_while_cond_18422785___redundant_placeholder26
2while_while_cond_18422785___redundant_placeholder3
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
while_body_18423289
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_452_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_452_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_452_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_452_matmul_readvariableop_resource:	ШG
4while_lstm_cell_452_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_452_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_452/BiasAdd/ReadVariableOpЂ)while/lstm_cell_452/MatMul/ReadVariableOpЂ+while/lstm_cell_452/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_452/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_452_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_452/MatMul/ReadVariableOpк
while/lstm_cell_452/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/MatMulв
+while/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_452_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_452/MatMul_1/ReadVariableOpУ
while/lstm_cell_452/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/MatMul_1М
while/lstm_cell_452/addAddV2$while/lstm_cell_452/MatMul:product:0&while/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/addЫ
*while/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_452_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_452/BiasAdd/ReadVariableOpЩ
while/lstm_cell_452/BiasAddBiasAddwhile/lstm_cell_452/add:z:02while/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/BiasAdd
#while/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_452/split/split_dim
while/lstm_cell_452/splitSplit,while/lstm_cell_452/split/split_dim:output:0$while/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_452/split
while/lstm_cell_452/SigmoidSigmoid"while/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Sigmoid
while/lstm_cell_452/Sigmoid_1Sigmoid"while/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Sigmoid_1Ѓ
while/lstm_cell_452/mulMul!while/lstm_cell_452/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/mul
while/lstm_cell_452/ReluRelu"while/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/ReluИ
while/lstm_cell_452/mul_1Mulwhile/lstm_cell_452/Sigmoid:y:0&while/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/mul_1­
while/lstm_cell_452/add_1AddV2while/lstm_cell_452/mul:z:0while/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/add_1
while/lstm_cell_452/Sigmoid_2Sigmoid"while/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Sigmoid_2
while/lstm_cell_452/Relu_1Reluwhile/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Relu_1М
while/lstm_cell_452/mul_2Mul!while/lstm_cell_452/Sigmoid_2:y:0(while/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_452/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_452/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_452/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_452/BiasAdd/ReadVariableOp*^while/lstm_cell_452/MatMul/ReadVariableOp,^while/lstm_cell_452/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_452_biasadd_readvariableop_resource5while_lstm_cell_452_biasadd_readvariableop_resource_0"n
4while_lstm_cell_452_matmul_1_readvariableop_resource6while_lstm_cell_452_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_452_matmul_readvariableop_resource4while_lstm_cell_452_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_452/BiasAdd/ReadVariableOp*while/lstm_cell_452/BiasAdd/ReadVariableOp2V
)while/lstm_cell_452/MatMul/ReadVariableOp)while/lstm_cell_452/MatMul/ReadVariableOp2Z
+while/lstm_cell_452/MatMul_1/ReadVariableOp+while/lstm_cell_452/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
ы	

4__inference_bidirectional_150_layer_call_fn_18420829
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
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_184193522
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
&
ј
while_body_18418055
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_451_18418079_0:	Ш1
while_lstm_cell_451_18418081_0:	2Ш-
while_lstm_cell_451_18418083_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_451_18418079:	Ш/
while_lstm_cell_451_18418081:	2Ш+
while_lstm_cell_451_18418083:	ШЂ+while/lstm_cell_451/StatefulPartitionedCallУ
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
+while/lstm_cell_451/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_451_18418079_0while_lstm_cell_451_18418081_0while_lstm_cell_451_18418083_0*
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
K__inference_lstm_cell_451_layer_call_and_return_conditional_losses_184179772-
+while/lstm_cell_451/StatefulPartitionedCallј
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_451/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_451/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѕ
while/Identity_5Identity4while/lstm_cell_451/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp,^while/lstm_cell_451/StatefulPartitionedCall*"
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
while_lstm_cell_451_18418079while_lstm_cell_451_18418079_0">
while_lstm_cell_451_18418081while_lstm_cell_451_18418081_0">
while_lstm_cell_451_18418083while_lstm_cell_451_18418083_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2Z
+while/lstm_cell_451/StatefulPartitionedCall+while/lstm_cell_451/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
while_cond_18423135
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_18423135___redundant_placeholder06
2while_while_cond_18423135___redundant_placeholder16
2while_while_cond_18423135___redundant_placeholder26
2while_while_cond_18423135___redundant_placeholder3
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
Щ
Ѕ
$forward_lstm_150_while_cond_18419915>
:forward_lstm_150_while_forward_lstm_150_while_loop_counterD
@forward_lstm_150_while_forward_lstm_150_while_maximum_iterations&
"forward_lstm_150_while_placeholder(
$forward_lstm_150_while_placeholder_1(
$forward_lstm_150_while_placeholder_2(
$forward_lstm_150_while_placeholder_3(
$forward_lstm_150_while_placeholder_4@
<forward_lstm_150_while_less_forward_lstm_150_strided_slice_1X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18419915___redundant_placeholder0X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18419915___redundant_placeholder1X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18419915___redundant_placeholder2X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18419915___redundant_placeholder3X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18419915___redundant_placeholder4#
forward_lstm_150_while_identity
Х
forward_lstm_150/while/LessLess"forward_lstm_150_while_placeholder<forward_lstm_150_while_less_forward_lstm_150_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_150/while/Less
forward_lstm_150/while/IdentityIdentityforward_lstm_150/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_150/while/Identity"K
forward_lstm_150_while_identity(forward_lstm_150/while/Identity:output:0*(
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
љ?
л
while_body_18419257
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_452_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_452_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_452_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_452_matmul_readvariableop_resource:	ШG
4while_lstm_cell_452_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_452_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_452/BiasAdd/ReadVariableOpЂ)while/lstm_cell_452/MatMul/ReadVariableOpЂ+while/lstm_cell_452/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_452/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_452_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_452/MatMul/ReadVariableOpк
while/lstm_cell_452/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/MatMulв
+while/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_452_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_452/MatMul_1/ReadVariableOpУ
while/lstm_cell_452/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/MatMul_1М
while/lstm_cell_452/addAddV2$while/lstm_cell_452/MatMul:product:0&while/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/addЫ
*while/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_452_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_452/BiasAdd/ReadVariableOpЩ
while/lstm_cell_452/BiasAddBiasAddwhile/lstm_cell_452/add:z:02while/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/BiasAdd
#while/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_452/split/split_dim
while/lstm_cell_452/splitSplit,while/lstm_cell_452/split/split_dim:output:0$while/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_452/split
while/lstm_cell_452/SigmoidSigmoid"while/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Sigmoid
while/lstm_cell_452/Sigmoid_1Sigmoid"while/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Sigmoid_1Ѓ
while/lstm_cell_452/mulMul!while/lstm_cell_452/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/mul
while/lstm_cell_452/ReluRelu"while/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/ReluИ
while/lstm_cell_452/mul_1Mulwhile/lstm_cell_452/Sigmoid:y:0&while/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/mul_1­
while/lstm_cell_452/add_1AddV2while/lstm_cell_452/mul:z:0while/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/add_1
while/lstm_cell_452/Sigmoid_2Sigmoid"while/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Sigmoid_2
while/lstm_cell_452/Relu_1Reluwhile/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Relu_1М
while/lstm_cell_452/mul_2Mul!while/lstm_cell_452/Sigmoid_2:y:0(while/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_452/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_452/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_452/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_452/BiasAdd/ReadVariableOp*^while/lstm_cell_452/MatMul/ReadVariableOp,^while/lstm_cell_452/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_452_biasadd_readvariableop_resource5while_lstm_cell_452_biasadd_readvariableop_resource_0"n
4while_lstm_cell_452_matmul_1_readvariableop_resource6while_lstm_cell_452_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_452_matmul_readvariableop_resource4while_lstm_cell_452_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_452/BiasAdd/ReadVariableOp*while/lstm_cell_452/BiasAdd/ReadVariableOp2V
)while/lstm_cell_452/MatMul/ReadVariableOp)while/lstm_cell_452/MatMul/ReadVariableOp2Z
+while/lstm_cell_452/MatMul_1/ReadVariableOp+while/lstm_cell_452/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_18419181

inputs?
,lstm_cell_451_matmul_readvariableop_resource:	ШA
.lstm_cell_451_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_451_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_451/BiasAdd/ReadVariableOpЂ#lstm_cell_451/MatMul/ReadVariableOpЂ%lstm_cell_451/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_451/MatMul/ReadVariableOpReadVariableOp,lstm_cell_451_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_451/MatMul/ReadVariableOpА
lstm_cell_451/MatMulMatMulstrided_slice_2:output:0+lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/MatMulО
%lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_451_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_451/MatMul_1/ReadVariableOpЌ
lstm_cell_451/MatMul_1MatMulzeros:output:0-lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/MatMul_1Є
lstm_cell_451/addAddV2lstm_cell_451/MatMul:product:0 lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/addЗ
$lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_451_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_451/BiasAdd/ReadVariableOpБ
lstm_cell_451/BiasAddBiasAddlstm_cell_451/add:z:0,lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_451/BiasAdd
lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_451/split/split_dimї
lstm_cell_451/splitSplit&lstm_cell_451/split/split_dim:output:0lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_451/split
lstm_cell_451/SigmoidSigmoidlstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Sigmoid
lstm_cell_451/Sigmoid_1Sigmoidlstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Sigmoid_1
lstm_cell_451/mulMullstm_cell_451/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/mul
lstm_cell_451/ReluRelulstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Relu 
lstm_cell_451/mul_1Mullstm_cell_451/Sigmoid:y:0 lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/mul_1
lstm_cell_451/add_1AddV2lstm_cell_451/mul:z:0lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/add_1
lstm_cell_451/Sigmoid_2Sigmoidlstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Sigmoid_2
lstm_cell_451/Relu_1Relulstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/Relu_1Є
lstm_cell_451/mul_2Mullstm_cell_451/Sigmoid_2:y:0"lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_451/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_451_matmul_readvariableop_resource.lstm_cell_451_matmul_1_readvariableop_resource-lstm_cell_451_biasadd_readvariableop_resource*
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
while_body_18419097*
condR
while_cond_18419096*K
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
NoOpNoOp%^lstm_cell_451/BiasAdd/ReadVariableOp$^lstm_cell_451/MatMul/ReadVariableOp&^lstm_cell_451/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_451/BiasAdd/ReadVariableOp$lstm_cell_451/BiasAdd/ReadVariableOp2J
#lstm_cell_451/MatMul/ReadVariableOp#lstm_cell_451/MatMul/ReadVariableOp2N
%lstm_cell_451/MatMul_1/ReadVariableOp%lstm_cell_451/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ї

,__inference_dense_150_layer_call_fn_18422211

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
G__inference_dense_150_layer_call_and_return_conditional_losses_184202172
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
п
Э
while_cond_18423441
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_18423441___redundant_placeholder06
2while_while_cond_18423441___redundant_placeholder16
2while_while_cond_18423441___redundant_placeholder26
2while_while_cond_18423441___redundant_placeholder3
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
і

K__inference_lstm_cell_452_layer_call_and_return_conditional_losses_18418463

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
&
ј
while_body_18417845
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_451_18417869_0:	Ш1
while_lstm_cell_451_18417871_0:	2Ш-
while_lstm_cell_451_18417873_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_451_18417869:	Ш/
while_lstm_cell_451_18417871:	2Ш+
while_lstm_cell_451_18417873:	ШЂ+while/lstm_cell_451/StatefulPartitionedCallУ
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
+while/lstm_cell_451/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_451_18417869_0while_lstm_cell_451_18417871_0while_lstm_cell_451_18417873_0*
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
K__inference_lstm_cell_451_layer_call_and_return_conditional_losses_184178312-
+while/lstm_cell_451/StatefulPartitionedCallј
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_451/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_451/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѕ
while/Identity_5Identity4while/lstm_cell_451/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp,^while/lstm_cell_451/StatefulPartitionedCall*"
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
while_lstm_cell_451_18417869while_lstm_cell_451_18417869_0">
while_lstm_cell_451_18417871while_lstm_cell_451_18417871_0">
while_lstm_cell_451_18417873while_lstm_cell_451_18417873_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2Z
+while/lstm_cell_451/StatefulPartitionedCall+while/lstm_cell_451/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
$forward_lstm_150_while_body_18419916>
:forward_lstm_150_while_forward_lstm_150_while_loop_counterD
@forward_lstm_150_while_forward_lstm_150_while_maximum_iterations&
"forward_lstm_150_while_placeholder(
$forward_lstm_150_while_placeholder_1(
$forward_lstm_150_while_placeholder_2(
$forward_lstm_150_while_placeholder_3(
$forward_lstm_150_while_placeholder_4=
9forward_lstm_150_while_forward_lstm_150_strided_slice_1_0y
uforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_150_while_greater_forward_lstm_150_cast_0X
Eforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0:	ШZ
Gforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0:	2ШU
Fforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0:	Ш#
forward_lstm_150_while_identity%
!forward_lstm_150_while_identity_1%
!forward_lstm_150_while_identity_2%
!forward_lstm_150_while_identity_3%
!forward_lstm_150_while_identity_4%
!forward_lstm_150_while_identity_5%
!forward_lstm_150_while_identity_6;
7forward_lstm_150_while_forward_lstm_150_strided_slice_1w
sforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_150_while_greater_forward_lstm_150_castV
Cforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource:	ШX
Eforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource:	2ШS
Dforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource:	ШЂ;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOpЂ:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOpЂ<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOpх
Hforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2J
Hforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
:forward_lstm_150/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_150_while_placeholderQforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02<
:forward_lstm_150/while/TensorArrayV2Read/TensorListGetItemе
forward_lstm_150/while/GreaterGreater6forward_lstm_150_while_greater_forward_lstm_150_cast_0"forward_lstm_150_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2 
forward_lstm_150/while/Greaterџ
:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOpReadVariableOpEforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp
+forward_lstm_150/while/lstm_cell_451/MatMulMatMulAforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem:item:0Bforward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_150/while/lstm_cell_451/MatMul
<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOpGforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp
-forward_lstm_150/while/lstm_cell_451/MatMul_1MatMul$forward_lstm_150_while_placeholder_3Dforward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-forward_lstm_150/while/lstm_cell_451/MatMul_1
(forward_lstm_150/while/lstm_cell_451/addAddV25forward_lstm_150/while/lstm_cell_451/MatMul:product:07forward_lstm_150/while/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_150/while/lstm_cell_451/addў
;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOpFforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp
,forward_lstm_150/while/lstm_cell_451/BiasAddBiasAdd,forward_lstm_150/while/lstm_cell_451/add:z:0Cforward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_150/while/lstm_cell_451/BiasAddЎ
4forward_lstm_150/while/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4forward_lstm_150/while/lstm_cell_451/split/split_dimг
*forward_lstm_150/while/lstm_cell_451/splitSplit=forward_lstm_150/while/lstm_cell_451/split/split_dim:output:05forward_lstm_150/while/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*forward_lstm_150/while/lstm_cell_451/splitЮ
,forward_lstm_150/while/lstm_cell_451/SigmoidSigmoid3forward_lstm_150/while/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,forward_lstm_150/while/lstm_cell_451/Sigmoidв
.forward_lstm_150/while/lstm_cell_451/Sigmoid_1Sigmoid3forward_lstm_150/while/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_150/while/lstm_cell_451/Sigmoid_1ч
(forward_lstm_150/while/lstm_cell_451/mulMul2forward_lstm_150/while/lstm_cell_451/Sigmoid_1:y:0$forward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_150/while/lstm_cell_451/mulХ
)forward_lstm_150/while/lstm_cell_451/ReluRelu3forward_lstm_150/while/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_150/while/lstm_cell_451/Reluќ
*forward_lstm_150/while/lstm_cell_451/mul_1Mul0forward_lstm_150/while/lstm_cell_451/Sigmoid:y:07forward_lstm_150/while/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_150/while/lstm_cell_451/mul_1ё
*forward_lstm_150/while/lstm_cell_451/add_1AddV2,forward_lstm_150/while/lstm_cell_451/mul:z:0.forward_lstm_150/while/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_150/while/lstm_cell_451/add_1в
.forward_lstm_150/while/lstm_cell_451/Sigmoid_2Sigmoid3forward_lstm_150/while/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_150/while/lstm_cell_451/Sigmoid_2Ф
+forward_lstm_150/while/lstm_cell_451/Relu_1Relu.forward_lstm_150/while/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_150/while/lstm_cell_451/Relu_1
*forward_lstm_150/while/lstm_cell_451/mul_2Mul2forward_lstm_150/while/lstm_cell_451/Sigmoid_2:y:09forward_lstm_150/while/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_150/while/lstm_cell_451/mul_2є
forward_lstm_150/while/SelectSelect"forward_lstm_150/while/Greater:z:0.forward_lstm_150/while/lstm_cell_451/mul_2:z:0$forward_lstm_150_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/while/Selectј
forward_lstm_150/while/Select_1Select"forward_lstm_150/while/Greater:z:0.forward_lstm_150/while/lstm_cell_451/mul_2:z:0$forward_lstm_150_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_150/while/Select_1ј
forward_lstm_150/while/Select_2Select"forward_lstm_150/while/Greater:z:0.forward_lstm_150/while/lstm_cell_451/add_1:z:0$forward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_150/while/Select_2Ў
;forward_lstm_150/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_150_while_placeholder_1"forward_lstm_150_while_placeholder&forward_lstm_150/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_150/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_150/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_150/while/add/y­
forward_lstm_150/while/addAddV2"forward_lstm_150_while_placeholder%forward_lstm_150/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/while/add
forward_lstm_150/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_150/while/add_1/yЫ
forward_lstm_150/while/add_1AddV2:forward_lstm_150_while_forward_lstm_150_while_loop_counter'forward_lstm_150/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/while/add_1Џ
forward_lstm_150/while/IdentityIdentity forward_lstm_150/while/add_1:z:0^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_150/while/Identityг
!forward_lstm_150/while/Identity_1Identity@forward_lstm_150_while_forward_lstm_150_while_maximum_iterations^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_150/while/Identity_1Б
!forward_lstm_150/while/Identity_2Identityforward_lstm_150/while/add:z:0^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_150/while/Identity_2о
!forward_lstm_150/while/Identity_3IdentityKforward_lstm_150/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_150/while/Identity_3Ъ
!forward_lstm_150/while/Identity_4Identity&forward_lstm_150/while/Select:output:0^forward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_150/while/Identity_4Ь
!forward_lstm_150/while/Identity_5Identity(forward_lstm_150/while/Select_1:output:0^forward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_150/while/Identity_5Ь
!forward_lstm_150/while/Identity_6Identity(forward_lstm_150/while/Select_2:output:0^forward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_150/while/Identity_6Ж
forward_lstm_150/while/NoOpNoOp<^forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp;^forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp=^forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_150/while/NoOp"t
7forward_lstm_150_while_forward_lstm_150_strided_slice_19forward_lstm_150_while_forward_lstm_150_strided_slice_1_0"n
4forward_lstm_150_while_greater_forward_lstm_150_cast6forward_lstm_150_while_greater_forward_lstm_150_cast_0"K
forward_lstm_150_while_identity(forward_lstm_150/while/Identity:output:0"O
!forward_lstm_150_while_identity_1*forward_lstm_150/while/Identity_1:output:0"O
!forward_lstm_150_while_identity_2*forward_lstm_150/while/Identity_2:output:0"O
!forward_lstm_150_while_identity_3*forward_lstm_150/while/Identity_3:output:0"O
!forward_lstm_150_while_identity_4*forward_lstm_150/while/Identity_4:output:0"O
!forward_lstm_150_while_identity_5*forward_lstm_150/while/Identity_5:output:0"O
!forward_lstm_150_while_identity_6*forward_lstm_150/while/Identity_6:output:0"
Dforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resourceFforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0"
Eforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resourceGforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0"
Cforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resourceEforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0"ь
sforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensoruforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2z
;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp2x
:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp2|
<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
ЯY

%backward_lstm_150_while_body_18421098@
<backward_lstm_150_while_backward_lstm_150_while_loop_counterF
Bbackward_lstm_150_while_backward_lstm_150_while_maximum_iterations'
#backward_lstm_150_while_placeholder)
%backward_lstm_150_while_placeholder_1)
%backward_lstm_150_while_placeholder_2)
%backward_lstm_150_while_placeholder_3?
;backward_lstm_150_while_backward_lstm_150_strided_slice_1_0{
wbackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0Y
Fbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0:	Ш[
Hbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0:	2ШV
Gbackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0:	Ш$
 backward_lstm_150_while_identity&
"backward_lstm_150_while_identity_1&
"backward_lstm_150_while_identity_2&
"backward_lstm_150_while_identity_3&
"backward_lstm_150_while_identity_4&
"backward_lstm_150_while_identity_5=
9backward_lstm_150_while_backward_lstm_150_strided_slice_1y
ubackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensorW
Dbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource:	ШY
Fbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource:	2ШT
Ebackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource:	ШЂ<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOpЂ;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpЂ=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOpч
Ibackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2K
Ibackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeШ
;backward_lstm_150/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_150_while_placeholderRbackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02=
;backward_lstm_150/while/TensorArrayV2Read/TensorListGetItem
;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpReadVariableOpFbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02=
;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpЂ
,backward_lstm_150/while/lstm_cell_452/MatMulMatMulBbackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem:item:0Cbackward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_150/while/lstm_cell_452/MatMul
=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOpHbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02?
=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp
.backward_lstm_150/while/lstm_cell_452/MatMul_1MatMul%backward_lstm_150_while_placeholder_2Ebackward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ20
.backward_lstm_150/while/lstm_cell_452/MatMul_1
)backward_lstm_150/while/lstm_cell_452/addAddV26backward_lstm_150/while/lstm_cell_452/MatMul:product:08backward_lstm_150/while/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_150/while/lstm_cell_452/add
<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOpGbackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02>
<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp
-backward_lstm_150/while/lstm_cell_452/BiasAddBiasAdd-backward_lstm_150/while/lstm_cell_452/add:z:0Dbackward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_150/while/lstm_cell_452/BiasAddА
5backward_lstm_150/while/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5backward_lstm_150/while/lstm_cell_452/split/split_dimз
+backward_lstm_150/while/lstm_cell_452/splitSplit>backward_lstm_150/while/lstm_cell_452/split/split_dim:output:06backward_lstm_150/while/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2-
+backward_lstm_150/while/lstm_cell_452/splitб
-backward_lstm_150/while/lstm_cell_452/SigmoidSigmoid4backward_lstm_150/while/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22/
-backward_lstm_150/while/lstm_cell_452/Sigmoidе
/backward_lstm_150/while/lstm_cell_452/Sigmoid_1Sigmoid4backward_lstm_150/while/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_150/while/lstm_cell_452/Sigmoid_1ы
)backward_lstm_150/while/lstm_cell_452/mulMul3backward_lstm_150/while/lstm_cell_452/Sigmoid_1:y:0%backward_lstm_150_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_150/while/lstm_cell_452/mulШ
*backward_lstm_150/while/lstm_cell_452/ReluRelu4backward_lstm_150/while/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_150/while/lstm_cell_452/Relu
+backward_lstm_150/while/lstm_cell_452/mul_1Mul1backward_lstm_150/while/lstm_cell_452/Sigmoid:y:08backward_lstm_150/while/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_150/while/lstm_cell_452/mul_1ѕ
+backward_lstm_150/while/lstm_cell_452/add_1AddV2-backward_lstm_150/while/lstm_cell_452/mul:z:0/backward_lstm_150/while/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_150/while/lstm_cell_452/add_1е
/backward_lstm_150/while/lstm_cell_452/Sigmoid_2Sigmoid4backward_lstm_150/while/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_150/while/lstm_cell_452/Sigmoid_2Ч
,backward_lstm_150/while/lstm_cell_452/Relu_1Relu/backward_lstm_150/while/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_150/while/lstm_cell_452/Relu_1
+backward_lstm_150/while/lstm_cell_452/mul_2Mul3backward_lstm_150/while/lstm_cell_452/Sigmoid_2:y:0:backward_lstm_150/while/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_150/while/lstm_cell_452/mul_2Л
<backward_lstm_150/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_150_while_placeholder_1#backward_lstm_150_while_placeholder/backward_lstm_150/while/lstm_cell_452/mul_2:z:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_150/while/TensorArrayV2Write/TensorListSetItem
backward_lstm_150/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_150/while/add/yБ
backward_lstm_150/while/addAddV2#backward_lstm_150_while_placeholder&backward_lstm_150/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/while/add
backward_lstm_150/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_150/while/add_1/yа
backward_lstm_150/while/add_1AddV2<backward_lstm_150_while_backward_lstm_150_while_loop_counter(backward_lstm_150/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/while/add_1Г
 backward_lstm_150/while/IdentityIdentity!backward_lstm_150/while/add_1:z:0^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_150/while/Identityи
"backward_lstm_150/while/Identity_1IdentityBbackward_lstm_150_while_backward_lstm_150_while_maximum_iterations^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_150/while/Identity_1Е
"backward_lstm_150/while/Identity_2Identitybackward_lstm_150/while/add:z:0^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_150/while/Identity_2т
"backward_lstm_150/while/Identity_3IdentityLbackward_lstm_150/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_150/while/Identity_3ж
"backward_lstm_150/while/Identity_4Identity/backward_lstm_150/while/lstm_cell_452/mul_2:z:0^backward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_150/while/Identity_4ж
"backward_lstm_150/while/Identity_5Identity/backward_lstm_150/while/lstm_cell_452/add_1:z:0^backward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_150/while/Identity_5Л
backward_lstm_150/while/NoOpNoOp=^backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp<^backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp>^backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_150/while/NoOp"x
9backward_lstm_150_while_backward_lstm_150_strided_slice_1;backward_lstm_150_while_backward_lstm_150_strided_slice_1_0"M
 backward_lstm_150_while_identity)backward_lstm_150/while/Identity:output:0"Q
"backward_lstm_150_while_identity_1+backward_lstm_150/while/Identity_1:output:0"Q
"backward_lstm_150_while_identity_2+backward_lstm_150/while/Identity_2:output:0"Q
"backward_lstm_150_while_identity_3+backward_lstm_150/while/Identity_3:output:0"Q
"backward_lstm_150_while_identity_4+backward_lstm_150/while/Identity_4:output:0"Q
"backward_lstm_150_while_identity_5+backward_lstm_150/while/Identity_5:output:0"
Ebackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resourceGbackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0"
Fbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resourceHbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0"
Dbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resourceFbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0"№
ubackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2|
<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp2z
;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp2~
=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_body_18422635
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_451_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_451_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_451_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_451_matmul_readvariableop_resource:	ШG
4while_lstm_cell_451_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_451_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_451/BiasAdd/ReadVariableOpЂ)while/lstm_cell_451/MatMul/ReadVariableOpЂ+while/lstm_cell_451/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_451/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_451_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_451/MatMul/ReadVariableOpк
while/lstm_cell_451/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/MatMulв
+while/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_451_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_451/MatMul_1/ReadVariableOpУ
while/lstm_cell_451/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/MatMul_1М
while/lstm_cell_451/addAddV2$while/lstm_cell_451/MatMul:product:0&while/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/addЫ
*while/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_451_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_451/BiasAdd/ReadVariableOpЩ
while/lstm_cell_451/BiasAddBiasAddwhile/lstm_cell_451/add:z:02while/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/BiasAdd
#while/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_451/split/split_dim
while/lstm_cell_451/splitSplit,while/lstm_cell_451/split/split_dim:output:0$while/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_451/split
while/lstm_cell_451/SigmoidSigmoid"while/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Sigmoid
while/lstm_cell_451/Sigmoid_1Sigmoid"while/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Sigmoid_1Ѓ
while/lstm_cell_451/mulMul!while/lstm_cell_451/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/mul
while/lstm_cell_451/ReluRelu"while/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/ReluИ
while/lstm_cell_451/mul_1Mulwhile/lstm_cell_451/Sigmoid:y:0&while/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/mul_1­
while/lstm_cell_451/add_1AddV2while/lstm_cell_451/mul:z:0while/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/add_1
while/lstm_cell_451/Sigmoid_2Sigmoid"while/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Sigmoid_2
while/lstm_cell_451/Relu_1Reluwhile/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Relu_1М
while/lstm_cell_451/mul_2Mul!while/lstm_cell_451/Sigmoid_2:y:0(while/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_451/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_451/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_451/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_451/BiasAdd/ReadVariableOp*^while/lstm_cell_451/MatMul/ReadVariableOp,^while/lstm_cell_451/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_451_biasadd_readvariableop_resource5while_lstm_cell_451_biasadd_readvariableop_resource_0"n
4while_lstm_cell_451_matmul_1_readvariableop_resource6while_lstm_cell_451_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_451_matmul_readvariableop_resource4while_lstm_cell_451_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_451/BiasAdd/ReadVariableOp*while/lstm_cell_451/BiasAdd/ReadVariableOp2V
)while/lstm_cell_451/MatMul/ReadVariableOp)while/lstm_cell_451/MatMul/ReadVariableOp2Z
+while/lstm_cell_451/MatMul_1/ReadVariableOp+while/lstm_cell_451/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Ў

Ѕ
4__inference_bidirectional_150_layer_call_fn_18420864

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
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_184201922
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
№?
л
while_body_18422484
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_451_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_451_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_451_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_451_matmul_readvariableop_resource:	ШG
4while_lstm_cell_451_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_451_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_451/BiasAdd/ReadVariableOpЂ)while/lstm_cell_451/MatMul/ReadVariableOpЂ+while/lstm_cell_451/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_451/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_451_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_451/MatMul/ReadVariableOpк
while/lstm_cell_451/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/MatMulв
+while/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_451_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_451/MatMul_1/ReadVariableOpУ
while/lstm_cell_451/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/MatMul_1М
while/lstm_cell_451/addAddV2$while/lstm_cell_451/MatMul:product:0&while/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/addЫ
*while/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_451_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_451/BiasAdd/ReadVariableOpЩ
while/lstm_cell_451/BiasAddBiasAddwhile/lstm_cell_451/add:z:02while/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_451/BiasAdd
#while/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_451/split/split_dim
while/lstm_cell_451/splitSplit,while/lstm_cell_451/split/split_dim:output:0$while/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_451/split
while/lstm_cell_451/SigmoidSigmoid"while/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Sigmoid
while/lstm_cell_451/Sigmoid_1Sigmoid"while/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Sigmoid_1Ѓ
while/lstm_cell_451/mulMul!while/lstm_cell_451/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/mul
while/lstm_cell_451/ReluRelu"while/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/ReluИ
while/lstm_cell_451/mul_1Mulwhile/lstm_cell_451/Sigmoid:y:0&while/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/mul_1­
while/lstm_cell_451/add_1AddV2while/lstm_cell_451/mul:z:0while/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/add_1
while/lstm_cell_451/Sigmoid_2Sigmoid"while/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Sigmoid_2
while/lstm_cell_451/Relu_1Reluwhile/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/Relu_1М
while/lstm_cell_451/mul_2Mul!while/lstm_cell_451/Sigmoid_2:y:0(while/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_451/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_451/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_451/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_451/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_451/BiasAdd/ReadVariableOp*^while/lstm_cell_451/MatMul/ReadVariableOp,^while/lstm_cell_451/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_451_biasadd_readvariableop_resource5while_lstm_cell_451_biasadd_readvariableop_resource_0"n
4while_lstm_cell_451_matmul_1_readvariableop_resource6while_lstm_cell_451_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_451_matmul_readvariableop_resource4while_lstm_cell_451_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_451/BiasAdd/ReadVariableOp*while/lstm_cell_451/BiasAdd/ReadVariableOp2V
)while/lstm_cell_451/MatMul/ReadVariableOp)while/lstm_cell_451/MatMul/ReadVariableOp2Z
+while/lstm_cell_451/MatMul_1/ReadVariableOp+while/lstm_cell_451/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
%backward_lstm_150_while_body_18421400@
<backward_lstm_150_while_backward_lstm_150_while_loop_counterF
Bbackward_lstm_150_while_backward_lstm_150_while_maximum_iterations'
#backward_lstm_150_while_placeholder)
%backward_lstm_150_while_placeholder_1)
%backward_lstm_150_while_placeholder_2)
%backward_lstm_150_while_placeholder_3?
;backward_lstm_150_while_backward_lstm_150_strided_slice_1_0{
wbackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0Y
Fbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0:	Ш[
Hbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0:	2ШV
Gbackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0:	Ш$
 backward_lstm_150_while_identity&
"backward_lstm_150_while_identity_1&
"backward_lstm_150_while_identity_2&
"backward_lstm_150_while_identity_3&
"backward_lstm_150_while_identity_4&
"backward_lstm_150_while_identity_5=
9backward_lstm_150_while_backward_lstm_150_strided_slice_1y
ubackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensorW
Dbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource:	ШY
Fbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource:	2ШT
Ebackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource:	ШЂ<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOpЂ;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpЂ=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOpч
Ibackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2K
Ibackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeШ
;backward_lstm_150/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_150_while_placeholderRbackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02=
;backward_lstm_150/while/TensorArrayV2Read/TensorListGetItem
;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpReadVariableOpFbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02=
;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpЂ
,backward_lstm_150/while/lstm_cell_452/MatMulMatMulBbackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem:item:0Cbackward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_150/while/lstm_cell_452/MatMul
=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOpHbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02?
=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp
.backward_lstm_150/while/lstm_cell_452/MatMul_1MatMul%backward_lstm_150_while_placeholder_2Ebackward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ20
.backward_lstm_150/while/lstm_cell_452/MatMul_1
)backward_lstm_150/while/lstm_cell_452/addAddV26backward_lstm_150/while/lstm_cell_452/MatMul:product:08backward_lstm_150/while/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_150/while/lstm_cell_452/add
<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOpGbackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02>
<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp
-backward_lstm_150/while/lstm_cell_452/BiasAddBiasAdd-backward_lstm_150/while/lstm_cell_452/add:z:0Dbackward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_150/while/lstm_cell_452/BiasAddА
5backward_lstm_150/while/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5backward_lstm_150/while/lstm_cell_452/split/split_dimз
+backward_lstm_150/while/lstm_cell_452/splitSplit>backward_lstm_150/while/lstm_cell_452/split/split_dim:output:06backward_lstm_150/while/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2-
+backward_lstm_150/while/lstm_cell_452/splitб
-backward_lstm_150/while/lstm_cell_452/SigmoidSigmoid4backward_lstm_150/while/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22/
-backward_lstm_150/while/lstm_cell_452/Sigmoidе
/backward_lstm_150/while/lstm_cell_452/Sigmoid_1Sigmoid4backward_lstm_150/while/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_150/while/lstm_cell_452/Sigmoid_1ы
)backward_lstm_150/while/lstm_cell_452/mulMul3backward_lstm_150/while/lstm_cell_452/Sigmoid_1:y:0%backward_lstm_150_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_150/while/lstm_cell_452/mulШ
*backward_lstm_150/while/lstm_cell_452/ReluRelu4backward_lstm_150/while/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_150/while/lstm_cell_452/Relu
+backward_lstm_150/while/lstm_cell_452/mul_1Mul1backward_lstm_150/while/lstm_cell_452/Sigmoid:y:08backward_lstm_150/while/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_150/while/lstm_cell_452/mul_1ѕ
+backward_lstm_150/while/lstm_cell_452/add_1AddV2-backward_lstm_150/while/lstm_cell_452/mul:z:0/backward_lstm_150/while/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_150/while/lstm_cell_452/add_1е
/backward_lstm_150/while/lstm_cell_452/Sigmoid_2Sigmoid4backward_lstm_150/while/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_150/while/lstm_cell_452/Sigmoid_2Ч
,backward_lstm_150/while/lstm_cell_452/Relu_1Relu/backward_lstm_150/while/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_150/while/lstm_cell_452/Relu_1
+backward_lstm_150/while/lstm_cell_452/mul_2Mul3backward_lstm_150/while/lstm_cell_452/Sigmoid_2:y:0:backward_lstm_150/while/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_150/while/lstm_cell_452/mul_2Л
<backward_lstm_150/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_150_while_placeholder_1#backward_lstm_150_while_placeholder/backward_lstm_150/while/lstm_cell_452/mul_2:z:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_150/while/TensorArrayV2Write/TensorListSetItem
backward_lstm_150/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_150/while/add/yБ
backward_lstm_150/while/addAddV2#backward_lstm_150_while_placeholder&backward_lstm_150/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/while/add
backward_lstm_150/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_150/while/add_1/yа
backward_lstm_150/while/add_1AddV2<backward_lstm_150_while_backward_lstm_150_while_loop_counter(backward_lstm_150/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/while/add_1Г
 backward_lstm_150/while/IdentityIdentity!backward_lstm_150/while/add_1:z:0^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_150/while/Identityи
"backward_lstm_150/while/Identity_1IdentityBbackward_lstm_150_while_backward_lstm_150_while_maximum_iterations^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_150/while/Identity_1Е
"backward_lstm_150/while/Identity_2Identitybackward_lstm_150/while/add:z:0^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_150/while/Identity_2т
"backward_lstm_150/while/Identity_3IdentityLbackward_lstm_150/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_150/while/Identity_3ж
"backward_lstm_150/while/Identity_4Identity/backward_lstm_150/while/lstm_cell_452/mul_2:z:0^backward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_150/while/Identity_4ж
"backward_lstm_150/while/Identity_5Identity/backward_lstm_150/while/lstm_cell_452/add_1:z:0^backward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_150/while/Identity_5Л
backward_lstm_150/while/NoOpNoOp=^backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp<^backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp>^backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_150/while/NoOp"x
9backward_lstm_150_while_backward_lstm_150_strided_slice_1;backward_lstm_150_while_backward_lstm_150_strided_slice_1_0"M
 backward_lstm_150_while_identity)backward_lstm_150/while/Identity:output:0"Q
"backward_lstm_150_while_identity_1+backward_lstm_150/while/Identity_1:output:0"Q
"backward_lstm_150_while_identity_2+backward_lstm_150/while/Identity_2:output:0"Q
"backward_lstm_150_while_identity_3+backward_lstm_150/while/Identity_3:output:0"Q
"backward_lstm_150_while_identity_4+backward_lstm_150/while/Identity_4:output:0"Q
"backward_lstm_150_while_identity_5+backward_lstm_150/while/Identity_5:output:0"
Ebackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resourceGbackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0"
Fbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resourceHbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0"
Dbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resourceFbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0"№
ubackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2|
<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp2z
;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp2~
=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Ф§
у
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_18421486
inputs_0P
=forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource:	ШR
?forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource:	2ШM
>forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource:	ШQ
>backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource:	ШS
@backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource:	2ШN
?backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource:	Ш
identityЂ6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpЂ5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpЂ7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpЂbackward_lstm_150/whileЂ5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpЂ4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpЂ6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOpЂforward_lstm_150/whileh
forward_lstm_150/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_150/Shape
$forward_lstm_150/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_150/strided_slice/stack
&forward_lstm_150/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_150/strided_slice/stack_1
&forward_lstm_150/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_150/strided_slice/stack_2Ш
forward_lstm_150/strided_sliceStridedSliceforward_lstm_150/Shape:output:0-forward_lstm_150/strided_slice/stack:output:0/forward_lstm_150/strided_slice/stack_1:output:0/forward_lstm_150/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_150/strided_slice~
forward_lstm_150/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_150/zeros/mul/yА
forward_lstm_150/zeros/mulMul'forward_lstm_150/strided_slice:output:0%forward_lstm_150/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros/mul
forward_lstm_150/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_150/zeros/Less/yЋ
forward_lstm_150/zeros/LessLessforward_lstm_150/zeros/mul:z:0&forward_lstm_150/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros/Less
forward_lstm_150/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_150/zeros/packed/1Ч
forward_lstm_150/zeros/packedPack'forward_lstm_150/strided_slice:output:0(forward_lstm_150/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_150/zeros/packed
forward_lstm_150/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_150/zeros/ConstЙ
forward_lstm_150/zerosFill&forward_lstm_150/zeros/packed:output:0%forward_lstm_150/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/zeros
forward_lstm_150/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_150/zeros_1/mul/yЖ
forward_lstm_150/zeros_1/mulMul'forward_lstm_150/strided_slice:output:0'forward_lstm_150/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros_1/mul
forward_lstm_150/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
forward_lstm_150/zeros_1/Less/yГ
forward_lstm_150/zeros_1/LessLess forward_lstm_150/zeros_1/mul:z:0(forward_lstm_150/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros_1/Less
!forward_lstm_150/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_150/zeros_1/packed/1Э
forward_lstm_150/zeros_1/packedPack'forward_lstm_150/strided_slice:output:0*forward_lstm_150/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_150/zeros_1/packed
forward_lstm_150/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_150/zeros_1/ConstС
forward_lstm_150/zeros_1Fill(forward_lstm_150/zeros_1/packed:output:0'forward_lstm_150/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/zeros_1
forward_lstm_150/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_150/transpose/permС
forward_lstm_150/transpose	Transposeinputs_0(forward_lstm_150/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
forward_lstm_150/transpose
forward_lstm_150/Shape_1Shapeforward_lstm_150/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_150/Shape_1
&forward_lstm_150/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_150/strided_slice_1/stack
(forward_lstm_150/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_1/stack_1
(forward_lstm_150/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_1/stack_2д
 forward_lstm_150/strided_slice_1StridedSlice!forward_lstm_150/Shape_1:output:0/forward_lstm_150/strided_slice_1/stack:output:01forward_lstm_150/strided_slice_1/stack_1:output:01forward_lstm_150/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_150/strided_slice_1Ї
,forward_lstm_150/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,forward_lstm_150/TensorArrayV2/element_shapeі
forward_lstm_150/TensorArrayV2TensorListReserve5forward_lstm_150/TensorArrayV2/element_shape:output:0)forward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_150/TensorArrayV2с
Fforward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2H
Fforward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8forward_lstm_150/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_150/transpose:y:0Oforward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_150/TensorArrayUnstack/TensorListFromTensor
&forward_lstm_150/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_150/strided_slice_2/stack
(forward_lstm_150/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_2/stack_1
(forward_lstm_150/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_2/stack_2ы
 forward_lstm_150/strided_slice_2StridedSliceforward_lstm_150/transpose:y:0/forward_lstm_150/strided_slice_2/stack:output:01forward_lstm_150/strided_slice_2/stack_1:output:01forward_lstm_150/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2"
 forward_lstm_150/strided_slice_2ы
4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpReadVariableOp=forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpє
%forward_lstm_150/lstm_cell_451/MatMulMatMul)forward_lstm_150/strided_slice_2:output:0<forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_150/lstm_cell_451/MatMulё
6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOp?forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp№
'forward_lstm_150/lstm_cell_451/MatMul_1MatMulforward_lstm_150/zeros:output:0>forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_150/lstm_cell_451/MatMul_1ш
"forward_lstm_150/lstm_cell_451/addAddV2/forward_lstm_150/lstm_cell_451/MatMul:product:01forward_lstm_150/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_150/lstm_cell_451/addъ
5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOp>forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpѕ
&forward_lstm_150/lstm_cell_451/BiasAddBiasAdd&forward_lstm_150/lstm_cell_451/add:z:0=forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_150/lstm_cell_451/BiasAddЂ
.forward_lstm_150/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.forward_lstm_150/lstm_cell_451/split/split_dimЛ
$forward_lstm_150/lstm_cell_451/splitSplit7forward_lstm_150/lstm_cell_451/split/split_dim:output:0/forward_lstm_150/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$forward_lstm_150/lstm_cell_451/splitМ
&forward_lstm_150/lstm_cell_451/SigmoidSigmoid-forward_lstm_150/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_150/lstm_cell_451/SigmoidР
(forward_lstm_150/lstm_cell_451/Sigmoid_1Sigmoid-forward_lstm_150/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_150/lstm_cell_451/Sigmoid_1в
"forward_lstm_150/lstm_cell_451/mulMul,forward_lstm_150/lstm_cell_451/Sigmoid_1:y:0!forward_lstm_150/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_150/lstm_cell_451/mulГ
#forward_lstm_150/lstm_cell_451/ReluRelu-forward_lstm_150/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_150/lstm_cell_451/Reluф
$forward_lstm_150/lstm_cell_451/mul_1Mul*forward_lstm_150/lstm_cell_451/Sigmoid:y:01forward_lstm_150/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_150/lstm_cell_451/mul_1й
$forward_lstm_150/lstm_cell_451/add_1AddV2&forward_lstm_150/lstm_cell_451/mul:z:0(forward_lstm_150/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_150/lstm_cell_451/add_1Р
(forward_lstm_150/lstm_cell_451/Sigmoid_2Sigmoid-forward_lstm_150/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_150/lstm_cell_451/Sigmoid_2В
%forward_lstm_150/lstm_cell_451/Relu_1Relu(forward_lstm_150/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_150/lstm_cell_451/Relu_1ш
$forward_lstm_150/lstm_cell_451/mul_2Mul,forward_lstm_150/lstm_cell_451/Sigmoid_2:y:03forward_lstm_150/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_150/lstm_cell_451/mul_2Б
.forward_lstm_150/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.forward_lstm_150/TensorArrayV2_1/element_shapeќ
 forward_lstm_150/TensorArrayV2_1TensorListReserve7forward_lstm_150/TensorArrayV2_1/element_shape:output:0)forward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_150/TensorArrayV2_1p
forward_lstm_150/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_150/timeЁ
)forward_lstm_150/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)forward_lstm_150/while/maximum_iterations
#forward_lstm_150/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_150/while/loop_counter
forward_lstm_150/whileWhile,forward_lstm_150/while/loop_counter:output:02forward_lstm_150/while/maximum_iterations:output:0forward_lstm_150/time:output:0)forward_lstm_150/TensorArrayV2_1:handle:0forward_lstm_150/zeros:output:0!forward_lstm_150/zeros_1:output:0)forward_lstm_150/strided_slice_1:output:0Hforward_lstm_150/TensorArrayUnstack/TensorListFromTensor:output_handle:0=forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource?forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource>forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource*
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
$forward_lstm_150_while_body_18421251*0
cond(R&
$forward_lstm_150_while_cond_18421250*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
forward_lstm_150/whileз
Aforward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Aforward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3forward_lstm_150/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_150/while:output:3Jforward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3forward_lstm_150/TensorArrayV2Stack/TensorListStackЃ
&forward_lstm_150/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&forward_lstm_150/strided_slice_3/stack
(forward_lstm_150/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_150/strided_slice_3/stack_1
(forward_lstm_150/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_3/stack_2
 forward_lstm_150/strided_slice_3StridedSlice<forward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_150/strided_slice_3/stack:output:01forward_lstm_150/strided_slice_3/stack_1:output:01forward_lstm_150/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 forward_lstm_150/strided_slice_3
!forward_lstm_150/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_150/transpose_1/permђ
forward_lstm_150/transpose_1	Transpose<forward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_150/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_150/transpose_1
forward_lstm_150/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_150/runtimej
backward_lstm_150/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_150/Shape
%backward_lstm_150/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_150/strided_slice/stack
'backward_lstm_150/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_150/strided_slice/stack_1
'backward_lstm_150/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_150/strided_slice/stack_2Ю
backward_lstm_150/strided_sliceStridedSlice backward_lstm_150/Shape:output:0.backward_lstm_150/strided_slice/stack:output:00backward_lstm_150/strided_slice/stack_1:output:00backward_lstm_150/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_150/strided_slice
backward_lstm_150/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_150/zeros/mul/yД
backward_lstm_150/zeros/mulMul(backward_lstm_150/strided_slice:output:0&backward_lstm_150/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/zeros/mul
backward_lstm_150/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_150/zeros/Less/yЏ
backward_lstm_150/zeros/LessLessbackward_lstm_150/zeros/mul:z:0'backward_lstm_150/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/zeros/Less
 backward_lstm_150/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_150/zeros/packed/1Ы
backward_lstm_150/zeros/packedPack(backward_lstm_150/strided_slice:output:0)backward_lstm_150/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_150/zeros/packed
backward_lstm_150/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_150/zeros/ConstН
backward_lstm_150/zerosFill'backward_lstm_150/zeros/packed:output:0&backward_lstm_150/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_150/zeros
backward_lstm_150/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_150/zeros_1/mul/yК
backward_lstm_150/zeros_1/mulMul(backward_lstm_150/strided_slice:output:0(backward_lstm_150/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/zeros_1/mul
 backward_lstm_150/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 backward_lstm_150/zeros_1/Less/yЗ
backward_lstm_150/zeros_1/LessLess!backward_lstm_150/zeros_1/mul:z:0)backward_lstm_150/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_150/zeros_1/Less
"backward_lstm_150/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_150/zeros_1/packed/1б
 backward_lstm_150/zeros_1/packedPack(backward_lstm_150/strided_slice:output:0+backward_lstm_150/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_150/zeros_1/packed
backward_lstm_150/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_150/zeros_1/ConstХ
backward_lstm_150/zeros_1Fill)backward_lstm_150/zeros_1/packed:output:0(backward_lstm_150/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_150/zeros_1
 backward_lstm_150/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_150/transpose/permФ
backward_lstm_150/transpose	Transposeinputs_0)backward_lstm_150/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_150/transpose
backward_lstm_150/Shape_1Shapebackward_lstm_150/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_150/Shape_1
'backward_lstm_150/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_150/strided_slice_1/stack 
)backward_lstm_150/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_1/stack_1 
)backward_lstm_150/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_1/stack_2к
!backward_lstm_150/strided_slice_1StridedSlice"backward_lstm_150/Shape_1:output:00backward_lstm_150/strided_slice_1/stack:output:02backward_lstm_150/strided_slice_1/stack_1:output:02backward_lstm_150/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_150/strided_slice_1Љ
-backward_lstm_150/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2/
-backward_lstm_150/TensorArrayV2/element_shapeњ
backward_lstm_150/TensorArrayV2TensorListReserve6backward_lstm_150/TensorArrayV2/element_shape:output:0*backward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_150/TensorArrayV2
 backward_lstm_150/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_150/ReverseV2/axisл
backward_lstm_150/ReverseV2	ReverseV2backward_lstm_150/transpose:y:0)backward_lstm_150/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_150/ReverseV2у
Gbackward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2I
Gbackward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeХ
9backward_lstm_150/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_150/ReverseV2:output:0Pbackward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_150/TensorArrayUnstack/TensorListFromTensor
'backward_lstm_150/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_150/strided_slice_2/stack 
)backward_lstm_150/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_2/stack_1 
)backward_lstm_150/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_2/stack_2ё
!backward_lstm_150/strided_slice_2StridedSlicebackward_lstm_150/transpose:y:00backward_lstm_150/strided_slice_2/stack:output:02backward_lstm_150/strided_slice_2/stack_1:output:02backward_lstm_150/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2#
!backward_lstm_150/strided_slice_2ю
5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpReadVariableOp>backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype027
5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpј
&backward_lstm_150/lstm_cell_452/MatMulMatMul*backward_lstm_150/strided_slice_2:output:0=backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_150/lstm_cell_452/MatMulє
7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOp@backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype029
7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpє
(backward_lstm_150/lstm_cell_452/MatMul_1MatMul backward_lstm_150/zeros:output:0?backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_150/lstm_cell_452/MatMul_1ь
#backward_lstm_150/lstm_cell_452/addAddV20backward_lstm_150/lstm_cell_452/MatMul:product:02backward_lstm_150/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_150/lstm_cell_452/addэ
6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOp?backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype028
6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpљ
'backward_lstm_150/lstm_cell_452/BiasAddBiasAdd'backward_lstm_150/lstm_cell_452/add:z:0>backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_150/lstm_cell_452/BiasAddЄ
/backward_lstm_150/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/backward_lstm_150/lstm_cell_452/split/split_dimП
%backward_lstm_150/lstm_cell_452/splitSplit8backward_lstm_150/lstm_cell_452/split/split_dim:output:00backward_lstm_150/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2'
%backward_lstm_150/lstm_cell_452/splitП
'backward_lstm_150/lstm_cell_452/SigmoidSigmoid.backward_lstm_150/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_150/lstm_cell_452/SigmoidУ
)backward_lstm_150/lstm_cell_452/Sigmoid_1Sigmoid.backward_lstm_150/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_150/lstm_cell_452/Sigmoid_1ж
#backward_lstm_150/lstm_cell_452/mulMul-backward_lstm_150/lstm_cell_452/Sigmoid_1:y:0"backward_lstm_150/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_150/lstm_cell_452/mulЖ
$backward_lstm_150/lstm_cell_452/ReluRelu.backward_lstm_150/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_150/lstm_cell_452/Reluш
%backward_lstm_150/lstm_cell_452/mul_1Mul+backward_lstm_150/lstm_cell_452/Sigmoid:y:02backward_lstm_150/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_150/lstm_cell_452/mul_1н
%backward_lstm_150/lstm_cell_452/add_1AddV2'backward_lstm_150/lstm_cell_452/mul:z:0)backward_lstm_150/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_150/lstm_cell_452/add_1У
)backward_lstm_150/lstm_cell_452/Sigmoid_2Sigmoid.backward_lstm_150/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_150/lstm_cell_452/Sigmoid_2Е
&backward_lstm_150/lstm_cell_452/Relu_1Relu)backward_lstm_150/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_150/lstm_cell_452/Relu_1ь
%backward_lstm_150/lstm_cell_452/mul_2Mul-backward_lstm_150/lstm_cell_452/Sigmoid_2:y:04backward_lstm_150/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_150/lstm_cell_452/mul_2Г
/backward_lstm_150/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   21
/backward_lstm_150/TensorArrayV2_1/element_shape
!backward_lstm_150/TensorArrayV2_1TensorListReserve8backward_lstm_150/TensorArrayV2_1/element_shape:output:0*backward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_150/TensorArrayV2_1r
backward_lstm_150/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_150/timeЃ
*backward_lstm_150/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*backward_lstm_150/while/maximum_iterations
$backward_lstm_150/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_150/while/loop_counter 
backward_lstm_150/whileWhile-backward_lstm_150/while/loop_counter:output:03backward_lstm_150/while/maximum_iterations:output:0backward_lstm_150/time:output:0*backward_lstm_150/TensorArrayV2_1:handle:0 backward_lstm_150/zeros:output:0"backward_lstm_150/zeros_1:output:0*backward_lstm_150/strided_slice_1:output:0Ibackward_lstm_150/TensorArrayUnstack/TensorListFromTensor:output_handle:0>backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource@backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource?backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource*
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
%backward_lstm_150_while_body_18421400*1
cond)R'
%backward_lstm_150_while_cond_18421399*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
backward_lstm_150/whileй
Bbackward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2D
Bbackward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeЙ
4backward_lstm_150/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_150/while:output:3Kbackward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype026
4backward_lstm_150/TensorArrayV2Stack/TensorListStackЅ
'backward_lstm_150/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2)
'backward_lstm_150/strided_slice_3/stack 
)backward_lstm_150/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_150/strided_slice_3/stack_1 
)backward_lstm_150/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_3/stack_2
!backward_lstm_150/strided_slice_3StridedSlice=backward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_150/strided_slice_3/stack:output:02backward_lstm_150/strided_slice_3/stack_1:output:02backward_lstm_150/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2#
!backward_lstm_150/strided_slice_3
"backward_lstm_150/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_150/transpose_1/permі
backward_lstm_150/transpose_1	Transpose=backward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_150/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_150/transpose_1
backward_lstm_150/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_150/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisФ
concatConcatV2)forward_lstm_150/strided_slice_3:output:0*backward_lstm_150/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp7^backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp6^backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp8^backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp^backward_lstm_150/while6^forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp5^forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp7^forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp^forward_lstm_150/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2p
6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp2n
5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp2r
7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp22
backward_lstm_150/whilebackward_lstm_150/while2n
5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp2l
4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp2p
6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp20
forward_lstm_150/whileforward_lstm_150/while:g c
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
д
Т
3__inference_forward_lstm_150_layer_call_fn_18422244
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
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_184181242
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
І_
­
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_18423373

inputs?
,lstm_cell_452_matmul_readvariableop_resource:	ШA
.lstm_cell_452_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_452_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_452/BiasAdd/ReadVariableOpЂ#lstm_cell_452/MatMul/ReadVariableOpЂ%lstm_cell_452/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_452/MatMul/ReadVariableOpReadVariableOp,lstm_cell_452_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_452/MatMul/ReadVariableOpА
lstm_cell_452/MatMulMatMulstrided_slice_2:output:0+lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/MatMulО
%lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_452_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_452/MatMul_1/ReadVariableOpЌ
lstm_cell_452/MatMul_1MatMulzeros:output:0-lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/MatMul_1Є
lstm_cell_452/addAddV2lstm_cell_452/MatMul:product:0 lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/addЗ
$lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_452_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_452/BiasAdd/ReadVariableOpБ
lstm_cell_452/BiasAddBiasAddlstm_cell_452/add:z:0,lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/BiasAdd
lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_452/split/split_dimї
lstm_cell_452/splitSplit&lstm_cell_452/split/split_dim:output:0lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_452/split
lstm_cell_452/SigmoidSigmoidlstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Sigmoid
lstm_cell_452/Sigmoid_1Sigmoidlstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Sigmoid_1
lstm_cell_452/mulMullstm_cell_452/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/mul
lstm_cell_452/ReluRelulstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Relu 
lstm_cell_452/mul_1Mullstm_cell_452/Sigmoid:y:0 lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/mul_1
lstm_cell_452/add_1AddV2lstm_cell_452/mul:z:0lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/add_1
lstm_cell_452/Sigmoid_2Sigmoidlstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Sigmoid_2
lstm_cell_452/Relu_1Relulstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Relu_1Є
lstm_cell_452/mul_2Mullstm_cell_452/Sigmoid_2:y:0"lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_452_matmul_readvariableop_resource.lstm_cell_452_matmul_1_readvariableop_resource-lstm_cell_452_biasadd_readvariableop_resource*
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
while_body_18423289*
condR
while_cond_18423288*K
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
NoOpNoOp%^lstm_cell_452/BiasAdd/ReadVariableOp$^lstm_cell_452/MatMul/ReadVariableOp&^lstm_cell_452/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_452/BiasAdd/ReadVariableOp$lstm_cell_452/BiasAdd/ReadVariableOp2J
#lstm_cell_452/MatMul/ReadVariableOp#lstm_cell_452/MatMul/ReadVariableOp2N
%lstm_cell_452/MatMul_1/ReadVariableOp%lstm_cell_452/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
М
љ
0__inference_lstm_cell_452_layer_call_fn_18423641

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
K__inference_lstm_cell_452_layer_call_and_return_conditional_losses_184184632
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
Щ
Ѕ
$forward_lstm_150_while_cond_18421567>
:forward_lstm_150_while_forward_lstm_150_while_loop_counterD
@forward_lstm_150_while_forward_lstm_150_while_maximum_iterations&
"forward_lstm_150_while_placeholder(
$forward_lstm_150_while_placeholder_1(
$forward_lstm_150_while_placeholder_2(
$forward_lstm_150_while_placeholder_3(
$forward_lstm_150_while_placeholder_4@
<forward_lstm_150_while_less_forward_lstm_150_strided_slice_1X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18421567___redundant_placeholder0X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18421567___redundant_placeholder1X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18421567___redundant_placeholder2X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18421567___redundant_placeholder3X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18421567___redundant_placeholder4#
forward_lstm_150_while_identity
Х
forward_lstm_150/while/LessLess"forward_lstm_150_while_placeholder<forward_lstm_150_while_less_forward_lstm_150_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_150/while/Less
forward_lstm_150/while/IdentityIdentityforward_lstm_150/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_150/while/Identity"K
forward_lstm_150_while_identity(forward_lstm_150/while/Identity:output:0*(
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

ј
G__inference_dense_150_layer_call_and_return_conditional_losses_18422222

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
п
Э
while_cond_18422634
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_18422634___redundant_placeholder06
2while_while_cond_18422634___redundant_placeholder16
2while_while_cond_18422634___redundant_placeholder26
2while_while_cond_18422634___redundant_placeholder3
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
while_cond_18417844
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_18417844___redundant_placeholder06
2while_while_cond_18417844___redundant_placeholder16
2while_while_cond_18417844___redundant_placeholder26
2while_while_cond_18417844___redundant_placeholder3
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
while_cond_18422332
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_18422332___redundant_placeholder06
2while_while_cond_18422332___redundant_placeholder16
2while_while_cond_18422332___redundant_placeholder26
2while_while_cond_18422332___redundant_placeholder3
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
і

K__inference_lstm_cell_452_layer_call_and_return_conditional_losses_18418609

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
аf
ф
%backward_lstm_150_while_body_18420535@
<backward_lstm_150_while_backward_lstm_150_while_loop_counterF
Bbackward_lstm_150_while_backward_lstm_150_while_maximum_iterations'
#backward_lstm_150_while_placeholder)
%backward_lstm_150_while_placeholder_1)
%backward_lstm_150_while_placeholder_2)
%backward_lstm_150_while_placeholder_3)
%backward_lstm_150_while_placeholder_4?
;backward_lstm_150_while_backward_lstm_150_strided_slice_1_0{
wbackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_150_while_less_backward_lstm_150_sub_1_0Y
Fbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0:	Ш[
Hbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0:	2ШV
Gbackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0:	Ш$
 backward_lstm_150_while_identity&
"backward_lstm_150_while_identity_1&
"backward_lstm_150_while_identity_2&
"backward_lstm_150_while_identity_3&
"backward_lstm_150_while_identity_4&
"backward_lstm_150_while_identity_5&
"backward_lstm_150_while_identity_6=
9backward_lstm_150_while_backward_lstm_150_strided_slice_1y
ubackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_150_while_less_backward_lstm_150_sub_1W
Dbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource:	ШY
Fbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource:	2ШT
Ebackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource:	ШЂ<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOpЂ;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpЂ=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOpч
Ibackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2K
Ibackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeП
;backward_lstm_150/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_150_while_placeholderRbackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02=
;backward_lstm_150/while/TensorArrayV2Read/TensorListGetItemЯ
backward_lstm_150/while/LessLess6backward_lstm_150_while_less_backward_lstm_150_sub_1_0#backward_lstm_150_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_150/while/Less
;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpReadVariableOpFbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02=
;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpЂ
,backward_lstm_150/while/lstm_cell_452/MatMulMatMulBbackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem:item:0Cbackward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_150/while/lstm_cell_452/MatMul
=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOpHbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02?
=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp
.backward_lstm_150/while/lstm_cell_452/MatMul_1MatMul%backward_lstm_150_while_placeholder_3Ebackward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ20
.backward_lstm_150/while/lstm_cell_452/MatMul_1
)backward_lstm_150/while/lstm_cell_452/addAddV26backward_lstm_150/while/lstm_cell_452/MatMul:product:08backward_lstm_150/while/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_150/while/lstm_cell_452/add
<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOpGbackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02>
<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp
-backward_lstm_150/while/lstm_cell_452/BiasAddBiasAdd-backward_lstm_150/while/lstm_cell_452/add:z:0Dbackward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_150/while/lstm_cell_452/BiasAddА
5backward_lstm_150/while/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5backward_lstm_150/while/lstm_cell_452/split/split_dimз
+backward_lstm_150/while/lstm_cell_452/splitSplit>backward_lstm_150/while/lstm_cell_452/split/split_dim:output:06backward_lstm_150/while/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2-
+backward_lstm_150/while/lstm_cell_452/splitб
-backward_lstm_150/while/lstm_cell_452/SigmoidSigmoid4backward_lstm_150/while/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22/
-backward_lstm_150/while/lstm_cell_452/Sigmoidе
/backward_lstm_150/while/lstm_cell_452/Sigmoid_1Sigmoid4backward_lstm_150/while/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_150/while/lstm_cell_452/Sigmoid_1ы
)backward_lstm_150/while/lstm_cell_452/mulMul3backward_lstm_150/while/lstm_cell_452/Sigmoid_1:y:0%backward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_150/while/lstm_cell_452/mulШ
*backward_lstm_150/while/lstm_cell_452/ReluRelu4backward_lstm_150/while/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_150/while/lstm_cell_452/Relu
+backward_lstm_150/while/lstm_cell_452/mul_1Mul1backward_lstm_150/while/lstm_cell_452/Sigmoid:y:08backward_lstm_150/while/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_150/while/lstm_cell_452/mul_1ѕ
+backward_lstm_150/while/lstm_cell_452/add_1AddV2-backward_lstm_150/while/lstm_cell_452/mul:z:0/backward_lstm_150/while/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_150/while/lstm_cell_452/add_1е
/backward_lstm_150/while/lstm_cell_452/Sigmoid_2Sigmoid4backward_lstm_150/while/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_150/while/lstm_cell_452/Sigmoid_2Ч
,backward_lstm_150/while/lstm_cell_452/Relu_1Relu/backward_lstm_150/while/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_150/while/lstm_cell_452/Relu_1
+backward_lstm_150/while/lstm_cell_452/mul_2Mul3backward_lstm_150/while/lstm_cell_452/Sigmoid_2:y:0:backward_lstm_150/while/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_150/while/lstm_cell_452/mul_2і
backward_lstm_150/while/SelectSelect backward_lstm_150/while/Less:z:0/backward_lstm_150/while/lstm_cell_452/mul_2:z:0%backward_lstm_150_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_150/while/Selectњ
 backward_lstm_150/while/Select_1Select backward_lstm_150/while/Less:z:0/backward_lstm_150/while/lstm_cell_452/mul_2:z:0%backward_lstm_150_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_150/while/Select_1њ
 backward_lstm_150/while/Select_2Select backward_lstm_150/while/Less:z:0/backward_lstm_150/while/lstm_cell_452/add_1:z:0%backward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_150/while/Select_2Г
<backward_lstm_150/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_150_while_placeholder_1#backward_lstm_150_while_placeholder'backward_lstm_150/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_150/while/TensorArrayV2Write/TensorListSetItem
backward_lstm_150/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_150/while/add/yБ
backward_lstm_150/while/addAddV2#backward_lstm_150_while_placeholder&backward_lstm_150/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/while/add
backward_lstm_150/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_150/while/add_1/yа
backward_lstm_150/while/add_1AddV2<backward_lstm_150_while_backward_lstm_150_while_loop_counter(backward_lstm_150/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/while/add_1Г
 backward_lstm_150/while/IdentityIdentity!backward_lstm_150/while/add_1:z:0^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_150/while/Identityи
"backward_lstm_150/while/Identity_1IdentityBbackward_lstm_150_while_backward_lstm_150_while_maximum_iterations^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_150/while/Identity_1Е
"backward_lstm_150/while/Identity_2Identitybackward_lstm_150/while/add:z:0^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_150/while/Identity_2т
"backward_lstm_150/while/Identity_3IdentityLbackward_lstm_150/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_150/while/Identity_3Ю
"backward_lstm_150/while/Identity_4Identity'backward_lstm_150/while/Select:output:0^backward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_150/while/Identity_4а
"backward_lstm_150/while/Identity_5Identity)backward_lstm_150/while/Select_1:output:0^backward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_150/while/Identity_5а
"backward_lstm_150/while/Identity_6Identity)backward_lstm_150/while/Select_2:output:0^backward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_150/while/Identity_6Л
backward_lstm_150/while/NoOpNoOp=^backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp<^backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp>^backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_150/while/NoOp"x
9backward_lstm_150_while_backward_lstm_150_strided_slice_1;backward_lstm_150_while_backward_lstm_150_strided_slice_1_0"M
 backward_lstm_150_while_identity)backward_lstm_150/while/Identity:output:0"Q
"backward_lstm_150_while_identity_1+backward_lstm_150/while/Identity_1:output:0"Q
"backward_lstm_150_while_identity_2+backward_lstm_150/while/Identity_2:output:0"Q
"backward_lstm_150_while_identity_3+backward_lstm_150/while/Identity_3:output:0"Q
"backward_lstm_150_while_identity_4+backward_lstm_150/while/Identity_4:output:0"Q
"backward_lstm_150_while_identity_5+backward_lstm_150/while/Identity_5:output:0"Q
"backward_lstm_150_while_identity_6+backward_lstm_150/while/Identity_6:output:0"n
4backward_lstm_150_while_less_backward_lstm_150_sub_16backward_lstm_150_while_less_backward_lstm_150_sub_1_0"
Ebackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resourceGbackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0"
Fbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resourceHbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0"
Dbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resourceFbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0"№
ubackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2|
<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp2z
;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp2~
=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Н
я
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_18422202

inputs
inputs_1	P
=forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource:	ШR
?forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource:	2ШM
>forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource:	ШQ
>backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource:	ШS
@backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource:	2ШN
?backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource:	Ш
identityЂ6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpЂ5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpЂ7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpЂbackward_lstm_150/whileЂ5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpЂ4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpЂ6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOpЂforward_lstm_150/while
%forward_lstm_150/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_150/RaggedToTensor/zeros
%forward_lstm_150/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2'
%forward_lstm_150/RaggedToTensor/Const
4forward_lstm_150/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_150/RaggedToTensor/Const:output:0inputs.forward_lstm_150/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_150/RaggedToTensor/RaggedTensorToTensorФ
;forward_lstm_150/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_150/RaggedNestedRowLengths/strided_slice/stackШ
=forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1Ш
=forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2Љ
5forward_lstm_150/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_150/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask27
5forward_lstm_150/RaggedNestedRowLengths/strided_sliceШ
=forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stackе
?forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2A
?forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1Ь
?forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2Е
7forward_lstm_150/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask29
7forward_lstm_150/RaggedNestedRowLengths/strided_slice_1
+forward_lstm_150/RaggedNestedRowLengths/subSub>forward_lstm_150/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_150/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2-
+forward_lstm_150/RaggedNestedRowLengths/subЄ
forward_lstm_150/CastCast/forward_lstm_150/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_150/Cast
forward_lstm_150/ShapeShape=forward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_150/Shape
$forward_lstm_150/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_150/strided_slice/stack
&forward_lstm_150/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_150/strided_slice/stack_1
&forward_lstm_150/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_150/strided_slice/stack_2Ш
forward_lstm_150/strided_sliceStridedSliceforward_lstm_150/Shape:output:0-forward_lstm_150/strided_slice/stack:output:0/forward_lstm_150/strided_slice/stack_1:output:0/forward_lstm_150/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_150/strided_slice~
forward_lstm_150/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_150/zeros/mul/yА
forward_lstm_150/zeros/mulMul'forward_lstm_150/strided_slice:output:0%forward_lstm_150/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros/mul
forward_lstm_150/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_150/zeros/Less/yЋ
forward_lstm_150/zeros/LessLessforward_lstm_150/zeros/mul:z:0&forward_lstm_150/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros/Less
forward_lstm_150/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_150/zeros/packed/1Ч
forward_lstm_150/zeros/packedPack'forward_lstm_150/strided_slice:output:0(forward_lstm_150/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_150/zeros/packed
forward_lstm_150/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_150/zeros/ConstЙ
forward_lstm_150/zerosFill&forward_lstm_150/zeros/packed:output:0%forward_lstm_150/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/zeros
forward_lstm_150/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_150/zeros_1/mul/yЖ
forward_lstm_150/zeros_1/mulMul'forward_lstm_150/strided_slice:output:0'forward_lstm_150/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros_1/mul
forward_lstm_150/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
forward_lstm_150/zeros_1/Less/yГ
forward_lstm_150/zeros_1/LessLess forward_lstm_150/zeros_1/mul:z:0(forward_lstm_150/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/zeros_1/Less
!forward_lstm_150/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_150/zeros_1/packed/1Э
forward_lstm_150/zeros_1/packedPack'forward_lstm_150/strided_slice:output:0*forward_lstm_150/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_150/zeros_1/packed
forward_lstm_150/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_150/zeros_1/ConstС
forward_lstm_150/zeros_1Fill(forward_lstm_150/zeros_1/packed:output:0'forward_lstm_150/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/zeros_1
forward_lstm_150/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_150/transpose/permэ
forward_lstm_150/transpose	Transpose=forward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_150/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_150/transpose
forward_lstm_150/Shape_1Shapeforward_lstm_150/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_150/Shape_1
&forward_lstm_150/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_150/strided_slice_1/stack
(forward_lstm_150/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_1/stack_1
(forward_lstm_150/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_1/stack_2д
 forward_lstm_150/strided_slice_1StridedSlice!forward_lstm_150/Shape_1:output:0/forward_lstm_150/strided_slice_1/stack:output:01forward_lstm_150/strided_slice_1/stack_1:output:01forward_lstm_150/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_150/strided_slice_1Ї
,forward_lstm_150/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,forward_lstm_150/TensorArrayV2/element_shapeі
forward_lstm_150/TensorArrayV2TensorListReserve5forward_lstm_150/TensorArrayV2/element_shape:output:0)forward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_150/TensorArrayV2с
Fforward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fforward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8forward_lstm_150/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_150/transpose:y:0Oforward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_150/TensorArrayUnstack/TensorListFromTensor
&forward_lstm_150/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_150/strided_slice_2/stack
(forward_lstm_150/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_2/stack_1
(forward_lstm_150/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_2/stack_2т
 forward_lstm_150/strided_slice_2StridedSliceforward_lstm_150/transpose:y:0/forward_lstm_150/strided_slice_2/stack:output:01forward_lstm_150/strided_slice_2/stack_1:output:01forward_lstm_150/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 forward_lstm_150/strided_slice_2ы
4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpReadVariableOp=forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOpє
%forward_lstm_150/lstm_cell_451/MatMulMatMul)forward_lstm_150/strided_slice_2:output:0<forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_150/lstm_cell_451/MatMulё
6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOp?forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp№
'forward_lstm_150/lstm_cell_451/MatMul_1MatMulforward_lstm_150/zeros:output:0>forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_150/lstm_cell_451/MatMul_1ш
"forward_lstm_150/lstm_cell_451/addAddV2/forward_lstm_150/lstm_cell_451/MatMul:product:01forward_lstm_150/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_150/lstm_cell_451/addъ
5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOp>forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOpѕ
&forward_lstm_150/lstm_cell_451/BiasAddBiasAdd&forward_lstm_150/lstm_cell_451/add:z:0=forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_150/lstm_cell_451/BiasAddЂ
.forward_lstm_150/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.forward_lstm_150/lstm_cell_451/split/split_dimЛ
$forward_lstm_150/lstm_cell_451/splitSplit7forward_lstm_150/lstm_cell_451/split/split_dim:output:0/forward_lstm_150/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$forward_lstm_150/lstm_cell_451/splitМ
&forward_lstm_150/lstm_cell_451/SigmoidSigmoid-forward_lstm_150/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_150/lstm_cell_451/SigmoidР
(forward_lstm_150/lstm_cell_451/Sigmoid_1Sigmoid-forward_lstm_150/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_150/lstm_cell_451/Sigmoid_1в
"forward_lstm_150/lstm_cell_451/mulMul,forward_lstm_150/lstm_cell_451/Sigmoid_1:y:0!forward_lstm_150/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_150/lstm_cell_451/mulГ
#forward_lstm_150/lstm_cell_451/ReluRelu-forward_lstm_150/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_150/lstm_cell_451/Reluф
$forward_lstm_150/lstm_cell_451/mul_1Mul*forward_lstm_150/lstm_cell_451/Sigmoid:y:01forward_lstm_150/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_150/lstm_cell_451/mul_1й
$forward_lstm_150/lstm_cell_451/add_1AddV2&forward_lstm_150/lstm_cell_451/mul:z:0(forward_lstm_150/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_150/lstm_cell_451/add_1Р
(forward_lstm_150/lstm_cell_451/Sigmoid_2Sigmoid-forward_lstm_150/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_150/lstm_cell_451/Sigmoid_2В
%forward_lstm_150/lstm_cell_451/Relu_1Relu(forward_lstm_150/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_150/lstm_cell_451/Relu_1ш
$forward_lstm_150/lstm_cell_451/mul_2Mul,forward_lstm_150/lstm_cell_451/Sigmoid_2:y:03forward_lstm_150/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_150/lstm_cell_451/mul_2Б
.forward_lstm_150/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.forward_lstm_150/TensorArrayV2_1/element_shapeќ
 forward_lstm_150/TensorArrayV2_1TensorListReserve7forward_lstm_150/TensorArrayV2_1/element_shape:output:0)forward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_150/TensorArrayV2_1p
forward_lstm_150/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_150/timeЃ
forward_lstm_150/zeros_like	ZerosLike(forward_lstm_150/lstm_cell_451/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/zeros_likeЁ
)forward_lstm_150/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)forward_lstm_150/while/maximum_iterations
#forward_lstm_150/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_150/while/loop_counter	
forward_lstm_150/whileWhile,forward_lstm_150/while/loop_counter:output:02forward_lstm_150/while/maximum_iterations:output:0forward_lstm_150/time:output:0)forward_lstm_150/TensorArrayV2_1:handle:0forward_lstm_150/zeros_like:y:0forward_lstm_150/zeros:output:0!forward_lstm_150/zeros_1:output:0)forward_lstm_150/strided_slice_1:output:0Hforward_lstm_150/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_150/Cast:y:0=forward_lstm_150_lstm_cell_451_matmul_readvariableop_resource?forward_lstm_150_lstm_cell_451_matmul_1_readvariableop_resource>forward_lstm_150_lstm_cell_451_biasadd_readvariableop_resource*
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
$forward_lstm_150_while_body_18421926*0
cond(R&
$forward_lstm_150_while_cond_18421925*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_150/whileз
Aforward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Aforward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3forward_lstm_150/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_150/while:output:3Jforward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3forward_lstm_150/TensorArrayV2Stack/TensorListStackЃ
&forward_lstm_150/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&forward_lstm_150/strided_slice_3/stack
(forward_lstm_150/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_150/strided_slice_3/stack_1
(forward_lstm_150/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_150/strided_slice_3/stack_2
 forward_lstm_150/strided_slice_3StridedSlice<forward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_150/strided_slice_3/stack:output:01forward_lstm_150/strided_slice_3/stack_1:output:01forward_lstm_150/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 forward_lstm_150/strided_slice_3
!forward_lstm_150/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_150/transpose_1/permђ
forward_lstm_150/transpose_1	Transpose<forward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_150/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_150/transpose_1
forward_lstm_150/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_150/runtime
&backward_lstm_150/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_150/RaggedToTensor/zeros
&backward_lstm_150/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2(
&backward_lstm_150/RaggedToTensor/Const
5backward_lstm_150/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_150/RaggedToTensor/Const:output:0inputs/backward_lstm_150/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_150/RaggedToTensor/RaggedTensorToTensorЦ
<backward_lstm_150/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_150/RaggedNestedRowLengths/strided_slice/stackЪ
>backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1Ъ
>backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2Ў
6backward_lstm_150/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_150/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_150/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask28
6backward_lstm_150/RaggedNestedRowLengths/strided_sliceЪ
>backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stackз
@backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2B
@backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1Ю
@backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2К
8backward_lstm_150/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_150/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2:
8backward_lstm_150/RaggedNestedRowLengths/strided_slice_1
,backward_lstm_150/RaggedNestedRowLengths/subSub?backward_lstm_150/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_150/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2.
,backward_lstm_150/RaggedNestedRowLengths/subЇ
backward_lstm_150/CastCast0backward_lstm_150/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_150/Cast 
backward_lstm_150/ShapeShape>backward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_150/Shape
%backward_lstm_150/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_150/strided_slice/stack
'backward_lstm_150/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_150/strided_slice/stack_1
'backward_lstm_150/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_150/strided_slice/stack_2Ю
backward_lstm_150/strided_sliceStridedSlice backward_lstm_150/Shape:output:0.backward_lstm_150/strided_slice/stack:output:00backward_lstm_150/strided_slice/stack_1:output:00backward_lstm_150/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_150/strided_slice
backward_lstm_150/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_150/zeros/mul/yД
backward_lstm_150/zeros/mulMul(backward_lstm_150/strided_slice:output:0&backward_lstm_150/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/zeros/mul
backward_lstm_150/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_150/zeros/Less/yЏ
backward_lstm_150/zeros/LessLessbackward_lstm_150/zeros/mul:z:0'backward_lstm_150/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/zeros/Less
 backward_lstm_150/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_150/zeros/packed/1Ы
backward_lstm_150/zeros/packedPack(backward_lstm_150/strided_slice:output:0)backward_lstm_150/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_150/zeros/packed
backward_lstm_150/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_150/zeros/ConstН
backward_lstm_150/zerosFill'backward_lstm_150/zeros/packed:output:0&backward_lstm_150/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_150/zeros
backward_lstm_150/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_150/zeros_1/mul/yК
backward_lstm_150/zeros_1/mulMul(backward_lstm_150/strided_slice:output:0(backward_lstm_150/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/zeros_1/mul
 backward_lstm_150/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 backward_lstm_150/zeros_1/Less/yЗ
backward_lstm_150/zeros_1/LessLess!backward_lstm_150/zeros_1/mul:z:0)backward_lstm_150/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_150/zeros_1/Less
"backward_lstm_150/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_150/zeros_1/packed/1б
 backward_lstm_150/zeros_1/packedPack(backward_lstm_150/strided_slice:output:0+backward_lstm_150/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_150/zeros_1/packed
backward_lstm_150/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_150/zeros_1/ConstХ
backward_lstm_150/zeros_1Fill)backward_lstm_150/zeros_1/packed:output:0(backward_lstm_150/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_150/zeros_1
 backward_lstm_150/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_150/transpose/permё
backward_lstm_150/transpose	Transpose>backward_lstm_150/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_150/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_150/transpose
backward_lstm_150/Shape_1Shapebackward_lstm_150/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_150/Shape_1
'backward_lstm_150/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_150/strided_slice_1/stack 
)backward_lstm_150/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_1/stack_1 
)backward_lstm_150/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_1/stack_2к
!backward_lstm_150/strided_slice_1StridedSlice"backward_lstm_150/Shape_1:output:00backward_lstm_150/strided_slice_1/stack:output:02backward_lstm_150/strided_slice_1/stack_1:output:02backward_lstm_150/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_150/strided_slice_1Љ
-backward_lstm_150/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2/
-backward_lstm_150/TensorArrayV2/element_shapeњ
backward_lstm_150/TensorArrayV2TensorListReserve6backward_lstm_150/TensorArrayV2/element_shape:output:0*backward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_150/TensorArrayV2
 backward_lstm_150/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_150/ReverseV2/axisв
backward_lstm_150/ReverseV2	ReverseV2backward_lstm_150/transpose:y:0)backward_lstm_150/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_150/ReverseV2у
Gbackward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gbackward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shapeХ
9backward_lstm_150/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_150/ReverseV2:output:0Pbackward_lstm_150/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_150/TensorArrayUnstack/TensorListFromTensor
'backward_lstm_150/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_150/strided_slice_2/stack 
)backward_lstm_150/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_2/stack_1 
)backward_lstm_150/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_2/stack_2ш
!backward_lstm_150/strided_slice_2StridedSlicebackward_lstm_150/transpose:y:00backward_lstm_150/strided_slice_2/stack:output:02backward_lstm_150/strided_slice_2/stack_1:output:02backward_lstm_150/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2#
!backward_lstm_150/strided_slice_2ю
5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpReadVariableOp>backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype027
5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOpј
&backward_lstm_150/lstm_cell_452/MatMulMatMul*backward_lstm_150/strided_slice_2:output:0=backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_150/lstm_cell_452/MatMulє
7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOp@backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype029
7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOpє
(backward_lstm_150/lstm_cell_452/MatMul_1MatMul backward_lstm_150/zeros:output:0?backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_150/lstm_cell_452/MatMul_1ь
#backward_lstm_150/lstm_cell_452/addAddV20backward_lstm_150/lstm_cell_452/MatMul:product:02backward_lstm_150/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_150/lstm_cell_452/addэ
6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOp?backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype028
6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOpљ
'backward_lstm_150/lstm_cell_452/BiasAddBiasAdd'backward_lstm_150/lstm_cell_452/add:z:0>backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_150/lstm_cell_452/BiasAddЄ
/backward_lstm_150/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/backward_lstm_150/lstm_cell_452/split/split_dimП
%backward_lstm_150/lstm_cell_452/splitSplit8backward_lstm_150/lstm_cell_452/split/split_dim:output:00backward_lstm_150/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2'
%backward_lstm_150/lstm_cell_452/splitП
'backward_lstm_150/lstm_cell_452/SigmoidSigmoid.backward_lstm_150/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_150/lstm_cell_452/SigmoidУ
)backward_lstm_150/lstm_cell_452/Sigmoid_1Sigmoid.backward_lstm_150/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_150/lstm_cell_452/Sigmoid_1ж
#backward_lstm_150/lstm_cell_452/mulMul-backward_lstm_150/lstm_cell_452/Sigmoid_1:y:0"backward_lstm_150/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_150/lstm_cell_452/mulЖ
$backward_lstm_150/lstm_cell_452/ReluRelu.backward_lstm_150/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_150/lstm_cell_452/Reluш
%backward_lstm_150/lstm_cell_452/mul_1Mul+backward_lstm_150/lstm_cell_452/Sigmoid:y:02backward_lstm_150/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_150/lstm_cell_452/mul_1н
%backward_lstm_150/lstm_cell_452/add_1AddV2'backward_lstm_150/lstm_cell_452/mul:z:0)backward_lstm_150/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_150/lstm_cell_452/add_1У
)backward_lstm_150/lstm_cell_452/Sigmoid_2Sigmoid.backward_lstm_150/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_150/lstm_cell_452/Sigmoid_2Е
&backward_lstm_150/lstm_cell_452/Relu_1Relu)backward_lstm_150/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_150/lstm_cell_452/Relu_1ь
%backward_lstm_150/lstm_cell_452/mul_2Mul-backward_lstm_150/lstm_cell_452/Sigmoid_2:y:04backward_lstm_150/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_150/lstm_cell_452/mul_2Г
/backward_lstm_150/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   21
/backward_lstm_150/TensorArrayV2_1/element_shape
!backward_lstm_150/TensorArrayV2_1TensorListReserve8backward_lstm_150/TensorArrayV2_1/element_shape:output:0*backward_lstm_150/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_150/TensorArrayV2_1r
backward_lstm_150/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_150/time
'backward_lstm_150/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_150/Max/reduction_indicesЄ
backward_lstm_150/MaxMaxbackward_lstm_150/Cast:y:00backward_lstm_150/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/Maxt
backward_lstm_150/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_150/sub/y
backward_lstm_150/subSubbackward_lstm_150/Max:output:0 backward_lstm_150/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/sub
backward_lstm_150/Sub_1Subbackward_lstm_150/sub:z:0backward_lstm_150/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_150/Sub_1І
backward_lstm_150/zeros_like	ZerosLike)backward_lstm_150/lstm_cell_452/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_150/zeros_likeЃ
*backward_lstm_150/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*backward_lstm_150/while/maximum_iterations
$backward_lstm_150/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_150/while/loop_counterЅ	
backward_lstm_150/whileWhile-backward_lstm_150/while/loop_counter:output:03backward_lstm_150/while/maximum_iterations:output:0backward_lstm_150/time:output:0*backward_lstm_150/TensorArrayV2_1:handle:0 backward_lstm_150/zeros_like:y:0 backward_lstm_150/zeros:output:0"backward_lstm_150/zeros_1:output:0*backward_lstm_150/strided_slice_1:output:0Ibackward_lstm_150/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_150/Sub_1:z:0>backward_lstm_150_lstm_cell_452_matmul_readvariableop_resource@backward_lstm_150_lstm_cell_452_matmul_1_readvariableop_resource?backward_lstm_150_lstm_cell_452_biasadd_readvariableop_resource*
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
%backward_lstm_150_while_body_18422105*1
cond)R'
%backward_lstm_150_while_cond_18422104*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_150/whileй
Bbackward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2D
Bbackward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shapeЙ
4backward_lstm_150/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_150/while:output:3Kbackward_lstm_150/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype026
4backward_lstm_150/TensorArrayV2Stack/TensorListStackЅ
'backward_lstm_150/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2)
'backward_lstm_150/strided_slice_3/stack 
)backward_lstm_150/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_150/strided_slice_3/stack_1 
)backward_lstm_150/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_150/strided_slice_3/stack_2
!backward_lstm_150/strided_slice_3StridedSlice=backward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_150/strided_slice_3/stack:output:02backward_lstm_150/strided_slice_3/stack_1:output:02backward_lstm_150/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2#
!backward_lstm_150/strided_slice_3
"backward_lstm_150/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_150/transpose_1/permі
backward_lstm_150/transpose_1	Transpose=backward_lstm_150/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_150/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_150/transpose_1
backward_lstm_150/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_150/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisФ
concatConcatV2)forward_lstm_150/strided_slice_3:output:0*backward_lstm_150/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp7^backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp6^backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp8^backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp^backward_lstm_150/while6^forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp5^forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp7^forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp^forward_lstm_150/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2p
6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp6backward_lstm_150/lstm_cell_452/BiasAdd/ReadVariableOp2n
5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp5backward_lstm_150/lstm_cell_452/MatMul/ReadVariableOp2r
7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp7backward_lstm_150/lstm_cell_452/MatMul_1/ReadVariableOp22
backward_lstm_150/whilebackward_lstm_150/while2n
5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp5forward_lstm_150/lstm_cell_451/BiasAdd/ReadVariableOp2l
4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp4forward_lstm_150/lstm_cell_451/MatMul/ReadVariableOp2p
6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp6forward_lstm_150/lstm_cell_451/MatMul_1/ReadVariableOp20
forward_lstm_150/whileforward_lstm_150/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
М
Т
Fsequential_150_bidirectional_150_backward_lstm_150_while_cond_18417651
~sequential_150_bidirectional_150_backward_lstm_150_while_sequential_150_bidirectional_150_backward_lstm_150_while_loop_counter
sequential_150_bidirectional_150_backward_lstm_150_while_sequential_150_bidirectional_150_backward_lstm_150_while_maximum_iterationsH
Dsequential_150_bidirectional_150_backward_lstm_150_while_placeholderJ
Fsequential_150_bidirectional_150_backward_lstm_150_while_placeholder_1J
Fsequential_150_bidirectional_150_backward_lstm_150_while_placeholder_2J
Fsequential_150_bidirectional_150_backward_lstm_150_while_placeholder_3J
Fsequential_150_bidirectional_150_backward_lstm_150_while_placeholder_4
sequential_150_bidirectional_150_backward_lstm_150_while_less_sequential_150_bidirectional_150_backward_lstm_150_strided_slice_1
sequential_150_bidirectional_150_backward_lstm_150_while_sequential_150_bidirectional_150_backward_lstm_150_while_cond_18417651___redundant_placeholder0
sequential_150_bidirectional_150_backward_lstm_150_while_sequential_150_bidirectional_150_backward_lstm_150_while_cond_18417651___redundant_placeholder1
sequential_150_bidirectional_150_backward_lstm_150_while_sequential_150_bidirectional_150_backward_lstm_150_while_cond_18417651___redundant_placeholder2
sequential_150_bidirectional_150_backward_lstm_150_while_sequential_150_bidirectional_150_backward_lstm_150_while_cond_18417651___redundant_placeholder3
sequential_150_bidirectional_150_backward_lstm_150_while_sequential_150_bidirectional_150_backward_lstm_150_while_cond_18417651___redundant_placeholder4E
Asequential_150_bidirectional_150_backward_lstm_150_while_identity
№
=sequential_150/bidirectional_150/backward_lstm_150/while/LessLessDsequential_150_bidirectional_150_backward_lstm_150_while_placeholdersequential_150_bidirectional_150_backward_lstm_150_while_less_sequential_150_bidirectional_150_backward_lstm_150_strided_slice_1*
T0*
_output_shapes
: 2?
=sequential_150/bidirectional_150/backward_lstm_150/while/Lessі
Asequential_150/bidirectional_150/backward_lstm_150/while/IdentityIdentityAsequential_150/bidirectional_150/backward_lstm_150/while/Less:z:0*
T0
*
_output_shapes
: 2C
Asequential_150/bidirectional_150/backward_lstm_150/while/Identity"
Asequential_150_bidirectional_150_backward_lstm_150_while_identityJsequential_150/bidirectional_150/backward_lstm_150/while/Identity:output:0*(
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
еH

O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_18418546

inputs)
lstm_cell_452_18418464:	Ш)
lstm_cell_452_18418466:	2Ш%
lstm_cell_452_18418468:	Ш
identityЂ%lstm_cell_452/StatefulPartitionedCallЂwhileD
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
%lstm_cell_452/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_452_18418464lstm_cell_452_18418466lstm_cell_452_18418468*
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
K__inference_lstm_cell_452_layer_call_and_return_conditional_losses_184184632'
%lstm_cell_452/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_452_18418464lstm_cell_452_18418466lstm_cell_452_18418468*
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
while_body_18418477*
condR
while_cond_18418476*K
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
NoOpNoOp&^lstm_cell_452/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2N
%lstm_cell_452/StatefulPartitionedCall%lstm_cell_452/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
§
Е
%backward_lstm_150_while_cond_18421399@
<backward_lstm_150_while_backward_lstm_150_while_loop_counterF
Bbackward_lstm_150_while_backward_lstm_150_while_maximum_iterations'
#backward_lstm_150_while_placeholder)
%backward_lstm_150_while_placeholder_1)
%backward_lstm_150_while_placeholder_2)
%backward_lstm_150_while_placeholder_3B
>backward_lstm_150_while_less_backward_lstm_150_strided_slice_1Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18421399___redundant_placeholder0Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18421399___redundant_placeholder1Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18421399___redundant_placeholder2Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18421399___redundant_placeholder3$
 backward_lstm_150_while_identity
Ъ
backward_lstm_150/while/LessLess#backward_lstm_150_while_placeholder>backward_lstm_150_while_less_backward_lstm_150_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_150/while/Less
 backward_lstm_150/while/IdentityIdentity backward_lstm_150/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_150/while/Identity"M
 backward_lstm_150_while_identity)backward_lstm_150/while/Identity:output:0*(
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
ъ
М
%backward_lstm_150_while_cond_18421746@
<backward_lstm_150_while_backward_lstm_150_while_loop_counterF
Bbackward_lstm_150_while_backward_lstm_150_while_maximum_iterations'
#backward_lstm_150_while_placeholder)
%backward_lstm_150_while_placeholder_1)
%backward_lstm_150_while_placeholder_2)
%backward_lstm_150_while_placeholder_3)
%backward_lstm_150_while_placeholder_4B
>backward_lstm_150_while_less_backward_lstm_150_strided_slice_1Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18421746___redundant_placeholder0Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18421746___redundant_placeholder1Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18421746___redundant_placeholder2Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18421746___redundant_placeholder3Z
Vbackward_lstm_150_while_backward_lstm_150_while_cond_18421746___redundant_placeholder4$
 backward_lstm_150_while_identity
Ъ
backward_lstm_150/while/LessLess#backward_lstm_150_while_placeholder>backward_lstm_150_while_less_backward_lstm_150_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_150/while/Less
 backward_lstm_150/while/IdentityIdentity backward_lstm_150/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_150/while/Identity"M
 backward_lstm_150_while_identity)backward_lstm_150/while/Identity:output:0*(
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
аf
ф
%backward_lstm_150_while_body_18421747@
<backward_lstm_150_while_backward_lstm_150_while_loop_counterF
Bbackward_lstm_150_while_backward_lstm_150_while_maximum_iterations'
#backward_lstm_150_while_placeholder)
%backward_lstm_150_while_placeholder_1)
%backward_lstm_150_while_placeholder_2)
%backward_lstm_150_while_placeholder_3)
%backward_lstm_150_while_placeholder_4?
;backward_lstm_150_while_backward_lstm_150_strided_slice_1_0{
wbackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_150_while_less_backward_lstm_150_sub_1_0Y
Fbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0:	Ш[
Hbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0:	2ШV
Gbackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0:	Ш$
 backward_lstm_150_while_identity&
"backward_lstm_150_while_identity_1&
"backward_lstm_150_while_identity_2&
"backward_lstm_150_while_identity_3&
"backward_lstm_150_while_identity_4&
"backward_lstm_150_while_identity_5&
"backward_lstm_150_while_identity_6=
9backward_lstm_150_while_backward_lstm_150_strided_slice_1y
ubackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_150_while_less_backward_lstm_150_sub_1W
Dbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource:	ШY
Fbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource:	2ШT
Ebackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource:	ШЂ<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOpЂ;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpЂ=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOpч
Ibackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2K
Ibackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeП
;backward_lstm_150/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_150_while_placeholderRbackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02=
;backward_lstm_150/while/TensorArrayV2Read/TensorListGetItemЯ
backward_lstm_150/while/LessLess6backward_lstm_150_while_less_backward_lstm_150_sub_1_0#backward_lstm_150_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_150/while/Less
;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpReadVariableOpFbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02=
;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOpЂ
,backward_lstm_150/while/lstm_cell_452/MatMulMatMulBbackward_lstm_150/while/TensorArrayV2Read/TensorListGetItem:item:0Cbackward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_150/while/lstm_cell_452/MatMul
=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOpHbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02?
=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp
.backward_lstm_150/while/lstm_cell_452/MatMul_1MatMul%backward_lstm_150_while_placeholder_3Ebackward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ20
.backward_lstm_150/while/lstm_cell_452/MatMul_1
)backward_lstm_150/while/lstm_cell_452/addAddV26backward_lstm_150/while/lstm_cell_452/MatMul:product:08backward_lstm_150/while/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_150/while/lstm_cell_452/add
<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOpGbackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02>
<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp
-backward_lstm_150/while/lstm_cell_452/BiasAddBiasAdd-backward_lstm_150/while/lstm_cell_452/add:z:0Dbackward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_150/while/lstm_cell_452/BiasAddА
5backward_lstm_150/while/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5backward_lstm_150/while/lstm_cell_452/split/split_dimз
+backward_lstm_150/while/lstm_cell_452/splitSplit>backward_lstm_150/while/lstm_cell_452/split/split_dim:output:06backward_lstm_150/while/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2-
+backward_lstm_150/while/lstm_cell_452/splitб
-backward_lstm_150/while/lstm_cell_452/SigmoidSigmoid4backward_lstm_150/while/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22/
-backward_lstm_150/while/lstm_cell_452/Sigmoidе
/backward_lstm_150/while/lstm_cell_452/Sigmoid_1Sigmoid4backward_lstm_150/while/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_150/while/lstm_cell_452/Sigmoid_1ы
)backward_lstm_150/while/lstm_cell_452/mulMul3backward_lstm_150/while/lstm_cell_452/Sigmoid_1:y:0%backward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_150/while/lstm_cell_452/mulШ
*backward_lstm_150/while/lstm_cell_452/ReluRelu4backward_lstm_150/while/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_150/while/lstm_cell_452/Relu
+backward_lstm_150/while/lstm_cell_452/mul_1Mul1backward_lstm_150/while/lstm_cell_452/Sigmoid:y:08backward_lstm_150/while/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_150/while/lstm_cell_452/mul_1ѕ
+backward_lstm_150/while/lstm_cell_452/add_1AddV2-backward_lstm_150/while/lstm_cell_452/mul:z:0/backward_lstm_150/while/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_150/while/lstm_cell_452/add_1е
/backward_lstm_150/while/lstm_cell_452/Sigmoid_2Sigmoid4backward_lstm_150/while/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_150/while/lstm_cell_452/Sigmoid_2Ч
,backward_lstm_150/while/lstm_cell_452/Relu_1Relu/backward_lstm_150/while/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_150/while/lstm_cell_452/Relu_1
+backward_lstm_150/while/lstm_cell_452/mul_2Mul3backward_lstm_150/while/lstm_cell_452/Sigmoid_2:y:0:backward_lstm_150/while/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_150/while/lstm_cell_452/mul_2і
backward_lstm_150/while/SelectSelect backward_lstm_150/while/Less:z:0/backward_lstm_150/while/lstm_cell_452/mul_2:z:0%backward_lstm_150_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_150/while/Selectњ
 backward_lstm_150/while/Select_1Select backward_lstm_150/while/Less:z:0/backward_lstm_150/while/lstm_cell_452/mul_2:z:0%backward_lstm_150_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_150/while/Select_1њ
 backward_lstm_150/while/Select_2Select backward_lstm_150/while/Less:z:0/backward_lstm_150/while/lstm_cell_452/add_1:z:0%backward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_150/while/Select_2Г
<backward_lstm_150/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_150_while_placeholder_1#backward_lstm_150_while_placeholder'backward_lstm_150/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_150/while/TensorArrayV2Write/TensorListSetItem
backward_lstm_150/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_150/while/add/yБ
backward_lstm_150/while/addAddV2#backward_lstm_150_while_placeholder&backward_lstm_150/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/while/add
backward_lstm_150/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_150/while/add_1/yа
backward_lstm_150/while/add_1AddV2<backward_lstm_150_while_backward_lstm_150_while_loop_counter(backward_lstm_150/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_150/while/add_1Г
 backward_lstm_150/while/IdentityIdentity!backward_lstm_150/while/add_1:z:0^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_150/while/Identityи
"backward_lstm_150/while/Identity_1IdentityBbackward_lstm_150_while_backward_lstm_150_while_maximum_iterations^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_150/while/Identity_1Е
"backward_lstm_150/while/Identity_2Identitybackward_lstm_150/while/add:z:0^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_150/while/Identity_2т
"backward_lstm_150/while/Identity_3IdentityLbackward_lstm_150/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_150/while/Identity_3Ю
"backward_lstm_150/while/Identity_4Identity'backward_lstm_150/while/Select:output:0^backward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_150/while/Identity_4а
"backward_lstm_150/while/Identity_5Identity)backward_lstm_150/while/Select_1:output:0^backward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_150/while/Identity_5а
"backward_lstm_150/while/Identity_6Identity)backward_lstm_150/while/Select_2:output:0^backward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_150/while/Identity_6Л
backward_lstm_150/while/NoOpNoOp=^backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp<^backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp>^backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_150/while/NoOp"x
9backward_lstm_150_while_backward_lstm_150_strided_slice_1;backward_lstm_150_while_backward_lstm_150_strided_slice_1_0"M
 backward_lstm_150_while_identity)backward_lstm_150/while/Identity:output:0"Q
"backward_lstm_150_while_identity_1+backward_lstm_150/while/Identity_1:output:0"Q
"backward_lstm_150_while_identity_2+backward_lstm_150/while/Identity_2:output:0"Q
"backward_lstm_150_while_identity_3+backward_lstm_150/while/Identity_3:output:0"Q
"backward_lstm_150_while_identity_4+backward_lstm_150/while/Identity_4:output:0"Q
"backward_lstm_150_while_identity_5+backward_lstm_150/while/Identity_5:output:0"Q
"backward_lstm_150_while_identity_6+backward_lstm_150/while/Identity_6:output:0"n
4backward_lstm_150_while_less_backward_lstm_150_sub_16backward_lstm_150_while_less_backward_lstm_150_sub_1_0"
Ebackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resourceGbackward_lstm_150_while_lstm_cell_452_biasadd_readvariableop_resource_0"
Fbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resourceHbackward_lstm_150_while_lstm_cell_452_matmul_1_readvariableop_resource_0"
Dbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resourceFbackward_lstm_150_while_lstm_cell_452_matmul_readvariableop_resource_0"№
ubackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2|
<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp<backward_lstm_150/while/lstm_cell_452/BiasAdd/ReadVariableOp2z
;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp;backward_lstm_150/while/lstm_cell_452/MatMul/ReadVariableOp2~
=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp=backward_lstm_150/while/lstm_cell_452/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
G__inference_dense_150_layer_call_and_return_conditional_losses_18420217

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
Ъ

Э
&__inference_signature_wrapper_18420812

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
#__inference__wrapped_model_184177562
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
М
љ
0__inference_lstm_cell_452_layer_call_fn_18423658

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
K__inference_lstm_cell_452_layer_call_and_return_conditional_losses_184186092
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
п
Ё
$forward_lstm_150_while_cond_18421250>
:forward_lstm_150_while_forward_lstm_150_while_loop_counterD
@forward_lstm_150_while_forward_lstm_150_while_maximum_iterations&
"forward_lstm_150_while_placeholder(
$forward_lstm_150_while_placeholder_1(
$forward_lstm_150_while_placeholder_2(
$forward_lstm_150_while_placeholder_3@
<forward_lstm_150_while_less_forward_lstm_150_strided_slice_1X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18421250___redundant_placeholder0X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18421250___redundant_placeholder1X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18421250___redundant_placeholder2X
Tforward_lstm_150_while_forward_lstm_150_while_cond_18421250___redundant_placeholder3#
forward_lstm_150_while_identity
Х
forward_lstm_150/while/LessLess"forward_lstm_150_while_placeholder<forward_lstm_150_while_less_forward_lstm_150_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_150/while/Less
forward_lstm_150/while/IdentityIdentityforward_lstm_150/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_150/while/Identity"K
forward_lstm_150_while_identity(forward_lstm_150/while/Identity:output:0*(
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
І_
­
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_18419341

inputs?
,lstm_cell_452_matmul_readvariableop_resource:	ШA
.lstm_cell_452_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_452_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_452/BiasAdd/ReadVariableOpЂ#lstm_cell_452/MatMul/ReadVariableOpЂ%lstm_cell_452/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_452/MatMul/ReadVariableOpReadVariableOp,lstm_cell_452_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_452/MatMul/ReadVariableOpА
lstm_cell_452/MatMulMatMulstrided_slice_2:output:0+lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/MatMulО
%lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_452_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_452/MatMul_1/ReadVariableOpЌ
lstm_cell_452/MatMul_1MatMulzeros:output:0-lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/MatMul_1Є
lstm_cell_452/addAddV2lstm_cell_452/MatMul:product:0 lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/addЗ
$lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_452_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_452/BiasAdd/ReadVariableOpБ
lstm_cell_452/BiasAddBiasAddlstm_cell_452/add:z:0,lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/BiasAdd
lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_452/split/split_dimї
lstm_cell_452/splitSplit&lstm_cell_452/split/split_dim:output:0lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_452/split
lstm_cell_452/SigmoidSigmoidlstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Sigmoid
lstm_cell_452/Sigmoid_1Sigmoidlstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Sigmoid_1
lstm_cell_452/mulMullstm_cell_452/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/mul
lstm_cell_452/ReluRelulstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Relu 
lstm_cell_452/mul_1Mullstm_cell_452/Sigmoid:y:0 lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/mul_1
lstm_cell_452/add_1AddV2lstm_cell_452/mul:z:0lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/add_1
lstm_cell_452/Sigmoid_2Sigmoidlstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Sigmoid_2
lstm_cell_452/Relu_1Relulstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Relu_1Є
lstm_cell_452/mul_2Mullstm_cell_452/Sigmoid_2:y:0"lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_452_matmul_readvariableop_resource.lstm_cell_452_matmul_1_readvariableop_resource-lstm_cell_452_biasadd_readvariableop_resource*
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
while_body_18419257*
condR
while_cond_18419256*K
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
NoOpNoOp%^lstm_cell_452/BiasAdd/ReadVariableOp$^lstm_cell_452/MatMul/ReadVariableOp&^lstm_cell_452/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_452/BiasAdd/ReadVariableOp$lstm_cell_452/BiasAdd/ReadVariableOp2J
#lstm_cell_452/MatMul/ReadVariableOp#lstm_cell_452/MatMul/ReadVariableOp2N
%lstm_cell_452/MatMul_1/ReadVariableOp%lstm_cell_452/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ў

K__inference_lstm_cell_452_layer_call_and_return_conditional_losses_18423690

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
_
Џ
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_18423067
inputs_0?
,lstm_cell_452_matmul_readvariableop_resource:	ШA
.lstm_cell_452_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_452_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_452/BiasAdd/ReadVariableOpЂ#lstm_cell_452/MatMul/ReadVariableOpЂ%lstm_cell_452/MatMul_1/ReadVariableOpЂwhileF
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
#lstm_cell_452/MatMul/ReadVariableOpReadVariableOp,lstm_cell_452_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_452/MatMul/ReadVariableOpА
lstm_cell_452/MatMulMatMulstrided_slice_2:output:0+lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/MatMulО
%lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_452_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_452/MatMul_1/ReadVariableOpЌ
lstm_cell_452/MatMul_1MatMulzeros:output:0-lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/MatMul_1Є
lstm_cell_452/addAddV2lstm_cell_452/MatMul:product:0 lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/addЗ
$lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_452_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_452/BiasAdd/ReadVariableOpБ
lstm_cell_452/BiasAddBiasAddlstm_cell_452/add:z:0,lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_452/BiasAdd
lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_452/split/split_dimї
lstm_cell_452/splitSplit&lstm_cell_452/split/split_dim:output:0lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_452/split
lstm_cell_452/SigmoidSigmoidlstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Sigmoid
lstm_cell_452/Sigmoid_1Sigmoidlstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Sigmoid_1
lstm_cell_452/mulMullstm_cell_452/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/mul
lstm_cell_452/ReluRelulstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Relu 
lstm_cell_452/mul_1Mullstm_cell_452/Sigmoid:y:0 lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/mul_1
lstm_cell_452/add_1AddV2lstm_cell_452/mul:z:0lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/add_1
lstm_cell_452/Sigmoid_2Sigmoidlstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Sigmoid_2
lstm_cell_452/Relu_1Relulstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/Relu_1Є
lstm_cell_452/mul_2Mullstm_cell_452/Sigmoid_2:y:0"lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_452/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_452_matmul_readvariableop_resource.lstm_cell_452_matmul_1_readvariableop_resource-lstm_cell_452_biasadd_readvariableop_resource*
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
while_body_18422983*
condR
while_cond_18422982*K
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
NoOpNoOp%^lstm_cell_452/BiasAdd/ReadVariableOp$^lstm_cell_452/MatMul/ReadVariableOp&^lstm_cell_452/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_452/BiasAdd/ReadVariableOp$lstm_cell_452/BiasAdd/ReadVariableOp2J
#lstm_cell_452/MatMul/ReadVariableOp#lstm_cell_452/MatMul/ReadVariableOp2N
%lstm_cell_452/MatMul_1/ReadVariableOp%lstm_cell_452/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
љ?
л
while_body_18423442
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_452_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_452_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_452_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_452_matmul_readvariableop_resource:	ШG
4while_lstm_cell_452_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_452_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_452/BiasAdd/ReadVariableOpЂ)while/lstm_cell_452/MatMul/ReadVariableOpЂ+while/lstm_cell_452/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_452/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_452_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_452/MatMul/ReadVariableOpк
while/lstm_cell_452/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_452/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/MatMulв
+while/lstm_cell_452/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_452_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_452/MatMul_1/ReadVariableOpУ
while/lstm_cell_452/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_452/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/MatMul_1М
while/lstm_cell_452/addAddV2$while/lstm_cell_452/MatMul:product:0&while/lstm_cell_452/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/addЫ
*while/lstm_cell_452/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_452_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_452/BiasAdd/ReadVariableOpЩ
while/lstm_cell_452/BiasAddBiasAddwhile/lstm_cell_452/add:z:02while/lstm_cell_452/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_452/BiasAdd
#while/lstm_cell_452/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_452/split/split_dim
while/lstm_cell_452/splitSplit,while/lstm_cell_452/split/split_dim:output:0$while/lstm_cell_452/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_452/split
while/lstm_cell_452/SigmoidSigmoid"while/lstm_cell_452/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Sigmoid
while/lstm_cell_452/Sigmoid_1Sigmoid"while/lstm_cell_452/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Sigmoid_1Ѓ
while/lstm_cell_452/mulMul!while/lstm_cell_452/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/mul
while/lstm_cell_452/ReluRelu"while/lstm_cell_452/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/ReluИ
while/lstm_cell_452/mul_1Mulwhile/lstm_cell_452/Sigmoid:y:0&while/lstm_cell_452/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/mul_1­
while/lstm_cell_452/add_1AddV2while/lstm_cell_452/mul:z:0while/lstm_cell_452/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/add_1
while/lstm_cell_452/Sigmoid_2Sigmoid"while/lstm_cell_452/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Sigmoid_2
while/lstm_cell_452/Relu_1Reluwhile/lstm_cell_452/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/Relu_1М
while/lstm_cell_452/mul_2Mul!while/lstm_cell_452/Sigmoid_2:y:0(while/lstm_cell_452/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_452/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_452/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_452/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_452/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_452/BiasAdd/ReadVariableOp*^while/lstm_cell_452/MatMul/ReadVariableOp,^while/lstm_cell_452/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_452_biasadd_readvariableop_resource5while_lstm_cell_452_biasadd_readvariableop_resource_0"n
4while_lstm_cell_452_matmul_1_readvariableop_resource6while_lstm_cell_452_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_452_matmul_readvariableop_resource4while_lstm_cell_452_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_452/BiasAdd/ReadVariableOp*while/lstm_cell_452/BiasAdd/ReadVariableOp2V
)while/lstm_cell_452/MatMul/ReadVariableOp)while/lstm_cell_452/MatMul/ReadVariableOp2Z
+while/lstm_cell_452/MatMul_1/ReadVariableOp+while/lstm_cell_452/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_cond_18418054
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_18418054___redundant_placeholder06
2while_while_cond_18418054___redundant_placeholder16
2while_while_cond_18418054___redundant_placeholder26
2while_while_cond_18418054___redundant_placeholder3
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
ЂИ
Ц 
$__inference__traced_restore_18423990
file_prefix3
!assignvariableop_dense_150_kernel:d/
!assignvariableop_1_dense_150_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: ]
Jassignvariableop_7_bidirectional_150_forward_lstm_150_lstm_cell_451_kernel:	Шg
Tassignvariableop_8_bidirectional_150_forward_lstm_150_lstm_cell_451_recurrent_kernel:	2ШW
Hassignvariableop_9_bidirectional_150_forward_lstm_150_lstm_cell_451_bias:	Ш_
Lassignvariableop_10_bidirectional_150_backward_lstm_150_lstm_cell_452_kernel:	Шi
Vassignvariableop_11_bidirectional_150_backward_lstm_150_lstm_cell_452_recurrent_kernel:	2ШY
Jassignvariableop_12_bidirectional_150_backward_lstm_150_lstm_cell_452_bias:	Ш#
assignvariableop_13_total: #
assignvariableop_14_count: =
+assignvariableop_15_adam_dense_150_kernel_m:d7
)assignvariableop_16_adam_dense_150_bias_m:e
Rassignvariableop_17_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_kernel_m:	Шo
\assignvariableop_18_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_recurrent_kernel_m:	2Ш_
Passignvariableop_19_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_bias_m:	Шf
Sassignvariableop_20_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_kernel_m:	Шp
]assignvariableop_21_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_recurrent_kernel_m:	2Ш`
Qassignvariableop_22_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_bias_m:	Ш=
+assignvariableop_23_adam_dense_150_kernel_v:d7
)assignvariableop_24_adam_dense_150_bias_v:e
Rassignvariableop_25_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_kernel_v:	Шo
\assignvariableop_26_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_recurrent_kernel_v:	2Ш_
Passignvariableop_27_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_bias_v:	Шf
Sassignvariableop_28_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_kernel_v:	Шp
]assignvariableop_29_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_recurrent_kernel_v:	2Ш`
Qassignvariableop_30_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_bias_v:	Ш@
.assignvariableop_31_adam_dense_150_kernel_vhat:d:
,assignvariableop_32_adam_dense_150_bias_vhat:h
Uassignvariableop_33_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_kernel_vhat:	Шr
_assignvariableop_34_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_recurrent_kernel_vhat:	2Шb
Sassignvariableop_35_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_bias_vhat:	Шi
Vassignvariableop_36_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_kernel_vhat:	Шs
`assignvariableop_37_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_recurrent_kernel_vhat:	2Шc
Tassignvariableop_38_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_bias_vhat:	Ш
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
AssignVariableOpAssignVariableOp!assignvariableop_dense_150_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1І
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_150_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOpJassignvariableop_7_bidirectional_150_forward_lstm_150_lstm_cell_451_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8й
AssignVariableOp_8AssignVariableOpTassignvariableop_8_bidirectional_150_forward_lstm_150_lstm_cell_451_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Э
AssignVariableOp_9AssignVariableOpHassignvariableop_9_bidirectional_150_forward_lstm_150_lstm_cell_451_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10д
AssignVariableOp_10AssignVariableOpLassignvariableop_10_bidirectional_150_backward_lstm_150_lstm_cell_452_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11о
AssignVariableOp_11AssignVariableOpVassignvariableop_11_bidirectional_150_backward_lstm_150_lstm_cell_452_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12в
AssignVariableOp_12AssignVariableOpJassignvariableop_12_bidirectional_150_backward_lstm_150_lstm_cell_452_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_dense_150_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Б
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_dense_150_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17к
AssignVariableOp_17AssignVariableOpRassignvariableop_17_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18ф
AssignVariableOp_18AssignVariableOp\assignvariableop_18_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19и
AssignVariableOp_19AssignVariableOpPassignvariableop_19_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20л
AssignVariableOp_20AssignVariableOpSassignvariableop_20_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21х
AssignVariableOp_21AssignVariableOp]assignvariableop_21_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22й
AssignVariableOp_22AssignVariableOpQassignvariableop_22_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Г
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_150_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Б
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_150_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25к
AssignVariableOp_25AssignVariableOpRassignvariableop_25_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26ф
AssignVariableOp_26AssignVariableOp\assignvariableop_26_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27и
AssignVariableOp_27AssignVariableOpPassignvariableop_27_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28л
AssignVariableOp_28AssignVariableOpSassignvariableop_28_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29х
AssignVariableOp_29AssignVariableOp]assignvariableop_29_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30й
AssignVariableOp_30AssignVariableOpQassignvariableop_30_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ж
AssignVariableOp_31AssignVariableOp.assignvariableop_31_adam_dense_150_kernel_vhatIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Д
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_dense_150_bias_vhatIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33н
AssignVariableOp_33AssignVariableOpUassignvariableop_33_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_kernel_vhatIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34ч
AssignVariableOp_34AssignVariableOp_assignvariableop_34_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_recurrent_kernel_vhatIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35л
AssignVariableOp_35AssignVariableOpSassignvariableop_35_adam_bidirectional_150_forward_lstm_150_lstm_cell_451_bias_vhatIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36о
AssignVariableOp_36AssignVariableOpVassignvariableop_36_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_kernel_vhatIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37ш
AssignVariableOp_37AssignVariableOp`assignvariableop_37_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_recurrent_kernel_vhatIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38м
AssignVariableOp_38AssignVariableOpTassignvariableop_38_adam_bidirectional_150_backward_lstm_150_lstm_cell_452_bias_vhatIdentity_38:output:0"/device:CPU:0*
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
еH

O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_18418758

inputs)
lstm_cell_452_18418676:	Ш)
lstm_cell_452_18418678:	2Ш%
lstm_cell_452_18418680:	Ш
identityЂ%lstm_cell_452/StatefulPartitionedCallЂwhileD
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
%lstm_cell_452/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_452_18418676lstm_cell_452_18418678lstm_cell_452_18418680*
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
K__inference_lstm_cell_452_layer_call_and_return_conditional_losses_184186092'
%lstm_cell_452/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_452_18418676lstm_cell_452_18418678lstm_cell_452_18418680*
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
while_body_18418689*
condR
while_cond_18418688*K
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
NoOpNoOp&^lstm_cell_452/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2N
%lstm_cell_452/StatefulPartitionedCall%lstm_cell_452/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ќ

и
1__inference_sequential_150_layer_call_fn_18420243

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
L__inference_sequential_150_layer_call_and_return_conditional_losses_184202242
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
п
Э
while_cond_18419096
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_18419096___redundant_placeholder06
2while_while_cond_18419096___redundant_placeholder16
2while_while_cond_18419096___redundant_placeholder26
2while_while_cond_18419096___redundant_placeholder3
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
e
Т
$forward_lstm_150_while_body_18420356>
:forward_lstm_150_while_forward_lstm_150_while_loop_counterD
@forward_lstm_150_while_forward_lstm_150_while_maximum_iterations&
"forward_lstm_150_while_placeholder(
$forward_lstm_150_while_placeholder_1(
$forward_lstm_150_while_placeholder_2(
$forward_lstm_150_while_placeholder_3(
$forward_lstm_150_while_placeholder_4=
9forward_lstm_150_while_forward_lstm_150_strided_slice_1_0y
uforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_150_while_greater_forward_lstm_150_cast_0X
Eforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0:	ШZ
Gforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0:	2ШU
Fforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0:	Ш#
forward_lstm_150_while_identity%
!forward_lstm_150_while_identity_1%
!forward_lstm_150_while_identity_2%
!forward_lstm_150_while_identity_3%
!forward_lstm_150_while_identity_4%
!forward_lstm_150_while_identity_5%
!forward_lstm_150_while_identity_6;
7forward_lstm_150_while_forward_lstm_150_strided_slice_1w
sforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_150_while_greater_forward_lstm_150_castV
Cforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource:	ШX
Eforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource:	2ШS
Dforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource:	ШЂ;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOpЂ:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOpЂ<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOpх
Hforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2J
Hforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
:forward_lstm_150/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_150_while_placeholderQforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02<
:forward_lstm_150/while/TensorArrayV2Read/TensorListGetItemе
forward_lstm_150/while/GreaterGreater6forward_lstm_150_while_greater_forward_lstm_150_cast_0"forward_lstm_150_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2 
forward_lstm_150/while/Greaterџ
:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOpReadVariableOpEforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp
+forward_lstm_150/while/lstm_cell_451/MatMulMatMulAforward_lstm_150/while/TensorArrayV2Read/TensorListGetItem:item:0Bforward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_150/while/lstm_cell_451/MatMul
<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOpReadVariableOpGforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp
-forward_lstm_150/while/lstm_cell_451/MatMul_1MatMul$forward_lstm_150_while_placeholder_3Dforward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-forward_lstm_150/while/lstm_cell_451/MatMul_1
(forward_lstm_150/while/lstm_cell_451/addAddV25forward_lstm_150/while/lstm_cell_451/MatMul:product:07forward_lstm_150/while/lstm_cell_451/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_150/while/lstm_cell_451/addў
;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOpReadVariableOpFforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp
,forward_lstm_150/while/lstm_cell_451/BiasAddBiasAdd,forward_lstm_150/while/lstm_cell_451/add:z:0Cforward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_150/while/lstm_cell_451/BiasAddЎ
4forward_lstm_150/while/lstm_cell_451/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4forward_lstm_150/while/lstm_cell_451/split/split_dimг
*forward_lstm_150/while/lstm_cell_451/splitSplit=forward_lstm_150/while/lstm_cell_451/split/split_dim:output:05forward_lstm_150/while/lstm_cell_451/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*forward_lstm_150/while/lstm_cell_451/splitЮ
,forward_lstm_150/while/lstm_cell_451/SigmoidSigmoid3forward_lstm_150/while/lstm_cell_451/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,forward_lstm_150/while/lstm_cell_451/Sigmoidв
.forward_lstm_150/while/lstm_cell_451/Sigmoid_1Sigmoid3forward_lstm_150/while/lstm_cell_451/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_150/while/lstm_cell_451/Sigmoid_1ч
(forward_lstm_150/while/lstm_cell_451/mulMul2forward_lstm_150/while/lstm_cell_451/Sigmoid_1:y:0$forward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_150/while/lstm_cell_451/mulХ
)forward_lstm_150/while/lstm_cell_451/ReluRelu3forward_lstm_150/while/lstm_cell_451/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_150/while/lstm_cell_451/Reluќ
*forward_lstm_150/while/lstm_cell_451/mul_1Mul0forward_lstm_150/while/lstm_cell_451/Sigmoid:y:07forward_lstm_150/while/lstm_cell_451/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_150/while/lstm_cell_451/mul_1ё
*forward_lstm_150/while/lstm_cell_451/add_1AddV2,forward_lstm_150/while/lstm_cell_451/mul:z:0.forward_lstm_150/while/lstm_cell_451/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_150/while/lstm_cell_451/add_1в
.forward_lstm_150/while/lstm_cell_451/Sigmoid_2Sigmoid3forward_lstm_150/while/lstm_cell_451/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_150/while/lstm_cell_451/Sigmoid_2Ф
+forward_lstm_150/while/lstm_cell_451/Relu_1Relu.forward_lstm_150/while/lstm_cell_451/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_150/while/lstm_cell_451/Relu_1
*forward_lstm_150/while/lstm_cell_451/mul_2Mul2forward_lstm_150/while/lstm_cell_451/Sigmoid_2:y:09forward_lstm_150/while/lstm_cell_451/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_150/while/lstm_cell_451/mul_2є
forward_lstm_150/while/SelectSelect"forward_lstm_150/while/Greater:z:0.forward_lstm_150/while/lstm_cell_451/mul_2:z:0$forward_lstm_150_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_150/while/Selectј
forward_lstm_150/while/Select_1Select"forward_lstm_150/while/Greater:z:0.forward_lstm_150/while/lstm_cell_451/mul_2:z:0$forward_lstm_150_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_150/while/Select_1ј
forward_lstm_150/while/Select_2Select"forward_lstm_150/while/Greater:z:0.forward_lstm_150/while/lstm_cell_451/add_1:z:0$forward_lstm_150_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_150/while/Select_2Ў
;forward_lstm_150/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_150_while_placeholder_1"forward_lstm_150_while_placeholder&forward_lstm_150/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_150/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_150/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_150/while/add/y­
forward_lstm_150/while/addAddV2"forward_lstm_150_while_placeholder%forward_lstm_150/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/while/add
forward_lstm_150/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_150/while/add_1/yЫ
forward_lstm_150/while/add_1AddV2:forward_lstm_150_while_forward_lstm_150_while_loop_counter'forward_lstm_150/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_150/while/add_1Џ
forward_lstm_150/while/IdentityIdentity forward_lstm_150/while/add_1:z:0^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_150/while/Identityг
!forward_lstm_150/while/Identity_1Identity@forward_lstm_150_while_forward_lstm_150_while_maximum_iterations^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_150/while/Identity_1Б
!forward_lstm_150/while/Identity_2Identityforward_lstm_150/while/add:z:0^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_150/while/Identity_2о
!forward_lstm_150/while/Identity_3IdentityKforward_lstm_150/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_150/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_150/while/Identity_3Ъ
!forward_lstm_150/while/Identity_4Identity&forward_lstm_150/while/Select:output:0^forward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_150/while/Identity_4Ь
!forward_lstm_150/while/Identity_5Identity(forward_lstm_150/while/Select_1:output:0^forward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_150/while/Identity_5Ь
!forward_lstm_150/while/Identity_6Identity(forward_lstm_150/while/Select_2:output:0^forward_lstm_150/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_150/while/Identity_6Ж
forward_lstm_150/while/NoOpNoOp<^forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp;^forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp=^forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_150/while/NoOp"t
7forward_lstm_150_while_forward_lstm_150_strided_slice_19forward_lstm_150_while_forward_lstm_150_strided_slice_1_0"n
4forward_lstm_150_while_greater_forward_lstm_150_cast6forward_lstm_150_while_greater_forward_lstm_150_cast_0"K
forward_lstm_150_while_identity(forward_lstm_150/while/Identity:output:0"O
!forward_lstm_150_while_identity_1*forward_lstm_150/while/Identity_1:output:0"O
!forward_lstm_150_while_identity_2*forward_lstm_150/while/Identity_2:output:0"O
!forward_lstm_150_while_identity_3*forward_lstm_150/while/Identity_3:output:0"O
!forward_lstm_150_while_identity_4*forward_lstm_150/while/Identity_4:output:0"O
!forward_lstm_150_while_identity_5*forward_lstm_150/while/Identity_5:output:0"O
!forward_lstm_150_while_identity_6*forward_lstm_150/while/Identity_6:output:0"
Dforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resourceFforward_lstm_150_while_lstm_cell_451_biasadd_readvariableop_resource_0"
Eforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resourceGforward_lstm_150_while_lstm_cell_451_matmul_1_readvariableop_resource_0"
Cforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resourceEforward_lstm_150_while_lstm_cell_451_matmul_readvariableop_resource_0"ь
sforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensoruforward_lstm_150_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_150_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2z
;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp;forward_lstm_150/while/lstm_cell_451/BiasAdd/ReadVariableOp2x
:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp:forward_lstm_150/while/lstm_cell_451/MatMul/ReadVariableOp2|
<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp<forward_lstm_150/while/lstm_cell_451/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
:џџџџџџџџџ"ЈL
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
	dense_1500
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
": d2dense_150/kernel
:2dense_150/bias
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
J:H	Ш27bidirectional_150/forward_lstm_150/lstm_cell_451/kernel
T:R	2Ш2Abidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel
D:BШ25bidirectional_150/forward_lstm_150/lstm_cell_451/bias
K:I	Ш28bidirectional_150/backward_lstm_150/lstm_cell_452/kernel
U:S	2Ш2Bbidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel
E:CШ26bidirectional_150/backward_lstm_150/lstm_cell_452/bias
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
':%d2Adam/dense_150/kernel/m
!:2Adam/dense_150/bias/m
O:M	Ш2>Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/m
Y:W	2Ш2HAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/m
I:GШ2<Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/m
P:N	Ш2?Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/m
Z:X	2Ш2IAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/m
J:HШ2=Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/m
':%d2Adam/dense_150/kernel/v
!:2Adam/dense_150/bias/v
O:M	Ш2>Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/v
Y:W	2Ш2HAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/v
I:GШ2<Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/v
P:N	Ш2?Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/v
Z:X	2Ш2IAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/v
J:HШ2=Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/v
*:(d2Adam/dense_150/kernel/vhat
$:"2Adam/dense_150/bias/vhat
R:P	Ш2AAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/kernel/vhat
\:Z	2Ш2KAdam/bidirectional_150/forward_lstm_150/lstm_cell_451/recurrent_kernel/vhat
L:JШ2?Adam/bidirectional_150/forward_lstm_150/lstm_cell_451/bias/vhat
S:Q	Ш2BAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/kernel/vhat
]:[	2Ш2LAdam/bidirectional_150/backward_lstm_150/lstm_cell_452/recurrent_kernel/vhat
M:KШ2@Adam/bidirectional_150/backward_lstm_150/lstm_cell_452/bias/vhat
Ќ2Љ
1__inference_sequential_150_layer_call_fn_18420243
1__inference_sequential_150_layer_call_fn_18420736Р
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
#__inference__wrapped_model_18417756args_0args_0_1"
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
L__inference_sequential_150_layer_call_and_return_conditional_losses_18420759
L__inference_sequential_150_layer_call_and_return_conditional_losses_18420782Р
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
4__inference_bidirectional_150_layer_call_fn_18420829
4__inference_bidirectional_150_layer_call_fn_18420846
4__inference_bidirectional_150_layer_call_fn_18420864
4__inference_bidirectional_150_layer_call_fn_18420882ц
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
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_18421184
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_18421486
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_18421844
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_18422202ц
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
,__inference_dense_150_layer_call_fn_18422211Ђ
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
G__inference_dense_150_layer_call_and_return_conditional_losses_18422222Ђ
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
&__inference_signature_wrapper_18420812args_0args_0_1"
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
3__inference_forward_lstm_150_layer_call_fn_18422233
3__inference_forward_lstm_150_layer_call_fn_18422244
3__inference_forward_lstm_150_layer_call_fn_18422255
3__inference_forward_lstm_150_layer_call_fn_18422266е
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
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_18422417
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_18422568
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_18422719
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_18422870е
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
4__inference_backward_lstm_150_layer_call_fn_18422881
4__inference_backward_lstm_150_layer_call_fn_18422892
4__inference_backward_lstm_150_layer_call_fn_18422903
4__inference_backward_lstm_150_layer_call_fn_18422914е
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
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_18423067
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_18423220
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_18423373
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_18423526е
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
0__inference_lstm_cell_451_layer_call_fn_18423543
0__inference_lstm_cell_451_layer_call_fn_18423560О
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
K__inference_lstm_cell_451_layer_call_and_return_conditional_losses_18423592
K__inference_lstm_cell_451_layer_call_and_return_conditional_losses_18423624О
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
0__inference_lstm_cell_452_layer_call_fn_18423641
0__inference_lstm_cell_452_layer_call_fn_18423658О
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
K__inference_lstm_cell_452_layer_call_and_return_conditional_losses_18423690
K__inference_lstm_cell_452_layer_call_and_return_conditional_losses_18423722О
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
#__inference__wrapped_model_18417756\ЂY
RЂO
MJ4Ђ1
!њџџџџџџџџџџџџџџџџџџ

`
	RaggedTensorSpec
Њ "5Њ2
0
	dense_150# 
	dense_150џџџџџџџџџа
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_18423067}OЂL
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
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_18423220}OЂL
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
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_18423373QЂN
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
O__inference_backward_lstm_150_layer_call_and_return_conditional_losses_18423526QЂN
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
4__inference_backward_lstm_150_layer_call_fn_18422881pOЂL
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
4__inference_backward_lstm_150_layer_call_fn_18422892pOЂL
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
4__inference_backward_lstm_150_layer_call_fn_18422903rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2Њ
4__inference_backward_lstm_150_layer_call_fn_18422914rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2с
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_18421184\ЂY
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
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_18421486\ЂY
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
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_18421844lЂi
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
O__inference_bidirectional_150_layer_call_and_return_conditional_losses_18422202lЂi
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
4__inference_bidirectional_150_layer_call_fn_18420829\ЂY
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
4__inference_bidirectional_150_layer_call_fn_18420846\ЂY
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
4__inference_bidirectional_150_layer_call_fn_18420864lЂi
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
4__inference_bidirectional_150_layer_call_fn_18420882lЂi
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
G__inference_dense_150_layer_call_and_return_conditional_losses_18422222\/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџ
 
,__inference_dense_150_layer_call_fn_18422211O/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџЯ
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_18422417}OЂL
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
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_18422568}OЂL
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
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_18422719QЂN
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
N__inference_forward_lstm_150_layer_call_and_return_conditional_losses_18422870QЂN
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
3__inference_forward_lstm_150_layer_call_fn_18422233pOЂL
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
3__inference_forward_lstm_150_layer_call_fn_18422244pOЂL
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
3__inference_forward_lstm_150_layer_call_fn_18422255rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2Љ
3__inference_forward_lstm_150_layer_call_fn_18422266rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2Э
K__inference_lstm_cell_451_layer_call_and_return_conditional_losses_18423592§Ђ}
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
K__inference_lstm_cell_451_layer_call_and_return_conditional_losses_18423624§Ђ}
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
0__inference_lstm_cell_451_layer_call_fn_18423543эЂ}
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
0__inference_lstm_cell_451_layer_call_fn_18423560эЂ}
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
K__inference_lstm_cell_452_layer_call_and_return_conditional_losses_18423690§Ђ}
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
K__inference_lstm_cell_452_layer_call_and_return_conditional_losses_18423722§Ђ}
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
0__inference_lstm_cell_452_layer_call_fn_18423641эЂ}
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
0__inference_lstm_cell_452_layer_call_fn_18423658эЂ}
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
L__inference_sequential_150_layer_call_and_return_conditional_losses_18420759dЂa
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
L__inference_sequential_150_layer_call_and_return_conditional_losses_18420782dЂa
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
1__inference_sequential_150_layer_call_fn_18420243dЂa
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
1__inference_sequential_150_layer_call_fn_18420736dЂa
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
&__inference_signature_wrapper_18420812ЈeЂb
Ђ 
[ЊX
*
args_0 
args_0џџџџџџџџџ
*
args_0_1
args_0_1џџџџџџџџџ	"5Њ2
0
	dense_150# 
	dense_150џџџџџџџџџ