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
dense_132/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*!
shared_namedense_132/kernel
u
$dense_132/kernel/Read/ReadVariableOpReadVariableOpdense_132/kernel*
_output_shapes

:d*
dtype0
t
dense_132/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_132/bias
m
"dense_132/bias/Read/ReadVariableOpReadVariableOpdense_132/bias*
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
7bidirectional_132/forward_lstm_132/lstm_cell_397/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*H
shared_name97bidirectional_132/forward_lstm_132/lstm_cell_397/kernel
Ф
Kbidirectional_132/forward_lstm_132/lstm_cell_397/kernel/Read/ReadVariableOpReadVariableOp7bidirectional_132/forward_lstm_132/lstm_cell_397/kernel*
_output_shapes
:	Ш*
dtype0
п
Abidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*R
shared_nameCAbidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel
и
Ubidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/Read/ReadVariableOpReadVariableOpAbidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel*
_output_shapes
:	2Ш*
dtype0
У
5bidirectional_132/forward_lstm_132/lstm_cell_397/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*F
shared_name75bidirectional_132/forward_lstm_132/lstm_cell_397/bias
М
Ibidirectional_132/forward_lstm_132/lstm_cell_397/bias/Read/ReadVariableOpReadVariableOp5bidirectional_132/forward_lstm_132/lstm_cell_397/bias*
_output_shapes	
:Ш*
dtype0
Э
8bidirectional_132/backward_lstm_132/lstm_cell_398/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*I
shared_name:8bidirectional_132/backward_lstm_132/lstm_cell_398/kernel
Ц
Lbidirectional_132/backward_lstm_132/lstm_cell_398/kernel/Read/ReadVariableOpReadVariableOp8bidirectional_132/backward_lstm_132/lstm_cell_398/kernel*
_output_shapes
:	Ш*
dtype0
с
Bbidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*S
shared_nameDBbidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel
к
Vbidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/Read/ReadVariableOpReadVariableOpBbidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel*
_output_shapes
:	2Ш*
dtype0
Х
6bidirectional_132/backward_lstm_132/lstm_cell_398/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*G
shared_name86bidirectional_132/backward_lstm_132/lstm_cell_398/bias
О
Jbidirectional_132/backward_lstm_132/lstm_cell_398/bias/Read/ReadVariableOpReadVariableOp6bidirectional_132/backward_lstm_132/lstm_cell_398/bias*
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
Adam/dense_132/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_132/kernel/m

+Adam/dense_132/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_132/kernel/m*
_output_shapes

:d*
dtype0

Adam/dense_132/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_132/bias/m
{
)Adam/dense_132/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_132/bias/m*
_output_shapes
:*
dtype0
й
>Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*O
shared_name@>Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/m
в
RAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/m/Read/ReadVariableOpReadVariableOp>Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/m*
_output_shapes
:	Ш*
dtype0
э
HAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*Y
shared_nameJHAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/m
ц
\Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpHAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/m*
_output_shapes
:	2Ш*
dtype0
б
<Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*M
shared_name><Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/m
Ъ
PAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/m/Read/ReadVariableOpReadVariableOp<Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/m*
_output_shapes	
:Ш*
dtype0
л
?Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*P
shared_nameA?Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/m
д
SAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/m/Read/ReadVariableOpReadVariableOp?Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/m*
_output_shapes
:	Ш*
dtype0
я
IAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*Z
shared_nameKIAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/m
ш
]Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpIAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/m*
_output_shapes
:	2Ш*
dtype0
г
=Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*N
shared_name?=Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/m
Ь
QAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/m/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/m*
_output_shapes	
:Ш*
dtype0

Adam/dense_132/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_132/kernel/v

+Adam/dense_132/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_132/kernel/v*
_output_shapes

:d*
dtype0

Adam/dense_132/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_132/bias/v
{
)Adam/dense_132/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_132/bias/v*
_output_shapes
:*
dtype0
й
>Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*O
shared_name@>Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/v
в
RAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/v/Read/ReadVariableOpReadVariableOp>Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/v*
_output_shapes
:	Ш*
dtype0
э
HAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*Y
shared_nameJHAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/v
ц
\Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpHAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/v*
_output_shapes
:	2Ш*
dtype0
б
<Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*M
shared_name><Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/v
Ъ
PAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/v/Read/ReadVariableOpReadVariableOp<Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/v*
_output_shapes	
:Ш*
dtype0
л
?Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*P
shared_nameA?Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/v
д
SAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/v/Read/ReadVariableOpReadVariableOp?Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/v*
_output_shapes
:	Ш*
dtype0
я
IAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*Z
shared_nameKIAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/v
ш
]Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpIAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/v*
_output_shapes
:	2Ш*
dtype0
г
=Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*N
shared_name?=Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/v
Ь
QAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/v/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/v*
_output_shapes	
:Ш*
dtype0

Adam/dense_132/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*+
shared_nameAdam/dense_132/kernel/vhat

.Adam/dense_132/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_132/kernel/vhat*
_output_shapes

:d*
dtype0

Adam/dense_132/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/dense_132/bias/vhat

,Adam/dense_132/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_132/bias/vhat*
_output_shapes
:*
dtype0
п
AAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*R
shared_nameCAAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/vhat
и
UAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/vhat/Read/ReadVariableOpReadVariableOpAAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/vhat*
_output_shapes
:	Ш*
dtype0
ѓ
KAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*\
shared_nameMKAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/vhat
ь
_Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpKAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/vhat*
_output_shapes
:	2Ш*
dtype0
з
?Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*P
shared_nameA?Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/vhat
а
SAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/vhat/Read/ReadVariableOpReadVariableOp?Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/vhat*
_output_shapes	
:Ш*
dtype0
с
BAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*S
shared_nameDBAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/vhat
к
VAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/vhat/Read/ReadVariableOpReadVariableOpBAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/vhat*
_output_shapes
:	Ш*
dtype0
ѕ
LAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*]
shared_nameNLAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/vhat
ю
`Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpLAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/vhat*
_output_shapes
:	2Ш*
dtype0
й
@Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*Q
shared_nameB@Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/vhat
в
TAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/vhat/Read/ReadVariableOpReadVariableOp@Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/vhat*
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
VARIABLE_VALUEdense_132/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_132/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUE7bidirectional_132/forward_lstm_132/lstm_cell_397/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAbidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5bidirectional_132/forward_lstm_132/lstm_cell_397/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE8bidirectional_132/backward_lstm_132/lstm_cell_398/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEBbidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6bidirectional_132/backward_lstm_132/lstm_cell_398/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_132/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_132/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ё
VARIABLE_VALUEHAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ђ
VARIABLE_VALUEIAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_132/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_132/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ё
VARIABLE_VALUEHAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ђ
VARIABLE_VALUEIAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_132/kernel/vhatUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_132/bias/vhatSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/vhatEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ЇЄ
VARIABLE_VALUEKAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/vhatEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/vhatEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEBAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/vhatEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ЈЅ
VARIABLE_VALUELAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/vhatEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE@Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/vhatEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_0_17bidirectional_132/forward_lstm_132/lstm_cell_397/kernelAbidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel5bidirectional_132/forward_lstm_132/lstm_cell_397/bias8bidirectional_132/backward_lstm_132/lstm_cell_398/kernelBbidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel6bidirectional_132/backward_lstm_132/lstm_cell_398/biasdense_132/kerneldense_132/bias*
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
&__inference_signature_wrapper_16716654
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_132/kernel/Read/ReadVariableOp"dense_132/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpKbidirectional_132/forward_lstm_132/lstm_cell_397/kernel/Read/ReadVariableOpUbidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/Read/ReadVariableOpIbidirectional_132/forward_lstm_132/lstm_cell_397/bias/Read/ReadVariableOpLbidirectional_132/backward_lstm_132/lstm_cell_398/kernel/Read/ReadVariableOpVbidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/Read/ReadVariableOpJbidirectional_132/backward_lstm_132/lstm_cell_398/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_132/kernel/m/Read/ReadVariableOp)Adam/dense_132/bias/m/Read/ReadVariableOpRAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/m/Read/ReadVariableOp\Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/m/Read/ReadVariableOpPAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/m/Read/ReadVariableOpSAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/m/Read/ReadVariableOp]Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/m/Read/ReadVariableOpQAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/m/Read/ReadVariableOp+Adam/dense_132/kernel/v/Read/ReadVariableOp)Adam/dense_132/bias/v/Read/ReadVariableOpRAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/v/Read/ReadVariableOp\Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/v/Read/ReadVariableOpPAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/v/Read/ReadVariableOpSAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/v/Read/ReadVariableOp]Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/v/Read/ReadVariableOpQAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/v/Read/ReadVariableOp.Adam/dense_132/kernel/vhat/Read/ReadVariableOp,Adam/dense_132/bias/vhat/Read/ReadVariableOpUAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/vhat/Read/ReadVariableOp_Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/vhat/Read/ReadVariableOpSAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/vhat/Read/ReadVariableOpVAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/vhat/Read/ReadVariableOp`Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/vhat/Read/ReadVariableOpTAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/vhat/Read/ReadVariableOpConst*4
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
!__inference__traced_save_16719705
ў
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_132/kerneldense_132/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate7bidirectional_132/forward_lstm_132/lstm_cell_397/kernelAbidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel5bidirectional_132/forward_lstm_132/lstm_cell_397/bias8bidirectional_132/backward_lstm_132/lstm_cell_398/kernelBbidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel6bidirectional_132/backward_lstm_132/lstm_cell_398/biastotalcountAdam/dense_132/kernel/mAdam/dense_132/bias/m>Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/mHAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/m<Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/m?Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/mIAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/m=Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/mAdam/dense_132/kernel/vAdam/dense_132/bias/v>Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/vHAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/v<Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/v?Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/vIAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/v=Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/vAdam/dense_132/kernel/vhatAdam/dense_132/bias/vhatAAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/vhatKAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/vhat?Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/vhatBAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/vhatLAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/vhat@Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/vhat*3
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
$__inference__traced_restore_16719832Њї8
ъ
М
%backward_lstm_132_while_cond_16716376@
<backward_lstm_132_while_backward_lstm_132_while_loop_counterF
Bbackward_lstm_132_while_backward_lstm_132_while_maximum_iterations'
#backward_lstm_132_while_placeholder)
%backward_lstm_132_while_placeholder_1)
%backward_lstm_132_while_placeholder_2)
%backward_lstm_132_while_placeholder_3)
%backward_lstm_132_while_placeholder_4B
>backward_lstm_132_while_less_backward_lstm_132_strided_slice_1Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16716376___redundant_placeholder0Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16716376___redundant_placeholder1Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16716376___redundant_placeholder2Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16716376___redundant_placeholder3Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16716376___redundant_placeholder4$
 backward_lstm_132_while_identity
Ъ
backward_lstm_132/while/LessLess#backward_lstm_132_while_placeholder>backward_lstm_132_while_less_backward_lstm_132_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_132/while/Less
 backward_lstm_132/while/IdentityIdentity backward_lstm_132/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_132/while/Identity"M
 backward_lstm_132_while_identity)backward_lstm_132/while/Identity:output:0*(
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
while_cond_16718325
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16718325___redundant_placeholder06
2while_while_cond_16718325___redundant_placeholder16
2while_while_cond_16718325___redundant_placeholder26
2while_while_cond_16718325___redundant_placeholder3
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
$forward_lstm_132_while_body_16717093>
:forward_lstm_132_while_forward_lstm_132_while_loop_counterD
@forward_lstm_132_while_forward_lstm_132_while_maximum_iterations&
"forward_lstm_132_while_placeholder(
$forward_lstm_132_while_placeholder_1(
$forward_lstm_132_while_placeholder_2(
$forward_lstm_132_while_placeholder_3=
9forward_lstm_132_while_forward_lstm_132_strided_slice_1_0y
uforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0X
Eforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0:	ШZ
Gforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0:	2ШU
Fforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0:	Ш#
forward_lstm_132_while_identity%
!forward_lstm_132_while_identity_1%
!forward_lstm_132_while_identity_2%
!forward_lstm_132_while_identity_3%
!forward_lstm_132_while_identity_4%
!forward_lstm_132_while_identity_5;
7forward_lstm_132_while_forward_lstm_132_strided_slice_1w
sforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensorV
Cforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource:	ШX
Eforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource:	2ШS
Dforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource:	ШЂ;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOpЂ:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOpЂ<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOpх
Hforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2J
Hforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeТ
:forward_lstm_132/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_132_while_placeholderQforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02<
:forward_lstm_132/while/TensorArrayV2Read/TensorListGetItemџ
:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOpReadVariableOpEforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp
+forward_lstm_132/while/lstm_cell_397/MatMulMatMulAforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem:item:0Bforward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_132/while/lstm_cell_397/MatMul
<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOpGforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp
-forward_lstm_132/while/lstm_cell_397/MatMul_1MatMul$forward_lstm_132_while_placeholder_2Dforward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-forward_lstm_132/while/lstm_cell_397/MatMul_1
(forward_lstm_132/while/lstm_cell_397/addAddV25forward_lstm_132/while/lstm_cell_397/MatMul:product:07forward_lstm_132/while/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_132/while/lstm_cell_397/addў
;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOpFforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp
,forward_lstm_132/while/lstm_cell_397/BiasAddBiasAdd,forward_lstm_132/while/lstm_cell_397/add:z:0Cforward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_132/while/lstm_cell_397/BiasAddЎ
4forward_lstm_132/while/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4forward_lstm_132/while/lstm_cell_397/split/split_dimг
*forward_lstm_132/while/lstm_cell_397/splitSplit=forward_lstm_132/while/lstm_cell_397/split/split_dim:output:05forward_lstm_132/while/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*forward_lstm_132/while/lstm_cell_397/splitЮ
,forward_lstm_132/while/lstm_cell_397/SigmoidSigmoid3forward_lstm_132/while/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,forward_lstm_132/while/lstm_cell_397/Sigmoidв
.forward_lstm_132/while/lstm_cell_397/Sigmoid_1Sigmoid3forward_lstm_132/while/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_132/while/lstm_cell_397/Sigmoid_1ч
(forward_lstm_132/while/lstm_cell_397/mulMul2forward_lstm_132/while/lstm_cell_397/Sigmoid_1:y:0$forward_lstm_132_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_132/while/lstm_cell_397/mulХ
)forward_lstm_132/while/lstm_cell_397/ReluRelu3forward_lstm_132/while/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_132/while/lstm_cell_397/Reluќ
*forward_lstm_132/while/lstm_cell_397/mul_1Mul0forward_lstm_132/while/lstm_cell_397/Sigmoid:y:07forward_lstm_132/while/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_132/while/lstm_cell_397/mul_1ё
*forward_lstm_132/while/lstm_cell_397/add_1AddV2,forward_lstm_132/while/lstm_cell_397/mul:z:0.forward_lstm_132/while/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_132/while/lstm_cell_397/add_1в
.forward_lstm_132/while/lstm_cell_397/Sigmoid_2Sigmoid3forward_lstm_132/while/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_132/while/lstm_cell_397/Sigmoid_2Ф
+forward_lstm_132/while/lstm_cell_397/Relu_1Relu.forward_lstm_132/while/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_132/while/lstm_cell_397/Relu_1
*forward_lstm_132/while/lstm_cell_397/mul_2Mul2forward_lstm_132/while/lstm_cell_397/Sigmoid_2:y:09forward_lstm_132/while/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_132/while/lstm_cell_397/mul_2Ж
;forward_lstm_132/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_132_while_placeholder_1"forward_lstm_132_while_placeholder.forward_lstm_132/while/lstm_cell_397/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_132/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_132/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_132/while/add/y­
forward_lstm_132/while/addAddV2"forward_lstm_132_while_placeholder%forward_lstm_132/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/while/add
forward_lstm_132/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_132/while/add_1/yЫ
forward_lstm_132/while/add_1AddV2:forward_lstm_132_while_forward_lstm_132_while_loop_counter'forward_lstm_132/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/while/add_1Џ
forward_lstm_132/while/IdentityIdentity forward_lstm_132/while/add_1:z:0^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_132/while/Identityг
!forward_lstm_132/while/Identity_1Identity@forward_lstm_132_while_forward_lstm_132_while_maximum_iterations^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_132/while/Identity_1Б
!forward_lstm_132/while/Identity_2Identityforward_lstm_132/while/add:z:0^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_132/while/Identity_2о
!forward_lstm_132/while/Identity_3IdentityKforward_lstm_132/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_132/while/Identity_3в
!forward_lstm_132/while/Identity_4Identity.forward_lstm_132/while/lstm_cell_397/mul_2:z:0^forward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_132/while/Identity_4в
!forward_lstm_132/while/Identity_5Identity.forward_lstm_132/while/lstm_cell_397/add_1:z:0^forward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_132/while/Identity_5Ж
forward_lstm_132/while/NoOpNoOp<^forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp;^forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp=^forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_132/while/NoOp"t
7forward_lstm_132_while_forward_lstm_132_strided_slice_19forward_lstm_132_while_forward_lstm_132_strided_slice_1_0"K
forward_lstm_132_while_identity(forward_lstm_132/while/Identity:output:0"O
!forward_lstm_132_while_identity_1*forward_lstm_132/while/Identity_1:output:0"O
!forward_lstm_132_while_identity_2*forward_lstm_132/while/Identity_2:output:0"O
!forward_lstm_132_while_identity_3*forward_lstm_132/while/Identity_3:output:0"O
!forward_lstm_132_while_identity_4*forward_lstm_132/while/Identity_4:output:0"O
!forward_lstm_132_while_identity_5*forward_lstm_132/while/Identity_5:output:0"
Dforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resourceFforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0"
Eforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resourceGforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0"
Cforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resourceEforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0"ь
sforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensoruforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2z
;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp2x
:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp2|
<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
4__inference_backward_lstm_132_layer_call_fn_16718723
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
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_167143882
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
Џc
ц
!__inference__traced_save_16719705
file_prefix/
+savev2_dense_132_kernel_read_readvariableop-
)savev2_dense_132_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopV
Rsavev2_bidirectional_132_forward_lstm_132_lstm_cell_397_kernel_read_readvariableop`
\savev2_bidirectional_132_forward_lstm_132_lstm_cell_397_recurrent_kernel_read_readvariableopT
Psavev2_bidirectional_132_forward_lstm_132_lstm_cell_397_bias_read_readvariableopW
Ssavev2_bidirectional_132_backward_lstm_132_lstm_cell_398_kernel_read_readvariableopa
]savev2_bidirectional_132_backward_lstm_132_lstm_cell_398_recurrent_kernel_read_readvariableopU
Qsavev2_bidirectional_132_backward_lstm_132_lstm_cell_398_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_132_kernel_m_read_readvariableop4
0savev2_adam_dense_132_bias_m_read_readvariableop]
Ysavev2_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_kernel_m_read_readvariableopg
csavev2_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_recurrent_kernel_m_read_readvariableop[
Wsavev2_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_bias_m_read_readvariableop^
Zsavev2_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_kernel_m_read_readvariableoph
dsavev2_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_recurrent_kernel_m_read_readvariableop\
Xsavev2_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_bias_m_read_readvariableop6
2savev2_adam_dense_132_kernel_v_read_readvariableop4
0savev2_adam_dense_132_bias_v_read_readvariableop]
Ysavev2_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_kernel_v_read_readvariableopg
csavev2_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_recurrent_kernel_v_read_readvariableop[
Wsavev2_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_bias_v_read_readvariableop^
Zsavev2_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_kernel_v_read_readvariableoph
dsavev2_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_recurrent_kernel_v_read_readvariableop\
Xsavev2_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_bias_v_read_readvariableop9
5savev2_adam_dense_132_kernel_vhat_read_readvariableop7
3savev2_adam_dense_132_bias_vhat_read_readvariableop`
\savev2_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_kernel_vhat_read_readvariableopj
fsavev2_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_recurrent_kernel_vhat_read_readvariableop^
Zsavev2_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_bias_vhat_read_readvariableopa
]savev2_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_kernel_vhat_read_readvariableopk
gsavev2_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_recurrent_kernel_vhat_read_readvariableop_
[savev2_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_bias_vhat_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_132_kernel_read_readvariableop)savev2_dense_132_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopRsavev2_bidirectional_132_forward_lstm_132_lstm_cell_397_kernel_read_readvariableop\savev2_bidirectional_132_forward_lstm_132_lstm_cell_397_recurrent_kernel_read_readvariableopPsavev2_bidirectional_132_forward_lstm_132_lstm_cell_397_bias_read_readvariableopSsavev2_bidirectional_132_backward_lstm_132_lstm_cell_398_kernel_read_readvariableop]savev2_bidirectional_132_backward_lstm_132_lstm_cell_398_recurrent_kernel_read_readvariableopQsavev2_bidirectional_132_backward_lstm_132_lstm_cell_398_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_132_kernel_m_read_readvariableop0savev2_adam_dense_132_bias_m_read_readvariableopYsavev2_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_kernel_m_read_readvariableopcsavev2_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_recurrent_kernel_m_read_readvariableopWsavev2_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_bias_m_read_readvariableopZsavev2_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_kernel_m_read_readvariableopdsavev2_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_recurrent_kernel_m_read_readvariableopXsavev2_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_bias_m_read_readvariableop2savev2_adam_dense_132_kernel_v_read_readvariableop0savev2_adam_dense_132_bias_v_read_readvariableopYsavev2_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_kernel_v_read_readvariableopcsavev2_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_recurrent_kernel_v_read_readvariableopWsavev2_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_bias_v_read_readvariableopZsavev2_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_kernel_v_read_readvariableopdsavev2_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_recurrent_kernel_v_read_readvariableopXsavev2_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_bias_v_read_readvariableop5savev2_adam_dense_132_kernel_vhat_read_readvariableop3savev2_adam_dense_132_bias_vhat_read_readvariableop\savev2_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_kernel_vhat_read_readvariableopfsavev2_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_recurrent_kernel_vhat_read_readvariableopZsavev2_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_bias_vhat_read_readvariableop]savev2_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_kernel_vhat_read_readvariableopgsavev2_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_recurrent_kernel_vhat_read_readvariableop[savev2_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_bias_vhat_read_readvariableopsavev2_const"/device:CPU:0*
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
Щ
Ѕ
$forward_lstm_132_while_cond_16716197>
:forward_lstm_132_while_forward_lstm_132_while_loop_counterD
@forward_lstm_132_while_forward_lstm_132_while_maximum_iterations&
"forward_lstm_132_while_placeholder(
$forward_lstm_132_while_placeholder_1(
$forward_lstm_132_while_placeholder_2(
$forward_lstm_132_while_placeholder_3(
$forward_lstm_132_while_placeholder_4@
<forward_lstm_132_while_less_forward_lstm_132_strided_slice_1X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16716197___redundant_placeholder0X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16716197___redundant_placeholder1X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16716197___redundant_placeholder2X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16716197___redundant_placeholder3X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16716197___redundant_placeholder4#
forward_lstm_132_while_identity
Х
forward_lstm_132/while/LessLess"forward_lstm_132_while_placeholder<forward_lstm_132_while_less_forward_lstm_132_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_132/while/Less
forward_lstm_132/while/IdentityIdentityforward_lstm_132/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_132/while/Identity"K
forward_lstm_132_while_identity(forward_lstm_132/while/Identity:output:0*(
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
$forward_lstm_132_while_cond_16717767>
:forward_lstm_132_while_forward_lstm_132_while_loop_counterD
@forward_lstm_132_while_forward_lstm_132_while_maximum_iterations&
"forward_lstm_132_while_placeholder(
$forward_lstm_132_while_placeholder_1(
$forward_lstm_132_while_placeholder_2(
$forward_lstm_132_while_placeholder_3(
$forward_lstm_132_while_placeholder_4@
<forward_lstm_132_while_less_forward_lstm_132_strided_slice_1X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16717767___redundant_placeholder0X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16717767___redundant_placeholder1X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16717767___redundant_placeholder2X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16717767___redundant_placeholder3X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16717767___redundant_placeholder4#
forward_lstm_132_while_identity
Х
forward_lstm_132/while/LessLess"forward_lstm_132_while_placeholder<forward_lstm_132_while_less_forward_lstm_132_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_132/while/Less
forward_lstm_132/while/IdentityIdentityforward_lstm_132/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_132/while/Identity"K
forward_lstm_132_while_identity(forward_lstm_132/while/Identity:output:0*(
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
X
ћ
$forward_lstm_132_while_body_16716791>
:forward_lstm_132_while_forward_lstm_132_while_loop_counterD
@forward_lstm_132_while_forward_lstm_132_while_maximum_iterations&
"forward_lstm_132_while_placeholder(
$forward_lstm_132_while_placeholder_1(
$forward_lstm_132_while_placeholder_2(
$forward_lstm_132_while_placeholder_3=
9forward_lstm_132_while_forward_lstm_132_strided_slice_1_0y
uforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0X
Eforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0:	ШZ
Gforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0:	2ШU
Fforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0:	Ш#
forward_lstm_132_while_identity%
!forward_lstm_132_while_identity_1%
!forward_lstm_132_while_identity_2%
!forward_lstm_132_while_identity_3%
!forward_lstm_132_while_identity_4%
!forward_lstm_132_while_identity_5;
7forward_lstm_132_while_forward_lstm_132_strided_slice_1w
sforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensorV
Cforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource:	ШX
Eforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource:	2ШS
Dforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource:	ШЂ;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOpЂ:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOpЂ<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOpх
Hforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2J
Hforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeТ
:forward_lstm_132/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_132_while_placeholderQforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02<
:forward_lstm_132/while/TensorArrayV2Read/TensorListGetItemџ
:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOpReadVariableOpEforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp
+forward_lstm_132/while/lstm_cell_397/MatMulMatMulAforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem:item:0Bforward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_132/while/lstm_cell_397/MatMul
<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOpGforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp
-forward_lstm_132/while/lstm_cell_397/MatMul_1MatMul$forward_lstm_132_while_placeholder_2Dforward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-forward_lstm_132/while/lstm_cell_397/MatMul_1
(forward_lstm_132/while/lstm_cell_397/addAddV25forward_lstm_132/while/lstm_cell_397/MatMul:product:07forward_lstm_132/while/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_132/while/lstm_cell_397/addў
;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOpFforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp
,forward_lstm_132/while/lstm_cell_397/BiasAddBiasAdd,forward_lstm_132/while/lstm_cell_397/add:z:0Cforward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_132/while/lstm_cell_397/BiasAddЎ
4forward_lstm_132/while/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4forward_lstm_132/while/lstm_cell_397/split/split_dimг
*forward_lstm_132/while/lstm_cell_397/splitSplit=forward_lstm_132/while/lstm_cell_397/split/split_dim:output:05forward_lstm_132/while/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*forward_lstm_132/while/lstm_cell_397/splitЮ
,forward_lstm_132/while/lstm_cell_397/SigmoidSigmoid3forward_lstm_132/while/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,forward_lstm_132/while/lstm_cell_397/Sigmoidв
.forward_lstm_132/while/lstm_cell_397/Sigmoid_1Sigmoid3forward_lstm_132/while/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_132/while/lstm_cell_397/Sigmoid_1ч
(forward_lstm_132/while/lstm_cell_397/mulMul2forward_lstm_132/while/lstm_cell_397/Sigmoid_1:y:0$forward_lstm_132_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_132/while/lstm_cell_397/mulХ
)forward_lstm_132/while/lstm_cell_397/ReluRelu3forward_lstm_132/while/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_132/while/lstm_cell_397/Reluќ
*forward_lstm_132/while/lstm_cell_397/mul_1Mul0forward_lstm_132/while/lstm_cell_397/Sigmoid:y:07forward_lstm_132/while/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_132/while/lstm_cell_397/mul_1ё
*forward_lstm_132/while/lstm_cell_397/add_1AddV2,forward_lstm_132/while/lstm_cell_397/mul:z:0.forward_lstm_132/while/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_132/while/lstm_cell_397/add_1в
.forward_lstm_132/while/lstm_cell_397/Sigmoid_2Sigmoid3forward_lstm_132/while/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_132/while/lstm_cell_397/Sigmoid_2Ф
+forward_lstm_132/while/lstm_cell_397/Relu_1Relu.forward_lstm_132/while/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_132/while/lstm_cell_397/Relu_1
*forward_lstm_132/while/lstm_cell_397/mul_2Mul2forward_lstm_132/while/lstm_cell_397/Sigmoid_2:y:09forward_lstm_132/while/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_132/while/lstm_cell_397/mul_2Ж
;forward_lstm_132/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_132_while_placeholder_1"forward_lstm_132_while_placeholder.forward_lstm_132/while/lstm_cell_397/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_132/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_132/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_132/while/add/y­
forward_lstm_132/while/addAddV2"forward_lstm_132_while_placeholder%forward_lstm_132/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/while/add
forward_lstm_132/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_132/while/add_1/yЫ
forward_lstm_132/while/add_1AddV2:forward_lstm_132_while_forward_lstm_132_while_loop_counter'forward_lstm_132/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/while/add_1Џ
forward_lstm_132/while/IdentityIdentity forward_lstm_132/while/add_1:z:0^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_132/while/Identityг
!forward_lstm_132/while/Identity_1Identity@forward_lstm_132_while_forward_lstm_132_while_maximum_iterations^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_132/while/Identity_1Б
!forward_lstm_132/while/Identity_2Identityforward_lstm_132/while/add:z:0^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_132/while/Identity_2о
!forward_lstm_132/while/Identity_3IdentityKforward_lstm_132/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_132/while/Identity_3в
!forward_lstm_132/while/Identity_4Identity.forward_lstm_132/while/lstm_cell_397/mul_2:z:0^forward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_132/while/Identity_4в
!forward_lstm_132/while/Identity_5Identity.forward_lstm_132/while/lstm_cell_397/add_1:z:0^forward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_132/while/Identity_5Ж
forward_lstm_132/while/NoOpNoOp<^forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp;^forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp=^forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_132/while/NoOp"t
7forward_lstm_132_while_forward_lstm_132_strided_slice_19forward_lstm_132_while_forward_lstm_132_strided_slice_1_0"K
forward_lstm_132_while_identity(forward_lstm_132/while/Identity:output:0"O
!forward_lstm_132_while_identity_1*forward_lstm_132/while/Identity_1:output:0"O
!forward_lstm_132_while_identity_2*forward_lstm_132/while/Identity_2:output:0"O
!forward_lstm_132_while_identity_3*forward_lstm_132/while/Identity_3:output:0"O
!forward_lstm_132_while_identity_4*forward_lstm_132/while/Identity_4:output:0"O
!forward_lstm_132_while_identity_5*forward_lstm_132/while/Identity_5:output:0"
Dforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resourceFforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0"
Eforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resourceGforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0"
Cforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resourceEforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0"ь
sforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensoruforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2z
;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp2x
:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp2|
<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
$forward_lstm_132_while_body_16717768>
:forward_lstm_132_while_forward_lstm_132_while_loop_counterD
@forward_lstm_132_while_forward_lstm_132_while_maximum_iterations&
"forward_lstm_132_while_placeholder(
$forward_lstm_132_while_placeholder_1(
$forward_lstm_132_while_placeholder_2(
$forward_lstm_132_while_placeholder_3(
$forward_lstm_132_while_placeholder_4=
9forward_lstm_132_while_forward_lstm_132_strided_slice_1_0y
uforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_132_while_greater_forward_lstm_132_cast_0X
Eforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0:	ШZ
Gforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0:	2ШU
Fforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0:	Ш#
forward_lstm_132_while_identity%
!forward_lstm_132_while_identity_1%
!forward_lstm_132_while_identity_2%
!forward_lstm_132_while_identity_3%
!forward_lstm_132_while_identity_4%
!forward_lstm_132_while_identity_5%
!forward_lstm_132_while_identity_6;
7forward_lstm_132_while_forward_lstm_132_strided_slice_1w
sforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_132_while_greater_forward_lstm_132_castV
Cforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource:	ШX
Eforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource:	2ШS
Dforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource:	ШЂ;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOpЂ:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOpЂ<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOpх
Hforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2J
Hforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
:forward_lstm_132/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_132_while_placeholderQforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02<
:forward_lstm_132/while/TensorArrayV2Read/TensorListGetItemе
forward_lstm_132/while/GreaterGreater6forward_lstm_132_while_greater_forward_lstm_132_cast_0"forward_lstm_132_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2 
forward_lstm_132/while/Greaterџ
:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOpReadVariableOpEforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp
+forward_lstm_132/while/lstm_cell_397/MatMulMatMulAforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem:item:0Bforward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_132/while/lstm_cell_397/MatMul
<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOpGforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp
-forward_lstm_132/while/lstm_cell_397/MatMul_1MatMul$forward_lstm_132_while_placeholder_3Dforward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-forward_lstm_132/while/lstm_cell_397/MatMul_1
(forward_lstm_132/while/lstm_cell_397/addAddV25forward_lstm_132/while/lstm_cell_397/MatMul:product:07forward_lstm_132/while/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_132/while/lstm_cell_397/addў
;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOpFforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp
,forward_lstm_132/while/lstm_cell_397/BiasAddBiasAdd,forward_lstm_132/while/lstm_cell_397/add:z:0Cforward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_132/while/lstm_cell_397/BiasAddЎ
4forward_lstm_132/while/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4forward_lstm_132/while/lstm_cell_397/split/split_dimг
*forward_lstm_132/while/lstm_cell_397/splitSplit=forward_lstm_132/while/lstm_cell_397/split/split_dim:output:05forward_lstm_132/while/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*forward_lstm_132/while/lstm_cell_397/splitЮ
,forward_lstm_132/while/lstm_cell_397/SigmoidSigmoid3forward_lstm_132/while/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,forward_lstm_132/while/lstm_cell_397/Sigmoidв
.forward_lstm_132/while/lstm_cell_397/Sigmoid_1Sigmoid3forward_lstm_132/while/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_132/while/lstm_cell_397/Sigmoid_1ч
(forward_lstm_132/while/lstm_cell_397/mulMul2forward_lstm_132/while/lstm_cell_397/Sigmoid_1:y:0$forward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_132/while/lstm_cell_397/mulХ
)forward_lstm_132/while/lstm_cell_397/ReluRelu3forward_lstm_132/while/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_132/while/lstm_cell_397/Reluќ
*forward_lstm_132/while/lstm_cell_397/mul_1Mul0forward_lstm_132/while/lstm_cell_397/Sigmoid:y:07forward_lstm_132/while/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_132/while/lstm_cell_397/mul_1ё
*forward_lstm_132/while/lstm_cell_397/add_1AddV2,forward_lstm_132/while/lstm_cell_397/mul:z:0.forward_lstm_132/while/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_132/while/lstm_cell_397/add_1в
.forward_lstm_132/while/lstm_cell_397/Sigmoid_2Sigmoid3forward_lstm_132/while/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_132/while/lstm_cell_397/Sigmoid_2Ф
+forward_lstm_132/while/lstm_cell_397/Relu_1Relu.forward_lstm_132/while/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_132/while/lstm_cell_397/Relu_1
*forward_lstm_132/while/lstm_cell_397/mul_2Mul2forward_lstm_132/while/lstm_cell_397/Sigmoid_2:y:09forward_lstm_132/while/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_132/while/lstm_cell_397/mul_2є
forward_lstm_132/while/SelectSelect"forward_lstm_132/while/Greater:z:0.forward_lstm_132/while/lstm_cell_397/mul_2:z:0$forward_lstm_132_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/while/Selectј
forward_lstm_132/while/Select_1Select"forward_lstm_132/while/Greater:z:0.forward_lstm_132/while/lstm_cell_397/mul_2:z:0$forward_lstm_132_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_132/while/Select_1ј
forward_lstm_132/while/Select_2Select"forward_lstm_132/while/Greater:z:0.forward_lstm_132/while/lstm_cell_397/add_1:z:0$forward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_132/while/Select_2Ў
;forward_lstm_132/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_132_while_placeholder_1"forward_lstm_132_while_placeholder&forward_lstm_132/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_132/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_132/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_132/while/add/y­
forward_lstm_132/while/addAddV2"forward_lstm_132_while_placeholder%forward_lstm_132/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/while/add
forward_lstm_132/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_132/while/add_1/yЫ
forward_lstm_132/while/add_1AddV2:forward_lstm_132_while_forward_lstm_132_while_loop_counter'forward_lstm_132/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/while/add_1Џ
forward_lstm_132/while/IdentityIdentity forward_lstm_132/while/add_1:z:0^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_132/while/Identityг
!forward_lstm_132/while/Identity_1Identity@forward_lstm_132_while_forward_lstm_132_while_maximum_iterations^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_132/while/Identity_1Б
!forward_lstm_132/while/Identity_2Identityforward_lstm_132/while/add:z:0^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_132/while/Identity_2о
!forward_lstm_132/while/Identity_3IdentityKforward_lstm_132/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_132/while/Identity_3Ъ
!forward_lstm_132/while/Identity_4Identity&forward_lstm_132/while/Select:output:0^forward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_132/while/Identity_4Ь
!forward_lstm_132/while/Identity_5Identity(forward_lstm_132/while/Select_1:output:0^forward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_132/while/Identity_5Ь
!forward_lstm_132/while/Identity_6Identity(forward_lstm_132/while/Select_2:output:0^forward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_132/while/Identity_6Ж
forward_lstm_132/while/NoOpNoOp<^forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp;^forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp=^forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_132/while/NoOp"t
7forward_lstm_132_while_forward_lstm_132_strided_slice_19forward_lstm_132_while_forward_lstm_132_strided_slice_1_0"n
4forward_lstm_132_while_greater_forward_lstm_132_cast6forward_lstm_132_while_greater_forward_lstm_132_cast_0"K
forward_lstm_132_while_identity(forward_lstm_132/while/Identity:output:0"O
!forward_lstm_132_while_identity_1*forward_lstm_132/while/Identity_1:output:0"O
!forward_lstm_132_while_identity_2*forward_lstm_132/while/Identity_2:output:0"O
!forward_lstm_132_while_identity_3*forward_lstm_132/while/Identity_3:output:0"O
!forward_lstm_132_while_identity_4*forward_lstm_132/while/Identity_4:output:0"O
!forward_lstm_132_while_identity_5*forward_lstm_132/while/Identity_5:output:0"O
!forward_lstm_132_while_identity_6*forward_lstm_132/while/Identity_6:output:0"
Dforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resourceFforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0"
Eforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resourceGforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0"
Cforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resourceEforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0"ь
sforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensoruforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2z
;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp2x
:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp2|
<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
д
Т
3__inference_forward_lstm_132_layer_call_fn_16718086
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
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_167139662
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
Ў

Ѕ
4__inference_bidirectional_132_layer_call_fn_16716724

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
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_167164742
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
п
Э
while_cond_16714530
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16714530___redundant_placeholder06
2while_while_cond_16714530___redundant_placeholder16
2while_while_cond_16714530___redundant_placeholder26
2while_while_cond_16714530___redundant_placeholder3
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
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_16714600

inputs)
lstm_cell_398_16714518:	Ш)
lstm_cell_398_16714520:	2Ш%
lstm_cell_398_16714522:	Ш
identityЂ%lstm_cell_398/StatefulPartitionedCallЂwhileD
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
%lstm_cell_398/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_398_16714518lstm_cell_398_16714520lstm_cell_398_16714522*
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
K__inference_lstm_cell_398_layer_call_and_return_conditional_losses_167144512'
%lstm_cell_398/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_398_16714518lstm_cell_398_16714520lstm_cell_398_16714522*
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
while_body_16714531*
condR
while_cond_16714530*K
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
NoOpNoOp&^lstm_cell_398/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2N
%lstm_cell_398/StatefulPartitionedCall%lstm_cell_398/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
і

K__inference_lstm_cell_398_layer_call_and_return_conditional_losses_16714305

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
]
Ў
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_16718259
inputs_0?
,lstm_cell_397_matmul_readvariableop_resource:	ШA
.lstm_cell_397_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_397_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_397/BiasAdd/ReadVariableOpЂ#lstm_cell_397/MatMul/ReadVariableOpЂ%lstm_cell_397/MatMul_1/ReadVariableOpЂwhileF
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
#lstm_cell_397/MatMul/ReadVariableOpReadVariableOp,lstm_cell_397_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_397/MatMul/ReadVariableOpА
lstm_cell_397/MatMulMatMulstrided_slice_2:output:0+lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/MatMulО
%lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_397_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_397/MatMul_1/ReadVariableOpЌ
lstm_cell_397/MatMul_1MatMulzeros:output:0-lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/MatMul_1Є
lstm_cell_397/addAddV2lstm_cell_397/MatMul:product:0 lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/addЗ
$lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_397_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_397/BiasAdd/ReadVariableOpБ
lstm_cell_397/BiasAddBiasAddlstm_cell_397/add:z:0,lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/BiasAdd
lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_397/split/split_dimї
lstm_cell_397/splitSplit&lstm_cell_397/split/split_dim:output:0lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_397/split
lstm_cell_397/SigmoidSigmoidlstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Sigmoid
lstm_cell_397/Sigmoid_1Sigmoidlstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Sigmoid_1
lstm_cell_397/mulMullstm_cell_397/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/mul
lstm_cell_397/ReluRelulstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Relu 
lstm_cell_397/mul_1Mullstm_cell_397/Sigmoid:y:0 lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/mul_1
lstm_cell_397/add_1AddV2lstm_cell_397/mul:z:0lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/add_1
lstm_cell_397/Sigmoid_2Sigmoidlstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Sigmoid_2
lstm_cell_397/Relu_1Relulstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Relu_1Є
lstm_cell_397/mul_2Mullstm_cell_397/Sigmoid_2:y:0"lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_397_matmul_readvariableop_resource.lstm_cell_397_matmul_1_readvariableop_resource-lstm_cell_397_biasadd_readvariableop_resource*
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
while_body_16718175*
condR
while_cond_16718174*K
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
NoOpNoOp%^lstm_cell_397/BiasAdd/ReadVariableOp$^lstm_cell_397/MatMul/ReadVariableOp&^lstm_cell_397/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_397/BiasAdd/ReadVariableOp$lstm_cell_397/BiasAdd/ReadVariableOp2J
#lstm_cell_397/MatMul/ReadVariableOp#lstm_cell_397/MatMul/ReadVariableOp2N
%lstm_cell_397/MatMul_1/ReadVariableOp%lstm_cell_397/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
п
Э
while_cond_16718476
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16718476___redundant_placeholder06
2while_while_cond_16718476___redundant_placeholder16
2while_while_cond_16718476___redundant_placeholder26
2while_while_cond_16718476___redundant_placeholder3
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
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_16715023

inputs?
,lstm_cell_397_matmul_readvariableop_resource:	ШA
.lstm_cell_397_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_397_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_397/BiasAdd/ReadVariableOpЂ#lstm_cell_397/MatMul/ReadVariableOpЂ%lstm_cell_397/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_397/MatMul/ReadVariableOpReadVariableOp,lstm_cell_397_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_397/MatMul/ReadVariableOpА
lstm_cell_397/MatMulMatMulstrided_slice_2:output:0+lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/MatMulО
%lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_397_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_397/MatMul_1/ReadVariableOpЌ
lstm_cell_397/MatMul_1MatMulzeros:output:0-lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/MatMul_1Є
lstm_cell_397/addAddV2lstm_cell_397/MatMul:product:0 lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/addЗ
$lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_397_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_397/BiasAdd/ReadVariableOpБ
lstm_cell_397/BiasAddBiasAddlstm_cell_397/add:z:0,lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/BiasAdd
lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_397/split/split_dimї
lstm_cell_397/splitSplit&lstm_cell_397/split/split_dim:output:0lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_397/split
lstm_cell_397/SigmoidSigmoidlstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Sigmoid
lstm_cell_397/Sigmoid_1Sigmoidlstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Sigmoid_1
lstm_cell_397/mulMullstm_cell_397/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/mul
lstm_cell_397/ReluRelulstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Relu 
lstm_cell_397/mul_1Mullstm_cell_397/Sigmoid:y:0 lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/mul_1
lstm_cell_397/add_1AddV2lstm_cell_397/mul:z:0lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/add_1
lstm_cell_397/Sigmoid_2Sigmoidlstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Sigmoid_2
lstm_cell_397/Relu_1Relulstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Relu_1Є
lstm_cell_397/mul_2Mullstm_cell_397/Sigmoid_2:y:0"lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_397_matmul_readvariableop_resource.lstm_cell_397_matmul_1_readvariableop_resource-lstm_cell_397_biasadd_readvariableop_resource*
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
while_body_16714939*
condR
while_cond_16714938*K
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
NoOpNoOp%^lstm_cell_397/BiasAdd/ReadVariableOp$^lstm_cell_397/MatMul/ReadVariableOp&^lstm_cell_397/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_397/BiasAdd/ReadVariableOp$lstm_cell_397/BiasAdd/ReadVariableOp2J
#lstm_cell_397/MatMul/ReadVariableOp#lstm_cell_397/MatMul/ReadVariableOp2N
%lstm_cell_397/MatMul_1/ReadVariableOp%lstm_cell_397/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ъ
М
%backward_lstm_132_while_cond_16717588@
<backward_lstm_132_while_backward_lstm_132_while_loop_counterF
Bbackward_lstm_132_while_backward_lstm_132_while_maximum_iterations'
#backward_lstm_132_while_placeholder)
%backward_lstm_132_while_placeholder_1)
%backward_lstm_132_while_placeholder_2)
%backward_lstm_132_while_placeholder_3)
%backward_lstm_132_while_placeholder_4B
>backward_lstm_132_while_less_backward_lstm_132_strided_slice_1Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16717588___redundant_placeholder0Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16717588___redundant_placeholder1Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16717588___redundant_placeholder2Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16717588___redundant_placeholder3Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16717588___redundant_placeholder4$
 backward_lstm_132_while_identity
Ъ
backward_lstm_132/while/LessLess#backward_lstm_132_while_placeholder>backward_lstm_132_while_less_backward_lstm_132_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_132/while/Less
 backward_lstm_132/while/IdentityIdentity backward_lstm_132/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_132/while/Identity"M
 backward_lstm_132_while_identity)backward_lstm_132/while/Identity:output:0*(
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
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_16718909
inputs_0?
,lstm_cell_398_matmul_readvariableop_resource:	ШA
.lstm_cell_398_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_398_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_398/BiasAdd/ReadVariableOpЂ#lstm_cell_398/MatMul/ReadVariableOpЂ%lstm_cell_398/MatMul_1/ReadVariableOpЂwhileF
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
#lstm_cell_398/MatMul/ReadVariableOpReadVariableOp,lstm_cell_398_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_398/MatMul/ReadVariableOpА
lstm_cell_398/MatMulMatMulstrided_slice_2:output:0+lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/MatMulО
%lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_398_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_398/MatMul_1/ReadVariableOpЌ
lstm_cell_398/MatMul_1MatMulzeros:output:0-lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/MatMul_1Є
lstm_cell_398/addAddV2lstm_cell_398/MatMul:product:0 lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/addЗ
$lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_398_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_398/BiasAdd/ReadVariableOpБ
lstm_cell_398/BiasAddBiasAddlstm_cell_398/add:z:0,lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/BiasAdd
lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_398/split/split_dimї
lstm_cell_398/splitSplit&lstm_cell_398/split/split_dim:output:0lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_398/split
lstm_cell_398/SigmoidSigmoidlstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Sigmoid
lstm_cell_398/Sigmoid_1Sigmoidlstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Sigmoid_1
lstm_cell_398/mulMullstm_cell_398/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/mul
lstm_cell_398/ReluRelulstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Relu 
lstm_cell_398/mul_1Mullstm_cell_398/Sigmoid:y:0 lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/mul_1
lstm_cell_398/add_1AddV2lstm_cell_398/mul:z:0lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/add_1
lstm_cell_398/Sigmoid_2Sigmoidlstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Sigmoid_2
lstm_cell_398/Relu_1Relulstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Relu_1Є
lstm_cell_398/mul_2Mullstm_cell_398/Sigmoid_2:y:0"lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_398_matmul_readvariableop_resource.lstm_cell_398_matmul_1_readvariableop_resource-lstm_cell_398_biasadd_readvariableop_resource*
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
while_body_16718825*
condR
while_cond_16718824*K
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
NoOpNoOp%^lstm_cell_398/BiasAdd/ReadVariableOp$^lstm_cell_398/MatMul/ReadVariableOp&^lstm_cell_398/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_398/BiasAdd/ReadVariableOp$lstm_cell_398/BiasAdd/ReadVariableOp2J
#lstm_cell_398/MatMul/ReadVariableOp#lstm_cell_398/MatMul/ReadVariableOp2N
%lstm_cell_398/MatMul_1/ReadVariableOp%lstm_cell_398/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
]
Ќ
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_16718712

inputs?
,lstm_cell_397_matmul_readvariableop_resource:	ШA
.lstm_cell_397_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_397_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_397/BiasAdd/ReadVariableOpЂ#lstm_cell_397/MatMul/ReadVariableOpЂ%lstm_cell_397/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_397/MatMul/ReadVariableOpReadVariableOp,lstm_cell_397_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_397/MatMul/ReadVariableOpА
lstm_cell_397/MatMulMatMulstrided_slice_2:output:0+lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/MatMulО
%lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_397_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_397/MatMul_1/ReadVariableOpЌ
lstm_cell_397/MatMul_1MatMulzeros:output:0-lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/MatMul_1Є
lstm_cell_397/addAddV2lstm_cell_397/MatMul:product:0 lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/addЗ
$lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_397_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_397/BiasAdd/ReadVariableOpБ
lstm_cell_397/BiasAddBiasAddlstm_cell_397/add:z:0,lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/BiasAdd
lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_397/split/split_dimї
lstm_cell_397/splitSplit&lstm_cell_397/split/split_dim:output:0lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_397/split
lstm_cell_397/SigmoidSigmoidlstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Sigmoid
lstm_cell_397/Sigmoid_1Sigmoidlstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Sigmoid_1
lstm_cell_397/mulMullstm_cell_397/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/mul
lstm_cell_397/ReluRelulstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Relu 
lstm_cell_397/mul_1Mullstm_cell_397/Sigmoid:y:0 lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/mul_1
lstm_cell_397/add_1AddV2lstm_cell_397/mul:z:0lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/add_1
lstm_cell_397/Sigmoid_2Sigmoidlstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Sigmoid_2
lstm_cell_397/Relu_1Relulstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Relu_1Є
lstm_cell_397/mul_2Mullstm_cell_397/Sigmoid_2:y:0"lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_397_matmul_readvariableop_resource.lstm_cell_397_matmul_1_readvariableop_resource-lstm_cell_397_biasadd_readvariableop_resource*
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
while_body_16718628*
condR
while_cond_16718627*K
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
NoOpNoOp%^lstm_cell_397/BiasAdd/ReadVariableOp$^lstm_cell_397/MatMul/ReadVariableOp&^lstm_cell_397/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_397/BiasAdd/ReadVariableOp$lstm_cell_397/BiasAdd/ReadVariableOp2J
#lstm_cell_397/MatMul/ReadVariableOp#lstm_cell_397/MatMul/ReadVariableOp2N
%lstm_cell_397/MatMul_1/ReadVariableOp%lstm_cell_397/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
М
Т
Fsequential_132_bidirectional_132_backward_lstm_132_while_cond_16713493
~sequential_132_bidirectional_132_backward_lstm_132_while_sequential_132_bidirectional_132_backward_lstm_132_while_loop_counter
sequential_132_bidirectional_132_backward_lstm_132_while_sequential_132_bidirectional_132_backward_lstm_132_while_maximum_iterationsH
Dsequential_132_bidirectional_132_backward_lstm_132_while_placeholderJ
Fsequential_132_bidirectional_132_backward_lstm_132_while_placeholder_1J
Fsequential_132_bidirectional_132_backward_lstm_132_while_placeholder_2J
Fsequential_132_bidirectional_132_backward_lstm_132_while_placeholder_3J
Fsequential_132_bidirectional_132_backward_lstm_132_while_placeholder_4
sequential_132_bidirectional_132_backward_lstm_132_while_less_sequential_132_bidirectional_132_backward_lstm_132_strided_slice_1
sequential_132_bidirectional_132_backward_lstm_132_while_sequential_132_bidirectional_132_backward_lstm_132_while_cond_16713493___redundant_placeholder0
sequential_132_bidirectional_132_backward_lstm_132_while_sequential_132_bidirectional_132_backward_lstm_132_while_cond_16713493___redundant_placeholder1
sequential_132_bidirectional_132_backward_lstm_132_while_sequential_132_bidirectional_132_backward_lstm_132_while_cond_16713493___redundant_placeholder2
sequential_132_bidirectional_132_backward_lstm_132_while_sequential_132_bidirectional_132_backward_lstm_132_while_cond_16713493___redundant_placeholder3
sequential_132_bidirectional_132_backward_lstm_132_while_sequential_132_bidirectional_132_backward_lstm_132_while_cond_16713493___redundant_placeholder4E
Asequential_132_bidirectional_132_backward_lstm_132_while_identity
№
=sequential_132/bidirectional_132/backward_lstm_132/while/LessLessDsequential_132_bidirectional_132_backward_lstm_132_while_placeholdersequential_132_bidirectional_132_backward_lstm_132_while_less_sequential_132_bidirectional_132_backward_lstm_132_strided_slice_1*
T0*
_output_shapes
: 2?
=sequential_132/bidirectional_132/backward_lstm_132/while/Lessі
Asequential_132/bidirectional_132/backward_lstm_132/while/IdentityIdentityAsequential_132/bidirectional_132/backward_lstm_132/while/Less:z:0*
T0
*
_output_shapes
: 2C
Asequential_132/bidirectional_132/backward_lstm_132/while/Identity"
Asequential_132_bidirectional_132_backward_lstm_132_while_identityJsequential_132/bidirectional_132/backward_lstm_132/while/Identity:output:0*(
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
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_16716034

inputs
inputs_1	P
=forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource:	ШR
?forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource:	2ШM
>forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource:	ШQ
>backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource:	ШS
@backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource:	2ШN
?backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource:	Ш
identityЂ6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpЂ5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpЂ7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpЂbackward_lstm_132/whileЂ5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpЂ4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpЂ6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOpЂforward_lstm_132/while
%forward_lstm_132/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_132/RaggedToTensor/zeros
%forward_lstm_132/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2'
%forward_lstm_132/RaggedToTensor/Const
4forward_lstm_132/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_132/RaggedToTensor/Const:output:0inputs.forward_lstm_132/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_132/RaggedToTensor/RaggedTensorToTensorФ
;forward_lstm_132/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_132/RaggedNestedRowLengths/strided_slice/stackШ
=forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1Ш
=forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2Љ
5forward_lstm_132/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_132/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask27
5forward_lstm_132/RaggedNestedRowLengths/strided_sliceШ
=forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stackе
?forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2A
?forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1Ь
?forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2Е
7forward_lstm_132/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask29
7forward_lstm_132/RaggedNestedRowLengths/strided_slice_1
+forward_lstm_132/RaggedNestedRowLengths/subSub>forward_lstm_132/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_132/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2-
+forward_lstm_132/RaggedNestedRowLengths/subЄ
forward_lstm_132/CastCast/forward_lstm_132/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_132/Cast
forward_lstm_132/ShapeShape=forward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_132/Shape
$forward_lstm_132/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_132/strided_slice/stack
&forward_lstm_132/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_132/strided_slice/stack_1
&forward_lstm_132/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_132/strided_slice/stack_2Ш
forward_lstm_132/strided_sliceStridedSliceforward_lstm_132/Shape:output:0-forward_lstm_132/strided_slice/stack:output:0/forward_lstm_132/strided_slice/stack_1:output:0/forward_lstm_132/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_132/strided_slice~
forward_lstm_132/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_132/zeros/mul/yА
forward_lstm_132/zeros/mulMul'forward_lstm_132/strided_slice:output:0%forward_lstm_132/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros/mul
forward_lstm_132/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_132/zeros/Less/yЋ
forward_lstm_132/zeros/LessLessforward_lstm_132/zeros/mul:z:0&forward_lstm_132/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros/Less
forward_lstm_132/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_132/zeros/packed/1Ч
forward_lstm_132/zeros/packedPack'forward_lstm_132/strided_slice:output:0(forward_lstm_132/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_132/zeros/packed
forward_lstm_132/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_132/zeros/ConstЙ
forward_lstm_132/zerosFill&forward_lstm_132/zeros/packed:output:0%forward_lstm_132/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/zeros
forward_lstm_132/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_132/zeros_1/mul/yЖ
forward_lstm_132/zeros_1/mulMul'forward_lstm_132/strided_slice:output:0'forward_lstm_132/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros_1/mul
forward_lstm_132/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
forward_lstm_132/zeros_1/Less/yГ
forward_lstm_132/zeros_1/LessLess forward_lstm_132/zeros_1/mul:z:0(forward_lstm_132/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros_1/Less
!forward_lstm_132/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_132/zeros_1/packed/1Э
forward_lstm_132/zeros_1/packedPack'forward_lstm_132/strided_slice:output:0*forward_lstm_132/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_132/zeros_1/packed
forward_lstm_132/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_132/zeros_1/ConstС
forward_lstm_132/zeros_1Fill(forward_lstm_132/zeros_1/packed:output:0'forward_lstm_132/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/zeros_1
forward_lstm_132/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_132/transpose/permэ
forward_lstm_132/transpose	Transpose=forward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_132/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_132/transpose
forward_lstm_132/Shape_1Shapeforward_lstm_132/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_132/Shape_1
&forward_lstm_132/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_132/strided_slice_1/stack
(forward_lstm_132/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_1/stack_1
(forward_lstm_132/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_1/stack_2д
 forward_lstm_132/strided_slice_1StridedSlice!forward_lstm_132/Shape_1:output:0/forward_lstm_132/strided_slice_1/stack:output:01forward_lstm_132/strided_slice_1/stack_1:output:01forward_lstm_132/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_132/strided_slice_1Ї
,forward_lstm_132/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,forward_lstm_132/TensorArrayV2/element_shapeі
forward_lstm_132/TensorArrayV2TensorListReserve5forward_lstm_132/TensorArrayV2/element_shape:output:0)forward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_132/TensorArrayV2с
Fforward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fforward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8forward_lstm_132/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_132/transpose:y:0Oforward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_132/TensorArrayUnstack/TensorListFromTensor
&forward_lstm_132/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_132/strided_slice_2/stack
(forward_lstm_132/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_2/stack_1
(forward_lstm_132/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_2/stack_2т
 forward_lstm_132/strided_slice_2StridedSliceforward_lstm_132/transpose:y:0/forward_lstm_132/strided_slice_2/stack:output:01forward_lstm_132/strided_slice_2/stack_1:output:01forward_lstm_132/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 forward_lstm_132/strided_slice_2ы
4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpReadVariableOp=forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpє
%forward_lstm_132/lstm_cell_397/MatMulMatMul)forward_lstm_132/strided_slice_2:output:0<forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_132/lstm_cell_397/MatMulё
6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOp?forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp№
'forward_lstm_132/lstm_cell_397/MatMul_1MatMulforward_lstm_132/zeros:output:0>forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_132/lstm_cell_397/MatMul_1ш
"forward_lstm_132/lstm_cell_397/addAddV2/forward_lstm_132/lstm_cell_397/MatMul:product:01forward_lstm_132/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_132/lstm_cell_397/addъ
5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOp>forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpѕ
&forward_lstm_132/lstm_cell_397/BiasAddBiasAdd&forward_lstm_132/lstm_cell_397/add:z:0=forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_132/lstm_cell_397/BiasAddЂ
.forward_lstm_132/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.forward_lstm_132/lstm_cell_397/split/split_dimЛ
$forward_lstm_132/lstm_cell_397/splitSplit7forward_lstm_132/lstm_cell_397/split/split_dim:output:0/forward_lstm_132/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$forward_lstm_132/lstm_cell_397/splitМ
&forward_lstm_132/lstm_cell_397/SigmoidSigmoid-forward_lstm_132/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_132/lstm_cell_397/SigmoidР
(forward_lstm_132/lstm_cell_397/Sigmoid_1Sigmoid-forward_lstm_132/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_132/lstm_cell_397/Sigmoid_1в
"forward_lstm_132/lstm_cell_397/mulMul,forward_lstm_132/lstm_cell_397/Sigmoid_1:y:0!forward_lstm_132/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_132/lstm_cell_397/mulГ
#forward_lstm_132/lstm_cell_397/ReluRelu-forward_lstm_132/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_132/lstm_cell_397/Reluф
$forward_lstm_132/lstm_cell_397/mul_1Mul*forward_lstm_132/lstm_cell_397/Sigmoid:y:01forward_lstm_132/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_132/lstm_cell_397/mul_1й
$forward_lstm_132/lstm_cell_397/add_1AddV2&forward_lstm_132/lstm_cell_397/mul:z:0(forward_lstm_132/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_132/lstm_cell_397/add_1Р
(forward_lstm_132/lstm_cell_397/Sigmoid_2Sigmoid-forward_lstm_132/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_132/lstm_cell_397/Sigmoid_2В
%forward_lstm_132/lstm_cell_397/Relu_1Relu(forward_lstm_132/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_132/lstm_cell_397/Relu_1ш
$forward_lstm_132/lstm_cell_397/mul_2Mul,forward_lstm_132/lstm_cell_397/Sigmoid_2:y:03forward_lstm_132/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_132/lstm_cell_397/mul_2Б
.forward_lstm_132/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.forward_lstm_132/TensorArrayV2_1/element_shapeќ
 forward_lstm_132/TensorArrayV2_1TensorListReserve7forward_lstm_132/TensorArrayV2_1/element_shape:output:0)forward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_132/TensorArrayV2_1p
forward_lstm_132/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_132/timeЃ
forward_lstm_132/zeros_like	ZerosLike(forward_lstm_132/lstm_cell_397/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/zeros_likeЁ
)forward_lstm_132/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)forward_lstm_132/while/maximum_iterations
#forward_lstm_132/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_132/while/loop_counter	
forward_lstm_132/whileWhile,forward_lstm_132/while/loop_counter:output:02forward_lstm_132/while/maximum_iterations:output:0forward_lstm_132/time:output:0)forward_lstm_132/TensorArrayV2_1:handle:0forward_lstm_132/zeros_like:y:0forward_lstm_132/zeros:output:0!forward_lstm_132/zeros_1:output:0)forward_lstm_132/strided_slice_1:output:0Hforward_lstm_132/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_132/Cast:y:0=forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource?forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource>forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource*
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
$forward_lstm_132_while_body_16715758*0
cond(R&
$forward_lstm_132_while_cond_16715757*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_132/whileз
Aforward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Aforward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3forward_lstm_132/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_132/while:output:3Jforward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3forward_lstm_132/TensorArrayV2Stack/TensorListStackЃ
&forward_lstm_132/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&forward_lstm_132/strided_slice_3/stack
(forward_lstm_132/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_132/strided_slice_3/stack_1
(forward_lstm_132/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_3/stack_2
 forward_lstm_132/strided_slice_3StridedSlice<forward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_132/strided_slice_3/stack:output:01forward_lstm_132/strided_slice_3/stack_1:output:01forward_lstm_132/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 forward_lstm_132/strided_slice_3
!forward_lstm_132/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_132/transpose_1/permђ
forward_lstm_132/transpose_1	Transpose<forward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_132/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_132/transpose_1
forward_lstm_132/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_132/runtime
&backward_lstm_132/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_132/RaggedToTensor/zeros
&backward_lstm_132/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2(
&backward_lstm_132/RaggedToTensor/Const
5backward_lstm_132/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_132/RaggedToTensor/Const:output:0inputs/backward_lstm_132/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_132/RaggedToTensor/RaggedTensorToTensorЦ
<backward_lstm_132/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_132/RaggedNestedRowLengths/strided_slice/stackЪ
>backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1Ъ
>backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2Ў
6backward_lstm_132/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_132/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask28
6backward_lstm_132/RaggedNestedRowLengths/strided_sliceЪ
>backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stackз
@backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2B
@backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1Ю
@backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2К
8backward_lstm_132/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2:
8backward_lstm_132/RaggedNestedRowLengths/strided_slice_1
,backward_lstm_132/RaggedNestedRowLengths/subSub?backward_lstm_132/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_132/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2.
,backward_lstm_132/RaggedNestedRowLengths/subЇ
backward_lstm_132/CastCast0backward_lstm_132/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_132/Cast 
backward_lstm_132/ShapeShape>backward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_132/Shape
%backward_lstm_132/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_132/strided_slice/stack
'backward_lstm_132/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_132/strided_slice/stack_1
'backward_lstm_132/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_132/strided_slice/stack_2Ю
backward_lstm_132/strided_sliceStridedSlice backward_lstm_132/Shape:output:0.backward_lstm_132/strided_slice/stack:output:00backward_lstm_132/strided_slice/stack_1:output:00backward_lstm_132/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_132/strided_slice
backward_lstm_132/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_132/zeros/mul/yД
backward_lstm_132/zeros/mulMul(backward_lstm_132/strided_slice:output:0&backward_lstm_132/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/zeros/mul
backward_lstm_132/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_132/zeros/Less/yЏ
backward_lstm_132/zeros/LessLessbackward_lstm_132/zeros/mul:z:0'backward_lstm_132/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/zeros/Less
 backward_lstm_132/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_132/zeros/packed/1Ы
backward_lstm_132/zeros/packedPack(backward_lstm_132/strided_slice:output:0)backward_lstm_132/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_132/zeros/packed
backward_lstm_132/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_132/zeros/ConstН
backward_lstm_132/zerosFill'backward_lstm_132/zeros/packed:output:0&backward_lstm_132/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_132/zeros
backward_lstm_132/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_132/zeros_1/mul/yК
backward_lstm_132/zeros_1/mulMul(backward_lstm_132/strided_slice:output:0(backward_lstm_132/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/zeros_1/mul
 backward_lstm_132/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 backward_lstm_132/zeros_1/Less/yЗ
backward_lstm_132/zeros_1/LessLess!backward_lstm_132/zeros_1/mul:z:0)backward_lstm_132/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_132/zeros_1/Less
"backward_lstm_132/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_132/zeros_1/packed/1б
 backward_lstm_132/zeros_1/packedPack(backward_lstm_132/strided_slice:output:0+backward_lstm_132/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_132/zeros_1/packed
backward_lstm_132/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_132/zeros_1/ConstХ
backward_lstm_132/zeros_1Fill)backward_lstm_132/zeros_1/packed:output:0(backward_lstm_132/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_132/zeros_1
 backward_lstm_132/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_132/transpose/permё
backward_lstm_132/transpose	Transpose>backward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_132/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_132/transpose
backward_lstm_132/Shape_1Shapebackward_lstm_132/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_132/Shape_1
'backward_lstm_132/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_132/strided_slice_1/stack 
)backward_lstm_132/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_1/stack_1 
)backward_lstm_132/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_1/stack_2к
!backward_lstm_132/strided_slice_1StridedSlice"backward_lstm_132/Shape_1:output:00backward_lstm_132/strided_slice_1/stack:output:02backward_lstm_132/strided_slice_1/stack_1:output:02backward_lstm_132/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_132/strided_slice_1Љ
-backward_lstm_132/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2/
-backward_lstm_132/TensorArrayV2/element_shapeњ
backward_lstm_132/TensorArrayV2TensorListReserve6backward_lstm_132/TensorArrayV2/element_shape:output:0*backward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_132/TensorArrayV2
 backward_lstm_132/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_132/ReverseV2/axisв
backward_lstm_132/ReverseV2	ReverseV2backward_lstm_132/transpose:y:0)backward_lstm_132/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_132/ReverseV2у
Gbackward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gbackward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeХ
9backward_lstm_132/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_132/ReverseV2:output:0Pbackward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_132/TensorArrayUnstack/TensorListFromTensor
'backward_lstm_132/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_132/strided_slice_2/stack 
)backward_lstm_132/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_2/stack_1 
)backward_lstm_132/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_2/stack_2ш
!backward_lstm_132/strided_slice_2StridedSlicebackward_lstm_132/transpose:y:00backward_lstm_132/strided_slice_2/stack:output:02backward_lstm_132/strided_slice_2/stack_1:output:02backward_lstm_132/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2#
!backward_lstm_132/strided_slice_2ю
5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpReadVariableOp>backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype027
5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpј
&backward_lstm_132/lstm_cell_398/MatMulMatMul*backward_lstm_132/strided_slice_2:output:0=backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_132/lstm_cell_398/MatMulє
7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOp@backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype029
7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpє
(backward_lstm_132/lstm_cell_398/MatMul_1MatMul backward_lstm_132/zeros:output:0?backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_132/lstm_cell_398/MatMul_1ь
#backward_lstm_132/lstm_cell_398/addAddV20backward_lstm_132/lstm_cell_398/MatMul:product:02backward_lstm_132/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_132/lstm_cell_398/addэ
6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOp?backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype028
6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpљ
'backward_lstm_132/lstm_cell_398/BiasAddBiasAdd'backward_lstm_132/lstm_cell_398/add:z:0>backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_132/lstm_cell_398/BiasAddЄ
/backward_lstm_132/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/backward_lstm_132/lstm_cell_398/split/split_dimП
%backward_lstm_132/lstm_cell_398/splitSplit8backward_lstm_132/lstm_cell_398/split/split_dim:output:00backward_lstm_132/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2'
%backward_lstm_132/lstm_cell_398/splitП
'backward_lstm_132/lstm_cell_398/SigmoidSigmoid.backward_lstm_132/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_132/lstm_cell_398/SigmoidУ
)backward_lstm_132/lstm_cell_398/Sigmoid_1Sigmoid.backward_lstm_132/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_132/lstm_cell_398/Sigmoid_1ж
#backward_lstm_132/lstm_cell_398/mulMul-backward_lstm_132/lstm_cell_398/Sigmoid_1:y:0"backward_lstm_132/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_132/lstm_cell_398/mulЖ
$backward_lstm_132/lstm_cell_398/ReluRelu.backward_lstm_132/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_132/lstm_cell_398/Reluш
%backward_lstm_132/lstm_cell_398/mul_1Mul+backward_lstm_132/lstm_cell_398/Sigmoid:y:02backward_lstm_132/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_132/lstm_cell_398/mul_1н
%backward_lstm_132/lstm_cell_398/add_1AddV2'backward_lstm_132/lstm_cell_398/mul:z:0)backward_lstm_132/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_132/lstm_cell_398/add_1У
)backward_lstm_132/lstm_cell_398/Sigmoid_2Sigmoid.backward_lstm_132/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_132/lstm_cell_398/Sigmoid_2Е
&backward_lstm_132/lstm_cell_398/Relu_1Relu)backward_lstm_132/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_132/lstm_cell_398/Relu_1ь
%backward_lstm_132/lstm_cell_398/mul_2Mul-backward_lstm_132/lstm_cell_398/Sigmoid_2:y:04backward_lstm_132/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_132/lstm_cell_398/mul_2Г
/backward_lstm_132/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   21
/backward_lstm_132/TensorArrayV2_1/element_shape
!backward_lstm_132/TensorArrayV2_1TensorListReserve8backward_lstm_132/TensorArrayV2_1/element_shape:output:0*backward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_132/TensorArrayV2_1r
backward_lstm_132/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_132/time
'backward_lstm_132/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_132/Max/reduction_indicesЄ
backward_lstm_132/MaxMaxbackward_lstm_132/Cast:y:00backward_lstm_132/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/Maxt
backward_lstm_132/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_132/sub/y
backward_lstm_132/subSubbackward_lstm_132/Max:output:0 backward_lstm_132/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/sub
backward_lstm_132/Sub_1Subbackward_lstm_132/sub:z:0backward_lstm_132/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_132/Sub_1І
backward_lstm_132/zeros_like	ZerosLike)backward_lstm_132/lstm_cell_398/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_132/zeros_likeЃ
*backward_lstm_132/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*backward_lstm_132/while/maximum_iterations
$backward_lstm_132/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_132/while/loop_counterЅ	
backward_lstm_132/whileWhile-backward_lstm_132/while/loop_counter:output:03backward_lstm_132/while/maximum_iterations:output:0backward_lstm_132/time:output:0*backward_lstm_132/TensorArrayV2_1:handle:0 backward_lstm_132/zeros_like:y:0 backward_lstm_132/zeros:output:0"backward_lstm_132/zeros_1:output:0*backward_lstm_132/strided_slice_1:output:0Ibackward_lstm_132/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_132/Sub_1:z:0>backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource@backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource?backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource*
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
%backward_lstm_132_while_body_16715937*1
cond)R'
%backward_lstm_132_while_cond_16715936*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_132/whileй
Bbackward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2D
Bbackward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeЙ
4backward_lstm_132/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_132/while:output:3Kbackward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype026
4backward_lstm_132/TensorArrayV2Stack/TensorListStackЅ
'backward_lstm_132/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2)
'backward_lstm_132/strided_slice_3/stack 
)backward_lstm_132/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_132/strided_slice_3/stack_1 
)backward_lstm_132/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_3/stack_2
!backward_lstm_132/strided_slice_3StridedSlice=backward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_132/strided_slice_3/stack:output:02backward_lstm_132/strided_slice_3/stack_1:output:02backward_lstm_132/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2#
!backward_lstm_132/strided_slice_3
"backward_lstm_132/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_132/transpose_1/permі
backward_lstm_132/transpose_1	Transpose=backward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_132/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_132/transpose_1
backward_lstm_132/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_132/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisФ
concatConcatV2)forward_lstm_132/strided_slice_3:output:0*backward_lstm_132/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp7^backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp6^backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp8^backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp^backward_lstm_132/while6^forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp5^forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp7^forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp^forward_lstm_132/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2p
6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp2n
5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp2r
7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp22
backward_lstm_132/whilebackward_lstm_132/while2n
5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp2l
4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp2p
6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp20
forward_lstm_132/whileforward_lstm_132/while:O K
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
Esequential_132_bidirectional_132_forward_lstm_132_while_cond_16713314
|sequential_132_bidirectional_132_forward_lstm_132_while_sequential_132_bidirectional_132_forward_lstm_132_while_loop_counter
sequential_132_bidirectional_132_forward_lstm_132_while_sequential_132_bidirectional_132_forward_lstm_132_while_maximum_iterationsG
Csequential_132_bidirectional_132_forward_lstm_132_while_placeholderI
Esequential_132_bidirectional_132_forward_lstm_132_while_placeholder_1I
Esequential_132_bidirectional_132_forward_lstm_132_while_placeholder_2I
Esequential_132_bidirectional_132_forward_lstm_132_while_placeholder_3I
Esequential_132_bidirectional_132_forward_lstm_132_while_placeholder_4
~sequential_132_bidirectional_132_forward_lstm_132_while_less_sequential_132_bidirectional_132_forward_lstm_132_strided_slice_1
sequential_132_bidirectional_132_forward_lstm_132_while_sequential_132_bidirectional_132_forward_lstm_132_while_cond_16713314___redundant_placeholder0
sequential_132_bidirectional_132_forward_lstm_132_while_sequential_132_bidirectional_132_forward_lstm_132_while_cond_16713314___redundant_placeholder1
sequential_132_bidirectional_132_forward_lstm_132_while_sequential_132_bidirectional_132_forward_lstm_132_while_cond_16713314___redundant_placeholder2
sequential_132_bidirectional_132_forward_lstm_132_while_sequential_132_bidirectional_132_forward_lstm_132_while_cond_16713314___redundant_placeholder3
sequential_132_bidirectional_132_forward_lstm_132_while_sequential_132_bidirectional_132_forward_lstm_132_while_cond_16713314___redundant_placeholder4D
@sequential_132_bidirectional_132_forward_lstm_132_while_identity
ъ
<sequential_132/bidirectional_132/forward_lstm_132/while/LessLessCsequential_132_bidirectional_132_forward_lstm_132_while_placeholder~sequential_132_bidirectional_132_forward_lstm_132_while_less_sequential_132_bidirectional_132_forward_lstm_132_strided_slice_1*
T0*
_output_shapes
: 2>
<sequential_132/bidirectional_132/forward_lstm_132/while/Lessѓ
@sequential_132/bidirectional_132/forward_lstm_132/while/IdentityIdentity@sequential_132/bidirectional_132/forward_lstm_132/while/Less:z:0*
T0
*
_output_shapes
: 2B
@sequential_132/bidirectional_132/forward_lstm_132/while/Identity"
@sequential_132_bidirectional_132_forward_lstm_132_while_identityIsequential_132/bidirectional_132/forward_lstm_132/while/Identity:output:0*(
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
while_body_16719131
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_398_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_398_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_398_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_398_matmul_readvariableop_resource:	ШG
4while_lstm_cell_398_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_398_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_398/BiasAdd/ReadVariableOpЂ)while/lstm_cell_398/MatMul/ReadVariableOpЂ+while/lstm_cell_398/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_398/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_398_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_398/MatMul/ReadVariableOpк
while/lstm_cell_398/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/MatMulв
+while/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_398_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_398/MatMul_1/ReadVariableOpУ
while/lstm_cell_398/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/MatMul_1М
while/lstm_cell_398/addAddV2$while/lstm_cell_398/MatMul:product:0&while/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/addЫ
*while/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_398_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_398/BiasAdd/ReadVariableOpЩ
while/lstm_cell_398/BiasAddBiasAddwhile/lstm_cell_398/add:z:02while/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/BiasAdd
#while/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_398/split/split_dim
while/lstm_cell_398/splitSplit,while/lstm_cell_398/split/split_dim:output:0$while/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_398/split
while/lstm_cell_398/SigmoidSigmoid"while/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Sigmoid
while/lstm_cell_398/Sigmoid_1Sigmoid"while/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Sigmoid_1Ѓ
while/lstm_cell_398/mulMul!while/lstm_cell_398/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/mul
while/lstm_cell_398/ReluRelu"while/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/ReluИ
while/lstm_cell_398/mul_1Mulwhile/lstm_cell_398/Sigmoid:y:0&while/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/mul_1­
while/lstm_cell_398/add_1AddV2while/lstm_cell_398/mul:z:0while/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/add_1
while/lstm_cell_398/Sigmoid_2Sigmoid"while/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Sigmoid_2
while/lstm_cell_398/Relu_1Reluwhile/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Relu_1М
while/lstm_cell_398/mul_2Mul!while/lstm_cell_398/Sigmoid_2:y:0(while/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_398/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_398/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_398/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_398/BiasAdd/ReadVariableOp*^while/lstm_cell_398/MatMul/ReadVariableOp,^while/lstm_cell_398/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_398_biasadd_readvariableop_resource5while_lstm_cell_398_biasadd_readvariableop_resource_0"n
4while_lstm_cell_398_matmul_1_readvariableop_resource6while_lstm_cell_398_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_398_matmul_readvariableop_resource4while_lstm_cell_398_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_398/BiasAdd/ReadVariableOp*while/lstm_cell_398/BiasAdd/ReadVariableOp2V
)while/lstm_cell_398/MatMul/ReadVariableOp)while/lstm_cell_398/MatMul/ReadVariableOp2Z
+while/lstm_cell_398/MatMul_1/ReadVariableOp+while/lstm_cell_398/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_body_16715291
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_398_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_398_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_398_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_398_matmul_readvariableop_resource:	ШG
4while_lstm_cell_398_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_398_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_398/BiasAdd/ReadVariableOpЂ)while/lstm_cell_398/MatMul/ReadVariableOpЂ+while/lstm_cell_398/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_398/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_398_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_398/MatMul/ReadVariableOpк
while/lstm_cell_398/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/MatMulв
+while/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_398_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_398/MatMul_1/ReadVariableOpУ
while/lstm_cell_398/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/MatMul_1М
while/lstm_cell_398/addAddV2$while/lstm_cell_398/MatMul:product:0&while/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/addЫ
*while/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_398_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_398/BiasAdd/ReadVariableOpЩ
while/lstm_cell_398/BiasAddBiasAddwhile/lstm_cell_398/add:z:02while/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/BiasAdd
#while/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_398/split/split_dim
while/lstm_cell_398/splitSplit,while/lstm_cell_398/split/split_dim:output:0$while/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_398/split
while/lstm_cell_398/SigmoidSigmoid"while/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Sigmoid
while/lstm_cell_398/Sigmoid_1Sigmoid"while/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Sigmoid_1Ѓ
while/lstm_cell_398/mulMul!while/lstm_cell_398/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/mul
while/lstm_cell_398/ReluRelu"while/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/ReluИ
while/lstm_cell_398/mul_1Mulwhile/lstm_cell_398/Sigmoid:y:0&while/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/mul_1­
while/lstm_cell_398/add_1AddV2while/lstm_cell_398/mul:z:0while/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/add_1
while/lstm_cell_398/Sigmoid_2Sigmoid"while/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Sigmoid_2
while/lstm_cell_398/Relu_1Reluwhile/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Relu_1М
while/lstm_cell_398/mul_2Mul!while/lstm_cell_398/Sigmoid_2:y:0(while/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_398/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_398/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_398/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_398/BiasAdd/ReadVariableOp*^while/lstm_cell_398/MatMul/ReadVariableOp,^while/lstm_cell_398/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_398_biasadd_readvariableop_resource5while_lstm_cell_398_biasadd_readvariableop_resource_0"n
4while_lstm_cell_398_matmul_1_readvariableop_resource6while_lstm_cell_398_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_398_matmul_readvariableop_resource4while_lstm_cell_398_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_398/BiasAdd/ReadVariableOp*while/lstm_cell_398/BiasAdd/ReadVariableOp2V
)while/lstm_cell_398/MatMul/ReadVariableOp)while/lstm_cell_398/MatMul/ReadVariableOp2Z
+while/lstm_cell_398/MatMul_1/ReadVariableOp+while/lstm_cell_398/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
K__inference_lstm_cell_397_layer_call_and_return_conditional_losses_16713673

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
while_body_16718326
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_397_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_397_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_397_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_397_matmul_readvariableop_resource:	ШG
4while_lstm_cell_397_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_397_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_397/BiasAdd/ReadVariableOpЂ)while/lstm_cell_397/MatMul/ReadVariableOpЂ+while/lstm_cell_397/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_397/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_397_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_397/MatMul/ReadVariableOpк
while/lstm_cell_397/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/MatMulв
+while/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_397_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_397/MatMul_1/ReadVariableOpУ
while/lstm_cell_397/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/MatMul_1М
while/lstm_cell_397/addAddV2$while/lstm_cell_397/MatMul:product:0&while/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/addЫ
*while/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_397_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_397/BiasAdd/ReadVariableOpЩ
while/lstm_cell_397/BiasAddBiasAddwhile/lstm_cell_397/add:z:02while/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/BiasAdd
#while/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_397/split/split_dim
while/lstm_cell_397/splitSplit,while/lstm_cell_397/split/split_dim:output:0$while/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_397/split
while/lstm_cell_397/SigmoidSigmoid"while/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Sigmoid
while/lstm_cell_397/Sigmoid_1Sigmoid"while/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Sigmoid_1Ѓ
while/lstm_cell_397/mulMul!while/lstm_cell_397/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/mul
while/lstm_cell_397/ReluRelu"while/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/ReluИ
while/lstm_cell_397/mul_1Mulwhile/lstm_cell_397/Sigmoid:y:0&while/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/mul_1­
while/lstm_cell_397/add_1AddV2while/lstm_cell_397/mul:z:0while/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/add_1
while/lstm_cell_397/Sigmoid_2Sigmoid"while/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Sigmoid_2
while/lstm_cell_397/Relu_1Reluwhile/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Relu_1М
while/lstm_cell_397/mul_2Mul!while/lstm_cell_397/Sigmoid_2:y:0(while/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_397/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_397/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_397/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_397/BiasAdd/ReadVariableOp*^while/lstm_cell_397/MatMul/ReadVariableOp,^while/lstm_cell_397/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_397_biasadd_readvariableop_resource5while_lstm_cell_397_biasadd_readvariableop_resource_0"n
4while_lstm_cell_397_matmul_1_readvariableop_resource6while_lstm_cell_397_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_397_matmul_readvariableop_resource4while_lstm_cell_397_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_397/BiasAdd/ReadVariableOp*while/lstm_cell_397/BiasAdd/ReadVariableOp2V
)while/lstm_cell_397/MatMul/ReadVariableOp)while/lstm_cell_397/MatMul/ReadVariableOp2Z
+while/lstm_cell_397/MatMul_1/ReadVariableOp+while/lstm_cell_397/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
K__inference_lstm_cell_397_layer_call_and_return_conditional_losses_16713819

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
ў

K__inference_lstm_cell_398_layer_call_and_return_conditional_losses_16719532

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
М
љ
0__inference_lstm_cell_398_layer_call_fn_16719483

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
K__inference_lstm_cell_398_layer_call_and_return_conditional_losses_167143052
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
%backward_lstm_132_while_cond_16716939@
<backward_lstm_132_while_backward_lstm_132_while_loop_counterF
Bbackward_lstm_132_while_backward_lstm_132_while_maximum_iterations'
#backward_lstm_132_while_placeholder)
%backward_lstm_132_while_placeholder_1)
%backward_lstm_132_while_placeholder_2)
%backward_lstm_132_while_placeholder_3B
>backward_lstm_132_while_less_backward_lstm_132_strided_slice_1Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16716939___redundant_placeholder0Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16716939___redundant_placeholder1Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16716939___redundant_placeholder2Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16716939___redundant_placeholder3$
 backward_lstm_132_while_identity
Ъ
backward_lstm_132/while/LessLess#backward_lstm_132_while_placeholder>backward_lstm_132_while_less_backward_lstm_132_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_132/while/Less
 backward_lstm_132/while/IdentityIdentity backward_lstm_132/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_132/while/Identity"M
 backward_lstm_132_while_identity)backward_lstm_132/while/Identity:output:0*(
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
Ф§
у
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_16717026
inputs_0P
=forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource:	ШR
?forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource:	2ШM
>forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource:	ШQ
>backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource:	ШS
@backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource:	2ШN
?backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource:	Ш
identityЂ6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpЂ5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpЂ7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpЂbackward_lstm_132/whileЂ5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpЂ4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpЂ6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOpЂforward_lstm_132/whileh
forward_lstm_132/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_132/Shape
$forward_lstm_132/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_132/strided_slice/stack
&forward_lstm_132/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_132/strided_slice/stack_1
&forward_lstm_132/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_132/strided_slice/stack_2Ш
forward_lstm_132/strided_sliceStridedSliceforward_lstm_132/Shape:output:0-forward_lstm_132/strided_slice/stack:output:0/forward_lstm_132/strided_slice/stack_1:output:0/forward_lstm_132/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_132/strided_slice~
forward_lstm_132/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_132/zeros/mul/yА
forward_lstm_132/zeros/mulMul'forward_lstm_132/strided_slice:output:0%forward_lstm_132/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros/mul
forward_lstm_132/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_132/zeros/Less/yЋ
forward_lstm_132/zeros/LessLessforward_lstm_132/zeros/mul:z:0&forward_lstm_132/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros/Less
forward_lstm_132/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_132/zeros/packed/1Ч
forward_lstm_132/zeros/packedPack'forward_lstm_132/strided_slice:output:0(forward_lstm_132/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_132/zeros/packed
forward_lstm_132/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_132/zeros/ConstЙ
forward_lstm_132/zerosFill&forward_lstm_132/zeros/packed:output:0%forward_lstm_132/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/zeros
forward_lstm_132/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_132/zeros_1/mul/yЖ
forward_lstm_132/zeros_1/mulMul'forward_lstm_132/strided_slice:output:0'forward_lstm_132/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros_1/mul
forward_lstm_132/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
forward_lstm_132/zeros_1/Less/yГ
forward_lstm_132/zeros_1/LessLess forward_lstm_132/zeros_1/mul:z:0(forward_lstm_132/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros_1/Less
!forward_lstm_132/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_132/zeros_1/packed/1Э
forward_lstm_132/zeros_1/packedPack'forward_lstm_132/strided_slice:output:0*forward_lstm_132/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_132/zeros_1/packed
forward_lstm_132/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_132/zeros_1/ConstС
forward_lstm_132/zeros_1Fill(forward_lstm_132/zeros_1/packed:output:0'forward_lstm_132/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/zeros_1
forward_lstm_132/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_132/transpose/permС
forward_lstm_132/transpose	Transposeinputs_0(forward_lstm_132/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
forward_lstm_132/transpose
forward_lstm_132/Shape_1Shapeforward_lstm_132/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_132/Shape_1
&forward_lstm_132/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_132/strided_slice_1/stack
(forward_lstm_132/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_1/stack_1
(forward_lstm_132/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_1/stack_2д
 forward_lstm_132/strided_slice_1StridedSlice!forward_lstm_132/Shape_1:output:0/forward_lstm_132/strided_slice_1/stack:output:01forward_lstm_132/strided_slice_1/stack_1:output:01forward_lstm_132/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_132/strided_slice_1Ї
,forward_lstm_132/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,forward_lstm_132/TensorArrayV2/element_shapeі
forward_lstm_132/TensorArrayV2TensorListReserve5forward_lstm_132/TensorArrayV2/element_shape:output:0)forward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_132/TensorArrayV2с
Fforward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2H
Fforward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8forward_lstm_132/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_132/transpose:y:0Oforward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_132/TensorArrayUnstack/TensorListFromTensor
&forward_lstm_132/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_132/strided_slice_2/stack
(forward_lstm_132/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_2/stack_1
(forward_lstm_132/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_2/stack_2ы
 forward_lstm_132/strided_slice_2StridedSliceforward_lstm_132/transpose:y:0/forward_lstm_132/strided_slice_2/stack:output:01forward_lstm_132/strided_slice_2/stack_1:output:01forward_lstm_132/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2"
 forward_lstm_132/strided_slice_2ы
4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpReadVariableOp=forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpє
%forward_lstm_132/lstm_cell_397/MatMulMatMul)forward_lstm_132/strided_slice_2:output:0<forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_132/lstm_cell_397/MatMulё
6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOp?forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp№
'forward_lstm_132/lstm_cell_397/MatMul_1MatMulforward_lstm_132/zeros:output:0>forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_132/lstm_cell_397/MatMul_1ш
"forward_lstm_132/lstm_cell_397/addAddV2/forward_lstm_132/lstm_cell_397/MatMul:product:01forward_lstm_132/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_132/lstm_cell_397/addъ
5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOp>forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpѕ
&forward_lstm_132/lstm_cell_397/BiasAddBiasAdd&forward_lstm_132/lstm_cell_397/add:z:0=forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_132/lstm_cell_397/BiasAddЂ
.forward_lstm_132/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.forward_lstm_132/lstm_cell_397/split/split_dimЛ
$forward_lstm_132/lstm_cell_397/splitSplit7forward_lstm_132/lstm_cell_397/split/split_dim:output:0/forward_lstm_132/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$forward_lstm_132/lstm_cell_397/splitМ
&forward_lstm_132/lstm_cell_397/SigmoidSigmoid-forward_lstm_132/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_132/lstm_cell_397/SigmoidР
(forward_lstm_132/lstm_cell_397/Sigmoid_1Sigmoid-forward_lstm_132/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_132/lstm_cell_397/Sigmoid_1в
"forward_lstm_132/lstm_cell_397/mulMul,forward_lstm_132/lstm_cell_397/Sigmoid_1:y:0!forward_lstm_132/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_132/lstm_cell_397/mulГ
#forward_lstm_132/lstm_cell_397/ReluRelu-forward_lstm_132/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_132/lstm_cell_397/Reluф
$forward_lstm_132/lstm_cell_397/mul_1Mul*forward_lstm_132/lstm_cell_397/Sigmoid:y:01forward_lstm_132/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_132/lstm_cell_397/mul_1й
$forward_lstm_132/lstm_cell_397/add_1AddV2&forward_lstm_132/lstm_cell_397/mul:z:0(forward_lstm_132/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_132/lstm_cell_397/add_1Р
(forward_lstm_132/lstm_cell_397/Sigmoid_2Sigmoid-forward_lstm_132/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_132/lstm_cell_397/Sigmoid_2В
%forward_lstm_132/lstm_cell_397/Relu_1Relu(forward_lstm_132/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_132/lstm_cell_397/Relu_1ш
$forward_lstm_132/lstm_cell_397/mul_2Mul,forward_lstm_132/lstm_cell_397/Sigmoid_2:y:03forward_lstm_132/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_132/lstm_cell_397/mul_2Б
.forward_lstm_132/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.forward_lstm_132/TensorArrayV2_1/element_shapeќ
 forward_lstm_132/TensorArrayV2_1TensorListReserve7forward_lstm_132/TensorArrayV2_1/element_shape:output:0)forward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_132/TensorArrayV2_1p
forward_lstm_132/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_132/timeЁ
)forward_lstm_132/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)forward_lstm_132/while/maximum_iterations
#forward_lstm_132/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_132/while/loop_counter
forward_lstm_132/whileWhile,forward_lstm_132/while/loop_counter:output:02forward_lstm_132/while/maximum_iterations:output:0forward_lstm_132/time:output:0)forward_lstm_132/TensorArrayV2_1:handle:0forward_lstm_132/zeros:output:0!forward_lstm_132/zeros_1:output:0)forward_lstm_132/strided_slice_1:output:0Hforward_lstm_132/TensorArrayUnstack/TensorListFromTensor:output_handle:0=forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource?forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource>forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource*
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
$forward_lstm_132_while_body_16716791*0
cond(R&
$forward_lstm_132_while_cond_16716790*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
forward_lstm_132/whileз
Aforward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Aforward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3forward_lstm_132/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_132/while:output:3Jforward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3forward_lstm_132/TensorArrayV2Stack/TensorListStackЃ
&forward_lstm_132/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&forward_lstm_132/strided_slice_3/stack
(forward_lstm_132/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_132/strided_slice_3/stack_1
(forward_lstm_132/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_3/stack_2
 forward_lstm_132/strided_slice_3StridedSlice<forward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_132/strided_slice_3/stack:output:01forward_lstm_132/strided_slice_3/stack_1:output:01forward_lstm_132/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 forward_lstm_132/strided_slice_3
!forward_lstm_132/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_132/transpose_1/permђ
forward_lstm_132/transpose_1	Transpose<forward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_132/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_132/transpose_1
forward_lstm_132/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_132/runtimej
backward_lstm_132/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_132/Shape
%backward_lstm_132/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_132/strided_slice/stack
'backward_lstm_132/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_132/strided_slice/stack_1
'backward_lstm_132/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_132/strided_slice/stack_2Ю
backward_lstm_132/strided_sliceStridedSlice backward_lstm_132/Shape:output:0.backward_lstm_132/strided_slice/stack:output:00backward_lstm_132/strided_slice/stack_1:output:00backward_lstm_132/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_132/strided_slice
backward_lstm_132/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_132/zeros/mul/yД
backward_lstm_132/zeros/mulMul(backward_lstm_132/strided_slice:output:0&backward_lstm_132/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/zeros/mul
backward_lstm_132/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_132/zeros/Less/yЏ
backward_lstm_132/zeros/LessLessbackward_lstm_132/zeros/mul:z:0'backward_lstm_132/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/zeros/Less
 backward_lstm_132/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_132/zeros/packed/1Ы
backward_lstm_132/zeros/packedPack(backward_lstm_132/strided_slice:output:0)backward_lstm_132/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_132/zeros/packed
backward_lstm_132/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_132/zeros/ConstН
backward_lstm_132/zerosFill'backward_lstm_132/zeros/packed:output:0&backward_lstm_132/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_132/zeros
backward_lstm_132/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_132/zeros_1/mul/yК
backward_lstm_132/zeros_1/mulMul(backward_lstm_132/strided_slice:output:0(backward_lstm_132/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/zeros_1/mul
 backward_lstm_132/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 backward_lstm_132/zeros_1/Less/yЗ
backward_lstm_132/zeros_1/LessLess!backward_lstm_132/zeros_1/mul:z:0)backward_lstm_132/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_132/zeros_1/Less
"backward_lstm_132/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_132/zeros_1/packed/1б
 backward_lstm_132/zeros_1/packedPack(backward_lstm_132/strided_slice:output:0+backward_lstm_132/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_132/zeros_1/packed
backward_lstm_132/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_132/zeros_1/ConstХ
backward_lstm_132/zeros_1Fill)backward_lstm_132/zeros_1/packed:output:0(backward_lstm_132/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_132/zeros_1
 backward_lstm_132/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_132/transpose/permФ
backward_lstm_132/transpose	Transposeinputs_0)backward_lstm_132/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_132/transpose
backward_lstm_132/Shape_1Shapebackward_lstm_132/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_132/Shape_1
'backward_lstm_132/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_132/strided_slice_1/stack 
)backward_lstm_132/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_1/stack_1 
)backward_lstm_132/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_1/stack_2к
!backward_lstm_132/strided_slice_1StridedSlice"backward_lstm_132/Shape_1:output:00backward_lstm_132/strided_slice_1/stack:output:02backward_lstm_132/strided_slice_1/stack_1:output:02backward_lstm_132/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_132/strided_slice_1Љ
-backward_lstm_132/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2/
-backward_lstm_132/TensorArrayV2/element_shapeњ
backward_lstm_132/TensorArrayV2TensorListReserve6backward_lstm_132/TensorArrayV2/element_shape:output:0*backward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_132/TensorArrayV2
 backward_lstm_132/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_132/ReverseV2/axisл
backward_lstm_132/ReverseV2	ReverseV2backward_lstm_132/transpose:y:0)backward_lstm_132/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_132/ReverseV2у
Gbackward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2I
Gbackward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeХ
9backward_lstm_132/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_132/ReverseV2:output:0Pbackward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_132/TensorArrayUnstack/TensorListFromTensor
'backward_lstm_132/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_132/strided_slice_2/stack 
)backward_lstm_132/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_2/stack_1 
)backward_lstm_132/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_2/stack_2ё
!backward_lstm_132/strided_slice_2StridedSlicebackward_lstm_132/transpose:y:00backward_lstm_132/strided_slice_2/stack:output:02backward_lstm_132/strided_slice_2/stack_1:output:02backward_lstm_132/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2#
!backward_lstm_132/strided_slice_2ю
5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpReadVariableOp>backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype027
5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpј
&backward_lstm_132/lstm_cell_398/MatMulMatMul*backward_lstm_132/strided_slice_2:output:0=backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_132/lstm_cell_398/MatMulє
7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOp@backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype029
7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpє
(backward_lstm_132/lstm_cell_398/MatMul_1MatMul backward_lstm_132/zeros:output:0?backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_132/lstm_cell_398/MatMul_1ь
#backward_lstm_132/lstm_cell_398/addAddV20backward_lstm_132/lstm_cell_398/MatMul:product:02backward_lstm_132/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_132/lstm_cell_398/addэ
6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOp?backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype028
6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpљ
'backward_lstm_132/lstm_cell_398/BiasAddBiasAdd'backward_lstm_132/lstm_cell_398/add:z:0>backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_132/lstm_cell_398/BiasAddЄ
/backward_lstm_132/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/backward_lstm_132/lstm_cell_398/split/split_dimП
%backward_lstm_132/lstm_cell_398/splitSplit8backward_lstm_132/lstm_cell_398/split/split_dim:output:00backward_lstm_132/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2'
%backward_lstm_132/lstm_cell_398/splitП
'backward_lstm_132/lstm_cell_398/SigmoidSigmoid.backward_lstm_132/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_132/lstm_cell_398/SigmoidУ
)backward_lstm_132/lstm_cell_398/Sigmoid_1Sigmoid.backward_lstm_132/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_132/lstm_cell_398/Sigmoid_1ж
#backward_lstm_132/lstm_cell_398/mulMul-backward_lstm_132/lstm_cell_398/Sigmoid_1:y:0"backward_lstm_132/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_132/lstm_cell_398/mulЖ
$backward_lstm_132/lstm_cell_398/ReluRelu.backward_lstm_132/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_132/lstm_cell_398/Reluш
%backward_lstm_132/lstm_cell_398/mul_1Mul+backward_lstm_132/lstm_cell_398/Sigmoid:y:02backward_lstm_132/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_132/lstm_cell_398/mul_1н
%backward_lstm_132/lstm_cell_398/add_1AddV2'backward_lstm_132/lstm_cell_398/mul:z:0)backward_lstm_132/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_132/lstm_cell_398/add_1У
)backward_lstm_132/lstm_cell_398/Sigmoid_2Sigmoid.backward_lstm_132/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_132/lstm_cell_398/Sigmoid_2Е
&backward_lstm_132/lstm_cell_398/Relu_1Relu)backward_lstm_132/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_132/lstm_cell_398/Relu_1ь
%backward_lstm_132/lstm_cell_398/mul_2Mul-backward_lstm_132/lstm_cell_398/Sigmoid_2:y:04backward_lstm_132/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_132/lstm_cell_398/mul_2Г
/backward_lstm_132/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   21
/backward_lstm_132/TensorArrayV2_1/element_shape
!backward_lstm_132/TensorArrayV2_1TensorListReserve8backward_lstm_132/TensorArrayV2_1/element_shape:output:0*backward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_132/TensorArrayV2_1r
backward_lstm_132/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_132/timeЃ
*backward_lstm_132/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*backward_lstm_132/while/maximum_iterations
$backward_lstm_132/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_132/while/loop_counter 
backward_lstm_132/whileWhile-backward_lstm_132/while/loop_counter:output:03backward_lstm_132/while/maximum_iterations:output:0backward_lstm_132/time:output:0*backward_lstm_132/TensorArrayV2_1:handle:0 backward_lstm_132/zeros:output:0"backward_lstm_132/zeros_1:output:0*backward_lstm_132/strided_slice_1:output:0Ibackward_lstm_132/TensorArrayUnstack/TensorListFromTensor:output_handle:0>backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource@backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource?backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource*
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
%backward_lstm_132_while_body_16716940*1
cond)R'
%backward_lstm_132_while_cond_16716939*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
backward_lstm_132/whileй
Bbackward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2D
Bbackward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeЙ
4backward_lstm_132/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_132/while:output:3Kbackward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype026
4backward_lstm_132/TensorArrayV2Stack/TensorListStackЅ
'backward_lstm_132/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2)
'backward_lstm_132/strided_slice_3/stack 
)backward_lstm_132/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_132/strided_slice_3/stack_1 
)backward_lstm_132/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_3/stack_2
!backward_lstm_132/strided_slice_3StridedSlice=backward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_132/strided_slice_3/stack:output:02backward_lstm_132/strided_slice_3/stack_1:output:02backward_lstm_132/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2#
!backward_lstm_132/strided_slice_3
"backward_lstm_132/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_132/transpose_1/permі
backward_lstm_132/transpose_1	Transpose=backward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_132/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_132/transpose_1
backward_lstm_132/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_132/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisФ
concatConcatV2)forward_lstm_132/strided_slice_3:output:0*backward_lstm_132/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp7^backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp6^backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp8^backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp^backward_lstm_132/while6^forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp5^forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp7^forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp^forward_lstm_132/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2p
6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp2n
5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp2r
7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp22
backward_lstm_132/whilebackward_lstm_132/while2n
5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp2l
4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp2p
6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp20
forward_lstm_132/whileforward_lstm_132/while:g c
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0

д
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_16715596

inputs,
forward_lstm_132_16715579:	Ш,
forward_lstm_132_16715581:	2Ш(
forward_lstm_132_16715583:	Ш-
backward_lstm_132_16715586:	Ш-
backward_lstm_132_16715588:	2Ш)
backward_lstm_132_16715590:	Ш
identityЂ)backward_lstm_132/StatefulPartitionedCallЂ(forward_lstm_132/StatefulPartitionedCallп
(forward_lstm_132/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_132_16715579forward_lstm_132_16715581forward_lstm_132_16715583*
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
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_167155482*
(forward_lstm_132/StatefulPartitionedCallх
)backward_lstm_132/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_132_16715586backward_lstm_132_16715588backward_lstm_132_16715590*
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
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_167153752+
)backward_lstm_132/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisд
concatConcatV21forward_lstm_132/StatefulPartitionedCall:output:02backward_lstm_132/StatefulPartitionedCall:output:0concat/axis:output:0*
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
NoOpNoOp*^backward_lstm_132/StatefulPartitionedCall)^forward_lstm_132/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2V
)backward_lstm_132/StatefulPartitionedCall)backward_lstm_132/StatefulPartitionedCall2T
(forward_lstm_132/StatefulPartitionedCall(forward_lstm_132/StatefulPartitionedCall:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
т
С
4__inference_backward_lstm_132_layer_call_fn_16718745

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
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_167151832
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
L__inference_sequential_132_layer_call_and_return_conditional_losses_16716624

inputs
inputs_1	-
bidirectional_132_16716605:	Ш-
bidirectional_132_16716607:	2Ш)
bidirectional_132_16716609:	Ш-
bidirectional_132_16716611:	Ш-
bidirectional_132_16716613:	2Ш)
bidirectional_132_16716615:	Ш$
dense_132_16716618:d 
dense_132_16716620:
identityЂ)bidirectional_132/StatefulPartitionedCallЂ!dense_132/StatefulPartitionedCallЪ
)bidirectional_132/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_132_16716605bidirectional_132_16716607bidirectional_132_16716609bidirectional_132_16716611bidirectional_132_16716613bidirectional_132_16716615*
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
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_167164742+
)bidirectional_132/StatefulPartitionedCallЫ
!dense_132/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_132/StatefulPartitionedCall:output:0dense_132_16716618dense_132_16716620*
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
G__inference_dense_132_layer_call_and_return_conditional_losses_167160592#
!dense_132/StatefulPartitionedCall
IdentityIdentity*dense_132/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp*^bidirectional_132/StatefulPartitionedCall"^dense_132/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2V
)bidirectional_132/StatefulPartitionedCall)bidirectional_132/StatefulPartitionedCall2F
!dense_132/StatefulPartitionedCall!dense_132/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ў

K__inference_lstm_cell_398_layer_call_and_return_conditional_losses_16719564

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
ж
У
4__inference_backward_lstm_132_layer_call_fn_16718734
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
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_167146002
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
Ф§
у
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_16717328
inputs_0P
=forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource:	ШR
?forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource:	2ШM
>forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource:	ШQ
>backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource:	ШS
@backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource:	2ШN
?backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource:	Ш
identityЂ6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpЂ5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpЂ7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpЂbackward_lstm_132/whileЂ5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpЂ4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpЂ6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOpЂforward_lstm_132/whileh
forward_lstm_132/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_132/Shape
$forward_lstm_132/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_132/strided_slice/stack
&forward_lstm_132/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_132/strided_slice/stack_1
&forward_lstm_132/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_132/strided_slice/stack_2Ш
forward_lstm_132/strided_sliceStridedSliceforward_lstm_132/Shape:output:0-forward_lstm_132/strided_slice/stack:output:0/forward_lstm_132/strided_slice/stack_1:output:0/forward_lstm_132/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_132/strided_slice~
forward_lstm_132/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_132/zeros/mul/yА
forward_lstm_132/zeros/mulMul'forward_lstm_132/strided_slice:output:0%forward_lstm_132/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros/mul
forward_lstm_132/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_132/zeros/Less/yЋ
forward_lstm_132/zeros/LessLessforward_lstm_132/zeros/mul:z:0&forward_lstm_132/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros/Less
forward_lstm_132/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_132/zeros/packed/1Ч
forward_lstm_132/zeros/packedPack'forward_lstm_132/strided_slice:output:0(forward_lstm_132/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_132/zeros/packed
forward_lstm_132/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_132/zeros/ConstЙ
forward_lstm_132/zerosFill&forward_lstm_132/zeros/packed:output:0%forward_lstm_132/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/zeros
forward_lstm_132/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_132/zeros_1/mul/yЖ
forward_lstm_132/zeros_1/mulMul'forward_lstm_132/strided_slice:output:0'forward_lstm_132/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros_1/mul
forward_lstm_132/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
forward_lstm_132/zeros_1/Less/yГ
forward_lstm_132/zeros_1/LessLess forward_lstm_132/zeros_1/mul:z:0(forward_lstm_132/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros_1/Less
!forward_lstm_132/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_132/zeros_1/packed/1Э
forward_lstm_132/zeros_1/packedPack'forward_lstm_132/strided_slice:output:0*forward_lstm_132/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_132/zeros_1/packed
forward_lstm_132/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_132/zeros_1/ConstС
forward_lstm_132/zeros_1Fill(forward_lstm_132/zeros_1/packed:output:0'forward_lstm_132/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/zeros_1
forward_lstm_132/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_132/transpose/permС
forward_lstm_132/transpose	Transposeinputs_0(forward_lstm_132/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
forward_lstm_132/transpose
forward_lstm_132/Shape_1Shapeforward_lstm_132/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_132/Shape_1
&forward_lstm_132/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_132/strided_slice_1/stack
(forward_lstm_132/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_1/stack_1
(forward_lstm_132/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_1/stack_2д
 forward_lstm_132/strided_slice_1StridedSlice!forward_lstm_132/Shape_1:output:0/forward_lstm_132/strided_slice_1/stack:output:01forward_lstm_132/strided_slice_1/stack_1:output:01forward_lstm_132/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_132/strided_slice_1Ї
,forward_lstm_132/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,forward_lstm_132/TensorArrayV2/element_shapeі
forward_lstm_132/TensorArrayV2TensorListReserve5forward_lstm_132/TensorArrayV2/element_shape:output:0)forward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_132/TensorArrayV2с
Fforward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2H
Fforward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8forward_lstm_132/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_132/transpose:y:0Oforward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_132/TensorArrayUnstack/TensorListFromTensor
&forward_lstm_132/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_132/strided_slice_2/stack
(forward_lstm_132/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_2/stack_1
(forward_lstm_132/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_2/stack_2ы
 forward_lstm_132/strided_slice_2StridedSliceforward_lstm_132/transpose:y:0/forward_lstm_132/strided_slice_2/stack:output:01forward_lstm_132/strided_slice_2/stack_1:output:01forward_lstm_132/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2"
 forward_lstm_132/strided_slice_2ы
4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpReadVariableOp=forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpє
%forward_lstm_132/lstm_cell_397/MatMulMatMul)forward_lstm_132/strided_slice_2:output:0<forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_132/lstm_cell_397/MatMulё
6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOp?forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp№
'forward_lstm_132/lstm_cell_397/MatMul_1MatMulforward_lstm_132/zeros:output:0>forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_132/lstm_cell_397/MatMul_1ш
"forward_lstm_132/lstm_cell_397/addAddV2/forward_lstm_132/lstm_cell_397/MatMul:product:01forward_lstm_132/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_132/lstm_cell_397/addъ
5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOp>forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpѕ
&forward_lstm_132/lstm_cell_397/BiasAddBiasAdd&forward_lstm_132/lstm_cell_397/add:z:0=forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_132/lstm_cell_397/BiasAddЂ
.forward_lstm_132/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.forward_lstm_132/lstm_cell_397/split/split_dimЛ
$forward_lstm_132/lstm_cell_397/splitSplit7forward_lstm_132/lstm_cell_397/split/split_dim:output:0/forward_lstm_132/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$forward_lstm_132/lstm_cell_397/splitМ
&forward_lstm_132/lstm_cell_397/SigmoidSigmoid-forward_lstm_132/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_132/lstm_cell_397/SigmoidР
(forward_lstm_132/lstm_cell_397/Sigmoid_1Sigmoid-forward_lstm_132/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_132/lstm_cell_397/Sigmoid_1в
"forward_lstm_132/lstm_cell_397/mulMul,forward_lstm_132/lstm_cell_397/Sigmoid_1:y:0!forward_lstm_132/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_132/lstm_cell_397/mulГ
#forward_lstm_132/lstm_cell_397/ReluRelu-forward_lstm_132/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_132/lstm_cell_397/Reluф
$forward_lstm_132/lstm_cell_397/mul_1Mul*forward_lstm_132/lstm_cell_397/Sigmoid:y:01forward_lstm_132/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_132/lstm_cell_397/mul_1й
$forward_lstm_132/lstm_cell_397/add_1AddV2&forward_lstm_132/lstm_cell_397/mul:z:0(forward_lstm_132/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_132/lstm_cell_397/add_1Р
(forward_lstm_132/lstm_cell_397/Sigmoid_2Sigmoid-forward_lstm_132/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_132/lstm_cell_397/Sigmoid_2В
%forward_lstm_132/lstm_cell_397/Relu_1Relu(forward_lstm_132/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_132/lstm_cell_397/Relu_1ш
$forward_lstm_132/lstm_cell_397/mul_2Mul,forward_lstm_132/lstm_cell_397/Sigmoid_2:y:03forward_lstm_132/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_132/lstm_cell_397/mul_2Б
.forward_lstm_132/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.forward_lstm_132/TensorArrayV2_1/element_shapeќ
 forward_lstm_132/TensorArrayV2_1TensorListReserve7forward_lstm_132/TensorArrayV2_1/element_shape:output:0)forward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_132/TensorArrayV2_1p
forward_lstm_132/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_132/timeЁ
)forward_lstm_132/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)forward_lstm_132/while/maximum_iterations
#forward_lstm_132/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_132/while/loop_counter
forward_lstm_132/whileWhile,forward_lstm_132/while/loop_counter:output:02forward_lstm_132/while/maximum_iterations:output:0forward_lstm_132/time:output:0)forward_lstm_132/TensorArrayV2_1:handle:0forward_lstm_132/zeros:output:0!forward_lstm_132/zeros_1:output:0)forward_lstm_132/strided_slice_1:output:0Hforward_lstm_132/TensorArrayUnstack/TensorListFromTensor:output_handle:0=forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource?forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource>forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource*
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
$forward_lstm_132_while_body_16717093*0
cond(R&
$forward_lstm_132_while_cond_16717092*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
forward_lstm_132/whileз
Aforward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Aforward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3forward_lstm_132/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_132/while:output:3Jforward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3forward_lstm_132/TensorArrayV2Stack/TensorListStackЃ
&forward_lstm_132/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&forward_lstm_132/strided_slice_3/stack
(forward_lstm_132/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_132/strided_slice_3/stack_1
(forward_lstm_132/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_3/stack_2
 forward_lstm_132/strided_slice_3StridedSlice<forward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_132/strided_slice_3/stack:output:01forward_lstm_132/strided_slice_3/stack_1:output:01forward_lstm_132/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 forward_lstm_132/strided_slice_3
!forward_lstm_132/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_132/transpose_1/permђ
forward_lstm_132/transpose_1	Transpose<forward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_132/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_132/transpose_1
forward_lstm_132/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_132/runtimej
backward_lstm_132/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_132/Shape
%backward_lstm_132/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_132/strided_slice/stack
'backward_lstm_132/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_132/strided_slice/stack_1
'backward_lstm_132/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_132/strided_slice/stack_2Ю
backward_lstm_132/strided_sliceStridedSlice backward_lstm_132/Shape:output:0.backward_lstm_132/strided_slice/stack:output:00backward_lstm_132/strided_slice/stack_1:output:00backward_lstm_132/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_132/strided_slice
backward_lstm_132/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_132/zeros/mul/yД
backward_lstm_132/zeros/mulMul(backward_lstm_132/strided_slice:output:0&backward_lstm_132/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/zeros/mul
backward_lstm_132/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_132/zeros/Less/yЏ
backward_lstm_132/zeros/LessLessbackward_lstm_132/zeros/mul:z:0'backward_lstm_132/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/zeros/Less
 backward_lstm_132/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_132/zeros/packed/1Ы
backward_lstm_132/zeros/packedPack(backward_lstm_132/strided_slice:output:0)backward_lstm_132/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_132/zeros/packed
backward_lstm_132/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_132/zeros/ConstН
backward_lstm_132/zerosFill'backward_lstm_132/zeros/packed:output:0&backward_lstm_132/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_132/zeros
backward_lstm_132/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_132/zeros_1/mul/yК
backward_lstm_132/zeros_1/mulMul(backward_lstm_132/strided_slice:output:0(backward_lstm_132/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/zeros_1/mul
 backward_lstm_132/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 backward_lstm_132/zeros_1/Less/yЗ
backward_lstm_132/zeros_1/LessLess!backward_lstm_132/zeros_1/mul:z:0)backward_lstm_132/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_132/zeros_1/Less
"backward_lstm_132/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_132/zeros_1/packed/1б
 backward_lstm_132/zeros_1/packedPack(backward_lstm_132/strided_slice:output:0+backward_lstm_132/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_132/zeros_1/packed
backward_lstm_132/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_132/zeros_1/ConstХ
backward_lstm_132/zeros_1Fill)backward_lstm_132/zeros_1/packed:output:0(backward_lstm_132/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_132/zeros_1
 backward_lstm_132/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_132/transpose/permФ
backward_lstm_132/transpose	Transposeinputs_0)backward_lstm_132/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_132/transpose
backward_lstm_132/Shape_1Shapebackward_lstm_132/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_132/Shape_1
'backward_lstm_132/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_132/strided_slice_1/stack 
)backward_lstm_132/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_1/stack_1 
)backward_lstm_132/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_1/stack_2к
!backward_lstm_132/strided_slice_1StridedSlice"backward_lstm_132/Shape_1:output:00backward_lstm_132/strided_slice_1/stack:output:02backward_lstm_132/strided_slice_1/stack_1:output:02backward_lstm_132/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_132/strided_slice_1Љ
-backward_lstm_132/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2/
-backward_lstm_132/TensorArrayV2/element_shapeњ
backward_lstm_132/TensorArrayV2TensorListReserve6backward_lstm_132/TensorArrayV2/element_shape:output:0*backward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_132/TensorArrayV2
 backward_lstm_132/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_132/ReverseV2/axisл
backward_lstm_132/ReverseV2	ReverseV2backward_lstm_132/transpose:y:0)backward_lstm_132/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_132/ReverseV2у
Gbackward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2I
Gbackward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeХ
9backward_lstm_132/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_132/ReverseV2:output:0Pbackward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_132/TensorArrayUnstack/TensorListFromTensor
'backward_lstm_132/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_132/strided_slice_2/stack 
)backward_lstm_132/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_2/stack_1 
)backward_lstm_132/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_2/stack_2ё
!backward_lstm_132/strided_slice_2StridedSlicebackward_lstm_132/transpose:y:00backward_lstm_132/strided_slice_2/stack:output:02backward_lstm_132/strided_slice_2/stack_1:output:02backward_lstm_132/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2#
!backward_lstm_132/strided_slice_2ю
5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpReadVariableOp>backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype027
5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpј
&backward_lstm_132/lstm_cell_398/MatMulMatMul*backward_lstm_132/strided_slice_2:output:0=backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_132/lstm_cell_398/MatMulє
7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOp@backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype029
7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpє
(backward_lstm_132/lstm_cell_398/MatMul_1MatMul backward_lstm_132/zeros:output:0?backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_132/lstm_cell_398/MatMul_1ь
#backward_lstm_132/lstm_cell_398/addAddV20backward_lstm_132/lstm_cell_398/MatMul:product:02backward_lstm_132/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_132/lstm_cell_398/addэ
6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOp?backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype028
6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpљ
'backward_lstm_132/lstm_cell_398/BiasAddBiasAdd'backward_lstm_132/lstm_cell_398/add:z:0>backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_132/lstm_cell_398/BiasAddЄ
/backward_lstm_132/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/backward_lstm_132/lstm_cell_398/split/split_dimП
%backward_lstm_132/lstm_cell_398/splitSplit8backward_lstm_132/lstm_cell_398/split/split_dim:output:00backward_lstm_132/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2'
%backward_lstm_132/lstm_cell_398/splitП
'backward_lstm_132/lstm_cell_398/SigmoidSigmoid.backward_lstm_132/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_132/lstm_cell_398/SigmoidУ
)backward_lstm_132/lstm_cell_398/Sigmoid_1Sigmoid.backward_lstm_132/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_132/lstm_cell_398/Sigmoid_1ж
#backward_lstm_132/lstm_cell_398/mulMul-backward_lstm_132/lstm_cell_398/Sigmoid_1:y:0"backward_lstm_132/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_132/lstm_cell_398/mulЖ
$backward_lstm_132/lstm_cell_398/ReluRelu.backward_lstm_132/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_132/lstm_cell_398/Reluш
%backward_lstm_132/lstm_cell_398/mul_1Mul+backward_lstm_132/lstm_cell_398/Sigmoid:y:02backward_lstm_132/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_132/lstm_cell_398/mul_1н
%backward_lstm_132/lstm_cell_398/add_1AddV2'backward_lstm_132/lstm_cell_398/mul:z:0)backward_lstm_132/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_132/lstm_cell_398/add_1У
)backward_lstm_132/lstm_cell_398/Sigmoid_2Sigmoid.backward_lstm_132/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_132/lstm_cell_398/Sigmoid_2Е
&backward_lstm_132/lstm_cell_398/Relu_1Relu)backward_lstm_132/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_132/lstm_cell_398/Relu_1ь
%backward_lstm_132/lstm_cell_398/mul_2Mul-backward_lstm_132/lstm_cell_398/Sigmoid_2:y:04backward_lstm_132/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_132/lstm_cell_398/mul_2Г
/backward_lstm_132/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   21
/backward_lstm_132/TensorArrayV2_1/element_shape
!backward_lstm_132/TensorArrayV2_1TensorListReserve8backward_lstm_132/TensorArrayV2_1/element_shape:output:0*backward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_132/TensorArrayV2_1r
backward_lstm_132/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_132/timeЃ
*backward_lstm_132/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*backward_lstm_132/while/maximum_iterations
$backward_lstm_132/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_132/while/loop_counter 
backward_lstm_132/whileWhile-backward_lstm_132/while/loop_counter:output:03backward_lstm_132/while/maximum_iterations:output:0backward_lstm_132/time:output:0*backward_lstm_132/TensorArrayV2_1:handle:0 backward_lstm_132/zeros:output:0"backward_lstm_132/zeros_1:output:0*backward_lstm_132/strided_slice_1:output:0Ibackward_lstm_132/TensorArrayUnstack/TensorListFromTensor:output_handle:0>backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource@backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource?backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource*
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
%backward_lstm_132_while_body_16717242*1
cond)R'
%backward_lstm_132_while_cond_16717241*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
backward_lstm_132/whileй
Bbackward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2D
Bbackward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeЙ
4backward_lstm_132/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_132/while:output:3Kbackward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype026
4backward_lstm_132/TensorArrayV2Stack/TensorListStackЅ
'backward_lstm_132/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2)
'backward_lstm_132/strided_slice_3/stack 
)backward_lstm_132/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_132/strided_slice_3/stack_1 
)backward_lstm_132/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_3/stack_2
!backward_lstm_132/strided_slice_3StridedSlice=backward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_132/strided_slice_3/stack:output:02backward_lstm_132/strided_slice_3/stack_1:output:02backward_lstm_132/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2#
!backward_lstm_132/strided_slice_3
"backward_lstm_132/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_132/transpose_1/permі
backward_lstm_132/transpose_1	Transpose=backward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_132/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_132/transpose_1
backward_lstm_132/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_132/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisФ
concatConcatV2)forward_lstm_132/strided_slice_3:output:0*backward_lstm_132/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp7^backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp6^backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp8^backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp^backward_lstm_132/while6^forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp5^forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp7^forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp^forward_lstm_132/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2p
6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp2n
5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp2r
7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp22
backward_lstm_132/whilebackward_lstm_132/while2n
5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp2l
4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp2p
6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp20
forward_lstm_132/whileforward_lstm_132/while:g c
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
аf
ф
%backward_lstm_132_while_body_16715937@
<backward_lstm_132_while_backward_lstm_132_while_loop_counterF
Bbackward_lstm_132_while_backward_lstm_132_while_maximum_iterations'
#backward_lstm_132_while_placeholder)
%backward_lstm_132_while_placeholder_1)
%backward_lstm_132_while_placeholder_2)
%backward_lstm_132_while_placeholder_3)
%backward_lstm_132_while_placeholder_4?
;backward_lstm_132_while_backward_lstm_132_strided_slice_1_0{
wbackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_132_while_less_backward_lstm_132_sub_1_0Y
Fbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0:	Ш[
Hbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0:	2ШV
Gbackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0:	Ш$
 backward_lstm_132_while_identity&
"backward_lstm_132_while_identity_1&
"backward_lstm_132_while_identity_2&
"backward_lstm_132_while_identity_3&
"backward_lstm_132_while_identity_4&
"backward_lstm_132_while_identity_5&
"backward_lstm_132_while_identity_6=
9backward_lstm_132_while_backward_lstm_132_strided_slice_1y
ubackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_132_while_less_backward_lstm_132_sub_1W
Dbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource:	ШY
Fbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource:	2ШT
Ebackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource:	ШЂ<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOpЂ;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpЂ=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOpч
Ibackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2K
Ibackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeП
;backward_lstm_132/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_132_while_placeholderRbackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02=
;backward_lstm_132/while/TensorArrayV2Read/TensorListGetItemЯ
backward_lstm_132/while/LessLess6backward_lstm_132_while_less_backward_lstm_132_sub_1_0#backward_lstm_132_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_132/while/Less
;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpReadVariableOpFbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02=
;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpЂ
,backward_lstm_132/while/lstm_cell_398/MatMulMatMulBbackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem:item:0Cbackward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_132/while/lstm_cell_398/MatMul
=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOpHbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02?
=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp
.backward_lstm_132/while/lstm_cell_398/MatMul_1MatMul%backward_lstm_132_while_placeholder_3Ebackward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ20
.backward_lstm_132/while/lstm_cell_398/MatMul_1
)backward_lstm_132/while/lstm_cell_398/addAddV26backward_lstm_132/while/lstm_cell_398/MatMul:product:08backward_lstm_132/while/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_132/while/lstm_cell_398/add
<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOpGbackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02>
<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp
-backward_lstm_132/while/lstm_cell_398/BiasAddBiasAdd-backward_lstm_132/while/lstm_cell_398/add:z:0Dbackward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_132/while/lstm_cell_398/BiasAddА
5backward_lstm_132/while/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5backward_lstm_132/while/lstm_cell_398/split/split_dimз
+backward_lstm_132/while/lstm_cell_398/splitSplit>backward_lstm_132/while/lstm_cell_398/split/split_dim:output:06backward_lstm_132/while/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2-
+backward_lstm_132/while/lstm_cell_398/splitб
-backward_lstm_132/while/lstm_cell_398/SigmoidSigmoid4backward_lstm_132/while/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22/
-backward_lstm_132/while/lstm_cell_398/Sigmoidе
/backward_lstm_132/while/lstm_cell_398/Sigmoid_1Sigmoid4backward_lstm_132/while/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_132/while/lstm_cell_398/Sigmoid_1ы
)backward_lstm_132/while/lstm_cell_398/mulMul3backward_lstm_132/while/lstm_cell_398/Sigmoid_1:y:0%backward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_132/while/lstm_cell_398/mulШ
*backward_lstm_132/while/lstm_cell_398/ReluRelu4backward_lstm_132/while/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_132/while/lstm_cell_398/Relu
+backward_lstm_132/while/lstm_cell_398/mul_1Mul1backward_lstm_132/while/lstm_cell_398/Sigmoid:y:08backward_lstm_132/while/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_132/while/lstm_cell_398/mul_1ѕ
+backward_lstm_132/while/lstm_cell_398/add_1AddV2-backward_lstm_132/while/lstm_cell_398/mul:z:0/backward_lstm_132/while/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_132/while/lstm_cell_398/add_1е
/backward_lstm_132/while/lstm_cell_398/Sigmoid_2Sigmoid4backward_lstm_132/while/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_132/while/lstm_cell_398/Sigmoid_2Ч
,backward_lstm_132/while/lstm_cell_398/Relu_1Relu/backward_lstm_132/while/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_132/while/lstm_cell_398/Relu_1
+backward_lstm_132/while/lstm_cell_398/mul_2Mul3backward_lstm_132/while/lstm_cell_398/Sigmoid_2:y:0:backward_lstm_132/while/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_132/while/lstm_cell_398/mul_2і
backward_lstm_132/while/SelectSelect backward_lstm_132/while/Less:z:0/backward_lstm_132/while/lstm_cell_398/mul_2:z:0%backward_lstm_132_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_132/while/Selectњ
 backward_lstm_132/while/Select_1Select backward_lstm_132/while/Less:z:0/backward_lstm_132/while/lstm_cell_398/mul_2:z:0%backward_lstm_132_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_132/while/Select_1њ
 backward_lstm_132/while/Select_2Select backward_lstm_132/while/Less:z:0/backward_lstm_132/while/lstm_cell_398/add_1:z:0%backward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_132/while/Select_2Г
<backward_lstm_132/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_132_while_placeholder_1#backward_lstm_132_while_placeholder'backward_lstm_132/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_132/while/TensorArrayV2Write/TensorListSetItem
backward_lstm_132/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_132/while/add/yБ
backward_lstm_132/while/addAddV2#backward_lstm_132_while_placeholder&backward_lstm_132/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/while/add
backward_lstm_132/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_132/while/add_1/yа
backward_lstm_132/while/add_1AddV2<backward_lstm_132_while_backward_lstm_132_while_loop_counter(backward_lstm_132/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/while/add_1Г
 backward_lstm_132/while/IdentityIdentity!backward_lstm_132/while/add_1:z:0^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_132/while/Identityи
"backward_lstm_132/while/Identity_1IdentityBbackward_lstm_132_while_backward_lstm_132_while_maximum_iterations^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_132/while/Identity_1Е
"backward_lstm_132/while/Identity_2Identitybackward_lstm_132/while/add:z:0^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_132/while/Identity_2т
"backward_lstm_132/while/Identity_3IdentityLbackward_lstm_132/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_132/while/Identity_3Ю
"backward_lstm_132/while/Identity_4Identity'backward_lstm_132/while/Select:output:0^backward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_132/while/Identity_4а
"backward_lstm_132/while/Identity_5Identity)backward_lstm_132/while/Select_1:output:0^backward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_132/while/Identity_5а
"backward_lstm_132/while/Identity_6Identity)backward_lstm_132/while/Select_2:output:0^backward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_132/while/Identity_6Л
backward_lstm_132/while/NoOpNoOp=^backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp<^backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp>^backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_132/while/NoOp"x
9backward_lstm_132_while_backward_lstm_132_strided_slice_1;backward_lstm_132_while_backward_lstm_132_strided_slice_1_0"M
 backward_lstm_132_while_identity)backward_lstm_132/while/Identity:output:0"Q
"backward_lstm_132_while_identity_1+backward_lstm_132/while/Identity_1:output:0"Q
"backward_lstm_132_while_identity_2+backward_lstm_132/while/Identity_2:output:0"Q
"backward_lstm_132_while_identity_3+backward_lstm_132/while/Identity_3:output:0"Q
"backward_lstm_132_while_identity_4+backward_lstm_132/while/Identity_4:output:0"Q
"backward_lstm_132_while_identity_5+backward_lstm_132/while/Identity_5:output:0"Q
"backward_lstm_132_while_identity_6+backward_lstm_132/while/Identity_6:output:0"n
4backward_lstm_132_while_less_backward_lstm_132_sub_16backward_lstm_132_while_less_backward_lstm_132_sub_1_0"
Ebackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resourceGbackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0"
Fbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resourceHbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0"
Dbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resourceFbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0"№
ubackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2|
<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp2z
;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp2~
=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
аf
ф
%backward_lstm_132_while_body_16717589@
<backward_lstm_132_while_backward_lstm_132_while_loop_counterF
Bbackward_lstm_132_while_backward_lstm_132_while_maximum_iterations'
#backward_lstm_132_while_placeholder)
%backward_lstm_132_while_placeholder_1)
%backward_lstm_132_while_placeholder_2)
%backward_lstm_132_while_placeholder_3)
%backward_lstm_132_while_placeholder_4?
;backward_lstm_132_while_backward_lstm_132_strided_slice_1_0{
wbackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_132_while_less_backward_lstm_132_sub_1_0Y
Fbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0:	Ш[
Hbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0:	2ШV
Gbackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0:	Ш$
 backward_lstm_132_while_identity&
"backward_lstm_132_while_identity_1&
"backward_lstm_132_while_identity_2&
"backward_lstm_132_while_identity_3&
"backward_lstm_132_while_identity_4&
"backward_lstm_132_while_identity_5&
"backward_lstm_132_while_identity_6=
9backward_lstm_132_while_backward_lstm_132_strided_slice_1y
ubackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_132_while_less_backward_lstm_132_sub_1W
Dbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource:	ШY
Fbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource:	2ШT
Ebackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource:	ШЂ<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOpЂ;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpЂ=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOpч
Ibackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2K
Ibackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeП
;backward_lstm_132/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_132_while_placeholderRbackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02=
;backward_lstm_132/while/TensorArrayV2Read/TensorListGetItemЯ
backward_lstm_132/while/LessLess6backward_lstm_132_while_less_backward_lstm_132_sub_1_0#backward_lstm_132_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_132/while/Less
;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpReadVariableOpFbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02=
;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpЂ
,backward_lstm_132/while/lstm_cell_398/MatMulMatMulBbackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem:item:0Cbackward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_132/while/lstm_cell_398/MatMul
=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOpHbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02?
=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp
.backward_lstm_132/while/lstm_cell_398/MatMul_1MatMul%backward_lstm_132_while_placeholder_3Ebackward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ20
.backward_lstm_132/while/lstm_cell_398/MatMul_1
)backward_lstm_132/while/lstm_cell_398/addAddV26backward_lstm_132/while/lstm_cell_398/MatMul:product:08backward_lstm_132/while/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_132/while/lstm_cell_398/add
<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOpGbackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02>
<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp
-backward_lstm_132/while/lstm_cell_398/BiasAddBiasAdd-backward_lstm_132/while/lstm_cell_398/add:z:0Dbackward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_132/while/lstm_cell_398/BiasAddА
5backward_lstm_132/while/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5backward_lstm_132/while/lstm_cell_398/split/split_dimз
+backward_lstm_132/while/lstm_cell_398/splitSplit>backward_lstm_132/while/lstm_cell_398/split/split_dim:output:06backward_lstm_132/while/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2-
+backward_lstm_132/while/lstm_cell_398/splitб
-backward_lstm_132/while/lstm_cell_398/SigmoidSigmoid4backward_lstm_132/while/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22/
-backward_lstm_132/while/lstm_cell_398/Sigmoidе
/backward_lstm_132/while/lstm_cell_398/Sigmoid_1Sigmoid4backward_lstm_132/while/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_132/while/lstm_cell_398/Sigmoid_1ы
)backward_lstm_132/while/lstm_cell_398/mulMul3backward_lstm_132/while/lstm_cell_398/Sigmoid_1:y:0%backward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_132/while/lstm_cell_398/mulШ
*backward_lstm_132/while/lstm_cell_398/ReluRelu4backward_lstm_132/while/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_132/while/lstm_cell_398/Relu
+backward_lstm_132/while/lstm_cell_398/mul_1Mul1backward_lstm_132/while/lstm_cell_398/Sigmoid:y:08backward_lstm_132/while/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_132/while/lstm_cell_398/mul_1ѕ
+backward_lstm_132/while/lstm_cell_398/add_1AddV2-backward_lstm_132/while/lstm_cell_398/mul:z:0/backward_lstm_132/while/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_132/while/lstm_cell_398/add_1е
/backward_lstm_132/while/lstm_cell_398/Sigmoid_2Sigmoid4backward_lstm_132/while/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_132/while/lstm_cell_398/Sigmoid_2Ч
,backward_lstm_132/while/lstm_cell_398/Relu_1Relu/backward_lstm_132/while/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_132/while/lstm_cell_398/Relu_1
+backward_lstm_132/while/lstm_cell_398/mul_2Mul3backward_lstm_132/while/lstm_cell_398/Sigmoid_2:y:0:backward_lstm_132/while/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_132/while/lstm_cell_398/mul_2і
backward_lstm_132/while/SelectSelect backward_lstm_132/while/Less:z:0/backward_lstm_132/while/lstm_cell_398/mul_2:z:0%backward_lstm_132_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_132/while/Selectњ
 backward_lstm_132/while/Select_1Select backward_lstm_132/while/Less:z:0/backward_lstm_132/while/lstm_cell_398/mul_2:z:0%backward_lstm_132_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_132/while/Select_1њ
 backward_lstm_132/while/Select_2Select backward_lstm_132/while/Less:z:0/backward_lstm_132/while/lstm_cell_398/add_1:z:0%backward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_132/while/Select_2Г
<backward_lstm_132/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_132_while_placeholder_1#backward_lstm_132_while_placeholder'backward_lstm_132/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_132/while/TensorArrayV2Write/TensorListSetItem
backward_lstm_132/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_132/while/add/yБ
backward_lstm_132/while/addAddV2#backward_lstm_132_while_placeholder&backward_lstm_132/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/while/add
backward_lstm_132/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_132/while/add_1/yа
backward_lstm_132/while/add_1AddV2<backward_lstm_132_while_backward_lstm_132_while_loop_counter(backward_lstm_132/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/while/add_1Г
 backward_lstm_132/while/IdentityIdentity!backward_lstm_132/while/add_1:z:0^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_132/while/Identityи
"backward_lstm_132/while/Identity_1IdentityBbackward_lstm_132_while_backward_lstm_132_while_maximum_iterations^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_132/while/Identity_1Е
"backward_lstm_132/while/Identity_2Identitybackward_lstm_132/while/add:z:0^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_132/while/Identity_2т
"backward_lstm_132/while/Identity_3IdentityLbackward_lstm_132/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_132/while/Identity_3Ю
"backward_lstm_132/while/Identity_4Identity'backward_lstm_132/while/Select:output:0^backward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_132/while/Identity_4а
"backward_lstm_132/while/Identity_5Identity)backward_lstm_132/while/Select_1:output:0^backward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_132/while/Identity_5а
"backward_lstm_132/while/Identity_6Identity)backward_lstm_132/while/Select_2:output:0^backward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_132/while/Identity_6Л
backward_lstm_132/while/NoOpNoOp=^backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp<^backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp>^backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_132/while/NoOp"x
9backward_lstm_132_while_backward_lstm_132_strided_slice_1;backward_lstm_132_while_backward_lstm_132_strided_slice_1_0"M
 backward_lstm_132_while_identity)backward_lstm_132/while/Identity:output:0"Q
"backward_lstm_132_while_identity_1+backward_lstm_132/while/Identity_1:output:0"Q
"backward_lstm_132_while_identity_2+backward_lstm_132/while/Identity_2:output:0"Q
"backward_lstm_132_while_identity_3+backward_lstm_132/while/Identity_3:output:0"Q
"backward_lstm_132_while_identity_4+backward_lstm_132/while/Identity_4:output:0"Q
"backward_lstm_132_while_identity_5+backward_lstm_132/while/Identity_5:output:0"Q
"backward_lstm_132_while_identity_6+backward_lstm_132/while/Identity_6:output:0"n
4backward_lstm_132_while_less_backward_lstm_132_sub_16backward_lstm_132_while_less_backward_lstm_132_sub_1_0"
Ebackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resourceGbackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0"
Fbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resourceHbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0"
Dbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resourceFbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0"№
ubackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2|
<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp2z
;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp2~
=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_body_16718978
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_398_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_398_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_398_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_398_matmul_readvariableop_resource:	ШG
4while_lstm_cell_398_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_398_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_398/BiasAdd/ReadVariableOpЂ)while/lstm_cell_398/MatMul/ReadVariableOpЂ+while/lstm_cell_398/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_398/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_398_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_398/MatMul/ReadVariableOpк
while/lstm_cell_398/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/MatMulв
+while/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_398_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_398/MatMul_1/ReadVariableOpУ
while/lstm_cell_398/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/MatMul_1М
while/lstm_cell_398/addAddV2$while/lstm_cell_398/MatMul:product:0&while/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/addЫ
*while/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_398_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_398/BiasAdd/ReadVariableOpЩ
while/lstm_cell_398/BiasAddBiasAddwhile/lstm_cell_398/add:z:02while/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/BiasAdd
#while/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_398/split/split_dim
while/lstm_cell_398/splitSplit,while/lstm_cell_398/split/split_dim:output:0$while/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_398/split
while/lstm_cell_398/SigmoidSigmoid"while/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Sigmoid
while/lstm_cell_398/Sigmoid_1Sigmoid"while/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Sigmoid_1Ѓ
while/lstm_cell_398/mulMul!while/lstm_cell_398/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/mul
while/lstm_cell_398/ReluRelu"while/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/ReluИ
while/lstm_cell_398/mul_1Mulwhile/lstm_cell_398/Sigmoid:y:0&while/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/mul_1­
while/lstm_cell_398/add_1AddV2while/lstm_cell_398/mul:z:0while/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/add_1
while/lstm_cell_398/Sigmoid_2Sigmoid"while/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Sigmoid_2
while/lstm_cell_398/Relu_1Reluwhile/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Relu_1М
while/lstm_cell_398/mul_2Mul!while/lstm_cell_398/Sigmoid_2:y:0(while/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_398/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_398/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_398/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_398/BiasAdd/ReadVariableOp*^while/lstm_cell_398/MatMul/ReadVariableOp,^while/lstm_cell_398/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_398_biasadd_readvariableop_resource5while_lstm_cell_398_biasadd_readvariableop_resource_0"n
4while_lstm_cell_398_matmul_1_readvariableop_resource6while_lstm_cell_398_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_398_matmul_readvariableop_resource4while_lstm_cell_398_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_398/BiasAdd/ReadVariableOp*while/lstm_cell_398/BiasAdd/ReadVariableOp2V
)while/lstm_cell_398/MatMul/ReadVariableOp)while/lstm_cell_398/MatMul/ReadVariableOp2Z
+while/lstm_cell_398/MatMul_1/ReadVariableOp+while/lstm_cell_398/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
$forward_lstm_132_while_body_16715758>
:forward_lstm_132_while_forward_lstm_132_while_loop_counterD
@forward_lstm_132_while_forward_lstm_132_while_maximum_iterations&
"forward_lstm_132_while_placeholder(
$forward_lstm_132_while_placeholder_1(
$forward_lstm_132_while_placeholder_2(
$forward_lstm_132_while_placeholder_3(
$forward_lstm_132_while_placeholder_4=
9forward_lstm_132_while_forward_lstm_132_strided_slice_1_0y
uforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_132_while_greater_forward_lstm_132_cast_0X
Eforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0:	ШZ
Gforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0:	2ШU
Fforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0:	Ш#
forward_lstm_132_while_identity%
!forward_lstm_132_while_identity_1%
!forward_lstm_132_while_identity_2%
!forward_lstm_132_while_identity_3%
!forward_lstm_132_while_identity_4%
!forward_lstm_132_while_identity_5%
!forward_lstm_132_while_identity_6;
7forward_lstm_132_while_forward_lstm_132_strided_slice_1w
sforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_132_while_greater_forward_lstm_132_castV
Cforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource:	ШX
Eforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource:	2ШS
Dforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource:	ШЂ;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOpЂ:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOpЂ<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOpх
Hforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2J
Hforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
:forward_lstm_132/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_132_while_placeholderQforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02<
:forward_lstm_132/while/TensorArrayV2Read/TensorListGetItemе
forward_lstm_132/while/GreaterGreater6forward_lstm_132_while_greater_forward_lstm_132_cast_0"forward_lstm_132_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2 
forward_lstm_132/while/Greaterџ
:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOpReadVariableOpEforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp
+forward_lstm_132/while/lstm_cell_397/MatMulMatMulAforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem:item:0Bforward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_132/while/lstm_cell_397/MatMul
<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOpGforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp
-forward_lstm_132/while/lstm_cell_397/MatMul_1MatMul$forward_lstm_132_while_placeholder_3Dforward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-forward_lstm_132/while/lstm_cell_397/MatMul_1
(forward_lstm_132/while/lstm_cell_397/addAddV25forward_lstm_132/while/lstm_cell_397/MatMul:product:07forward_lstm_132/while/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_132/while/lstm_cell_397/addў
;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOpFforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp
,forward_lstm_132/while/lstm_cell_397/BiasAddBiasAdd,forward_lstm_132/while/lstm_cell_397/add:z:0Cforward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_132/while/lstm_cell_397/BiasAddЎ
4forward_lstm_132/while/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4forward_lstm_132/while/lstm_cell_397/split/split_dimг
*forward_lstm_132/while/lstm_cell_397/splitSplit=forward_lstm_132/while/lstm_cell_397/split/split_dim:output:05forward_lstm_132/while/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*forward_lstm_132/while/lstm_cell_397/splitЮ
,forward_lstm_132/while/lstm_cell_397/SigmoidSigmoid3forward_lstm_132/while/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,forward_lstm_132/while/lstm_cell_397/Sigmoidв
.forward_lstm_132/while/lstm_cell_397/Sigmoid_1Sigmoid3forward_lstm_132/while/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_132/while/lstm_cell_397/Sigmoid_1ч
(forward_lstm_132/while/lstm_cell_397/mulMul2forward_lstm_132/while/lstm_cell_397/Sigmoid_1:y:0$forward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_132/while/lstm_cell_397/mulХ
)forward_lstm_132/while/lstm_cell_397/ReluRelu3forward_lstm_132/while/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_132/while/lstm_cell_397/Reluќ
*forward_lstm_132/while/lstm_cell_397/mul_1Mul0forward_lstm_132/while/lstm_cell_397/Sigmoid:y:07forward_lstm_132/while/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_132/while/lstm_cell_397/mul_1ё
*forward_lstm_132/while/lstm_cell_397/add_1AddV2,forward_lstm_132/while/lstm_cell_397/mul:z:0.forward_lstm_132/while/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_132/while/lstm_cell_397/add_1в
.forward_lstm_132/while/lstm_cell_397/Sigmoid_2Sigmoid3forward_lstm_132/while/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_132/while/lstm_cell_397/Sigmoid_2Ф
+forward_lstm_132/while/lstm_cell_397/Relu_1Relu.forward_lstm_132/while/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_132/while/lstm_cell_397/Relu_1
*forward_lstm_132/while/lstm_cell_397/mul_2Mul2forward_lstm_132/while/lstm_cell_397/Sigmoid_2:y:09forward_lstm_132/while/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_132/while/lstm_cell_397/mul_2є
forward_lstm_132/while/SelectSelect"forward_lstm_132/while/Greater:z:0.forward_lstm_132/while/lstm_cell_397/mul_2:z:0$forward_lstm_132_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/while/Selectј
forward_lstm_132/while/Select_1Select"forward_lstm_132/while/Greater:z:0.forward_lstm_132/while/lstm_cell_397/mul_2:z:0$forward_lstm_132_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_132/while/Select_1ј
forward_lstm_132/while/Select_2Select"forward_lstm_132/while/Greater:z:0.forward_lstm_132/while/lstm_cell_397/add_1:z:0$forward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_132/while/Select_2Ў
;forward_lstm_132/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_132_while_placeholder_1"forward_lstm_132_while_placeholder&forward_lstm_132/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_132/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_132/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_132/while/add/y­
forward_lstm_132/while/addAddV2"forward_lstm_132_while_placeholder%forward_lstm_132/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/while/add
forward_lstm_132/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_132/while/add_1/yЫ
forward_lstm_132/while/add_1AddV2:forward_lstm_132_while_forward_lstm_132_while_loop_counter'forward_lstm_132/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/while/add_1Џ
forward_lstm_132/while/IdentityIdentity forward_lstm_132/while/add_1:z:0^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_132/while/Identityг
!forward_lstm_132/while/Identity_1Identity@forward_lstm_132_while_forward_lstm_132_while_maximum_iterations^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_132/while/Identity_1Б
!forward_lstm_132/while/Identity_2Identityforward_lstm_132/while/add:z:0^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_132/while/Identity_2о
!forward_lstm_132/while/Identity_3IdentityKforward_lstm_132/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_132/while/Identity_3Ъ
!forward_lstm_132/while/Identity_4Identity&forward_lstm_132/while/Select:output:0^forward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_132/while/Identity_4Ь
!forward_lstm_132/while/Identity_5Identity(forward_lstm_132/while/Select_1:output:0^forward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_132/while/Identity_5Ь
!forward_lstm_132/while/Identity_6Identity(forward_lstm_132/while/Select_2:output:0^forward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_132/while/Identity_6Ж
forward_lstm_132/while/NoOpNoOp<^forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp;^forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp=^forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_132/while/NoOp"t
7forward_lstm_132_while_forward_lstm_132_strided_slice_19forward_lstm_132_while_forward_lstm_132_strided_slice_1_0"n
4forward_lstm_132_while_greater_forward_lstm_132_cast6forward_lstm_132_while_greater_forward_lstm_132_cast_0"K
forward_lstm_132_while_identity(forward_lstm_132/while/Identity:output:0"O
!forward_lstm_132_while_identity_1*forward_lstm_132/while/Identity_1:output:0"O
!forward_lstm_132_while_identity_2*forward_lstm_132/while/Identity_2:output:0"O
!forward_lstm_132_while_identity_3*forward_lstm_132/while/Identity_3:output:0"O
!forward_lstm_132_while_identity_4*forward_lstm_132/while/Identity_4:output:0"O
!forward_lstm_132_while_identity_5*forward_lstm_132/while/Identity_5:output:0"O
!forward_lstm_132_while_identity_6*forward_lstm_132/while/Identity_6:output:0"
Dforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resourceFforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0"
Eforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resourceGforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0"
Cforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resourceEforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0"ь
sforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensoruforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2z
;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp2x
:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp2|
<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
&
ј
while_body_16713687
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_397_16713711_0:	Ш1
while_lstm_cell_397_16713713_0:	2Ш-
while_lstm_cell_397_16713715_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_397_16713711:	Ш/
while_lstm_cell_397_16713713:	2Ш+
while_lstm_cell_397_16713715:	ШЂ+while/lstm_cell_397/StatefulPartitionedCallУ
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
+while/lstm_cell_397/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_397_16713711_0while_lstm_cell_397_16713713_0while_lstm_cell_397_16713715_0*
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
K__inference_lstm_cell_397_layer_call_and_return_conditional_losses_167136732-
+while/lstm_cell_397/StatefulPartitionedCallј
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_397/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_397/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѕ
while/Identity_5Identity4while/lstm_cell_397/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp,^while/lstm_cell_397/StatefulPartitionedCall*"
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
while_lstm_cell_397_16713711while_lstm_cell_397_16713711_0">
while_lstm_cell_397_16713713while_lstm_cell_397_16713713_0">
while_lstm_cell_397_16713715while_lstm_cell_397_16713715_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2Z
+while/lstm_cell_397/StatefulPartitionedCall+while/lstm_cell_397/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
%backward_lstm_132_while_body_16716940@
<backward_lstm_132_while_backward_lstm_132_while_loop_counterF
Bbackward_lstm_132_while_backward_lstm_132_while_maximum_iterations'
#backward_lstm_132_while_placeholder)
%backward_lstm_132_while_placeholder_1)
%backward_lstm_132_while_placeholder_2)
%backward_lstm_132_while_placeholder_3?
;backward_lstm_132_while_backward_lstm_132_strided_slice_1_0{
wbackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0Y
Fbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0:	Ш[
Hbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0:	2ШV
Gbackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0:	Ш$
 backward_lstm_132_while_identity&
"backward_lstm_132_while_identity_1&
"backward_lstm_132_while_identity_2&
"backward_lstm_132_while_identity_3&
"backward_lstm_132_while_identity_4&
"backward_lstm_132_while_identity_5=
9backward_lstm_132_while_backward_lstm_132_strided_slice_1y
ubackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensorW
Dbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource:	ШY
Fbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource:	2ШT
Ebackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource:	ШЂ<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOpЂ;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpЂ=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOpч
Ibackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2K
Ibackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeШ
;backward_lstm_132/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_132_while_placeholderRbackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02=
;backward_lstm_132/while/TensorArrayV2Read/TensorListGetItem
;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpReadVariableOpFbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02=
;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpЂ
,backward_lstm_132/while/lstm_cell_398/MatMulMatMulBbackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem:item:0Cbackward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_132/while/lstm_cell_398/MatMul
=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOpHbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02?
=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp
.backward_lstm_132/while/lstm_cell_398/MatMul_1MatMul%backward_lstm_132_while_placeholder_2Ebackward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ20
.backward_lstm_132/while/lstm_cell_398/MatMul_1
)backward_lstm_132/while/lstm_cell_398/addAddV26backward_lstm_132/while/lstm_cell_398/MatMul:product:08backward_lstm_132/while/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_132/while/lstm_cell_398/add
<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOpGbackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02>
<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp
-backward_lstm_132/while/lstm_cell_398/BiasAddBiasAdd-backward_lstm_132/while/lstm_cell_398/add:z:0Dbackward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_132/while/lstm_cell_398/BiasAddА
5backward_lstm_132/while/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5backward_lstm_132/while/lstm_cell_398/split/split_dimз
+backward_lstm_132/while/lstm_cell_398/splitSplit>backward_lstm_132/while/lstm_cell_398/split/split_dim:output:06backward_lstm_132/while/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2-
+backward_lstm_132/while/lstm_cell_398/splitб
-backward_lstm_132/while/lstm_cell_398/SigmoidSigmoid4backward_lstm_132/while/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22/
-backward_lstm_132/while/lstm_cell_398/Sigmoidе
/backward_lstm_132/while/lstm_cell_398/Sigmoid_1Sigmoid4backward_lstm_132/while/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_132/while/lstm_cell_398/Sigmoid_1ы
)backward_lstm_132/while/lstm_cell_398/mulMul3backward_lstm_132/while/lstm_cell_398/Sigmoid_1:y:0%backward_lstm_132_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_132/while/lstm_cell_398/mulШ
*backward_lstm_132/while/lstm_cell_398/ReluRelu4backward_lstm_132/while/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_132/while/lstm_cell_398/Relu
+backward_lstm_132/while/lstm_cell_398/mul_1Mul1backward_lstm_132/while/lstm_cell_398/Sigmoid:y:08backward_lstm_132/while/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_132/while/lstm_cell_398/mul_1ѕ
+backward_lstm_132/while/lstm_cell_398/add_1AddV2-backward_lstm_132/while/lstm_cell_398/mul:z:0/backward_lstm_132/while/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_132/while/lstm_cell_398/add_1е
/backward_lstm_132/while/lstm_cell_398/Sigmoid_2Sigmoid4backward_lstm_132/while/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_132/while/lstm_cell_398/Sigmoid_2Ч
,backward_lstm_132/while/lstm_cell_398/Relu_1Relu/backward_lstm_132/while/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_132/while/lstm_cell_398/Relu_1
+backward_lstm_132/while/lstm_cell_398/mul_2Mul3backward_lstm_132/while/lstm_cell_398/Sigmoid_2:y:0:backward_lstm_132/while/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_132/while/lstm_cell_398/mul_2Л
<backward_lstm_132/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_132_while_placeholder_1#backward_lstm_132_while_placeholder/backward_lstm_132/while/lstm_cell_398/mul_2:z:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_132/while/TensorArrayV2Write/TensorListSetItem
backward_lstm_132/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_132/while/add/yБ
backward_lstm_132/while/addAddV2#backward_lstm_132_while_placeholder&backward_lstm_132/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/while/add
backward_lstm_132/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_132/while/add_1/yа
backward_lstm_132/while/add_1AddV2<backward_lstm_132_while_backward_lstm_132_while_loop_counter(backward_lstm_132/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/while/add_1Г
 backward_lstm_132/while/IdentityIdentity!backward_lstm_132/while/add_1:z:0^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_132/while/Identityи
"backward_lstm_132/while/Identity_1IdentityBbackward_lstm_132_while_backward_lstm_132_while_maximum_iterations^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_132/while/Identity_1Е
"backward_lstm_132/while/Identity_2Identitybackward_lstm_132/while/add:z:0^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_132/while/Identity_2т
"backward_lstm_132/while/Identity_3IdentityLbackward_lstm_132/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_132/while/Identity_3ж
"backward_lstm_132/while/Identity_4Identity/backward_lstm_132/while/lstm_cell_398/mul_2:z:0^backward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_132/while/Identity_4ж
"backward_lstm_132/while/Identity_5Identity/backward_lstm_132/while/lstm_cell_398/add_1:z:0^backward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_132/while/Identity_5Л
backward_lstm_132/while/NoOpNoOp=^backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp<^backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp>^backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_132/while/NoOp"x
9backward_lstm_132_while_backward_lstm_132_strided_slice_1;backward_lstm_132_while_backward_lstm_132_strided_slice_1_0"M
 backward_lstm_132_while_identity)backward_lstm_132/while/Identity:output:0"Q
"backward_lstm_132_while_identity_1+backward_lstm_132/while/Identity_1:output:0"Q
"backward_lstm_132_while_identity_2+backward_lstm_132/while/Identity_2:output:0"Q
"backward_lstm_132_while_identity_3+backward_lstm_132/while/Identity_3:output:0"Q
"backward_lstm_132_while_identity_4+backward_lstm_132/while/Identity_4:output:0"Q
"backward_lstm_132_while_identity_5+backward_lstm_132/while/Identity_5:output:0"
Ebackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resourceGbackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0"
Fbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resourceHbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0"
Dbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resourceFbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0"№
ubackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2|
<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp2z
;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp2~
=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_body_16715099
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_398_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_398_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_398_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_398_matmul_readvariableop_resource:	ШG
4while_lstm_cell_398_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_398_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_398/BiasAdd/ReadVariableOpЂ)while/lstm_cell_398/MatMul/ReadVariableOpЂ+while/lstm_cell_398/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_398/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_398_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_398/MatMul/ReadVariableOpк
while/lstm_cell_398/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/MatMulв
+while/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_398_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_398/MatMul_1/ReadVariableOpУ
while/lstm_cell_398/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/MatMul_1М
while/lstm_cell_398/addAddV2$while/lstm_cell_398/MatMul:product:0&while/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/addЫ
*while/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_398_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_398/BiasAdd/ReadVariableOpЩ
while/lstm_cell_398/BiasAddBiasAddwhile/lstm_cell_398/add:z:02while/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/BiasAdd
#while/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_398/split/split_dim
while/lstm_cell_398/splitSplit,while/lstm_cell_398/split/split_dim:output:0$while/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_398/split
while/lstm_cell_398/SigmoidSigmoid"while/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Sigmoid
while/lstm_cell_398/Sigmoid_1Sigmoid"while/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Sigmoid_1Ѓ
while/lstm_cell_398/mulMul!while/lstm_cell_398/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/mul
while/lstm_cell_398/ReluRelu"while/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/ReluИ
while/lstm_cell_398/mul_1Mulwhile/lstm_cell_398/Sigmoid:y:0&while/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/mul_1­
while/lstm_cell_398/add_1AddV2while/lstm_cell_398/mul:z:0while/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/add_1
while/lstm_cell_398/Sigmoid_2Sigmoid"while/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Sigmoid_2
while/lstm_cell_398/Relu_1Reluwhile/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Relu_1М
while/lstm_cell_398/mul_2Mul!while/lstm_cell_398/Sigmoid_2:y:0(while/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_398/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_398/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_398/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_398/BiasAdd/ReadVariableOp*^while/lstm_cell_398/MatMul/ReadVariableOp,^while/lstm_cell_398/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_398_biasadd_readvariableop_resource5while_lstm_cell_398_biasadd_readvariableop_resource_0"n
4while_lstm_cell_398_matmul_1_readvariableop_resource6while_lstm_cell_398_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_398_matmul_readvariableop_resource4while_lstm_cell_398_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_398/BiasAdd/ReadVariableOp*while/lstm_cell_398/BiasAdd/ReadVariableOp2V
)while/lstm_cell_398/MatMul/ReadVariableOp)while/lstm_cell_398/MatMul/ReadVariableOp2Z
+while/lstm_cell_398/MatMul_1/ReadVariableOp+while/lstm_cell_398/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
§
Е
%backward_lstm_132_while_cond_16717241@
<backward_lstm_132_while_backward_lstm_132_while_loop_counterF
Bbackward_lstm_132_while_backward_lstm_132_while_maximum_iterations'
#backward_lstm_132_while_placeholder)
%backward_lstm_132_while_placeholder_1)
%backward_lstm_132_while_placeholder_2)
%backward_lstm_132_while_placeholder_3B
>backward_lstm_132_while_less_backward_lstm_132_strided_slice_1Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16717241___redundant_placeholder0Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16717241___redundant_placeholder1Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16717241___redundant_placeholder2Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16717241___redundant_placeholder3$
 backward_lstm_132_while_identity
Ъ
backward_lstm_132/while/LessLess#backward_lstm_132_while_placeholder>backward_lstm_132_while_less_backward_lstm_132_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_132/while/Less
 backward_lstm_132/while/IdentityIdentity backward_lstm_132/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_132/while/Identity"M
 backward_lstm_132_while_identity)backward_lstm_132/while/Identity:output:0*(
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
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_16717686

inputs
inputs_1	P
=forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource:	ШR
?forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource:	2ШM
>forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource:	ШQ
>backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource:	ШS
@backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource:	2ШN
?backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource:	Ш
identityЂ6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpЂ5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpЂ7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpЂbackward_lstm_132/whileЂ5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpЂ4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpЂ6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOpЂforward_lstm_132/while
%forward_lstm_132/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_132/RaggedToTensor/zeros
%forward_lstm_132/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2'
%forward_lstm_132/RaggedToTensor/Const
4forward_lstm_132/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_132/RaggedToTensor/Const:output:0inputs.forward_lstm_132/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_132/RaggedToTensor/RaggedTensorToTensorФ
;forward_lstm_132/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_132/RaggedNestedRowLengths/strided_slice/stackШ
=forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1Ш
=forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2Љ
5forward_lstm_132/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_132/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask27
5forward_lstm_132/RaggedNestedRowLengths/strided_sliceШ
=forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stackе
?forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2A
?forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1Ь
?forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2Е
7forward_lstm_132/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask29
7forward_lstm_132/RaggedNestedRowLengths/strided_slice_1
+forward_lstm_132/RaggedNestedRowLengths/subSub>forward_lstm_132/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_132/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2-
+forward_lstm_132/RaggedNestedRowLengths/subЄ
forward_lstm_132/CastCast/forward_lstm_132/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_132/Cast
forward_lstm_132/ShapeShape=forward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_132/Shape
$forward_lstm_132/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_132/strided_slice/stack
&forward_lstm_132/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_132/strided_slice/stack_1
&forward_lstm_132/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_132/strided_slice/stack_2Ш
forward_lstm_132/strided_sliceStridedSliceforward_lstm_132/Shape:output:0-forward_lstm_132/strided_slice/stack:output:0/forward_lstm_132/strided_slice/stack_1:output:0/forward_lstm_132/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_132/strided_slice~
forward_lstm_132/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_132/zeros/mul/yА
forward_lstm_132/zeros/mulMul'forward_lstm_132/strided_slice:output:0%forward_lstm_132/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros/mul
forward_lstm_132/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_132/zeros/Less/yЋ
forward_lstm_132/zeros/LessLessforward_lstm_132/zeros/mul:z:0&forward_lstm_132/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros/Less
forward_lstm_132/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_132/zeros/packed/1Ч
forward_lstm_132/zeros/packedPack'forward_lstm_132/strided_slice:output:0(forward_lstm_132/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_132/zeros/packed
forward_lstm_132/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_132/zeros/ConstЙ
forward_lstm_132/zerosFill&forward_lstm_132/zeros/packed:output:0%forward_lstm_132/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/zeros
forward_lstm_132/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_132/zeros_1/mul/yЖ
forward_lstm_132/zeros_1/mulMul'forward_lstm_132/strided_slice:output:0'forward_lstm_132/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros_1/mul
forward_lstm_132/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
forward_lstm_132/zeros_1/Less/yГ
forward_lstm_132/zeros_1/LessLess forward_lstm_132/zeros_1/mul:z:0(forward_lstm_132/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros_1/Less
!forward_lstm_132/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_132/zeros_1/packed/1Э
forward_lstm_132/zeros_1/packedPack'forward_lstm_132/strided_slice:output:0*forward_lstm_132/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_132/zeros_1/packed
forward_lstm_132/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_132/zeros_1/ConstС
forward_lstm_132/zeros_1Fill(forward_lstm_132/zeros_1/packed:output:0'forward_lstm_132/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/zeros_1
forward_lstm_132/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_132/transpose/permэ
forward_lstm_132/transpose	Transpose=forward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_132/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_132/transpose
forward_lstm_132/Shape_1Shapeforward_lstm_132/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_132/Shape_1
&forward_lstm_132/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_132/strided_slice_1/stack
(forward_lstm_132/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_1/stack_1
(forward_lstm_132/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_1/stack_2д
 forward_lstm_132/strided_slice_1StridedSlice!forward_lstm_132/Shape_1:output:0/forward_lstm_132/strided_slice_1/stack:output:01forward_lstm_132/strided_slice_1/stack_1:output:01forward_lstm_132/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_132/strided_slice_1Ї
,forward_lstm_132/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,forward_lstm_132/TensorArrayV2/element_shapeі
forward_lstm_132/TensorArrayV2TensorListReserve5forward_lstm_132/TensorArrayV2/element_shape:output:0)forward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_132/TensorArrayV2с
Fforward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fforward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8forward_lstm_132/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_132/transpose:y:0Oforward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_132/TensorArrayUnstack/TensorListFromTensor
&forward_lstm_132/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_132/strided_slice_2/stack
(forward_lstm_132/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_2/stack_1
(forward_lstm_132/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_2/stack_2т
 forward_lstm_132/strided_slice_2StridedSliceforward_lstm_132/transpose:y:0/forward_lstm_132/strided_slice_2/stack:output:01forward_lstm_132/strided_slice_2/stack_1:output:01forward_lstm_132/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 forward_lstm_132/strided_slice_2ы
4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpReadVariableOp=forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpє
%forward_lstm_132/lstm_cell_397/MatMulMatMul)forward_lstm_132/strided_slice_2:output:0<forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_132/lstm_cell_397/MatMulё
6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOp?forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp№
'forward_lstm_132/lstm_cell_397/MatMul_1MatMulforward_lstm_132/zeros:output:0>forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_132/lstm_cell_397/MatMul_1ш
"forward_lstm_132/lstm_cell_397/addAddV2/forward_lstm_132/lstm_cell_397/MatMul:product:01forward_lstm_132/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_132/lstm_cell_397/addъ
5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOp>forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpѕ
&forward_lstm_132/lstm_cell_397/BiasAddBiasAdd&forward_lstm_132/lstm_cell_397/add:z:0=forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_132/lstm_cell_397/BiasAddЂ
.forward_lstm_132/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.forward_lstm_132/lstm_cell_397/split/split_dimЛ
$forward_lstm_132/lstm_cell_397/splitSplit7forward_lstm_132/lstm_cell_397/split/split_dim:output:0/forward_lstm_132/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$forward_lstm_132/lstm_cell_397/splitМ
&forward_lstm_132/lstm_cell_397/SigmoidSigmoid-forward_lstm_132/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_132/lstm_cell_397/SigmoidР
(forward_lstm_132/lstm_cell_397/Sigmoid_1Sigmoid-forward_lstm_132/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_132/lstm_cell_397/Sigmoid_1в
"forward_lstm_132/lstm_cell_397/mulMul,forward_lstm_132/lstm_cell_397/Sigmoid_1:y:0!forward_lstm_132/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_132/lstm_cell_397/mulГ
#forward_lstm_132/lstm_cell_397/ReluRelu-forward_lstm_132/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_132/lstm_cell_397/Reluф
$forward_lstm_132/lstm_cell_397/mul_1Mul*forward_lstm_132/lstm_cell_397/Sigmoid:y:01forward_lstm_132/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_132/lstm_cell_397/mul_1й
$forward_lstm_132/lstm_cell_397/add_1AddV2&forward_lstm_132/lstm_cell_397/mul:z:0(forward_lstm_132/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_132/lstm_cell_397/add_1Р
(forward_lstm_132/lstm_cell_397/Sigmoid_2Sigmoid-forward_lstm_132/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_132/lstm_cell_397/Sigmoid_2В
%forward_lstm_132/lstm_cell_397/Relu_1Relu(forward_lstm_132/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_132/lstm_cell_397/Relu_1ш
$forward_lstm_132/lstm_cell_397/mul_2Mul,forward_lstm_132/lstm_cell_397/Sigmoid_2:y:03forward_lstm_132/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_132/lstm_cell_397/mul_2Б
.forward_lstm_132/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.forward_lstm_132/TensorArrayV2_1/element_shapeќ
 forward_lstm_132/TensorArrayV2_1TensorListReserve7forward_lstm_132/TensorArrayV2_1/element_shape:output:0)forward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_132/TensorArrayV2_1p
forward_lstm_132/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_132/timeЃ
forward_lstm_132/zeros_like	ZerosLike(forward_lstm_132/lstm_cell_397/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/zeros_likeЁ
)forward_lstm_132/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)forward_lstm_132/while/maximum_iterations
#forward_lstm_132/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_132/while/loop_counter	
forward_lstm_132/whileWhile,forward_lstm_132/while/loop_counter:output:02forward_lstm_132/while/maximum_iterations:output:0forward_lstm_132/time:output:0)forward_lstm_132/TensorArrayV2_1:handle:0forward_lstm_132/zeros_like:y:0forward_lstm_132/zeros:output:0!forward_lstm_132/zeros_1:output:0)forward_lstm_132/strided_slice_1:output:0Hforward_lstm_132/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_132/Cast:y:0=forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource?forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource>forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource*
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
$forward_lstm_132_while_body_16717410*0
cond(R&
$forward_lstm_132_while_cond_16717409*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_132/whileз
Aforward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Aforward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3forward_lstm_132/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_132/while:output:3Jforward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3forward_lstm_132/TensorArrayV2Stack/TensorListStackЃ
&forward_lstm_132/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&forward_lstm_132/strided_slice_3/stack
(forward_lstm_132/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_132/strided_slice_3/stack_1
(forward_lstm_132/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_3/stack_2
 forward_lstm_132/strided_slice_3StridedSlice<forward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_132/strided_slice_3/stack:output:01forward_lstm_132/strided_slice_3/stack_1:output:01forward_lstm_132/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 forward_lstm_132/strided_slice_3
!forward_lstm_132/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_132/transpose_1/permђ
forward_lstm_132/transpose_1	Transpose<forward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_132/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_132/transpose_1
forward_lstm_132/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_132/runtime
&backward_lstm_132/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_132/RaggedToTensor/zeros
&backward_lstm_132/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2(
&backward_lstm_132/RaggedToTensor/Const
5backward_lstm_132/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_132/RaggedToTensor/Const:output:0inputs/backward_lstm_132/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_132/RaggedToTensor/RaggedTensorToTensorЦ
<backward_lstm_132/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_132/RaggedNestedRowLengths/strided_slice/stackЪ
>backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1Ъ
>backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2Ў
6backward_lstm_132/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_132/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask28
6backward_lstm_132/RaggedNestedRowLengths/strided_sliceЪ
>backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stackз
@backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2B
@backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1Ю
@backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2К
8backward_lstm_132/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2:
8backward_lstm_132/RaggedNestedRowLengths/strided_slice_1
,backward_lstm_132/RaggedNestedRowLengths/subSub?backward_lstm_132/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_132/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2.
,backward_lstm_132/RaggedNestedRowLengths/subЇ
backward_lstm_132/CastCast0backward_lstm_132/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_132/Cast 
backward_lstm_132/ShapeShape>backward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_132/Shape
%backward_lstm_132/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_132/strided_slice/stack
'backward_lstm_132/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_132/strided_slice/stack_1
'backward_lstm_132/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_132/strided_slice/stack_2Ю
backward_lstm_132/strided_sliceStridedSlice backward_lstm_132/Shape:output:0.backward_lstm_132/strided_slice/stack:output:00backward_lstm_132/strided_slice/stack_1:output:00backward_lstm_132/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_132/strided_slice
backward_lstm_132/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_132/zeros/mul/yД
backward_lstm_132/zeros/mulMul(backward_lstm_132/strided_slice:output:0&backward_lstm_132/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/zeros/mul
backward_lstm_132/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_132/zeros/Less/yЏ
backward_lstm_132/zeros/LessLessbackward_lstm_132/zeros/mul:z:0'backward_lstm_132/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/zeros/Less
 backward_lstm_132/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_132/zeros/packed/1Ы
backward_lstm_132/zeros/packedPack(backward_lstm_132/strided_slice:output:0)backward_lstm_132/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_132/zeros/packed
backward_lstm_132/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_132/zeros/ConstН
backward_lstm_132/zerosFill'backward_lstm_132/zeros/packed:output:0&backward_lstm_132/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_132/zeros
backward_lstm_132/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_132/zeros_1/mul/yК
backward_lstm_132/zeros_1/mulMul(backward_lstm_132/strided_slice:output:0(backward_lstm_132/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/zeros_1/mul
 backward_lstm_132/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 backward_lstm_132/zeros_1/Less/yЗ
backward_lstm_132/zeros_1/LessLess!backward_lstm_132/zeros_1/mul:z:0)backward_lstm_132/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_132/zeros_1/Less
"backward_lstm_132/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_132/zeros_1/packed/1б
 backward_lstm_132/zeros_1/packedPack(backward_lstm_132/strided_slice:output:0+backward_lstm_132/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_132/zeros_1/packed
backward_lstm_132/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_132/zeros_1/ConstХ
backward_lstm_132/zeros_1Fill)backward_lstm_132/zeros_1/packed:output:0(backward_lstm_132/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_132/zeros_1
 backward_lstm_132/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_132/transpose/permё
backward_lstm_132/transpose	Transpose>backward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_132/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_132/transpose
backward_lstm_132/Shape_1Shapebackward_lstm_132/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_132/Shape_1
'backward_lstm_132/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_132/strided_slice_1/stack 
)backward_lstm_132/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_1/stack_1 
)backward_lstm_132/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_1/stack_2к
!backward_lstm_132/strided_slice_1StridedSlice"backward_lstm_132/Shape_1:output:00backward_lstm_132/strided_slice_1/stack:output:02backward_lstm_132/strided_slice_1/stack_1:output:02backward_lstm_132/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_132/strided_slice_1Љ
-backward_lstm_132/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2/
-backward_lstm_132/TensorArrayV2/element_shapeњ
backward_lstm_132/TensorArrayV2TensorListReserve6backward_lstm_132/TensorArrayV2/element_shape:output:0*backward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_132/TensorArrayV2
 backward_lstm_132/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_132/ReverseV2/axisв
backward_lstm_132/ReverseV2	ReverseV2backward_lstm_132/transpose:y:0)backward_lstm_132/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_132/ReverseV2у
Gbackward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gbackward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeХ
9backward_lstm_132/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_132/ReverseV2:output:0Pbackward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_132/TensorArrayUnstack/TensorListFromTensor
'backward_lstm_132/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_132/strided_slice_2/stack 
)backward_lstm_132/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_2/stack_1 
)backward_lstm_132/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_2/stack_2ш
!backward_lstm_132/strided_slice_2StridedSlicebackward_lstm_132/transpose:y:00backward_lstm_132/strided_slice_2/stack:output:02backward_lstm_132/strided_slice_2/stack_1:output:02backward_lstm_132/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2#
!backward_lstm_132/strided_slice_2ю
5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpReadVariableOp>backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype027
5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpј
&backward_lstm_132/lstm_cell_398/MatMulMatMul*backward_lstm_132/strided_slice_2:output:0=backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_132/lstm_cell_398/MatMulє
7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOp@backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype029
7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpє
(backward_lstm_132/lstm_cell_398/MatMul_1MatMul backward_lstm_132/zeros:output:0?backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_132/lstm_cell_398/MatMul_1ь
#backward_lstm_132/lstm_cell_398/addAddV20backward_lstm_132/lstm_cell_398/MatMul:product:02backward_lstm_132/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_132/lstm_cell_398/addэ
6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOp?backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype028
6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpљ
'backward_lstm_132/lstm_cell_398/BiasAddBiasAdd'backward_lstm_132/lstm_cell_398/add:z:0>backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_132/lstm_cell_398/BiasAddЄ
/backward_lstm_132/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/backward_lstm_132/lstm_cell_398/split/split_dimП
%backward_lstm_132/lstm_cell_398/splitSplit8backward_lstm_132/lstm_cell_398/split/split_dim:output:00backward_lstm_132/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2'
%backward_lstm_132/lstm_cell_398/splitП
'backward_lstm_132/lstm_cell_398/SigmoidSigmoid.backward_lstm_132/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_132/lstm_cell_398/SigmoidУ
)backward_lstm_132/lstm_cell_398/Sigmoid_1Sigmoid.backward_lstm_132/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_132/lstm_cell_398/Sigmoid_1ж
#backward_lstm_132/lstm_cell_398/mulMul-backward_lstm_132/lstm_cell_398/Sigmoid_1:y:0"backward_lstm_132/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_132/lstm_cell_398/mulЖ
$backward_lstm_132/lstm_cell_398/ReluRelu.backward_lstm_132/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_132/lstm_cell_398/Reluш
%backward_lstm_132/lstm_cell_398/mul_1Mul+backward_lstm_132/lstm_cell_398/Sigmoid:y:02backward_lstm_132/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_132/lstm_cell_398/mul_1н
%backward_lstm_132/lstm_cell_398/add_1AddV2'backward_lstm_132/lstm_cell_398/mul:z:0)backward_lstm_132/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_132/lstm_cell_398/add_1У
)backward_lstm_132/lstm_cell_398/Sigmoid_2Sigmoid.backward_lstm_132/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_132/lstm_cell_398/Sigmoid_2Е
&backward_lstm_132/lstm_cell_398/Relu_1Relu)backward_lstm_132/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_132/lstm_cell_398/Relu_1ь
%backward_lstm_132/lstm_cell_398/mul_2Mul-backward_lstm_132/lstm_cell_398/Sigmoid_2:y:04backward_lstm_132/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_132/lstm_cell_398/mul_2Г
/backward_lstm_132/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   21
/backward_lstm_132/TensorArrayV2_1/element_shape
!backward_lstm_132/TensorArrayV2_1TensorListReserve8backward_lstm_132/TensorArrayV2_1/element_shape:output:0*backward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_132/TensorArrayV2_1r
backward_lstm_132/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_132/time
'backward_lstm_132/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_132/Max/reduction_indicesЄ
backward_lstm_132/MaxMaxbackward_lstm_132/Cast:y:00backward_lstm_132/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/Maxt
backward_lstm_132/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_132/sub/y
backward_lstm_132/subSubbackward_lstm_132/Max:output:0 backward_lstm_132/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/sub
backward_lstm_132/Sub_1Subbackward_lstm_132/sub:z:0backward_lstm_132/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_132/Sub_1І
backward_lstm_132/zeros_like	ZerosLike)backward_lstm_132/lstm_cell_398/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_132/zeros_likeЃ
*backward_lstm_132/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*backward_lstm_132/while/maximum_iterations
$backward_lstm_132/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_132/while/loop_counterЅ	
backward_lstm_132/whileWhile-backward_lstm_132/while/loop_counter:output:03backward_lstm_132/while/maximum_iterations:output:0backward_lstm_132/time:output:0*backward_lstm_132/TensorArrayV2_1:handle:0 backward_lstm_132/zeros_like:y:0 backward_lstm_132/zeros:output:0"backward_lstm_132/zeros_1:output:0*backward_lstm_132/strided_slice_1:output:0Ibackward_lstm_132/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_132/Sub_1:z:0>backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource@backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource?backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource*
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
%backward_lstm_132_while_body_16717589*1
cond)R'
%backward_lstm_132_while_cond_16717588*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_132/whileй
Bbackward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2D
Bbackward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeЙ
4backward_lstm_132/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_132/while:output:3Kbackward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype026
4backward_lstm_132/TensorArrayV2Stack/TensorListStackЅ
'backward_lstm_132/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2)
'backward_lstm_132/strided_slice_3/stack 
)backward_lstm_132/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_132/strided_slice_3/stack_1 
)backward_lstm_132/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_3/stack_2
!backward_lstm_132/strided_slice_3StridedSlice=backward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_132/strided_slice_3/stack:output:02backward_lstm_132/strided_slice_3/stack_1:output:02backward_lstm_132/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2#
!backward_lstm_132/strided_slice_3
"backward_lstm_132/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_132/transpose_1/permі
backward_lstm_132/transpose_1	Transpose=backward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_132/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_132/transpose_1
backward_lstm_132/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_132/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisФ
concatConcatV2)forward_lstm_132/strided_slice_3:output:0*backward_lstm_132/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp7^backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp6^backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp8^backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp^backward_lstm_132/while6^forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp5^forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp7^forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp^forward_lstm_132/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2p
6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp2n
5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp2r
7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp22
backward_lstm_132/whilebackward_lstm_132/while2n
5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp2l
4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp2p
6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp20
forward_lstm_132/whileforward_lstm_132/while:O K
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
4__inference_backward_lstm_132_layer_call_fn_16718756

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
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_167153752
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
L__inference_sequential_132_layer_call_and_return_conditional_losses_16716537

inputs
inputs_1	-
bidirectional_132_16716518:	Ш-
bidirectional_132_16716520:	2Ш)
bidirectional_132_16716522:	Ш-
bidirectional_132_16716524:	Ш-
bidirectional_132_16716526:	2Ш)
bidirectional_132_16716528:	Ш$
dense_132_16716531:d 
dense_132_16716533:
identityЂ)bidirectional_132/StatefulPartitionedCallЂ!dense_132/StatefulPartitionedCallЪ
)bidirectional_132/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_132_16716518bidirectional_132_16716520bidirectional_132_16716522bidirectional_132_16716524bidirectional_132_16716526bidirectional_132_16716528*
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
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_167164742+
)bidirectional_132/StatefulPartitionedCallЫ
!dense_132/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_132/StatefulPartitionedCall:output:0dense_132_16716531dense_132_16716533*
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
G__inference_dense_132_layer_call_and_return_conditional_losses_167160592#
!dense_132/StatefulPartitionedCall
IdentityIdentity*dense_132/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp*^bidirectional_132/StatefulPartitionedCall"^dense_132/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2V
)bidirectional_132/StatefulPartitionedCall)bidirectional_132/StatefulPartitionedCall2F
!dense_132/StatefulPartitionedCall!dense_132/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ
Ѕ
$forward_lstm_132_while_cond_16717409>
:forward_lstm_132_while_forward_lstm_132_while_loop_counterD
@forward_lstm_132_while_forward_lstm_132_while_maximum_iterations&
"forward_lstm_132_while_placeholder(
$forward_lstm_132_while_placeholder_1(
$forward_lstm_132_while_placeholder_2(
$forward_lstm_132_while_placeholder_3(
$forward_lstm_132_while_placeholder_4@
<forward_lstm_132_while_less_forward_lstm_132_strided_slice_1X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16717409___redundant_placeholder0X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16717409___redundant_placeholder1X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16717409___redundant_placeholder2X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16717409___redundant_placeholder3X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16717409___redundant_placeholder4#
forward_lstm_132_while_identity
Х
forward_lstm_132/while/LessLess"forward_lstm_132_while_placeholder<forward_lstm_132_while_less_forward_lstm_132_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_132/while/Less
forward_lstm_132/while/IdentityIdentityforward_lstm_132/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_132/while/Identity"K
forward_lstm_132_while_identity(forward_lstm_132/while/Identity:output:0*(
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
while_cond_16718824
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16718824___redundant_placeholder06
2while_while_cond_16718824___redundant_placeholder16
2while_while_cond_16718824___redundant_placeholder26
2while_while_cond_16718824___redundant_placeholder3
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
while_body_16714531
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_398_16714555_0:	Ш1
while_lstm_cell_398_16714557_0:	2Ш-
while_lstm_cell_398_16714559_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_398_16714555:	Ш/
while_lstm_cell_398_16714557:	2Ш+
while_lstm_cell_398_16714559:	ШЂ+while/lstm_cell_398/StatefulPartitionedCallУ
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
+while/lstm_cell_398/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_398_16714555_0while_lstm_cell_398_16714557_0while_lstm_cell_398_16714559_0*
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
K__inference_lstm_cell_398_layer_call_and_return_conditional_losses_167144512-
+while/lstm_cell_398/StatefulPartitionedCallј
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_398/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_398/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѕ
while/Identity_5Identity4while/lstm_cell_398/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp,^while/lstm_cell_398/StatefulPartitionedCall*"
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
while_lstm_cell_398_16714555while_lstm_cell_398_16714555_0">
while_lstm_cell_398_16714557while_lstm_cell_398_16714557_0">
while_lstm_cell_398_16714559while_lstm_cell_398_16714559_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2Z
+while/lstm_cell_398/StatefulPartitionedCall+while/lstm_cell_398/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
І_
­
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_16719368

inputs?
,lstm_cell_398_matmul_readvariableop_resource:	ШA
.lstm_cell_398_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_398_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_398/BiasAdd/ReadVariableOpЂ#lstm_cell_398/MatMul/ReadVariableOpЂ%lstm_cell_398/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_398/MatMul/ReadVariableOpReadVariableOp,lstm_cell_398_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_398/MatMul/ReadVariableOpА
lstm_cell_398/MatMulMatMulstrided_slice_2:output:0+lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/MatMulО
%lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_398_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_398/MatMul_1/ReadVariableOpЌ
lstm_cell_398/MatMul_1MatMulzeros:output:0-lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/MatMul_1Є
lstm_cell_398/addAddV2lstm_cell_398/MatMul:product:0 lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/addЗ
$lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_398_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_398/BiasAdd/ReadVariableOpБ
lstm_cell_398/BiasAddBiasAddlstm_cell_398/add:z:0,lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/BiasAdd
lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_398/split/split_dimї
lstm_cell_398/splitSplit&lstm_cell_398/split/split_dim:output:0lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_398/split
lstm_cell_398/SigmoidSigmoidlstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Sigmoid
lstm_cell_398/Sigmoid_1Sigmoidlstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Sigmoid_1
lstm_cell_398/mulMullstm_cell_398/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/mul
lstm_cell_398/ReluRelulstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Relu 
lstm_cell_398/mul_1Mullstm_cell_398/Sigmoid:y:0 lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/mul_1
lstm_cell_398/add_1AddV2lstm_cell_398/mul:z:0lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/add_1
lstm_cell_398/Sigmoid_2Sigmoidlstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Sigmoid_2
lstm_cell_398/Relu_1Relulstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Relu_1Є
lstm_cell_398/mul_2Mullstm_cell_398/Sigmoid_2:y:0"lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_398_matmul_readvariableop_resource.lstm_cell_398_matmul_1_readvariableop_resource-lstm_cell_398_biasadd_readvariableop_resource*
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
while_body_16719284*
condR
while_cond_16719283*K
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
NoOpNoOp%^lstm_cell_398/BiasAdd/ReadVariableOp$^lstm_cell_398/MatMul/ReadVariableOp&^lstm_cell_398/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_398/BiasAdd/ReadVariableOp$lstm_cell_398/BiasAdd/ReadVariableOp2J
#lstm_cell_398/MatMul/ReadVariableOp#lstm_cell_398/MatMul/ReadVariableOp2N
%lstm_cell_398/MatMul_1/ReadVariableOp%lstm_cell_398/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ъ
М
%backward_lstm_132_while_cond_16715936@
<backward_lstm_132_while_backward_lstm_132_while_loop_counterF
Bbackward_lstm_132_while_backward_lstm_132_while_maximum_iterations'
#backward_lstm_132_while_placeholder)
%backward_lstm_132_while_placeholder_1)
%backward_lstm_132_while_placeholder_2)
%backward_lstm_132_while_placeholder_3)
%backward_lstm_132_while_placeholder_4B
>backward_lstm_132_while_less_backward_lstm_132_strided_slice_1Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16715936___redundant_placeholder0Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16715936___redundant_placeholder1Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16715936___redundant_placeholder2Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16715936___redundant_placeholder3Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16715936___redundant_placeholder4$
 backward_lstm_132_while_identity
Ъ
backward_lstm_132/while/LessLess#backward_lstm_132_while_placeholder>backward_lstm_132_while_less_backward_lstm_132_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_132/while/Less
 backward_lstm_132/while/IdentityIdentity backward_lstm_132/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_132/while/Identity"M
 backward_lstm_132_while_identity)backward_lstm_132/while/Identity:output:0*(
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
п
Ё
$forward_lstm_132_while_cond_16716790>
:forward_lstm_132_while_forward_lstm_132_while_loop_counterD
@forward_lstm_132_while_forward_lstm_132_while_maximum_iterations&
"forward_lstm_132_while_placeholder(
$forward_lstm_132_while_placeholder_1(
$forward_lstm_132_while_placeholder_2(
$forward_lstm_132_while_placeholder_3@
<forward_lstm_132_while_less_forward_lstm_132_strided_slice_1X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16716790___redundant_placeholder0X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16716790___redundant_placeholder1X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16716790___redundant_placeholder2X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16716790___redundant_placeholder3#
forward_lstm_132_while_identity
Х
forward_lstm_132/while/LessLess"forward_lstm_132_while_placeholder<forward_lstm_132_while_less_forward_lstm_132_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_132/while/Less
forward_lstm_132/while/IdentityIdentityforward_lstm_132/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_132/while/Identity"K
forward_lstm_132_while_identity(forward_lstm_132/while/Identity:output:0*(
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
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_16718044

inputs
inputs_1	P
=forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource:	ШR
?forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource:	2ШM
>forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource:	ШQ
>backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource:	ШS
@backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource:	2ШN
?backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource:	Ш
identityЂ6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpЂ5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpЂ7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpЂbackward_lstm_132/whileЂ5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpЂ4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpЂ6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOpЂforward_lstm_132/while
%forward_lstm_132/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_132/RaggedToTensor/zeros
%forward_lstm_132/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2'
%forward_lstm_132/RaggedToTensor/Const
4forward_lstm_132/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_132/RaggedToTensor/Const:output:0inputs.forward_lstm_132/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_132/RaggedToTensor/RaggedTensorToTensorФ
;forward_lstm_132/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_132/RaggedNestedRowLengths/strided_slice/stackШ
=forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1Ш
=forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2Љ
5forward_lstm_132/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_132/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask27
5forward_lstm_132/RaggedNestedRowLengths/strided_sliceШ
=forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stackе
?forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2A
?forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1Ь
?forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2Е
7forward_lstm_132/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask29
7forward_lstm_132/RaggedNestedRowLengths/strided_slice_1
+forward_lstm_132/RaggedNestedRowLengths/subSub>forward_lstm_132/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_132/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2-
+forward_lstm_132/RaggedNestedRowLengths/subЄ
forward_lstm_132/CastCast/forward_lstm_132/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_132/Cast
forward_lstm_132/ShapeShape=forward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_132/Shape
$forward_lstm_132/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_132/strided_slice/stack
&forward_lstm_132/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_132/strided_slice/stack_1
&forward_lstm_132/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_132/strided_slice/stack_2Ш
forward_lstm_132/strided_sliceStridedSliceforward_lstm_132/Shape:output:0-forward_lstm_132/strided_slice/stack:output:0/forward_lstm_132/strided_slice/stack_1:output:0/forward_lstm_132/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_132/strided_slice~
forward_lstm_132/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_132/zeros/mul/yА
forward_lstm_132/zeros/mulMul'forward_lstm_132/strided_slice:output:0%forward_lstm_132/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros/mul
forward_lstm_132/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_132/zeros/Less/yЋ
forward_lstm_132/zeros/LessLessforward_lstm_132/zeros/mul:z:0&forward_lstm_132/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros/Less
forward_lstm_132/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_132/zeros/packed/1Ч
forward_lstm_132/zeros/packedPack'forward_lstm_132/strided_slice:output:0(forward_lstm_132/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_132/zeros/packed
forward_lstm_132/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_132/zeros/ConstЙ
forward_lstm_132/zerosFill&forward_lstm_132/zeros/packed:output:0%forward_lstm_132/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/zeros
forward_lstm_132/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_132/zeros_1/mul/yЖ
forward_lstm_132/zeros_1/mulMul'forward_lstm_132/strided_slice:output:0'forward_lstm_132/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros_1/mul
forward_lstm_132/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
forward_lstm_132/zeros_1/Less/yГ
forward_lstm_132/zeros_1/LessLess forward_lstm_132/zeros_1/mul:z:0(forward_lstm_132/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros_1/Less
!forward_lstm_132/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_132/zeros_1/packed/1Э
forward_lstm_132/zeros_1/packedPack'forward_lstm_132/strided_slice:output:0*forward_lstm_132/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_132/zeros_1/packed
forward_lstm_132/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_132/zeros_1/ConstС
forward_lstm_132/zeros_1Fill(forward_lstm_132/zeros_1/packed:output:0'forward_lstm_132/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/zeros_1
forward_lstm_132/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_132/transpose/permэ
forward_lstm_132/transpose	Transpose=forward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_132/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_132/transpose
forward_lstm_132/Shape_1Shapeforward_lstm_132/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_132/Shape_1
&forward_lstm_132/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_132/strided_slice_1/stack
(forward_lstm_132/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_1/stack_1
(forward_lstm_132/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_1/stack_2д
 forward_lstm_132/strided_slice_1StridedSlice!forward_lstm_132/Shape_1:output:0/forward_lstm_132/strided_slice_1/stack:output:01forward_lstm_132/strided_slice_1/stack_1:output:01forward_lstm_132/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_132/strided_slice_1Ї
,forward_lstm_132/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,forward_lstm_132/TensorArrayV2/element_shapeі
forward_lstm_132/TensorArrayV2TensorListReserve5forward_lstm_132/TensorArrayV2/element_shape:output:0)forward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_132/TensorArrayV2с
Fforward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fforward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8forward_lstm_132/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_132/transpose:y:0Oforward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_132/TensorArrayUnstack/TensorListFromTensor
&forward_lstm_132/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_132/strided_slice_2/stack
(forward_lstm_132/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_2/stack_1
(forward_lstm_132/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_2/stack_2т
 forward_lstm_132/strided_slice_2StridedSliceforward_lstm_132/transpose:y:0/forward_lstm_132/strided_slice_2/stack:output:01forward_lstm_132/strided_slice_2/stack_1:output:01forward_lstm_132/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 forward_lstm_132/strided_slice_2ы
4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpReadVariableOp=forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpє
%forward_lstm_132/lstm_cell_397/MatMulMatMul)forward_lstm_132/strided_slice_2:output:0<forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_132/lstm_cell_397/MatMulё
6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOp?forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp№
'forward_lstm_132/lstm_cell_397/MatMul_1MatMulforward_lstm_132/zeros:output:0>forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_132/lstm_cell_397/MatMul_1ш
"forward_lstm_132/lstm_cell_397/addAddV2/forward_lstm_132/lstm_cell_397/MatMul:product:01forward_lstm_132/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_132/lstm_cell_397/addъ
5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOp>forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpѕ
&forward_lstm_132/lstm_cell_397/BiasAddBiasAdd&forward_lstm_132/lstm_cell_397/add:z:0=forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_132/lstm_cell_397/BiasAddЂ
.forward_lstm_132/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.forward_lstm_132/lstm_cell_397/split/split_dimЛ
$forward_lstm_132/lstm_cell_397/splitSplit7forward_lstm_132/lstm_cell_397/split/split_dim:output:0/forward_lstm_132/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$forward_lstm_132/lstm_cell_397/splitМ
&forward_lstm_132/lstm_cell_397/SigmoidSigmoid-forward_lstm_132/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_132/lstm_cell_397/SigmoidР
(forward_lstm_132/lstm_cell_397/Sigmoid_1Sigmoid-forward_lstm_132/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_132/lstm_cell_397/Sigmoid_1в
"forward_lstm_132/lstm_cell_397/mulMul,forward_lstm_132/lstm_cell_397/Sigmoid_1:y:0!forward_lstm_132/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_132/lstm_cell_397/mulГ
#forward_lstm_132/lstm_cell_397/ReluRelu-forward_lstm_132/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_132/lstm_cell_397/Reluф
$forward_lstm_132/lstm_cell_397/mul_1Mul*forward_lstm_132/lstm_cell_397/Sigmoid:y:01forward_lstm_132/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_132/lstm_cell_397/mul_1й
$forward_lstm_132/lstm_cell_397/add_1AddV2&forward_lstm_132/lstm_cell_397/mul:z:0(forward_lstm_132/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_132/lstm_cell_397/add_1Р
(forward_lstm_132/lstm_cell_397/Sigmoid_2Sigmoid-forward_lstm_132/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_132/lstm_cell_397/Sigmoid_2В
%forward_lstm_132/lstm_cell_397/Relu_1Relu(forward_lstm_132/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_132/lstm_cell_397/Relu_1ш
$forward_lstm_132/lstm_cell_397/mul_2Mul,forward_lstm_132/lstm_cell_397/Sigmoid_2:y:03forward_lstm_132/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_132/lstm_cell_397/mul_2Б
.forward_lstm_132/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.forward_lstm_132/TensorArrayV2_1/element_shapeќ
 forward_lstm_132/TensorArrayV2_1TensorListReserve7forward_lstm_132/TensorArrayV2_1/element_shape:output:0)forward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_132/TensorArrayV2_1p
forward_lstm_132/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_132/timeЃ
forward_lstm_132/zeros_like	ZerosLike(forward_lstm_132/lstm_cell_397/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/zeros_likeЁ
)forward_lstm_132/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)forward_lstm_132/while/maximum_iterations
#forward_lstm_132/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_132/while/loop_counter	
forward_lstm_132/whileWhile,forward_lstm_132/while/loop_counter:output:02forward_lstm_132/while/maximum_iterations:output:0forward_lstm_132/time:output:0)forward_lstm_132/TensorArrayV2_1:handle:0forward_lstm_132/zeros_like:y:0forward_lstm_132/zeros:output:0!forward_lstm_132/zeros_1:output:0)forward_lstm_132/strided_slice_1:output:0Hforward_lstm_132/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_132/Cast:y:0=forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource?forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource>forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource*
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
$forward_lstm_132_while_body_16717768*0
cond(R&
$forward_lstm_132_while_cond_16717767*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_132/whileз
Aforward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Aforward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3forward_lstm_132/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_132/while:output:3Jforward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3forward_lstm_132/TensorArrayV2Stack/TensorListStackЃ
&forward_lstm_132/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&forward_lstm_132/strided_slice_3/stack
(forward_lstm_132/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_132/strided_slice_3/stack_1
(forward_lstm_132/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_3/stack_2
 forward_lstm_132/strided_slice_3StridedSlice<forward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_132/strided_slice_3/stack:output:01forward_lstm_132/strided_slice_3/stack_1:output:01forward_lstm_132/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 forward_lstm_132/strided_slice_3
!forward_lstm_132/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_132/transpose_1/permђ
forward_lstm_132/transpose_1	Transpose<forward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_132/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_132/transpose_1
forward_lstm_132/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_132/runtime
&backward_lstm_132/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_132/RaggedToTensor/zeros
&backward_lstm_132/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2(
&backward_lstm_132/RaggedToTensor/Const
5backward_lstm_132/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_132/RaggedToTensor/Const:output:0inputs/backward_lstm_132/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_132/RaggedToTensor/RaggedTensorToTensorЦ
<backward_lstm_132/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_132/RaggedNestedRowLengths/strided_slice/stackЪ
>backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1Ъ
>backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2Ў
6backward_lstm_132/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_132/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask28
6backward_lstm_132/RaggedNestedRowLengths/strided_sliceЪ
>backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stackз
@backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2B
@backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1Ю
@backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2К
8backward_lstm_132/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2:
8backward_lstm_132/RaggedNestedRowLengths/strided_slice_1
,backward_lstm_132/RaggedNestedRowLengths/subSub?backward_lstm_132/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_132/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2.
,backward_lstm_132/RaggedNestedRowLengths/subЇ
backward_lstm_132/CastCast0backward_lstm_132/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_132/Cast 
backward_lstm_132/ShapeShape>backward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_132/Shape
%backward_lstm_132/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_132/strided_slice/stack
'backward_lstm_132/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_132/strided_slice/stack_1
'backward_lstm_132/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_132/strided_slice/stack_2Ю
backward_lstm_132/strided_sliceStridedSlice backward_lstm_132/Shape:output:0.backward_lstm_132/strided_slice/stack:output:00backward_lstm_132/strided_slice/stack_1:output:00backward_lstm_132/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_132/strided_slice
backward_lstm_132/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_132/zeros/mul/yД
backward_lstm_132/zeros/mulMul(backward_lstm_132/strided_slice:output:0&backward_lstm_132/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/zeros/mul
backward_lstm_132/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_132/zeros/Less/yЏ
backward_lstm_132/zeros/LessLessbackward_lstm_132/zeros/mul:z:0'backward_lstm_132/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/zeros/Less
 backward_lstm_132/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_132/zeros/packed/1Ы
backward_lstm_132/zeros/packedPack(backward_lstm_132/strided_slice:output:0)backward_lstm_132/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_132/zeros/packed
backward_lstm_132/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_132/zeros/ConstН
backward_lstm_132/zerosFill'backward_lstm_132/zeros/packed:output:0&backward_lstm_132/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_132/zeros
backward_lstm_132/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_132/zeros_1/mul/yК
backward_lstm_132/zeros_1/mulMul(backward_lstm_132/strided_slice:output:0(backward_lstm_132/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/zeros_1/mul
 backward_lstm_132/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 backward_lstm_132/zeros_1/Less/yЗ
backward_lstm_132/zeros_1/LessLess!backward_lstm_132/zeros_1/mul:z:0)backward_lstm_132/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_132/zeros_1/Less
"backward_lstm_132/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_132/zeros_1/packed/1б
 backward_lstm_132/zeros_1/packedPack(backward_lstm_132/strided_slice:output:0+backward_lstm_132/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_132/zeros_1/packed
backward_lstm_132/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_132/zeros_1/ConstХ
backward_lstm_132/zeros_1Fill)backward_lstm_132/zeros_1/packed:output:0(backward_lstm_132/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_132/zeros_1
 backward_lstm_132/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_132/transpose/permё
backward_lstm_132/transpose	Transpose>backward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_132/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_132/transpose
backward_lstm_132/Shape_1Shapebackward_lstm_132/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_132/Shape_1
'backward_lstm_132/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_132/strided_slice_1/stack 
)backward_lstm_132/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_1/stack_1 
)backward_lstm_132/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_1/stack_2к
!backward_lstm_132/strided_slice_1StridedSlice"backward_lstm_132/Shape_1:output:00backward_lstm_132/strided_slice_1/stack:output:02backward_lstm_132/strided_slice_1/stack_1:output:02backward_lstm_132/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_132/strided_slice_1Љ
-backward_lstm_132/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2/
-backward_lstm_132/TensorArrayV2/element_shapeњ
backward_lstm_132/TensorArrayV2TensorListReserve6backward_lstm_132/TensorArrayV2/element_shape:output:0*backward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_132/TensorArrayV2
 backward_lstm_132/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_132/ReverseV2/axisв
backward_lstm_132/ReverseV2	ReverseV2backward_lstm_132/transpose:y:0)backward_lstm_132/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_132/ReverseV2у
Gbackward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gbackward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeХ
9backward_lstm_132/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_132/ReverseV2:output:0Pbackward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_132/TensorArrayUnstack/TensorListFromTensor
'backward_lstm_132/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_132/strided_slice_2/stack 
)backward_lstm_132/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_2/stack_1 
)backward_lstm_132/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_2/stack_2ш
!backward_lstm_132/strided_slice_2StridedSlicebackward_lstm_132/transpose:y:00backward_lstm_132/strided_slice_2/stack:output:02backward_lstm_132/strided_slice_2/stack_1:output:02backward_lstm_132/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2#
!backward_lstm_132/strided_slice_2ю
5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpReadVariableOp>backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype027
5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpј
&backward_lstm_132/lstm_cell_398/MatMulMatMul*backward_lstm_132/strided_slice_2:output:0=backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_132/lstm_cell_398/MatMulє
7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOp@backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype029
7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpє
(backward_lstm_132/lstm_cell_398/MatMul_1MatMul backward_lstm_132/zeros:output:0?backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_132/lstm_cell_398/MatMul_1ь
#backward_lstm_132/lstm_cell_398/addAddV20backward_lstm_132/lstm_cell_398/MatMul:product:02backward_lstm_132/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_132/lstm_cell_398/addэ
6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOp?backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype028
6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpљ
'backward_lstm_132/lstm_cell_398/BiasAddBiasAdd'backward_lstm_132/lstm_cell_398/add:z:0>backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_132/lstm_cell_398/BiasAddЄ
/backward_lstm_132/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/backward_lstm_132/lstm_cell_398/split/split_dimП
%backward_lstm_132/lstm_cell_398/splitSplit8backward_lstm_132/lstm_cell_398/split/split_dim:output:00backward_lstm_132/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2'
%backward_lstm_132/lstm_cell_398/splitП
'backward_lstm_132/lstm_cell_398/SigmoidSigmoid.backward_lstm_132/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_132/lstm_cell_398/SigmoidУ
)backward_lstm_132/lstm_cell_398/Sigmoid_1Sigmoid.backward_lstm_132/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_132/lstm_cell_398/Sigmoid_1ж
#backward_lstm_132/lstm_cell_398/mulMul-backward_lstm_132/lstm_cell_398/Sigmoid_1:y:0"backward_lstm_132/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_132/lstm_cell_398/mulЖ
$backward_lstm_132/lstm_cell_398/ReluRelu.backward_lstm_132/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_132/lstm_cell_398/Reluш
%backward_lstm_132/lstm_cell_398/mul_1Mul+backward_lstm_132/lstm_cell_398/Sigmoid:y:02backward_lstm_132/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_132/lstm_cell_398/mul_1н
%backward_lstm_132/lstm_cell_398/add_1AddV2'backward_lstm_132/lstm_cell_398/mul:z:0)backward_lstm_132/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_132/lstm_cell_398/add_1У
)backward_lstm_132/lstm_cell_398/Sigmoid_2Sigmoid.backward_lstm_132/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_132/lstm_cell_398/Sigmoid_2Е
&backward_lstm_132/lstm_cell_398/Relu_1Relu)backward_lstm_132/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_132/lstm_cell_398/Relu_1ь
%backward_lstm_132/lstm_cell_398/mul_2Mul-backward_lstm_132/lstm_cell_398/Sigmoid_2:y:04backward_lstm_132/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_132/lstm_cell_398/mul_2Г
/backward_lstm_132/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   21
/backward_lstm_132/TensorArrayV2_1/element_shape
!backward_lstm_132/TensorArrayV2_1TensorListReserve8backward_lstm_132/TensorArrayV2_1/element_shape:output:0*backward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_132/TensorArrayV2_1r
backward_lstm_132/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_132/time
'backward_lstm_132/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_132/Max/reduction_indicesЄ
backward_lstm_132/MaxMaxbackward_lstm_132/Cast:y:00backward_lstm_132/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/Maxt
backward_lstm_132/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_132/sub/y
backward_lstm_132/subSubbackward_lstm_132/Max:output:0 backward_lstm_132/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/sub
backward_lstm_132/Sub_1Subbackward_lstm_132/sub:z:0backward_lstm_132/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_132/Sub_1І
backward_lstm_132/zeros_like	ZerosLike)backward_lstm_132/lstm_cell_398/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_132/zeros_likeЃ
*backward_lstm_132/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*backward_lstm_132/while/maximum_iterations
$backward_lstm_132/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_132/while/loop_counterЅ	
backward_lstm_132/whileWhile-backward_lstm_132/while/loop_counter:output:03backward_lstm_132/while/maximum_iterations:output:0backward_lstm_132/time:output:0*backward_lstm_132/TensorArrayV2_1:handle:0 backward_lstm_132/zeros_like:y:0 backward_lstm_132/zeros:output:0"backward_lstm_132/zeros_1:output:0*backward_lstm_132/strided_slice_1:output:0Ibackward_lstm_132/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_132/Sub_1:z:0>backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource@backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource?backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource*
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
%backward_lstm_132_while_body_16717947*1
cond)R'
%backward_lstm_132_while_cond_16717946*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_132/whileй
Bbackward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2D
Bbackward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeЙ
4backward_lstm_132/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_132/while:output:3Kbackward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype026
4backward_lstm_132/TensorArrayV2Stack/TensorListStackЅ
'backward_lstm_132/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2)
'backward_lstm_132/strided_slice_3/stack 
)backward_lstm_132/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_132/strided_slice_3/stack_1 
)backward_lstm_132/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_3/stack_2
!backward_lstm_132/strided_slice_3StridedSlice=backward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_132/strided_slice_3/stack:output:02backward_lstm_132/strided_slice_3/stack_1:output:02backward_lstm_132/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2#
!backward_lstm_132/strided_slice_3
"backward_lstm_132/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_132/transpose_1/permі
backward_lstm_132/transpose_1	Transpose=backward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_132/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_132/transpose_1
backward_lstm_132/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_132/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisФ
concatConcatV2)forward_lstm_132/strided_slice_3:output:0*backward_lstm_132/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp7^backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp6^backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp8^backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp^backward_lstm_132/while6^forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp5^forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp7^forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp^forward_lstm_132/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2p
6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp2n
5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp2r
7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp22
backward_lstm_132/whilebackward_lstm_132/while2n
5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp2l
4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp2p
6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp20
forward_lstm_132/whileforward_lstm_132/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
зу

#__inference__wrapped_model_16713598

args_0
args_0_1	q
^sequential_132_bidirectional_132_forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource:	Шs
`sequential_132_bidirectional_132_forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource:	2Шn
_sequential_132_bidirectional_132_forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource:	Шr
_sequential_132_bidirectional_132_backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource:	Шt
asequential_132_bidirectional_132_backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource:	2Шo
`sequential_132_bidirectional_132_backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource:	ШI
7sequential_132_dense_132_matmul_readvariableop_resource:dF
8sequential_132_dense_132_biasadd_readvariableop_resource:
identityЂWsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpЂVsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpЂXsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpЂ8sequential_132/bidirectional_132/backward_lstm_132/whileЂVsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpЂUsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpЂWsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOpЂ7sequential_132/bidirectional_132/forward_lstm_132/whileЂ/sequential_132/dense_132/BiasAdd/ReadVariableOpЂ.sequential_132/dense_132/MatMul/ReadVariableOpй
Fsequential_132/bidirectional_132/forward_lstm_132/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2H
Fsequential_132/bidirectional_132/forward_lstm_132/RaggedToTensor/zerosл
Fsequential_132/bidirectional_132/forward_lstm_132/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2H
Fsequential_132/bidirectional_132/forward_lstm_132/RaggedToTensor/Const
Usequential_132/bidirectional_132/forward_lstm_132/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorOsequential_132/bidirectional_132/forward_lstm_132/RaggedToTensor/Const:output:0args_0Osequential_132/bidirectional_132/forward_lstm_132/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2W
Usequential_132/bidirectional_132/forward_lstm_132/RaggedToTensor/RaggedTensorToTensor
\sequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2^
\sequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack
^sequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2`
^sequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1
^sequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2`
^sequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2Ю
Vsequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1esequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack:output:0gsequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1:output:0gsequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask2X
Vsequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice
^sequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2`
^sequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack
`sequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2b
`sequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1
`sequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2b
`sequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2к
Xsequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1gsequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack:output:0isequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0isequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2Z
Xsequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice_1
Lsequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/subSub_sequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice:output:0asequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2N
Lsequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/sub
6sequential_132/bidirectional_132/forward_lstm_132/CastCastPsequential_132/bidirectional_132/forward_lstm_132/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ28
6sequential_132/bidirectional_132/forward_lstm_132/Cast
7sequential_132/bidirectional_132/forward_lstm_132/ShapeShape^sequential_132/bidirectional_132/forward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:29
7sequential_132/bidirectional_132/forward_lstm_132/Shapeи
Esequential_132/bidirectional_132/forward_lstm_132/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_132/bidirectional_132/forward_lstm_132/strided_slice/stackм
Gsequential_132/bidirectional_132/forward_lstm_132/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_132/bidirectional_132/forward_lstm_132/strided_slice/stack_1м
Gsequential_132/bidirectional_132/forward_lstm_132/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_132/bidirectional_132/forward_lstm_132/strided_slice/stack_2
?sequential_132/bidirectional_132/forward_lstm_132/strided_sliceStridedSlice@sequential_132/bidirectional_132/forward_lstm_132/Shape:output:0Nsequential_132/bidirectional_132/forward_lstm_132/strided_slice/stack:output:0Psequential_132/bidirectional_132/forward_lstm_132/strided_slice/stack_1:output:0Psequential_132/bidirectional_132/forward_lstm_132/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?sequential_132/bidirectional_132/forward_lstm_132/strided_sliceР
=sequential_132/bidirectional_132/forward_lstm_132/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22?
=sequential_132/bidirectional_132/forward_lstm_132/zeros/mul/yД
;sequential_132/bidirectional_132/forward_lstm_132/zeros/mulMulHsequential_132/bidirectional_132/forward_lstm_132/strided_slice:output:0Fsequential_132/bidirectional_132/forward_lstm_132/zeros/mul/y:output:0*
T0*
_output_shapes
: 2=
;sequential_132/bidirectional_132/forward_lstm_132/zeros/mulУ
>sequential_132/bidirectional_132/forward_lstm_132/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2@
>sequential_132/bidirectional_132/forward_lstm_132/zeros/Less/yЏ
<sequential_132/bidirectional_132/forward_lstm_132/zeros/LessLess?sequential_132/bidirectional_132/forward_lstm_132/zeros/mul:z:0Gsequential_132/bidirectional_132/forward_lstm_132/zeros/Less/y:output:0*
T0*
_output_shapes
: 2>
<sequential_132/bidirectional_132/forward_lstm_132/zeros/LessЦ
@sequential_132/bidirectional_132/forward_lstm_132/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22B
@sequential_132/bidirectional_132/forward_lstm_132/zeros/packed/1Ы
>sequential_132/bidirectional_132/forward_lstm_132/zeros/packedPackHsequential_132/bidirectional_132/forward_lstm_132/strided_slice:output:0Isequential_132/bidirectional_132/forward_lstm_132/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2@
>sequential_132/bidirectional_132/forward_lstm_132/zeros/packedЧ
=sequential_132/bidirectional_132/forward_lstm_132/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2?
=sequential_132/bidirectional_132/forward_lstm_132/zeros/ConstН
7sequential_132/bidirectional_132/forward_lstm_132/zerosFillGsequential_132/bidirectional_132/forward_lstm_132/zeros/packed:output:0Fsequential_132/bidirectional_132/forward_lstm_132/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ229
7sequential_132/bidirectional_132/forward_lstm_132/zerosФ
?sequential_132/bidirectional_132/forward_lstm_132/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22A
?sequential_132/bidirectional_132/forward_lstm_132/zeros_1/mul/yК
=sequential_132/bidirectional_132/forward_lstm_132/zeros_1/mulMulHsequential_132/bidirectional_132/forward_lstm_132/strided_slice:output:0Hsequential_132/bidirectional_132/forward_lstm_132/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2?
=sequential_132/bidirectional_132/forward_lstm_132/zeros_1/mulЧ
@sequential_132/bidirectional_132/forward_lstm_132/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2B
@sequential_132/bidirectional_132/forward_lstm_132/zeros_1/Less/yЗ
>sequential_132/bidirectional_132/forward_lstm_132/zeros_1/LessLessAsequential_132/bidirectional_132/forward_lstm_132/zeros_1/mul:z:0Isequential_132/bidirectional_132/forward_lstm_132/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2@
>sequential_132/bidirectional_132/forward_lstm_132/zeros_1/LessЪ
Bsequential_132/bidirectional_132/forward_lstm_132/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22D
Bsequential_132/bidirectional_132/forward_lstm_132/zeros_1/packed/1б
@sequential_132/bidirectional_132/forward_lstm_132/zeros_1/packedPackHsequential_132/bidirectional_132/forward_lstm_132/strided_slice:output:0Ksequential_132/bidirectional_132/forward_lstm_132/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2B
@sequential_132/bidirectional_132/forward_lstm_132/zeros_1/packedЫ
?sequential_132/bidirectional_132/forward_lstm_132/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2A
?sequential_132/bidirectional_132/forward_lstm_132/zeros_1/ConstХ
9sequential_132/bidirectional_132/forward_lstm_132/zeros_1FillIsequential_132/bidirectional_132/forward_lstm_132/zeros_1/packed:output:0Hsequential_132/bidirectional_132/forward_lstm_132/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22;
9sequential_132/bidirectional_132/forward_lstm_132/zeros_1й
@sequential_132/bidirectional_132/forward_lstm_132/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2B
@sequential_132/bidirectional_132/forward_lstm_132/transpose/permё
;sequential_132/bidirectional_132/forward_lstm_132/transpose	Transpose^sequential_132/bidirectional_132/forward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0Isequential_132/bidirectional_132/forward_lstm_132/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2=
;sequential_132/bidirectional_132/forward_lstm_132/transposeх
9sequential_132/bidirectional_132/forward_lstm_132/Shape_1Shape?sequential_132/bidirectional_132/forward_lstm_132/transpose:y:0*
T0*
_output_shapes
:2;
9sequential_132/bidirectional_132/forward_lstm_132/Shape_1м
Gsequential_132/bidirectional_132/forward_lstm_132/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2I
Gsequential_132/bidirectional_132/forward_lstm_132/strided_slice_1/stackр
Isequential_132/bidirectional_132/forward_lstm_132/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_132/bidirectional_132/forward_lstm_132/strided_slice_1/stack_1р
Isequential_132/bidirectional_132/forward_lstm_132/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_132/bidirectional_132/forward_lstm_132/strided_slice_1/stack_2
Asequential_132/bidirectional_132/forward_lstm_132/strided_slice_1StridedSliceBsequential_132/bidirectional_132/forward_lstm_132/Shape_1:output:0Psequential_132/bidirectional_132/forward_lstm_132/strided_slice_1/stack:output:0Rsequential_132/bidirectional_132/forward_lstm_132/strided_slice_1/stack_1:output:0Rsequential_132/bidirectional_132/forward_lstm_132/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2C
Asequential_132/bidirectional_132/forward_lstm_132/strided_slice_1щ
Msequential_132/bidirectional_132/forward_lstm_132/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2O
Msequential_132/bidirectional_132/forward_lstm_132/TensorArrayV2/element_shapeњ
?sequential_132/bidirectional_132/forward_lstm_132/TensorArrayV2TensorListReserveVsequential_132/bidirectional_132/forward_lstm_132/TensorArrayV2/element_shape:output:0Jsequential_132/bidirectional_132/forward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?sequential_132/bidirectional_132/forward_lstm_132/TensorArrayV2Ѓ
gsequential_132/bidirectional_132/forward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2i
gsequential_132/bidirectional_132/forward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeР
Ysequential_132/bidirectional_132/forward_lstm_132/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor?sequential_132/bidirectional_132/forward_lstm_132/transpose:y:0psequential_132/bidirectional_132/forward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02[
Ysequential_132/bidirectional_132/forward_lstm_132/TensorArrayUnstack/TensorListFromTensorм
Gsequential_132/bidirectional_132/forward_lstm_132/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2I
Gsequential_132/bidirectional_132/forward_lstm_132/strided_slice_2/stackр
Isequential_132/bidirectional_132/forward_lstm_132/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_132/bidirectional_132/forward_lstm_132/strided_slice_2/stack_1р
Isequential_132/bidirectional_132/forward_lstm_132/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_132/bidirectional_132/forward_lstm_132/strided_slice_2/stack_2Ј
Asequential_132/bidirectional_132/forward_lstm_132/strided_slice_2StridedSlice?sequential_132/bidirectional_132/forward_lstm_132/transpose:y:0Psequential_132/bidirectional_132/forward_lstm_132/strided_slice_2/stack:output:0Rsequential_132/bidirectional_132/forward_lstm_132/strided_slice_2/stack_1:output:0Rsequential_132/bidirectional_132/forward_lstm_132/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2C
Asequential_132/bidirectional_132/forward_lstm_132/strided_slice_2Ю
Usequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpReadVariableOp^sequential_132_bidirectional_132_forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02W
Usequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpј
Fsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMulMatMulJsequential_132/bidirectional_132/forward_lstm_132/strided_slice_2:output:0]sequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2H
Fsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMulд
Wsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOp`sequential_132_bidirectional_132_forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02Y
Wsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOpє
Hsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMul_1MatMul@sequential_132/bidirectional_132/forward_lstm_132/zeros:output:0_sequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2J
Hsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMul_1ь
Csequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/addAddV2Psequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMul:product:0Rsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2E
Csequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/addЭ
Vsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOp_sequential_132_bidirectional_132_forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02X
Vsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpљ
Gsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/BiasAddBiasAddGsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/add:z:0^sequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2I
Gsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/BiasAddф
Osequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2Q
Osequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/split/split_dimП
Esequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/splitSplitXsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/split/split_dim:output:0Psequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2G
Esequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/split
Gsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/SigmoidSigmoidNsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22I
Gsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/SigmoidЃ
Isequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/Sigmoid_1SigmoidNsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22K
Isequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/Sigmoid_1ж
Csequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/mulMulMsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/Sigmoid_1:y:0Bsequential_132/bidirectional_132/forward_lstm_132/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/mul
Dsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/ReluReluNsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22F
Dsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/Reluш
Esequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/mul_1MulKsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/Sigmoid:y:0Rsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22G
Esequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/mul_1н
Esequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/add_1AddV2Gsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/mul:z:0Isequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22G
Esequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/add_1Ѓ
Isequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/Sigmoid_2SigmoidNsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22K
Isequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/Sigmoid_2
Fsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/Relu_1ReluIsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22H
Fsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/Relu_1ь
Esequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/mul_2MulMsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/Sigmoid_2:y:0Tsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22G
Esequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/mul_2ѓ
Osequential_132/bidirectional_132/forward_lstm_132/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2Q
Osequential_132/bidirectional_132/forward_lstm_132/TensorArrayV2_1/element_shape
Asequential_132/bidirectional_132/forward_lstm_132/TensorArrayV2_1TensorListReserveXsequential_132/bidirectional_132/forward_lstm_132/TensorArrayV2_1/element_shape:output:0Jsequential_132/bidirectional_132/forward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02C
Asequential_132/bidirectional_132/forward_lstm_132/TensorArrayV2_1В
6sequential_132/bidirectional_132/forward_lstm_132/timeConst*
_output_shapes
: *
dtype0*
value	B : 28
6sequential_132/bidirectional_132/forward_lstm_132/time
<sequential_132/bidirectional_132/forward_lstm_132/zeros_like	ZerosLikeIsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22>
<sequential_132/bidirectional_132/forward_lstm_132/zeros_likeу
Jsequential_132/bidirectional_132/forward_lstm_132/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2L
Jsequential_132/bidirectional_132/forward_lstm_132/while/maximum_iterationsЮ
Dsequential_132/bidirectional_132/forward_lstm_132/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dsequential_132/bidirectional_132/forward_lstm_132/while/loop_counterФ
7sequential_132/bidirectional_132/forward_lstm_132/whileWhileMsequential_132/bidirectional_132/forward_lstm_132/while/loop_counter:output:0Ssequential_132/bidirectional_132/forward_lstm_132/while/maximum_iterations:output:0?sequential_132/bidirectional_132/forward_lstm_132/time:output:0Jsequential_132/bidirectional_132/forward_lstm_132/TensorArrayV2_1:handle:0@sequential_132/bidirectional_132/forward_lstm_132/zeros_like:y:0@sequential_132/bidirectional_132/forward_lstm_132/zeros:output:0Bsequential_132/bidirectional_132/forward_lstm_132/zeros_1:output:0Jsequential_132/bidirectional_132/forward_lstm_132/strided_slice_1:output:0isequential_132/bidirectional_132/forward_lstm_132/TensorArrayUnstack/TensorListFromTensor:output_handle:0:sequential_132/bidirectional_132/forward_lstm_132/Cast:y:0^sequential_132_bidirectional_132_forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource`sequential_132_bidirectional_132_forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource_sequential_132_bidirectional_132_forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource*
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
Esequential_132_bidirectional_132_forward_lstm_132_while_body_16713315*Q
condIRG
Esequential_132_bidirectional_132_forward_lstm_132_while_cond_16713314*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 29
7sequential_132/bidirectional_132/forward_lstm_132/while
bsequential_132/bidirectional_132/forward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2d
bsequential_132/bidirectional_132/forward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeЙ
Tsequential_132/bidirectional_132/forward_lstm_132/TensorArrayV2Stack/TensorListStackTensorListStack@sequential_132/bidirectional_132/forward_lstm_132/while:output:3ksequential_132/bidirectional_132/forward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02V
Tsequential_132/bidirectional_132/forward_lstm_132/TensorArrayV2Stack/TensorListStackх
Gsequential_132/bidirectional_132/forward_lstm_132/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2I
Gsequential_132/bidirectional_132/forward_lstm_132/strided_slice_3/stackр
Isequential_132/bidirectional_132/forward_lstm_132/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2K
Isequential_132/bidirectional_132/forward_lstm_132/strided_slice_3/stack_1р
Isequential_132/bidirectional_132/forward_lstm_132/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_132/bidirectional_132/forward_lstm_132/strided_slice_3/stack_2Ц
Asequential_132/bidirectional_132/forward_lstm_132/strided_slice_3StridedSlice]sequential_132/bidirectional_132/forward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0Psequential_132/bidirectional_132/forward_lstm_132/strided_slice_3/stack:output:0Rsequential_132/bidirectional_132/forward_lstm_132/strided_slice_3/stack_1:output:0Rsequential_132/bidirectional_132/forward_lstm_132/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2C
Asequential_132/bidirectional_132/forward_lstm_132/strided_slice_3н
Bsequential_132/bidirectional_132/forward_lstm_132/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2D
Bsequential_132/bidirectional_132/forward_lstm_132/transpose_1/permі
=sequential_132/bidirectional_132/forward_lstm_132/transpose_1	Transpose]sequential_132/bidirectional_132/forward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0Ksequential_132/bidirectional_132/forward_lstm_132/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22?
=sequential_132/bidirectional_132/forward_lstm_132/transpose_1Ъ
9sequential_132/bidirectional_132/forward_lstm_132/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2;
9sequential_132/bidirectional_132/forward_lstm_132/runtimeл
Gsequential_132/bidirectional_132/backward_lstm_132/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2I
Gsequential_132/bidirectional_132/backward_lstm_132/RaggedToTensor/zerosн
Gsequential_132/bidirectional_132/backward_lstm_132/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2I
Gsequential_132/bidirectional_132/backward_lstm_132/RaggedToTensor/ConstЁ
Vsequential_132/bidirectional_132/backward_lstm_132/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorPsequential_132/bidirectional_132/backward_lstm_132/RaggedToTensor/Const:output:0args_0Psequential_132/bidirectional_132/backward_lstm_132/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2X
Vsequential_132/bidirectional_132/backward_lstm_132/RaggedToTensor/RaggedTensorToTensor
]sequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2_
]sequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack
_sequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2a
_sequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1
_sequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2a
_sequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2г
Wsequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1fsequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack:output:0hsequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1:output:0hsequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask2Y
Wsequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice
_sequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2a
_sequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack
asequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2c
asequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1
asequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2c
asequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2п
Ysequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1hsequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack:output:0jsequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0jsequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2[
Ysequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice_1
Msequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/subSub`sequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice:output:0bsequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2O
Msequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/sub
7sequential_132/bidirectional_132/backward_lstm_132/CastCastQsequential_132/bidirectional_132/backward_lstm_132/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ29
7sequential_132/bidirectional_132/backward_lstm_132/Cast
8sequential_132/bidirectional_132/backward_lstm_132/ShapeShape_sequential_132/bidirectional_132/backward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2:
8sequential_132/bidirectional_132/backward_lstm_132/Shapeк
Fsequential_132/bidirectional_132/backward_lstm_132/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fsequential_132/bidirectional_132/backward_lstm_132/strided_slice/stackо
Hsequential_132/bidirectional_132/backward_lstm_132/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_132/bidirectional_132/backward_lstm_132/strided_slice/stack_1о
Hsequential_132/bidirectional_132/backward_lstm_132/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_132/bidirectional_132/backward_lstm_132/strided_slice/stack_2
@sequential_132/bidirectional_132/backward_lstm_132/strided_sliceStridedSliceAsequential_132/bidirectional_132/backward_lstm_132/Shape:output:0Osequential_132/bidirectional_132/backward_lstm_132/strided_slice/stack:output:0Qsequential_132/bidirectional_132/backward_lstm_132/strided_slice/stack_1:output:0Qsequential_132/bidirectional_132/backward_lstm_132/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2B
@sequential_132/bidirectional_132/backward_lstm_132/strided_sliceТ
>sequential_132/bidirectional_132/backward_lstm_132/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22@
>sequential_132/bidirectional_132/backward_lstm_132/zeros/mul/yИ
<sequential_132/bidirectional_132/backward_lstm_132/zeros/mulMulIsequential_132/bidirectional_132/backward_lstm_132/strided_slice:output:0Gsequential_132/bidirectional_132/backward_lstm_132/zeros/mul/y:output:0*
T0*
_output_shapes
: 2>
<sequential_132/bidirectional_132/backward_lstm_132/zeros/mulХ
?sequential_132/bidirectional_132/backward_lstm_132/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2A
?sequential_132/bidirectional_132/backward_lstm_132/zeros/Less/yГ
=sequential_132/bidirectional_132/backward_lstm_132/zeros/LessLess@sequential_132/bidirectional_132/backward_lstm_132/zeros/mul:z:0Hsequential_132/bidirectional_132/backward_lstm_132/zeros/Less/y:output:0*
T0*
_output_shapes
: 2?
=sequential_132/bidirectional_132/backward_lstm_132/zeros/LessШ
Asequential_132/bidirectional_132/backward_lstm_132/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22C
Asequential_132/bidirectional_132/backward_lstm_132/zeros/packed/1Я
?sequential_132/bidirectional_132/backward_lstm_132/zeros/packedPackIsequential_132/bidirectional_132/backward_lstm_132/strided_slice:output:0Jsequential_132/bidirectional_132/backward_lstm_132/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2A
?sequential_132/bidirectional_132/backward_lstm_132/zeros/packedЩ
>sequential_132/bidirectional_132/backward_lstm_132/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2@
>sequential_132/bidirectional_132/backward_lstm_132/zeros/ConstС
8sequential_132/bidirectional_132/backward_lstm_132/zerosFillHsequential_132/bidirectional_132/backward_lstm_132/zeros/packed:output:0Gsequential_132/bidirectional_132/backward_lstm_132/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22:
8sequential_132/bidirectional_132/backward_lstm_132/zerosЦ
@sequential_132/bidirectional_132/backward_lstm_132/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22B
@sequential_132/bidirectional_132/backward_lstm_132/zeros_1/mul/yО
>sequential_132/bidirectional_132/backward_lstm_132/zeros_1/mulMulIsequential_132/bidirectional_132/backward_lstm_132/strided_slice:output:0Isequential_132/bidirectional_132/backward_lstm_132/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2@
>sequential_132/bidirectional_132/backward_lstm_132/zeros_1/mulЩ
Asequential_132/bidirectional_132/backward_lstm_132/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2C
Asequential_132/bidirectional_132/backward_lstm_132/zeros_1/Less/yЛ
?sequential_132/bidirectional_132/backward_lstm_132/zeros_1/LessLessBsequential_132/bidirectional_132/backward_lstm_132/zeros_1/mul:z:0Jsequential_132/bidirectional_132/backward_lstm_132/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2A
?sequential_132/bidirectional_132/backward_lstm_132/zeros_1/LessЬ
Csequential_132/bidirectional_132/backward_lstm_132/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22E
Csequential_132/bidirectional_132/backward_lstm_132/zeros_1/packed/1е
Asequential_132/bidirectional_132/backward_lstm_132/zeros_1/packedPackIsequential_132/bidirectional_132/backward_lstm_132/strided_slice:output:0Lsequential_132/bidirectional_132/backward_lstm_132/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2C
Asequential_132/bidirectional_132/backward_lstm_132/zeros_1/packedЭ
@sequential_132/bidirectional_132/backward_lstm_132/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2B
@sequential_132/bidirectional_132/backward_lstm_132/zeros_1/ConstЩ
:sequential_132/bidirectional_132/backward_lstm_132/zeros_1FillJsequential_132/bidirectional_132/backward_lstm_132/zeros_1/packed:output:0Isequential_132/bidirectional_132/backward_lstm_132/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22<
:sequential_132/bidirectional_132/backward_lstm_132/zeros_1л
Asequential_132/bidirectional_132/backward_lstm_132/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2C
Asequential_132/bidirectional_132/backward_lstm_132/transpose/permѕ
<sequential_132/bidirectional_132/backward_lstm_132/transpose	Transpose_sequential_132/bidirectional_132/backward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0Jsequential_132/bidirectional_132/backward_lstm_132/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2>
<sequential_132/bidirectional_132/backward_lstm_132/transposeш
:sequential_132/bidirectional_132/backward_lstm_132/Shape_1Shape@sequential_132/bidirectional_132/backward_lstm_132/transpose:y:0*
T0*
_output_shapes
:2<
:sequential_132/bidirectional_132/backward_lstm_132/Shape_1о
Hsequential_132/bidirectional_132/backward_lstm_132/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_132/bidirectional_132/backward_lstm_132/strided_slice_1/stackт
Jsequential_132/bidirectional_132/backward_lstm_132/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_132/bidirectional_132/backward_lstm_132/strided_slice_1/stack_1т
Jsequential_132/bidirectional_132/backward_lstm_132/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_132/bidirectional_132/backward_lstm_132/strided_slice_1/stack_2 
Bsequential_132/bidirectional_132/backward_lstm_132/strided_slice_1StridedSliceCsequential_132/bidirectional_132/backward_lstm_132/Shape_1:output:0Qsequential_132/bidirectional_132/backward_lstm_132/strided_slice_1/stack:output:0Ssequential_132/bidirectional_132/backward_lstm_132/strided_slice_1/stack_1:output:0Ssequential_132/bidirectional_132/backward_lstm_132/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2D
Bsequential_132/bidirectional_132/backward_lstm_132/strided_slice_1ы
Nsequential_132/bidirectional_132/backward_lstm_132/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2P
Nsequential_132/bidirectional_132/backward_lstm_132/TensorArrayV2/element_shapeў
@sequential_132/bidirectional_132/backward_lstm_132/TensorArrayV2TensorListReserveWsequential_132/bidirectional_132/backward_lstm_132/TensorArrayV2/element_shape:output:0Ksequential_132/bidirectional_132/backward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02B
@sequential_132/bidirectional_132/backward_lstm_132/TensorArrayV2а
Asequential_132/bidirectional_132/backward_lstm_132/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential_132/bidirectional_132/backward_lstm_132/ReverseV2/axisж
<sequential_132/bidirectional_132/backward_lstm_132/ReverseV2	ReverseV2@sequential_132/bidirectional_132/backward_lstm_132/transpose:y:0Jsequential_132/bidirectional_132/backward_lstm_132/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2>
<sequential_132/bidirectional_132/backward_lstm_132/ReverseV2Ѕ
hsequential_132/bidirectional_132/backward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2j
hsequential_132/bidirectional_132/backward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeЩ
Zsequential_132/bidirectional_132/backward_lstm_132/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorEsequential_132/bidirectional_132/backward_lstm_132/ReverseV2:output:0qsequential_132/bidirectional_132/backward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02\
Zsequential_132/bidirectional_132/backward_lstm_132/TensorArrayUnstack/TensorListFromTensorо
Hsequential_132/bidirectional_132/backward_lstm_132/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_132/bidirectional_132/backward_lstm_132/strided_slice_2/stackт
Jsequential_132/bidirectional_132/backward_lstm_132/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_132/bidirectional_132/backward_lstm_132/strided_slice_2/stack_1т
Jsequential_132/bidirectional_132/backward_lstm_132/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_132/bidirectional_132/backward_lstm_132/strided_slice_2/stack_2Ў
Bsequential_132/bidirectional_132/backward_lstm_132/strided_slice_2StridedSlice@sequential_132/bidirectional_132/backward_lstm_132/transpose:y:0Qsequential_132/bidirectional_132/backward_lstm_132/strided_slice_2/stack:output:0Ssequential_132/bidirectional_132/backward_lstm_132/strided_slice_2/stack_1:output:0Ssequential_132/bidirectional_132/backward_lstm_132/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2D
Bsequential_132/bidirectional_132/backward_lstm_132/strided_slice_2б
Vsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpReadVariableOp_sequential_132_bidirectional_132_backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02X
Vsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpќ
Gsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMulMatMulKsequential_132/bidirectional_132/backward_lstm_132/strided_slice_2:output:0^sequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2I
Gsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMulз
Xsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOpasequential_132_bidirectional_132_backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02Z
Xsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpј
Isequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMul_1MatMulAsequential_132/bidirectional_132/backward_lstm_132/zeros:output:0`sequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2K
Isequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMul_1№
Dsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/addAddV2Qsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMul:product:0Ssequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2F
Dsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/addа
Wsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOp`sequential_132_bidirectional_132_backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02Y
Wsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp§
Hsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/BiasAddBiasAddHsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/add:z:0_sequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2J
Hsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/BiasAddц
Psequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2R
Psequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/split/split_dimУ
Fsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/splitSplitYsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/split/split_dim:output:0Qsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2H
Fsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/splitЂ
Hsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/SigmoidSigmoidOsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22J
Hsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/SigmoidІ
Jsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/Sigmoid_1SigmoidOsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22L
Jsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/Sigmoid_1к
Dsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/mulMulNsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/Sigmoid_1:y:0Csequential_132/bidirectional_132/backward_lstm_132/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22F
Dsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/mul
Esequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/ReluReluOsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22G
Esequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/Reluь
Fsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/mul_1MulLsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/Sigmoid:y:0Ssequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22H
Fsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/mul_1с
Fsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/add_1AddV2Hsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/mul:z:0Jsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22H
Fsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/add_1І
Jsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/Sigmoid_2SigmoidOsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22L
Jsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/Sigmoid_2
Gsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/Relu_1ReluJsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22I
Gsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/Relu_1№
Fsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/mul_2MulNsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/Sigmoid_2:y:0Usequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22H
Fsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/mul_2ѕ
Psequential_132/bidirectional_132/backward_lstm_132/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2R
Psequential_132/bidirectional_132/backward_lstm_132/TensorArrayV2_1/element_shape
Bsequential_132/bidirectional_132/backward_lstm_132/TensorArrayV2_1TensorListReserveYsequential_132/bidirectional_132/backward_lstm_132/TensorArrayV2_1/element_shape:output:0Ksequential_132/bidirectional_132/backward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02D
Bsequential_132/bidirectional_132/backward_lstm_132/TensorArrayV2_1Д
7sequential_132/bidirectional_132/backward_lstm_132/timeConst*
_output_shapes
: *
dtype0*
value	B : 29
7sequential_132/bidirectional_132/backward_lstm_132/timeж
Hsequential_132/bidirectional_132/backward_lstm_132/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hsequential_132/bidirectional_132/backward_lstm_132/Max/reduction_indicesЈ
6sequential_132/bidirectional_132/backward_lstm_132/MaxMax;sequential_132/bidirectional_132/backward_lstm_132/Cast:y:0Qsequential_132/bidirectional_132/backward_lstm_132/Max/reduction_indices:output:0*
T0*
_output_shapes
: 28
6sequential_132/bidirectional_132/backward_lstm_132/MaxЖ
8sequential_132/bidirectional_132/backward_lstm_132/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_132/bidirectional_132/backward_lstm_132/sub/y
6sequential_132/bidirectional_132/backward_lstm_132/subSub?sequential_132/bidirectional_132/backward_lstm_132/Max:output:0Asequential_132/bidirectional_132/backward_lstm_132/sub/y:output:0*
T0*
_output_shapes
: 28
6sequential_132/bidirectional_132/backward_lstm_132/subЂ
8sequential_132/bidirectional_132/backward_lstm_132/Sub_1Sub:sequential_132/bidirectional_132/backward_lstm_132/sub:z:0;sequential_132/bidirectional_132/backward_lstm_132/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2:
8sequential_132/bidirectional_132/backward_lstm_132/Sub_1
=sequential_132/bidirectional_132/backward_lstm_132/zeros_like	ZerosLikeJsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22?
=sequential_132/bidirectional_132/backward_lstm_132/zeros_likeх
Ksequential_132/bidirectional_132/backward_lstm_132/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2M
Ksequential_132/bidirectional_132/backward_lstm_132/while/maximum_iterationsа
Esequential_132/bidirectional_132/backward_lstm_132/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2G
Esequential_132/bidirectional_132/backward_lstm_132/while/loop_counterж
8sequential_132/bidirectional_132/backward_lstm_132/whileWhileNsequential_132/bidirectional_132/backward_lstm_132/while/loop_counter:output:0Tsequential_132/bidirectional_132/backward_lstm_132/while/maximum_iterations:output:0@sequential_132/bidirectional_132/backward_lstm_132/time:output:0Ksequential_132/bidirectional_132/backward_lstm_132/TensorArrayV2_1:handle:0Asequential_132/bidirectional_132/backward_lstm_132/zeros_like:y:0Asequential_132/bidirectional_132/backward_lstm_132/zeros:output:0Csequential_132/bidirectional_132/backward_lstm_132/zeros_1:output:0Ksequential_132/bidirectional_132/backward_lstm_132/strided_slice_1:output:0jsequential_132/bidirectional_132/backward_lstm_132/TensorArrayUnstack/TensorListFromTensor:output_handle:0<sequential_132/bidirectional_132/backward_lstm_132/Sub_1:z:0_sequential_132_bidirectional_132_backward_lstm_132_lstm_cell_398_matmul_readvariableop_resourceasequential_132_bidirectional_132_backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource`sequential_132_bidirectional_132_backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource*
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
Fsequential_132_bidirectional_132_backward_lstm_132_while_body_16713494*R
condJRH
Fsequential_132_bidirectional_132_backward_lstm_132_while_cond_16713493*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2:
8sequential_132/bidirectional_132/backward_lstm_132/while
csequential_132/bidirectional_132/backward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2e
csequential_132/bidirectional_132/backward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeН
Usequential_132/bidirectional_132/backward_lstm_132/TensorArrayV2Stack/TensorListStackTensorListStackAsequential_132/bidirectional_132/backward_lstm_132/while:output:3lsequential_132/bidirectional_132/backward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02W
Usequential_132/bidirectional_132/backward_lstm_132/TensorArrayV2Stack/TensorListStackч
Hsequential_132/bidirectional_132/backward_lstm_132/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2J
Hsequential_132/bidirectional_132/backward_lstm_132/strided_slice_3/stackт
Jsequential_132/bidirectional_132/backward_lstm_132/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jsequential_132/bidirectional_132/backward_lstm_132/strided_slice_3/stack_1т
Jsequential_132/bidirectional_132/backward_lstm_132/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_132/bidirectional_132/backward_lstm_132/strided_slice_3/stack_2Ь
Bsequential_132/bidirectional_132/backward_lstm_132/strided_slice_3StridedSlice^sequential_132/bidirectional_132/backward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0Qsequential_132/bidirectional_132/backward_lstm_132/strided_slice_3/stack:output:0Ssequential_132/bidirectional_132/backward_lstm_132/strided_slice_3/stack_1:output:0Ssequential_132/bidirectional_132/backward_lstm_132/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2D
Bsequential_132/bidirectional_132/backward_lstm_132/strided_slice_3п
Csequential_132/bidirectional_132/backward_lstm_132/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2E
Csequential_132/bidirectional_132/backward_lstm_132/transpose_1/permњ
>sequential_132/bidirectional_132/backward_lstm_132/transpose_1	Transpose^sequential_132/bidirectional_132/backward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0Lsequential_132/bidirectional_132/backward_lstm_132/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22@
>sequential_132/bidirectional_132/backward_lstm_132/transpose_1Ь
:sequential_132/bidirectional_132/backward_lstm_132/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2<
:sequential_132/bidirectional_132/backward_lstm_132/runtime
,sequential_132/bidirectional_132/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_132/bidirectional_132/concat/axisщ
'sequential_132/bidirectional_132/concatConcatV2Jsequential_132/bidirectional_132/forward_lstm_132/strided_slice_3:output:0Ksequential_132/bidirectional_132/backward_lstm_132/strided_slice_3:output:05sequential_132/bidirectional_132/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2)
'sequential_132/bidirectional_132/concatи
.sequential_132/dense_132/MatMul/ReadVariableOpReadVariableOp7sequential_132_dense_132_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_132/dense_132/MatMul/ReadVariableOpш
sequential_132/dense_132/MatMulMatMul0sequential_132/bidirectional_132/concat:output:06sequential_132/dense_132/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
sequential_132/dense_132/MatMulз
/sequential_132/dense_132/BiasAdd/ReadVariableOpReadVariableOp8sequential_132_dense_132_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_132/dense_132/BiasAdd/ReadVariableOpх
 sequential_132/dense_132/BiasAddBiasAdd)sequential_132/dense_132/MatMul:product:07sequential_132/dense_132/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 sequential_132/dense_132/BiasAddЌ
 sequential_132/dense_132/SigmoidSigmoid)sequential_132/dense_132/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 sequential_132/dense_132/Sigmoid
IdentityIdentity$sequential_132/dense_132/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityП
NoOpNoOpX^sequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpW^sequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpY^sequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp9^sequential_132/bidirectional_132/backward_lstm_132/whileW^sequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpV^sequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpX^sequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp8^sequential_132/bidirectional_132/forward_lstm_132/while0^sequential_132/dense_132/BiasAdd/ReadVariableOp/^sequential_132/dense_132/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2В
Wsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpWsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp2А
Vsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpVsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp2Д
Xsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpXsequential_132/bidirectional_132/backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp2t
8sequential_132/bidirectional_132/backward_lstm_132/while8sequential_132/bidirectional_132/backward_lstm_132/while2А
Vsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpVsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp2Ў
Usequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpUsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp2В
Wsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOpWsequential_132/bidirectional_132/forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp2r
7sequential_132/bidirectional_132/forward_lstm_132/while7sequential_132/bidirectional_132/forward_lstm_132/while2b
/sequential_132/dense_132/BiasAdd/ReadVariableOp/sequential_132/dense_132/BiasAdd/ReadVariableOp2`
.sequential_132/dense_132/MatMul/ReadVariableOp.sequential_132/dense_132/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameargs_0:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameargs_0
аf
ф
%backward_lstm_132_while_body_16717947@
<backward_lstm_132_while_backward_lstm_132_while_loop_counterF
Bbackward_lstm_132_while_backward_lstm_132_while_maximum_iterations'
#backward_lstm_132_while_placeholder)
%backward_lstm_132_while_placeholder_1)
%backward_lstm_132_while_placeholder_2)
%backward_lstm_132_while_placeholder_3)
%backward_lstm_132_while_placeholder_4?
;backward_lstm_132_while_backward_lstm_132_strided_slice_1_0{
wbackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_132_while_less_backward_lstm_132_sub_1_0Y
Fbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0:	Ш[
Hbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0:	2ШV
Gbackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0:	Ш$
 backward_lstm_132_while_identity&
"backward_lstm_132_while_identity_1&
"backward_lstm_132_while_identity_2&
"backward_lstm_132_while_identity_3&
"backward_lstm_132_while_identity_4&
"backward_lstm_132_while_identity_5&
"backward_lstm_132_while_identity_6=
9backward_lstm_132_while_backward_lstm_132_strided_slice_1y
ubackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_132_while_less_backward_lstm_132_sub_1W
Dbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource:	ШY
Fbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource:	2ШT
Ebackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource:	ШЂ<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOpЂ;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpЂ=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOpч
Ibackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2K
Ibackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeП
;backward_lstm_132/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_132_while_placeholderRbackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02=
;backward_lstm_132/while/TensorArrayV2Read/TensorListGetItemЯ
backward_lstm_132/while/LessLess6backward_lstm_132_while_less_backward_lstm_132_sub_1_0#backward_lstm_132_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_132/while/Less
;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpReadVariableOpFbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02=
;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpЂ
,backward_lstm_132/while/lstm_cell_398/MatMulMatMulBbackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem:item:0Cbackward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_132/while/lstm_cell_398/MatMul
=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOpHbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02?
=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp
.backward_lstm_132/while/lstm_cell_398/MatMul_1MatMul%backward_lstm_132_while_placeholder_3Ebackward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ20
.backward_lstm_132/while/lstm_cell_398/MatMul_1
)backward_lstm_132/while/lstm_cell_398/addAddV26backward_lstm_132/while/lstm_cell_398/MatMul:product:08backward_lstm_132/while/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_132/while/lstm_cell_398/add
<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOpGbackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02>
<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp
-backward_lstm_132/while/lstm_cell_398/BiasAddBiasAdd-backward_lstm_132/while/lstm_cell_398/add:z:0Dbackward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_132/while/lstm_cell_398/BiasAddА
5backward_lstm_132/while/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5backward_lstm_132/while/lstm_cell_398/split/split_dimз
+backward_lstm_132/while/lstm_cell_398/splitSplit>backward_lstm_132/while/lstm_cell_398/split/split_dim:output:06backward_lstm_132/while/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2-
+backward_lstm_132/while/lstm_cell_398/splitб
-backward_lstm_132/while/lstm_cell_398/SigmoidSigmoid4backward_lstm_132/while/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22/
-backward_lstm_132/while/lstm_cell_398/Sigmoidе
/backward_lstm_132/while/lstm_cell_398/Sigmoid_1Sigmoid4backward_lstm_132/while/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_132/while/lstm_cell_398/Sigmoid_1ы
)backward_lstm_132/while/lstm_cell_398/mulMul3backward_lstm_132/while/lstm_cell_398/Sigmoid_1:y:0%backward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_132/while/lstm_cell_398/mulШ
*backward_lstm_132/while/lstm_cell_398/ReluRelu4backward_lstm_132/while/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_132/while/lstm_cell_398/Relu
+backward_lstm_132/while/lstm_cell_398/mul_1Mul1backward_lstm_132/while/lstm_cell_398/Sigmoid:y:08backward_lstm_132/while/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_132/while/lstm_cell_398/mul_1ѕ
+backward_lstm_132/while/lstm_cell_398/add_1AddV2-backward_lstm_132/while/lstm_cell_398/mul:z:0/backward_lstm_132/while/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_132/while/lstm_cell_398/add_1е
/backward_lstm_132/while/lstm_cell_398/Sigmoid_2Sigmoid4backward_lstm_132/while/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_132/while/lstm_cell_398/Sigmoid_2Ч
,backward_lstm_132/while/lstm_cell_398/Relu_1Relu/backward_lstm_132/while/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_132/while/lstm_cell_398/Relu_1
+backward_lstm_132/while/lstm_cell_398/mul_2Mul3backward_lstm_132/while/lstm_cell_398/Sigmoid_2:y:0:backward_lstm_132/while/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_132/while/lstm_cell_398/mul_2і
backward_lstm_132/while/SelectSelect backward_lstm_132/while/Less:z:0/backward_lstm_132/while/lstm_cell_398/mul_2:z:0%backward_lstm_132_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_132/while/Selectњ
 backward_lstm_132/while/Select_1Select backward_lstm_132/while/Less:z:0/backward_lstm_132/while/lstm_cell_398/mul_2:z:0%backward_lstm_132_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_132/while/Select_1њ
 backward_lstm_132/while/Select_2Select backward_lstm_132/while/Less:z:0/backward_lstm_132/while/lstm_cell_398/add_1:z:0%backward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_132/while/Select_2Г
<backward_lstm_132/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_132_while_placeholder_1#backward_lstm_132_while_placeholder'backward_lstm_132/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_132/while/TensorArrayV2Write/TensorListSetItem
backward_lstm_132/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_132/while/add/yБ
backward_lstm_132/while/addAddV2#backward_lstm_132_while_placeholder&backward_lstm_132/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/while/add
backward_lstm_132/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_132/while/add_1/yа
backward_lstm_132/while/add_1AddV2<backward_lstm_132_while_backward_lstm_132_while_loop_counter(backward_lstm_132/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/while/add_1Г
 backward_lstm_132/while/IdentityIdentity!backward_lstm_132/while/add_1:z:0^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_132/while/Identityи
"backward_lstm_132/while/Identity_1IdentityBbackward_lstm_132_while_backward_lstm_132_while_maximum_iterations^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_132/while/Identity_1Е
"backward_lstm_132/while/Identity_2Identitybackward_lstm_132/while/add:z:0^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_132/while/Identity_2т
"backward_lstm_132/while/Identity_3IdentityLbackward_lstm_132/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_132/while/Identity_3Ю
"backward_lstm_132/while/Identity_4Identity'backward_lstm_132/while/Select:output:0^backward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_132/while/Identity_4а
"backward_lstm_132/while/Identity_5Identity)backward_lstm_132/while/Select_1:output:0^backward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_132/while/Identity_5а
"backward_lstm_132/while/Identity_6Identity)backward_lstm_132/while/Select_2:output:0^backward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_132/while/Identity_6Л
backward_lstm_132/while/NoOpNoOp=^backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp<^backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp>^backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_132/while/NoOp"x
9backward_lstm_132_while_backward_lstm_132_strided_slice_1;backward_lstm_132_while_backward_lstm_132_strided_slice_1_0"M
 backward_lstm_132_while_identity)backward_lstm_132/while/Identity:output:0"Q
"backward_lstm_132_while_identity_1+backward_lstm_132/while/Identity_1:output:0"Q
"backward_lstm_132_while_identity_2+backward_lstm_132/while/Identity_2:output:0"Q
"backward_lstm_132_while_identity_3+backward_lstm_132/while/Identity_3:output:0"Q
"backward_lstm_132_while_identity_4+backward_lstm_132/while/Identity_4:output:0"Q
"backward_lstm_132_while_identity_5+backward_lstm_132/while/Identity_5:output:0"Q
"backward_lstm_132_while_identity_6+backward_lstm_132/while/Identity_6:output:0"n
4backward_lstm_132_while_less_backward_lstm_132_sub_16backward_lstm_132_while_less_backward_lstm_132_sub_1_0"
Ebackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resourceGbackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0"
Fbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resourceHbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0"
Dbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resourceFbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0"№
ubackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2|
<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp2z
;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp2~
=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
ї

,__inference_dense_132_layer_call_fn_16718053

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
G__inference_dense_132_layer_call_and_return_conditional_losses_167160592
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
У
Џ
Esequential_132_bidirectional_132_forward_lstm_132_while_body_16713315
|sequential_132_bidirectional_132_forward_lstm_132_while_sequential_132_bidirectional_132_forward_lstm_132_while_loop_counter
sequential_132_bidirectional_132_forward_lstm_132_while_sequential_132_bidirectional_132_forward_lstm_132_while_maximum_iterationsG
Csequential_132_bidirectional_132_forward_lstm_132_while_placeholderI
Esequential_132_bidirectional_132_forward_lstm_132_while_placeholder_1I
Esequential_132_bidirectional_132_forward_lstm_132_while_placeholder_2I
Esequential_132_bidirectional_132_forward_lstm_132_while_placeholder_3I
Esequential_132_bidirectional_132_forward_lstm_132_while_placeholder_4
{sequential_132_bidirectional_132_forward_lstm_132_while_sequential_132_bidirectional_132_forward_lstm_132_strided_slice_1_0М
Зsequential_132_bidirectional_132_forward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_sequential_132_bidirectional_132_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0|
xsequential_132_bidirectional_132_forward_lstm_132_while_greater_sequential_132_bidirectional_132_forward_lstm_132_cast_0y
fsequential_132_bidirectional_132_forward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0:	Ш{
hsequential_132_bidirectional_132_forward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0:	2Шv
gsequential_132_bidirectional_132_forward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0:	ШD
@sequential_132_bidirectional_132_forward_lstm_132_while_identityF
Bsequential_132_bidirectional_132_forward_lstm_132_while_identity_1F
Bsequential_132_bidirectional_132_forward_lstm_132_while_identity_2F
Bsequential_132_bidirectional_132_forward_lstm_132_while_identity_3F
Bsequential_132_bidirectional_132_forward_lstm_132_while_identity_4F
Bsequential_132_bidirectional_132_forward_lstm_132_while_identity_5F
Bsequential_132_bidirectional_132_forward_lstm_132_while_identity_6}
ysequential_132_bidirectional_132_forward_lstm_132_while_sequential_132_bidirectional_132_forward_lstm_132_strided_slice_1К
Еsequential_132_bidirectional_132_forward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_sequential_132_bidirectional_132_forward_lstm_132_tensorarrayunstack_tensorlistfromtensorz
vsequential_132_bidirectional_132_forward_lstm_132_while_greater_sequential_132_bidirectional_132_forward_lstm_132_castw
dsequential_132_bidirectional_132_forward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource:	Шy
fsequential_132_bidirectional_132_forward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource:	2Шt
esequential_132_bidirectional_132_forward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource:	ШЂ\sequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOpЂ[sequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOpЂ]sequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOpЇ
isequential_132/bidirectional_132/forward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2k
isequential_132/bidirectional_132/forward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shape
[sequential_132/bidirectional_132/forward_lstm_132/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЗsequential_132_bidirectional_132_forward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_sequential_132_bidirectional_132_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0Csequential_132_bidirectional_132_forward_lstm_132_while_placeholderrsequential_132/bidirectional_132/forward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02]
[sequential_132/bidirectional_132/forward_lstm_132/while/TensorArrayV2Read/TensorListGetItemњ
?sequential_132/bidirectional_132/forward_lstm_132/while/GreaterGreaterxsequential_132_bidirectional_132_forward_lstm_132_while_greater_sequential_132_bidirectional_132_forward_lstm_132_cast_0Csequential_132_bidirectional_132_forward_lstm_132_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2A
?sequential_132/bidirectional_132/forward_lstm_132/while/Greaterт
[sequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOpReadVariableOpfsequential_132_bidirectional_132_forward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02]
[sequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOpЂ
Lsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMulMatMulbsequential_132/bidirectional_132/forward_lstm_132/while/TensorArrayV2Read/TensorListGetItem:item:0csequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2N
Lsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMulш
]sequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOphsequential_132_bidirectional_132_forward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02_
]sequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp
Nsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMul_1MatMulEsequential_132_bidirectional_132_forward_lstm_132_while_placeholder_3esequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2P
Nsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMul_1
Isequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/addAddV2Vsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMul:product:0Xsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2K
Isequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/addс
\sequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOpgsequential_132_bidirectional_132_forward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02^
\sequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp
Msequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/BiasAddBiasAddMsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/add:z:0dsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2O
Msequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/BiasAdd№
Usequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2W
Usequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/split/split_dimз
Ksequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/splitSplit^sequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/split/split_dim:output:0Vsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2M
Ksequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/splitБ
Msequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/SigmoidSigmoidTsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22O
Msequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/SigmoidЕ
Osequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/Sigmoid_1SigmoidTsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22Q
Osequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/Sigmoid_1ы
Isequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/mulMulSsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/Sigmoid_1:y:0Esequential_132_bidirectional_132_forward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22K
Isequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/mulЈ
Jsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/ReluReluTsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22L
Jsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/Relu
Ksequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/mul_1MulQsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/Sigmoid:y:0Xsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22M
Ksequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/mul_1ѕ
Ksequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/add_1AddV2Msequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/mul:z:0Osequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22M
Ksequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/add_1Е
Osequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/Sigmoid_2SigmoidTsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22Q
Osequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/Sigmoid_2Ї
Lsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/Relu_1ReluOsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22N
Lsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/Relu_1
Ksequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/mul_2MulSsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/Sigmoid_2:y:0Zsequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22M
Ksequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/mul_2
>sequential_132/bidirectional_132/forward_lstm_132/while/SelectSelectCsequential_132/bidirectional_132/forward_lstm_132/while/Greater:z:0Osequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/mul_2:z:0Esequential_132_bidirectional_132_forward_lstm_132_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22@
>sequential_132/bidirectional_132/forward_lstm_132/while/Select
@sequential_132/bidirectional_132/forward_lstm_132/while/Select_1SelectCsequential_132/bidirectional_132/forward_lstm_132/while/Greater:z:0Osequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/mul_2:z:0Esequential_132_bidirectional_132_forward_lstm_132_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22B
@sequential_132/bidirectional_132/forward_lstm_132/while/Select_1
@sequential_132/bidirectional_132/forward_lstm_132/while/Select_2SelectCsequential_132/bidirectional_132/forward_lstm_132/while/Greater:z:0Osequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/add_1:z:0Esequential_132_bidirectional_132_forward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22B
@sequential_132/bidirectional_132/forward_lstm_132/while/Select_2г
\sequential_132/bidirectional_132/forward_lstm_132/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemEsequential_132_bidirectional_132_forward_lstm_132_while_placeholder_1Csequential_132_bidirectional_132_forward_lstm_132_while_placeholderGsequential_132/bidirectional_132/forward_lstm_132/while/Select:output:0*
_output_shapes
: *
element_dtype02^
\sequential_132/bidirectional_132/forward_lstm_132/while/TensorArrayV2Write/TensorListSetItemР
=sequential_132/bidirectional_132/forward_lstm_132/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2?
=sequential_132/bidirectional_132/forward_lstm_132/while/add/yБ
;sequential_132/bidirectional_132/forward_lstm_132/while/addAddV2Csequential_132_bidirectional_132_forward_lstm_132_while_placeholderFsequential_132/bidirectional_132/forward_lstm_132/while/add/y:output:0*
T0*
_output_shapes
: 2=
;sequential_132/bidirectional_132/forward_lstm_132/while/addФ
?sequential_132/bidirectional_132/forward_lstm_132/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2A
?sequential_132/bidirectional_132/forward_lstm_132/while/add_1/y№
=sequential_132/bidirectional_132/forward_lstm_132/while/add_1AddV2|sequential_132_bidirectional_132_forward_lstm_132_while_sequential_132_bidirectional_132_forward_lstm_132_while_loop_counterHsequential_132/bidirectional_132/forward_lstm_132/while/add_1/y:output:0*
T0*
_output_shapes
: 2?
=sequential_132/bidirectional_132/forward_lstm_132/while/add_1Г
@sequential_132/bidirectional_132/forward_lstm_132/while/IdentityIdentityAsequential_132/bidirectional_132/forward_lstm_132/while/add_1:z:0=^sequential_132/bidirectional_132/forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2B
@sequential_132/bidirectional_132/forward_lstm_132/while/Identityљ
Bsequential_132/bidirectional_132/forward_lstm_132/while/Identity_1Identitysequential_132_bidirectional_132_forward_lstm_132_while_sequential_132_bidirectional_132_forward_lstm_132_while_maximum_iterations=^sequential_132/bidirectional_132/forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2D
Bsequential_132/bidirectional_132/forward_lstm_132/while/Identity_1Е
Bsequential_132/bidirectional_132/forward_lstm_132/while/Identity_2Identity?sequential_132/bidirectional_132/forward_lstm_132/while/add:z:0=^sequential_132/bidirectional_132/forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2D
Bsequential_132/bidirectional_132/forward_lstm_132/while/Identity_2т
Bsequential_132/bidirectional_132/forward_lstm_132/while/Identity_3Identitylsequential_132/bidirectional_132/forward_lstm_132/while/TensorArrayV2Write/TensorListSetItem:output_handle:0=^sequential_132/bidirectional_132/forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2D
Bsequential_132/bidirectional_132/forward_lstm_132/while/Identity_3Ю
Bsequential_132/bidirectional_132/forward_lstm_132/while/Identity_4IdentityGsequential_132/bidirectional_132/forward_lstm_132/while/Select:output:0=^sequential_132/bidirectional_132/forward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22D
Bsequential_132/bidirectional_132/forward_lstm_132/while/Identity_4а
Bsequential_132/bidirectional_132/forward_lstm_132/while/Identity_5IdentityIsequential_132/bidirectional_132/forward_lstm_132/while/Select_1:output:0=^sequential_132/bidirectional_132/forward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22D
Bsequential_132/bidirectional_132/forward_lstm_132/while/Identity_5а
Bsequential_132/bidirectional_132/forward_lstm_132/while/Identity_6IdentityIsequential_132/bidirectional_132/forward_lstm_132/while/Select_2:output:0=^sequential_132/bidirectional_132/forward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22D
Bsequential_132/bidirectional_132/forward_lstm_132/while/Identity_6л
<sequential_132/bidirectional_132/forward_lstm_132/while/NoOpNoOp]^sequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp\^sequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp^^sequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2>
<sequential_132/bidirectional_132/forward_lstm_132/while/NoOp"ђ
vsequential_132_bidirectional_132_forward_lstm_132_while_greater_sequential_132_bidirectional_132_forward_lstm_132_castxsequential_132_bidirectional_132_forward_lstm_132_while_greater_sequential_132_bidirectional_132_forward_lstm_132_cast_0"
@sequential_132_bidirectional_132_forward_lstm_132_while_identityIsequential_132/bidirectional_132/forward_lstm_132/while/Identity:output:0"
Bsequential_132_bidirectional_132_forward_lstm_132_while_identity_1Ksequential_132/bidirectional_132/forward_lstm_132/while/Identity_1:output:0"
Bsequential_132_bidirectional_132_forward_lstm_132_while_identity_2Ksequential_132/bidirectional_132/forward_lstm_132/while/Identity_2:output:0"
Bsequential_132_bidirectional_132_forward_lstm_132_while_identity_3Ksequential_132/bidirectional_132/forward_lstm_132/while/Identity_3:output:0"
Bsequential_132_bidirectional_132_forward_lstm_132_while_identity_4Ksequential_132/bidirectional_132/forward_lstm_132/while/Identity_4:output:0"
Bsequential_132_bidirectional_132_forward_lstm_132_while_identity_5Ksequential_132/bidirectional_132/forward_lstm_132/while/Identity_5:output:0"
Bsequential_132_bidirectional_132_forward_lstm_132_while_identity_6Ksequential_132/bidirectional_132/forward_lstm_132/while/Identity_6:output:0"а
esequential_132_bidirectional_132_forward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resourcegsequential_132_bidirectional_132_forward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0"в
fsequential_132_bidirectional_132_forward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resourcehsequential_132_bidirectional_132_forward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0"Ю
dsequential_132_bidirectional_132_forward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resourcefsequential_132_bidirectional_132_forward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0"ј
ysequential_132_bidirectional_132_forward_lstm_132_while_sequential_132_bidirectional_132_forward_lstm_132_strided_slice_1{sequential_132_bidirectional_132_forward_lstm_132_while_sequential_132_bidirectional_132_forward_lstm_132_strided_slice_1_0"ђ
Еsequential_132_bidirectional_132_forward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_sequential_132_bidirectional_132_forward_lstm_132_tensorarrayunstack_tensorlistfromtensorЗsequential_132_bidirectional_132_forward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_sequential_132_bidirectional_132_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2М
\sequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp\sequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp2К
[sequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp[sequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp2О
]sequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp]sequential_132/bidirectional_132/forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
ќ

и
1__inference_sequential_132_layer_call_fn_16716578

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
L__inference_sequential_132_layer_call_and_return_conditional_losses_167165372
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
И
Ѓ
L__inference_sequential_132_layer_call_and_return_conditional_losses_16716066

inputs
inputs_1	-
bidirectional_132_16716035:	Ш-
bidirectional_132_16716037:	2Ш)
bidirectional_132_16716039:	Ш-
bidirectional_132_16716041:	Ш-
bidirectional_132_16716043:	2Ш)
bidirectional_132_16716045:	Ш$
dense_132_16716060:d 
dense_132_16716062:
identityЂ)bidirectional_132/StatefulPartitionedCallЂ!dense_132/StatefulPartitionedCallЪ
)bidirectional_132/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_132_16716035bidirectional_132_16716037bidirectional_132_16716039bidirectional_132_16716041bidirectional_132_16716043bidirectional_132_16716045*
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
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_167160342+
)bidirectional_132/StatefulPartitionedCallЫ
!dense_132/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_132/StatefulPartitionedCall:output:0dense_132_16716060dense_132_16716062*
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
G__inference_dense_132_layer_call_and_return_conditional_losses_167160592#
!dense_132/StatefulPartitionedCall
IdentityIdentity*dense_132/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp*^bidirectional_132/StatefulPartitionedCall"^dense_132/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2V
)bidirectional_132/StatefulPartitionedCall)bidirectional_132/StatefulPartitionedCall2F
!dense_132/StatefulPartitionedCall!dense_132/StatefulPartitionedCall:O K
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
while_cond_16713896
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16713896___redundant_placeholder06
2while_while_cond_16713896___redundant_placeholder16
2while_while_cond_16713896___redundant_placeholder26
2while_while_cond_16713896___redundant_placeholder3
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
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_16715548

inputs?
,lstm_cell_397_matmul_readvariableop_resource:	ШA
.lstm_cell_397_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_397_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_397/BiasAdd/ReadVariableOpЂ#lstm_cell_397/MatMul/ReadVariableOpЂ%lstm_cell_397/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_397/MatMul/ReadVariableOpReadVariableOp,lstm_cell_397_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_397/MatMul/ReadVariableOpА
lstm_cell_397/MatMulMatMulstrided_slice_2:output:0+lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/MatMulО
%lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_397_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_397/MatMul_1/ReadVariableOpЌ
lstm_cell_397/MatMul_1MatMulzeros:output:0-lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/MatMul_1Є
lstm_cell_397/addAddV2lstm_cell_397/MatMul:product:0 lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/addЗ
$lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_397_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_397/BiasAdd/ReadVariableOpБ
lstm_cell_397/BiasAddBiasAddlstm_cell_397/add:z:0,lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/BiasAdd
lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_397/split/split_dimї
lstm_cell_397/splitSplit&lstm_cell_397/split/split_dim:output:0lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_397/split
lstm_cell_397/SigmoidSigmoidlstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Sigmoid
lstm_cell_397/Sigmoid_1Sigmoidlstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Sigmoid_1
lstm_cell_397/mulMullstm_cell_397/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/mul
lstm_cell_397/ReluRelulstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Relu 
lstm_cell_397/mul_1Mullstm_cell_397/Sigmoid:y:0 lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/mul_1
lstm_cell_397/add_1AddV2lstm_cell_397/mul:z:0lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/add_1
lstm_cell_397/Sigmoid_2Sigmoidlstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Sigmoid_2
lstm_cell_397/Relu_1Relulstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Relu_1Є
lstm_cell_397/mul_2Mullstm_cell_397/Sigmoid_2:y:0"lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_397_matmul_readvariableop_resource.lstm_cell_397_matmul_1_readvariableop_resource-lstm_cell_397_biasadd_readvariableop_resource*
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
while_body_16715464*
condR
while_cond_16715463*K
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
NoOpNoOp%^lstm_cell_397/BiasAdd/ReadVariableOp$^lstm_cell_397/MatMul/ReadVariableOp&^lstm_cell_397/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_397/BiasAdd/ReadVariableOp$lstm_cell_397/BiasAdd/ReadVariableOp2J
#lstm_cell_397/MatMul/ReadVariableOp#lstm_cell_397/MatMul/ReadVariableOp2N
%lstm_cell_397/MatMul_1/ReadVariableOp%lstm_cell_397/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
п
Э
while_cond_16713686
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16713686___redundant_placeholder06
2while_while_cond_16713686___redundant_placeholder16
2while_while_cond_16713686___redundant_placeholder26
2while_while_cond_16713686___redundant_placeholder3
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
І_
­
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_16719215

inputs?
,lstm_cell_398_matmul_readvariableop_resource:	ШA
.lstm_cell_398_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_398_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_398/BiasAdd/ReadVariableOpЂ#lstm_cell_398/MatMul/ReadVariableOpЂ%lstm_cell_398/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_398/MatMul/ReadVariableOpReadVariableOp,lstm_cell_398_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_398/MatMul/ReadVariableOpА
lstm_cell_398/MatMulMatMulstrided_slice_2:output:0+lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/MatMulО
%lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_398_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_398/MatMul_1/ReadVariableOpЌ
lstm_cell_398/MatMul_1MatMulzeros:output:0-lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/MatMul_1Є
lstm_cell_398/addAddV2lstm_cell_398/MatMul:product:0 lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/addЗ
$lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_398_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_398/BiasAdd/ReadVariableOpБ
lstm_cell_398/BiasAddBiasAddlstm_cell_398/add:z:0,lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/BiasAdd
lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_398/split/split_dimї
lstm_cell_398/splitSplit&lstm_cell_398/split/split_dim:output:0lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_398/split
lstm_cell_398/SigmoidSigmoidlstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Sigmoid
lstm_cell_398/Sigmoid_1Sigmoidlstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Sigmoid_1
lstm_cell_398/mulMullstm_cell_398/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/mul
lstm_cell_398/ReluRelulstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Relu 
lstm_cell_398/mul_1Mullstm_cell_398/Sigmoid:y:0 lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/mul_1
lstm_cell_398/add_1AddV2lstm_cell_398/mul:z:0lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/add_1
lstm_cell_398/Sigmoid_2Sigmoidlstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Sigmoid_2
lstm_cell_398/Relu_1Relulstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Relu_1Є
lstm_cell_398/mul_2Mullstm_cell_398/Sigmoid_2:y:0"lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_398_matmul_readvariableop_resource.lstm_cell_398_matmul_1_readvariableop_resource-lstm_cell_398_biasadd_readvariableop_resource*
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
while_body_16719131*
condR
while_cond_16719130*K
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
NoOpNoOp%^lstm_cell_398/BiasAdd/ReadVariableOp$^lstm_cell_398/MatMul/ReadVariableOp&^lstm_cell_398/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_398/BiasAdd/ReadVariableOp$lstm_cell_398/BiasAdd/ReadVariableOp2J
#lstm_cell_398/MatMul/ReadVariableOp#lstm_cell_398/MatMul/ReadVariableOp2N
%lstm_cell_398/MatMul_1/ReadVariableOp%lstm_cell_398/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ЂИ
Ц 
$__inference__traced_restore_16719832
file_prefix3
!assignvariableop_dense_132_kernel:d/
!assignvariableop_1_dense_132_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: ]
Jassignvariableop_7_bidirectional_132_forward_lstm_132_lstm_cell_397_kernel:	Шg
Tassignvariableop_8_bidirectional_132_forward_lstm_132_lstm_cell_397_recurrent_kernel:	2ШW
Hassignvariableop_9_bidirectional_132_forward_lstm_132_lstm_cell_397_bias:	Ш_
Lassignvariableop_10_bidirectional_132_backward_lstm_132_lstm_cell_398_kernel:	Шi
Vassignvariableop_11_bidirectional_132_backward_lstm_132_lstm_cell_398_recurrent_kernel:	2ШY
Jassignvariableop_12_bidirectional_132_backward_lstm_132_lstm_cell_398_bias:	Ш#
assignvariableop_13_total: #
assignvariableop_14_count: =
+assignvariableop_15_adam_dense_132_kernel_m:d7
)assignvariableop_16_adam_dense_132_bias_m:e
Rassignvariableop_17_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_kernel_m:	Шo
\assignvariableop_18_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_recurrent_kernel_m:	2Ш_
Passignvariableop_19_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_bias_m:	Шf
Sassignvariableop_20_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_kernel_m:	Шp
]assignvariableop_21_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_recurrent_kernel_m:	2Ш`
Qassignvariableop_22_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_bias_m:	Ш=
+assignvariableop_23_adam_dense_132_kernel_v:d7
)assignvariableop_24_adam_dense_132_bias_v:e
Rassignvariableop_25_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_kernel_v:	Шo
\assignvariableop_26_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_recurrent_kernel_v:	2Ш_
Passignvariableop_27_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_bias_v:	Шf
Sassignvariableop_28_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_kernel_v:	Шp
]assignvariableop_29_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_recurrent_kernel_v:	2Ш`
Qassignvariableop_30_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_bias_v:	Ш@
.assignvariableop_31_adam_dense_132_kernel_vhat:d:
,assignvariableop_32_adam_dense_132_bias_vhat:h
Uassignvariableop_33_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_kernel_vhat:	Шr
_assignvariableop_34_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_recurrent_kernel_vhat:	2Шb
Sassignvariableop_35_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_bias_vhat:	Шi
Vassignvariableop_36_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_kernel_vhat:	Шs
`assignvariableop_37_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_recurrent_kernel_vhat:	2Шc
Tassignvariableop_38_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_bias_vhat:	Ш
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
AssignVariableOpAssignVariableOp!assignvariableop_dense_132_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1І
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_132_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOpJassignvariableop_7_bidirectional_132_forward_lstm_132_lstm_cell_397_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8й
AssignVariableOp_8AssignVariableOpTassignvariableop_8_bidirectional_132_forward_lstm_132_lstm_cell_397_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Э
AssignVariableOp_9AssignVariableOpHassignvariableop_9_bidirectional_132_forward_lstm_132_lstm_cell_397_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10д
AssignVariableOp_10AssignVariableOpLassignvariableop_10_bidirectional_132_backward_lstm_132_lstm_cell_398_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11о
AssignVariableOp_11AssignVariableOpVassignvariableop_11_bidirectional_132_backward_lstm_132_lstm_cell_398_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12в
AssignVariableOp_12AssignVariableOpJassignvariableop_12_bidirectional_132_backward_lstm_132_lstm_cell_398_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_dense_132_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Б
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_dense_132_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17к
AssignVariableOp_17AssignVariableOpRassignvariableop_17_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18ф
AssignVariableOp_18AssignVariableOp\assignvariableop_18_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19и
AssignVariableOp_19AssignVariableOpPassignvariableop_19_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20л
AssignVariableOp_20AssignVariableOpSassignvariableop_20_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21х
AssignVariableOp_21AssignVariableOp]assignvariableop_21_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22й
AssignVariableOp_22AssignVariableOpQassignvariableop_22_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Г
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_132_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Б
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_132_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25к
AssignVariableOp_25AssignVariableOpRassignvariableop_25_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26ф
AssignVariableOp_26AssignVariableOp\assignvariableop_26_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27и
AssignVariableOp_27AssignVariableOpPassignvariableop_27_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28л
AssignVariableOp_28AssignVariableOpSassignvariableop_28_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29х
AssignVariableOp_29AssignVariableOp]assignvariableop_29_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30й
AssignVariableOp_30AssignVariableOpQassignvariableop_30_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ж
AssignVariableOp_31AssignVariableOp.assignvariableop_31_adam_dense_132_kernel_vhatIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Д
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_dense_132_bias_vhatIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33н
AssignVariableOp_33AssignVariableOpUassignvariableop_33_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_kernel_vhatIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34ч
AssignVariableOp_34AssignVariableOp_assignvariableop_34_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_recurrent_kernel_vhatIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35л
AssignVariableOp_35AssignVariableOpSassignvariableop_35_adam_bidirectional_132_forward_lstm_132_lstm_cell_397_bias_vhatIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36о
AssignVariableOp_36AssignVariableOpVassignvariableop_36_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_kernel_vhatIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37ш
AssignVariableOp_37AssignVariableOp`assignvariableop_37_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_recurrent_kernel_vhatIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38м
AssignVariableOp_38AssignVariableOpTassignvariableop_38_adam_bidirectional_132_backward_lstm_132_lstm_cell_398_bias_vhatIdentity_38:output:0"/device:CPU:0*
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
п
Э
while_cond_16719283
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16719283___redundant_placeholder06
2while_while_cond_16719283___redundant_placeholder16
2while_while_cond_16719283___redundant_placeholder26
2while_while_cond_16719283___redundant_placeholder3
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
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_16718561

inputs?
,lstm_cell_397_matmul_readvariableop_resource:	ШA
.lstm_cell_397_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_397_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_397/BiasAdd/ReadVariableOpЂ#lstm_cell_397/MatMul/ReadVariableOpЂ%lstm_cell_397/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_397/MatMul/ReadVariableOpReadVariableOp,lstm_cell_397_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_397/MatMul/ReadVariableOpА
lstm_cell_397/MatMulMatMulstrided_slice_2:output:0+lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/MatMulО
%lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_397_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_397/MatMul_1/ReadVariableOpЌ
lstm_cell_397/MatMul_1MatMulzeros:output:0-lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/MatMul_1Є
lstm_cell_397/addAddV2lstm_cell_397/MatMul:product:0 lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/addЗ
$lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_397_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_397/BiasAdd/ReadVariableOpБ
lstm_cell_397/BiasAddBiasAddlstm_cell_397/add:z:0,lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/BiasAdd
lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_397/split/split_dimї
lstm_cell_397/splitSplit&lstm_cell_397/split/split_dim:output:0lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_397/split
lstm_cell_397/SigmoidSigmoidlstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Sigmoid
lstm_cell_397/Sigmoid_1Sigmoidlstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Sigmoid_1
lstm_cell_397/mulMullstm_cell_397/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/mul
lstm_cell_397/ReluRelulstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Relu 
lstm_cell_397/mul_1Mullstm_cell_397/Sigmoid:y:0 lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/mul_1
lstm_cell_397/add_1AddV2lstm_cell_397/mul:z:0lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/add_1
lstm_cell_397/Sigmoid_2Sigmoidlstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Sigmoid_2
lstm_cell_397/Relu_1Relulstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Relu_1Є
lstm_cell_397/mul_2Mullstm_cell_397/Sigmoid_2:y:0"lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_397_matmul_readvariableop_resource.lstm_cell_397_matmul_1_readvariableop_resource-lstm_cell_397_biasadd_readvariableop_resource*
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
while_body_16718477*
condR
while_cond_16718476*K
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
NoOpNoOp%^lstm_cell_397/BiasAdd/ReadVariableOp$^lstm_cell_397/MatMul/ReadVariableOp&^lstm_cell_397/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_397/BiasAdd/ReadVariableOp$lstm_cell_397/BiasAdd/ReadVariableOp2J
#lstm_cell_397/MatMul/ReadVariableOp#lstm_cell_397/MatMul/ReadVariableOp2N
%lstm_cell_397/MatMul_1/ReadVariableOp%lstm_cell_397/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ў

Ѕ
4__inference_bidirectional_132_layer_call_fn_16716706

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
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_167160342
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
$forward_lstm_132_while_body_16716198>
:forward_lstm_132_while_forward_lstm_132_while_loop_counterD
@forward_lstm_132_while_forward_lstm_132_while_maximum_iterations&
"forward_lstm_132_while_placeholder(
$forward_lstm_132_while_placeholder_1(
$forward_lstm_132_while_placeholder_2(
$forward_lstm_132_while_placeholder_3(
$forward_lstm_132_while_placeholder_4=
9forward_lstm_132_while_forward_lstm_132_strided_slice_1_0y
uforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_132_while_greater_forward_lstm_132_cast_0X
Eforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0:	ШZ
Gforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0:	2ШU
Fforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0:	Ш#
forward_lstm_132_while_identity%
!forward_lstm_132_while_identity_1%
!forward_lstm_132_while_identity_2%
!forward_lstm_132_while_identity_3%
!forward_lstm_132_while_identity_4%
!forward_lstm_132_while_identity_5%
!forward_lstm_132_while_identity_6;
7forward_lstm_132_while_forward_lstm_132_strided_slice_1w
sforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_132_while_greater_forward_lstm_132_castV
Cforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource:	ШX
Eforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource:	2ШS
Dforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource:	ШЂ;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOpЂ:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOpЂ<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOpх
Hforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2J
Hforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
:forward_lstm_132/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_132_while_placeholderQforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02<
:forward_lstm_132/while/TensorArrayV2Read/TensorListGetItemе
forward_lstm_132/while/GreaterGreater6forward_lstm_132_while_greater_forward_lstm_132_cast_0"forward_lstm_132_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2 
forward_lstm_132/while/Greaterџ
:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOpReadVariableOpEforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp
+forward_lstm_132/while/lstm_cell_397/MatMulMatMulAforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem:item:0Bforward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_132/while/lstm_cell_397/MatMul
<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOpGforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp
-forward_lstm_132/while/lstm_cell_397/MatMul_1MatMul$forward_lstm_132_while_placeholder_3Dforward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-forward_lstm_132/while/lstm_cell_397/MatMul_1
(forward_lstm_132/while/lstm_cell_397/addAddV25forward_lstm_132/while/lstm_cell_397/MatMul:product:07forward_lstm_132/while/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_132/while/lstm_cell_397/addў
;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOpFforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp
,forward_lstm_132/while/lstm_cell_397/BiasAddBiasAdd,forward_lstm_132/while/lstm_cell_397/add:z:0Cforward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_132/while/lstm_cell_397/BiasAddЎ
4forward_lstm_132/while/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4forward_lstm_132/while/lstm_cell_397/split/split_dimг
*forward_lstm_132/while/lstm_cell_397/splitSplit=forward_lstm_132/while/lstm_cell_397/split/split_dim:output:05forward_lstm_132/while/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*forward_lstm_132/while/lstm_cell_397/splitЮ
,forward_lstm_132/while/lstm_cell_397/SigmoidSigmoid3forward_lstm_132/while/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,forward_lstm_132/while/lstm_cell_397/Sigmoidв
.forward_lstm_132/while/lstm_cell_397/Sigmoid_1Sigmoid3forward_lstm_132/while/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_132/while/lstm_cell_397/Sigmoid_1ч
(forward_lstm_132/while/lstm_cell_397/mulMul2forward_lstm_132/while/lstm_cell_397/Sigmoid_1:y:0$forward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_132/while/lstm_cell_397/mulХ
)forward_lstm_132/while/lstm_cell_397/ReluRelu3forward_lstm_132/while/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_132/while/lstm_cell_397/Reluќ
*forward_lstm_132/while/lstm_cell_397/mul_1Mul0forward_lstm_132/while/lstm_cell_397/Sigmoid:y:07forward_lstm_132/while/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_132/while/lstm_cell_397/mul_1ё
*forward_lstm_132/while/lstm_cell_397/add_1AddV2,forward_lstm_132/while/lstm_cell_397/mul:z:0.forward_lstm_132/while/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_132/while/lstm_cell_397/add_1в
.forward_lstm_132/while/lstm_cell_397/Sigmoid_2Sigmoid3forward_lstm_132/while/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_132/while/lstm_cell_397/Sigmoid_2Ф
+forward_lstm_132/while/lstm_cell_397/Relu_1Relu.forward_lstm_132/while/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_132/while/lstm_cell_397/Relu_1
*forward_lstm_132/while/lstm_cell_397/mul_2Mul2forward_lstm_132/while/lstm_cell_397/Sigmoid_2:y:09forward_lstm_132/while/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_132/while/lstm_cell_397/mul_2є
forward_lstm_132/while/SelectSelect"forward_lstm_132/while/Greater:z:0.forward_lstm_132/while/lstm_cell_397/mul_2:z:0$forward_lstm_132_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/while/Selectј
forward_lstm_132/while/Select_1Select"forward_lstm_132/while/Greater:z:0.forward_lstm_132/while/lstm_cell_397/mul_2:z:0$forward_lstm_132_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_132/while/Select_1ј
forward_lstm_132/while/Select_2Select"forward_lstm_132/while/Greater:z:0.forward_lstm_132/while/lstm_cell_397/add_1:z:0$forward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_132/while/Select_2Ў
;forward_lstm_132/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_132_while_placeholder_1"forward_lstm_132_while_placeholder&forward_lstm_132/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_132/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_132/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_132/while/add/y­
forward_lstm_132/while/addAddV2"forward_lstm_132_while_placeholder%forward_lstm_132/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/while/add
forward_lstm_132/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_132/while/add_1/yЫ
forward_lstm_132/while/add_1AddV2:forward_lstm_132_while_forward_lstm_132_while_loop_counter'forward_lstm_132/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/while/add_1Џ
forward_lstm_132/while/IdentityIdentity forward_lstm_132/while/add_1:z:0^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_132/while/Identityг
!forward_lstm_132/while/Identity_1Identity@forward_lstm_132_while_forward_lstm_132_while_maximum_iterations^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_132/while/Identity_1Б
!forward_lstm_132/while/Identity_2Identityforward_lstm_132/while/add:z:0^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_132/while/Identity_2о
!forward_lstm_132/while/Identity_3IdentityKforward_lstm_132/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_132/while/Identity_3Ъ
!forward_lstm_132/while/Identity_4Identity&forward_lstm_132/while/Select:output:0^forward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_132/while/Identity_4Ь
!forward_lstm_132/while/Identity_5Identity(forward_lstm_132/while/Select_1:output:0^forward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_132/while/Identity_5Ь
!forward_lstm_132/while/Identity_6Identity(forward_lstm_132/while/Select_2:output:0^forward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_132/while/Identity_6Ж
forward_lstm_132/while/NoOpNoOp<^forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp;^forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp=^forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_132/while/NoOp"t
7forward_lstm_132_while_forward_lstm_132_strided_slice_19forward_lstm_132_while_forward_lstm_132_strided_slice_1_0"n
4forward_lstm_132_while_greater_forward_lstm_132_cast6forward_lstm_132_while_greater_forward_lstm_132_cast_0"K
forward_lstm_132_while_identity(forward_lstm_132/while/Identity:output:0"O
!forward_lstm_132_while_identity_1*forward_lstm_132/while/Identity_1:output:0"O
!forward_lstm_132_while_identity_2*forward_lstm_132/while/Identity_2:output:0"O
!forward_lstm_132_while_identity_3*forward_lstm_132/while/Identity_3:output:0"O
!forward_lstm_132_while_identity_4*forward_lstm_132/while/Identity_4:output:0"O
!forward_lstm_132_while_identity_5*forward_lstm_132/while/Identity_5:output:0"O
!forward_lstm_132_while_identity_6*forward_lstm_132/while/Identity_6:output:0"
Dforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resourceFforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0"
Eforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resourceGforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0"
Cforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resourceEforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0"ь
sforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensoruforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2z
;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp2x
:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp2|
<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
М
љ
0__inference_lstm_cell_397_layer_call_fn_16719385

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
K__inference_lstm_cell_397_layer_call_and_return_conditional_losses_167136732
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
$forward_lstm_132_while_cond_16717092>
:forward_lstm_132_while_forward_lstm_132_while_loop_counterD
@forward_lstm_132_while_forward_lstm_132_while_maximum_iterations&
"forward_lstm_132_while_placeholder(
$forward_lstm_132_while_placeholder_1(
$forward_lstm_132_while_placeholder_2(
$forward_lstm_132_while_placeholder_3@
<forward_lstm_132_while_less_forward_lstm_132_strided_slice_1X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16717092___redundant_placeholder0X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16717092___redundant_placeholder1X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16717092___redundant_placeholder2X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16717092___redundant_placeholder3#
forward_lstm_132_while_identity
Х
forward_lstm_132/while/LessLess"forward_lstm_132_while_placeholder<forward_lstm_132_while_less_forward_lstm_132_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_132/while/Less
forward_lstm_132/while/IdentityIdentityforward_lstm_132/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_132/while/Identity"K
forward_lstm_132_while_identity(forward_lstm_132/while/Identity:output:0*(
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

ј
G__inference_dense_132_layer_call_and_return_conditional_losses_16718064

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
жF

N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_16713966

inputs)
lstm_cell_397_16713884:	Ш)
lstm_cell_397_16713886:	2Ш%
lstm_cell_397_16713888:	Ш
identityЂ%lstm_cell_397/StatefulPartitionedCallЂwhileD
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
%lstm_cell_397/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_397_16713884lstm_cell_397_16713886lstm_cell_397_16713888*
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
K__inference_lstm_cell_397_layer_call_and_return_conditional_losses_167138192'
%lstm_cell_397/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_397_16713884lstm_cell_397_16713886lstm_cell_397_16713888*
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
while_body_16713897*
condR
while_cond_16713896*K
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
NoOpNoOp&^lstm_cell_397/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2N
%lstm_cell_397/StatefulPartitionedCall%lstm_cell_397/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
&
ј
while_body_16713897
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_397_16713921_0:	Ш1
while_lstm_cell_397_16713923_0:	2Ш-
while_lstm_cell_397_16713925_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_397_16713921:	Ш/
while_lstm_cell_397_16713923:	2Ш+
while_lstm_cell_397_16713925:	ШЂ+while/lstm_cell_397/StatefulPartitionedCallУ
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
+while/lstm_cell_397/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_397_16713921_0while_lstm_cell_397_16713923_0while_lstm_cell_397_16713925_0*
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
K__inference_lstm_cell_397_layer_call_and_return_conditional_losses_167138192-
+while/lstm_cell_397/StatefulPartitionedCallј
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_397/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_397/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѕ
while/Identity_5Identity4while/lstm_cell_397/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp,^while/lstm_cell_397/StatefulPartitionedCall*"
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
while_lstm_cell_397_16713921while_lstm_cell_397_16713921_0">
while_lstm_cell_397_16713923while_lstm_cell_397_16713923_0">
while_lstm_cell_397_16713925while_lstm_cell_397_16713925_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2Z
+while/lstm_cell_397/StatefulPartitionedCall+while/lstm_cell_397/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
аf
ф
%backward_lstm_132_while_body_16716377@
<backward_lstm_132_while_backward_lstm_132_while_loop_counterF
Bbackward_lstm_132_while_backward_lstm_132_while_maximum_iterations'
#backward_lstm_132_while_placeholder)
%backward_lstm_132_while_placeholder_1)
%backward_lstm_132_while_placeholder_2)
%backward_lstm_132_while_placeholder_3)
%backward_lstm_132_while_placeholder_4?
;backward_lstm_132_while_backward_lstm_132_strided_slice_1_0{
wbackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_132_while_less_backward_lstm_132_sub_1_0Y
Fbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0:	Ш[
Hbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0:	2ШV
Gbackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0:	Ш$
 backward_lstm_132_while_identity&
"backward_lstm_132_while_identity_1&
"backward_lstm_132_while_identity_2&
"backward_lstm_132_while_identity_3&
"backward_lstm_132_while_identity_4&
"backward_lstm_132_while_identity_5&
"backward_lstm_132_while_identity_6=
9backward_lstm_132_while_backward_lstm_132_strided_slice_1y
ubackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_132_while_less_backward_lstm_132_sub_1W
Dbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource:	ШY
Fbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource:	2ШT
Ebackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource:	ШЂ<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOpЂ;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpЂ=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOpч
Ibackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2K
Ibackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeП
;backward_lstm_132/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_132_while_placeholderRbackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02=
;backward_lstm_132/while/TensorArrayV2Read/TensorListGetItemЯ
backward_lstm_132/while/LessLess6backward_lstm_132_while_less_backward_lstm_132_sub_1_0#backward_lstm_132_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_132/while/Less
;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpReadVariableOpFbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02=
;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpЂ
,backward_lstm_132/while/lstm_cell_398/MatMulMatMulBbackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem:item:0Cbackward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_132/while/lstm_cell_398/MatMul
=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOpHbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02?
=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp
.backward_lstm_132/while/lstm_cell_398/MatMul_1MatMul%backward_lstm_132_while_placeholder_3Ebackward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ20
.backward_lstm_132/while/lstm_cell_398/MatMul_1
)backward_lstm_132/while/lstm_cell_398/addAddV26backward_lstm_132/while/lstm_cell_398/MatMul:product:08backward_lstm_132/while/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_132/while/lstm_cell_398/add
<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOpGbackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02>
<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp
-backward_lstm_132/while/lstm_cell_398/BiasAddBiasAdd-backward_lstm_132/while/lstm_cell_398/add:z:0Dbackward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_132/while/lstm_cell_398/BiasAddА
5backward_lstm_132/while/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5backward_lstm_132/while/lstm_cell_398/split/split_dimз
+backward_lstm_132/while/lstm_cell_398/splitSplit>backward_lstm_132/while/lstm_cell_398/split/split_dim:output:06backward_lstm_132/while/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2-
+backward_lstm_132/while/lstm_cell_398/splitб
-backward_lstm_132/while/lstm_cell_398/SigmoidSigmoid4backward_lstm_132/while/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22/
-backward_lstm_132/while/lstm_cell_398/Sigmoidе
/backward_lstm_132/while/lstm_cell_398/Sigmoid_1Sigmoid4backward_lstm_132/while/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_132/while/lstm_cell_398/Sigmoid_1ы
)backward_lstm_132/while/lstm_cell_398/mulMul3backward_lstm_132/while/lstm_cell_398/Sigmoid_1:y:0%backward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_132/while/lstm_cell_398/mulШ
*backward_lstm_132/while/lstm_cell_398/ReluRelu4backward_lstm_132/while/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_132/while/lstm_cell_398/Relu
+backward_lstm_132/while/lstm_cell_398/mul_1Mul1backward_lstm_132/while/lstm_cell_398/Sigmoid:y:08backward_lstm_132/while/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_132/while/lstm_cell_398/mul_1ѕ
+backward_lstm_132/while/lstm_cell_398/add_1AddV2-backward_lstm_132/while/lstm_cell_398/mul:z:0/backward_lstm_132/while/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_132/while/lstm_cell_398/add_1е
/backward_lstm_132/while/lstm_cell_398/Sigmoid_2Sigmoid4backward_lstm_132/while/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_132/while/lstm_cell_398/Sigmoid_2Ч
,backward_lstm_132/while/lstm_cell_398/Relu_1Relu/backward_lstm_132/while/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_132/while/lstm_cell_398/Relu_1
+backward_lstm_132/while/lstm_cell_398/mul_2Mul3backward_lstm_132/while/lstm_cell_398/Sigmoid_2:y:0:backward_lstm_132/while/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_132/while/lstm_cell_398/mul_2і
backward_lstm_132/while/SelectSelect backward_lstm_132/while/Less:z:0/backward_lstm_132/while/lstm_cell_398/mul_2:z:0%backward_lstm_132_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_132/while/Selectњ
 backward_lstm_132/while/Select_1Select backward_lstm_132/while/Less:z:0/backward_lstm_132/while/lstm_cell_398/mul_2:z:0%backward_lstm_132_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_132/while/Select_1њ
 backward_lstm_132/while/Select_2Select backward_lstm_132/while/Less:z:0/backward_lstm_132/while/lstm_cell_398/add_1:z:0%backward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_132/while/Select_2Г
<backward_lstm_132/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_132_while_placeholder_1#backward_lstm_132_while_placeholder'backward_lstm_132/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_132/while/TensorArrayV2Write/TensorListSetItem
backward_lstm_132/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_132/while/add/yБ
backward_lstm_132/while/addAddV2#backward_lstm_132_while_placeholder&backward_lstm_132/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/while/add
backward_lstm_132/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_132/while/add_1/yа
backward_lstm_132/while/add_1AddV2<backward_lstm_132_while_backward_lstm_132_while_loop_counter(backward_lstm_132/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/while/add_1Г
 backward_lstm_132/while/IdentityIdentity!backward_lstm_132/while/add_1:z:0^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_132/while/Identityи
"backward_lstm_132/while/Identity_1IdentityBbackward_lstm_132_while_backward_lstm_132_while_maximum_iterations^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_132/while/Identity_1Е
"backward_lstm_132/while/Identity_2Identitybackward_lstm_132/while/add:z:0^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_132/while/Identity_2т
"backward_lstm_132/while/Identity_3IdentityLbackward_lstm_132/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_132/while/Identity_3Ю
"backward_lstm_132/while/Identity_4Identity'backward_lstm_132/while/Select:output:0^backward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_132/while/Identity_4а
"backward_lstm_132/while/Identity_5Identity)backward_lstm_132/while/Select_1:output:0^backward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_132/while/Identity_5а
"backward_lstm_132/while/Identity_6Identity)backward_lstm_132/while/Select_2:output:0^backward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_132/while/Identity_6Л
backward_lstm_132/while/NoOpNoOp=^backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp<^backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp>^backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_132/while/NoOp"x
9backward_lstm_132_while_backward_lstm_132_strided_slice_1;backward_lstm_132_while_backward_lstm_132_strided_slice_1_0"M
 backward_lstm_132_while_identity)backward_lstm_132/while/Identity:output:0"Q
"backward_lstm_132_while_identity_1+backward_lstm_132/while/Identity_1:output:0"Q
"backward_lstm_132_while_identity_2+backward_lstm_132/while/Identity_2:output:0"Q
"backward_lstm_132_while_identity_3+backward_lstm_132/while/Identity_3:output:0"Q
"backward_lstm_132_while_identity_4+backward_lstm_132/while/Identity_4:output:0"Q
"backward_lstm_132_while_identity_5+backward_lstm_132/while/Identity_5:output:0"Q
"backward_lstm_132_while_identity_6+backward_lstm_132/while/Identity_6:output:0"n
4backward_lstm_132_while_less_backward_lstm_132_sub_16backward_lstm_132_while_less_backward_lstm_132_sub_1_0"
Ebackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resourceGbackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0"
Fbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resourceHbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0"
Dbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resourceFbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0"№
ubackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2|
<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp2z
;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp2~
=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
ы	

4__inference_bidirectional_132_layer_call_fn_16716688
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
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_167155962
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
І_
­
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_16715375

inputs?
,lstm_cell_398_matmul_readvariableop_resource:	ШA
.lstm_cell_398_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_398_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_398/BiasAdd/ReadVariableOpЂ#lstm_cell_398/MatMul/ReadVariableOpЂ%lstm_cell_398/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_398/MatMul/ReadVariableOpReadVariableOp,lstm_cell_398_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_398/MatMul/ReadVariableOpА
lstm_cell_398/MatMulMatMulstrided_slice_2:output:0+lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/MatMulО
%lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_398_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_398/MatMul_1/ReadVariableOpЌ
lstm_cell_398/MatMul_1MatMulzeros:output:0-lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/MatMul_1Є
lstm_cell_398/addAddV2lstm_cell_398/MatMul:product:0 lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/addЗ
$lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_398_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_398/BiasAdd/ReadVariableOpБ
lstm_cell_398/BiasAddBiasAddlstm_cell_398/add:z:0,lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/BiasAdd
lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_398/split/split_dimї
lstm_cell_398/splitSplit&lstm_cell_398/split/split_dim:output:0lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_398/split
lstm_cell_398/SigmoidSigmoidlstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Sigmoid
lstm_cell_398/Sigmoid_1Sigmoidlstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Sigmoid_1
lstm_cell_398/mulMullstm_cell_398/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/mul
lstm_cell_398/ReluRelulstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Relu 
lstm_cell_398/mul_1Mullstm_cell_398/Sigmoid:y:0 lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/mul_1
lstm_cell_398/add_1AddV2lstm_cell_398/mul:z:0lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/add_1
lstm_cell_398/Sigmoid_2Sigmoidlstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Sigmoid_2
lstm_cell_398/Relu_1Relulstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Relu_1Є
lstm_cell_398/mul_2Mullstm_cell_398/Sigmoid_2:y:0"lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_398_matmul_readvariableop_resource.lstm_cell_398_matmul_1_readvariableop_resource-lstm_cell_398_biasadd_readvariableop_resource*
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
while_body_16715291*
condR
while_cond_16715290*K
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
NoOpNoOp%^lstm_cell_398/BiasAdd/ReadVariableOp$^lstm_cell_398/MatMul/ReadVariableOp&^lstm_cell_398/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_398/BiasAdd/ReadVariableOp$lstm_cell_398/BiasAdd/ReadVariableOp2J
#lstm_cell_398/MatMul/ReadVariableOp#lstm_cell_398/MatMul/ReadVariableOp2N
%lstm_cell_398/MatMul_1/ReadVariableOp%lstm_cell_398/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
љ?
л
while_body_16718628
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_397_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_397_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_397_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_397_matmul_readvariableop_resource:	ШG
4while_lstm_cell_397_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_397_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_397/BiasAdd/ReadVariableOpЂ)while/lstm_cell_397/MatMul/ReadVariableOpЂ+while/lstm_cell_397/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_397/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_397_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_397/MatMul/ReadVariableOpк
while/lstm_cell_397/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/MatMulв
+while/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_397_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_397/MatMul_1/ReadVariableOpУ
while/lstm_cell_397/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/MatMul_1М
while/lstm_cell_397/addAddV2$while/lstm_cell_397/MatMul:product:0&while/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/addЫ
*while/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_397_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_397/BiasAdd/ReadVariableOpЩ
while/lstm_cell_397/BiasAddBiasAddwhile/lstm_cell_397/add:z:02while/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/BiasAdd
#while/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_397/split/split_dim
while/lstm_cell_397/splitSplit,while/lstm_cell_397/split/split_dim:output:0$while/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_397/split
while/lstm_cell_397/SigmoidSigmoid"while/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Sigmoid
while/lstm_cell_397/Sigmoid_1Sigmoid"while/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Sigmoid_1Ѓ
while/lstm_cell_397/mulMul!while/lstm_cell_397/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/mul
while/lstm_cell_397/ReluRelu"while/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/ReluИ
while/lstm_cell_397/mul_1Mulwhile/lstm_cell_397/Sigmoid:y:0&while/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/mul_1­
while/lstm_cell_397/add_1AddV2while/lstm_cell_397/mul:z:0while/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/add_1
while/lstm_cell_397/Sigmoid_2Sigmoid"while/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Sigmoid_2
while/lstm_cell_397/Relu_1Reluwhile/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Relu_1М
while/lstm_cell_397/mul_2Mul!while/lstm_cell_397/Sigmoid_2:y:0(while/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_397/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_397/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_397/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_397/BiasAdd/ReadVariableOp*^while/lstm_cell_397/MatMul/ReadVariableOp,^while/lstm_cell_397/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_397_biasadd_readvariableop_resource5while_lstm_cell_397_biasadd_readvariableop_resource_0"n
4while_lstm_cell_397_matmul_1_readvariableop_resource6while_lstm_cell_397_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_397_matmul_readvariableop_resource4while_lstm_cell_397_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_397/BiasAdd/ReadVariableOp*while/lstm_cell_397/BiasAdd/ReadVariableOp2V
)while/lstm_cell_397/MatMul/ReadVariableOp)while/lstm_cell_397/MatMul/ReadVariableOp2Z
+while/lstm_cell_397/MatMul_1/ReadVariableOp+while/lstm_cell_397/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
д
Т
3__inference_forward_lstm_132_layer_call_fn_16718075
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
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_167137562
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
еH

O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_16714388

inputs)
lstm_cell_398_16714306:	Ш)
lstm_cell_398_16714308:	2Ш%
lstm_cell_398_16714310:	Ш
identityЂ%lstm_cell_398/StatefulPartitionedCallЂwhileD
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
%lstm_cell_398/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_398_16714306lstm_cell_398_16714308lstm_cell_398_16714310*
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
K__inference_lstm_cell_398_layer_call_and_return_conditional_losses_167143052'
%lstm_cell_398/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_398_16714306lstm_cell_398_16714308lstm_cell_398_16714310*
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
while_body_16714319*
condR
while_cond_16714318*K
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
NoOpNoOp&^lstm_cell_398/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2N
%lstm_cell_398/StatefulPartitionedCall%lstm_cell_398/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
№?
л
while_body_16718825
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_398_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_398_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_398_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_398_matmul_readvariableop_resource:	ШG
4while_lstm_cell_398_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_398_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_398/BiasAdd/ReadVariableOpЂ)while/lstm_cell_398/MatMul/ReadVariableOpЂ+while/lstm_cell_398/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_398/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_398_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_398/MatMul/ReadVariableOpк
while/lstm_cell_398/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/MatMulв
+while/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_398_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_398/MatMul_1/ReadVariableOpУ
while/lstm_cell_398/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/MatMul_1М
while/lstm_cell_398/addAddV2$while/lstm_cell_398/MatMul:product:0&while/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/addЫ
*while/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_398_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_398/BiasAdd/ReadVariableOpЩ
while/lstm_cell_398/BiasAddBiasAddwhile/lstm_cell_398/add:z:02while/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/BiasAdd
#while/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_398/split/split_dim
while/lstm_cell_398/splitSplit,while/lstm_cell_398/split/split_dim:output:0$while/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_398/split
while/lstm_cell_398/SigmoidSigmoid"while/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Sigmoid
while/lstm_cell_398/Sigmoid_1Sigmoid"while/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Sigmoid_1Ѓ
while/lstm_cell_398/mulMul!while/lstm_cell_398/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/mul
while/lstm_cell_398/ReluRelu"while/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/ReluИ
while/lstm_cell_398/mul_1Mulwhile/lstm_cell_398/Sigmoid:y:0&while/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/mul_1­
while/lstm_cell_398/add_1AddV2while/lstm_cell_398/mul:z:0while/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/add_1
while/lstm_cell_398/Sigmoid_2Sigmoid"while/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Sigmoid_2
while/lstm_cell_398/Relu_1Reluwhile/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Relu_1М
while/lstm_cell_398/mul_2Mul!while/lstm_cell_398/Sigmoid_2:y:0(while/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_398/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_398/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_398/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_398/BiasAdd/ReadVariableOp*^while/lstm_cell_398/MatMul/ReadVariableOp,^while/lstm_cell_398/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_398_biasadd_readvariableop_resource5while_lstm_cell_398_biasadd_readvariableop_resource_0"n
4while_lstm_cell_398_matmul_1_readvariableop_resource6while_lstm_cell_398_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_398_matmul_readvariableop_resource4while_lstm_cell_398_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_398/BiasAdd/ReadVariableOp*while/lstm_cell_398/BiasAdd/ReadVariableOp2V
)while/lstm_cell_398/MatMul/ReadVariableOp)while/lstm_cell_398/MatMul/ReadVariableOp2Z
+while/lstm_cell_398/MatMul_1/ReadVariableOp+while/lstm_cell_398/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_cond_16718174
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16718174___redundant_placeholder06
2while_while_cond_16718174___redundant_placeholder16
2while_while_cond_16718174___redundant_placeholder26
2while_while_cond_16718174___redundant_placeholder3
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

в
Fsequential_132_bidirectional_132_backward_lstm_132_while_body_16713494
~sequential_132_bidirectional_132_backward_lstm_132_while_sequential_132_bidirectional_132_backward_lstm_132_while_loop_counter
sequential_132_bidirectional_132_backward_lstm_132_while_sequential_132_bidirectional_132_backward_lstm_132_while_maximum_iterationsH
Dsequential_132_bidirectional_132_backward_lstm_132_while_placeholderJ
Fsequential_132_bidirectional_132_backward_lstm_132_while_placeholder_1J
Fsequential_132_bidirectional_132_backward_lstm_132_while_placeholder_2J
Fsequential_132_bidirectional_132_backward_lstm_132_while_placeholder_3J
Fsequential_132_bidirectional_132_backward_lstm_132_while_placeholder_4
}sequential_132_bidirectional_132_backward_lstm_132_while_sequential_132_bidirectional_132_backward_lstm_132_strided_slice_1_0О
Йsequential_132_bidirectional_132_backward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_sequential_132_bidirectional_132_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0|
xsequential_132_bidirectional_132_backward_lstm_132_while_less_sequential_132_bidirectional_132_backward_lstm_132_sub_1_0z
gsequential_132_bidirectional_132_backward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0:	Ш|
isequential_132_bidirectional_132_backward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0:	2Шw
hsequential_132_bidirectional_132_backward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0:	ШE
Asequential_132_bidirectional_132_backward_lstm_132_while_identityG
Csequential_132_bidirectional_132_backward_lstm_132_while_identity_1G
Csequential_132_bidirectional_132_backward_lstm_132_while_identity_2G
Csequential_132_bidirectional_132_backward_lstm_132_while_identity_3G
Csequential_132_bidirectional_132_backward_lstm_132_while_identity_4G
Csequential_132_bidirectional_132_backward_lstm_132_while_identity_5G
Csequential_132_bidirectional_132_backward_lstm_132_while_identity_6
{sequential_132_bidirectional_132_backward_lstm_132_while_sequential_132_bidirectional_132_backward_lstm_132_strided_slice_1М
Зsequential_132_bidirectional_132_backward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_sequential_132_bidirectional_132_backward_lstm_132_tensorarrayunstack_tensorlistfromtensorz
vsequential_132_bidirectional_132_backward_lstm_132_while_less_sequential_132_bidirectional_132_backward_lstm_132_sub_1x
esequential_132_bidirectional_132_backward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource:	Шz
gsequential_132_bidirectional_132_backward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource:	2Шu
fsequential_132_bidirectional_132_backward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource:	ШЂ]sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOpЂ\sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpЂ^sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOpЉ
jsequential_132/bidirectional_132/backward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2l
jsequential_132/bidirectional_132/backward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shape
\sequential_132/bidirectional_132/backward_lstm_132/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЙsequential_132_bidirectional_132_backward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_sequential_132_bidirectional_132_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0Dsequential_132_bidirectional_132_backward_lstm_132_while_placeholderssequential_132/bidirectional_132/backward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02^
\sequential_132/bidirectional_132/backward_lstm_132/while/TensorArrayV2Read/TensorListGetItemє
=sequential_132/bidirectional_132/backward_lstm_132/while/LessLessxsequential_132_bidirectional_132_backward_lstm_132_while_less_sequential_132_bidirectional_132_backward_lstm_132_sub_1_0Dsequential_132_bidirectional_132_backward_lstm_132_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2?
=sequential_132/bidirectional_132/backward_lstm_132/while/Lessх
\sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpReadVariableOpgsequential_132_bidirectional_132_backward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02^
\sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpІ
Msequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMulMatMulcsequential_132/bidirectional_132/backward_lstm_132/while/TensorArrayV2Read/TensorListGetItem:item:0dsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2O
Msequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMulы
^sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOpisequential_132_bidirectional_132_backward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02`
^sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp
Osequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMul_1MatMulFsequential_132_bidirectional_132_backward_lstm_132_while_placeholder_3fsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2Q
Osequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMul_1
Jsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/addAddV2Wsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMul:product:0Ysequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2L
Jsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/addф
]sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOphsequential_132_bidirectional_132_backward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02_
]sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp
Nsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/BiasAddBiasAddNsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/add:z:0esequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2P
Nsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/BiasAddђ
Vsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2X
Vsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/split/split_dimл
Lsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/splitSplit_sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/split/split_dim:output:0Wsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2N
Lsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/splitД
Nsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/SigmoidSigmoidUsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22P
Nsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/SigmoidИ
Psequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/Sigmoid_1SigmoidUsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22R
Psequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/Sigmoid_1я
Jsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/mulMulTsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/Sigmoid_1:y:0Fsequential_132_bidirectional_132_backward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22L
Jsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/mulЋ
Ksequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/ReluReluUsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22M
Ksequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/Relu
Lsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/mul_1MulRsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/Sigmoid:y:0Ysequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22N
Lsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/mul_1љ
Lsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/add_1AddV2Nsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/mul:z:0Psequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22N
Lsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/add_1И
Psequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/Sigmoid_2SigmoidUsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22R
Psequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/Sigmoid_2Њ
Msequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/Relu_1ReluPsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22O
Msequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/Relu_1
Lsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/mul_2MulTsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/Sigmoid_2:y:0[sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22N
Lsequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/mul_2
?sequential_132/bidirectional_132/backward_lstm_132/while/SelectSelectAsequential_132/bidirectional_132/backward_lstm_132/while/Less:z:0Psequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/mul_2:z:0Fsequential_132_bidirectional_132_backward_lstm_132_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22A
?sequential_132/bidirectional_132/backward_lstm_132/while/Select
Asequential_132/bidirectional_132/backward_lstm_132/while/Select_1SelectAsequential_132/bidirectional_132/backward_lstm_132/while/Less:z:0Psequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/mul_2:z:0Fsequential_132_bidirectional_132_backward_lstm_132_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22C
Asequential_132/bidirectional_132/backward_lstm_132/while/Select_1
Asequential_132/bidirectional_132/backward_lstm_132/while/Select_2SelectAsequential_132/bidirectional_132/backward_lstm_132/while/Less:z:0Psequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/add_1:z:0Fsequential_132_bidirectional_132_backward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22C
Asequential_132/bidirectional_132/backward_lstm_132/while/Select_2и
]sequential_132/bidirectional_132/backward_lstm_132/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemFsequential_132_bidirectional_132_backward_lstm_132_while_placeholder_1Dsequential_132_bidirectional_132_backward_lstm_132_while_placeholderHsequential_132/bidirectional_132/backward_lstm_132/while/Select:output:0*
_output_shapes
: *
element_dtype02_
]sequential_132/bidirectional_132/backward_lstm_132/while/TensorArrayV2Write/TensorListSetItemТ
>sequential_132/bidirectional_132/backward_lstm_132/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_132/bidirectional_132/backward_lstm_132/while/add/yЕ
<sequential_132/bidirectional_132/backward_lstm_132/while/addAddV2Dsequential_132_bidirectional_132_backward_lstm_132_while_placeholderGsequential_132/bidirectional_132/backward_lstm_132/while/add/y:output:0*
T0*
_output_shapes
: 2>
<sequential_132/bidirectional_132/backward_lstm_132/while/addЦ
@sequential_132/bidirectional_132/backward_lstm_132/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2B
@sequential_132/bidirectional_132/backward_lstm_132/while/add_1/yѕ
>sequential_132/bidirectional_132/backward_lstm_132/while/add_1AddV2~sequential_132_bidirectional_132_backward_lstm_132_while_sequential_132_bidirectional_132_backward_lstm_132_while_loop_counterIsequential_132/bidirectional_132/backward_lstm_132/while/add_1/y:output:0*
T0*
_output_shapes
: 2@
>sequential_132/bidirectional_132/backward_lstm_132/while/add_1З
Asequential_132/bidirectional_132/backward_lstm_132/while/IdentityIdentityBsequential_132/bidirectional_132/backward_lstm_132/while/add_1:z:0>^sequential_132/bidirectional_132/backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2C
Asequential_132/bidirectional_132/backward_lstm_132/while/Identityў
Csequential_132/bidirectional_132/backward_lstm_132/while/Identity_1Identitysequential_132_bidirectional_132_backward_lstm_132_while_sequential_132_bidirectional_132_backward_lstm_132_while_maximum_iterations>^sequential_132/bidirectional_132/backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2E
Csequential_132/bidirectional_132/backward_lstm_132/while/Identity_1Й
Csequential_132/bidirectional_132/backward_lstm_132/while/Identity_2Identity@sequential_132/bidirectional_132/backward_lstm_132/while/add:z:0>^sequential_132/bidirectional_132/backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2E
Csequential_132/bidirectional_132/backward_lstm_132/while/Identity_2ц
Csequential_132/bidirectional_132/backward_lstm_132/while/Identity_3Identitymsequential_132/bidirectional_132/backward_lstm_132/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential_132/bidirectional_132/backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2E
Csequential_132/bidirectional_132/backward_lstm_132/while/Identity_3в
Csequential_132/bidirectional_132/backward_lstm_132/while/Identity_4IdentityHsequential_132/bidirectional_132/backward_lstm_132/while/Select:output:0>^sequential_132/bidirectional_132/backward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_132/bidirectional_132/backward_lstm_132/while/Identity_4д
Csequential_132/bidirectional_132/backward_lstm_132/while/Identity_5IdentityJsequential_132/bidirectional_132/backward_lstm_132/while/Select_1:output:0>^sequential_132/bidirectional_132/backward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_132/bidirectional_132/backward_lstm_132/while/Identity_5д
Csequential_132/bidirectional_132/backward_lstm_132/while/Identity_6IdentityJsequential_132/bidirectional_132/backward_lstm_132/while/Select_2:output:0>^sequential_132/bidirectional_132/backward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_132/bidirectional_132/backward_lstm_132/while/Identity_6р
=sequential_132/bidirectional_132/backward_lstm_132/while/NoOpNoOp^^sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp]^sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp_^sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2?
=sequential_132/bidirectional_132/backward_lstm_132/while/NoOp"
Asequential_132_bidirectional_132_backward_lstm_132_while_identityJsequential_132/bidirectional_132/backward_lstm_132/while/Identity:output:0"
Csequential_132_bidirectional_132_backward_lstm_132_while_identity_1Lsequential_132/bidirectional_132/backward_lstm_132/while/Identity_1:output:0"
Csequential_132_bidirectional_132_backward_lstm_132_while_identity_2Lsequential_132/bidirectional_132/backward_lstm_132/while/Identity_2:output:0"
Csequential_132_bidirectional_132_backward_lstm_132_while_identity_3Lsequential_132/bidirectional_132/backward_lstm_132/while/Identity_3:output:0"
Csequential_132_bidirectional_132_backward_lstm_132_while_identity_4Lsequential_132/bidirectional_132/backward_lstm_132/while/Identity_4:output:0"
Csequential_132_bidirectional_132_backward_lstm_132_while_identity_5Lsequential_132/bidirectional_132/backward_lstm_132/while/Identity_5:output:0"
Csequential_132_bidirectional_132_backward_lstm_132_while_identity_6Lsequential_132/bidirectional_132/backward_lstm_132/while/Identity_6:output:0"ђ
vsequential_132_bidirectional_132_backward_lstm_132_while_less_sequential_132_bidirectional_132_backward_lstm_132_sub_1xsequential_132_bidirectional_132_backward_lstm_132_while_less_sequential_132_bidirectional_132_backward_lstm_132_sub_1_0"в
fsequential_132_bidirectional_132_backward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resourcehsequential_132_bidirectional_132_backward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0"д
gsequential_132_bidirectional_132_backward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resourceisequential_132_bidirectional_132_backward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0"а
esequential_132_bidirectional_132_backward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resourcegsequential_132_bidirectional_132_backward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0"ќ
{sequential_132_bidirectional_132_backward_lstm_132_while_sequential_132_bidirectional_132_backward_lstm_132_strided_slice_1}sequential_132_bidirectional_132_backward_lstm_132_while_sequential_132_bidirectional_132_backward_lstm_132_strided_slice_1_0"і
Зsequential_132_bidirectional_132_backward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_sequential_132_bidirectional_132_backward_lstm_132_tensorarrayunstack_tensorlistfromtensorЙsequential_132_bidirectional_132_backward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_sequential_132_bidirectional_132_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2О
]sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp]sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp2М
\sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp\sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp2Р
^sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp^sequential_132/bidirectional_132/backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_body_16715464
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_397_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_397_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_397_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_397_matmul_readvariableop_resource:	ШG
4while_lstm_cell_397_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_397_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_397/BiasAdd/ReadVariableOpЂ)while/lstm_cell_397/MatMul/ReadVariableOpЂ+while/lstm_cell_397/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_397/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_397_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_397/MatMul/ReadVariableOpк
while/lstm_cell_397/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/MatMulв
+while/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_397_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_397/MatMul_1/ReadVariableOpУ
while/lstm_cell_397/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/MatMul_1М
while/lstm_cell_397/addAddV2$while/lstm_cell_397/MatMul:product:0&while/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/addЫ
*while/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_397_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_397/BiasAdd/ReadVariableOpЩ
while/lstm_cell_397/BiasAddBiasAddwhile/lstm_cell_397/add:z:02while/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/BiasAdd
#while/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_397/split/split_dim
while/lstm_cell_397/splitSplit,while/lstm_cell_397/split/split_dim:output:0$while/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_397/split
while/lstm_cell_397/SigmoidSigmoid"while/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Sigmoid
while/lstm_cell_397/Sigmoid_1Sigmoid"while/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Sigmoid_1Ѓ
while/lstm_cell_397/mulMul!while/lstm_cell_397/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/mul
while/lstm_cell_397/ReluRelu"while/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/ReluИ
while/lstm_cell_397/mul_1Mulwhile/lstm_cell_397/Sigmoid:y:0&while/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/mul_1­
while/lstm_cell_397/add_1AddV2while/lstm_cell_397/mul:z:0while/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/add_1
while/lstm_cell_397/Sigmoid_2Sigmoid"while/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Sigmoid_2
while/lstm_cell_397/Relu_1Reluwhile/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Relu_1М
while/lstm_cell_397/mul_2Mul!while/lstm_cell_397/Sigmoid_2:y:0(while/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_397/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_397/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_397/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_397/BiasAdd/ReadVariableOp*^while/lstm_cell_397/MatMul/ReadVariableOp,^while/lstm_cell_397/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_397_biasadd_readvariableop_resource5while_lstm_cell_397_biasadd_readvariableop_resource_0"n
4while_lstm_cell_397_matmul_1_readvariableop_resource6while_lstm_cell_397_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_397_matmul_readvariableop_resource4while_lstm_cell_397_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_397/BiasAdd/ReadVariableOp*while/lstm_cell_397/BiasAdd/ReadVariableOp2V
)while/lstm_cell_397/MatMul/ReadVariableOp)while/lstm_cell_397/MatMul/ReadVariableOp2Z
+while/lstm_cell_397/MatMul_1/ReadVariableOp+while/lstm_cell_397/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_cond_16714318
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16714318___redundant_placeholder06
2while_while_cond_16714318___redundant_placeholder16
2while_while_cond_16714318___redundant_placeholder26
2while_while_cond_16714318___redundant_placeholder3
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
І_
­
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_16715183

inputs?
,lstm_cell_398_matmul_readvariableop_resource:	ШA
.lstm_cell_398_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_398_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_398/BiasAdd/ReadVariableOpЂ#lstm_cell_398/MatMul/ReadVariableOpЂ%lstm_cell_398/MatMul_1/ReadVariableOpЂwhileD
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
#lstm_cell_398/MatMul/ReadVariableOpReadVariableOp,lstm_cell_398_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_398/MatMul/ReadVariableOpА
lstm_cell_398/MatMulMatMulstrided_slice_2:output:0+lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/MatMulО
%lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_398_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_398/MatMul_1/ReadVariableOpЌ
lstm_cell_398/MatMul_1MatMulzeros:output:0-lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/MatMul_1Є
lstm_cell_398/addAddV2lstm_cell_398/MatMul:product:0 lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/addЗ
$lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_398_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_398/BiasAdd/ReadVariableOpБ
lstm_cell_398/BiasAddBiasAddlstm_cell_398/add:z:0,lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/BiasAdd
lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_398/split/split_dimї
lstm_cell_398/splitSplit&lstm_cell_398/split/split_dim:output:0lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_398/split
lstm_cell_398/SigmoidSigmoidlstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Sigmoid
lstm_cell_398/Sigmoid_1Sigmoidlstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Sigmoid_1
lstm_cell_398/mulMullstm_cell_398/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/mul
lstm_cell_398/ReluRelulstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Relu 
lstm_cell_398/mul_1Mullstm_cell_398/Sigmoid:y:0 lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/mul_1
lstm_cell_398/add_1AddV2lstm_cell_398/mul:z:0lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/add_1
lstm_cell_398/Sigmoid_2Sigmoidlstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Sigmoid_2
lstm_cell_398/Relu_1Relulstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Relu_1Є
lstm_cell_398/mul_2Mullstm_cell_398/Sigmoid_2:y:0"lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_398_matmul_readvariableop_resource.lstm_cell_398_matmul_1_readvariableop_resource-lstm_cell_398_biasadd_readvariableop_resource*
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
while_body_16715099*
condR
while_cond_16715098*K
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
NoOpNoOp%^lstm_cell_398/BiasAdd/ReadVariableOp$^lstm_cell_398/MatMul/ReadVariableOp&^lstm_cell_398/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_398/BiasAdd/ReadVariableOp$lstm_cell_398/BiasAdd/ReadVariableOp2J
#lstm_cell_398/MatMul/ReadVariableOp#lstm_cell_398/MatMul/ReadVariableOp2N
%lstm_cell_398/MatMul_1/ReadVariableOp%lstm_cell_398/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
п
Э
while_cond_16718977
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16718977___redundant_placeholder06
2while_while_cond_16718977___redundant_placeholder16
2while_while_cond_16718977___redundant_placeholder26
2while_while_cond_16718977___redundant_placeholder3
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
K__inference_lstm_cell_398_layer_call_and_return_conditional_losses_16714451

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
М
љ
0__inference_lstm_cell_397_layer_call_fn_16719402

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
K__inference_lstm_cell_397_layer_call_and_return_conditional_losses_167138192
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
ў

K__inference_lstm_cell_397_layer_call_and_return_conditional_losses_16719434

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
1__inference_sequential_132_layer_call_fn_16716085

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
L__inference_sequential_132_layer_call_and_return_conditional_losses_167160662
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
while_cond_16714938
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16714938___redundant_placeholder06
2while_while_cond_16714938___redundant_placeholder16
2while_while_cond_16714938___redundant_placeholder26
2while_while_cond_16714938___redundant_placeholder3
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
while_cond_16718627
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16718627___redundant_placeholder06
2while_while_cond_16718627___redundant_placeholder16
2while_while_cond_16718627___redundant_placeholder26
2while_while_cond_16718627___redundant_placeholder3
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

ј
G__inference_dense_132_layer_call_and_return_conditional_losses_16716059

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
while_cond_16715098
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16715098___redundant_placeholder06
2while_while_cond_16715098___redundant_placeholder16
2while_while_cond_16715098___redundant_placeholder26
2while_while_cond_16715098___redundant_placeholder3
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
while_body_16714939
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_397_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_397_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_397_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_397_matmul_readvariableop_resource:	ШG
4while_lstm_cell_397_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_397_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_397/BiasAdd/ReadVariableOpЂ)while/lstm_cell_397/MatMul/ReadVariableOpЂ+while/lstm_cell_397/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_397/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_397_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_397/MatMul/ReadVariableOpк
while/lstm_cell_397/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/MatMulв
+while/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_397_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_397/MatMul_1/ReadVariableOpУ
while/lstm_cell_397/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/MatMul_1М
while/lstm_cell_397/addAddV2$while/lstm_cell_397/MatMul:product:0&while/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/addЫ
*while/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_397_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_397/BiasAdd/ReadVariableOpЩ
while/lstm_cell_397/BiasAddBiasAddwhile/lstm_cell_397/add:z:02while/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/BiasAdd
#while/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_397/split/split_dim
while/lstm_cell_397/splitSplit,while/lstm_cell_397/split/split_dim:output:0$while/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_397/split
while/lstm_cell_397/SigmoidSigmoid"while/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Sigmoid
while/lstm_cell_397/Sigmoid_1Sigmoid"while/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Sigmoid_1Ѓ
while/lstm_cell_397/mulMul!while/lstm_cell_397/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/mul
while/lstm_cell_397/ReluRelu"while/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/ReluИ
while/lstm_cell_397/mul_1Mulwhile/lstm_cell_397/Sigmoid:y:0&while/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/mul_1­
while/lstm_cell_397/add_1AddV2while/lstm_cell_397/mul:z:0while/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/add_1
while/lstm_cell_397/Sigmoid_2Sigmoid"while/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Sigmoid_2
while/lstm_cell_397/Relu_1Reluwhile/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Relu_1М
while/lstm_cell_397/mul_2Mul!while/lstm_cell_397/Sigmoid_2:y:0(while/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_397/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_397/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_397/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_397/BiasAdd/ReadVariableOp*^while/lstm_cell_397/MatMul/ReadVariableOp,^while/lstm_cell_397/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_397_biasadd_readvariableop_resource5while_lstm_cell_397_biasadd_readvariableop_resource_0"n
4while_lstm_cell_397_matmul_1_readvariableop_resource6while_lstm_cell_397_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_397_matmul_readvariableop_resource4while_lstm_cell_397_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_397/BiasAdd/ReadVariableOp*while/lstm_cell_397/BiasAdd/ReadVariableOp2V
)while/lstm_cell_397/MatMul/ReadVariableOp)while/lstm_cell_397/MatMul/ReadVariableOp2Z
+while/lstm_cell_397/MatMul_1/ReadVariableOp+while/lstm_cell_397/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
&
ј
while_body_16714319
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_398_16714343_0:	Ш1
while_lstm_cell_398_16714345_0:	2Ш-
while_lstm_cell_398_16714347_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_398_16714343:	Ш/
while_lstm_cell_398_16714345:	2Ш+
while_lstm_cell_398_16714347:	ШЂ+while/lstm_cell_398/StatefulPartitionedCallУ
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
+while/lstm_cell_398/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_398_16714343_0while_lstm_cell_398_16714345_0while_lstm_cell_398_16714347_0*
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
K__inference_lstm_cell_398_layer_call_and_return_conditional_losses_167143052-
+while/lstm_cell_398/StatefulPartitionedCallј
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_398/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_398/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѕ
while/Identity_5Identity4while/lstm_cell_398/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp,^while/lstm_cell_398/StatefulPartitionedCall*"
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
while_lstm_cell_398_16714343while_lstm_cell_398_16714343_0">
while_lstm_cell_398_16714345while_lstm_cell_398_16714345_0">
while_lstm_cell_398_16714347while_lstm_cell_398_16714347_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2Z
+while/lstm_cell_398/StatefulPartitionedCall+while/lstm_cell_398/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_16715194

inputs,
forward_lstm_132_16715024:	Ш,
forward_lstm_132_16715026:	2Ш(
forward_lstm_132_16715028:	Ш-
backward_lstm_132_16715184:	Ш-
backward_lstm_132_16715186:	2Ш)
backward_lstm_132_16715188:	Ш
identityЂ)backward_lstm_132/StatefulPartitionedCallЂ(forward_lstm_132/StatefulPartitionedCallп
(forward_lstm_132/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_132_16715024forward_lstm_132_16715026forward_lstm_132_16715028*
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
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_167150232*
(forward_lstm_132/StatefulPartitionedCallх
)backward_lstm_132/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_132_16715184backward_lstm_132_16715186backward_lstm_132_16715188*
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
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_167151832+
)backward_lstm_132/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisд
concatConcatV21forward_lstm_132/StatefulPartitionedCall:output:02backward_lstm_132/StatefulPartitionedCall:output:0concat/axis:output:0*
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
NoOpNoOp*^backward_lstm_132/StatefulPartitionedCall)^forward_lstm_132/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2V
)backward_lstm_132/StatefulPartitionedCall)backward_lstm_132/StatefulPartitionedCall2T
(forward_lstm_132/StatefulPartitionedCall(forward_lstm_132/StatefulPartitionedCall:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
]
Ў
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_16718410
inputs_0?
,lstm_cell_397_matmul_readvariableop_resource:	ШA
.lstm_cell_397_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_397_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_397/BiasAdd/ReadVariableOpЂ#lstm_cell_397/MatMul/ReadVariableOpЂ%lstm_cell_397/MatMul_1/ReadVariableOpЂwhileF
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
#lstm_cell_397/MatMul/ReadVariableOpReadVariableOp,lstm_cell_397_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_397/MatMul/ReadVariableOpА
lstm_cell_397/MatMulMatMulstrided_slice_2:output:0+lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/MatMulО
%lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_397_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_397/MatMul_1/ReadVariableOpЌ
lstm_cell_397/MatMul_1MatMulzeros:output:0-lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/MatMul_1Є
lstm_cell_397/addAddV2lstm_cell_397/MatMul:product:0 lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/addЗ
$lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_397_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_397/BiasAdd/ReadVariableOpБ
lstm_cell_397/BiasAddBiasAddlstm_cell_397/add:z:0,lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_397/BiasAdd
lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_397/split/split_dimї
lstm_cell_397/splitSplit&lstm_cell_397/split/split_dim:output:0lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_397/split
lstm_cell_397/SigmoidSigmoidlstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Sigmoid
lstm_cell_397/Sigmoid_1Sigmoidlstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Sigmoid_1
lstm_cell_397/mulMullstm_cell_397/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/mul
lstm_cell_397/ReluRelulstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Relu 
lstm_cell_397/mul_1Mullstm_cell_397/Sigmoid:y:0 lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/mul_1
lstm_cell_397/add_1AddV2lstm_cell_397/mul:z:0lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/add_1
lstm_cell_397/Sigmoid_2Sigmoidlstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Sigmoid_2
lstm_cell_397/Relu_1Relulstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/Relu_1Є
lstm_cell_397/mul_2Mullstm_cell_397/Sigmoid_2:y:0"lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_397/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_397_matmul_readvariableop_resource.lstm_cell_397_matmul_1_readvariableop_resource-lstm_cell_397_biasadd_readvariableop_resource*
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
while_body_16718326*
condR
while_cond_16718325*K
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
NoOpNoOp%^lstm_cell_397/BiasAdd/ReadVariableOp$^lstm_cell_397/MatMul/ReadVariableOp&^lstm_cell_397/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_397/BiasAdd/ReadVariableOp$lstm_cell_397/BiasAdd/ReadVariableOp2J
#lstm_cell_397/MatMul/ReadVariableOp#lstm_cell_397/MatMul/ReadVariableOp2N
%lstm_cell_397/MatMul_1/ReadVariableOp%lstm_cell_397/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
ЯY

%backward_lstm_132_while_body_16717242@
<backward_lstm_132_while_backward_lstm_132_while_loop_counterF
Bbackward_lstm_132_while_backward_lstm_132_while_maximum_iterations'
#backward_lstm_132_while_placeholder)
%backward_lstm_132_while_placeholder_1)
%backward_lstm_132_while_placeholder_2)
%backward_lstm_132_while_placeholder_3?
;backward_lstm_132_while_backward_lstm_132_strided_slice_1_0{
wbackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0Y
Fbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0:	Ш[
Hbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0:	2ШV
Gbackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0:	Ш$
 backward_lstm_132_while_identity&
"backward_lstm_132_while_identity_1&
"backward_lstm_132_while_identity_2&
"backward_lstm_132_while_identity_3&
"backward_lstm_132_while_identity_4&
"backward_lstm_132_while_identity_5=
9backward_lstm_132_while_backward_lstm_132_strided_slice_1y
ubackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensorW
Dbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource:	ШY
Fbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource:	2ШT
Ebackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource:	ШЂ<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOpЂ;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpЂ=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOpч
Ibackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2K
Ibackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeШ
;backward_lstm_132/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_132_while_placeholderRbackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02=
;backward_lstm_132/while/TensorArrayV2Read/TensorListGetItem
;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpReadVariableOpFbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02=
;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOpЂ
,backward_lstm_132/while/lstm_cell_398/MatMulMatMulBbackward_lstm_132/while/TensorArrayV2Read/TensorListGetItem:item:0Cbackward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,backward_lstm_132/while/lstm_cell_398/MatMul
=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOpHbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02?
=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp
.backward_lstm_132/while/lstm_cell_398/MatMul_1MatMul%backward_lstm_132_while_placeholder_2Ebackward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ20
.backward_lstm_132/while/lstm_cell_398/MatMul_1
)backward_lstm_132/while/lstm_cell_398/addAddV26backward_lstm_132/while/lstm_cell_398/MatMul:product:08backward_lstm_132/while/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_132/while/lstm_cell_398/add
<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOpGbackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02>
<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp
-backward_lstm_132/while/lstm_cell_398/BiasAddBiasAdd-backward_lstm_132/while/lstm_cell_398/add:z:0Dbackward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-backward_lstm_132/while/lstm_cell_398/BiasAddА
5backward_lstm_132/while/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5backward_lstm_132/while/lstm_cell_398/split/split_dimз
+backward_lstm_132/while/lstm_cell_398/splitSplit>backward_lstm_132/while/lstm_cell_398/split/split_dim:output:06backward_lstm_132/while/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2-
+backward_lstm_132/while/lstm_cell_398/splitб
-backward_lstm_132/while/lstm_cell_398/SigmoidSigmoid4backward_lstm_132/while/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22/
-backward_lstm_132/while/lstm_cell_398/Sigmoidе
/backward_lstm_132/while/lstm_cell_398/Sigmoid_1Sigmoid4backward_lstm_132/while/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_132/while/lstm_cell_398/Sigmoid_1ы
)backward_lstm_132/while/lstm_cell_398/mulMul3backward_lstm_132/while/lstm_cell_398/Sigmoid_1:y:0%backward_lstm_132_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_132/while/lstm_cell_398/mulШ
*backward_lstm_132/while/lstm_cell_398/ReluRelu4backward_lstm_132/while/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22,
*backward_lstm_132/while/lstm_cell_398/Relu
+backward_lstm_132/while/lstm_cell_398/mul_1Mul1backward_lstm_132/while/lstm_cell_398/Sigmoid:y:08backward_lstm_132/while/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_132/while/lstm_cell_398/mul_1ѕ
+backward_lstm_132/while/lstm_cell_398/add_1AddV2-backward_lstm_132/while/lstm_cell_398/mul:z:0/backward_lstm_132/while/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_132/while/lstm_cell_398/add_1е
/backward_lstm_132/while/lstm_cell_398/Sigmoid_2Sigmoid4backward_lstm_132/while/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ221
/backward_lstm_132/while/lstm_cell_398/Sigmoid_2Ч
,backward_lstm_132/while/lstm_cell_398/Relu_1Relu/backward_lstm_132/while/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,backward_lstm_132/while/lstm_cell_398/Relu_1
+backward_lstm_132/while/lstm_cell_398/mul_2Mul3backward_lstm_132/while/lstm_cell_398/Sigmoid_2:y:0:backward_lstm_132/while/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_132/while/lstm_cell_398/mul_2Л
<backward_lstm_132/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_132_while_placeholder_1#backward_lstm_132_while_placeholder/backward_lstm_132/while/lstm_cell_398/mul_2:z:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_132/while/TensorArrayV2Write/TensorListSetItem
backward_lstm_132/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_132/while/add/yБ
backward_lstm_132/while/addAddV2#backward_lstm_132_while_placeholder&backward_lstm_132/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/while/add
backward_lstm_132/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_132/while/add_1/yа
backward_lstm_132/while/add_1AddV2<backward_lstm_132_while_backward_lstm_132_while_loop_counter(backward_lstm_132/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/while/add_1Г
 backward_lstm_132/while/IdentityIdentity!backward_lstm_132/while/add_1:z:0^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_132/while/Identityи
"backward_lstm_132/while/Identity_1IdentityBbackward_lstm_132_while_backward_lstm_132_while_maximum_iterations^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_132/while/Identity_1Е
"backward_lstm_132/while/Identity_2Identitybackward_lstm_132/while/add:z:0^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_132/while/Identity_2т
"backward_lstm_132/while/Identity_3IdentityLbackward_lstm_132/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_132/while/Identity_3ж
"backward_lstm_132/while/Identity_4Identity/backward_lstm_132/while/lstm_cell_398/mul_2:z:0^backward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_132/while/Identity_4ж
"backward_lstm_132/while/Identity_5Identity/backward_lstm_132/while/lstm_cell_398/add_1:z:0^backward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_132/while/Identity_5Л
backward_lstm_132/while/NoOpNoOp=^backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp<^backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp>^backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_132/while/NoOp"x
9backward_lstm_132_while_backward_lstm_132_strided_slice_1;backward_lstm_132_while_backward_lstm_132_strided_slice_1_0"M
 backward_lstm_132_while_identity)backward_lstm_132/while/Identity:output:0"Q
"backward_lstm_132_while_identity_1+backward_lstm_132/while/Identity_1:output:0"Q
"backward_lstm_132_while_identity_2+backward_lstm_132/while/Identity_2:output:0"Q
"backward_lstm_132_while_identity_3+backward_lstm_132/while/Identity_3:output:0"Q
"backward_lstm_132_while_identity_4+backward_lstm_132/while/Identity_4:output:0"Q
"backward_lstm_132_while_identity_5+backward_lstm_132/while/Identity_5:output:0"
Ebackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resourceGbackward_lstm_132_while_lstm_cell_398_biasadd_readvariableop_resource_0"
Fbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resourceHbackward_lstm_132_while_lstm_cell_398_matmul_1_readvariableop_resource_0"
Dbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resourceFbackward_lstm_132_while_lstm_cell_398_matmul_readvariableop_resource_0"№
ubackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2|
<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp<backward_lstm_132/while/lstm_cell_398/BiasAdd/ReadVariableOp2z
;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp;backward_lstm_132/while/lstm_cell_398/MatMul/ReadVariableOp2~
=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp=backward_lstm_132/while/lstm_cell_398/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_16713756

inputs)
lstm_cell_397_16713674:	Ш)
lstm_cell_397_16713676:	2Ш%
lstm_cell_397_16713678:	Ш
identityЂ%lstm_cell_397/StatefulPartitionedCallЂwhileD
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
%lstm_cell_397/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_397_16713674lstm_cell_397_16713676lstm_cell_397_16713678*
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
K__inference_lstm_cell_397_layer_call_and_return_conditional_losses_167136732'
%lstm_cell_397/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_397_16713674lstm_cell_397_16713676lstm_cell_397_16713678*
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
while_body_16713687*
condR
while_cond_16713686*K
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
NoOpNoOp&^lstm_cell_397/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2N
%lstm_cell_397/StatefulPartitionedCall%lstm_cell_397/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
e
Т
$forward_lstm_132_while_body_16717410>
:forward_lstm_132_while_forward_lstm_132_while_loop_counterD
@forward_lstm_132_while_forward_lstm_132_while_maximum_iterations&
"forward_lstm_132_while_placeholder(
$forward_lstm_132_while_placeholder_1(
$forward_lstm_132_while_placeholder_2(
$forward_lstm_132_while_placeholder_3(
$forward_lstm_132_while_placeholder_4=
9forward_lstm_132_while_forward_lstm_132_strided_slice_1_0y
uforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_132_while_greater_forward_lstm_132_cast_0X
Eforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0:	ШZ
Gforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0:	2ШU
Fforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0:	Ш#
forward_lstm_132_while_identity%
!forward_lstm_132_while_identity_1%
!forward_lstm_132_while_identity_2%
!forward_lstm_132_while_identity_3%
!forward_lstm_132_while_identity_4%
!forward_lstm_132_while_identity_5%
!forward_lstm_132_while_identity_6;
7forward_lstm_132_while_forward_lstm_132_strided_slice_1w
sforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_132_while_greater_forward_lstm_132_castV
Cforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource:	ШX
Eforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource:	2ШS
Dforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource:	ШЂ;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOpЂ:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOpЂ<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOpх
Hforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2J
Hforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
:forward_lstm_132/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_132_while_placeholderQforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02<
:forward_lstm_132/while/TensorArrayV2Read/TensorListGetItemе
forward_lstm_132/while/GreaterGreater6forward_lstm_132_while_greater_forward_lstm_132_cast_0"forward_lstm_132_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2 
forward_lstm_132/while/Greaterџ
:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOpReadVariableOpEforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02<
:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp
+forward_lstm_132/while/lstm_cell_397/MatMulMatMulAforward_lstm_132/while/TensorArrayV2Read/TensorListGetItem:item:0Bforward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+forward_lstm_132/while/lstm_cell_397/MatMul
<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOpGforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02>
<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp
-forward_lstm_132/while/lstm_cell_397/MatMul_1MatMul$forward_lstm_132_while_placeholder_3Dforward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2/
-forward_lstm_132/while/lstm_cell_397/MatMul_1
(forward_lstm_132/while/lstm_cell_397/addAddV25forward_lstm_132/while/lstm_cell_397/MatMul:product:07forward_lstm_132/while/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_132/while/lstm_cell_397/addў
;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOpFforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02=
;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp
,forward_lstm_132/while/lstm_cell_397/BiasAddBiasAdd,forward_lstm_132/while/lstm_cell_397/add:z:0Cforward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2.
,forward_lstm_132/while/lstm_cell_397/BiasAddЎ
4forward_lstm_132/while/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4forward_lstm_132/while/lstm_cell_397/split/split_dimг
*forward_lstm_132/while/lstm_cell_397/splitSplit=forward_lstm_132/while/lstm_cell_397/split/split_dim:output:05forward_lstm_132/while/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2,
*forward_lstm_132/while/lstm_cell_397/splitЮ
,forward_lstm_132/while/lstm_cell_397/SigmoidSigmoid3forward_lstm_132/while/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22.
,forward_lstm_132/while/lstm_cell_397/Sigmoidв
.forward_lstm_132/while/lstm_cell_397/Sigmoid_1Sigmoid3forward_lstm_132/while/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_132/while/lstm_cell_397/Sigmoid_1ч
(forward_lstm_132/while/lstm_cell_397/mulMul2forward_lstm_132/while/lstm_cell_397/Sigmoid_1:y:0$forward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_132/while/lstm_cell_397/mulХ
)forward_lstm_132/while/lstm_cell_397/ReluRelu3forward_lstm_132/while/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22+
)forward_lstm_132/while/lstm_cell_397/Reluќ
*forward_lstm_132/while/lstm_cell_397/mul_1Mul0forward_lstm_132/while/lstm_cell_397/Sigmoid:y:07forward_lstm_132/while/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_132/while/lstm_cell_397/mul_1ё
*forward_lstm_132/while/lstm_cell_397/add_1AddV2,forward_lstm_132/while/lstm_cell_397/mul:z:0.forward_lstm_132/while/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_132/while/lstm_cell_397/add_1в
.forward_lstm_132/while/lstm_cell_397/Sigmoid_2Sigmoid3forward_lstm_132/while/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ220
.forward_lstm_132/while/lstm_cell_397/Sigmoid_2Ф
+forward_lstm_132/while/lstm_cell_397/Relu_1Relu.forward_lstm_132/while/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22-
+forward_lstm_132/while/lstm_cell_397/Relu_1
*forward_lstm_132/while/lstm_cell_397/mul_2Mul2forward_lstm_132/while/lstm_cell_397/Sigmoid_2:y:09forward_lstm_132/while/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_132/while/lstm_cell_397/mul_2є
forward_lstm_132/while/SelectSelect"forward_lstm_132/while/Greater:z:0.forward_lstm_132/while/lstm_cell_397/mul_2:z:0$forward_lstm_132_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/while/Selectј
forward_lstm_132/while/Select_1Select"forward_lstm_132/while/Greater:z:0.forward_lstm_132/while/lstm_cell_397/mul_2:z:0$forward_lstm_132_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_132/while/Select_1ј
forward_lstm_132/while/Select_2Select"forward_lstm_132/while/Greater:z:0.forward_lstm_132/while/lstm_cell_397/add_1:z:0$forward_lstm_132_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_132/while/Select_2Ў
;forward_lstm_132/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_132_while_placeholder_1"forward_lstm_132_while_placeholder&forward_lstm_132/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_132/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_132/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_132/while/add/y­
forward_lstm_132/while/addAddV2"forward_lstm_132_while_placeholder%forward_lstm_132/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/while/add
forward_lstm_132/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_132/while/add_1/yЫ
forward_lstm_132/while/add_1AddV2:forward_lstm_132_while_forward_lstm_132_while_loop_counter'forward_lstm_132/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/while/add_1Џ
forward_lstm_132/while/IdentityIdentity forward_lstm_132/while/add_1:z:0^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_132/while/Identityг
!forward_lstm_132/while/Identity_1Identity@forward_lstm_132_while_forward_lstm_132_while_maximum_iterations^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_132/while/Identity_1Б
!forward_lstm_132/while/Identity_2Identityforward_lstm_132/while/add:z:0^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_132/while/Identity_2о
!forward_lstm_132/while/Identity_3IdentityKforward_lstm_132/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_132/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_132/while/Identity_3Ъ
!forward_lstm_132/while/Identity_4Identity&forward_lstm_132/while/Select:output:0^forward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_132/while/Identity_4Ь
!forward_lstm_132/while/Identity_5Identity(forward_lstm_132/while/Select_1:output:0^forward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_132/while/Identity_5Ь
!forward_lstm_132/while/Identity_6Identity(forward_lstm_132/while/Select_2:output:0^forward_lstm_132/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_132/while/Identity_6Ж
forward_lstm_132/while/NoOpNoOp<^forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp;^forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp=^forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_132/while/NoOp"t
7forward_lstm_132_while_forward_lstm_132_strided_slice_19forward_lstm_132_while_forward_lstm_132_strided_slice_1_0"n
4forward_lstm_132_while_greater_forward_lstm_132_cast6forward_lstm_132_while_greater_forward_lstm_132_cast_0"K
forward_lstm_132_while_identity(forward_lstm_132/while/Identity:output:0"O
!forward_lstm_132_while_identity_1*forward_lstm_132/while/Identity_1:output:0"O
!forward_lstm_132_while_identity_2*forward_lstm_132/while/Identity_2:output:0"O
!forward_lstm_132_while_identity_3*forward_lstm_132/while/Identity_3:output:0"O
!forward_lstm_132_while_identity_4*forward_lstm_132/while/Identity_4:output:0"O
!forward_lstm_132_while_identity_5*forward_lstm_132/while/Identity_5:output:0"O
!forward_lstm_132_while_identity_6*forward_lstm_132/while/Identity_6:output:0"
Dforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resourceFforward_lstm_132_while_lstm_cell_397_biasadd_readvariableop_resource_0"
Eforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resourceGforward_lstm_132_while_lstm_cell_397_matmul_1_readvariableop_resource_0"
Cforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resourceEforward_lstm_132_while_lstm_cell_397_matmul_readvariableop_resource_0"ь
sforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensoruforward_lstm_132_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_132_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2z
;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp;forward_lstm_132/while/lstm_cell_397/BiasAdd/ReadVariableOp2x
:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp:forward_lstm_132/while/lstm_cell_397/MatMul/ReadVariableOp2|
<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp<forward_lstm_132/while/lstm_cell_397/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
_
Џ
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_16719062
inputs_0?
,lstm_cell_398_matmul_readvariableop_resource:	ШA
.lstm_cell_398_matmul_1_readvariableop_resource:	2Ш<
-lstm_cell_398_biasadd_readvariableop_resource:	Ш
identityЂ$lstm_cell_398/BiasAdd/ReadVariableOpЂ#lstm_cell_398/MatMul/ReadVariableOpЂ%lstm_cell_398/MatMul_1/ReadVariableOpЂwhileF
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
#lstm_cell_398/MatMul/ReadVariableOpReadVariableOp,lstm_cell_398_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02%
#lstm_cell_398/MatMul/ReadVariableOpА
lstm_cell_398/MatMulMatMulstrided_slice_2:output:0+lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/MatMulО
%lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_398_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02'
%lstm_cell_398/MatMul_1/ReadVariableOpЌ
lstm_cell_398/MatMul_1MatMulzeros:output:0-lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/MatMul_1Є
lstm_cell_398/addAddV2lstm_cell_398/MatMul:product:0 lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/addЗ
$lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_398_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02&
$lstm_cell_398/BiasAdd/ReadVariableOpБ
lstm_cell_398/BiasAddBiasAddlstm_cell_398/add:z:0,lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_398/BiasAdd
lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_398/split/split_dimї
lstm_cell_398/splitSplit&lstm_cell_398/split/split_dim:output:0lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_398/split
lstm_cell_398/SigmoidSigmoidlstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Sigmoid
lstm_cell_398/Sigmoid_1Sigmoidlstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Sigmoid_1
lstm_cell_398/mulMullstm_cell_398/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/mul
lstm_cell_398/ReluRelulstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Relu 
lstm_cell_398/mul_1Mullstm_cell_398/Sigmoid:y:0 lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/mul_1
lstm_cell_398/add_1AddV2lstm_cell_398/mul:z:0lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/add_1
lstm_cell_398/Sigmoid_2Sigmoidlstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Sigmoid_2
lstm_cell_398/Relu_1Relulstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/Relu_1Є
lstm_cell_398/mul_2Mullstm_cell_398/Sigmoid_2:y:0"lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_398/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_398_matmul_readvariableop_resource.lstm_cell_398_matmul_1_readvariableop_resource-lstm_cell_398_biasadd_readvariableop_resource*
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
while_body_16718978*
condR
while_cond_16718977*K
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
NoOpNoOp%^lstm_cell_398/BiasAdd/ReadVariableOp$^lstm_cell_398/MatMul/ReadVariableOp&^lstm_cell_398/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_398/BiasAdd/ReadVariableOp$lstm_cell_398/BiasAdd/ReadVariableOp2J
#lstm_cell_398/MatMul/ReadVariableOp#lstm_cell_398/MatMul/ReadVariableOp2N
%lstm_cell_398/MatMul_1/ReadVariableOp%lstm_cell_398/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
М
љ
0__inference_lstm_cell_398_layer_call_fn_16719500

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
K__inference_lstm_cell_398_layer_call_and_return_conditional_losses_167144512
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
ъ
М
%backward_lstm_132_while_cond_16717946@
<backward_lstm_132_while_backward_lstm_132_while_loop_counterF
Bbackward_lstm_132_while_backward_lstm_132_while_maximum_iterations'
#backward_lstm_132_while_placeholder)
%backward_lstm_132_while_placeholder_1)
%backward_lstm_132_while_placeholder_2)
%backward_lstm_132_while_placeholder_3)
%backward_lstm_132_while_placeholder_4B
>backward_lstm_132_while_less_backward_lstm_132_strided_slice_1Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16717946___redundant_placeholder0Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16717946___redundant_placeholder1Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16717946___redundant_placeholder2Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16717946___redundant_placeholder3Z
Vbackward_lstm_132_while_backward_lstm_132_while_cond_16717946___redundant_placeholder4$
 backward_lstm_132_while_identity
Ъ
backward_lstm_132/while/LessLess#backward_lstm_132_while_placeholder>backward_lstm_132_while_less_backward_lstm_132_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_132/while/Less
 backward_lstm_132/while/IdentityIdentity backward_lstm_132/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_132/while/Identity"M
 backward_lstm_132_while_identity)backward_lstm_132/while/Identity:output:0*(
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
р
Р
3__inference_forward_lstm_132_layer_call_fn_16718097

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
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_167150232
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
Ъ

Э
&__inference_signature_wrapper_16716654

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
#__inference__wrapped_model_167135982
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
р
Р
3__inference_forward_lstm_132_layer_call_fn_16718108

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
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_167155482
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
ы	

4__inference_bidirectional_132_layer_call_fn_16716671
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
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_167151942
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
љ?
л
while_body_16719284
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_398_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_398_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_398_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_398_matmul_readvariableop_resource:	ШG
4while_lstm_cell_398_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_398_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_398/BiasAdd/ReadVariableOpЂ)while/lstm_cell_398/MatMul/ReadVariableOpЂ+while/lstm_cell_398/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_398/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_398_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_398/MatMul/ReadVariableOpк
while/lstm_cell_398/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/MatMulв
+while/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_398_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_398/MatMul_1/ReadVariableOpУ
while/lstm_cell_398/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/MatMul_1М
while/lstm_cell_398/addAddV2$while/lstm_cell_398/MatMul:product:0&while/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/addЫ
*while/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_398_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_398/BiasAdd/ReadVariableOpЩ
while/lstm_cell_398/BiasAddBiasAddwhile/lstm_cell_398/add:z:02while/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_398/BiasAdd
#while/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_398/split/split_dim
while/lstm_cell_398/splitSplit,while/lstm_cell_398/split/split_dim:output:0$while/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_398/split
while/lstm_cell_398/SigmoidSigmoid"while/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Sigmoid
while/lstm_cell_398/Sigmoid_1Sigmoid"while/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Sigmoid_1Ѓ
while/lstm_cell_398/mulMul!while/lstm_cell_398/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/mul
while/lstm_cell_398/ReluRelu"while/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/ReluИ
while/lstm_cell_398/mul_1Mulwhile/lstm_cell_398/Sigmoid:y:0&while/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/mul_1­
while/lstm_cell_398/add_1AddV2while/lstm_cell_398/mul:z:0while/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/add_1
while/lstm_cell_398/Sigmoid_2Sigmoid"while/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Sigmoid_2
while/lstm_cell_398/Relu_1Reluwhile/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/Relu_1М
while/lstm_cell_398/mul_2Mul!while/lstm_cell_398/Sigmoid_2:y:0(while/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_398/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_398/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_398/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_398/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_398/BiasAdd/ReadVariableOp*^while/lstm_cell_398/MatMul/ReadVariableOp,^while/lstm_cell_398/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_398_biasadd_readvariableop_resource5while_lstm_cell_398_biasadd_readvariableop_resource_0"n
4while_lstm_cell_398_matmul_1_readvariableop_resource6while_lstm_cell_398_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_398_matmul_readvariableop_resource4while_lstm_cell_398_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_398/BiasAdd/ReadVariableOp*while/lstm_cell_398/BiasAdd/ReadVariableOp2V
)while/lstm_cell_398/MatMul/ReadVariableOp)while/lstm_cell_398/MatMul/ReadVariableOp2Z
+while/lstm_cell_398/MatMul_1/ReadVariableOp+while/lstm_cell_398/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_cond_16719130
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16719130___redundant_placeholder06
2while_while_cond_16719130___redundant_placeholder16
2while_while_cond_16719130___redundant_placeholder26
2while_while_cond_16719130___redundant_placeholder3
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
while_body_16718477
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_397_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_397_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_397_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_397_matmul_readvariableop_resource:	ШG
4while_lstm_cell_397_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_397_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_397/BiasAdd/ReadVariableOpЂ)while/lstm_cell_397/MatMul/ReadVariableOpЂ+while/lstm_cell_397/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_397/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_397_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_397/MatMul/ReadVariableOpк
while/lstm_cell_397/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/MatMulв
+while/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_397_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_397/MatMul_1/ReadVariableOpУ
while/lstm_cell_397/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/MatMul_1М
while/lstm_cell_397/addAddV2$while/lstm_cell_397/MatMul:product:0&while/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/addЫ
*while/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_397_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_397/BiasAdd/ReadVariableOpЩ
while/lstm_cell_397/BiasAddBiasAddwhile/lstm_cell_397/add:z:02while/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/BiasAdd
#while/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_397/split/split_dim
while/lstm_cell_397/splitSplit,while/lstm_cell_397/split/split_dim:output:0$while/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_397/split
while/lstm_cell_397/SigmoidSigmoid"while/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Sigmoid
while/lstm_cell_397/Sigmoid_1Sigmoid"while/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Sigmoid_1Ѓ
while/lstm_cell_397/mulMul!while/lstm_cell_397/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/mul
while/lstm_cell_397/ReluRelu"while/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/ReluИ
while/lstm_cell_397/mul_1Mulwhile/lstm_cell_397/Sigmoid:y:0&while/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/mul_1­
while/lstm_cell_397/add_1AddV2while/lstm_cell_397/mul:z:0while/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/add_1
while/lstm_cell_397/Sigmoid_2Sigmoid"while/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Sigmoid_2
while/lstm_cell_397/Relu_1Reluwhile/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Relu_1М
while/lstm_cell_397/mul_2Mul!while/lstm_cell_397/Sigmoid_2:y:0(while/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_397/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_397/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_397/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_397/BiasAdd/ReadVariableOp*^while/lstm_cell_397/MatMul/ReadVariableOp,^while/lstm_cell_397/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_397_biasadd_readvariableop_resource5while_lstm_cell_397_biasadd_readvariableop_resource_0"n
4while_lstm_cell_397_matmul_1_readvariableop_resource6while_lstm_cell_397_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_397_matmul_readvariableop_resource4while_lstm_cell_397_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_397/BiasAdd/ReadVariableOp*while/lstm_cell_397/BiasAdd/ReadVariableOp2V
)while/lstm_cell_397/MatMul/ReadVariableOp)while/lstm_cell_397/MatMul/ReadVariableOp2Z
+while/lstm_cell_397/MatMul_1/ReadVariableOp+while/lstm_cell_397/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
K__inference_lstm_cell_397_layer_call_and_return_conditional_losses_16719466

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
while_cond_16715290
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16715290___redundant_placeholder06
2while_while_cond_16715290___redundant_placeholder16
2while_while_cond_16715290___redundant_placeholder26
2while_while_cond_16715290___redundant_placeholder3
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
№?
л
while_body_16718175
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_397_matmul_readvariableop_resource_0:	ШI
6while_lstm_cell_397_matmul_1_readvariableop_resource_0:	2ШD
5while_lstm_cell_397_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_397_matmul_readvariableop_resource:	ШG
4while_lstm_cell_397_matmul_1_readvariableop_resource:	2ШB
3while_lstm_cell_397_biasadd_readvariableop_resource:	ШЂ*while/lstm_cell_397/BiasAdd/ReadVariableOpЂ)while/lstm_cell_397/MatMul/ReadVariableOpЂ+while/lstm_cell_397/MatMul_1/ReadVariableOpУ
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
)while/lstm_cell_397/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_397_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02+
)while/lstm_cell_397/MatMul/ReadVariableOpк
while/lstm_cell_397/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/MatMulв
+while/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_397_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02-
+while/lstm_cell_397/MatMul_1/ReadVariableOpУ
while/lstm_cell_397/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/MatMul_1М
while/lstm_cell_397/addAddV2$while/lstm_cell_397/MatMul:product:0&while/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/addЫ
*while/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_397_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02,
*while/lstm_cell_397/BiasAdd/ReadVariableOpЩ
while/lstm_cell_397/BiasAddBiasAddwhile/lstm_cell_397/add:z:02while/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_397/BiasAdd
#while/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_397/split/split_dim
while/lstm_cell_397/splitSplit,while/lstm_cell_397/split/split_dim:output:0$while/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_397/split
while/lstm_cell_397/SigmoidSigmoid"while/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Sigmoid
while/lstm_cell_397/Sigmoid_1Sigmoid"while/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Sigmoid_1Ѓ
while/lstm_cell_397/mulMul!while/lstm_cell_397/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/mul
while/lstm_cell_397/ReluRelu"while/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/ReluИ
while/lstm_cell_397/mul_1Mulwhile/lstm_cell_397/Sigmoid:y:0&while/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/mul_1­
while/lstm_cell_397/add_1AddV2while/lstm_cell_397/mul:z:0while/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/add_1
while/lstm_cell_397/Sigmoid_2Sigmoid"while/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Sigmoid_2
while/lstm_cell_397/Relu_1Reluwhile/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/Relu_1М
while/lstm_cell_397/mul_2Mul!while/lstm_cell_397/Sigmoid_2:y:0(while/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_397/mul_2с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_397/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_397/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_397/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5с

while/NoOpNoOp+^while/lstm_cell_397/BiasAdd/ReadVariableOp*^while/lstm_cell_397/MatMul/ReadVariableOp,^while/lstm_cell_397/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_397_biasadd_readvariableop_resource5while_lstm_cell_397_biasadd_readvariableop_resource_0"n
4while_lstm_cell_397_matmul_1_readvariableop_resource6while_lstm_cell_397_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_397_matmul_readvariableop_resource4while_lstm_cell_397_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2X
*while/lstm_cell_397/BiasAdd/ReadVariableOp*while/lstm_cell_397/BiasAdd/ReadVariableOp2V
)while/lstm_cell_397/MatMul/ReadVariableOp)while/lstm_cell_397/MatMul/ReadVariableOp2Z
+while/lstm_cell_397/MatMul_1/ReadVariableOp+while/lstm_cell_397/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_16716474

inputs
inputs_1	P
=forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource:	ШR
?forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource:	2ШM
>forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource:	ШQ
>backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource:	ШS
@backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource:	2ШN
?backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource:	Ш
identityЂ6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpЂ5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpЂ7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpЂbackward_lstm_132/whileЂ5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpЂ4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpЂ6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOpЂforward_lstm_132/while
%forward_lstm_132/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_132/RaggedToTensor/zeros
%forward_lstm_132/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2'
%forward_lstm_132/RaggedToTensor/Const
4forward_lstm_132/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_132/RaggedToTensor/Const:output:0inputs.forward_lstm_132/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_132/RaggedToTensor/RaggedTensorToTensorФ
;forward_lstm_132/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_132/RaggedNestedRowLengths/strided_slice/stackШ
=forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1Ш
=forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2Љ
5forward_lstm_132/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_132/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask27
5forward_lstm_132/RaggedNestedRowLengths/strided_sliceШ
=forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stackе
?forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2A
?forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1Ь
?forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2Е
7forward_lstm_132/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask29
7forward_lstm_132/RaggedNestedRowLengths/strided_slice_1
+forward_lstm_132/RaggedNestedRowLengths/subSub>forward_lstm_132/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_132/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2-
+forward_lstm_132/RaggedNestedRowLengths/subЄ
forward_lstm_132/CastCast/forward_lstm_132/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_132/Cast
forward_lstm_132/ShapeShape=forward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_132/Shape
$forward_lstm_132/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_132/strided_slice/stack
&forward_lstm_132/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_132/strided_slice/stack_1
&forward_lstm_132/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_132/strided_slice/stack_2Ш
forward_lstm_132/strided_sliceStridedSliceforward_lstm_132/Shape:output:0-forward_lstm_132/strided_slice/stack:output:0/forward_lstm_132/strided_slice/stack_1:output:0/forward_lstm_132/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_132/strided_slice~
forward_lstm_132/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_132/zeros/mul/yА
forward_lstm_132/zeros/mulMul'forward_lstm_132/strided_slice:output:0%forward_lstm_132/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros/mul
forward_lstm_132/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_132/zeros/Less/yЋ
forward_lstm_132/zeros/LessLessforward_lstm_132/zeros/mul:z:0&forward_lstm_132/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros/Less
forward_lstm_132/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_132/zeros/packed/1Ч
forward_lstm_132/zeros/packedPack'forward_lstm_132/strided_slice:output:0(forward_lstm_132/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_132/zeros/packed
forward_lstm_132/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_132/zeros/ConstЙ
forward_lstm_132/zerosFill&forward_lstm_132/zeros/packed:output:0%forward_lstm_132/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/zeros
forward_lstm_132/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_132/zeros_1/mul/yЖ
forward_lstm_132/zeros_1/mulMul'forward_lstm_132/strided_slice:output:0'forward_lstm_132/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros_1/mul
forward_lstm_132/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2!
forward_lstm_132/zeros_1/Less/yГ
forward_lstm_132/zeros_1/LessLess forward_lstm_132/zeros_1/mul:z:0(forward_lstm_132/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_132/zeros_1/Less
!forward_lstm_132/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_132/zeros_1/packed/1Э
forward_lstm_132/zeros_1/packedPack'forward_lstm_132/strided_slice:output:0*forward_lstm_132/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_132/zeros_1/packed
forward_lstm_132/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_132/zeros_1/ConstС
forward_lstm_132/zeros_1Fill(forward_lstm_132/zeros_1/packed:output:0'forward_lstm_132/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/zeros_1
forward_lstm_132/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_132/transpose/permэ
forward_lstm_132/transpose	Transpose=forward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_132/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_132/transpose
forward_lstm_132/Shape_1Shapeforward_lstm_132/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_132/Shape_1
&forward_lstm_132/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_132/strided_slice_1/stack
(forward_lstm_132/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_1/stack_1
(forward_lstm_132/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_1/stack_2д
 forward_lstm_132/strided_slice_1StridedSlice!forward_lstm_132/Shape_1:output:0/forward_lstm_132/strided_slice_1/stack:output:01forward_lstm_132/strided_slice_1/stack_1:output:01forward_lstm_132/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_132/strided_slice_1Ї
,forward_lstm_132/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,forward_lstm_132/TensorArrayV2/element_shapeі
forward_lstm_132/TensorArrayV2TensorListReserve5forward_lstm_132/TensorArrayV2/element_shape:output:0)forward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_132/TensorArrayV2с
Fforward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fforward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8forward_lstm_132/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_132/transpose:y:0Oforward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_132/TensorArrayUnstack/TensorListFromTensor
&forward_lstm_132/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_132/strided_slice_2/stack
(forward_lstm_132/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_2/stack_1
(forward_lstm_132/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_2/stack_2т
 forward_lstm_132/strided_slice_2StridedSliceforward_lstm_132/transpose:y:0/forward_lstm_132/strided_slice_2/stack:output:01forward_lstm_132/strided_slice_2/stack_1:output:01forward_lstm_132/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 forward_lstm_132/strided_slice_2ы
4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpReadVariableOp=forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype026
4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOpє
%forward_lstm_132/lstm_cell_397/MatMulMatMul)forward_lstm_132/strided_slice_2:output:0<forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%forward_lstm_132/lstm_cell_397/MatMulё
6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOpReadVariableOp?forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype028
6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp№
'forward_lstm_132/lstm_cell_397/MatMul_1MatMulforward_lstm_132/zeros:output:0>forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_132/lstm_cell_397/MatMul_1ш
"forward_lstm_132/lstm_cell_397/addAddV2/forward_lstm_132/lstm_cell_397/MatMul:product:01forward_lstm_132/lstm_cell_397/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_132/lstm_cell_397/addъ
5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpReadVariableOp>forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype027
5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOpѕ
&forward_lstm_132/lstm_cell_397/BiasAddBiasAdd&forward_lstm_132/lstm_cell_397/add:z:0=forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&forward_lstm_132/lstm_cell_397/BiasAddЂ
.forward_lstm_132/lstm_cell_397/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.forward_lstm_132/lstm_cell_397/split/split_dimЛ
$forward_lstm_132/lstm_cell_397/splitSplit7forward_lstm_132/lstm_cell_397/split/split_dim:output:0/forward_lstm_132/lstm_cell_397/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$forward_lstm_132/lstm_cell_397/splitМ
&forward_lstm_132/lstm_cell_397/SigmoidSigmoid-forward_lstm_132/lstm_cell_397/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_132/lstm_cell_397/SigmoidР
(forward_lstm_132/lstm_cell_397/Sigmoid_1Sigmoid-forward_lstm_132/lstm_cell_397/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_132/lstm_cell_397/Sigmoid_1в
"forward_lstm_132/lstm_cell_397/mulMul,forward_lstm_132/lstm_cell_397/Sigmoid_1:y:0!forward_lstm_132/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_132/lstm_cell_397/mulГ
#forward_lstm_132/lstm_cell_397/ReluRelu-forward_lstm_132/lstm_cell_397/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22%
#forward_lstm_132/lstm_cell_397/Reluф
$forward_lstm_132/lstm_cell_397/mul_1Mul*forward_lstm_132/lstm_cell_397/Sigmoid:y:01forward_lstm_132/lstm_cell_397/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_132/lstm_cell_397/mul_1й
$forward_lstm_132/lstm_cell_397/add_1AddV2&forward_lstm_132/lstm_cell_397/mul:z:0(forward_lstm_132/lstm_cell_397/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_132/lstm_cell_397/add_1Р
(forward_lstm_132/lstm_cell_397/Sigmoid_2Sigmoid-forward_lstm_132/lstm_cell_397/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_132/lstm_cell_397/Sigmoid_2В
%forward_lstm_132/lstm_cell_397/Relu_1Relu(forward_lstm_132/lstm_cell_397/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_132/lstm_cell_397/Relu_1ш
$forward_lstm_132/lstm_cell_397/mul_2Mul,forward_lstm_132/lstm_cell_397/Sigmoid_2:y:03forward_lstm_132/lstm_cell_397/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_132/lstm_cell_397/mul_2Б
.forward_lstm_132/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   20
.forward_lstm_132/TensorArrayV2_1/element_shapeќ
 forward_lstm_132/TensorArrayV2_1TensorListReserve7forward_lstm_132/TensorArrayV2_1/element_shape:output:0)forward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_132/TensorArrayV2_1p
forward_lstm_132/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_132/timeЃ
forward_lstm_132/zeros_like	ZerosLike(forward_lstm_132/lstm_cell_397/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_132/zeros_likeЁ
)forward_lstm_132/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)forward_lstm_132/while/maximum_iterations
#forward_lstm_132/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_132/while/loop_counter	
forward_lstm_132/whileWhile,forward_lstm_132/while/loop_counter:output:02forward_lstm_132/while/maximum_iterations:output:0forward_lstm_132/time:output:0)forward_lstm_132/TensorArrayV2_1:handle:0forward_lstm_132/zeros_like:y:0forward_lstm_132/zeros:output:0!forward_lstm_132/zeros_1:output:0)forward_lstm_132/strided_slice_1:output:0Hforward_lstm_132/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_132/Cast:y:0=forward_lstm_132_lstm_cell_397_matmul_readvariableop_resource?forward_lstm_132_lstm_cell_397_matmul_1_readvariableop_resource>forward_lstm_132_lstm_cell_397_biasadd_readvariableop_resource*
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
$forward_lstm_132_while_body_16716198*0
cond(R&
$forward_lstm_132_while_cond_16716197*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_132/whileз
Aforward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2C
Aforward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeЕ
3forward_lstm_132/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_132/while:output:3Jforward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype025
3forward_lstm_132/TensorArrayV2Stack/TensorListStackЃ
&forward_lstm_132/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&forward_lstm_132/strided_slice_3/stack
(forward_lstm_132/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_132/strided_slice_3/stack_1
(forward_lstm_132/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_132/strided_slice_3/stack_2
 forward_lstm_132/strided_slice_3StridedSlice<forward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_132/strided_slice_3/stack:output:01forward_lstm_132/strided_slice_3/stack_1:output:01forward_lstm_132/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2"
 forward_lstm_132/strided_slice_3
!forward_lstm_132/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_132/transpose_1/permђ
forward_lstm_132/transpose_1	Transpose<forward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_132/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_132/transpose_1
forward_lstm_132/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_132/runtime
&backward_lstm_132/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_132/RaggedToTensor/zeros
&backward_lstm_132/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2(
&backward_lstm_132/RaggedToTensor/Const
5backward_lstm_132/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_132/RaggedToTensor/Const:output:0inputs/backward_lstm_132/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_132/RaggedToTensor/RaggedTensorToTensorЦ
<backward_lstm_132/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_132/RaggedNestedRowLengths/strided_slice/stackЪ
>backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1Ъ
>backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2Ў
6backward_lstm_132/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_132/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_132/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask28
6backward_lstm_132/RaggedNestedRowLengths/strided_sliceЪ
>backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stackз
@backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2B
@backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1Ю
@backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2К
8backward_lstm_132/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_132/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2:
8backward_lstm_132/RaggedNestedRowLengths/strided_slice_1
,backward_lstm_132/RaggedNestedRowLengths/subSub?backward_lstm_132/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_132/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2.
,backward_lstm_132/RaggedNestedRowLengths/subЇ
backward_lstm_132/CastCast0backward_lstm_132/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_132/Cast 
backward_lstm_132/ShapeShape>backward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_132/Shape
%backward_lstm_132/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_132/strided_slice/stack
'backward_lstm_132/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_132/strided_slice/stack_1
'backward_lstm_132/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_132/strided_slice/stack_2Ю
backward_lstm_132/strided_sliceStridedSlice backward_lstm_132/Shape:output:0.backward_lstm_132/strided_slice/stack:output:00backward_lstm_132/strided_slice/stack_1:output:00backward_lstm_132/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_132/strided_slice
backward_lstm_132/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_132/zeros/mul/yД
backward_lstm_132/zeros/mulMul(backward_lstm_132/strided_slice:output:0&backward_lstm_132/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/zeros/mul
backward_lstm_132/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_132/zeros/Less/yЏ
backward_lstm_132/zeros/LessLessbackward_lstm_132/zeros/mul:z:0'backward_lstm_132/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/zeros/Less
 backward_lstm_132/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_132/zeros/packed/1Ы
backward_lstm_132/zeros/packedPack(backward_lstm_132/strided_slice:output:0)backward_lstm_132/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_132/zeros/packed
backward_lstm_132/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_132/zeros/ConstН
backward_lstm_132/zerosFill'backward_lstm_132/zeros/packed:output:0&backward_lstm_132/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_132/zeros
backward_lstm_132/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_132/zeros_1/mul/yК
backward_lstm_132/zeros_1/mulMul(backward_lstm_132/strided_slice:output:0(backward_lstm_132/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/zeros_1/mul
 backward_lstm_132/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 backward_lstm_132/zeros_1/Less/yЗ
backward_lstm_132/zeros_1/LessLess!backward_lstm_132/zeros_1/mul:z:0)backward_lstm_132/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_132/zeros_1/Less
"backward_lstm_132/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_132/zeros_1/packed/1б
 backward_lstm_132/zeros_1/packedPack(backward_lstm_132/strided_slice:output:0+backward_lstm_132/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_132/zeros_1/packed
backward_lstm_132/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_132/zeros_1/ConstХ
backward_lstm_132/zeros_1Fill)backward_lstm_132/zeros_1/packed:output:0(backward_lstm_132/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_132/zeros_1
 backward_lstm_132/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_132/transpose/permё
backward_lstm_132/transpose	Transpose>backward_lstm_132/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_132/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_132/transpose
backward_lstm_132/Shape_1Shapebackward_lstm_132/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_132/Shape_1
'backward_lstm_132/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_132/strided_slice_1/stack 
)backward_lstm_132/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_1/stack_1 
)backward_lstm_132/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_1/stack_2к
!backward_lstm_132/strided_slice_1StridedSlice"backward_lstm_132/Shape_1:output:00backward_lstm_132/strided_slice_1/stack:output:02backward_lstm_132/strided_slice_1/stack_1:output:02backward_lstm_132/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_132/strided_slice_1Љ
-backward_lstm_132/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2/
-backward_lstm_132/TensorArrayV2/element_shapeњ
backward_lstm_132/TensorArrayV2TensorListReserve6backward_lstm_132/TensorArrayV2/element_shape:output:0*backward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_132/TensorArrayV2
 backward_lstm_132/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_132/ReverseV2/axisв
backward_lstm_132/ReverseV2	ReverseV2backward_lstm_132/transpose:y:0)backward_lstm_132/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_132/ReverseV2у
Gbackward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gbackward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shapeХ
9backward_lstm_132/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_132/ReverseV2:output:0Pbackward_lstm_132/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_132/TensorArrayUnstack/TensorListFromTensor
'backward_lstm_132/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_132/strided_slice_2/stack 
)backward_lstm_132/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_2/stack_1 
)backward_lstm_132/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_2/stack_2ш
!backward_lstm_132/strided_slice_2StridedSlicebackward_lstm_132/transpose:y:00backward_lstm_132/strided_slice_2/stack:output:02backward_lstm_132/strided_slice_2/stack_1:output:02backward_lstm_132/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2#
!backward_lstm_132/strided_slice_2ю
5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpReadVariableOp>backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype027
5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOpј
&backward_lstm_132/lstm_cell_398/MatMulMatMul*backward_lstm_132/strided_slice_2:output:0=backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&backward_lstm_132/lstm_cell_398/MatMulє
7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpReadVariableOp@backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype029
7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOpє
(backward_lstm_132/lstm_cell_398/MatMul_1MatMul backward_lstm_132/zeros:output:0?backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_132/lstm_cell_398/MatMul_1ь
#backward_lstm_132/lstm_cell_398/addAddV20backward_lstm_132/lstm_cell_398/MatMul:product:02backward_lstm_132/lstm_cell_398/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_132/lstm_cell_398/addэ
6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpReadVariableOp?backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype028
6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOpљ
'backward_lstm_132/lstm_cell_398/BiasAddBiasAdd'backward_lstm_132/lstm_cell_398/add:z:0>backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'backward_lstm_132/lstm_cell_398/BiasAddЄ
/backward_lstm_132/lstm_cell_398/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/backward_lstm_132/lstm_cell_398/split/split_dimП
%backward_lstm_132/lstm_cell_398/splitSplit8backward_lstm_132/lstm_cell_398/split/split_dim:output:00backward_lstm_132/lstm_cell_398/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2'
%backward_lstm_132/lstm_cell_398/splitП
'backward_lstm_132/lstm_cell_398/SigmoidSigmoid.backward_lstm_132/lstm_cell_398/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_132/lstm_cell_398/SigmoidУ
)backward_lstm_132/lstm_cell_398/Sigmoid_1Sigmoid.backward_lstm_132/lstm_cell_398/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_132/lstm_cell_398/Sigmoid_1ж
#backward_lstm_132/lstm_cell_398/mulMul-backward_lstm_132/lstm_cell_398/Sigmoid_1:y:0"backward_lstm_132/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_132/lstm_cell_398/mulЖ
$backward_lstm_132/lstm_cell_398/ReluRelu.backward_lstm_132/lstm_cell_398/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$backward_lstm_132/lstm_cell_398/Reluш
%backward_lstm_132/lstm_cell_398/mul_1Mul+backward_lstm_132/lstm_cell_398/Sigmoid:y:02backward_lstm_132/lstm_cell_398/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_132/lstm_cell_398/mul_1н
%backward_lstm_132/lstm_cell_398/add_1AddV2'backward_lstm_132/lstm_cell_398/mul:z:0)backward_lstm_132/lstm_cell_398/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_132/lstm_cell_398/add_1У
)backward_lstm_132/lstm_cell_398/Sigmoid_2Sigmoid.backward_lstm_132/lstm_cell_398/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_132/lstm_cell_398/Sigmoid_2Е
&backward_lstm_132/lstm_cell_398/Relu_1Relu)backward_lstm_132/lstm_cell_398/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_132/lstm_cell_398/Relu_1ь
%backward_lstm_132/lstm_cell_398/mul_2Mul-backward_lstm_132/lstm_cell_398/Sigmoid_2:y:04backward_lstm_132/lstm_cell_398/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_132/lstm_cell_398/mul_2Г
/backward_lstm_132/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   21
/backward_lstm_132/TensorArrayV2_1/element_shape
!backward_lstm_132/TensorArrayV2_1TensorListReserve8backward_lstm_132/TensorArrayV2_1/element_shape:output:0*backward_lstm_132/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_132/TensorArrayV2_1r
backward_lstm_132/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_132/time
'backward_lstm_132/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_132/Max/reduction_indicesЄ
backward_lstm_132/MaxMaxbackward_lstm_132/Cast:y:00backward_lstm_132/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/Maxt
backward_lstm_132/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_132/sub/y
backward_lstm_132/subSubbackward_lstm_132/Max:output:0 backward_lstm_132/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_132/sub
backward_lstm_132/Sub_1Subbackward_lstm_132/sub:z:0backward_lstm_132/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_132/Sub_1І
backward_lstm_132/zeros_like	ZerosLike)backward_lstm_132/lstm_cell_398/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_132/zeros_likeЃ
*backward_lstm_132/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*backward_lstm_132/while/maximum_iterations
$backward_lstm_132/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_132/while/loop_counterЅ	
backward_lstm_132/whileWhile-backward_lstm_132/while/loop_counter:output:03backward_lstm_132/while/maximum_iterations:output:0backward_lstm_132/time:output:0*backward_lstm_132/TensorArrayV2_1:handle:0 backward_lstm_132/zeros_like:y:0 backward_lstm_132/zeros:output:0"backward_lstm_132/zeros_1:output:0*backward_lstm_132/strided_slice_1:output:0Ibackward_lstm_132/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_132/Sub_1:z:0>backward_lstm_132_lstm_cell_398_matmul_readvariableop_resource@backward_lstm_132_lstm_cell_398_matmul_1_readvariableop_resource?backward_lstm_132_lstm_cell_398_biasadd_readvariableop_resource*
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
%backward_lstm_132_while_body_16716377*1
cond)R'
%backward_lstm_132_while_cond_16716376*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_132/whileй
Bbackward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2D
Bbackward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shapeЙ
4backward_lstm_132/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_132/while:output:3Kbackward_lstm_132/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype026
4backward_lstm_132/TensorArrayV2Stack/TensorListStackЅ
'backward_lstm_132/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2)
'backward_lstm_132/strided_slice_3/stack 
)backward_lstm_132/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_132/strided_slice_3/stack_1 
)backward_lstm_132/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_132/strided_slice_3/stack_2
!backward_lstm_132/strided_slice_3StridedSlice=backward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_132/strided_slice_3/stack:output:02backward_lstm_132/strided_slice_3/stack_1:output:02backward_lstm_132/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2#
!backward_lstm_132/strided_slice_3
"backward_lstm_132/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_132/transpose_1/permі
backward_lstm_132/transpose_1	Transpose=backward_lstm_132/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_132/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_132/transpose_1
backward_lstm_132/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_132/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisФ
concatConcatV2)forward_lstm_132/strided_slice_3:output:0*backward_lstm_132/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp7^backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp6^backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp8^backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp^backward_lstm_132/while6^forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp5^forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp7^forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp^forward_lstm_132/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2p
6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp6backward_lstm_132/lstm_cell_398/BiasAdd/ReadVariableOp2n
5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp5backward_lstm_132/lstm_cell_398/MatMul/ReadVariableOp2r
7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp7backward_lstm_132/lstm_cell_398/MatMul_1/ReadVariableOp22
backward_lstm_132/whilebackward_lstm_132/while2n
5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp5forward_lstm_132/lstm_cell_397/BiasAdd/ReadVariableOp2l
4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp4forward_lstm_132/lstm_cell_397/MatMul/ReadVariableOp2p
6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp6forward_lstm_132/lstm_cell_397/MatMul_1/ReadVariableOp20
forward_lstm_132/whileforward_lstm_132/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
И
Ѓ
L__inference_sequential_132_layer_call_and_return_conditional_losses_16716601

inputs
inputs_1	-
bidirectional_132_16716582:	Ш-
bidirectional_132_16716584:	2Ш)
bidirectional_132_16716586:	Ш-
bidirectional_132_16716588:	Ш-
bidirectional_132_16716590:	2Ш)
bidirectional_132_16716592:	Ш$
dense_132_16716595:d 
dense_132_16716597:
identityЂ)bidirectional_132/StatefulPartitionedCallЂ!dense_132/StatefulPartitionedCallЪ
)bidirectional_132/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_132_16716582bidirectional_132_16716584bidirectional_132_16716586bidirectional_132_16716588bidirectional_132_16716590bidirectional_132_16716592*
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
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_167160342+
)bidirectional_132/StatefulPartitionedCallЫ
!dense_132/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_132/StatefulPartitionedCall:output:0dense_132_16716595dense_132_16716597*
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
G__inference_dense_132_layer_call_and_return_conditional_losses_167160592#
!dense_132/StatefulPartitionedCall
IdentityIdentity*dense_132/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp*^bidirectional_132/StatefulPartitionedCall"^dense_132/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2V
)bidirectional_132/StatefulPartitionedCall)bidirectional_132/StatefulPartitionedCall2F
!dense_132/StatefulPartitionedCall!dense_132/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ
Ѕ
$forward_lstm_132_while_cond_16715757>
:forward_lstm_132_while_forward_lstm_132_while_loop_counterD
@forward_lstm_132_while_forward_lstm_132_while_maximum_iterations&
"forward_lstm_132_while_placeholder(
$forward_lstm_132_while_placeholder_1(
$forward_lstm_132_while_placeholder_2(
$forward_lstm_132_while_placeholder_3(
$forward_lstm_132_while_placeholder_4@
<forward_lstm_132_while_less_forward_lstm_132_strided_slice_1X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16715757___redundant_placeholder0X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16715757___redundant_placeholder1X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16715757___redundant_placeholder2X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16715757___redundant_placeholder3X
Tforward_lstm_132_while_forward_lstm_132_while_cond_16715757___redundant_placeholder4#
forward_lstm_132_while_identity
Х
forward_lstm_132/while/LessLess"forward_lstm_132_while_placeholder<forward_lstm_132_while_less_forward_lstm_132_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_132/while/Less
forward_lstm_132/while/IdentityIdentityforward_lstm_132/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_132/while/Identity"K
forward_lstm_132_while_identity(forward_lstm_132/while/Identity:output:0*(
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
while_cond_16715463
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_16715463___redundant_placeholder06
2while_while_cond_16715463___redundant_placeholder16
2while_while_cond_16715463___redundant_placeholder26
2while_while_cond_16715463___redundant_placeholder3
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
	dense_1320
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
": d2dense_132/kernel
:2dense_132/bias
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
J:H	Ш27bidirectional_132/forward_lstm_132/lstm_cell_397/kernel
T:R	2Ш2Abidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel
D:BШ25bidirectional_132/forward_lstm_132/lstm_cell_397/bias
K:I	Ш28bidirectional_132/backward_lstm_132/lstm_cell_398/kernel
U:S	2Ш2Bbidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel
E:CШ26bidirectional_132/backward_lstm_132/lstm_cell_398/bias
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
':%d2Adam/dense_132/kernel/m
!:2Adam/dense_132/bias/m
O:M	Ш2>Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/m
Y:W	2Ш2HAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/m
I:GШ2<Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/m
P:N	Ш2?Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/m
Z:X	2Ш2IAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/m
J:HШ2=Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/m
':%d2Adam/dense_132/kernel/v
!:2Adam/dense_132/bias/v
O:M	Ш2>Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/v
Y:W	2Ш2HAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/v
I:GШ2<Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/v
P:N	Ш2?Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/v
Z:X	2Ш2IAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/v
J:HШ2=Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/v
*:(d2Adam/dense_132/kernel/vhat
$:"2Adam/dense_132/bias/vhat
R:P	Ш2AAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/kernel/vhat
\:Z	2Ш2KAdam/bidirectional_132/forward_lstm_132/lstm_cell_397/recurrent_kernel/vhat
L:JШ2?Adam/bidirectional_132/forward_lstm_132/lstm_cell_397/bias/vhat
S:Q	Ш2BAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/kernel/vhat
]:[	2Ш2LAdam/bidirectional_132/backward_lstm_132/lstm_cell_398/recurrent_kernel/vhat
M:KШ2@Adam/bidirectional_132/backward_lstm_132/lstm_cell_398/bias/vhat
Ќ2Љ
1__inference_sequential_132_layer_call_fn_16716085
1__inference_sequential_132_layer_call_fn_16716578Р
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
#__inference__wrapped_model_16713598args_0args_0_1"
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
L__inference_sequential_132_layer_call_and_return_conditional_losses_16716601
L__inference_sequential_132_layer_call_and_return_conditional_losses_16716624Р
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
4__inference_bidirectional_132_layer_call_fn_16716671
4__inference_bidirectional_132_layer_call_fn_16716688
4__inference_bidirectional_132_layer_call_fn_16716706
4__inference_bidirectional_132_layer_call_fn_16716724ц
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
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_16717026
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_16717328
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_16717686
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_16718044ц
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
,__inference_dense_132_layer_call_fn_16718053Ђ
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
G__inference_dense_132_layer_call_and_return_conditional_losses_16718064Ђ
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
&__inference_signature_wrapper_16716654args_0args_0_1"
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
3__inference_forward_lstm_132_layer_call_fn_16718075
3__inference_forward_lstm_132_layer_call_fn_16718086
3__inference_forward_lstm_132_layer_call_fn_16718097
3__inference_forward_lstm_132_layer_call_fn_16718108е
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
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_16718259
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_16718410
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_16718561
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_16718712е
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
4__inference_backward_lstm_132_layer_call_fn_16718723
4__inference_backward_lstm_132_layer_call_fn_16718734
4__inference_backward_lstm_132_layer_call_fn_16718745
4__inference_backward_lstm_132_layer_call_fn_16718756е
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
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_16718909
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_16719062
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_16719215
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_16719368е
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
0__inference_lstm_cell_397_layer_call_fn_16719385
0__inference_lstm_cell_397_layer_call_fn_16719402О
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
K__inference_lstm_cell_397_layer_call_and_return_conditional_losses_16719434
K__inference_lstm_cell_397_layer_call_and_return_conditional_losses_16719466О
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
0__inference_lstm_cell_398_layer_call_fn_16719483
0__inference_lstm_cell_398_layer_call_fn_16719500О
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
K__inference_lstm_cell_398_layer_call_and_return_conditional_losses_16719532
K__inference_lstm_cell_398_layer_call_and_return_conditional_losses_16719564О
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
#__inference__wrapped_model_16713598\ЂY
RЂO
MJ4Ђ1
!њџџџџџџџџџџџџџџџџџџ

`
	RaggedTensorSpec
Њ "5Њ2
0
	dense_132# 
	dense_132џџџџџџџџџа
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_16718909}OЂL
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
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_16719062}OЂL
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
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_16719215QЂN
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
O__inference_backward_lstm_132_layer_call_and_return_conditional_losses_16719368QЂN
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
4__inference_backward_lstm_132_layer_call_fn_16718723pOЂL
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
4__inference_backward_lstm_132_layer_call_fn_16718734pOЂL
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
4__inference_backward_lstm_132_layer_call_fn_16718745rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2Њ
4__inference_backward_lstm_132_layer_call_fn_16718756rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2с
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_16717026\ЂY
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
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_16717328\ЂY
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
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_16717686lЂi
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
O__inference_bidirectional_132_layer_call_and_return_conditional_losses_16718044lЂi
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
4__inference_bidirectional_132_layer_call_fn_16716671\ЂY
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
4__inference_bidirectional_132_layer_call_fn_16716688\ЂY
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
4__inference_bidirectional_132_layer_call_fn_16716706lЂi
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
4__inference_bidirectional_132_layer_call_fn_16716724lЂi
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
G__inference_dense_132_layer_call_and_return_conditional_losses_16718064\/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџ
 
,__inference_dense_132_layer_call_fn_16718053O/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџЯ
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_16718259}OЂL
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
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_16718410}OЂL
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
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_16718561QЂN
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
N__inference_forward_lstm_132_layer_call_and_return_conditional_losses_16718712QЂN
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
3__inference_forward_lstm_132_layer_call_fn_16718075pOЂL
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
3__inference_forward_lstm_132_layer_call_fn_16718086pOЂL
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
3__inference_forward_lstm_132_layer_call_fn_16718097rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2Љ
3__inference_forward_lstm_132_layer_call_fn_16718108rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2Э
K__inference_lstm_cell_397_layer_call_and_return_conditional_losses_16719434§Ђ}
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
K__inference_lstm_cell_397_layer_call_and_return_conditional_losses_16719466§Ђ}
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
0__inference_lstm_cell_397_layer_call_fn_16719385эЂ}
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
0__inference_lstm_cell_397_layer_call_fn_16719402эЂ}
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
K__inference_lstm_cell_398_layer_call_and_return_conditional_losses_16719532§Ђ}
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
K__inference_lstm_cell_398_layer_call_and_return_conditional_losses_16719564§Ђ}
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
0__inference_lstm_cell_398_layer_call_fn_16719483эЂ}
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
0__inference_lstm_cell_398_layer_call_fn_16719500эЂ}
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
L__inference_sequential_132_layer_call_and_return_conditional_losses_16716601dЂa
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
L__inference_sequential_132_layer_call_and_return_conditional_losses_16716624dЂa
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
1__inference_sequential_132_layer_call_fn_16716085dЂa
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
1__inference_sequential_132_layer_call_fn_16716578dЂa
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
&__inference_signature_wrapper_16716654ЈeЂb
Ђ 
[ЊX
*
args_0 
args_0џџџџџџџџџ
*
args_0_1
args_0_1џџџџџџџџџ	"5Њ2
0
	dense_132# 
	dense_132џџџџџџџџџ