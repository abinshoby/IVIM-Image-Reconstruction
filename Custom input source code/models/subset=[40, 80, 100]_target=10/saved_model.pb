ЈЏ<
Іш
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
М
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
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
Ч
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
dtypetypeИ
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
list(type)(0И
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
list(type)(0И
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
Њ
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
ц
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
Ђ
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handleКйelement_dtype"
element_dtypetype"

shape_typetype:
2	
Ъ
TensorListReserve
element_shape"
shape_type
num_elements#
handleКйelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint€€€€€€€€€
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
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
И
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.6.02v2.6.0-rc2-32-g919f693420e8эы:
|
dense_379/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*!
shared_namedense_379/kernel
u
$dense_379/kernel/Read/ReadVariableOpReadVariableOpdense_379/kernel*
_output_shapes

:d*
dtype0
t
dense_379/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_379/bias
m
"dense_379/bias/Read/ReadVariableOpReadVariableOpdense_379/bias*
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
Ќ
8bidirectional_379/forward_lstm_379/lstm_cell_1138/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*I
shared_name:8bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel
∆
Lbidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/Read/ReadVariableOpReadVariableOp8bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel*
_output_shapes
:	»*
dtype0
б
Bbidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*S
shared_nameDBbidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel
Џ
Vbidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/Read/ReadVariableOpReadVariableOpBbidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel*
_output_shapes
:	2»*
dtype0
≈
6bidirectional_379/forward_lstm_379/lstm_cell_1138/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*G
shared_name86bidirectional_379/forward_lstm_379/lstm_cell_1138/bias
Њ
Jbidirectional_379/forward_lstm_379/lstm_cell_1138/bias/Read/ReadVariableOpReadVariableOp6bidirectional_379/forward_lstm_379/lstm_cell_1138/bias*
_output_shapes	
:»*
dtype0
ѕ
9bidirectional_379/backward_lstm_379/lstm_cell_1139/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*J
shared_name;9bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel
»
Mbidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/Read/ReadVariableOpReadVariableOp9bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel*
_output_shapes
:	»*
dtype0
г
Cbidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*T
shared_nameECbidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel
№
Wbidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/Read/ReadVariableOpReadVariableOpCbidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel*
_output_shapes
:	2»*
dtype0
«
7bidirectional_379/backward_lstm_379/lstm_cell_1139/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*H
shared_name97bidirectional_379/backward_lstm_379/lstm_cell_1139/bias
ј
Kbidirectional_379/backward_lstm_379/lstm_cell_1139/bias/Read/ReadVariableOpReadVariableOp7bidirectional_379/backward_lstm_379/lstm_cell_1139/bias*
_output_shapes	
:»*
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
К
Adam/dense_379/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_379/kernel/m
Г
+Adam/dense_379/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_379/kernel/m*
_output_shapes

:d*
dtype0
В
Adam/dense_379/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_379/bias/m
{
)Adam/dense_379/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_379/bias/m*
_output_shapes
:*
dtype0
џ
?Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*P
shared_nameA?Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/m
‘
SAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/m/Read/ReadVariableOpReadVariableOp?Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/m*
_output_shapes
:	»*
dtype0
п
IAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*Z
shared_nameKIAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/m
и
]Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpIAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/m*
_output_shapes
:	2»*
dtype0
”
=Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*N
shared_name?=Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/m
ћ
QAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/m/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/m*
_output_shapes	
:»*
dtype0
Ё
@Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*Q
shared_nameB@Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/m
÷
TAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/m/Read/ReadVariableOpReadVariableOp@Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/m*
_output_shapes
:	»*
dtype0
с
JAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*[
shared_nameLJAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/m
к
^Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpJAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/m*
_output_shapes
:	2»*
dtype0
’
>Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*O
shared_name@>Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/m
ќ
RAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/m/Read/ReadVariableOpReadVariableOp>Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/m*
_output_shapes	
:»*
dtype0
К
Adam/dense_379/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_379/kernel/v
Г
+Adam/dense_379/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_379/kernel/v*
_output_shapes

:d*
dtype0
В
Adam/dense_379/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_379/bias/v
{
)Adam/dense_379/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_379/bias/v*
_output_shapes
:*
dtype0
џ
?Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*P
shared_nameA?Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/v
‘
SAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/v/Read/ReadVariableOpReadVariableOp?Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/v*
_output_shapes
:	»*
dtype0
п
IAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*Z
shared_nameKIAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/v
и
]Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpIAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/v*
_output_shapes
:	2»*
dtype0
”
=Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*N
shared_name?=Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/v
ћ
QAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/v/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/v*
_output_shapes	
:»*
dtype0
Ё
@Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*Q
shared_nameB@Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/v
÷
TAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/v/Read/ReadVariableOpReadVariableOp@Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/v*
_output_shapes
:	»*
dtype0
с
JAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*[
shared_nameLJAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/v
к
^Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpJAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/v*
_output_shapes
:	2»*
dtype0
’
>Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*O
shared_name@>Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/v
ќ
RAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/v/Read/ReadVariableOpReadVariableOp>Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/v*
_output_shapes	
:»*
dtype0
Р
Adam/dense_379/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*+
shared_nameAdam/dense_379/kernel/vhat
Й
.Adam/dense_379/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_379/kernel/vhat*
_output_shapes

:d*
dtype0
И
Adam/dense_379/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/dense_379/bias/vhat
Б
,Adam/dense_379/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_379/bias/vhat*
_output_shapes
:*
dtype0
б
BAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*S
shared_nameDBAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/vhat
Џ
VAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/vhat/Read/ReadVariableOpReadVariableOpBAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/vhat*
_output_shapes
:	»*
dtype0
х
LAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*]
shared_nameNLAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/vhat
о
`Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpLAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/vhat*
_output_shapes
:	2»*
dtype0
ў
@Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*Q
shared_nameB@Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/vhat
“
TAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/vhat/Read/ReadVariableOpReadVariableOp@Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/vhat*
_output_shapes	
:»*
dtype0
г
CAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*T
shared_nameECAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/vhat
№
WAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/vhat/Read/ReadVariableOpReadVariableOpCAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/vhat*
_output_shapes
:	»*
dtype0
ч
MAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*^
shared_nameOMAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/vhat
р
aAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpMAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/vhat*
_output_shapes
:	2»*
dtype0
џ
AAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*R
shared_nameCAAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/vhat
‘
UAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/vhat/Read/ReadVariableOpReadVariableOpAAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/vhat*
_output_shapes	
:»*
dtype0

NoOpNoOp
≤A
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*н@
valueг@Bа@ Bў@
њ
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
∞
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
≠
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
≠
1layer_metrics
2non_trainable_variables
regularization_losses
	variables
3metrics
trainable_variables
4layer_regularization_losses

5layers
\Z
VARIABLE_VALUEdense_379/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_379/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
≠
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
tr
VARIABLE_VALUE8bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEBbidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6bidirectional_379/forward_lstm_379/lstm_cell_1138/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE9bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUECbidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE7bidirectional_379/backward_lstm_379/lstm_cell_1139/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 

;0
 

0
1
О
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
є
Alayer_metrics
Bnon_trainable_variables

Cstates
'regularization_losses
(	variables
Dmetrics
)trainable_variables
Elayer_regularization_losses

Flayers
О
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
є
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
≠
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
≠
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
VARIABLE_VALUEAdam/dense_379/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_379/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE?Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ҐЯ
VARIABLE_VALUEIAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE@Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
£†
VARIABLE_VALUEJAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE>Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_379/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_379/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE?Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ҐЯ
VARIABLE_VALUEIAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE@Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
£†
VARIABLE_VALUEJAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE>Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEAdam/dense_379/kernel/vhatUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/dense_379/bias/vhatSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUEBAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/vhatEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
®•
VARIABLE_VALUELAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/vhatEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ЬЩ
VARIABLE_VALUE@Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/vhatEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUECAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/vhatEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
©¶
VARIABLE_VALUEMAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/vhatEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ЭЪ
VARIABLE_VALUEAAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/vhatEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
y
serving_default_args_0Placeholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
s
serving_default_args_0_1Placeholder*#
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€
п
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_0_18bidirectional_379/forward_lstm_379/lstm_cell_1138/kernelBbidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel6bidirectional_379/forward_lstm_379/lstm_cell_1138/bias9bidirectional_379/backward_lstm_379/lstm_cell_1139/kernelCbidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel7bidirectional_379/backward_lstm_379/lstm_cell_1139/biasdense_379/kerneldense_379/bias*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8В */
f*R(
&__inference_signature_wrapper_42660894
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
І
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_379/kernel/Read/ReadVariableOp"dense_379/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpLbidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/Read/ReadVariableOpVbidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/Read/ReadVariableOpJbidirectional_379/forward_lstm_379/lstm_cell_1138/bias/Read/ReadVariableOpMbidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/Read/ReadVariableOpWbidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/Read/ReadVariableOpKbidirectional_379/backward_lstm_379/lstm_cell_1139/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_379/kernel/m/Read/ReadVariableOp)Adam/dense_379/bias/m/Read/ReadVariableOpSAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/m/Read/ReadVariableOp]Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/m/Read/ReadVariableOpQAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/m/Read/ReadVariableOpTAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/m/Read/ReadVariableOp^Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/m/Read/ReadVariableOpRAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/m/Read/ReadVariableOp+Adam/dense_379/kernel/v/Read/ReadVariableOp)Adam/dense_379/bias/v/Read/ReadVariableOpSAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/v/Read/ReadVariableOp]Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/v/Read/ReadVariableOpQAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/v/Read/ReadVariableOpTAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/v/Read/ReadVariableOp^Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/v/Read/ReadVariableOpRAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/v/Read/ReadVariableOp.Adam/dense_379/kernel/vhat/Read/ReadVariableOp,Adam/dense_379/bias/vhat/Read/ReadVariableOpVAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/vhat/Read/ReadVariableOp`Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/vhat/Read/ReadVariableOpTAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/vhat/Read/ReadVariableOpWAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/vhat/Read/ReadVariableOpaAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/vhat/Read/ReadVariableOpUAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/vhat/Read/ReadVariableOpConst*4
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
GPU 2J 8В **
f%R#
!__inference__traced_save_42663945
Ц
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_379/kerneldense_379/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate8bidirectional_379/forward_lstm_379/lstm_cell_1138/kernelBbidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel6bidirectional_379/forward_lstm_379/lstm_cell_1138/bias9bidirectional_379/backward_lstm_379/lstm_cell_1139/kernelCbidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel7bidirectional_379/backward_lstm_379/lstm_cell_1139/biastotalcountAdam/dense_379/kernel/mAdam/dense_379/bias/m?Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/mIAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/m=Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/m@Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/mJAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/m>Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/mAdam/dense_379/kernel/vAdam/dense_379/bias/v?Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/vIAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/v=Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/v@Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/vJAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/v>Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/vAdam/dense_379/kernel/vhatAdam/dense_379/bias/vhatBAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/vhatLAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/vhat@Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/vhatCAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/vhatMAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/vhatAAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/vhat*3
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
GPU 2J 8В *-
f(R&
$__inference__traced_restore_42664072иЪ9
ѕ@
д
while_body_42659531
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1139_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1139_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1139_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1139_matmul_readvariableop_resource:	»H
5while_lstm_cell_1139_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1139_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1139/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1139/MatMul/ReadVariableOpҐ,while/lstm_cell_1139/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€29
7while/TensorArrayV2Read/TensorListGetItem/element_shape№
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemѕ
*while/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1139_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1139/MatMul/ReadVariableOpЁ
while/lstm_cell_1139/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/MatMul’
,while/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1139_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1139/MatMul_1/ReadVariableOp∆
while/lstm_cell_1139/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/MatMul_1ј
while/lstm_cell_1139/addAddV2%while/lstm_cell_1139/MatMul:product:0'while/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/addќ
+while/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1139_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1139/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1139/BiasAddBiasAddwhile/lstm_cell_1139/add:z:03while/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/BiasAddО
$while/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1139/split/split_dimУ
while/lstm_cell_1139/splitSplit-while/lstm_cell_1139/split/split_dim:output:0%while/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1139/splitЮ
while/lstm_cell_1139/SigmoidSigmoid#while/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/SigmoidҐ
while/lstm_cell_1139/Sigmoid_1Sigmoid#while/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1139/Sigmoid_1¶
while/lstm_cell_1139/mulMul"while/lstm_cell_1139/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/mulХ
while/lstm_cell_1139/ReluRelu#while/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/ReluЉ
while/lstm_cell_1139/mul_1Mul while/lstm_cell_1139/Sigmoid:y:0'while/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/mul_1±
while/lstm_cell_1139/add_1AddV2while/lstm_cell_1139/mul:z:0while/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/add_1Ґ
while/lstm_cell_1139/Sigmoid_2Sigmoid#while/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1139/Sigmoid_2Ф
while/lstm_cell_1139/Relu_1Reluwhile/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/Relu_1ј
while/lstm_cell_1139/mul_2Mul"while/lstm_cell_1139/Sigmoid_2:y:0)while/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1139/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3П
while/Identity_4Identitywhile/lstm_cell_1139/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1139/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1139/BiasAdd/ReadVariableOp+^while/lstm_cell_1139/MatMul/ReadVariableOp-^while/lstm_cell_1139/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"n
4while_lstm_cell_1139_biasadd_readvariableop_resource6while_lstm_cell_1139_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1139_matmul_1_readvariableop_resource7while_lstm_cell_1139_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1139_matmul_readvariableop_resource5while_lstm_cell_1139_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1139/BiasAdd/ReadVariableOp+while/lstm_cell_1139/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1139/MatMul/ReadVariableOp*while/lstm_cell_1139/MatMul/ReadVariableOp2\
,while/lstm_cell_1139/MatMul_1/ReadVariableOp,while/lstm_cell_1139/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
л	
Щ
4__inference_bidirectional_379_layer_call_fn_42660928
inputs_0
unknown:	»
	unknown_0:	2»
	unknown_1:	»
	unknown_2:	»
	unknown_3:	2»
	unknown_4:	»
identityИҐStatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_426598362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
ч
И
L__inference_lstm_cell_1139_layer_call_and_return_conditional_losses_42658691

inputs

states
states_11
matmul_readvariableop_resource:	»3
 matmul_1_readvariableop_resource:	2».
biasadd_readvariableop_resource:	»
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	»*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_2Щ
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
?:€€€€€€€€€:€€€€€€€€€2:€€€€€€€€€2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_namestates
я
Ќ
while_cond_42663370
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_42663370___redundant_placeholder06
2while_while_cond_42663370___redundant_placeholder16
2while_while_cond_42663370___redundant_placeholder26
2while_while_cond_42663370___redundant_placeholder3
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
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:
 

Ќ
&__inference_signature_wrapper_42660894

args_0
args_0_1	
unknown:	»
	unknown_0:	2»
	unknown_1:	»
	unknown_2:	»
	unknown_3:	2»
	unknown_4:	»
	unknown_5:d
	unknown_6:
identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallargs_0args_0_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__wrapped_model_426578382
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€:€€€€€€€€€: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameargs_0:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
args_0_1
жF
Ю
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_42657996

inputs*
lstm_cell_1138_42657914:	»*
lstm_cell_1138_42657916:	2»&
lstm_cell_1138_42657918:	»
identityИҐ&lstm_cell_1138/StatefulPartitionedCallҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€22
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
B :и2
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
zeros_1/packed/1Й
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
:€€€€€€€€€22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2±
&lstm_cell_1138/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1138_42657914lstm_cell_1138_42657916lstm_cell_1138_42657918*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_lstm_cell_1138_layer_call_and_return_conditional_losses_426579132(
&lstm_cell_1138/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter–
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1138_42657914lstm_cell_1138_42657916lstm_cell_1138_42657918*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42657927*
condR
while_cond_42657926*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
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
:€€€€€€€€€22

Identity
NoOpNoOp'^lstm_cell_1138/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2P
&lstm_cell_1138/StatefulPartitionedCall&lstm_cell_1138/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
¶Z
§
%backward_lstm_379_while_body_42661482@
<backward_lstm_379_while_backward_lstm_379_while_loop_counterF
Bbackward_lstm_379_while_backward_lstm_379_while_maximum_iterations'
#backward_lstm_379_while_placeholder)
%backward_lstm_379_while_placeholder_1)
%backward_lstm_379_while_placeholder_2)
%backward_lstm_379_while_placeholder_3?
;backward_lstm_379_while_backward_lstm_379_strided_slice_1_0{
wbackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0Z
Gbackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0:	»\
Ibackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0:	2»W
Hbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0:	»$
 backward_lstm_379_while_identity&
"backward_lstm_379_while_identity_1&
"backward_lstm_379_while_identity_2&
"backward_lstm_379_while_identity_3&
"backward_lstm_379_while_identity_4&
"backward_lstm_379_while_identity_5=
9backward_lstm_379_while_backward_lstm_379_strided_slice_1y
ubackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensorX
Ebackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource:	»Z
Gbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource:	2»U
Fbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource:	»ИҐ=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpҐ<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOpҐ>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpз
Ibackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2K
Ibackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shape»
;backward_lstm_379/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_379_while_placeholderRbackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02=
;backward_lstm_379/while/TensorArrayV2Read/TensorListGetItemЕ
<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOpGbackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02>
<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp•
-backward_lstm_379/while/lstm_cell_1139/MatMulMatMulBbackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem:item:0Dbackward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-backward_lstm_379/while/lstm_cell_1139/MatMulЛ
>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOpIbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02@
>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpО
/backward_lstm_379/while/lstm_cell_1139/MatMul_1MatMul%backward_lstm_379_while_placeholder_2Fbackward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»21
/backward_lstm_379/while/lstm_cell_1139/MatMul_1И
*backward_lstm_379/while/lstm_cell_1139/addAddV27backward_lstm_379/while/lstm_cell_1139/MatMul:product:09backward_lstm_379/while/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2,
*backward_lstm_379/while/lstm_cell_1139/addД
=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOpHbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02?
=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpХ
.backward_lstm_379/while/lstm_cell_1139/BiasAddBiasAdd.backward_lstm_379/while/lstm_cell_1139/add:z:0Ebackward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.backward_lstm_379/while/lstm_cell_1139/BiasAdd≤
6backward_lstm_379/while/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6backward_lstm_379/while/lstm_cell_1139/split/split_dimџ
,backward_lstm_379/while/lstm_cell_1139/splitSplit?backward_lstm_379/while/lstm_cell_1139/split/split_dim:output:07backward_lstm_379/while/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2.
,backward_lstm_379/while/lstm_cell_1139/split‘
.backward_lstm_379/while/lstm_cell_1139/SigmoidSigmoid5backward_lstm_379/while/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€220
.backward_lstm_379/while/lstm_cell_1139/SigmoidЎ
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_1Sigmoid5backward_lstm_379/while/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_1о
*backward_lstm_379/while/lstm_cell_1139/mulMul4backward_lstm_379/while/lstm_cell_1139/Sigmoid_1:y:0%backward_lstm_379_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_379/while/lstm_cell_1139/mulЋ
+backward_lstm_379/while/lstm_cell_1139/ReluRelu5backward_lstm_379/while/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22-
+backward_lstm_379/while/lstm_cell_1139/ReluД
,backward_lstm_379/while/lstm_cell_1139/mul_1Mul2backward_lstm_379/while/lstm_cell_1139/Sigmoid:y:09backward_lstm_379/while/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_379/while/lstm_cell_1139/mul_1щ
,backward_lstm_379/while/lstm_cell_1139/add_1AddV2.backward_lstm_379/while/lstm_cell_1139/mul:z:00backward_lstm_379/while/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_379/while/lstm_cell_1139/add_1Ў
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_2Sigmoid5backward_lstm_379/while/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_2 
-backward_lstm_379/while/lstm_cell_1139/Relu_1Relu0backward_lstm_379/while/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-backward_lstm_379/while/lstm_cell_1139/Relu_1И
,backward_lstm_379/while/lstm_cell_1139/mul_2Mul4backward_lstm_379/while/lstm_cell_1139/Sigmoid_2:y:0;backward_lstm_379/while/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_379/while/lstm_cell_1139/mul_2Љ
<backward_lstm_379/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_379_while_placeholder_1#backward_lstm_379_while_placeholder0backward_lstm_379/while/lstm_cell_1139/mul_2:z:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_379/while/TensorArrayV2Write/TensorListSetItemА
backward_lstm_379/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_379/while/add/y±
backward_lstm_379/while/addAddV2#backward_lstm_379_while_placeholder&backward_lstm_379/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/while/addД
backward_lstm_379/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_379/while/add_1/y–
backward_lstm_379/while/add_1AddV2<backward_lstm_379_while_backward_lstm_379_while_loop_counter(backward_lstm_379/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/while/add_1≥
 backward_lstm_379/while/IdentityIdentity!backward_lstm_379/while/add_1:z:0^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_379/while/IdentityЎ
"backward_lstm_379/while/Identity_1IdentityBbackward_lstm_379_while_backward_lstm_379_while_maximum_iterations^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_379/while/Identity_1µ
"backward_lstm_379/while/Identity_2Identitybackward_lstm_379/while/add:z:0^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_379/while/Identity_2в
"backward_lstm_379/while/Identity_3IdentityLbackward_lstm_379/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_379/while/Identity_3„
"backward_lstm_379/while/Identity_4Identity0backward_lstm_379/while/lstm_cell_1139/mul_2:z:0^backward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_379/while/Identity_4„
"backward_lstm_379/while/Identity_5Identity0backward_lstm_379/while/lstm_cell_1139/add_1:z:0^backward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_379/while/Identity_5Њ
backward_lstm_379/while/NoOpNoOp>^backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp=^backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp?^backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_379/while/NoOp"x
9backward_lstm_379_while_backward_lstm_379_strided_slice_1;backward_lstm_379_while_backward_lstm_379_strided_slice_1_0"M
 backward_lstm_379_while_identity)backward_lstm_379/while/Identity:output:0"Q
"backward_lstm_379_while_identity_1+backward_lstm_379/while/Identity_1:output:0"Q
"backward_lstm_379_while_identity_2+backward_lstm_379/while/Identity_2:output:0"Q
"backward_lstm_379_while_identity_3+backward_lstm_379/while/Identity_3:output:0"Q
"backward_lstm_379_while_identity_4+backward_lstm_379/while/Identity_4:output:0"Q
"backward_lstm_379_while_identity_5+backward_lstm_379/while/Identity_5:output:0"Т
Fbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resourceHbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0"Ф
Gbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resourceIbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0"Р
Ebackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resourceGbackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0"р
ubackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2~
=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp2|
<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp2А
>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
жF
Ю
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_42658206

inputs*
lstm_cell_1138_42658124:	»*
lstm_cell_1138_42658126:	2»&
lstm_cell_1138_42658128:	»
identityИҐ&lstm_cell_1138/StatefulPartitionedCallҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€22
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
B :и2
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
zeros_1/packed/1Й
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
:€€€€€€€€€22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2±
&lstm_cell_1138/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1138_42658124lstm_cell_1138_42658126lstm_cell_1138_42658128*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_lstm_cell_1138_layer_call_and_return_conditional_losses_426580592(
&lstm_cell_1138/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter–
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1138_42658124lstm_cell_1138_42658126lstm_cell_1138_42658128*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42658137*
condR
while_cond_42658136*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
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
:€€€€€€€€€22

Identity
NoOpNoOp'^lstm_cell_1138/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2P
&lstm_cell_1138/StatefulPartitionedCall&lstm_cell_1138/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
э
µ
%backward_lstm_379_while_cond_42661179@
<backward_lstm_379_while_backward_lstm_379_while_loop_counterF
Bbackward_lstm_379_while_backward_lstm_379_while_maximum_iterations'
#backward_lstm_379_while_placeholder)
%backward_lstm_379_while_placeholder_1)
%backward_lstm_379_while_placeholder_2)
%backward_lstm_379_while_placeholder_3B
>backward_lstm_379_while_less_backward_lstm_379_strided_slice_1Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42661179___redundant_placeholder0Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42661179___redundant_placeholder1Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42661179___redundant_placeholder2Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42661179___redundant_placeholder3$
 backward_lstm_379_while_identity
 
backward_lstm_379/while/LessLess#backward_lstm_379_while_placeholder>backward_lstm_379_while_less_backward_lstm_379_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_379/while/LessУ
 backward_lstm_379/while/IdentityIdentity backward_lstm_379/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_379/while/Identity"M
 backward_lstm_379_while_identity)backward_lstm_379/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:
я
°
$forward_lstm_379_while_cond_42661332>
:forward_lstm_379_while_forward_lstm_379_while_loop_counterD
@forward_lstm_379_while_forward_lstm_379_while_maximum_iterations&
"forward_lstm_379_while_placeholder(
$forward_lstm_379_while_placeholder_1(
$forward_lstm_379_while_placeholder_2(
$forward_lstm_379_while_placeholder_3@
<forward_lstm_379_while_less_forward_lstm_379_strided_slice_1X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42661332___redundant_placeholder0X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42661332___redundant_placeholder1X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42661332___redundant_placeholder2X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42661332___redundant_placeholder3#
forward_lstm_379_while_identity
≈
forward_lstm_379/while/LessLess"forward_lstm_379_while_placeholder<forward_lstm_379_while_less_forward_lstm_379_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_379/while/LessР
forward_lstm_379/while/IdentityIdentityforward_lstm_379/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_379/while/Identity"K
forward_lstm_379_while_identity(forward_lstm_379/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:
зe
Ћ
$forward_lstm_379_while_body_42661650>
:forward_lstm_379_while_forward_lstm_379_while_loop_counterD
@forward_lstm_379_while_forward_lstm_379_while_maximum_iterations&
"forward_lstm_379_while_placeholder(
$forward_lstm_379_while_placeholder_1(
$forward_lstm_379_while_placeholder_2(
$forward_lstm_379_while_placeholder_3(
$forward_lstm_379_while_placeholder_4=
9forward_lstm_379_while_forward_lstm_379_strided_slice_1_0y
uforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_379_while_greater_forward_lstm_379_cast_0Y
Fforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0:	»[
Hforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0:	2»V
Gforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0:	»#
forward_lstm_379_while_identity%
!forward_lstm_379_while_identity_1%
!forward_lstm_379_while_identity_2%
!forward_lstm_379_while_identity_3%
!forward_lstm_379_while_identity_4%
!forward_lstm_379_while_identity_5%
!forward_lstm_379_while_identity_6;
7forward_lstm_379_while_forward_lstm_379_strided_slice_1w
sforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_379_while_greater_forward_lstm_379_castW
Dforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource:	»Y
Fforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource:	2»T
Eforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource:	»ИҐ<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpҐ;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOpҐ=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpе
Hforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2J
Hforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeє
:forward_lstm_379/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_379_while_placeholderQforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02<
:forward_lstm_379/while/TensorArrayV2Read/TensorListGetItem’
forward_lstm_379/while/GreaterGreater6forward_lstm_379_while_greater_forward_lstm_379_cast_0"forward_lstm_379_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2 
forward_lstm_379/while/GreaterВ
;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOpFforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02=
;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp°
,forward_lstm_379/while/lstm_cell_1138/MatMulMatMulAforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem:item:0Cforward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2.
,forward_lstm_379/while/lstm_cell_1138/MatMulИ
=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOpHforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02?
=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpК
.forward_lstm_379/while/lstm_cell_1138/MatMul_1MatMul$forward_lstm_379_while_placeholder_3Eforward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.forward_lstm_379/while/lstm_cell_1138/MatMul_1Д
)forward_lstm_379/while/lstm_cell_1138/addAddV26forward_lstm_379/while/lstm_cell_1138/MatMul:product:08forward_lstm_379/while/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)forward_lstm_379/while/lstm_cell_1138/addБ
<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOpGforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02>
<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpС
-forward_lstm_379/while/lstm_cell_1138/BiasAddBiasAdd-forward_lstm_379/while/lstm_cell_1138/add:z:0Dforward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-forward_lstm_379/while/lstm_cell_1138/BiasAdd∞
5forward_lstm_379/while/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5forward_lstm_379/while/lstm_cell_1138/split/split_dim„
+forward_lstm_379/while/lstm_cell_1138/splitSplit>forward_lstm_379/while/lstm_cell_1138/split/split_dim:output:06forward_lstm_379/while/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2-
+forward_lstm_379/while/lstm_cell_1138/split—
-forward_lstm_379/while/lstm_cell_1138/SigmoidSigmoid4forward_lstm_379/while/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-forward_lstm_379/while/lstm_cell_1138/Sigmoid’
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_1Sigmoid4forward_lstm_379/while/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_1к
)forward_lstm_379/while/lstm_cell_1138/mulMul3forward_lstm_379/while/lstm_cell_1138/Sigmoid_1:y:0$forward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_379/while/lstm_cell_1138/mul»
*forward_lstm_379/while/lstm_cell_1138/ReluRelu4forward_lstm_379/while/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22,
*forward_lstm_379/while/lstm_cell_1138/ReluА
+forward_lstm_379/while/lstm_cell_1138/mul_1Mul1forward_lstm_379/while/lstm_cell_1138/Sigmoid:y:08forward_lstm_379/while/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_379/while/lstm_cell_1138/mul_1х
+forward_lstm_379/while/lstm_cell_1138/add_1AddV2-forward_lstm_379/while/lstm_cell_1138/mul:z:0/forward_lstm_379/while/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_379/while/lstm_cell_1138/add_1’
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_2Sigmoid4forward_lstm_379/while/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_2«
,forward_lstm_379/while/lstm_cell_1138/Relu_1Relu/forward_lstm_379/while/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,forward_lstm_379/while/lstm_cell_1138/Relu_1Д
+forward_lstm_379/while/lstm_cell_1138/mul_2Mul3forward_lstm_379/while/lstm_cell_1138/Sigmoid_2:y:0:forward_lstm_379/while/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_379/while/lstm_cell_1138/mul_2х
forward_lstm_379/while/SelectSelect"forward_lstm_379/while/Greater:z:0/forward_lstm_379/while/lstm_cell_1138/mul_2:z:0$forward_lstm_379_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/while/Selectщ
forward_lstm_379/while/Select_1Select"forward_lstm_379/while/Greater:z:0/forward_lstm_379/while/lstm_cell_1138/mul_2:z:0$forward_lstm_379_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_379/while/Select_1щ
forward_lstm_379/while/Select_2Select"forward_lstm_379/while/Greater:z:0/forward_lstm_379/while/lstm_cell_1138/add_1:z:0$forward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_379/while/Select_2Ѓ
;forward_lstm_379/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_379_while_placeholder_1"forward_lstm_379_while_placeholder&forward_lstm_379/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_379/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_379/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_379/while/add/y≠
forward_lstm_379/while/addAddV2"forward_lstm_379_while_placeholder%forward_lstm_379/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/while/addВ
forward_lstm_379/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_379/while/add_1/yЋ
forward_lstm_379/while/add_1AddV2:forward_lstm_379_while_forward_lstm_379_while_loop_counter'forward_lstm_379/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/while/add_1ѓ
forward_lstm_379/while/IdentityIdentity forward_lstm_379/while/add_1:z:0^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_379/while/Identity”
!forward_lstm_379/while/Identity_1Identity@forward_lstm_379_while_forward_lstm_379_while_maximum_iterations^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_379/while/Identity_1±
!forward_lstm_379/while/Identity_2Identityforward_lstm_379/while/add:z:0^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_379/while/Identity_2ё
!forward_lstm_379/while/Identity_3IdentityKforward_lstm_379/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_379/while/Identity_3 
!forward_lstm_379/while/Identity_4Identity&forward_lstm_379/while/Select:output:0^forward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_379/while/Identity_4ћ
!forward_lstm_379/while/Identity_5Identity(forward_lstm_379/while/Select_1:output:0^forward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_379/while/Identity_5ћ
!forward_lstm_379/while/Identity_6Identity(forward_lstm_379/while/Select_2:output:0^forward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_379/while/Identity_6є
forward_lstm_379/while/NoOpNoOp=^forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp<^forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp>^forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_379/while/NoOp"t
7forward_lstm_379_while_forward_lstm_379_strided_slice_19forward_lstm_379_while_forward_lstm_379_strided_slice_1_0"n
4forward_lstm_379_while_greater_forward_lstm_379_cast6forward_lstm_379_while_greater_forward_lstm_379_cast_0"K
forward_lstm_379_while_identity(forward_lstm_379/while/Identity:output:0"O
!forward_lstm_379_while_identity_1*forward_lstm_379/while/Identity_1:output:0"O
!forward_lstm_379_while_identity_2*forward_lstm_379/while/Identity_2:output:0"O
!forward_lstm_379_while_identity_3*forward_lstm_379/while/Identity_3:output:0"O
!forward_lstm_379_while_identity_4*forward_lstm_379/while/Identity_4:output:0"O
!forward_lstm_379_while_identity_5*forward_lstm_379/while/Identity_5:output:0"O
!forward_lstm_379_while_identity_6*forward_lstm_379/while/Identity_6:output:0"Р
Eforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resourceGforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0"Т
Fforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resourceHforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0"О
Dforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resourceFforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0"м
sforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensoruforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2|
<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp2z
;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp2~
=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:€€€€€€€€€
Іg
н
%backward_lstm_379_while_body_42660617@
<backward_lstm_379_while_backward_lstm_379_while_loop_counterF
Bbackward_lstm_379_while_backward_lstm_379_while_maximum_iterations'
#backward_lstm_379_while_placeholder)
%backward_lstm_379_while_placeholder_1)
%backward_lstm_379_while_placeholder_2)
%backward_lstm_379_while_placeholder_3)
%backward_lstm_379_while_placeholder_4?
;backward_lstm_379_while_backward_lstm_379_strided_slice_1_0{
wbackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_379_while_less_backward_lstm_379_sub_1_0Z
Gbackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0:	»\
Ibackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0:	2»W
Hbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0:	»$
 backward_lstm_379_while_identity&
"backward_lstm_379_while_identity_1&
"backward_lstm_379_while_identity_2&
"backward_lstm_379_while_identity_3&
"backward_lstm_379_while_identity_4&
"backward_lstm_379_while_identity_5&
"backward_lstm_379_while_identity_6=
9backward_lstm_379_while_backward_lstm_379_strided_slice_1y
ubackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_379_while_less_backward_lstm_379_sub_1X
Ebackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource:	»Z
Gbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource:	2»U
Fbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource:	»ИҐ=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpҐ<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOpҐ>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpз
Ibackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2K
Ibackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeњ
;backward_lstm_379/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_379_while_placeholderRbackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02=
;backward_lstm_379/while/TensorArrayV2Read/TensorListGetItemѕ
backward_lstm_379/while/LessLess6backward_lstm_379_while_less_backward_lstm_379_sub_1_0#backward_lstm_379_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_379/while/LessЕ
<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOpGbackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02>
<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp•
-backward_lstm_379/while/lstm_cell_1139/MatMulMatMulBbackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem:item:0Dbackward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-backward_lstm_379/while/lstm_cell_1139/MatMulЛ
>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOpIbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02@
>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpО
/backward_lstm_379/while/lstm_cell_1139/MatMul_1MatMul%backward_lstm_379_while_placeholder_3Fbackward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»21
/backward_lstm_379/while/lstm_cell_1139/MatMul_1И
*backward_lstm_379/while/lstm_cell_1139/addAddV27backward_lstm_379/while/lstm_cell_1139/MatMul:product:09backward_lstm_379/while/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2,
*backward_lstm_379/while/lstm_cell_1139/addД
=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOpHbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02?
=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpХ
.backward_lstm_379/while/lstm_cell_1139/BiasAddBiasAdd.backward_lstm_379/while/lstm_cell_1139/add:z:0Ebackward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.backward_lstm_379/while/lstm_cell_1139/BiasAdd≤
6backward_lstm_379/while/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6backward_lstm_379/while/lstm_cell_1139/split/split_dimџ
,backward_lstm_379/while/lstm_cell_1139/splitSplit?backward_lstm_379/while/lstm_cell_1139/split/split_dim:output:07backward_lstm_379/while/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2.
,backward_lstm_379/while/lstm_cell_1139/split‘
.backward_lstm_379/while/lstm_cell_1139/SigmoidSigmoid5backward_lstm_379/while/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€220
.backward_lstm_379/while/lstm_cell_1139/SigmoidЎ
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_1Sigmoid5backward_lstm_379/while/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_1о
*backward_lstm_379/while/lstm_cell_1139/mulMul4backward_lstm_379/while/lstm_cell_1139/Sigmoid_1:y:0%backward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_379/while/lstm_cell_1139/mulЋ
+backward_lstm_379/while/lstm_cell_1139/ReluRelu5backward_lstm_379/while/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22-
+backward_lstm_379/while/lstm_cell_1139/ReluД
,backward_lstm_379/while/lstm_cell_1139/mul_1Mul2backward_lstm_379/while/lstm_cell_1139/Sigmoid:y:09backward_lstm_379/while/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_379/while/lstm_cell_1139/mul_1щ
,backward_lstm_379/while/lstm_cell_1139/add_1AddV2.backward_lstm_379/while/lstm_cell_1139/mul:z:00backward_lstm_379/while/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_379/while/lstm_cell_1139/add_1Ў
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_2Sigmoid5backward_lstm_379/while/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_2 
-backward_lstm_379/while/lstm_cell_1139/Relu_1Relu0backward_lstm_379/while/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-backward_lstm_379/while/lstm_cell_1139/Relu_1И
,backward_lstm_379/while/lstm_cell_1139/mul_2Mul4backward_lstm_379/while/lstm_cell_1139/Sigmoid_2:y:0;backward_lstm_379/while/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_379/while/lstm_cell_1139/mul_2ч
backward_lstm_379/while/SelectSelect backward_lstm_379/while/Less:z:00backward_lstm_379/while/lstm_cell_1139/mul_2:z:0%backward_lstm_379_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22 
backward_lstm_379/while/Selectы
 backward_lstm_379/while/Select_1Select backward_lstm_379/while/Less:z:00backward_lstm_379/while/lstm_cell_1139/mul_2:z:0%backward_lstm_379_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_379/while/Select_1ы
 backward_lstm_379/while/Select_2Select backward_lstm_379/while/Less:z:00backward_lstm_379/while/lstm_cell_1139/add_1:z:0%backward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_379/while/Select_2≥
<backward_lstm_379/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_379_while_placeholder_1#backward_lstm_379_while_placeholder'backward_lstm_379/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_379/while/TensorArrayV2Write/TensorListSetItemА
backward_lstm_379/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_379/while/add/y±
backward_lstm_379/while/addAddV2#backward_lstm_379_while_placeholder&backward_lstm_379/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/while/addД
backward_lstm_379/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_379/while/add_1/y–
backward_lstm_379/while/add_1AddV2<backward_lstm_379_while_backward_lstm_379_while_loop_counter(backward_lstm_379/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/while/add_1≥
 backward_lstm_379/while/IdentityIdentity!backward_lstm_379/while/add_1:z:0^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_379/while/IdentityЎ
"backward_lstm_379/while/Identity_1IdentityBbackward_lstm_379_while_backward_lstm_379_while_maximum_iterations^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_379/while/Identity_1µ
"backward_lstm_379/while/Identity_2Identitybackward_lstm_379/while/add:z:0^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_379/while/Identity_2в
"backward_lstm_379/while/Identity_3IdentityLbackward_lstm_379/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_379/while/Identity_3ќ
"backward_lstm_379/while/Identity_4Identity'backward_lstm_379/while/Select:output:0^backward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_379/while/Identity_4–
"backward_lstm_379/while/Identity_5Identity)backward_lstm_379/while/Select_1:output:0^backward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_379/while/Identity_5–
"backward_lstm_379/while/Identity_6Identity)backward_lstm_379/while/Select_2:output:0^backward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_379/while/Identity_6Њ
backward_lstm_379/while/NoOpNoOp>^backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp=^backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp?^backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_379/while/NoOp"x
9backward_lstm_379_while_backward_lstm_379_strided_slice_1;backward_lstm_379_while_backward_lstm_379_strided_slice_1_0"M
 backward_lstm_379_while_identity)backward_lstm_379/while/Identity:output:0"Q
"backward_lstm_379_while_identity_1+backward_lstm_379/while/Identity_1:output:0"Q
"backward_lstm_379_while_identity_2+backward_lstm_379/while/Identity_2:output:0"Q
"backward_lstm_379_while_identity_3+backward_lstm_379/while/Identity_3:output:0"Q
"backward_lstm_379_while_identity_4+backward_lstm_379/while/Identity_4:output:0"Q
"backward_lstm_379_while_identity_5+backward_lstm_379/while/Identity_5:output:0"Q
"backward_lstm_379_while_identity_6+backward_lstm_379/while/Identity_6:output:0"n
4backward_lstm_379_while_less_backward_lstm_379_sub_16backward_lstm_379_while_less_backward_lstm_379_sub_1_0"Т
Fbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resourceHbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0"Ф
Gbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resourceIbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0"Р
Ebackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resourceGbackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0"р
ubackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2~
=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp2|
<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp2А
>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:€€€€€€€€€
зe
Ћ
$forward_lstm_379_while_body_42659998>
:forward_lstm_379_while_forward_lstm_379_while_loop_counterD
@forward_lstm_379_while_forward_lstm_379_while_maximum_iterations&
"forward_lstm_379_while_placeholder(
$forward_lstm_379_while_placeholder_1(
$forward_lstm_379_while_placeholder_2(
$forward_lstm_379_while_placeholder_3(
$forward_lstm_379_while_placeholder_4=
9forward_lstm_379_while_forward_lstm_379_strided_slice_1_0y
uforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_379_while_greater_forward_lstm_379_cast_0Y
Fforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0:	»[
Hforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0:	2»V
Gforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0:	»#
forward_lstm_379_while_identity%
!forward_lstm_379_while_identity_1%
!forward_lstm_379_while_identity_2%
!forward_lstm_379_while_identity_3%
!forward_lstm_379_while_identity_4%
!forward_lstm_379_while_identity_5%
!forward_lstm_379_while_identity_6;
7forward_lstm_379_while_forward_lstm_379_strided_slice_1w
sforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_379_while_greater_forward_lstm_379_castW
Dforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource:	»Y
Fforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource:	2»T
Eforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource:	»ИҐ<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpҐ;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOpҐ=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpе
Hforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2J
Hforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeє
:forward_lstm_379/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_379_while_placeholderQforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02<
:forward_lstm_379/while/TensorArrayV2Read/TensorListGetItem’
forward_lstm_379/while/GreaterGreater6forward_lstm_379_while_greater_forward_lstm_379_cast_0"forward_lstm_379_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2 
forward_lstm_379/while/GreaterВ
;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOpFforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02=
;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp°
,forward_lstm_379/while/lstm_cell_1138/MatMulMatMulAforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem:item:0Cforward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2.
,forward_lstm_379/while/lstm_cell_1138/MatMulИ
=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOpHforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02?
=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpК
.forward_lstm_379/while/lstm_cell_1138/MatMul_1MatMul$forward_lstm_379_while_placeholder_3Eforward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.forward_lstm_379/while/lstm_cell_1138/MatMul_1Д
)forward_lstm_379/while/lstm_cell_1138/addAddV26forward_lstm_379/while/lstm_cell_1138/MatMul:product:08forward_lstm_379/while/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)forward_lstm_379/while/lstm_cell_1138/addБ
<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOpGforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02>
<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpС
-forward_lstm_379/while/lstm_cell_1138/BiasAddBiasAdd-forward_lstm_379/while/lstm_cell_1138/add:z:0Dforward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-forward_lstm_379/while/lstm_cell_1138/BiasAdd∞
5forward_lstm_379/while/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5forward_lstm_379/while/lstm_cell_1138/split/split_dim„
+forward_lstm_379/while/lstm_cell_1138/splitSplit>forward_lstm_379/while/lstm_cell_1138/split/split_dim:output:06forward_lstm_379/while/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2-
+forward_lstm_379/while/lstm_cell_1138/split—
-forward_lstm_379/while/lstm_cell_1138/SigmoidSigmoid4forward_lstm_379/while/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-forward_lstm_379/while/lstm_cell_1138/Sigmoid’
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_1Sigmoid4forward_lstm_379/while/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_1к
)forward_lstm_379/while/lstm_cell_1138/mulMul3forward_lstm_379/while/lstm_cell_1138/Sigmoid_1:y:0$forward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_379/while/lstm_cell_1138/mul»
*forward_lstm_379/while/lstm_cell_1138/ReluRelu4forward_lstm_379/while/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22,
*forward_lstm_379/while/lstm_cell_1138/ReluА
+forward_lstm_379/while/lstm_cell_1138/mul_1Mul1forward_lstm_379/while/lstm_cell_1138/Sigmoid:y:08forward_lstm_379/while/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_379/while/lstm_cell_1138/mul_1х
+forward_lstm_379/while/lstm_cell_1138/add_1AddV2-forward_lstm_379/while/lstm_cell_1138/mul:z:0/forward_lstm_379/while/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_379/while/lstm_cell_1138/add_1’
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_2Sigmoid4forward_lstm_379/while/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_2«
,forward_lstm_379/while/lstm_cell_1138/Relu_1Relu/forward_lstm_379/while/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,forward_lstm_379/while/lstm_cell_1138/Relu_1Д
+forward_lstm_379/while/lstm_cell_1138/mul_2Mul3forward_lstm_379/while/lstm_cell_1138/Sigmoid_2:y:0:forward_lstm_379/while/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_379/while/lstm_cell_1138/mul_2х
forward_lstm_379/while/SelectSelect"forward_lstm_379/while/Greater:z:0/forward_lstm_379/while/lstm_cell_1138/mul_2:z:0$forward_lstm_379_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/while/Selectщ
forward_lstm_379/while/Select_1Select"forward_lstm_379/while/Greater:z:0/forward_lstm_379/while/lstm_cell_1138/mul_2:z:0$forward_lstm_379_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_379/while/Select_1щ
forward_lstm_379/while/Select_2Select"forward_lstm_379/while/Greater:z:0/forward_lstm_379/while/lstm_cell_1138/add_1:z:0$forward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_379/while/Select_2Ѓ
;forward_lstm_379/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_379_while_placeholder_1"forward_lstm_379_while_placeholder&forward_lstm_379/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_379/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_379/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_379/while/add/y≠
forward_lstm_379/while/addAddV2"forward_lstm_379_while_placeholder%forward_lstm_379/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/while/addВ
forward_lstm_379/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_379/while/add_1/yЋ
forward_lstm_379/while/add_1AddV2:forward_lstm_379_while_forward_lstm_379_while_loop_counter'forward_lstm_379/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/while/add_1ѓ
forward_lstm_379/while/IdentityIdentity forward_lstm_379/while/add_1:z:0^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_379/while/Identity”
!forward_lstm_379/while/Identity_1Identity@forward_lstm_379_while_forward_lstm_379_while_maximum_iterations^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_379/while/Identity_1±
!forward_lstm_379/while/Identity_2Identityforward_lstm_379/while/add:z:0^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_379/while/Identity_2ё
!forward_lstm_379/while/Identity_3IdentityKforward_lstm_379/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_379/while/Identity_3 
!forward_lstm_379/while/Identity_4Identity&forward_lstm_379/while/Select:output:0^forward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_379/while/Identity_4ћ
!forward_lstm_379/while/Identity_5Identity(forward_lstm_379/while/Select_1:output:0^forward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_379/while/Identity_5ћ
!forward_lstm_379/while/Identity_6Identity(forward_lstm_379/while/Select_2:output:0^forward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_379/while/Identity_6є
forward_lstm_379/while/NoOpNoOp=^forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp<^forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp>^forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_379/while/NoOp"t
7forward_lstm_379_while_forward_lstm_379_strided_slice_19forward_lstm_379_while_forward_lstm_379_strided_slice_1_0"n
4forward_lstm_379_while_greater_forward_lstm_379_cast6forward_lstm_379_while_greater_forward_lstm_379_cast_0"K
forward_lstm_379_while_identity(forward_lstm_379/while/Identity:output:0"O
!forward_lstm_379_while_identity_1*forward_lstm_379/while/Identity_1:output:0"O
!forward_lstm_379_while_identity_2*forward_lstm_379/while/Identity_2:output:0"O
!forward_lstm_379_while_identity_3*forward_lstm_379/while/Identity_3:output:0"O
!forward_lstm_379_while_identity_4*forward_lstm_379/while/Identity_4:output:0"O
!forward_lstm_379_while_identity_5*forward_lstm_379/while/Identity_5:output:0"O
!forward_lstm_379_while_identity_6*forward_lstm_379/while/Identity_6:output:0"Р
Eforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resourceGforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0"Т
Fforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resourceHforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0"О
Dforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resourceFforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0"м
sforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensoruforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2|
<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp2z
;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp2~
=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:€€€€€€€€€
м]
≤
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_42659263

inputs@
-lstm_cell_1138_matmul_readvariableop_resource:	»B
/lstm_cell_1138_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1138_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1138/BiasAdd/ReadVariableOpҐ$lstm_cell_1138/MatMul/ReadVariableOpҐ&lstm_cell_1138/MatMul_1/ReadVariableOpҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€22
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
B :и2
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
zeros_1/packed/1Й
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
:€€€€€€€€€22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permМ
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2Е
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2
strided_slice_2ї
$lstm_cell_1138/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1138_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1138/MatMul/ReadVariableOp≥
lstm_cell_1138/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/MatMulЅ
&lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1138_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1138/MatMul_1/ReadVariableOpѓ
lstm_cell_1138/MatMul_1MatMulzeros:output:0.lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/MatMul_1®
lstm_cell_1138/addAddV2lstm_cell_1138/MatMul:product:0!lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/addЇ
%lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1138_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1138/BiasAdd/ReadVariableOpµ
lstm_cell_1138/BiasAddBiasAddlstm_cell_1138/add:z:0-lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/BiasAddВ
lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1138/split/split_dimы
lstm_cell_1138/splitSplit'lstm_cell_1138/split/split_dim:output:0lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1138/splitМ
lstm_cell_1138/SigmoidSigmoidlstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/SigmoidР
lstm_cell_1138/Sigmoid_1Sigmoidlstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Sigmoid_1С
lstm_cell_1138/mulMullstm_cell_1138/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/mulГ
lstm_cell_1138/ReluRelulstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Relu§
lstm_cell_1138/mul_1Mullstm_cell_1138/Sigmoid:y:0!lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/mul_1Щ
lstm_cell_1138/add_1AddV2lstm_cell_1138/mul:z:0lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/add_1Р
lstm_cell_1138/Sigmoid_2Sigmoidlstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Sigmoid_2В
lstm_cell_1138/Relu_1Relulstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Relu_1®
lstm_cell_1138/mul_2Mullstm_cell_1138/Sigmoid_2:y:0#lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterХ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1138_matmul_readvariableop_resource/lstm_cell_1138_matmul_1_readvariableop_resource.lstm_cell_1138_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42659179*
condR
while_cond_42659178*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
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
:€€€€€€€€€22

Identityќ
NoOpNoOp&^lstm_cell_1138/BiasAdd/ReadVariableOp%^lstm_cell_1138/MatMul/ReadVariableOp'^lstm_cell_1138/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1138/BiasAdd/ReadVariableOp%lstm_cell_1138/BiasAdd/ReadVariableOp2L
$lstm_cell_1138/MatMul/ReadVariableOp$lstm_cell_1138/MatMul/ReadVariableOp2P
&lstm_cell_1138/MatMul_1/ReadVariableOp&lstm_cell_1138/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
зe
Ћ
$forward_lstm_379_while_body_42662008>
:forward_lstm_379_while_forward_lstm_379_while_loop_counterD
@forward_lstm_379_while_forward_lstm_379_while_maximum_iterations&
"forward_lstm_379_while_placeholder(
$forward_lstm_379_while_placeholder_1(
$forward_lstm_379_while_placeholder_2(
$forward_lstm_379_while_placeholder_3(
$forward_lstm_379_while_placeholder_4=
9forward_lstm_379_while_forward_lstm_379_strided_slice_1_0y
uforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_379_while_greater_forward_lstm_379_cast_0Y
Fforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0:	»[
Hforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0:	2»V
Gforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0:	»#
forward_lstm_379_while_identity%
!forward_lstm_379_while_identity_1%
!forward_lstm_379_while_identity_2%
!forward_lstm_379_while_identity_3%
!forward_lstm_379_while_identity_4%
!forward_lstm_379_while_identity_5%
!forward_lstm_379_while_identity_6;
7forward_lstm_379_while_forward_lstm_379_strided_slice_1w
sforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_379_while_greater_forward_lstm_379_castW
Dforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource:	»Y
Fforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource:	2»T
Eforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource:	»ИҐ<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpҐ;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOpҐ=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpе
Hforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2J
Hforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeє
:forward_lstm_379/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_379_while_placeholderQforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02<
:forward_lstm_379/while/TensorArrayV2Read/TensorListGetItem’
forward_lstm_379/while/GreaterGreater6forward_lstm_379_while_greater_forward_lstm_379_cast_0"forward_lstm_379_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2 
forward_lstm_379/while/GreaterВ
;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOpFforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02=
;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp°
,forward_lstm_379/while/lstm_cell_1138/MatMulMatMulAforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem:item:0Cforward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2.
,forward_lstm_379/while/lstm_cell_1138/MatMulИ
=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOpHforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02?
=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpК
.forward_lstm_379/while/lstm_cell_1138/MatMul_1MatMul$forward_lstm_379_while_placeholder_3Eforward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.forward_lstm_379/while/lstm_cell_1138/MatMul_1Д
)forward_lstm_379/while/lstm_cell_1138/addAddV26forward_lstm_379/while/lstm_cell_1138/MatMul:product:08forward_lstm_379/while/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)forward_lstm_379/while/lstm_cell_1138/addБ
<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOpGforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02>
<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpС
-forward_lstm_379/while/lstm_cell_1138/BiasAddBiasAdd-forward_lstm_379/while/lstm_cell_1138/add:z:0Dforward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-forward_lstm_379/while/lstm_cell_1138/BiasAdd∞
5forward_lstm_379/while/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5forward_lstm_379/while/lstm_cell_1138/split/split_dim„
+forward_lstm_379/while/lstm_cell_1138/splitSplit>forward_lstm_379/while/lstm_cell_1138/split/split_dim:output:06forward_lstm_379/while/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2-
+forward_lstm_379/while/lstm_cell_1138/split—
-forward_lstm_379/while/lstm_cell_1138/SigmoidSigmoid4forward_lstm_379/while/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-forward_lstm_379/while/lstm_cell_1138/Sigmoid’
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_1Sigmoid4forward_lstm_379/while/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_1к
)forward_lstm_379/while/lstm_cell_1138/mulMul3forward_lstm_379/while/lstm_cell_1138/Sigmoid_1:y:0$forward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_379/while/lstm_cell_1138/mul»
*forward_lstm_379/while/lstm_cell_1138/ReluRelu4forward_lstm_379/while/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22,
*forward_lstm_379/while/lstm_cell_1138/ReluА
+forward_lstm_379/while/lstm_cell_1138/mul_1Mul1forward_lstm_379/while/lstm_cell_1138/Sigmoid:y:08forward_lstm_379/while/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_379/while/lstm_cell_1138/mul_1х
+forward_lstm_379/while/lstm_cell_1138/add_1AddV2-forward_lstm_379/while/lstm_cell_1138/mul:z:0/forward_lstm_379/while/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_379/while/lstm_cell_1138/add_1’
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_2Sigmoid4forward_lstm_379/while/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_2«
,forward_lstm_379/while/lstm_cell_1138/Relu_1Relu/forward_lstm_379/while/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,forward_lstm_379/while/lstm_cell_1138/Relu_1Д
+forward_lstm_379/while/lstm_cell_1138/mul_2Mul3forward_lstm_379/while/lstm_cell_1138/Sigmoid_2:y:0:forward_lstm_379/while/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_379/while/lstm_cell_1138/mul_2х
forward_lstm_379/while/SelectSelect"forward_lstm_379/while/Greater:z:0/forward_lstm_379/while/lstm_cell_1138/mul_2:z:0$forward_lstm_379_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/while/Selectщ
forward_lstm_379/while/Select_1Select"forward_lstm_379/while/Greater:z:0/forward_lstm_379/while/lstm_cell_1138/mul_2:z:0$forward_lstm_379_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_379/while/Select_1щ
forward_lstm_379/while/Select_2Select"forward_lstm_379/while/Greater:z:0/forward_lstm_379/while/lstm_cell_1138/add_1:z:0$forward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_379/while/Select_2Ѓ
;forward_lstm_379/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_379_while_placeholder_1"forward_lstm_379_while_placeholder&forward_lstm_379/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_379/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_379/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_379/while/add/y≠
forward_lstm_379/while/addAddV2"forward_lstm_379_while_placeholder%forward_lstm_379/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/while/addВ
forward_lstm_379/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_379/while/add_1/yЋ
forward_lstm_379/while/add_1AddV2:forward_lstm_379_while_forward_lstm_379_while_loop_counter'forward_lstm_379/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/while/add_1ѓ
forward_lstm_379/while/IdentityIdentity forward_lstm_379/while/add_1:z:0^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_379/while/Identity”
!forward_lstm_379/while/Identity_1Identity@forward_lstm_379_while_forward_lstm_379_while_maximum_iterations^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_379/while/Identity_1±
!forward_lstm_379/while/Identity_2Identityforward_lstm_379/while/add:z:0^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_379/while/Identity_2ё
!forward_lstm_379/while/Identity_3IdentityKforward_lstm_379/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_379/while/Identity_3 
!forward_lstm_379/while/Identity_4Identity&forward_lstm_379/while/Select:output:0^forward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_379/while/Identity_4ћ
!forward_lstm_379/while/Identity_5Identity(forward_lstm_379/while/Select_1:output:0^forward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_379/while/Identity_5ћ
!forward_lstm_379/while/Identity_6Identity(forward_lstm_379/while/Select_2:output:0^forward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_379/while/Identity_6є
forward_lstm_379/while/NoOpNoOp=^forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp<^forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp>^forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_379/while/NoOp"t
7forward_lstm_379_while_forward_lstm_379_strided_slice_19forward_lstm_379_while_forward_lstm_379_strided_slice_1_0"n
4forward_lstm_379_while_greater_forward_lstm_379_cast6forward_lstm_379_while_greater_forward_lstm_379_cast_0"K
forward_lstm_379_while_identity(forward_lstm_379/while/Identity:output:0"O
!forward_lstm_379_while_identity_1*forward_lstm_379/while/Identity_1:output:0"O
!forward_lstm_379_while_identity_2*forward_lstm_379/while/Identity_2:output:0"O
!forward_lstm_379_while_identity_3*forward_lstm_379/while/Identity_3:output:0"O
!forward_lstm_379_while_identity_4*forward_lstm_379/while/Identity_4:output:0"O
!forward_lstm_379_while_identity_5*forward_lstm_379/while/Identity_5:output:0"O
!forward_lstm_379_while_identity_6*forward_lstm_379/while/Identity_6:output:0"Р
Eforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resourceGforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0"Т
Fforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resourceHforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0"О
Dforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resourceFforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0"м
sforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensoruforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2|
<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp2z
;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp2~
=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:€€€€€€€€€
я
Ќ
while_cond_42658770
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_42658770___redundant_placeholder06
2while_while_cond_42658770___redundant_placeholder16
2while_while_cond_42658770___redundant_placeholder26
2while_while_cond_42658770___redundant_placeholder3
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
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:
уд
У
#__inference__wrapped_model_42657838

args_0
args_0_1	r
_sequential_379_bidirectional_379_forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource:	»t
asequential_379_bidirectional_379_forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource:	2»o
`sequential_379_bidirectional_379_forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource:	»s
`sequential_379_bidirectional_379_backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resource:	»u
bsequential_379_bidirectional_379_backward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource:	2»p
asequential_379_bidirectional_379_backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource:	»I
7sequential_379_dense_379_matmul_readvariableop_resource:dF
8sequential_379_dense_379_biasadd_readvariableop_resource:
identityИҐXsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpҐWsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpҐYsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpҐ8sequential_379/bidirectional_379/backward_lstm_379/whileҐWsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpҐVsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpҐXsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpҐ7sequential_379/bidirectional_379/forward_lstm_379/whileҐ/sequential_379/dense_379/BiasAdd/ReadVariableOpҐ.sequential_379/dense_379/MatMul/ReadVariableOpў
Fsequential_379/bidirectional_379/forward_lstm_379/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2H
Fsequential_379/bidirectional_379/forward_lstm_379/RaggedToTensor/zerosџ
Fsequential_379/bidirectional_379/forward_lstm_379/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2H
Fsequential_379/bidirectional_379/forward_lstm_379/RaggedToTensor/ConstЭ
Usequential_379/bidirectional_379/forward_lstm_379/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorOsequential_379/bidirectional_379/forward_lstm_379/RaggedToTensor/Const:output:0args_0Osequential_379/bidirectional_379/forward_lstm_379/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2W
Usequential_379/bidirectional_379/forward_lstm_379/RaggedToTensor/RaggedTensorToTensorЖ
\sequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2^
\sequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice/stackК
^sequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2`
^sequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1К
^sequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2`
^sequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2ќ
Vsequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1esequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack:output:0gsequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1:output:0gsequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask2X
Vsequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_sliceК
^sequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2`
^sequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stackЧ
`sequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2b
`sequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1О
`sequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2b
`sequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2Џ
Xsequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1gsequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack:output:0isequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0isequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask2Z
Xsequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice_1Х
Lsequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/subSub_sequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice:output:0asequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2N
Lsequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/subЗ
6sequential_379/bidirectional_379/forward_lstm_379/CastCastPsequential_379/bidirectional_379/forward_lstm_379/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€28
6sequential_379/bidirectional_379/forward_lstm_379/CastА
7sequential_379/bidirectional_379/forward_lstm_379/ShapeShape^sequential_379/bidirectional_379/forward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:29
7sequential_379/bidirectional_379/forward_lstm_379/ShapeЎ
Esequential_379/bidirectional_379/forward_lstm_379/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_379/bidirectional_379/forward_lstm_379/strided_slice/stack№
Gsequential_379/bidirectional_379/forward_lstm_379/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_379/bidirectional_379/forward_lstm_379/strided_slice/stack_1№
Gsequential_379/bidirectional_379/forward_lstm_379/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_379/bidirectional_379/forward_lstm_379/strided_slice/stack_2О
?sequential_379/bidirectional_379/forward_lstm_379/strided_sliceStridedSlice@sequential_379/bidirectional_379/forward_lstm_379/Shape:output:0Nsequential_379/bidirectional_379/forward_lstm_379/strided_slice/stack:output:0Psequential_379/bidirectional_379/forward_lstm_379/strided_slice/stack_1:output:0Psequential_379/bidirectional_379/forward_lstm_379/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?sequential_379/bidirectional_379/forward_lstm_379/strided_sliceј
=sequential_379/bidirectional_379/forward_lstm_379/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22?
=sequential_379/bidirectional_379/forward_lstm_379/zeros/mul/yі
;sequential_379/bidirectional_379/forward_lstm_379/zeros/mulMulHsequential_379/bidirectional_379/forward_lstm_379/strided_slice:output:0Fsequential_379/bidirectional_379/forward_lstm_379/zeros/mul/y:output:0*
T0*
_output_shapes
: 2=
;sequential_379/bidirectional_379/forward_lstm_379/zeros/mul√
>sequential_379/bidirectional_379/forward_lstm_379/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2@
>sequential_379/bidirectional_379/forward_lstm_379/zeros/Less/yѓ
<sequential_379/bidirectional_379/forward_lstm_379/zeros/LessLess?sequential_379/bidirectional_379/forward_lstm_379/zeros/mul:z:0Gsequential_379/bidirectional_379/forward_lstm_379/zeros/Less/y:output:0*
T0*
_output_shapes
: 2>
<sequential_379/bidirectional_379/forward_lstm_379/zeros/Less∆
@sequential_379/bidirectional_379/forward_lstm_379/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22B
@sequential_379/bidirectional_379/forward_lstm_379/zeros/packed/1Ћ
>sequential_379/bidirectional_379/forward_lstm_379/zeros/packedPackHsequential_379/bidirectional_379/forward_lstm_379/strided_slice:output:0Isequential_379/bidirectional_379/forward_lstm_379/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2@
>sequential_379/bidirectional_379/forward_lstm_379/zeros/packed«
=sequential_379/bidirectional_379/forward_lstm_379/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2?
=sequential_379/bidirectional_379/forward_lstm_379/zeros/Constљ
7sequential_379/bidirectional_379/forward_lstm_379/zerosFillGsequential_379/bidirectional_379/forward_lstm_379/zeros/packed:output:0Fsequential_379/bidirectional_379/forward_lstm_379/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€229
7sequential_379/bidirectional_379/forward_lstm_379/zerosƒ
?sequential_379/bidirectional_379/forward_lstm_379/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22A
?sequential_379/bidirectional_379/forward_lstm_379/zeros_1/mul/yЇ
=sequential_379/bidirectional_379/forward_lstm_379/zeros_1/mulMulHsequential_379/bidirectional_379/forward_lstm_379/strided_slice:output:0Hsequential_379/bidirectional_379/forward_lstm_379/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2?
=sequential_379/bidirectional_379/forward_lstm_379/zeros_1/mul«
@sequential_379/bidirectional_379/forward_lstm_379/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2B
@sequential_379/bidirectional_379/forward_lstm_379/zeros_1/Less/yЈ
>sequential_379/bidirectional_379/forward_lstm_379/zeros_1/LessLessAsequential_379/bidirectional_379/forward_lstm_379/zeros_1/mul:z:0Isequential_379/bidirectional_379/forward_lstm_379/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2@
>sequential_379/bidirectional_379/forward_lstm_379/zeros_1/Less 
Bsequential_379/bidirectional_379/forward_lstm_379/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22D
Bsequential_379/bidirectional_379/forward_lstm_379/zeros_1/packed/1—
@sequential_379/bidirectional_379/forward_lstm_379/zeros_1/packedPackHsequential_379/bidirectional_379/forward_lstm_379/strided_slice:output:0Ksequential_379/bidirectional_379/forward_lstm_379/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2B
@sequential_379/bidirectional_379/forward_lstm_379/zeros_1/packedЋ
?sequential_379/bidirectional_379/forward_lstm_379/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2A
?sequential_379/bidirectional_379/forward_lstm_379/zeros_1/Const≈
9sequential_379/bidirectional_379/forward_lstm_379/zeros_1FillIsequential_379/bidirectional_379/forward_lstm_379/zeros_1/packed:output:0Hsequential_379/bidirectional_379/forward_lstm_379/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22;
9sequential_379/bidirectional_379/forward_lstm_379/zeros_1ў
@sequential_379/bidirectional_379/forward_lstm_379/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2B
@sequential_379/bidirectional_379/forward_lstm_379/transpose/permс
;sequential_379/bidirectional_379/forward_lstm_379/transpose	Transpose^sequential_379/bidirectional_379/forward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0Isequential_379/bidirectional_379/forward_lstm_379/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2=
;sequential_379/bidirectional_379/forward_lstm_379/transposeе
9sequential_379/bidirectional_379/forward_lstm_379/Shape_1Shape?sequential_379/bidirectional_379/forward_lstm_379/transpose:y:0*
T0*
_output_shapes
:2;
9sequential_379/bidirectional_379/forward_lstm_379/Shape_1№
Gsequential_379/bidirectional_379/forward_lstm_379/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2I
Gsequential_379/bidirectional_379/forward_lstm_379/strided_slice_1/stackа
Isequential_379/bidirectional_379/forward_lstm_379/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_379/bidirectional_379/forward_lstm_379/strided_slice_1/stack_1а
Isequential_379/bidirectional_379/forward_lstm_379/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_379/bidirectional_379/forward_lstm_379/strided_slice_1/stack_2Ъ
Asequential_379/bidirectional_379/forward_lstm_379/strided_slice_1StridedSliceBsequential_379/bidirectional_379/forward_lstm_379/Shape_1:output:0Psequential_379/bidirectional_379/forward_lstm_379/strided_slice_1/stack:output:0Rsequential_379/bidirectional_379/forward_lstm_379/strided_slice_1/stack_1:output:0Rsequential_379/bidirectional_379/forward_lstm_379/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2C
Asequential_379/bidirectional_379/forward_lstm_379/strided_slice_1й
Msequential_379/bidirectional_379/forward_lstm_379/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2O
Msequential_379/bidirectional_379/forward_lstm_379/TensorArrayV2/element_shapeъ
?sequential_379/bidirectional_379/forward_lstm_379/TensorArrayV2TensorListReserveVsequential_379/bidirectional_379/forward_lstm_379/TensorArrayV2/element_shape:output:0Jsequential_379/bidirectional_379/forward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?sequential_379/bidirectional_379/forward_lstm_379/TensorArrayV2£
gsequential_379/bidirectional_379/forward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2i
gsequential_379/bidirectional_379/forward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeј
Ysequential_379/bidirectional_379/forward_lstm_379/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor?sequential_379/bidirectional_379/forward_lstm_379/transpose:y:0psequential_379/bidirectional_379/forward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02[
Ysequential_379/bidirectional_379/forward_lstm_379/TensorArrayUnstack/TensorListFromTensor№
Gsequential_379/bidirectional_379/forward_lstm_379/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2I
Gsequential_379/bidirectional_379/forward_lstm_379/strided_slice_2/stackа
Isequential_379/bidirectional_379/forward_lstm_379/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_379/bidirectional_379/forward_lstm_379/strided_slice_2/stack_1а
Isequential_379/bidirectional_379/forward_lstm_379/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_379/bidirectional_379/forward_lstm_379/strided_slice_2/stack_2®
Asequential_379/bidirectional_379/forward_lstm_379/strided_slice_2StridedSlice?sequential_379/bidirectional_379/forward_lstm_379/transpose:y:0Psequential_379/bidirectional_379/forward_lstm_379/strided_slice_2/stack:output:0Rsequential_379/bidirectional_379/forward_lstm_379/strided_slice_2/stack_1:output:0Rsequential_379/bidirectional_379/forward_lstm_379/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2C
Asequential_379/bidirectional_379/forward_lstm_379/strided_slice_2—
Vsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOp_sequential_379_bidirectional_379_forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02X
Vsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpы
Gsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMulMatMulJsequential_379/bidirectional_379/forward_lstm_379/strided_slice_2:output:0^sequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2I
Gsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMul„
Xsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOpasequential_379_bidirectional_379_forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02Z
Xsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpч
Isequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMul_1MatMul@sequential_379/bidirectional_379/forward_lstm_379/zeros:output:0`sequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2K
Isequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMul_1р
Dsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/addAddV2Qsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMul:product:0Ssequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2F
Dsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/add–
Wsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOp`sequential_379_bidirectional_379_forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02Y
Wsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpэ
Hsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/BiasAddBiasAddHsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/add:z:0_sequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2J
Hsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/BiasAddж
Psequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2R
Psequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/split/split_dim√
Fsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/splitSplitYsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/split/split_dim:output:0Qsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2H
Fsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/splitҐ
Hsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/SigmoidSigmoidOsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22J
Hsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/Sigmoid¶
Jsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/Sigmoid_1SigmoidOsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22L
Jsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/Sigmoid_1ў
Dsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/mulMulNsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/Sigmoid_1:y:0Bsequential_379/bidirectional_379/forward_lstm_379/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22F
Dsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/mulЩ
Esequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/ReluReluOsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22G
Esequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/Reluм
Fsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/mul_1MulLsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/Sigmoid:y:0Ssequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22H
Fsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/mul_1б
Fsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/add_1AddV2Hsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/mul:z:0Jsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22H
Fsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/add_1¶
Jsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/Sigmoid_2SigmoidOsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22L
Jsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/Sigmoid_2Ш
Gsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/Relu_1ReluJsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22I
Gsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/Relu_1р
Fsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/mul_2MulNsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/Sigmoid_2:y:0Usequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22H
Fsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/mul_2у
Osequential_379/bidirectional_379/forward_lstm_379/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2Q
Osequential_379/bidirectional_379/forward_lstm_379/TensorArrayV2_1/element_shapeА
Asequential_379/bidirectional_379/forward_lstm_379/TensorArrayV2_1TensorListReserveXsequential_379/bidirectional_379/forward_lstm_379/TensorArrayV2_1/element_shape:output:0Jsequential_379/bidirectional_379/forward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02C
Asequential_379/bidirectional_379/forward_lstm_379/TensorArrayV2_1≤
6sequential_379/bidirectional_379/forward_lstm_379/timeConst*
_output_shapes
: *
dtype0*
value	B : 28
6sequential_379/bidirectional_379/forward_lstm_379/timeЗ
<sequential_379/bidirectional_379/forward_lstm_379/zeros_like	ZerosLikeJsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22>
<sequential_379/bidirectional_379/forward_lstm_379/zeros_likeг
Jsequential_379/bidirectional_379/forward_lstm_379/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2L
Jsequential_379/bidirectional_379/forward_lstm_379/while/maximum_iterationsќ
Dsequential_379/bidirectional_379/forward_lstm_379/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dsequential_379/bidirectional_379/forward_lstm_379/while/loop_counter«
7sequential_379/bidirectional_379/forward_lstm_379/whileWhileMsequential_379/bidirectional_379/forward_lstm_379/while/loop_counter:output:0Ssequential_379/bidirectional_379/forward_lstm_379/while/maximum_iterations:output:0?sequential_379/bidirectional_379/forward_lstm_379/time:output:0Jsequential_379/bidirectional_379/forward_lstm_379/TensorArrayV2_1:handle:0@sequential_379/bidirectional_379/forward_lstm_379/zeros_like:y:0@sequential_379/bidirectional_379/forward_lstm_379/zeros:output:0Bsequential_379/bidirectional_379/forward_lstm_379/zeros_1:output:0Jsequential_379/bidirectional_379/forward_lstm_379/strided_slice_1:output:0isequential_379/bidirectional_379/forward_lstm_379/TensorArrayUnstack/TensorListFromTensor:output_handle:0:sequential_379/bidirectional_379/forward_lstm_379/Cast:y:0_sequential_379_bidirectional_379_forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resourceasequential_379_bidirectional_379_forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource`sequential_379_bidirectional_379_forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *Q
bodyIRG
Esequential_379_bidirectional_379_forward_lstm_379_while_body_42657555*Q
condIRG
Esequential_379_bidirectional_379_forward_lstm_379_while_cond_42657554*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 29
7sequential_379/bidirectional_379/forward_lstm_379/whileЩ
bsequential_379/bidirectional_379/forward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2d
bsequential_379/bidirectional_379/forward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeє
Tsequential_379/bidirectional_379/forward_lstm_379/TensorArrayV2Stack/TensorListStackTensorListStack@sequential_379/bidirectional_379/forward_lstm_379/while:output:3ksequential_379/bidirectional_379/forward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02V
Tsequential_379/bidirectional_379/forward_lstm_379/TensorArrayV2Stack/TensorListStackе
Gsequential_379/bidirectional_379/forward_lstm_379/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2I
Gsequential_379/bidirectional_379/forward_lstm_379/strided_slice_3/stackа
Isequential_379/bidirectional_379/forward_lstm_379/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2K
Isequential_379/bidirectional_379/forward_lstm_379/strided_slice_3/stack_1а
Isequential_379/bidirectional_379/forward_lstm_379/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_379/bidirectional_379/forward_lstm_379/strided_slice_3/stack_2∆
Asequential_379/bidirectional_379/forward_lstm_379/strided_slice_3StridedSlice]sequential_379/bidirectional_379/forward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0Psequential_379/bidirectional_379/forward_lstm_379/strided_slice_3/stack:output:0Rsequential_379/bidirectional_379/forward_lstm_379/strided_slice_3/stack_1:output:0Rsequential_379/bidirectional_379/forward_lstm_379/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2C
Asequential_379/bidirectional_379/forward_lstm_379/strided_slice_3Ё
Bsequential_379/bidirectional_379/forward_lstm_379/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2D
Bsequential_379/bidirectional_379/forward_lstm_379/transpose_1/permц
=sequential_379/bidirectional_379/forward_lstm_379/transpose_1	Transpose]sequential_379/bidirectional_379/forward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0Ksequential_379/bidirectional_379/forward_lstm_379/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22?
=sequential_379/bidirectional_379/forward_lstm_379/transpose_1 
9sequential_379/bidirectional_379/forward_lstm_379/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2;
9sequential_379/bidirectional_379/forward_lstm_379/runtimeџ
Gsequential_379/bidirectional_379/backward_lstm_379/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2I
Gsequential_379/bidirectional_379/backward_lstm_379/RaggedToTensor/zerosЁ
Gsequential_379/bidirectional_379/backward_lstm_379/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2I
Gsequential_379/bidirectional_379/backward_lstm_379/RaggedToTensor/Const°
Vsequential_379/bidirectional_379/backward_lstm_379/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorPsequential_379/bidirectional_379/backward_lstm_379/RaggedToTensor/Const:output:0args_0Psequential_379/bidirectional_379/backward_lstm_379/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2X
Vsequential_379/bidirectional_379/backward_lstm_379/RaggedToTensor/RaggedTensorToTensorИ
]sequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2_
]sequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice/stackМ
_sequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2a
_sequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1М
_sequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2a
_sequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2”
Wsequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1fsequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack:output:0hsequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1:output:0hsequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask2Y
Wsequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_sliceМ
_sequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2a
_sequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stackЩ
asequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2c
asequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1Р
asequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2c
asequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2я
Ysequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1hsequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack:output:0jsequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0jsequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask2[
Ysequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice_1Щ
Msequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/subSub`sequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice:output:0bsequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2O
Msequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/subК
7sequential_379/bidirectional_379/backward_lstm_379/CastCastQsequential_379/bidirectional_379/backward_lstm_379/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€29
7sequential_379/bidirectional_379/backward_lstm_379/CastГ
8sequential_379/bidirectional_379/backward_lstm_379/ShapeShape_sequential_379/bidirectional_379/backward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2:
8sequential_379/bidirectional_379/backward_lstm_379/ShapeЏ
Fsequential_379/bidirectional_379/backward_lstm_379/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fsequential_379/bidirectional_379/backward_lstm_379/strided_slice/stackё
Hsequential_379/bidirectional_379/backward_lstm_379/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_379/bidirectional_379/backward_lstm_379/strided_slice/stack_1ё
Hsequential_379/bidirectional_379/backward_lstm_379/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_379/bidirectional_379/backward_lstm_379/strided_slice/stack_2Ф
@sequential_379/bidirectional_379/backward_lstm_379/strided_sliceStridedSliceAsequential_379/bidirectional_379/backward_lstm_379/Shape:output:0Osequential_379/bidirectional_379/backward_lstm_379/strided_slice/stack:output:0Qsequential_379/bidirectional_379/backward_lstm_379/strided_slice/stack_1:output:0Qsequential_379/bidirectional_379/backward_lstm_379/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2B
@sequential_379/bidirectional_379/backward_lstm_379/strided_slice¬
>sequential_379/bidirectional_379/backward_lstm_379/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22@
>sequential_379/bidirectional_379/backward_lstm_379/zeros/mul/yЄ
<sequential_379/bidirectional_379/backward_lstm_379/zeros/mulMulIsequential_379/bidirectional_379/backward_lstm_379/strided_slice:output:0Gsequential_379/bidirectional_379/backward_lstm_379/zeros/mul/y:output:0*
T0*
_output_shapes
: 2>
<sequential_379/bidirectional_379/backward_lstm_379/zeros/mul≈
?sequential_379/bidirectional_379/backward_lstm_379/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2A
?sequential_379/bidirectional_379/backward_lstm_379/zeros/Less/y≥
=sequential_379/bidirectional_379/backward_lstm_379/zeros/LessLess@sequential_379/bidirectional_379/backward_lstm_379/zeros/mul:z:0Hsequential_379/bidirectional_379/backward_lstm_379/zeros/Less/y:output:0*
T0*
_output_shapes
: 2?
=sequential_379/bidirectional_379/backward_lstm_379/zeros/Less»
Asequential_379/bidirectional_379/backward_lstm_379/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22C
Asequential_379/bidirectional_379/backward_lstm_379/zeros/packed/1ѕ
?sequential_379/bidirectional_379/backward_lstm_379/zeros/packedPackIsequential_379/bidirectional_379/backward_lstm_379/strided_slice:output:0Jsequential_379/bidirectional_379/backward_lstm_379/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2A
?sequential_379/bidirectional_379/backward_lstm_379/zeros/packed…
>sequential_379/bidirectional_379/backward_lstm_379/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2@
>sequential_379/bidirectional_379/backward_lstm_379/zeros/ConstЅ
8sequential_379/bidirectional_379/backward_lstm_379/zerosFillHsequential_379/bidirectional_379/backward_lstm_379/zeros/packed:output:0Gsequential_379/bidirectional_379/backward_lstm_379/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22:
8sequential_379/bidirectional_379/backward_lstm_379/zeros∆
@sequential_379/bidirectional_379/backward_lstm_379/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22B
@sequential_379/bidirectional_379/backward_lstm_379/zeros_1/mul/yЊ
>sequential_379/bidirectional_379/backward_lstm_379/zeros_1/mulMulIsequential_379/bidirectional_379/backward_lstm_379/strided_slice:output:0Isequential_379/bidirectional_379/backward_lstm_379/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2@
>sequential_379/bidirectional_379/backward_lstm_379/zeros_1/mul…
Asequential_379/bidirectional_379/backward_lstm_379/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2C
Asequential_379/bidirectional_379/backward_lstm_379/zeros_1/Less/yї
?sequential_379/bidirectional_379/backward_lstm_379/zeros_1/LessLessBsequential_379/bidirectional_379/backward_lstm_379/zeros_1/mul:z:0Jsequential_379/bidirectional_379/backward_lstm_379/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2A
?sequential_379/bidirectional_379/backward_lstm_379/zeros_1/Lessћ
Csequential_379/bidirectional_379/backward_lstm_379/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22E
Csequential_379/bidirectional_379/backward_lstm_379/zeros_1/packed/1’
Asequential_379/bidirectional_379/backward_lstm_379/zeros_1/packedPackIsequential_379/bidirectional_379/backward_lstm_379/strided_slice:output:0Lsequential_379/bidirectional_379/backward_lstm_379/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2C
Asequential_379/bidirectional_379/backward_lstm_379/zeros_1/packedЌ
@sequential_379/bidirectional_379/backward_lstm_379/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2B
@sequential_379/bidirectional_379/backward_lstm_379/zeros_1/Const…
:sequential_379/bidirectional_379/backward_lstm_379/zeros_1FillJsequential_379/bidirectional_379/backward_lstm_379/zeros_1/packed:output:0Isequential_379/bidirectional_379/backward_lstm_379/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22<
:sequential_379/bidirectional_379/backward_lstm_379/zeros_1џ
Asequential_379/bidirectional_379/backward_lstm_379/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2C
Asequential_379/bidirectional_379/backward_lstm_379/transpose/permх
<sequential_379/bidirectional_379/backward_lstm_379/transpose	Transpose_sequential_379/bidirectional_379/backward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0Jsequential_379/bidirectional_379/backward_lstm_379/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2>
<sequential_379/bidirectional_379/backward_lstm_379/transposeи
:sequential_379/bidirectional_379/backward_lstm_379/Shape_1Shape@sequential_379/bidirectional_379/backward_lstm_379/transpose:y:0*
T0*
_output_shapes
:2<
:sequential_379/bidirectional_379/backward_lstm_379/Shape_1ё
Hsequential_379/bidirectional_379/backward_lstm_379/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_379/bidirectional_379/backward_lstm_379/strided_slice_1/stackв
Jsequential_379/bidirectional_379/backward_lstm_379/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_379/bidirectional_379/backward_lstm_379/strided_slice_1/stack_1в
Jsequential_379/bidirectional_379/backward_lstm_379/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_379/bidirectional_379/backward_lstm_379/strided_slice_1/stack_2†
Bsequential_379/bidirectional_379/backward_lstm_379/strided_slice_1StridedSliceCsequential_379/bidirectional_379/backward_lstm_379/Shape_1:output:0Qsequential_379/bidirectional_379/backward_lstm_379/strided_slice_1/stack:output:0Ssequential_379/bidirectional_379/backward_lstm_379/strided_slice_1/stack_1:output:0Ssequential_379/bidirectional_379/backward_lstm_379/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2D
Bsequential_379/bidirectional_379/backward_lstm_379/strided_slice_1л
Nsequential_379/bidirectional_379/backward_lstm_379/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2P
Nsequential_379/bidirectional_379/backward_lstm_379/TensorArrayV2/element_shapeю
@sequential_379/bidirectional_379/backward_lstm_379/TensorArrayV2TensorListReserveWsequential_379/bidirectional_379/backward_lstm_379/TensorArrayV2/element_shape:output:0Ksequential_379/bidirectional_379/backward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02B
@sequential_379/bidirectional_379/backward_lstm_379/TensorArrayV2–
Asequential_379/bidirectional_379/backward_lstm_379/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential_379/bidirectional_379/backward_lstm_379/ReverseV2/axis÷
<sequential_379/bidirectional_379/backward_lstm_379/ReverseV2	ReverseV2@sequential_379/bidirectional_379/backward_lstm_379/transpose:y:0Jsequential_379/bidirectional_379/backward_lstm_379/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2>
<sequential_379/bidirectional_379/backward_lstm_379/ReverseV2•
hsequential_379/bidirectional_379/backward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2j
hsequential_379/bidirectional_379/backward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape…
Zsequential_379/bidirectional_379/backward_lstm_379/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorEsequential_379/bidirectional_379/backward_lstm_379/ReverseV2:output:0qsequential_379/bidirectional_379/backward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02\
Zsequential_379/bidirectional_379/backward_lstm_379/TensorArrayUnstack/TensorListFromTensorё
Hsequential_379/bidirectional_379/backward_lstm_379/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_379/bidirectional_379/backward_lstm_379/strided_slice_2/stackв
Jsequential_379/bidirectional_379/backward_lstm_379/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_379/bidirectional_379/backward_lstm_379/strided_slice_2/stack_1в
Jsequential_379/bidirectional_379/backward_lstm_379/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_379/bidirectional_379/backward_lstm_379/strided_slice_2/stack_2Ѓ
Bsequential_379/bidirectional_379/backward_lstm_379/strided_slice_2StridedSlice@sequential_379/bidirectional_379/backward_lstm_379/transpose:y:0Qsequential_379/bidirectional_379/backward_lstm_379/strided_slice_2/stack:output:0Ssequential_379/bidirectional_379/backward_lstm_379/strided_slice_2/stack_1:output:0Ssequential_379/bidirectional_379/backward_lstm_379/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2D
Bsequential_379/bidirectional_379/backward_lstm_379/strided_slice_2‘
Wsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOp`sequential_379_bidirectional_379_backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02Y
Wsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp€
Hsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMulMatMulKsequential_379/bidirectional_379/backward_lstm_379/strided_slice_2:output:0_sequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2J
Hsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMulЏ
Ysequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOpbsequential_379_bidirectional_379_backward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02[
Ysequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpы
Jsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMul_1MatMulAsequential_379/bidirectional_379/backward_lstm_379/zeros:output:0asequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2L
Jsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMul_1ф
Esequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/addAddV2Rsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMul:product:0Tsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2G
Esequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/add”
Xsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOpasequential_379_bidirectional_379_backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02Z
Xsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpБ
Isequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/BiasAddBiasAddIsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/add:z:0`sequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2K
Isequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/BiasAddи
Qsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2S
Qsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/split/split_dim«
Gsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/splitSplitZsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/split/split_dim:output:0Rsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2I
Gsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/split•
Isequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/SigmoidSigmoidPsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22K
Isequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/Sigmoid©
Ksequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/Sigmoid_1SigmoidPsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22M
Ksequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/Sigmoid_1Ё
Esequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/mulMulOsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/Sigmoid_1:y:0Csequential_379/bidirectional_379/backward_lstm_379/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22G
Esequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/mulЬ
Fsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/ReluReluPsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22H
Fsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/Reluр
Gsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/mul_1MulMsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/Sigmoid:y:0Tsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22I
Gsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/mul_1е
Gsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/add_1AddV2Isequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/mul:z:0Ksequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22I
Gsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/add_1©
Ksequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/Sigmoid_2SigmoidPsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22M
Ksequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/Sigmoid_2Ы
Hsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/Relu_1ReluKsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22J
Hsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/Relu_1ф
Gsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/mul_2MulOsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/Sigmoid_2:y:0Vsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22I
Gsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/mul_2х
Psequential_379/bidirectional_379/backward_lstm_379/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2R
Psequential_379/bidirectional_379/backward_lstm_379/TensorArrayV2_1/element_shapeД
Bsequential_379/bidirectional_379/backward_lstm_379/TensorArrayV2_1TensorListReserveYsequential_379/bidirectional_379/backward_lstm_379/TensorArrayV2_1/element_shape:output:0Ksequential_379/bidirectional_379/backward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02D
Bsequential_379/bidirectional_379/backward_lstm_379/TensorArrayV2_1і
7sequential_379/bidirectional_379/backward_lstm_379/timeConst*
_output_shapes
: *
dtype0*
value	B : 29
7sequential_379/bidirectional_379/backward_lstm_379/time÷
Hsequential_379/bidirectional_379/backward_lstm_379/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hsequential_379/bidirectional_379/backward_lstm_379/Max/reduction_indices®
6sequential_379/bidirectional_379/backward_lstm_379/MaxMax;sequential_379/bidirectional_379/backward_lstm_379/Cast:y:0Qsequential_379/bidirectional_379/backward_lstm_379/Max/reduction_indices:output:0*
T0*
_output_shapes
: 28
6sequential_379/bidirectional_379/backward_lstm_379/Maxґ
8sequential_379/bidirectional_379/backward_lstm_379/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_379/bidirectional_379/backward_lstm_379/sub/yЬ
6sequential_379/bidirectional_379/backward_lstm_379/subSub?sequential_379/bidirectional_379/backward_lstm_379/Max:output:0Asequential_379/bidirectional_379/backward_lstm_379/sub/y:output:0*
T0*
_output_shapes
: 28
6sequential_379/bidirectional_379/backward_lstm_379/subҐ
8sequential_379/bidirectional_379/backward_lstm_379/Sub_1Sub:sequential_379/bidirectional_379/backward_lstm_379/sub:z:0;sequential_379/bidirectional_379/backward_lstm_379/Cast:y:0*
T0*#
_output_shapes
:€€€€€€€€€2:
8sequential_379/bidirectional_379/backward_lstm_379/Sub_1К
=sequential_379/bidirectional_379/backward_lstm_379/zeros_like	ZerosLikeKsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22?
=sequential_379/bidirectional_379/backward_lstm_379/zeros_likeе
Ksequential_379/bidirectional_379/backward_lstm_379/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2M
Ksequential_379/bidirectional_379/backward_lstm_379/while/maximum_iterations–
Esequential_379/bidirectional_379/backward_lstm_379/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2G
Esequential_379/bidirectional_379/backward_lstm_379/while/loop_counterў
8sequential_379/bidirectional_379/backward_lstm_379/whileWhileNsequential_379/bidirectional_379/backward_lstm_379/while/loop_counter:output:0Tsequential_379/bidirectional_379/backward_lstm_379/while/maximum_iterations:output:0@sequential_379/bidirectional_379/backward_lstm_379/time:output:0Ksequential_379/bidirectional_379/backward_lstm_379/TensorArrayV2_1:handle:0Asequential_379/bidirectional_379/backward_lstm_379/zeros_like:y:0Asequential_379/bidirectional_379/backward_lstm_379/zeros:output:0Csequential_379/bidirectional_379/backward_lstm_379/zeros_1:output:0Ksequential_379/bidirectional_379/backward_lstm_379/strided_slice_1:output:0jsequential_379/bidirectional_379/backward_lstm_379/TensorArrayUnstack/TensorListFromTensor:output_handle:0<sequential_379/bidirectional_379/backward_lstm_379/Sub_1:z:0`sequential_379_bidirectional_379_backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resourcebsequential_379_bidirectional_379_backward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resourceasequential_379_bidirectional_379_backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *R
bodyJRH
Fsequential_379_bidirectional_379_backward_lstm_379_while_body_42657734*R
condJRH
Fsequential_379_bidirectional_379_backward_lstm_379_while_cond_42657733*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2:
8sequential_379/bidirectional_379/backward_lstm_379/whileЫ
csequential_379/bidirectional_379/backward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2e
csequential_379/bidirectional_379/backward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeљ
Usequential_379/bidirectional_379/backward_lstm_379/TensorArrayV2Stack/TensorListStackTensorListStackAsequential_379/bidirectional_379/backward_lstm_379/while:output:3lsequential_379/bidirectional_379/backward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02W
Usequential_379/bidirectional_379/backward_lstm_379/TensorArrayV2Stack/TensorListStackз
Hsequential_379/bidirectional_379/backward_lstm_379/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2J
Hsequential_379/bidirectional_379/backward_lstm_379/strided_slice_3/stackв
Jsequential_379/bidirectional_379/backward_lstm_379/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jsequential_379/bidirectional_379/backward_lstm_379/strided_slice_3/stack_1в
Jsequential_379/bidirectional_379/backward_lstm_379/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_379/bidirectional_379/backward_lstm_379/strided_slice_3/stack_2ћ
Bsequential_379/bidirectional_379/backward_lstm_379/strided_slice_3StridedSlice^sequential_379/bidirectional_379/backward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0Qsequential_379/bidirectional_379/backward_lstm_379/strided_slice_3/stack:output:0Ssequential_379/bidirectional_379/backward_lstm_379/strided_slice_3/stack_1:output:0Ssequential_379/bidirectional_379/backward_lstm_379/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2D
Bsequential_379/bidirectional_379/backward_lstm_379/strided_slice_3я
Csequential_379/bidirectional_379/backward_lstm_379/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2E
Csequential_379/bidirectional_379/backward_lstm_379/transpose_1/permъ
>sequential_379/bidirectional_379/backward_lstm_379/transpose_1	Transpose^sequential_379/bidirectional_379/backward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0Lsequential_379/bidirectional_379/backward_lstm_379/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22@
>sequential_379/bidirectional_379/backward_lstm_379/transpose_1ћ
:sequential_379/bidirectional_379/backward_lstm_379/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2<
:sequential_379/bidirectional_379/backward_lstm_379/runtimeЮ
,sequential_379/bidirectional_379/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_379/bidirectional_379/concat/axisй
'sequential_379/bidirectional_379/concatConcatV2Jsequential_379/bidirectional_379/forward_lstm_379/strided_slice_3:output:0Ksequential_379/bidirectional_379/backward_lstm_379/strided_slice_3:output:05sequential_379/bidirectional_379/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€d2)
'sequential_379/bidirectional_379/concatЎ
.sequential_379/dense_379/MatMul/ReadVariableOpReadVariableOp7sequential_379_dense_379_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_379/dense_379/MatMul/ReadVariableOpи
sequential_379/dense_379/MatMulMatMul0sequential_379/bidirectional_379/concat:output:06sequential_379/dense_379/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2!
sequential_379/dense_379/MatMul„
/sequential_379/dense_379/BiasAdd/ReadVariableOpReadVariableOp8sequential_379_dense_379_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_379/dense_379/BiasAdd/ReadVariableOpе
 sequential_379/dense_379/BiasAddBiasAdd)sequential_379/dense_379/MatMul:product:07sequential_379/dense_379/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2"
 sequential_379/dense_379/BiasAddђ
 sequential_379/dense_379/SigmoidSigmoid)sequential_379/dense_379/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2"
 sequential_379/dense_379/Sigmoid
IdentityIdentity$sequential_379/dense_379/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity≈
NoOpNoOpY^sequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpX^sequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpZ^sequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp9^sequential_379/bidirectional_379/backward_lstm_379/whileX^sequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpW^sequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpY^sequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp8^sequential_379/bidirectional_379/forward_lstm_379/while0^sequential_379/dense_379/BiasAdd/ReadVariableOp/^sequential_379/dense_379/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€:€€€€€€€€€: : : : : : : : 2і
Xsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpXsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp2≤
Wsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpWsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp2ґ
Ysequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpYsequential_379/bidirectional_379/backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp2t
8sequential_379/bidirectional_379/backward_lstm_379/while8sequential_379/bidirectional_379/backward_lstm_379/while2≤
Wsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpWsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp2∞
Vsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpVsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp2і
Xsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpXsequential_379/bidirectional_379/forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp2r
7sequential_379/bidirectional_379/forward_lstm_379/while7sequential_379/bidirectional_379/forward_lstm_379/while2b
/sequential_379/dense_379/BiasAdd/ReadVariableOp/sequential_379/dense_379/BiasAdd/ReadVariableOp2`
.sequential_379/dense_379/MatMul/ReadVariableOp.sequential_379/dense_379/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameargs_0:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameargs_0
∞Њ
ы
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_42660714

inputs
inputs_1	Q
>forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource:	»S
@forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource:	2»N
?forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource:	»R
?backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resource:	»T
Abackward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource:	2»O
@backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource:	»
identityИҐ7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpҐ6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpҐ8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpҐbackward_lstm_379/whileҐ6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpҐ5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpҐ7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpҐforward_lstm_379/whileЧ
%forward_lstm_379/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_379/RaggedToTensor/zerosЩ
%forward_lstm_379/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2'
%forward_lstm_379/RaggedToTensor/ConstЩ
4forward_lstm_379/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_379/RaggedToTensor/Const:output:0inputs.forward_lstm_379/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_379/RaggedToTensor/RaggedTensorToTensorƒ
;forward_lstm_379/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack»
=forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1»
=forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2©
5forward_lstm_379/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_379/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask27
5forward_lstm_379/RaggedNestedRowLengths/strided_slice»
=forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack’
?forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2A
?forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1ћ
?forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2µ
7forward_lstm_379/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask29
7forward_lstm_379/RaggedNestedRowLengths/strided_slice_1С
+forward_lstm_379/RaggedNestedRowLengths/subSub>forward_lstm_379/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_379/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2-
+forward_lstm_379/RaggedNestedRowLengths/sub§
forward_lstm_379/CastCast/forward_lstm_379/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
forward_lstm_379/CastЭ
forward_lstm_379/ShapeShape=forward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_379/ShapeЦ
$forward_lstm_379/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_379/strided_slice/stackЪ
&forward_lstm_379/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_379/strided_slice/stack_1Ъ
&forward_lstm_379/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_379/strided_slice/stack_2»
forward_lstm_379/strided_sliceStridedSliceforward_lstm_379/Shape:output:0-forward_lstm_379/strided_slice/stack:output:0/forward_lstm_379/strided_slice/stack_1:output:0/forward_lstm_379/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_379/strided_slice~
forward_lstm_379/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_379/zeros/mul/y∞
forward_lstm_379/zeros/mulMul'forward_lstm_379/strided_slice:output:0%forward_lstm_379/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros/mulБ
forward_lstm_379/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
forward_lstm_379/zeros/Less/yЂ
forward_lstm_379/zeros/LessLessforward_lstm_379/zeros/mul:z:0&forward_lstm_379/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros/LessД
forward_lstm_379/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_379/zeros/packed/1«
forward_lstm_379/zeros/packedPack'forward_lstm_379/strided_slice:output:0(forward_lstm_379/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_379/zeros/packedЕ
forward_lstm_379/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_379/zeros/Constє
forward_lstm_379/zerosFill&forward_lstm_379/zeros/packed:output:0%forward_lstm_379/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/zerosВ
forward_lstm_379/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_379/zeros_1/mul/yґ
forward_lstm_379/zeros_1/mulMul'forward_lstm_379/strided_slice:output:0'forward_lstm_379/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros_1/mulЕ
forward_lstm_379/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2!
forward_lstm_379/zeros_1/Less/y≥
forward_lstm_379/zeros_1/LessLess forward_lstm_379/zeros_1/mul:z:0(forward_lstm_379/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros_1/LessИ
!forward_lstm_379/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_379/zeros_1/packed/1Ќ
forward_lstm_379/zeros_1/packedPack'forward_lstm_379/strided_slice:output:0*forward_lstm_379/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_379/zeros_1/packedЙ
forward_lstm_379/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_379/zeros_1/ConstЅ
forward_lstm_379/zeros_1Fill(forward_lstm_379/zeros_1/packed:output:0'forward_lstm_379/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/zeros_1Ч
forward_lstm_379/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_379/transpose/permн
forward_lstm_379/transpose	Transpose=forward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_379/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
forward_lstm_379/transposeВ
forward_lstm_379/Shape_1Shapeforward_lstm_379/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_379/Shape_1Ъ
&forward_lstm_379/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_379/strided_slice_1/stackЮ
(forward_lstm_379/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_1/stack_1Ю
(forward_lstm_379/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_1/stack_2‘
 forward_lstm_379/strided_slice_1StridedSlice!forward_lstm_379/Shape_1:output:0/forward_lstm_379/strided_slice_1/stack:output:01forward_lstm_379/strided_slice_1/stack_1:output:01forward_lstm_379/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_379/strided_slice_1І
,forward_lstm_379/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,forward_lstm_379/TensorArrayV2/element_shapeц
forward_lstm_379/TensorArrayV2TensorListReserve5forward_lstm_379/TensorArrayV2/element_shape:output:0)forward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_379/TensorArrayV2б
Fforward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2H
Fforward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeЉ
8forward_lstm_379/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_379/transpose:y:0Oforward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_379/TensorArrayUnstack/TensorListFromTensorЪ
&forward_lstm_379/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_379/strided_slice_2/stackЮ
(forward_lstm_379/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_2/stack_1Ю
(forward_lstm_379/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_2/stack_2в
 forward_lstm_379/strided_slice_2StridedSliceforward_lstm_379/transpose:y:0/forward_lstm_379/strided_slice_2/stack:output:01forward_lstm_379/strided_slice_2/stack_1:output:01forward_lstm_379/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2"
 forward_lstm_379/strided_slice_2о
5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOp>forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype027
5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpч
&forward_lstm_379/lstm_cell_1138/MatMulMatMul)forward_lstm_379/strided_slice_2:output:0=forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2(
&forward_lstm_379/lstm_cell_1138/MatMulф
7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOp@forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype029
7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpу
(forward_lstm_379/lstm_cell_1138/MatMul_1MatMulforward_lstm_379/zeros:output:0?forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(forward_lstm_379/lstm_cell_1138/MatMul_1м
#forward_lstm_379/lstm_cell_1138/addAddV20forward_lstm_379/lstm_cell_1138/MatMul:product:02forward_lstm_379/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2%
#forward_lstm_379/lstm_cell_1138/addн
6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOp?forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype028
6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpщ
'forward_lstm_379/lstm_cell_1138/BiasAddBiasAdd'forward_lstm_379/lstm_cell_1138/add:z:0>forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'forward_lstm_379/lstm_cell_1138/BiasAdd§
/forward_lstm_379/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/forward_lstm_379/lstm_cell_1138/split/split_dimњ
%forward_lstm_379/lstm_cell_1138/splitSplit8forward_lstm_379/lstm_cell_1138/split/split_dim:output:00forward_lstm_379/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2'
%forward_lstm_379/lstm_cell_1138/splitњ
'forward_lstm_379/lstm_cell_1138/SigmoidSigmoid.forward_lstm_379/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'forward_lstm_379/lstm_cell_1138/Sigmoid√
)forward_lstm_379/lstm_cell_1138/Sigmoid_1Sigmoid.forward_lstm_379/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_379/lstm_cell_1138/Sigmoid_1’
#forward_lstm_379/lstm_cell_1138/mulMul-forward_lstm_379/lstm_cell_1138/Sigmoid_1:y:0!forward_lstm_379/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22%
#forward_lstm_379/lstm_cell_1138/mulґ
$forward_lstm_379/lstm_cell_1138/ReluRelu.forward_lstm_379/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22&
$forward_lstm_379/lstm_cell_1138/Reluи
%forward_lstm_379/lstm_cell_1138/mul_1Mul+forward_lstm_379/lstm_cell_1138/Sigmoid:y:02forward_lstm_379/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_379/lstm_cell_1138/mul_1Ё
%forward_lstm_379/lstm_cell_1138/add_1AddV2'forward_lstm_379/lstm_cell_1138/mul:z:0)forward_lstm_379/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_379/lstm_cell_1138/add_1√
)forward_lstm_379/lstm_cell_1138/Sigmoid_2Sigmoid.forward_lstm_379/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_379/lstm_cell_1138/Sigmoid_2µ
&forward_lstm_379/lstm_cell_1138/Relu_1Relu)forward_lstm_379/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&forward_lstm_379/lstm_cell_1138/Relu_1м
%forward_lstm_379/lstm_cell_1138/mul_2Mul-forward_lstm_379/lstm_cell_1138/Sigmoid_2:y:04forward_lstm_379/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_379/lstm_cell_1138/mul_2±
.forward_lstm_379/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   20
.forward_lstm_379/TensorArrayV2_1/element_shapeь
 forward_lstm_379/TensorArrayV2_1TensorListReserve7forward_lstm_379/TensorArrayV2_1/element_shape:output:0)forward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_379/TensorArrayV2_1p
forward_lstm_379/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_379/time§
forward_lstm_379/zeros_like	ZerosLike)forward_lstm_379/lstm_cell_1138/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/zeros_like°
)forward_lstm_379/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2+
)forward_lstm_379/while/maximum_iterationsМ
#forward_lstm_379/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_379/while/loop_counterЦ	
forward_lstm_379/whileWhile,forward_lstm_379/while/loop_counter:output:02forward_lstm_379/while/maximum_iterations:output:0forward_lstm_379/time:output:0)forward_lstm_379/TensorArrayV2_1:handle:0forward_lstm_379/zeros_like:y:0forward_lstm_379/zeros:output:0!forward_lstm_379/zeros_1:output:0)forward_lstm_379/strided_slice_1:output:0Hforward_lstm_379/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_379/Cast:y:0>forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource@forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource?forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *0
body(R&
$forward_lstm_379_while_body_42660438*0
cond(R&
$forward_lstm_379_while_cond_42660437*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
forward_lstm_379/while„
Aforward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2C
Aforward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeµ
3forward_lstm_379/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_379/while:output:3Jforward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype025
3forward_lstm_379/TensorArrayV2Stack/TensorListStack£
&forward_lstm_379/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2(
&forward_lstm_379/strided_slice_3/stackЮ
(forward_lstm_379/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_379/strided_slice_3/stack_1Ю
(forward_lstm_379/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_3/stack_2А
 forward_lstm_379/strided_slice_3StridedSlice<forward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_379/strided_slice_3/stack:output:01forward_lstm_379/strided_slice_3/stack_1:output:01forward_lstm_379/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2"
 forward_lstm_379/strided_slice_3Ы
!forward_lstm_379/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_379/transpose_1/permт
forward_lstm_379/transpose_1	Transpose<forward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_379/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
forward_lstm_379/transpose_1И
forward_lstm_379/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_379/runtimeЩ
&backward_lstm_379/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_379/RaggedToTensor/zerosЫ
&backward_lstm_379/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2(
&backward_lstm_379/RaggedToTensor/ConstЭ
5backward_lstm_379/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_379/RaggedToTensor/Const:output:0inputs/backward_lstm_379/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_379/RaggedToTensor/RaggedTensorToTensor∆
<backward_lstm_379/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack 
>backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1 
>backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2Ѓ
6backward_lstm_379/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_379/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask28
6backward_lstm_379/RaggedNestedRowLengths/strided_slice 
>backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack„
@backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2B
@backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1ќ
@backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2Ї
8backward_lstm_379/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask2:
8backward_lstm_379/RaggedNestedRowLengths/strided_slice_1Х
,backward_lstm_379/RaggedNestedRowLengths/subSub?backward_lstm_379/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_379/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2.
,backward_lstm_379/RaggedNestedRowLengths/subІ
backward_lstm_379/CastCast0backward_lstm_379/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
backward_lstm_379/Cast†
backward_lstm_379/ShapeShape>backward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_379/ShapeШ
%backward_lstm_379/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_379/strided_slice/stackЬ
'backward_lstm_379/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_379/strided_slice/stack_1Ь
'backward_lstm_379/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_379/strided_slice/stack_2ќ
backward_lstm_379/strided_sliceStridedSlice backward_lstm_379/Shape:output:0.backward_lstm_379/strided_slice/stack:output:00backward_lstm_379/strided_slice/stack_1:output:00backward_lstm_379/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_379/strided_sliceА
backward_lstm_379/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_379/zeros/mul/yі
backward_lstm_379/zeros/mulMul(backward_lstm_379/strided_slice:output:0&backward_lstm_379/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/zeros/mulГ
backward_lstm_379/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
backward_lstm_379/zeros/Less/yѓ
backward_lstm_379/zeros/LessLessbackward_lstm_379/zeros/mul:z:0'backward_lstm_379/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/zeros/LessЖ
 backward_lstm_379/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_379/zeros/packed/1Ћ
backward_lstm_379/zeros/packedPack(backward_lstm_379/strided_slice:output:0)backward_lstm_379/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_379/zeros/packedЗ
backward_lstm_379/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_379/zeros/Constљ
backward_lstm_379/zerosFill'backward_lstm_379/zeros/packed:output:0&backward_lstm_379/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_379/zerosД
backward_lstm_379/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_379/zeros_1/mul/yЇ
backward_lstm_379/zeros_1/mulMul(backward_lstm_379/strided_slice:output:0(backward_lstm_379/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/zeros_1/mulЗ
 backward_lstm_379/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 backward_lstm_379/zeros_1/Less/yЈ
backward_lstm_379/zeros_1/LessLess!backward_lstm_379/zeros_1/mul:z:0)backward_lstm_379/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_379/zeros_1/LessК
"backward_lstm_379/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_379/zeros_1/packed/1—
 backward_lstm_379/zeros_1/packedPack(backward_lstm_379/strided_slice:output:0+backward_lstm_379/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_379/zeros_1/packedЛ
backward_lstm_379/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_379/zeros_1/Const≈
backward_lstm_379/zeros_1Fill)backward_lstm_379/zeros_1/packed:output:0(backward_lstm_379/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_379/zeros_1Щ
 backward_lstm_379/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_379/transpose/permс
backward_lstm_379/transpose	Transpose>backward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_379/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_379/transposeЕ
backward_lstm_379/Shape_1Shapebackward_lstm_379/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_379/Shape_1Ь
'backward_lstm_379/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_379/strided_slice_1/stack†
)backward_lstm_379/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_1/stack_1†
)backward_lstm_379/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_1/stack_2Џ
!backward_lstm_379/strided_slice_1StridedSlice"backward_lstm_379/Shape_1:output:00backward_lstm_379/strided_slice_1/stack:output:02backward_lstm_379/strided_slice_1/stack_1:output:02backward_lstm_379/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_379/strided_slice_1©
-backward_lstm_379/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-backward_lstm_379/TensorArrayV2/element_shapeъ
backward_lstm_379/TensorArrayV2TensorListReserve6backward_lstm_379/TensorArrayV2/element_shape:output:0*backward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_379/TensorArrayV2О
 backward_lstm_379/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_379/ReverseV2/axis“
backward_lstm_379/ReverseV2	ReverseV2backward_lstm_379/transpose:y:0)backward_lstm_379/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_379/ReverseV2г
Gbackward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2I
Gbackward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape≈
9backward_lstm_379/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_379/ReverseV2:output:0Pbackward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_379/TensorArrayUnstack/TensorListFromTensorЬ
'backward_lstm_379/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_379/strided_slice_2/stack†
)backward_lstm_379/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_2/stack_1†
)backward_lstm_379/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_2/stack_2и
!backward_lstm_379/strided_slice_2StridedSlicebackward_lstm_379/transpose:y:00backward_lstm_379/strided_slice_2/stack:output:02backward_lstm_379/strided_slice_2/stack_1:output:02backward_lstm_379/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2#
!backward_lstm_379/strided_slice_2с
6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOp?backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype028
6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpы
'backward_lstm_379/lstm_cell_1139/MatMulMatMul*backward_lstm_379/strided_slice_2:output:0>backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'backward_lstm_379/lstm_cell_1139/MatMulч
8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOpAbackward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02:
8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpч
)backward_lstm_379/lstm_cell_1139/MatMul_1MatMul backward_lstm_379/zeros:output:0@backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)backward_lstm_379/lstm_cell_1139/MatMul_1р
$backward_lstm_379/lstm_cell_1139/addAddV21backward_lstm_379/lstm_cell_1139/MatMul:product:03backward_lstm_379/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2&
$backward_lstm_379/lstm_cell_1139/addр
7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOp@backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype029
7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpэ
(backward_lstm_379/lstm_cell_1139/BiasAddBiasAdd(backward_lstm_379/lstm_cell_1139/add:z:0?backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(backward_lstm_379/lstm_cell_1139/BiasAdd¶
0backward_lstm_379/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0backward_lstm_379/lstm_cell_1139/split/split_dim√
&backward_lstm_379/lstm_cell_1139/splitSplit9backward_lstm_379/lstm_cell_1139/split/split_dim:output:01backward_lstm_379/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2(
&backward_lstm_379/lstm_cell_1139/split¬
(backward_lstm_379/lstm_cell_1139/SigmoidSigmoid/backward_lstm_379/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(backward_lstm_379/lstm_cell_1139/Sigmoid∆
*backward_lstm_379/lstm_cell_1139/Sigmoid_1Sigmoid/backward_lstm_379/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_379/lstm_cell_1139/Sigmoid_1ў
$backward_lstm_379/lstm_cell_1139/mulMul.backward_lstm_379/lstm_cell_1139/Sigmoid_1:y:0"backward_lstm_379/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$backward_lstm_379/lstm_cell_1139/mulє
%backward_lstm_379/lstm_cell_1139/ReluRelu/backward_lstm_379/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22'
%backward_lstm_379/lstm_cell_1139/Reluм
&backward_lstm_379/lstm_cell_1139/mul_1Mul,backward_lstm_379/lstm_cell_1139/Sigmoid:y:03backward_lstm_379/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_379/lstm_cell_1139/mul_1б
&backward_lstm_379/lstm_cell_1139/add_1AddV2(backward_lstm_379/lstm_cell_1139/mul:z:0*backward_lstm_379/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_379/lstm_cell_1139/add_1∆
*backward_lstm_379/lstm_cell_1139/Sigmoid_2Sigmoid/backward_lstm_379/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_379/lstm_cell_1139/Sigmoid_2Є
'backward_lstm_379/lstm_cell_1139/Relu_1Relu*backward_lstm_379/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'backward_lstm_379/lstm_cell_1139/Relu_1р
&backward_lstm_379/lstm_cell_1139/mul_2Mul.backward_lstm_379/lstm_cell_1139/Sigmoid_2:y:05backward_lstm_379/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_379/lstm_cell_1139/mul_2≥
/backward_lstm_379/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   21
/backward_lstm_379/TensorArrayV2_1/element_shapeА
!backward_lstm_379/TensorArrayV2_1TensorListReserve8backward_lstm_379/TensorArrayV2_1/element_shape:output:0*backward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_379/TensorArrayV2_1r
backward_lstm_379/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_379/timeФ
'backward_lstm_379/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_379/Max/reduction_indices§
backward_lstm_379/MaxMaxbackward_lstm_379/Cast:y:00backward_lstm_379/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/Maxt
backward_lstm_379/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_379/sub/yШ
backward_lstm_379/subSubbackward_lstm_379/Max:output:0 backward_lstm_379/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/subЮ
backward_lstm_379/Sub_1Subbackward_lstm_379/sub:z:0backward_lstm_379/Cast:y:0*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_379/Sub_1І
backward_lstm_379/zeros_like	ZerosLike*backward_lstm_379/lstm_cell_1139/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_379/zeros_like£
*backward_lstm_379/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2,
*backward_lstm_379/while/maximum_iterationsО
$backward_lstm_379/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_379/while/loop_counter®	
backward_lstm_379/whileWhile-backward_lstm_379/while/loop_counter:output:03backward_lstm_379/while/maximum_iterations:output:0backward_lstm_379/time:output:0*backward_lstm_379/TensorArrayV2_1:handle:0 backward_lstm_379/zeros_like:y:0 backward_lstm_379/zeros:output:0"backward_lstm_379/zeros_1:output:0*backward_lstm_379/strided_slice_1:output:0Ibackward_lstm_379/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_379/Sub_1:z:0?backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resourceAbackward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource@backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *1
body)R'
%backward_lstm_379_while_body_42660617*1
cond)R'
%backward_lstm_379_while_cond_42660616*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
backward_lstm_379/whileў
Bbackward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2D
Bbackward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeє
4backward_lstm_379/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_379/while:output:3Kbackward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype026
4backward_lstm_379/TensorArrayV2Stack/TensorListStack•
'backward_lstm_379/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2)
'backward_lstm_379/strided_slice_3/stack†
)backward_lstm_379/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_379/strided_slice_3/stack_1†
)backward_lstm_379/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_3/stack_2Ж
!backward_lstm_379/strided_slice_3StridedSlice=backward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_379/strided_slice_3/stack:output:02backward_lstm_379/strided_slice_3/stack_1:output:02backward_lstm_379/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2#
!backward_lstm_379/strided_slice_3Э
"backward_lstm_379/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_379/transpose_1/permц
backward_lstm_379/transpose_1	Transpose=backward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_379/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
backward_lstm_379/transpose_1К
backward_lstm_379/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_379/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisƒ
concatConcatV2)forward_lstm_379/strided_slice_3:output:0*backward_lstm_379/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d2

IdentityЏ
NoOpNoOp8^backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp7^backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp9^backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp^backward_lstm_379/while7^forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp6^forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp8^forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp^forward_lstm_379/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€:€€€€€€€€€: : : : : : 2r
7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp2p
6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp2t
8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp22
backward_lstm_379/whilebackward_lstm_379/while2p
6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp2n
5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp2r
7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp20
forward_lstm_379/whileforward_lstm_379/while:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ф_
≥
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_42659615

inputs@
-lstm_cell_1139_matmul_readvariableop_resource:	»B
/lstm_cell_1139_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1139_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1139/BiasAdd/ReadVariableOpҐ$lstm_cell_1139/MatMul/ReadVariableOpҐ&lstm_cell_1139/MatMul_1/ReadVariableOpҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€22
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
B :и2
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
zeros_1/packed/1Й
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
:€€€€€€€€€22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permМ
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
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
ReverseV2/axisУ
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
	ReverseV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€27
5TensorArrayUnstack/TensorListFromTensor/element_shapeэ
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
strided_slice_2/stack_2Е
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2
strided_slice_2ї
$lstm_cell_1139/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1139_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1139/MatMul/ReadVariableOp≥
lstm_cell_1139/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/MatMulЅ
&lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1139_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1139/MatMul_1/ReadVariableOpѓ
lstm_cell_1139/MatMul_1MatMulzeros:output:0.lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/MatMul_1®
lstm_cell_1139/addAddV2lstm_cell_1139/MatMul:product:0!lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/addЇ
%lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1139_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1139/BiasAdd/ReadVariableOpµ
lstm_cell_1139/BiasAddBiasAddlstm_cell_1139/add:z:0-lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/BiasAddВ
lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1139/split/split_dimы
lstm_cell_1139/splitSplit'lstm_cell_1139/split/split_dim:output:0lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1139/splitМ
lstm_cell_1139/SigmoidSigmoidlstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/SigmoidР
lstm_cell_1139/Sigmoid_1Sigmoidlstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Sigmoid_1С
lstm_cell_1139/mulMullstm_cell_1139/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/mulГ
lstm_cell_1139/ReluRelulstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Relu§
lstm_cell_1139/mul_1Mullstm_cell_1139/Sigmoid:y:0!lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/mul_1Щ
lstm_cell_1139/add_1AddV2lstm_cell_1139/mul:z:0lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/add_1Р
lstm_cell_1139/Sigmoid_2Sigmoidlstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Sigmoid_2В
lstm_cell_1139/Relu_1Relulstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Relu_1®
lstm_cell_1139/mul_2Mullstm_cell_1139/Sigmoid_2:y:0#lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterХ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1139_matmul_readvariableop_resource/lstm_cell_1139_matmul_1_readvariableop_resource.lstm_cell_1139_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42659531*
condR
while_cond_42659530*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
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
:€€€€€€€€€22

Identityќ
NoOpNoOp&^lstm_cell_1139/BiasAdd/ReadVariableOp%^lstm_cell_1139/MatMul/ReadVariableOp'^lstm_cell_1139/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1139/BiasAdd/ReadVariableOp%lstm_cell_1139/BiasAdd/ReadVariableOp2L
$lstm_cell_1139/MatMul/ReadVariableOp$lstm_cell_1139/MatMul/ReadVariableOp2P
&lstm_cell_1139/MatMul_1/ReadVariableOp&lstm_cell_1139/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ф_
≥
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_42663455

inputs@
-lstm_cell_1139_matmul_readvariableop_resource:	»B
/lstm_cell_1139_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1139_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1139/BiasAdd/ReadVariableOpҐ$lstm_cell_1139/MatMul/ReadVariableOpҐ&lstm_cell_1139/MatMul_1/ReadVariableOpҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€22
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
B :и2
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
zeros_1/packed/1Й
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
:€€€€€€€€€22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permМ
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
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
ReverseV2/axisУ
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
	ReverseV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€27
5TensorArrayUnstack/TensorListFromTensor/element_shapeэ
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
strided_slice_2/stack_2Е
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2
strided_slice_2ї
$lstm_cell_1139/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1139_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1139/MatMul/ReadVariableOp≥
lstm_cell_1139/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/MatMulЅ
&lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1139_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1139/MatMul_1/ReadVariableOpѓ
lstm_cell_1139/MatMul_1MatMulzeros:output:0.lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/MatMul_1®
lstm_cell_1139/addAddV2lstm_cell_1139/MatMul:product:0!lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/addЇ
%lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1139_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1139/BiasAdd/ReadVariableOpµ
lstm_cell_1139/BiasAddBiasAddlstm_cell_1139/add:z:0-lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/BiasAddВ
lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1139/split/split_dimы
lstm_cell_1139/splitSplit'lstm_cell_1139/split/split_dim:output:0lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1139/splitМ
lstm_cell_1139/SigmoidSigmoidlstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/SigmoidР
lstm_cell_1139/Sigmoid_1Sigmoidlstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Sigmoid_1С
lstm_cell_1139/mulMullstm_cell_1139/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/mulГ
lstm_cell_1139/ReluRelulstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Relu§
lstm_cell_1139/mul_1Mullstm_cell_1139/Sigmoid:y:0!lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/mul_1Щ
lstm_cell_1139/add_1AddV2lstm_cell_1139/mul:z:0lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/add_1Р
lstm_cell_1139/Sigmoid_2Sigmoidlstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Sigmoid_2В
lstm_cell_1139/Relu_1Relulstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Relu_1®
lstm_cell_1139/mul_2Mullstm_cell_1139/Sigmoid_2:y:0#lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterХ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1139_matmul_readvariableop_resource/lstm_cell_1139_matmul_1_readvariableop_resource.lstm_cell_1139_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42663371*
condR
while_cond_42663370*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
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
:€€€€€€€€€22

Identityќ
NoOpNoOp&^lstm_cell_1139/BiasAdd/ReadVariableOp%^lstm_cell_1139/MatMul/ReadVariableOp'^lstm_cell_1139/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1139/BiasAdd/ReadVariableOp%lstm_cell_1139/BiasAdd/ReadVariableOp2L
$lstm_cell_1139/MatMul/ReadVariableOp$lstm_cell_1139/MatMul/ReadVariableOp2P
&lstm_cell_1139/MatMul_1/ReadVariableOp&lstm_cell_1139/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞Њ
ы
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_42662284

inputs
inputs_1	Q
>forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource:	»S
@forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource:	2»N
?forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource:	»R
?backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resource:	»T
Abackward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource:	2»O
@backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource:	»
identityИҐ7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpҐ6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpҐ8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpҐbackward_lstm_379/whileҐ6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpҐ5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpҐ7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpҐforward_lstm_379/whileЧ
%forward_lstm_379/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_379/RaggedToTensor/zerosЩ
%forward_lstm_379/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2'
%forward_lstm_379/RaggedToTensor/ConstЩ
4forward_lstm_379/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_379/RaggedToTensor/Const:output:0inputs.forward_lstm_379/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_379/RaggedToTensor/RaggedTensorToTensorƒ
;forward_lstm_379/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack»
=forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1»
=forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2©
5forward_lstm_379/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_379/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask27
5forward_lstm_379/RaggedNestedRowLengths/strided_slice»
=forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack’
?forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2A
?forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1ћ
?forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2µ
7forward_lstm_379/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask29
7forward_lstm_379/RaggedNestedRowLengths/strided_slice_1С
+forward_lstm_379/RaggedNestedRowLengths/subSub>forward_lstm_379/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_379/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2-
+forward_lstm_379/RaggedNestedRowLengths/sub§
forward_lstm_379/CastCast/forward_lstm_379/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
forward_lstm_379/CastЭ
forward_lstm_379/ShapeShape=forward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_379/ShapeЦ
$forward_lstm_379/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_379/strided_slice/stackЪ
&forward_lstm_379/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_379/strided_slice/stack_1Ъ
&forward_lstm_379/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_379/strided_slice/stack_2»
forward_lstm_379/strided_sliceStridedSliceforward_lstm_379/Shape:output:0-forward_lstm_379/strided_slice/stack:output:0/forward_lstm_379/strided_slice/stack_1:output:0/forward_lstm_379/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_379/strided_slice~
forward_lstm_379/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_379/zeros/mul/y∞
forward_lstm_379/zeros/mulMul'forward_lstm_379/strided_slice:output:0%forward_lstm_379/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros/mulБ
forward_lstm_379/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
forward_lstm_379/zeros/Less/yЂ
forward_lstm_379/zeros/LessLessforward_lstm_379/zeros/mul:z:0&forward_lstm_379/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros/LessД
forward_lstm_379/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_379/zeros/packed/1«
forward_lstm_379/zeros/packedPack'forward_lstm_379/strided_slice:output:0(forward_lstm_379/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_379/zeros/packedЕ
forward_lstm_379/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_379/zeros/Constє
forward_lstm_379/zerosFill&forward_lstm_379/zeros/packed:output:0%forward_lstm_379/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/zerosВ
forward_lstm_379/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_379/zeros_1/mul/yґ
forward_lstm_379/zeros_1/mulMul'forward_lstm_379/strided_slice:output:0'forward_lstm_379/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros_1/mulЕ
forward_lstm_379/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2!
forward_lstm_379/zeros_1/Less/y≥
forward_lstm_379/zeros_1/LessLess forward_lstm_379/zeros_1/mul:z:0(forward_lstm_379/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros_1/LessИ
!forward_lstm_379/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_379/zeros_1/packed/1Ќ
forward_lstm_379/zeros_1/packedPack'forward_lstm_379/strided_slice:output:0*forward_lstm_379/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_379/zeros_1/packedЙ
forward_lstm_379/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_379/zeros_1/ConstЅ
forward_lstm_379/zeros_1Fill(forward_lstm_379/zeros_1/packed:output:0'forward_lstm_379/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/zeros_1Ч
forward_lstm_379/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_379/transpose/permн
forward_lstm_379/transpose	Transpose=forward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_379/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
forward_lstm_379/transposeВ
forward_lstm_379/Shape_1Shapeforward_lstm_379/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_379/Shape_1Ъ
&forward_lstm_379/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_379/strided_slice_1/stackЮ
(forward_lstm_379/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_1/stack_1Ю
(forward_lstm_379/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_1/stack_2‘
 forward_lstm_379/strided_slice_1StridedSlice!forward_lstm_379/Shape_1:output:0/forward_lstm_379/strided_slice_1/stack:output:01forward_lstm_379/strided_slice_1/stack_1:output:01forward_lstm_379/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_379/strided_slice_1І
,forward_lstm_379/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,forward_lstm_379/TensorArrayV2/element_shapeц
forward_lstm_379/TensorArrayV2TensorListReserve5forward_lstm_379/TensorArrayV2/element_shape:output:0)forward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_379/TensorArrayV2б
Fforward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2H
Fforward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeЉ
8forward_lstm_379/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_379/transpose:y:0Oforward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_379/TensorArrayUnstack/TensorListFromTensorЪ
&forward_lstm_379/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_379/strided_slice_2/stackЮ
(forward_lstm_379/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_2/stack_1Ю
(forward_lstm_379/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_2/stack_2в
 forward_lstm_379/strided_slice_2StridedSliceforward_lstm_379/transpose:y:0/forward_lstm_379/strided_slice_2/stack:output:01forward_lstm_379/strided_slice_2/stack_1:output:01forward_lstm_379/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2"
 forward_lstm_379/strided_slice_2о
5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOp>forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype027
5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpч
&forward_lstm_379/lstm_cell_1138/MatMulMatMul)forward_lstm_379/strided_slice_2:output:0=forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2(
&forward_lstm_379/lstm_cell_1138/MatMulф
7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOp@forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype029
7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpу
(forward_lstm_379/lstm_cell_1138/MatMul_1MatMulforward_lstm_379/zeros:output:0?forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(forward_lstm_379/lstm_cell_1138/MatMul_1м
#forward_lstm_379/lstm_cell_1138/addAddV20forward_lstm_379/lstm_cell_1138/MatMul:product:02forward_lstm_379/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2%
#forward_lstm_379/lstm_cell_1138/addн
6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOp?forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype028
6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpщ
'forward_lstm_379/lstm_cell_1138/BiasAddBiasAdd'forward_lstm_379/lstm_cell_1138/add:z:0>forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'forward_lstm_379/lstm_cell_1138/BiasAdd§
/forward_lstm_379/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/forward_lstm_379/lstm_cell_1138/split/split_dimњ
%forward_lstm_379/lstm_cell_1138/splitSplit8forward_lstm_379/lstm_cell_1138/split/split_dim:output:00forward_lstm_379/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2'
%forward_lstm_379/lstm_cell_1138/splitњ
'forward_lstm_379/lstm_cell_1138/SigmoidSigmoid.forward_lstm_379/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'forward_lstm_379/lstm_cell_1138/Sigmoid√
)forward_lstm_379/lstm_cell_1138/Sigmoid_1Sigmoid.forward_lstm_379/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_379/lstm_cell_1138/Sigmoid_1’
#forward_lstm_379/lstm_cell_1138/mulMul-forward_lstm_379/lstm_cell_1138/Sigmoid_1:y:0!forward_lstm_379/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22%
#forward_lstm_379/lstm_cell_1138/mulґ
$forward_lstm_379/lstm_cell_1138/ReluRelu.forward_lstm_379/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22&
$forward_lstm_379/lstm_cell_1138/Reluи
%forward_lstm_379/lstm_cell_1138/mul_1Mul+forward_lstm_379/lstm_cell_1138/Sigmoid:y:02forward_lstm_379/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_379/lstm_cell_1138/mul_1Ё
%forward_lstm_379/lstm_cell_1138/add_1AddV2'forward_lstm_379/lstm_cell_1138/mul:z:0)forward_lstm_379/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_379/lstm_cell_1138/add_1√
)forward_lstm_379/lstm_cell_1138/Sigmoid_2Sigmoid.forward_lstm_379/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_379/lstm_cell_1138/Sigmoid_2µ
&forward_lstm_379/lstm_cell_1138/Relu_1Relu)forward_lstm_379/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&forward_lstm_379/lstm_cell_1138/Relu_1м
%forward_lstm_379/lstm_cell_1138/mul_2Mul-forward_lstm_379/lstm_cell_1138/Sigmoid_2:y:04forward_lstm_379/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_379/lstm_cell_1138/mul_2±
.forward_lstm_379/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   20
.forward_lstm_379/TensorArrayV2_1/element_shapeь
 forward_lstm_379/TensorArrayV2_1TensorListReserve7forward_lstm_379/TensorArrayV2_1/element_shape:output:0)forward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_379/TensorArrayV2_1p
forward_lstm_379/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_379/time§
forward_lstm_379/zeros_like	ZerosLike)forward_lstm_379/lstm_cell_1138/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/zeros_like°
)forward_lstm_379/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2+
)forward_lstm_379/while/maximum_iterationsМ
#forward_lstm_379/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_379/while/loop_counterЦ	
forward_lstm_379/whileWhile,forward_lstm_379/while/loop_counter:output:02forward_lstm_379/while/maximum_iterations:output:0forward_lstm_379/time:output:0)forward_lstm_379/TensorArrayV2_1:handle:0forward_lstm_379/zeros_like:y:0forward_lstm_379/zeros:output:0!forward_lstm_379/zeros_1:output:0)forward_lstm_379/strided_slice_1:output:0Hforward_lstm_379/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_379/Cast:y:0>forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource@forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource?forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *0
body(R&
$forward_lstm_379_while_body_42662008*0
cond(R&
$forward_lstm_379_while_cond_42662007*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
forward_lstm_379/while„
Aforward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2C
Aforward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeµ
3forward_lstm_379/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_379/while:output:3Jforward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype025
3forward_lstm_379/TensorArrayV2Stack/TensorListStack£
&forward_lstm_379/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2(
&forward_lstm_379/strided_slice_3/stackЮ
(forward_lstm_379/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_379/strided_slice_3/stack_1Ю
(forward_lstm_379/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_3/stack_2А
 forward_lstm_379/strided_slice_3StridedSlice<forward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_379/strided_slice_3/stack:output:01forward_lstm_379/strided_slice_3/stack_1:output:01forward_lstm_379/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2"
 forward_lstm_379/strided_slice_3Ы
!forward_lstm_379/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_379/transpose_1/permт
forward_lstm_379/transpose_1	Transpose<forward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_379/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
forward_lstm_379/transpose_1И
forward_lstm_379/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_379/runtimeЩ
&backward_lstm_379/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_379/RaggedToTensor/zerosЫ
&backward_lstm_379/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2(
&backward_lstm_379/RaggedToTensor/ConstЭ
5backward_lstm_379/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_379/RaggedToTensor/Const:output:0inputs/backward_lstm_379/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_379/RaggedToTensor/RaggedTensorToTensor∆
<backward_lstm_379/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack 
>backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1 
>backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2Ѓ
6backward_lstm_379/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_379/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask28
6backward_lstm_379/RaggedNestedRowLengths/strided_slice 
>backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack„
@backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2B
@backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1ќ
@backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2Ї
8backward_lstm_379/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask2:
8backward_lstm_379/RaggedNestedRowLengths/strided_slice_1Х
,backward_lstm_379/RaggedNestedRowLengths/subSub?backward_lstm_379/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_379/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2.
,backward_lstm_379/RaggedNestedRowLengths/subІ
backward_lstm_379/CastCast0backward_lstm_379/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
backward_lstm_379/Cast†
backward_lstm_379/ShapeShape>backward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_379/ShapeШ
%backward_lstm_379/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_379/strided_slice/stackЬ
'backward_lstm_379/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_379/strided_slice/stack_1Ь
'backward_lstm_379/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_379/strided_slice/stack_2ќ
backward_lstm_379/strided_sliceStridedSlice backward_lstm_379/Shape:output:0.backward_lstm_379/strided_slice/stack:output:00backward_lstm_379/strided_slice/stack_1:output:00backward_lstm_379/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_379/strided_sliceА
backward_lstm_379/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_379/zeros/mul/yі
backward_lstm_379/zeros/mulMul(backward_lstm_379/strided_slice:output:0&backward_lstm_379/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/zeros/mulГ
backward_lstm_379/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
backward_lstm_379/zeros/Less/yѓ
backward_lstm_379/zeros/LessLessbackward_lstm_379/zeros/mul:z:0'backward_lstm_379/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/zeros/LessЖ
 backward_lstm_379/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_379/zeros/packed/1Ћ
backward_lstm_379/zeros/packedPack(backward_lstm_379/strided_slice:output:0)backward_lstm_379/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_379/zeros/packedЗ
backward_lstm_379/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_379/zeros/Constљ
backward_lstm_379/zerosFill'backward_lstm_379/zeros/packed:output:0&backward_lstm_379/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_379/zerosД
backward_lstm_379/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_379/zeros_1/mul/yЇ
backward_lstm_379/zeros_1/mulMul(backward_lstm_379/strided_slice:output:0(backward_lstm_379/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/zeros_1/mulЗ
 backward_lstm_379/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 backward_lstm_379/zeros_1/Less/yЈ
backward_lstm_379/zeros_1/LessLess!backward_lstm_379/zeros_1/mul:z:0)backward_lstm_379/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_379/zeros_1/LessК
"backward_lstm_379/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_379/zeros_1/packed/1—
 backward_lstm_379/zeros_1/packedPack(backward_lstm_379/strided_slice:output:0+backward_lstm_379/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_379/zeros_1/packedЛ
backward_lstm_379/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_379/zeros_1/Const≈
backward_lstm_379/zeros_1Fill)backward_lstm_379/zeros_1/packed:output:0(backward_lstm_379/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_379/zeros_1Щ
 backward_lstm_379/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_379/transpose/permс
backward_lstm_379/transpose	Transpose>backward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_379/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_379/transposeЕ
backward_lstm_379/Shape_1Shapebackward_lstm_379/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_379/Shape_1Ь
'backward_lstm_379/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_379/strided_slice_1/stack†
)backward_lstm_379/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_1/stack_1†
)backward_lstm_379/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_1/stack_2Џ
!backward_lstm_379/strided_slice_1StridedSlice"backward_lstm_379/Shape_1:output:00backward_lstm_379/strided_slice_1/stack:output:02backward_lstm_379/strided_slice_1/stack_1:output:02backward_lstm_379/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_379/strided_slice_1©
-backward_lstm_379/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-backward_lstm_379/TensorArrayV2/element_shapeъ
backward_lstm_379/TensorArrayV2TensorListReserve6backward_lstm_379/TensorArrayV2/element_shape:output:0*backward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_379/TensorArrayV2О
 backward_lstm_379/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_379/ReverseV2/axis“
backward_lstm_379/ReverseV2	ReverseV2backward_lstm_379/transpose:y:0)backward_lstm_379/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_379/ReverseV2г
Gbackward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2I
Gbackward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape≈
9backward_lstm_379/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_379/ReverseV2:output:0Pbackward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_379/TensorArrayUnstack/TensorListFromTensorЬ
'backward_lstm_379/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_379/strided_slice_2/stack†
)backward_lstm_379/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_2/stack_1†
)backward_lstm_379/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_2/stack_2и
!backward_lstm_379/strided_slice_2StridedSlicebackward_lstm_379/transpose:y:00backward_lstm_379/strided_slice_2/stack:output:02backward_lstm_379/strided_slice_2/stack_1:output:02backward_lstm_379/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2#
!backward_lstm_379/strided_slice_2с
6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOp?backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype028
6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpы
'backward_lstm_379/lstm_cell_1139/MatMulMatMul*backward_lstm_379/strided_slice_2:output:0>backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'backward_lstm_379/lstm_cell_1139/MatMulч
8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOpAbackward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02:
8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpч
)backward_lstm_379/lstm_cell_1139/MatMul_1MatMul backward_lstm_379/zeros:output:0@backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)backward_lstm_379/lstm_cell_1139/MatMul_1р
$backward_lstm_379/lstm_cell_1139/addAddV21backward_lstm_379/lstm_cell_1139/MatMul:product:03backward_lstm_379/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2&
$backward_lstm_379/lstm_cell_1139/addр
7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOp@backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype029
7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpэ
(backward_lstm_379/lstm_cell_1139/BiasAddBiasAdd(backward_lstm_379/lstm_cell_1139/add:z:0?backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(backward_lstm_379/lstm_cell_1139/BiasAdd¶
0backward_lstm_379/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0backward_lstm_379/lstm_cell_1139/split/split_dim√
&backward_lstm_379/lstm_cell_1139/splitSplit9backward_lstm_379/lstm_cell_1139/split/split_dim:output:01backward_lstm_379/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2(
&backward_lstm_379/lstm_cell_1139/split¬
(backward_lstm_379/lstm_cell_1139/SigmoidSigmoid/backward_lstm_379/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(backward_lstm_379/lstm_cell_1139/Sigmoid∆
*backward_lstm_379/lstm_cell_1139/Sigmoid_1Sigmoid/backward_lstm_379/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_379/lstm_cell_1139/Sigmoid_1ў
$backward_lstm_379/lstm_cell_1139/mulMul.backward_lstm_379/lstm_cell_1139/Sigmoid_1:y:0"backward_lstm_379/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$backward_lstm_379/lstm_cell_1139/mulє
%backward_lstm_379/lstm_cell_1139/ReluRelu/backward_lstm_379/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22'
%backward_lstm_379/lstm_cell_1139/Reluм
&backward_lstm_379/lstm_cell_1139/mul_1Mul,backward_lstm_379/lstm_cell_1139/Sigmoid:y:03backward_lstm_379/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_379/lstm_cell_1139/mul_1б
&backward_lstm_379/lstm_cell_1139/add_1AddV2(backward_lstm_379/lstm_cell_1139/mul:z:0*backward_lstm_379/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_379/lstm_cell_1139/add_1∆
*backward_lstm_379/lstm_cell_1139/Sigmoid_2Sigmoid/backward_lstm_379/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_379/lstm_cell_1139/Sigmoid_2Є
'backward_lstm_379/lstm_cell_1139/Relu_1Relu*backward_lstm_379/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'backward_lstm_379/lstm_cell_1139/Relu_1р
&backward_lstm_379/lstm_cell_1139/mul_2Mul.backward_lstm_379/lstm_cell_1139/Sigmoid_2:y:05backward_lstm_379/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_379/lstm_cell_1139/mul_2≥
/backward_lstm_379/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   21
/backward_lstm_379/TensorArrayV2_1/element_shapeА
!backward_lstm_379/TensorArrayV2_1TensorListReserve8backward_lstm_379/TensorArrayV2_1/element_shape:output:0*backward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_379/TensorArrayV2_1r
backward_lstm_379/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_379/timeФ
'backward_lstm_379/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_379/Max/reduction_indices§
backward_lstm_379/MaxMaxbackward_lstm_379/Cast:y:00backward_lstm_379/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/Maxt
backward_lstm_379/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_379/sub/yШ
backward_lstm_379/subSubbackward_lstm_379/Max:output:0 backward_lstm_379/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/subЮ
backward_lstm_379/Sub_1Subbackward_lstm_379/sub:z:0backward_lstm_379/Cast:y:0*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_379/Sub_1І
backward_lstm_379/zeros_like	ZerosLike*backward_lstm_379/lstm_cell_1139/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_379/zeros_like£
*backward_lstm_379/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2,
*backward_lstm_379/while/maximum_iterationsО
$backward_lstm_379/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_379/while/loop_counter®	
backward_lstm_379/whileWhile-backward_lstm_379/while/loop_counter:output:03backward_lstm_379/while/maximum_iterations:output:0backward_lstm_379/time:output:0*backward_lstm_379/TensorArrayV2_1:handle:0 backward_lstm_379/zeros_like:y:0 backward_lstm_379/zeros:output:0"backward_lstm_379/zeros_1:output:0*backward_lstm_379/strided_slice_1:output:0Ibackward_lstm_379/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_379/Sub_1:z:0?backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resourceAbackward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource@backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *1
body)R'
%backward_lstm_379_while_body_42662187*1
cond)R'
%backward_lstm_379_while_cond_42662186*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
backward_lstm_379/whileў
Bbackward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2D
Bbackward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeє
4backward_lstm_379/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_379/while:output:3Kbackward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype026
4backward_lstm_379/TensorArrayV2Stack/TensorListStack•
'backward_lstm_379/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2)
'backward_lstm_379/strided_slice_3/stack†
)backward_lstm_379/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_379/strided_slice_3/stack_1†
)backward_lstm_379/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_3/stack_2Ж
!backward_lstm_379/strided_slice_3StridedSlice=backward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_379/strided_slice_3/stack:output:02backward_lstm_379/strided_slice_3/stack_1:output:02backward_lstm_379/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2#
!backward_lstm_379/strided_slice_3Э
"backward_lstm_379/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_379/transpose_1/permц
backward_lstm_379/transpose_1	Transpose=backward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_379/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
backward_lstm_379/transpose_1К
backward_lstm_379/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_379/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisƒ
concatConcatV2)forward_lstm_379/strided_slice_3:output:0*backward_lstm_379/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d2

IdentityЏ
NoOpNoOp8^backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp7^backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp9^backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp^backward_lstm_379/while7^forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp6^forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp8^forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp^forward_lstm_379/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€:€€€€€€€€€: : : : : : 2r
7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp2p
6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp2t
8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp22
backward_lstm_379/whilebackward_lstm_379/while2p
6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp2n
5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp2r
7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp20
forward_lstm_379/whileforward_lstm_379/while:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Є
£
L__inference_sequential_379_layer_call_and_return_conditional_losses_42660306

inputs
inputs_1	-
bidirectional_379_42660275:	»-
bidirectional_379_42660277:	2»)
bidirectional_379_42660279:	»-
bidirectional_379_42660281:	»-
bidirectional_379_42660283:	2»)
bidirectional_379_42660285:	»$
dense_379_42660300:d 
dense_379_42660302:
identityИҐ)bidirectional_379/StatefulPartitionedCallҐ!dense_379/StatefulPartitionedCall 
)bidirectional_379/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_379_42660275bidirectional_379_42660277bidirectional_379_42660279bidirectional_379_42660281bidirectional_379_42660283bidirectional_379_42660285*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_426602742+
)bidirectional_379/StatefulPartitionedCallЋ
!dense_379/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_379/StatefulPartitionedCall:output:0dense_379_42660300dense_379_42660302*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dense_379_layer_call_and_return_conditional_losses_426602992#
!dense_379/StatefulPartitionedCallЕ
IdentityIdentity*dense_379/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityЮ
NoOpNoOp*^bidirectional_379/StatefulPartitionedCall"^dense_379/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€:€€€€€€€€€: : : : : : : : 2V
)bidirectional_379/StatefulPartitionedCall)bidirectional_379/StatefulPartitionedCall2F
!dense_379/StatefulPartitionedCall!dense_379/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
…
•
$forward_lstm_379_while_cond_42660437>
:forward_lstm_379_while_forward_lstm_379_while_loop_counterD
@forward_lstm_379_while_forward_lstm_379_while_maximum_iterations&
"forward_lstm_379_while_placeholder(
$forward_lstm_379_while_placeholder_1(
$forward_lstm_379_while_placeholder_2(
$forward_lstm_379_while_placeholder_3(
$forward_lstm_379_while_placeholder_4@
<forward_lstm_379_while_less_forward_lstm_379_strided_slice_1X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42660437___redundant_placeholder0X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42660437___redundant_placeholder1X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42660437___redundant_placeholder2X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42660437___redundant_placeholder3X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42660437___redundant_placeholder4#
forward_lstm_379_while_identity
≈
forward_lstm_379/while/LessLess"forward_lstm_379_while_placeholder<forward_lstm_379_while_less_forward_lstm_379_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_379/while/LessР
forward_lstm_379/while/IdentityIdentityforward_lstm_379/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_379/while/Identity"K
forward_lstm_379_while_identity(forward_lstm_379/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
зe
Ћ
$forward_lstm_379_while_body_42660438>
:forward_lstm_379_while_forward_lstm_379_while_loop_counterD
@forward_lstm_379_while_forward_lstm_379_while_maximum_iterations&
"forward_lstm_379_while_placeholder(
$forward_lstm_379_while_placeholder_1(
$forward_lstm_379_while_placeholder_2(
$forward_lstm_379_while_placeholder_3(
$forward_lstm_379_while_placeholder_4=
9forward_lstm_379_while_forward_lstm_379_strided_slice_1_0y
uforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_379_while_greater_forward_lstm_379_cast_0Y
Fforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0:	»[
Hforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0:	2»V
Gforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0:	»#
forward_lstm_379_while_identity%
!forward_lstm_379_while_identity_1%
!forward_lstm_379_while_identity_2%
!forward_lstm_379_while_identity_3%
!forward_lstm_379_while_identity_4%
!forward_lstm_379_while_identity_5%
!forward_lstm_379_while_identity_6;
7forward_lstm_379_while_forward_lstm_379_strided_slice_1w
sforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_379_while_greater_forward_lstm_379_castW
Dforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource:	»Y
Fforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource:	2»T
Eforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource:	»ИҐ<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpҐ;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOpҐ=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpе
Hforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2J
Hforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeє
:forward_lstm_379/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_379_while_placeholderQforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02<
:forward_lstm_379/while/TensorArrayV2Read/TensorListGetItem’
forward_lstm_379/while/GreaterGreater6forward_lstm_379_while_greater_forward_lstm_379_cast_0"forward_lstm_379_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2 
forward_lstm_379/while/GreaterВ
;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOpFforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02=
;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp°
,forward_lstm_379/while/lstm_cell_1138/MatMulMatMulAforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem:item:0Cforward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2.
,forward_lstm_379/while/lstm_cell_1138/MatMulИ
=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOpHforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02?
=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpК
.forward_lstm_379/while/lstm_cell_1138/MatMul_1MatMul$forward_lstm_379_while_placeholder_3Eforward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.forward_lstm_379/while/lstm_cell_1138/MatMul_1Д
)forward_lstm_379/while/lstm_cell_1138/addAddV26forward_lstm_379/while/lstm_cell_1138/MatMul:product:08forward_lstm_379/while/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)forward_lstm_379/while/lstm_cell_1138/addБ
<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOpGforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02>
<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpС
-forward_lstm_379/while/lstm_cell_1138/BiasAddBiasAdd-forward_lstm_379/while/lstm_cell_1138/add:z:0Dforward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-forward_lstm_379/while/lstm_cell_1138/BiasAdd∞
5forward_lstm_379/while/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5forward_lstm_379/while/lstm_cell_1138/split/split_dim„
+forward_lstm_379/while/lstm_cell_1138/splitSplit>forward_lstm_379/while/lstm_cell_1138/split/split_dim:output:06forward_lstm_379/while/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2-
+forward_lstm_379/while/lstm_cell_1138/split—
-forward_lstm_379/while/lstm_cell_1138/SigmoidSigmoid4forward_lstm_379/while/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-forward_lstm_379/while/lstm_cell_1138/Sigmoid’
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_1Sigmoid4forward_lstm_379/while/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_1к
)forward_lstm_379/while/lstm_cell_1138/mulMul3forward_lstm_379/while/lstm_cell_1138/Sigmoid_1:y:0$forward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_379/while/lstm_cell_1138/mul»
*forward_lstm_379/while/lstm_cell_1138/ReluRelu4forward_lstm_379/while/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22,
*forward_lstm_379/while/lstm_cell_1138/ReluА
+forward_lstm_379/while/lstm_cell_1138/mul_1Mul1forward_lstm_379/while/lstm_cell_1138/Sigmoid:y:08forward_lstm_379/while/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_379/while/lstm_cell_1138/mul_1х
+forward_lstm_379/while/lstm_cell_1138/add_1AddV2-forward_lstm_379/while/lstm_cell_1138/mul:z:0/forward_lstm_379/while/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_379/while/lstm_cell_1138/add_1’
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_2Sigmoid4forward_lstm_379/while/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_2«
,forward_lstm_379/while/lstm_cell_1138/Relu_1Relu/forward_lstm_379/while/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,forward_lstm_379/while/lstm_cell_1138/Relu_1Д
+forward_lstm_379/while/lstm_cell_1138/mul_2Mul3forward_lstm_379/while/lstm_cell_1138/Sigmoid_2:y:0:forward_lstm_379/while/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_379/while/lstm_cell_1138/mul_2х
forward_lstm_379/while/SelectSelect"forward_lstm_379/while/Greater:z:0/forward_lstm_379/while/lstm_cell_1138/mul_2:z:0$forward_lstm_379_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/while/Selectщ
forward_lstm_379/while/Select_1Select"forward_lstm_379/while/Greater:z:0/forward_lstm_379/while/lstm_cell_1138/mul_2:z:0$forward_lstm_379_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_379/while/Select_1щ
forward_lstm_379/while/Select_2Select"forward_lstm_379/while/Greater:z:0/forward_lstm_379/while/lstm_cell_1138/add_1:z:0$forward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_379/while/Select_2Ѓ
;forward_lstm_379/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_379_while_placeholder_1"forward_lstm_379_while_placeholder&forward_lstm_379/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_379/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_379/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_379/while/add/y≠
forward_lstm_379/while/addAddV2"forward_lstm_379_while_placeholder%forward_lstm_379/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/while/addВ
forward_lstm_379/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_379/while/add_1/yЋ
forward_lstm_379/while/add_1AddV2:forward_lstm_379_while_forward_lstm_379_while_loop_counter'forward_lstm_379/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/while/add_1ѓ
forward_lstm_379/while/IdentityIdentity forward_lstm_379/while/add_1:z:0^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_379/while/Identity”
!forward_lstm_379/while/Identity_1Identity@forward_lstm_379_while_forward_lstm_379_while_maximum_iterations^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_379/while/Identity_1±
!forward_lstm_379/while/Identity_2Identityforward_lstm_379/while/add:z:0^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_379/while/Identity_2ё
!forward_lstm_379/while/Identity_3IdentityKforward_lstm_379/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_379/while/Identity_3 
!forward_lstm_379/while/Identity_4Identity&forward_lstm_379/while/Select:output:0^forward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_379/while/Identity_4ћ
!forward_lstm_379/while/Identity_5Identity(forward_lstm_379/while/Select_1:output:0^forward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_379/while/Identity_5ћ
!forward_lstm_379/while/Identity_6Identity(forward_lstm_379/while/Select_2:output:0^forward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_379/while/Identity_6є
forward_lstm_379/while/NoOpNoOp=^forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp<^forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp>^forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_379/while/NoOp"t
7forward_lstm_379_while_forward_lstm_379_strided_slice_19forward_lstm_379_while_forward_lstm_379_strided_slice_1_0"n
4forward_lstm_379_while_greater_forward_lstm_379_cast6forward_lstm_379_while_greater_forward_lstm_379_cast_0"K
forward_lstm_379_while_identity(forward_lstm_379/while/Identity:output:0"O
!forward_lstm_379_while_identity_1*forward_lstm_379/while/Identity_1:output:0"O
!forward_lstm_379_while_identity_2*forward_lstm_379/while/Identity_2:output:0"O
!forward_lstm_379_while_identity_3*forward_lstm_379/while/Identity_3:output:0"O
!forward_lstm_379_while_identity_4*forward_lstm_379/while/Identity_4:output:0"O
!forward_lstm_379_while_identity_5*forward_lstm_379/while/Identity_5:output:0"O
!forward_lstm_379_while_identity_6*forward_lstm_379/while/Identity_6:output:0"Р
Eforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resourceGforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0"Т
Fforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resourceHforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0"О
Dforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resourceFforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0"м
sforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensoruforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2|
<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp2z
;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp2~
=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:€€€€€€€€€
ђ&
€
while_body_42658559
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_lstm_cell_1139_42658583_0:	»2
while_lstm_cell_1139_42658585_0:	2».
while_lstm_cell_1139_42658587_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_lstm_cell_1139_42658583:	»0
while_lstm_cell_1139_42658585:	2»,
while_lstm_cell_1139_42658587:	»ИҐ,while/lstm_cell_1139/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemх
,while/lstm_cell_1139/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1139_42658583_0while_lstm_cell_1139_42658585_0while_lstm_cell_1139_42658587_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_lstm_cell_1139_layer_call_and_return_conditional_losses_426585452.
,while/lstm_cell_1139/StatefulPartitionedCallщ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder5while/lstm_cell_1139/StatefulPartitionedCall:output:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¶
while/Identity_4Identity5while/lstm_cell_1139/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4¶
while/Identity_5Identity5while/lstm_cell_1139/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5Й

while/NoOpNoOp-^while/lstm_cell_1139/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"@
while_lstm_cell_1139_42658583while_lstm_cell_1139_42658583_0"@
while_lstm_cell_1139_42658585while_lstm_cell_1139_42658585_0"@
while_lstm_cell_1139_42658587while_lstm_cell_1139_42658587_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2\
,while/lstm_cell_1139/StatefulPartitionedCall,while/lstm_cell_1139/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
З
ш
G__inference_dense_379_layer_call_and_return_conditional_losses_42662304

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
ЩЮ
Є
Esequential_379_bidirectional_379_forward_lstm_379_while_body_42657555А
|sequential_379_bidirectional_379_forward_lstm_379_while_sequential_379_bidirectional_379_forward_lstm_379_while_loop_counterЗ
Вsequential_379_bidirectional_379_forward_lstm_379_while_sequential_379_bidirectional_379_forward_lstm_379_while_maximum_iterationsG
Csequential_379_bidirectional_379_forward_lstm_379_while_placeholderI
Esequential_379_bidirectional_379_forward_lstm_379_while_placeholder_1I
Esequential_379_bidirectional_379_forward_lstm_379_while_placeholder_2I
Esequential_379_bidirectional_379_forward_lstm_379_while_placeholder_3I
Esequential_379_bidirectional_379_forward_lstm_379_while_placeholder_4
{sequential_379_bidirectional_379_forward_lstm_379_while_sequential_379_bidirectional_379_forward_lstm_379_strided_slice_1_0Љ
Јsequential_379_bidirectional_379_forward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_sequential_379_bidirectional_379_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0|
xsequential_379_bidirectional_379_forward_lstm_379_while_greater_sequential_379_bidirectional_379_forward_lstm_379_cast_0z
gsequential_379_bidirectional_379_forward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0:	»|
isequential_379_bidirectional_379_forward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0:	2»w
hsequential_379_bidirectional_379_forward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0:	»D
@sequential_379_bidirectional_379_forward_lstm_379_while_identityF
Bsequential_379_bidirectional_379_forward_lstm_379_while_identity_1F
Bsequential_379_bidirectional_379_forward_lstm_379_while_identity_2F
Bsequential_379_bidirectional_379_forward_lstm_379_while_identity_3F
Bsequential_379_bidirectional_379_forward_lstm_379_while_identity_4F
Bsequential_379_bidirectional_379_forward_lstm_379_while_identity_5F
Bsequential_379_bidirectional_379_forward_lstm_379_while_identity_6}
ysequential_379_bidirectional_379_forward_lstm_379_while_sequential_379_bidirectional_379_forward_lstm_379_strided_slice_1Ї
µsequential_379_bidirectional_379_forward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_sequential_379_bidirectional_379_forward_lstm_379_tensorarrayunstack_tensorlistfromtensorz
vsequential_379_bidirectional_379_forward_lstm_379_while_greater_sequential_379_bidirectional_379_forward_lstm_379_castx
esequential_379_bidirectional_379_forward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource:	»z
gsequential_379_bidirectional_379_forward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource:	2»u
fsequential_379_bidirectional_379_forward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource:	»ИҐ]sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpҐ\sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOpҐ^sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpІ
isequential_379/bidirectional_379/forward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2k
isequential_379/bidirectional_379/forward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeА
[sequential_379/bidirectional_379/forward_lstm_379/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЈsequential_379_bidirectional_379_forward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_sequential_379_bidirectional_379_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0Csequential_379_bidirectional_379_forward_lstm_379_while_placeholderrsequential_379/bidirectional_379/forward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02]
[sequential_379/bidirectional_379/forward_lstm_379/while/TensorArrayV2Read/TensorListGetItemъ
?sequential_379/bidirectional_379/forward_lstm_379/while/GreaterGreaterxsequential_379_bidirectional_379_forward_lstm_379_while_greater_sequential_379_bidirectional_379_forward_lstm_379_cast_0Csequential_379_bidirectional_379_forward_lstm_379_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2A
?sequential_379/bidirectional_379/forward_lstm_379/while/Greaterе
\sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOpgsequential_379_bidirectional_379_forward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02^
\sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp•
Msequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMulMatMulbsequential_379/bidirectional_379/forward_lstm_379/while/TensorArrayV2Read/TensorListGetItem:item:0dsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2O
Msequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMulл
^sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOpisequential_379_bidirectional_379_forward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02`
^sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpО
Osequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMul_1MatMulEsequential_379_bidirectional_379_forward_lstm_379_while_placeholder_3fsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2Q
Osequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMul_1И
Jsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/addAddV2Wsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMul:product:0Ysequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2L
Jsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/addд
]sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOphsequential_379_bidirectional_379_forward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02_
]sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpХ
Nsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/BiasAddBiasAddNsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/add:z:0esequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2P
Nsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/BiasAddт
Vsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2X
Vsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/split/split_dimџ
Lsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/splitSplit_sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/split/split_dim:output:0Wsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2N
Lsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/splitі
Nsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/SigmoidSigmoidUsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22P
Nsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/SigmoidЄ
Psequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/Sigmoid_1SigmoidUsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22R
Psequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/Sigmoid_1о
Jsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/mulMulTsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/Sigmoid_1:y:0Esequential_379_bidirectional_379_forward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22L
Jsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/mulЂ
Ksequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/ReluReluUsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22M
Ksequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/ReluД
Lsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/mul_1MulRsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/Sigmoid:y:0Ysequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22N
Lsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/mul_1щ
Lsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/add_1AddV2Nsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/mul:z:0Psequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22N
Lsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/add_1Є
Psequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/Sigmoid_2SigmoidUsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22R
Psequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/Sigmoid_2™
Msequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/Relu_1ReluPsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22O
Msequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/Relu_1И
Lsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/mul_2MulTsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/Sigmoid_2:y:0[sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22N
Lsequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/mul_2Ъ
>sequential_379/bidirectional_379/forward_lstm_379/while/SelectSelectCsequential_379/bidirectional_379/forward_lstm_379/while/Greater:z:0Psequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/mul_2:z:0Esequential_379_bidirectional_379_forward_lstm_379_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22@
>sequential_379/bidirectional_379/forward_lstm_379/while/SelectЮ
@sequential_379/bidirectional_379/forward_lstm_379/while/Select_1SelectCsequential_379/bidirectional_379/forward_lstm_379/while/Greater:z:0Psequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/mul_2:z:0Esequential_379_bidirectional_379_forward_lstm_379_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22B
@sequential_379/bidirectional_379/forward_lstm_379/while/Select_1Ю
@sequential_379/bidirectional_379/forward_lstm_379/while/Select_2SelectCsequential_379/bidirectional_379/forward_lstm_379/while/Greater:z:0Psequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/add_1:z:0Esequential_379_bidirectional_379_forward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22B
@sequential_379/bidirectional_379/forward_lstm_379/while/Select_2”
\sequential_379/bidirectional_379/forward_lstm_379/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemEsequential_379_bidirectional_379_forward_lstm_379_while_placeholder_1Csequential_379_bidirectional_379_forward_lstm_379_while_placeholderGsequential_379/bidirectional_379/forward_lstm_379/while/Select:output:0*
_output_shapes
: *
element_dtype02^
\sequential_379/bidirectional_379/forward_lstm_379/while/TensorArrayV2Write/TensorListSetItemј
=sequential_379/bidirectional_379/forward_lstm_379/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2?
=sequential_379/bidirectional_379/forward_lstm_379/while/add/y±
;sequential_379/bidirectional_379/forward_lstm_379/while/addAddV2Csequential_379_bidirectional_379_forward_lstm_379_while_placeholderFsequential_379/bidirectional_379/forward_lstm_379/while/add/y:output:0*
T0*
_output_shapes
: 2=
;sequential_379/bidirectional_379/forward_lstm_379/while/addƒ
?sequential_379/bidirectional_379/forward_lstm_379/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2A
?sequential_379/bidirectional_379/forward_lstm_379/while/add_1/yр
=sequential_379/bidirectional_379/forward_lstm_379/while/add_1AddV2|sequential_379_bidirectional_379_forward_lstm_379_while_sequential_379_bidirectional_379_forward_lstm_379_while_loop_counterHsequential_379/bidirectional_379/forward_lstm_379/while/add_1/y:output:0*
T0*
_output_shapes
: 2?
=sequential_379/bidirectional_379/forward_lstm_379/while/add_1≥
@sequential_379/bidirectional_379/forward_lstm_379/while/IdentityIdentityAsequential_379/bidirectional_379/forward_lstm_379/while/add_1:z:0=^sequential_379/bidirectional_379/forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2B
@sequential_379/bidirectional_379/forward_lstm_379/while/Identityщ
Bsequential_379/bidirectional_379/forward_lstm_379/while/Identity_1IdentityВsequential_379_bidirectional_379_forward_lstm_379_while_sequential_379_bidirectional_379_forward_lstm_379_while_maximum_iterations=^sequential_379/bidirectional_379/forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2D
Bsequential_379/bidirectional_379/forward_lstm_379/while/Identity_1µ
Bsequential_379/bidirectional_379/forward_lstm_379/while/Identity_2Identity?sequential_379/bidirectional_379/forward_lstm_379/while/add:z:0=^sequential_379/bidirectional_379/forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2D
Bsequential_379/bidirectional_379/forward_lstm_379/while/Identity_2в
Bsequential_379/bidirectional_379/forward_lstm_379/while/Identity_3Identitylsequential_379/bidirectional_379/forward_lstm_379/while/TensorArrayV2Write/TensorListSetItem:output_handle:0=^sequential_379/bidirectional_379/forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2D
Bsequential_379/bidirectional_379/forward_lstm_379/while/Identity_3ќ
Bsequential_379/bidirectional_379/forward_lstm_379/while/Identity_4IdentityGsequential_379/bidirectional_379/forward_lstm_379/while/Select:output:0=^sequential_379/bidirectional_379/forward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22D
Bsequential_379/bidirectional_379/forward_lstm_379/while/Identity_4–
Bsequential_379/bidirectional_379/forward_lstm_379/while/Identity_5IdentityIsequential_379/bidirectional_379/forward_lstm_379/while/Select_1:output:0=^sequential_379/bidirectional_379/forward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22D
Bsequential_379/bidirectional_379/forward_lstm_379/while/Identity_5–
Bsequential_379/bidirectional_379/forward_lstm_379/while/Identity_6IdentityIsequential_379/bidirectional_379/forward_lstm_379/while/Select_2:output:0=^sequential_379/bidirectional_379/forward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22D
Bsequential_379/bidirectional_379/forward_lstm_379/while/Identity_6ё
<sequential_379/bidirectional_379/forward_lstm_379/while/NoOpNoOp^^sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp]^sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp_^sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2>
<sequential_379/bidirectional_379/forward_lstm_379/while/NoOp"т
vsequential_379_bidirectional_379_forward_lstm_379_while_greater_sequential_379_bidirectional_379_forward_lstm_379_castxsequential_379_bidirectional_379_forward_lstm_379_while_greater_sequential_379_bidirectional_379_forward_lstm_379_cast_0"Н
@sequential_379_bidirectional_379_forward_lstm_379_while_identityIsequential_379/bidirectional_379/forward_lstm_379/while/Identity:output:0"С
Bsequential_379_bidirectional_379_forward_lstm_379_while_identity_1Ksequential_379/bidirectional_379/forward_lstm_379/while/Identity_1:output:0"С
Bsequential_379_bidirectional_379_forward_lstm_379_while_identity_2Ksequential_379/bidirectional_379/forward_lstm_379/while/Identity_2:output:0"С
Bsequential_379_bidirectional_379_forward_lstm_379_while_identity_3Ksequential_379/bidirectional_379/forward_lstm_379/while/Identity_3:output:0"С
Bsequential_379_bidirectional_379_forward_lstm_379_while_identity_4Ksequential_379/bidirectional_379/forward_lstm_379/while/Identity_4:output:0"С
Bsequential_379_bidirectional_379_forward_lstm_379_while_identity_5Ksequential_379/bidirectional_379/forward_lstm_379/while/Identity_5:output:0"С
Bsequential_379_bidirectional_379_forward_lstm_379_while_identity_6Ksequential_379/bidirectional_379/forward_lstm_379/while/Identity_6:output:0"“
fsequential_379_bidirectional_379_forward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resourcehsequential_379_bidirectional_379_forward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0"‘
gsequential_379_bidirectional_379_forward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resourceisequential_379_bidirectional_379_forward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0"–
esequential_379_bidirectional_379_forward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resourcegsequential_379_bidirectional_379_forward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0"ш
ysequential_379_bidirectional_379_forward_lstm_379_while_sequential_379_bidirectional_379_forward_lstm_379_strided_slice_1{sequential_379_bidirectional_379_forward_lstm_379_while_sequential_379_bidirectional_379_forward_lstm_379_strided_slice_1_0"т
µsequential_379_bidirectional_379_forward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_sequential_379_bidirectional_379_forward_lstm_379_tensorarrayunstack_tensorlistfromtensorЈsequential_379_bidirectional_379_forward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_sequential_379_bidirectional_379_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2Њ
]sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp]sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp2Љ
\sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp\sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp2ј
^sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp^sequential_379/bidirectional_379/forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:€€€€€€€€€
я
Ќ
while_cond_42657926
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_42657926___redundant_placeholder06
2while_while_cond_42657926___redundant_placeholder16
2while_while_cond_42657926___redundant_placeholder26
2while_while_cond_42657926___redundant_placeholder3
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
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:
оX
Д
$forward_lstm_379_while_body_42661333>
:forward_lstm_379_while_forward_lstm_379_while_loop_counterD
@forward_lstm_379_while_forward_lstm_379_while_maximum_iterations&
"forward_lstm_379_while_placeholder(
$forward_lstm_379_while_placeholder_1(
$forward_lstm_379_while_placeholder_2(
$forward_lstm_379_while_placeholder_3=
9forward_lstm_379_while_forward_lstm_379_strided_slice_1_0y
uforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0Y
Fforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0:	»[
Hforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0:	2»V
Gforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0:	»#
forward_lstm_379_while_identity%
!forward_lstm_379_while_identity_1%
!forward_lstm_379_while_identity_2%
!forward_lstm_379_while_identity_3%
!forward_lstm_379_while_identity_4%
!forward_lstm_379_while_identity_5;
7forward_lstm_379_while_forward_lstm_379_strided_slice_1w
sforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensorW
Dforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource:	»Y
Fforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource:	2»T
Eforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource:	»ИҐ<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpҐ;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOpҐ=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpе
Hforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2J
Hforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shape¬
:forward_lstm_379/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_379_while_placeholderQforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02<
:forward_lstm_379/while/TensorArrayV2Read/TensorListGetItemВ
;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOpFforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02=
;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp°
,forward_lstm_379/while/lstm_cell_1138/MatMulMatMulAforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem:item:0Cforward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2.
,forward_lstm_379/while/lstm_cell_1138/MatMulИ
=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOpHforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02?
=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpК
.forward_lstm_379/while/lstm_cell_1138/MatMul_1MatMul$forward_lstm_379_while_placeholder_2Eforward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.forward_lstm_379/while/lstm_cell_1138/MatMul_1Д
)forward_lstm_379/while/lstm_cell_1138/addAddV26forward_lstm_379/while/lstm_cell_1138/MatMul:product:08forward_lstm_379/while/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)forward_lstm_379/while/lstm_cell_1138/addБ
<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOpGforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02>
<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpС
-forward_lstm_379/while/lstm_cell_1138/BiasAddBiasAdd-forward_lstm_379/while/lstm_cell_1138/add:z:0Dforward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-forward_lstm_379/while/lstm_cell_1138/BiasAdd∞
5forward_lstm_379/while/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5forward_lstm_379/while/lstm_cell_1138/split/split_dim„
+forward_lstm_379/while/lstm_cell_1138/splitSplit>forward_lstm_379/while/lstm_cell_1138/split/split_dim:output:06forward_lstm_379/while/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2-
+forward_lstm_379/while/lstm_cell_1138/split—
-forward_lstm_379/while/lstm_cell_1138/SigmoidSigmoid4forward_lstm_379/while/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-forward_lstm_379/while/lstm_cell_1138/Sigmoid’
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_1Sigmoid4forward_lstm_379/while/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_1к
)forward_lstm_379/while/lstm_cell_1138/mulMul3forward_lstm_379/while/lstm_cell_1138/Sigmoid_1:y:0$forward_lstm_379_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_379/while/lstm_cell_1138/mul»
*forward_lstm_379/while/lstm_cell_1138/ReluRelu4forward_lstm_379/while/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22,
*forward_lstm_379/while/lstm_cell_1138/ReluА
+forward_lstm_379/while/lstm_cell_1138/mul_1Mul1forward_lstm_379/while/lstm_cell_1138/Sigmoid:y:08forward_lstm_379/while/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_379/while/lstm_cell_1138/mul_1х
+forward_lstm_379/while/lstm_cell_1138/add_1AddV2-forward_lstm_379/while/lstm_cell_1138/mul:z:0/forward_lstm_379/while/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_379/while/lstm_cell_1138/add_1’
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_2Sigmoid4forward_lstm_379/while/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_2«
,forward_lstm_379/while/lstm_cell_1138/Relu_1Relu/forward_lstm_379/while/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,forward_lstm_379/while/lstm_cell_1138/Relu_1Д
+forward_lstm_379/while/lstm_cell_1138/mul_2Mul3forward_lstm_379/while/lstm_cell_1138/Sigmoid_2:y:0:forward_lstm_379/while/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_379/while/lstm_cell_1138/mul_2Ј
;forward_lstm_379/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_379_while_placeholder_1"forward_lstm_379_while_placeholder/forward_lstm_379/while/lstm_cell_1138/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_379/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_379/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_379/while/add/y≠
forward_lstm_379/while/addAddV2"forward_lstm_379_while_placeholder%forward_lstm_379/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/while/addВ
forward_lstm_379/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_379/while/add_1/yЋ
forward_lstm_379/while/add_1AddV2:forward_lstm_379_while_forward_lstm_379_while_loop_counter'forward_lstm_379/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/while/add_1ѓ
forward_lstm_379/while/IdentityIdentity forward_lstm_379/while/add_1:z:0^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_379/while/Identity”
!forward_lstm_379/while/Identity_1Identity@forward_lstm_379_while_forward_lstm_379_while_maximum_iterations^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_379/while/Identity_1±
!forward_lstm_379/while/Identity_2Identityforward_lstm_379/while/add:z:0^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_379/while/Identity_2ё
!forward_lstm_379/while/Identity_3IdentityKforward_lstm_379/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_379/while/Identity_3”
!forward_lstm_379/while/Identity_4Identity/forward_lstm_379/while/lstm_cell_1138/mul_2:z:0^forward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_379/while/Identity_4”
!forward_lstm_379/while/Identity_5Identity/forward_lstm_379/while/lstm_cell_1138/add_1:z:0^forward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_379/while/Identity_5є
forward_lstm_379/while/NoOpNoOp=^forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp<^forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp>^forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_379/while/NoOp"t
7forward_lstm_379_while_forward_lstm_379_strided_slice_19forward_lstm_379_while_forward_lstm_379_strided_slice_1_0"K
forward_lstm_379_while_identity(forward_lstm_379/while/Identity:output:0"O
!forward_lstm_379_while_identity_1*forward_lstm_379/while/Identity_1:output:0"O
!forward_lstm_379_while_identity_2*forward_lstm_379/while/Identity_2:output:0"O
!forward_lstm_379_while_identity_3*forward_lstm_379/while/Identity_3:output:0"O
!forward_lstm_379_while_identity_4*forward_lstm_379/while/Identity_4:output:0"O
!forward_lstm_379_while_identity_5*forward_lstm_379/while/Identity_5:output:0"Р
Eforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resourceGforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0"Т
Fforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resourceHforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0"О
Dforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resourceFforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0"м
sforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensoruforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2|
<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp2z
;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp2~
=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
÷
√
4__inference_backward_lstm_379_layer_call_fn_42662974
inputs_0
unknown:	»
	unknown_0:	2»
	unknown_1:	»
identityИҐStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_426588402
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
я
°
$forward_lstm_379_while_cond_42661030>
:forward_lstm_379_while_forward_lstm_379_while_loop_counterD
@forward_lstm_379_while_forward_lstm_379_while_maximum_iterations&
"forward_lstm_379_while_placeholder(
$forward_lstm_379_while_placeholder_1(
$forward_lstm_379_while_placeholder_2(
$forward_lstm_379_while_placeholder_3@
<forward_lstm_379_while_less_forward_lstm_379_strided_slice_1X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42661030___redundant_placeholder0X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42661030___redundant_placeholder1X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42661030___redundant_placeholder2X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42661030___redundant_placeholder3#
forward_lstm_379_while_identity
≈
forward_lstm_379/while/LessLess"forward_lstm_379_while_placeholder<forward_lstm_379_while_less_forward_lstm_379_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_379/while/LessР
forward_lstm_379/while/IdentityIdentityforward_lstm_379/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_379/while/Identity"K
forward_lstm_379_while_identity(forward_lstm_379/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:
¶Z
§
%backward_lstm_379_while_body_42661180@
<backward_lstm_379_while_backward_lstm_379_while_loop_counterF
Bbackward_lstm_379_while_backward_lstm_379_while_maximum_iterations'
#backward_lstm_379_while_placeholder)
%backward_lstm_379_while_placeholder_1)
%backward_lstm_379_while_placeholder_2)
%backward_lstm_379_while_placeholder_3?
;backward_lstm_379_while_backward_lstm_379_strided_slice_1_0{
wbackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0Z
Gbackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0:	»\
Ibackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0:	2»W
Hbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0:	»$
 backward_lstm_379_while_identity&
"backward_lstm_379_while_identity_1&
"backward_lstm_379_while_identity_2&
"backward_lstm_379_while_identity_3&
"backward_lstm_379_while_identity_4&
"backward_lstm_379_while_identity_5=
9backward_lstm_379_while_backward_lstm_379_strided_slice_1y
ubackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensorX
Ebackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource:	»Z
Gbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource:	2»U
Fbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource:	»ИҐ=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpҐ<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOpҐ>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpз
Ibackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2K
Ibackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shape»
;backward_lstm_379/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_379_while_placeholderRbackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02=
;backward_lstm_379/while/TensorArrayV2Read/TensorListGetItemЕ
<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOpGbackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02>
<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp•
-backward_lstm_379/while/lstm_cell_1139/MatMulMatMulBbackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem:item:0Dbackward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-backward_lstm_379/while/lstm_cell_1139/MatMulЛ
>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOpIbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02@
>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpО
/backward_lstm_379/while/lstm_cell_1139/MatMul_1MatMul%backward_lstm_379_while_placeholder_2Fbackward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»21
/backward_lstm_379/while/lstm_cell_1139/MatMul_1И
*backward_lstm_379/while/lstm_cell_1139/addAddV27backward_lstm_379/while/lstm_cell_1139/MatMul:product:09backward_lstm_379/while/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2,
*backward_lstm_379/while/lstm_cell_1139/addД
=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOpHbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02?
=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpХ
.backward_lstm_379/while/lstm_cell_1139/BiasAddBiasAdd.backward_lstm_379/while/lstm_cell_1139/add:z:0Ebackward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.backward_lstm_379/while/lstm_cell_1139/BiasAdd≤
6backward_lstm_379/while/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6backward_lstm_379/while/lstm_cell_1139/split/split_dimџ
,backward_lstm_379/while/lstm_cell_1139/splitSplit?backward_lstm_379/while/lstm_cell_1139/split/split_dim:output:07backward_lstm_379/while/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2.
,backward_lstm_379/while/lstm_cell_1139/split‘
.backward_lstm_379/while/lstm_cell_1139/SigmoidSigmoid5backward_lstm_379/while/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€220
.backward_lstm_379/while/lstm_cell_1139/SigmoidЎ
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_1Sigmoid5backward_lstm_379/while/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_1о
*backward_lstm_379/while/lstm_cell_1139/mulMul4backward_lstm_379/while/lstm_cell_1139/Sigmoid_1:y:0%backward_lstm_379_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_379/while/lstm_cell_1139/mulЋ
+backward_lstm_379/while/lstm_cell_1139/ReluRelu5backward_lstm_379/while/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22-
+backward_lstm_379/while/lstm_cell_1139/ReluД
,backward_lstm_379/while/lstm_cell_1139/mul_1Mul2backward_lstm_379/while/lstm_cell_1139/Sigmoid:y:09backward_lstm_379/while/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_379/while/lstm_cell_1139/mul_1щ
,backward_lstm_379/while/lstm_cell_1139/add_1AddV2.backward_lstm_379/while/lstm_cell_1139/mul:z:00backward_lstm_379/while/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_379/while/lstm_cell_1139/add_1Ў
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_2Sigmoid5backward_lstm_379/while/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_2 
-backward_lstm_379/while/lstm_cell_1139/Relu_1Relu0backward_lstm_379/while/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-backward_lstm_379/while/lstm_cell_1139/Relu_1И
,backward_lstm_379/while/lstm_cell_1139/mul_2Mul4backward_lstm_379/while/lstm_cell_1139/Sigmoid_2:y:0;backward_lstm_379/while/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_379/while/lstm_cell_1139/mul_2Љ
<backward_lstm_379/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_379_while_placeholder_1#backward_lstm_379_while_placeholder0backward_lstm_379/while/lstm_cell_1139/mul_2:z:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_379/while/TensorArrayV2Write/TensorListSetItemА
backward_lstm_379/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_379/while/add/y±
backward_lstm_379/while/addAddV2#backward_lstm_379_while_placeholder&backward_lstm_379/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/while/addД
backward_lstm_379/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_379/while/add_1/y–
backward_lstm_379/while/add_1AddV2<backward_lstm_379_while_backward_lstm_379_while_loop_counter(backward_lstm_379/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/while/add_1≥
 backward_lstm_379/while/IdentityIdentity!backward_lstm_379/while/add_1:z:0^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_379/while/IdentityЎ
"backward_lstm_379/while/Identity_1IdentityBbackward_lstm_379_while_backward_lstm_379_while_maximum_iterations^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_379/while/Identity_1µ
"backward_lstm_379/while/Identity_2Identitybackward_lstm_379/while/add:z:0^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_379/while/Identity_2в
"backward_lstm_379/while/Identity_3IdentityLbackward_lstm_379/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_379/while/Identity_3„
"backward_lstm_379/while/Identity_4Identity0backward_lstm_379/while/lstm_cell_1139/mul_2:z:0^backward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_379/while/Identity_4„
"backward_lstm_379/while/Identity_5Identity0backward_lstm_379/while/lstm_cell_1139/add_1:z:0^backward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_379/while/Identity_5Њ
backward_lstm_379/while/NoOpNoOp>^backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp=^backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp?^backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_379/while/NoOp"x
9backward_lstm_379_while_backward_lstm_379_strided_slice_1;backward_lstm_379_while_backward_lstm_379_strided_slice_1_0"M
 backward_lstm_379_while_identity)backward_lstm_379/while/Identity:output:0"Q
"backward_lstm_379_while_identity_1+backward_lstm_379/while/Identity_1:output:0"Q
"backward_lstm_379_while_identity_2+backward_lstm_379/while/Identity_2:output:0"Q
"backward_lstm_379_while_identity_3+backward_lstm_379/while/Identity_3:output:0"Q
"backward_lstm_379_while_identity_4+backward_lstm_379/while/Identity_4:output:0"Q
"backward_lstm_379_while_identity_5+backward_lstm_379/while/Identity_5:output:0"Т
Fbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resourceHbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0"Ф
Gbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resourceIbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0"Р
Ebackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resourceGbackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0"р
ubackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2~
=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp2|
<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp2А
>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
я
Ќ
while_cond_42662565
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_42662565___redundant_placeholder06
2while_while_cond_42662565___redundant_placeholder16
2while_while_cond_42662565___redundant_placeholder26
2while_while_cond_42662565___redundant_placeholder3
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
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:
–]
і
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_42662650
inputs_0@
-lstm_cell_1138_matmul_readvariableop_resource:	»B
/lstm_cell_1138_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1138_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1138/BiasAdd/ReadVariableOpҐ$lstm_cell_1138/MatMul/ReadVariableOpҐ&lstm_cell_1138/MatMul_1/ReadVariableOpҐwhileF
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€22
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
B :и2
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
zeros_1/packed/1Й
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
:€€€€€€€€€22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЕ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2ї
$lstm_cell_1138/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1138_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1138/MatMul/ReadVariableOp≥
lstm_cell_1138/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/MatMulЅ
&lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1138_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1138/MatMul_1/ReadVariableOpѓ
lstm_cell_1138/MatMul_1MatMulzeros:output:0.lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/MatMul_1®
lstm_cell_1138/addAddV2lstm_cell_1138/MatMul:product:0!lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/addЇ
%lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1138_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1138/BiasAdd/ReadVariableOpµ
lstm_cell_1138/BiasAddBiasAddlstm_cell_1138/add:z:0-lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/BiasAddВ
lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1138/split/split_dimы
lstm_cell_1138/splitSplit'lstm_cell_1138/split/split_dim:output:0lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1138/splitМ
lstm_cell_1138/SigmoidSigmoidlstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/SigmoidР
lstm_cell_1138/Sigmoid_1Sigmoidlstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Sigmoid_1С
lstm_cell_1138/mulMullstm_cell_1138/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/mulГ
lstm_cell_1138/ReluRelulstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Relu§
lstm_cell_1138/mul_1Mullstm_cell_1138/Sigmoid:y:0!lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/mul_1Щ
lstm_cell_1138/add_1AddV2lstm_cell_1138/mul:z:0lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/add_1Р
lstm_cell_1138/Sigmoid_2Sigmoidlstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Sigmoid_2В
lstm_cell_1138/Relu_1Relulstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Relu_1®
lstm_cell_1138/mul_2Mullstm_cell_1138/Sigmoid_2:y:0#lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterХ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1138_matmul_readvariableop_resource/lstm_cell_1138_matmul_1_readvariableop_resource.lstm_cell_1138_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42662566*
condR
while_cond_42662565*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
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
:€€€€€€€€€22

Identityќ
NoOpNoOp&^lstm_cell_1138/BiasAdd/ReadVariableOp%^lstm_cell_1138/MatMul/ReadVariableOp'^lstm_cell_1138/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1138/BiasAdd/ReadVariableOp%lstm_cell_1138/BiasAdd/ReadVariableOp2L
$lstm_cell_1138/MatMul/ReadVariableOp$lstm_cell_1138/MatMul/ReadVariableOp2P
&lstm_cell_1138/MatMul_1/ReadVariableOp&lstm_cell_1138/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
ч
И
L__inference_lstm_cell_1138_layer_call_and_return_conditional_losses_42658059

inputs

states
states_11
matmul_readvariableop_resource:	»3
 matmul_1_readvariableop_resource:	2».
biasadd_readvariableop_resource:	»
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	»*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_2Щ
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
?:€€€€€€€€€:€€€€€€€€€2:€€€€€€€€€2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_namestates
∞Њ
ы
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_42660274

inputs
inputs_1	Q
>forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource:	»S
@forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource:	2»N
?forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource:	»R
?backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resource:	»T
Abackward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource:	2»O
@backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource:	»
identityИҐ7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpҐ6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpҐ8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpҐbackward_lstm_379/whileҐ6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpҐ5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpҐ7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpҐforward_lstm_379/whileЧ
%forward_lstm_379/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_379/RaggedToTensor/zerosЩ
%forward_lstm_379/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2'
%forward_lstm_379/RaggedToTensor/ConstЩ
4forward_lstm_379/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_379/RaggedToTensor/Const:output:0inputs.forward_lstm_379/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_379/RaggedToTensor/RaggedTensorToTensorƒ
;forward_lstm_379/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack»
=forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1»
=forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2©
5forward_lstm_379/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_379/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask27
5forward_lstm_379/RaggedNestedRowLengths/strided_slice»
=forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack’
?forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2A
?forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1ћ
?forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2µ
7forward_lstm_379/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask29
7forward_lstm_379/RaggedNestedRowLengths/strided_slice_1С
+forward_lstm_379/RaggedNestedRowLengths/subSub>forward_lstm_379/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_379/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2-
+forward_lstm_379/RaggedNestedRowLengths/sub§
forward_lstm_379/CastCast/forward_lstm_379/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
forward_lstm_379/CastЭ
forward_lstm_379/ShapeShape=forward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_379/ShapeЦ
$forward_lstm_379/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_379/strided_slice/stackЪ
&forward_lstm_379/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_379/strided_slice/stack_1Ъ
&forward_lstm_379/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_379/strided_slice/stack_2»
forward_lstm_379/strided_sliceStridedSliceforward_lstm_379/Shape:output:0-forward_lstm_379/strided_slice/stack:output:0/forward_lstm_379/strided_slice/stack_1:output:0/forward_lstm_379/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_379/strided_slice~
forward_lstm_379/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_379/zeros/mul/y∞
forward_lstm_379/zeros/mulMul'forward_lstm_379/strided_slice:output:0%forward_lstm_379/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros/mulБ
forward_lstm_379/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
forward_lstm_379/zeros/Less/yЂ
forward_lstm_379/zeros/LessLessforward_lstm_379/zeros/mul:z:0&forward_lstm_379/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros/LessД
forward_lstm_379/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_379/zeros/packed/1«
forward_lstm_379/zeros/packedPack'forward_lstm_379/strided_slice:output:0(forward_lstm_379/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_379/zeros/packedЕ
forward_lstm_379/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_379/zeros/Constє
forward_lstm_379/zerosFill&forward_lstm_379/zeros/packed:output:0%forward_lstm_379/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/zerosВ
forward_lstm_379/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_379/zeros_1/mul/yґ
forward_lstm_379/zeros_1/mulMul'forward_lstm_379/strided_slice:output:0'forward_lstm_379/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros_1/mulЕ
forward_lstm_379/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2!
forward_lstm_379/zeros_1/Less/y≥
forward_lstm_379/zeros_1/LessLess forward_lstm_379/zeros_1/mul:z:0(forward_lstm_379/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros_1/LessИ
!forward_lstm_379/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_379/zeros_1/packed/1Ќ
forward_lstm_379/zeros_1/packedPack'forward_lstm_379/strided_slice:output:0*forward_lstm_379/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_379/zeros_1/packedЙ
forward_lstm_379/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_379/zeros_1/ConstЅ
forward_lstm_379/zeros_1Fill(forward_lstm_379/zeros_1/packed:output:0'forward_lstm_379/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/zeros_1Ч
forward_lstm_379/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_379/transpose/permн
forward_lstm_379/transpose	Transpose=forward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_379/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
forward_lstm_379/transposeВ
forward_lstm_379/Shape_1Shapeforward_lstm_379/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_379/Shape_1Ъ
&forward_lstm_379/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_379/strided_slice_1/stackЮ
(forward_lstm_379/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_1/stack_1Ю
(forward_lstm_379/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_1/stack_2‘
 forward_lstm_379/strided_slice_1StridedSlice!forward_lstm_379/Shape_1:output:0/forward_lstm_379/strided_slice_1/stack:output:01forward_lstm_379/strided_slice_1/stack_1:output:01forward_lstm_379/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_379/strided_slice_1І
,forward_lstm_379/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,forward_lstm_379/TensorArrayV2/element_shapeц
forward_lstm_379/TensorArrayV2TensorListReserve5forward_lstm_379/TensorArrayV2/element_shape:output:0)forward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_379/TensorArrayV2б
Fforward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2H
Fforward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeЉ
8forward_lstm_379/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_379/transpose:y:0Oforward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_379/TensorArrayUnstack/TensorListFromTensorЪ
&forward_lstm_379/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_379/strided_slice_2/stackЮ
(forward_lstm_379/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_2/stack_1Ю
(forward_lstm_379/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_2/stack_2в
 forward_lstm_379/strided_slice_2StridedSliceforward_lstm_379/transpose:y:0/forward_lstm_379/strided_slice_2/stack:output:01forward_lstm_379/strided_slice_2/stack_1:output:01forward_lstm_379/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2"
 forward_lstm_379/strided_slice_2о
5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOp>forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype027
5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpч
&forward_lstm_379/lstm_cell_1138/MatMulMatMul)forward_lstm_379/strided_slice_2:output:0=forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2(
&forward_lstm_379/lstm_cell_1138/MatMulф
7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOp@forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype029
7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpу
(forward_lstm_379/lstm_cell_1138/MatMul_1MatMulforward_lstm_379/zeros:output:0?forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(forward_lstm_379/lstm_cell_1138/MatMul_1м
#forward_lstm_379/lstm_cell_1138/addAddV20forward_lstm_379/lstm_cell_1138/MatMul:product:02forward_lstm_379/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2%
#forward_lstm_379/lstm_cell_1138/addн
6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOp?forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype028
6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpщ
'forward_lstm_379/lstm_cell_1138/BiasAddBiasAdd'forward_lstm_379/lstm_cell_1138/add:z:0>forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'forward_lstm_379/lstm_cell_1138/BiasAdd§
/forward_lstm_379/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/forward_lstm_379/lstm_cell_1138/split/split_dimњ
%forward_lstm_379/lstm_cell_1138/splitSplit8forward_lstm_379/lstm_cell_1138/split/split_dim:output:00forward_lstm_379/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2'
%forward_lstm_379/lstm_cell_1138/splitњ
'forward_lstm_379/lstm_cell_1138/SigmoidSigmoid.forward_lstm_379/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'forward_lstm_379/lstm_cell_1138/Sigmoid√
)forward_lstm_379/lstm_cell_1138/Sigmoid_1Sigmoid.forward_lstm_379/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_379/lstm_cell_1138/Sigmoid_1’
#forward_lstm_379/lstm_cell_1138/mulMul-forward_lstm_379/lstm_cell_1138/Sigmoid_1:y:0!forward_lstm_379/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22%
#forward_lstm_379/lstm_cell_1138/mulґ
$forward_lstm_379/lstm_cell_1138/ReluRelu.forward_lstm_379/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22&
$forward_lstm_379/lstm_cell_1138/Reluи
%forward_lstm_379/lstm_cell_1138/mul_1Mul+forward_lstm_379/lstm_cell_1138/Sigmoid:y:02forward_lstm_379/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_379/lstm_cell_1138/mul_1Ё
%forward_lstm_379/lstm_cell_1138/add_1AddV2'forward_lstm_379/lstm_cell_1138/mul:z:0)forward_lstm_379/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_379/lstm_cell_1138/add_1√
)forward_lstm_379/lstm_cell_1138/Sigmoid_2Sigmoid.forward_lstm_379/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_379/lstm_cell_1138/Sigmoid_2µ
&forward_lstm_379/lstm_cell_1138/Relu_1Relu)forward_lstm_379/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&forward_lstm_379/lstm_cell_1138/Relu_1м
%forward_lstm_379/lstm_cell_1138/mul_2Mul-forward_lstm_379/lstm_cell_1138/Sigmoid_2:y:04forward_lstm_379/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_379/lstm_cell_1138/mul_2±
.forward_lstm_379/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   20
.forward_lstm_379/TensorArrayV2_1/element_shapeь
 forward_lstm_379/TensorArrayV2_1TensorListReserve7forward_lstm_379/TensorArrayV2_1/element_shape:output:0)forward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_379/TensorArrayV2_1p
forward_lstm_379/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_379/time§
forward_lstm_379/zeros_like	ZerosLike)forward_lstm_379/lstm_cell_1138/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/zeros_like°
)forward_lstm_379/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2+
)forward_lstm_379/while/maximum_iterationsМ
#forward_lstm_379/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_379/while/loop_counterЦ	
forward_lstm_379/whileWhile,forward_lstm_379/while/loop_counter:output:02forward_lstm_379/while/maximum_iterations:output:0forward_lstm_379/time:output:0)forward_lstm_379/TensorArrayV2_1:handle:0forward_lstm_379/zeros_like:y:0forward_lstm_379/zeros:output:0!forward_lstm_379/zeros_1:output:0)forward_lstm_379/strided_slice_1:output:0Hforward_lstm_379/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_379/Cast:y:0>forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource@forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource?forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *0
body(R&
$forward_lstm_379_while_body_42659998*0
cond(R&
$forward_lstm_379_while_cond_42659997*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
forward_lstm_379/while„
Aforward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2C
Aforward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeµ
3forward_lstm_379/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_379/while:output:3Jforward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype025
3forward_lstm_379/TensorArrayV2Stack/TensorListStack£
&forward_lstm_379/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2(
&forward_lstm_379/strided_slice_3/stackЮ
(forward_lstm_379/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_379/strided_slice_3/stack_1Ю
(forward_lstm_379/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_3/stack_2А
 forward_lstm_379/strided_slice_3StridedSlice<forward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_379/strided_slice_3/stack:output:01forward_lstm_379/strided_slice_3/stack_1:output:01forward_lstm_379/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2"
 forward_lstm_379/strided_slice_3Ы
!forward_lstm_379/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_379/transpose_1/permт
forward_lstm_379/transpose_1	Transpose<forward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_379/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
forward_lstm_379/transpose_1И
forward_lstm_379/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_379/runtimeЩ
&backward_lstm_379/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_379/RaggedToTensor/zerosЫ
&backward_lstm_379/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2(
&backward_lstm_379/RaggedToTensor/ConstЭ
5backward_lstm_379/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_379/RaggedToTensor/Const:output:0inputs/backward_lstm_379/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_379/RaggedToTensor/RaggedTensorToTensor∆
<backward_lstm_379/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack 
>backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1 
>backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2Ѓ
6backward_lstm_379/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_379/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask28
6backward_lstm_379/RaggedNestedRowLengths/strided_slice 
>backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack„
@backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2B
@backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1ќ
@backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2Ї
8backward_lstm_379/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask2:
8backward_lstm_379/RaggedNestedRowLengths/strided_slice_1Х
,backward_lstm_379/RaggedNestedRowLengths/subSub?backward_lstm_379/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_379/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2.
,backward_lstm_379/RaggedNestedRowLengths/subІ
backward_lstm_379/CastCast0backward_lstm_379/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
backward_lstm_379/Cast†
backward_lstm_379/ShapeShape>backward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_379/ShapeШ
%backward_lstm_379/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_379/strided_slice/stackЬ
'backward_lstm_379/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_379/strided_slice/stack_1Ь
'backward_lstm_379/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_379/strided_slice/stack_2ќ
backward_lstm_379/strided_sliceStridedSlice backward_lstm_379/Shape:output:0.backward_lstm_379/strided_slice/stack:output:00backward_lstm_379/strided_slice/stack_1:output:00backward_lstm_379/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_379/strided_sliceА
backward_lstm_379/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_379/zeros/mul/yі
backward_lstm_379/zeros/mulMul(backward_lstm_379/strided_slice:output:0&backward_lstm_379/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/zeros/mulГ
backward_lstm_379/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
backward_lstm_379/zeros/Less/yѓ
backward_lstm_379/zeros/LessLessbackward_lstm_379/zeros/mul:z:0'backward_lstm_379/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/zeros/LessЖ
 backward_lstm_379/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_379/zeros/packed/1Ћ
backward_lstm_379/zeros/packedPack(backward_lstm_379/strided_slice:output:0)backward_lstm_379/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_379/zeros/packedЗ
backward_lstm_379/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_379/zeros/Constљ
backward_lstm_379/zerosFill'backward_lstm_379/zeros/packed:output:0&backward_lstm_379/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_379/zerosД
backward_lstm_379/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_379/zeros_1/mul/yЇ
backward_lstm_379/zeros_1/mulMul(backward_lstm_379/strided_slice:output:0(backward_lstm_379/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/zeros_1/mulЗ
 backward_lstm_379/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 backward_lstm_379/zeros_1/Less/yЈ
backward_lstm_379/zeros_1/LessLess!backward_lstm_379/zeros_1/mul:z:0)backward_lstm_379/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_379/zeros_1/LessК
"backward_lstm_379/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_379/zeros_1/packed/1—
 backward_lstm_379/zeros_1/packedPack(backward_lstm_379/strided_slice:output:0+backward_lstm_379/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_379/zeros_1/packedЛ
backward_lstm_379/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_379/zeros_1/Const≈
backward_lstm_379/zeros_1Fill)backward_lstm_379/zeros_1/packed:output:0(backward_lstm_379/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_379/zeros_1Щ
 backward_lstm_379/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_379/transpose/permс
backward_lstm_379/transpose	Transpose>backward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_379/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_379/transposeЕ
backward_lstm_379/Shape_1Shapebackward_lstm_379/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_379/Shape_1Ь
'backward_lstm_379/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_379/strided_slice_1/stack†
)backward_lstm_379/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_1/stack_1†
)backward_lstm_379/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_1/stack_2Џ
!backward_lstm_379/strided_slice_1StridedSlice"backward_lstm_379/Shape_1:output:00backward_lstm_379/strided_slice_1/stack:output:02backward_lstm_379/strided_slice_1/stack_1:output:02backward_lstm_379/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_379/strided_slice_1©
-backward_lstm_379/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-backward_lstm_379/TensorArrayV2/element_shapeъ
backward_lstm_379/TensorArrayV2TensorListReserve6backward_lstm_379/TensorArrayV2/element_shape:output:0*backward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_379/TensorArrayV2О
 backward_lstm_379/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_379/ReverseV2/axis“
backward_lstm_379/ReverseV2	ReverseV2backward_lstm_379/transpose:y:0)backward_lstm_379/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_379/ReverseV2г
Gbackward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2I
Gbackward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape≈
9backward_lstm_379/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_379/ReverseV2:output:0Pbackward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_379/TensorArrayUnstack/TensorListFromTensorЬ
'backward_lstm_379/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_379/strided_slice_2/stack†
)backward_lstm_379/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_2/stack_1†
)backward_lstm_379/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_2/stack_2и
!backward_lstm_379/strided_slice_2StridedSlicebackward_lstm_379/transpose:y:00backward_lstm_379/strided_slice_2/stack:output:02backward_lstm_379/strided_slice_2/stack_1:output:02backward_lstm_379/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2#
!backward_lstm_379/strided_slice_2с
6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOp?backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype028
6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpы
'backward_lstm_379/lstm_cell_1139/MatMulMatMul*backward_lstm_379/strided_slice_2:output:0>backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'backward_lstm_379/lstm_cell_1139/MatMulч
8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOpAbackward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02:
8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpч
)backward_lstm_379/lstm_cell_1139/MatMul_1MatMul backward_lstm_379/zeros:output:0@backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)backward_lstm_379/lstm_cell_1139/MatMul_1р
$backward_lstm_379/lstm_cell_1139/addAddV21backward_lstm_379/lstm_cell_1139/MatMul:product:03backward_lstm_379/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2&
$backward_lstm_379/lstm_cell_1139/addр
7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOp@backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype029
7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpэ
(backward_lstm_379/lstm_cell_1139/BiasAddBiasAdd(backward_lstm_379/lstm_cell_1139/add:z:0?backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(backward_lstm_379/lstm_cell_1139/BiasAdd¶
0backward_lstm_379/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0backward_lstm_379/lstm_cell_1139/split/split_dim√
&backward_lstm_379/lstm_cell_1139/splitSplit9backward_lstm_379/lstm_cell_1139/split/split_dim:output:01backward_lstm_379/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2(
&backward_lstm_379/lstm_cell_1139/split¬
(backward_lstm_379/lstm_cell_1139/SigmoidSigmoid/backward_lstm_379/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(backward_lstm_379/lstm_cell_1139/Sigmoid∆
*backward_lstm_379/lstm_cell_1139/Sigmoid_1Sigmoid/backward_lstm_379/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_379/lstm_cell_1139/Sigmoid_1ў
$backward_lstm_379/lstm_cell_1139/mulMul.backward_lstm_379/lstm_cell_1139/Sigmoid_1:y:0"backward_lstm_379/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$backward_lstm_379/lstm_cell_1139/mulє
%backward_lstm_379/lstm_cell_1139/ReluRelu/backward_lstm_379/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22'
%backward_lstm_379/lstm_cell_1139/Reluм
&backward_lstm_379/lstm_cell_1139/mul_1Mul,backward_lstm_379/lstm_cell_1139/Sigmoid:y:03backward_lstm_379/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_379/lstm_cell_1139/mul_1б
&backward_lstm_379/lstm_cell_1139/add_1AddV2(backward_lstm_379/lstm_cell_1139/mul:z:0*backward_lstm_379/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_379/lstm_cell_1139/add_1∆
*backward_lstm_379/lstm_cell_1139/Sigmoid_2Sigmoid/backward_lstm_379/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_379/lstm_cell_1139/Sigmoid_2Є
'backward_lstm_379/lstm_cell_1139/Relu_1Relu*backward_lstm_379/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'backward_lstm_379/lstm_cell_1139/Relu_1р
&backward_lstm_379/lstm_cell_1139/mul_2Mul.backward_lstm_379/lstm_cell_1139/Sigmoid_2:y:05backward_lstm_379/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_379/lstm_cell_1139/mul_2≥
/backward_lstm_379/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   21
/backward_lstm_379/TensorArrayV2_1/element_shapeА
!backward_lstm_379/TensorArrayV2_1TensorListReserve8backward_lstm_379/TensorArrayV2_1/element_shape:output:0*backward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_379/TensorArrayV2_1r
backward_lstm_379/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_379/timeФ
'backward_lstm_379/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_379/Max/reduction_indices§
backward_lstm_379/MaxMaxbackward_lstm_379/Cast:y:00backward_lstm_379/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/Maxt
backward_lstm_379/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_379/sub/yШ
backward_lstm_379/subSubbackward_lstm_379/Max:output:0 backward_lstm_379/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/subЮ
backward_lstm_379/Sub_1Subbackward_lstm_379/sub:z:0backward_lstm_379/Cast:y:0*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_379/Sub_1І
backward_lstm_379/zeros_like	ZerosLike*backward_lstm_379/lstm_cell_1139/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_379/zeros_like£
*backward_lstm_379/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2,
*backward_lstm_379/while/maximum_iterationsО
$backward_lstm_379/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_379/while/loop_counter®	
backward_lstm_379/whileWhile-backward_lstm_379/while/loop_counter:output:03backward_lstm_379/while/maximum_iterations:output:0backward_lstm_379/time:output:0*backward_lstm_379/TensorArrayV2_1:handle:0 backward_lstm_379/zeros_like:y:0 backward_lstm_379/zeros:output:0"backward_lstm_379/zeros_1:output:0*backward_lstm_379/strided_slice_1:output:0Ibackward_lstm_379/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_379/Sub_1:z:0?backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resourceAbackward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource@backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *1
body)R'
%backward_lstm_379_while_body_42660177*1
cond)R'
%backward_lstm_379_while_cond_42660176*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
backward_lstm_379/whileў
Bbackward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2D
Bbackward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeє
4backward_lstm_379/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_379/while:output:3Kbackward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype026
4backward_lstm_379/TensorArrayV2Stack/TensorListStack•
'backward_lstm_379/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2)
'backward_lstm_379/strided_slice_3/stack†
)backward_lstm_379/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_379/strided_slice_3/stack_1†
)backward_lstm_379/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_3/stack_2Ж
!backward_lstm_379/strided_slice_3StridedSlice=backward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_379/strided_slice_3/stack:output:02backward_lstm_379/strided_slice_3/stack_1:output:02backward_lstm_379/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2#
!backward_lstm_379/strided_slice_3Э
"backward_lstm_379/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_379/transpose_1/permц
backward_lstm_379/transpose_1	Transpose=backward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_379/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
backward_lstm_379/transpose_1К
backward_lstm_379/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_379/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisƒ
concatConcatV2)forward_lstm_379/strided_slice_3:output:0*backward_lstm_379/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d2

IdentityЏ
NoOpNoOp8^backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp7^backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp9^backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp^backward_lstm_379/while7^forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp6^forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp8^forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp^forward_lstm_379/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€:€€€€€€€€€: : : : : : 2r
7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp2p
6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp2t
8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp22
backward_lstm_379/whilebackward_lstm_379/while2p
6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp2n
5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp2r
7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp20
forward_lstm_379/whileforward_lstm_379/while:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ѓ

•
4__inference_bidirectional_379_layer_call_fn_42660964

inputs
inputs_1	
unknown:	»
	unknown_0:	2»
	unknown_1:	»
	unknown_2:	»
	unknown_3:	2»
	unknown_4:	»
identityИҐStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_426607142
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€:€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ѕ_
µ
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_42663149
inputs_0@
-lstm_cell_1139_matmul_readvariableop_resource:	»B
/lstm_cell_1139_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1139_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1139/BiasAdd/ReadVariableOpҐ$lstm_cell_1139/MatMul/ReadVariableOpҐ&lstm_cell_1139/MatMul_1/ReadVariableOpҐwhileF
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€22
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
B :и2
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
zeros_1/packed/1Й
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
:€€€€€€€€€22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЕ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
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
ReverseV2/axisК
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
	ReverseV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeэ
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
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2ї
$lstm_cell_1139/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1139_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1139/MatMul/ReadVariableOp≥
lstm_cell_1139/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/MatMulЅ
&lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1139_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1139/MatMul_1/ReadVariableOpѓ
lstm_cell_1139/MatMul_1MatMulzeros:output:0.lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/MatMul_1®
lstm_cell_1139/addAddV2lstm_cell_1139/MatMul:product:0!lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/addЇ
%lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1139_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1139/BiasAdd/ReadVariableOpµ
lstm_cell_1139/BiasAddBiasAddlstm_cell_1139/add:z:0-lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/BiasAddВ
lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1139/split/split_dimы
lstm_cell_1139/splitSplit'lstm_cell_1139/split/split_dim:output:0lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1139/splitМ
lstm_cell_1139/SigmoidSigmoidlstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/SigmoidР
lstm_cell_1139/Sigmoid_1Sigmoidlstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Sigmoid_1С
lstm_cell_1139/mulMullstm_cell_1139/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/mulГ
lstm_cell_1139/ReluRelulstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Relu§
lstm_cell_1139/mul_1Mullstm_cell_1139/Sigmoid:y:0!lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/mul_1Щ
lstm_cell_1139/add_1AddV2lstm_cell_1139/mul:z:0lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/add_1Р
lstm_cell_1139/Sigmoid_2Sigmoidlstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Sigmoid_2В
lstm_cell_1139/Relu_1Relulstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Relu_1®
lstm_cell_1139/mul_2Mullstm_cell_1139/Sigmoid_2:y:0#lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterХ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1139_matmul_readvariableop_resource/lstm_cell_1139_matmul_1_readvariableop_resource.lstm_cell_1139_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42663065*
condR
while_cond_42663064*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
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
:€€€€€€€€€22

Identityќ
NoOpNoOp&^lstm_cell_1139/BiasAdd/ReadVariableOp%^lstm_cell_1139/MatMul/ReadVariableOp'^lstm_cell_1139/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1139/BiasAdd/ReadVariableOp%lstm_cell_1139/BiasAdd/ReadVariableOp2L
$lstm_cell_1139/MatMul/ReadVariableOp$lstm_cell_1139/MatMul/ReadVariableOp2P
&lstm_cell_1139/MatMul_1/ReadVariableOp&lstm_cell_1139/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
Є
£
L__inference_sequential_379_layer_call_and_return_conditional_losses_42660777

inputs
inputs_1	-
bidirectional_379_42660758:	»-
bidirectional_379_42660760:	2»)
bidirectional_379_42660762:	»-
bidirectional_379_42660764:	»-
bidirectional_379_42660766:	2»)
bidirectional_379_42660768:	»$
dense_379_42660771:d 
dense_379_42660773:
identityИҐ)bidirectional_379/StatefulPartitionedCallҐ!dense_379/StatefulPartitionedCall 
)bidirectional_379/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_379_42660758bidirectional_379_42660760bidirectional_379_42660762bidirectional_379_42660764bidirectional_379_42660766bidirectional_379_42660768*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_426607142+
)bidirectional_379/StatefulPartitionedCallЋ
!dense_379/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_379/StatefulPartitionedCall:output:0dense_379_42660771dense_379_42660773*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dense_379_layer_call_and_return_conditional_losses_426602992#
!dense_379/StatefulPartitionedCallЕ
IdentityIdentity*dense_379/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityЮ
NoOpNoOp*^bidirectional_379/StatefulPartitionedCall"^dense_379/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€:€€€€€€€€€: : : : : : : : 2V
)bidirectional_379/StatefulPartitionedCall)bidirectional_379/StatefulPartitionedCall2F
!dense_379/StatefulPartitionedCall!dense_379/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
в
Ѕ
4__inference_backward_lstm_379_layer_call_fn_42662996

inputs
unknown:	»
	unknown_0:	2»
	unknown_1:	»
identityИҐStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_426596152
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
л	
Щ
4__inference_bidirectional_379_layer_call_fn_42660911
inputs_0
unknown:	»
	unknown_0:	2»
	unknown_1:	»
	unknown_2:	»
	unknown_3:	2»
	unknown_4:	»
identityИҐStatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_426594342
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
∆@
д
while_body_42662566
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1138_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1138_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1138_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1138_matmul_readvariableop_resource:	»H
5while_lstm_cell_1138_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1138_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1138/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1138/MatMul/ReadVariableOpҐ,while/lstm_cell_1138/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemѕ
*while/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1138_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1138/MatMul/ReadVariableOpЁ
while/lstm_cell_1138/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/MatMul’
,while/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1138_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1138/MatMul_1/ReadVariableOp∆
while/lstm_cell_1138/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/MatMul_1ј
while/lstm_cell_1138/addAddV2%while/lstm_cell_1138/MatMul:product:0'while/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/addќ
+while/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1138_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1138/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1138/BiasAddBiasAddwhile/lstm_cell_1138/add:z:03while/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/BiasAddО
$while/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1138/split/split_dimУ
while/lstm_cell_1138/splitSplit-while/lstm_cell_1138/split/split_dim:output:0%while/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1138/splitЮ
while/lstm_cell_1138/SigmoidSigmoid#while/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/SigmoidҐ
while/lstm_cell_1138/Sigmoid_1Sigmoid#while/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1138/Sigmoid_1¶
while/lstm_cell_1138/mulMul"while/lstm_cell_1138/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/mulХ
while/lstm_cell_1138/ReluRelu#while/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/ReluЉ
while/lstm_cell_1138/mul_1Mul while/lstm_cell_1138/Sigmoid:y:0'while/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/mul_1±
while/lstm_cell_1138/add_1AddV2while/lstm_cell_1138/mul:z:0while/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/add_1Ґ
while/lstm_cell_1138/Sigmoid_2Sigmoid#while/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1138/Sigmoid_2Ф
while/lstm_cell_1138/Relu_1Reluwhile/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/Relu_1ј
while/lstm_cell_1138/mul_2Mul"while/lstm_cell_1138/Sigmoid_2:y:0)while/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1138/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3П
while/Identity_4Identitywhile/lstm_cell_1138/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1138/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1138/BiasAdd/ReadVariableOp+^while/lstm_cell_1138/MatMul/ReadVariableOp-^while/lstm_cell_1138/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"n
4while_lstm_cell_1138_biasadd_readvariableop_resource6while_lstm_cell_1138_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1138_matmul_1_readvariableop_resource7while_lstm_cell_1138_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1138_matmul_readvariableop_resource5while_lstm_cell_1138_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1138/BiasAdd/ReadVariableOp+while/lstm_cell_1138/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1138/MatMul/ReadVariableOp*while/lstm_cell_1138/MatMul/ReadVariableOp2\
,while/lstm_cell_1138/MatMul_1/ReadVariableOp,while/lstm_cell_1138/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
€
К
L__inference_lstm_cell_1138_layer_call_and_return_conditional_losses_42663706

inputs
states_0
states_11
matmul_readvariableop_resource:	»3
 matmul_1_readvariableop_resource:	2».
biasadd_readvariableop_resource:	»
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	»*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_2Щ
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
?:€€€€€€€€€:€€€€€€€€€2:€€€€€€€€€2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€2
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€2
"
_user_specified_name
states/1
к
Љ
%backward_lstm_379_while_cond_42661828@
<backward_lstm_379_while_backward_lstm_379_while_loop_counterF
Bbackward_lstm_379_while_backward_lstm_379_while_maximum_iterations'
#backward_lstm_379_while_placeholder)
%backward_lstm_379_while_placeholder_1)
%backward_lstm_379_while_placeholder_2)
%backward_lstm_379_while_placeholder_3)
%backward_lstm_379_while_placeholder_4B
>backward_lstm_379_while_less_backward_lstm_379_strided_slice_1Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42661828___redundant_placeholder0Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42661828___redundant_placeholder1Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42661828___redundant_placeholder2Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42661828___redundant_placeholder3Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42661828___redundant_placeholder4$
 backward_lstm_379_while_identity
 
backward_lstm_379/while/LessLess#backward_lstm_379_while_placeholder>backward_lstm_379_while_less_backward_lstm_379_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_379/while/LessУ
 backward_lstm_379/while/IdentityIdentity backward_lstm_379/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_379/while/Identity"M
 backward_lstm_379_while_identity)backward_lstm_379/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
ѕ@
д
while_body_42659339
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1139_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1139_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1139_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1139_matmul_readvariableop_resource:	»H
5while_lstm_cell_1139_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1139_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1139/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1139/MatMul/ReadVariableOpҐ,while/lstm_cell_1139/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€29
7while/TensorArrayV2Read/TensorListGetItem/element_shape№
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemѕ
*while/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1139_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1139/MatMul/ReadVariableOpЁ
while/lstm_cell_1139/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/MatMul’
,while/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1139_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1139/MatMul_1/ReadVariableOp∆
while/lstm_cell_1139/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/MatMul_1ј
while/lstm_cell_1139/addAddV2%while/lstm_cell_1139/MatMul:product:0'while/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/addќ
+while/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1139_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1139/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1139/BiasAddBiasAddwhile/lstm_cell_1139/add:z:03while/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/BiasAddО
$while/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1139/split/split_dimУ
while/lstm_cell_1139/splitSplit-while/lstm_cell_1139/split/split_dim:output:0%while/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1139/splitЮ
while/lstm_cell_1139/SigmoidSigmoid#while/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/SigmoidҐ
while/lstm_cell_1139/Sigmoid_1Sigmoid#while/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1139/Sigmoid_1¶
while/lstm_cell_1139/mulMul"while/lstm_cell_1139/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/mulХ
while/lstm_cell_1139/ReluRelu#while/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/ReluЉ
while/lstm_cell_1139/mul_1Mul while/lstm_cell_1139/Sigmoid:y:0'while/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/mul_1±
while/lstm_cell_1139/add_1AddV2while/lstm_cell_1139/mul:z:0while/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/add_1Ґ
while/lstm_cell_1139/Sigmoid_2Sigmoid#while/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1139/Sigmoid_2Ф
while/lstm_cell_1139/Relu_1Reluwhile/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/Relu_1ј
while/lstm_cell_1139/mul_2Mul"while/lstm_cell_1139/Sigmoid_2:y:0)while/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1139/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3П
while/Identity_4Identitywhile/lstm_cell_1139/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1139/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1139/BiasAdd/ReadVariableOp+^while/lstm_cell_1139/MatMul/ReadVariableOp-^while/lstm_cell_1139/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"n
4while_lstm_cell_1139_biasadd_readvariableop_resource6while_lstm_cell_1139_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1139_matmul_1_readvariableop_resource7while_lstm_cell_1139_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1139_matmul_readvariableop_resource5while_lstm_cell_1139_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1139/BiasAdd/ReadVariableOp+while/lstm_cell_1139/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1139/MatMul/ReadVariableOp*while/lstm_cell_1139/MatMul/ReadVariableOp2\
,while/lstm_cell_1139/MatMul_1/ReadVariableOp,while/lstm_cell_1139/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
я
Ќ
while_cond_42662716
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_42662716___redundant_placeholder06
2while_while_cond_42662716___redundant_placeholder16
2while_while_cond_42662716___redundant_placeholder26
2while_while_cond_42662716___redundant_placeholder3
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
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:
ч
Щ
,__inference_dense_379_layer_call_fn_42662293

inputs
unknown:d
	unknown_0:
identityИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dense_379_layer_call_and_return_conditional_losses_426602992
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
Іg
н
%backward_lstm_379_while_body_42662187@
<backward_lstm_379_while_backward_lstm_379_while_loop_counterF
Bbackward_lstm_379_while_backward_lstm_379_while_maximum_iterations'
#backward_lstm_379_while_placeholder)
%backward_lstm_379_while_placeholder_1)
%backward_lstm_379_while_placeholder_2)
%backward_lstm_379_while_placeholder_3)
%backward_lstm_379_while_placeholder_4?
;backward_lstm_379_while_backward_lstm_379_strided_slice_1_0{
wbackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_379_while_less_backward_lstm_379_sub_1_0Z
Gbackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0:	»\
Ibackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0:	2»W
Hbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0:	»$
 backward_lstm_379_while_identity&
"backward_lstm_379_while_identity_1&
"backward_lstm_379_while_identity_2&
"backward_lstm_379_while_identity_3&
"backward_lstm_379_while_identity_4&
"backward_lstm_379_while_identity_5&
"backward_lstm_379_while_identity_6=
9backward_lstm_379_while_backward_lstm_379_strided_slice_1y
ubackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_379_while_less_backward_lstm_379_sub_1X
Ebackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource:	»Z
Gbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource:	2»U
Fbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource:	»ИҐ=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpҐ<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOpҐ>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpз
Ibackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2K
Ibackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeњ
;backward_lstm_379/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_379_while_placeholderRbackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02=
;backward_lstm_379/while/TensorArrayV2Read/TensorListGetItemѕ
backward_lstm_379/while/LessLess6backward_lstm_379_while_less_backward_lstm_379_sub_1_0#backward_lstm_379_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_379/while/LessЕ
<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOpGbackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02>
<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp•
-backward_lstm_379/while/lstm_cell_1139/MatMulMatMulBbackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem:item:0Dbackward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-backward_lstm_379/while/lstm_cell_1139/MatMulЛ
>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOpIbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02@
>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpО
/backward_lstm_379/while/lstm_cell_1139/MatMul_1MatMul%backward_lstm_379_while_placeholder_3Fbackward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»21
/backward_lstm_379/while/lstm_cell_1139/MatMul_1И
*backward_lstm_379/while/lstm_cell_1139/addAddV27backward_lstm_379/while/lstm_cell_1139/MatMul:product:09backward_lstm_379/while/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2,
*backward_lstm_379/while/lstm_cell_1139/addД
=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOpHbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02?
=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpХ
.backward_lstm_379/while/lstm_cell_1139/BiasAddBiasAdd.backward_lstm_379/while/lstm_cell_1139/add:z:0Ebackward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.backward_lstm_379/while/lstm_cell_1139/BiasAdd≤
6backward_lstm_379/while/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6backward_lstm_379/while/lstm_cell_1139/split/split_dimџ
,backward_lstm_379/while/lstm_cell_1139/splitSplit?backward_lstm_379/while/lstm_cell_1139/split/split_dim:output:07backward_lstm_379/while/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2.
,backward_lstm_379/while/lstm_cell_1139/split‘
.backward_lstm_379/while/lstm_cell_1139/SigmoidSigmoid5backward_lstm_379/while/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€220
.backward_lstm_379/while/lstm_cell_1139/SigmoidЎ
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_1Sigmoid5backward_lstm_379/while/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_1о
*backward_lstm_379/while/lstm_cell_1139/mulMul4backward_lstm_379/while/lstm_cell_1139/Sigmoid_1:y:0%backward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_379/while/lstm_cell_1139/mulЋ
+backward_lstm_379/while/lstm_cell_1139/ReluRelu5backward_lstm_379/while/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22-
+backward_lstm_379/while/lstm_cell_1139/ReluД
,backward_lstm_379/while/lstm_cell_1139/mul_1Mul2backward_lstm_379/while/lstm_cell_1139/Sigmoid:y:09backward_lstm_379/while/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_379/while/lstm_cell_1139/mul_1щ
,backward_lstm_379/while/lstm_cell_1139/add_1AddV2.backward_lstm_379/while/lstm_cell_1139/mul:z:00backward_lstm_379/while/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_379/while/lstm_cell_1139/add_1Ў
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_2Sigmoid5backward_lstm_379/while/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_2 
-backward_lstm_379/while/lstm_cell_1139/Relu_1Relu0backward_lstm_379/while/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-backward_lstm_379/while/lstm_cell_1139/Relu_1И
,backward_lstm_379/while/lstm_cell_1139/mul_2Mul4backward_lstm_379/while/lstm_cell_1139/Sigmoid_2:y:0;backward_lstm_379/while/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_379/while/lstm_cell_1139/mul_2ч
backward_lstm_379/while/SelectSelect backward_lstm_379/while/Less:z:00backward_lstm_379/while/lstm_cell_1139/mul_2:z:0%backward_lstm_379_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22 
backward_lstm_379/while/Selectы
 backward_lstm_379/while/Select_1Select backward_lstm_379/while/Less:z:00backward_lstm_379/while/lstm_cell_1139/mul_2:z:0%backward_lstm_379_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_379/while/Select_1ы
 backward_lstm_379/while/Select_2Select backward_lstm_379/while/Less:z:00backward_lstm_379/while/lstm_cell_1139/add_1:z:0%backward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_379/while/Select_2≥
<backward_lstm_379/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_379_while_placeholder_1#backward_lstm_379_while_placeholder'backward_lstm_379/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_379/while/TensorArrayV2Write/TensorListSetItemА
backward_lstm_379/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_379/while/add/y±
backward_lstm_379/while/addAddV2#backward_lstm_379_while_placeholder&backward_lstm_379/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/while/addД
backward_lstm_379/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_379/while/add_1/y–
backward_lstm_379/while/add_1AddV2<backward_lstm_379_while_backward_lstm_379_while_loop_counter(backward_lstm_379/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/while/add_1≥
 backward_lstm_379/while/IdentityIdentity!backward_lstm_379/while/add_1:z:0^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_379/while/IdentityЎ
"backward_lstm_379/while/Identity_1IdentityBbackward_lstm_379_while_backward_lstm_379_while_maximum_iterations^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_379/while/Identity_1µ
"backward_lstm_379/while/Identity_2Identitybackward_lstm_379/while/add:z:0^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_379/while/Identity_2в
"backward_lstm_379/while/Identity_3IdentityLbackward_lstm_379/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_379/while/Identity_3ќ
"backward_lstm_379/while/Identity_4Identity'backward_lstm_379/while/Select:output:0^backward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_379/while/Identity_4–
"backward_lstm_379/while/Identity_5Identity)backward_lstm_379/while/Select_1:output:0^backward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_379/while/Identity_5–
"backward_lstm_379/while/Identity_6Identity)backward_lstm_379/while/Select_2:output:0^backward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_379/while/Identity_6Њ
backward_lstm_379/while/NoOpNoOp>^backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp=^backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp?^backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_379/while/NoOp"x
9backward_lstm_379_while_backward_lstm_379_strided_slice_1;backward_lstm_379_while_backward_lstm_379_strided_slice_1_0"M
 backward_lstm_379_while_identity)backward_lstm_379/while/Identity:output:0"Q
"backward_lstm_379_while_identity_1+backward_lstm_379/while/Identity_1:output:0"Q
"backward_lstm_379_while_identity_2+backward_lstm_379/while/Identity_2:output:0"Q
"backward_lstm_379_while_identity_3+backward_lstm_379/while/Identity_3:output:0"Q
"backward_lstm_379_while_identity_4+backward_lstm_379/while/Identity_4:output:0"Q
"backward_lstm_379_while_identity_5+backward_lstm_379/while/Identity_5:output:0"Q
"backward_lstm_379_while_identity_6+backward_lstm_379/while/Identity_6:output:0"n
4backward_lstm_379_while_less_backward_lstm_379_sub_16backward_lstm_379_while_less_backward_lstm_379_sub_1_0"Т
Fbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resourceHbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0"Ф
Gbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resourceIbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0"Р
Ebackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resourceGbackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0"р
ubackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2~
=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp2|
<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp2А
>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:€€€€€€€€€
яc
ю
!__inference__traced_save_42663945
file_prefix/
+savev2_dense_379_kernel_read_readvariableop-
)savev2_dense_379_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopW
Ssavev2_bidirectional_379_forward_lstm_379_lstm_cell_1138_kernel_read_readvariableopa
]savev2_bidirectional_379_forward_lstm_379_lstm_cell_1138_recurrent_kernel_read_readvariableopU
Qsavev2_bidirectional_379_forward_lstm_379_lstm_cell_1138_bias_read_readvariableopX
Tsavev2_bidirectional_379_backward_lstm_379_lstm_cell_1139_kernel_read_readvariableopb
^savev2_bidirectional_379_backward_lstm_379_lstm_cell_1139_recurrent_kernel_read_readvariableopV
Rsavev2_bidirectional_379_backward_lstm_379_lstm_cell_1139_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_379_kernel_m_read_readvariableop4
0savev2_adam_dense_379_bias_m_read_readvariableop^
Zsavev2_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_kernel_m_read_readvariableoph
dsavev2_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_recurrent_kernel_m_read_readvariableop\
Xsavev2_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_bias_m_read_readvariableop_
[savev2_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_kernel_m_read_readvariableopi
esavev2_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_recurrent_kernel_m_read_readvariableop]
Ysavev2_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_bias_m_read_readvariableop6
2savev2_adam_dense_379_kernel_v_read_readvariableop4
0savev2_adam_dense_379_bias_v_read_readvariableop^
Zsavev2_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_kernel_v_read_readvariableoph
dsavev2_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_recurrent_kernel_v_read_readvariableop\
Xsavev2_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_bias_v_read_readvariableop_
[savev2_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_kernel_v_read_readvariableopi
esavev2_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_recurrent_kernel_v_read_readvariableop]
Ysavev2_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_bias_v_read_readvariableop9
5savev2_adam_dense_379_kernel_vhat_read_readvariableop7
3savev2_adam_dense_379_bias_vhat_read_readvariableopa
]savev2_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_kernel_vhat_read_readvariableopk
gsavev2_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_recurrent_kernel_vhat_read_readvariableop_
[savev2_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_bias_vhat_read_readvariableopb
^savev2_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_kernel_vhat_read_readvariableopl
hsavev2_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_recurrent_kernel_vhat_read_readvariableop`
\savev2_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_bias_vhat_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameҐ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*і
value™BІ(B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЎ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices’
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_379_kernel_read_readvariableop)savev2_dense_379_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopSsavev2_bidirectional_379_forward_lstm_379_lstm_cell_1138_kernel_read_readvariableop]savev2_bidirectional_379_forward_lstm_379_lstm_cell_1138_recurrent_kernel_read_readvariableopQsavev2_bidirectional_379_forward_lstm_379_lstm_cell_1138_bias_read_readvariableopTsavev2_bidirectional_379_backward_lstm_379_lstm_cell_1139_kernel_read_readvariableop^savev2_bidirectional_379_backward_lstm_379_lstm_cell_1139_recurrent_kernel_read_readvariableopRsavev2_bidirectional_379_backward_lstm_379_lstm_cell_1139_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_379_kernel_m_read_readvariableop0savev2_adam_dense_379_bias_m_read_readvariableopZsavev2_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_kernel_m_read_readvariableopdsavev2_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_recurrent_kernel_m_read_readvariableopXsavev2_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_bias_m_read_readvariableop[savev2_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_kernel_m_read_readvariableopesavev2_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_recurrent_kernel_m_read_readvariableopYsavev2_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_bias_m_read_readvariableop2savev2_adam_dense_379_kernel_v_read_readvariableop0savev2_adam_dense_379_bias_v_read_readvariableopZsavev2_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_kernel_v_read_readvariableopdsavev2_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_recurrent_kernel_v_read_readvariableopXsavev2_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_bias_v_read_readvariableop[savev2_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_kernel_v_read_readvariableopesavev2_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_recurrent_kernel_v_read_readvariableopYsavev2_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_bias_v_read_readvariableop5savev2_adam_dense_379_kernel_vhat_read_readvariableop3savev2_adam_dense_379_bias_vhat_read_readvariableop]savev2_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_kernel_vhat_read_readvariableopgsavev2_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_recurrent_kernel_vhat_read_readvariableop[savev2_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_bias_vhat_read_readvariableop^savev2_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_kernel_vhat_read_readvariableophsavev2_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_recurrent_kernel_vhat_read_readvariableop\savev2_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_bias_vhat_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*ѕ
_input_shapesљ
Ї: :d:: : : : : :	»:	2»:»:	»:	2»:»: : :d::	»:	2»:»:	»:	2»:»:d::	»:	2»:»:	»:	2»:»:d::	»:	2»:»:	»:	2»:»: 2(
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
:	»:%	!

_output_shapes
:	2»:!


_output_shapes	
:»:%!

_output_shapes
:	»:%!

_output_shapes
:	2»:!

_output_shapes	
:»:
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
:	»:%!

_output_shapes
:	2»:!

_output_shapes	
:»:%!

_output_shapes
:	»:%!

_output_shapes
:	2»:!

_output_shapes	
:»:$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	»:%!

_output_shapes
:	2»:!

_output_shapes	
:»:%!

_output_shapes
:	»:%!

_output_shapes
:	2»:!

_output_shapes	
:»:$  

_output_shapes

:d: !

_output_shapes
::%"!

_output_shapes
:	»:%#!

_output_shapes
:	2»:!$

_output_shapes	
:»:%%!

_output_shapes
:	»:%&!

_output_shapes
:	2»:!'

_output_shapes	
:»:(

_output_shapes
: 
€
К
L__inference_lstm_cell_1139_layer_call_and_return_conditional_losses_42663772

inputs
states_0
states_11
matmul_readvariableop_resource:	»3
 matmul_1_readvariableop_resource:	2».
biasadd_readvariableop_resource:	»
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	»*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_2Щ
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
?:€€€€€€€€€:€€€€€€€€€2:€€€€€€€€€2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€2
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€2
"
_user_specified_name
states/1
Є
£
L__inference_sequential_379_layer_call_and_return_conditional_losses_42660841

inputs
inputs_1	-
bidirectional_379_42660822:	»-
bidirectional_379_42660824:	2»)
bidirectional_379_42660826:	»-
bidirectional_379_42660828:	»-
bidirectional_379_42660830:	2»)
bidirectional_379_42660832:	»$
dense_379_42660835:d 
dense_379_42660837:
identityИҐ)bidirectional_379/StatefulPartitionedCallҐ!dense_379/StatefulPartitionedCall 
)bidirectional_379/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_379_42660822bidirectional_379_42660824bidirectional_379_42660826bidirectional_379_42660828bidirectional_379_42660830bidirectional_379_42660832*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_426602742+
)bidirectional_379/StatefulPartitionedCallЋ
!dense_379/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_379/StatefulPartitionedCall:output:0dense_379_42660835dense_379_42660837*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dense_379_layer_call_and_return_conditional_losses_426602992#
!dense_379/StatefulPartitionedCallЕ
IdentityIdentity*dense_379/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityЮ
NoOpNoOp*^bidirectional_379/StatefulPartitionedCall"^dense_379/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€:€€€€€€€€€: : : : : : : : 2V
)bidirectional_379/StatefulPartitionedCall)bidirectional_379/StatefulPartitionedCall2F
!dense_379/StatefulPartitionedCall!dense_379/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
…
•
$forward_lstm_379_while_cond_42659997>
:forward_lstm_379_while_forward_lstm_379_while_loop_counterD
@forward_lstm_379_while_forward_lstm_379_while_maximum_iterations&
"forward_lstm_379_while_placeholder(
$forward_lstm_379_while_placeholder_1(
$forward_lstm_379_while_placeholder_2(
$forward_lstm_379_while_placeholder_3(
$forward_lstm_379_while_placeholder_4@
<forward_lstm_379_while_less_forward_lstm_379_strided_slice_1X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42659997___redundant_placeholder0X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42659997___redundant_placeholder1X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42659997___redundant_placeholder2X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42659997___redundant_placeholder3X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42659997___redundant_placeholder4#
forward_lstm_379_while_identity
≈
forward_lstm_379/while/LessLess"forward_lstm_379_while_placeholder<forward_lstm_379_while_less_forward_lstm_379_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_379/while/LessР
forward_lstm_379/while/IdentityIdentityforward_lstm_379/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_379/while/Identity"K
forward_lstm_379_while_identity(forward_lstm_379/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
€
К
L__inference_lstm_cell_1139_layer_call_and_return_conditional_losses_42663804

inputs
states_0
states_11
matmul_readvariableop_resource:	»3
 matmul_1_readvariableop_resource:	2».
biasadd_readvariableop_resource:	»
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	»*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_2Щ
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
?:€€€€€€€€€:€€€€€€€€€2:€€€€€€€€€2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€2
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€2
"
_user_specified_name
states/1
Љ
¬
Fsequential_379_bidirectional_379_backward_lstm_379_while_cond_42657733В
~sequential_379_bidirectional_379_backward_lstm_379_while_sequential_379_bidirectional_379_backward_lstm_379_while_loop_counterЙ
Дsequential_379_bidirectional_379_backward_lstm_379_while_sequential_379_bidirectional_379_backward_lstm_379_while_maximum_iterationsH
Dsequential_379_bidirectional_379_backward_lstm_379_while_placeholderJ
Fsequential_379_bidirectional_379_backward_lstm_379_while_placeholder_1J
Fsequential_379_bidirectional_379_backward_lstm_379_while_placeholder_2J
Fsequential_379_bidirectional_379_backward_lstm_379_while_placeholder_3J
Fsequential_379_bidirectional_379_backward_lstm_379_while_placeholder_4Е
Аsequential_379_bidirectional_379_backward_lstm_379_while_less_sequential_379_bidirectional_379_backward_lstm_379_strided_slice_1Э
Шsequential_379_bidirectional_379_backward_lstm_379_while_sequential_379_bidirectional_379_backward_lstm_379_while_cond_42657733___redundant_placeholder0Э
Шsequential_379_bidirectional_379_backward_lstm_379_while_sequential_379_bidirectional_379_backward_lstm_379_while_cond_42657733___redundant_placeholder1Э
Шsequential_379_bidirectional_379_backward_lstm_379_while_sequential_379_bidirectional_379_backward_lstm_379_while_cond_42657733___redundant_placeholder2Э
Шsequential_379_bidirectional_379_backward_lstm_379_while_sequential_379_bidirectional_379_backward_lstm_379_while_cond_42657733___redundant_placeholder3Э
Шsequential_379_bidirectional_379_backward_lstm_379_while_sequential_379_bidirectional_379_backward_lstm_379_while_cond_42657733___redundant_placeholder4E
Asequential_379_bidirectional_379_backward_lstm_379_while_identity
р
=sequential_379/bidirectional_379/backward_lstm_379/while/LessLessDsequential_379_bidirectional_379_backward_lstm_379_while_placeholderАsequential_379_bidirectional_379_backward_lstm_379_while_less_sequential_379_bidirectional_379_backward_lstm_379_strided_slice_1*
T0*
_output_shapes
: 2?
=sequential_379/bidirectional_379/backward_lstm_379/while/Lessц
Asequential_379/bidirectional_379/backward_lstm_379/while/IdentityIdentityAsequential_379/bidirectional_379/backward_lstm_379/while/Less:z:0*
T0
*
_output_shapes
: 2C
Asequential_379/bidirectional_379/backward_lstm_379/while/Identity"П
Asequential_379_bidirectional_379_backward_lstm_379_while_identityJsequential_379/bidirectional_379/backward_lstm_379/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
ѕ@
д
while_body_42663371
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1139_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1139_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1139_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1139_matmul_readvariableop_resource:	»H
5while_lstm_cell_1139_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1139_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1139/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1139/MatMul/ReadVariableOpҐ,while/lstm_cell_1139/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€29
7while/TensorArrayV2Read/TensorListGetItem/element_shape№
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemѕ
*while/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1139_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1139/MatMul/ReadVariableOpЁ
while/lstm_cell_1139/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/MatMul’
,while/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1139_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1139/MatMul_1/ReadVariableOp∆
while/lstm_cell_1139/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/MatMul_1ј
while/lstm_cell_1139/addAddV2%while/lstm_cell_1139/MatMul:product:0'while/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/addќ
+while/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1139_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1139/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1139/BiasAddBiasAddwhile/lstm_cell_1139/add:z:03while/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/BiasAddО
$while/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1139/split/split_dimУ
while/lstm_cell_1139/splitSplit-while/lstm_cell_1139/split/split_dim:output:0%while/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1139/splitЮ
while/lstm_cell_1139/SigmoidSigmoid#while/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/SigmoidҐ
while/lstm_cell_1139/Sigmoid_1Sigmoid#while/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1139/Sigmoid_1¶
while/lstm_cell_1139/mulMul"while/lstm_cell_1139/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/mulХ
while/lstm_cell_1139/ReluRelu#while/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/ReluЉ
while/lstm_cell_1139/mul_1Mul while/lstm_cell_1139/Sigmoid:y:0'while/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/mul_1±
while/lstm_cell_1139/add_1AddV2while/lstm_cell_1139/mul:z:0while/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/add_1Ґ
while/lstm_cell_1139/Sigmoid_2Sigmoid#while/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1139/Sigmoid_2Ф
while/lstm_cell_1139/Relu_1Reluwhile/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/Relu_1ј
while/lstm_cell_1139/mul_2Mul"while/lstm_cell_1139/Sigmoid_2:y:0)while/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1139/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3П
while/Identity_4Identitywhile/lstm_cell_1139/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1139/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1139/BiasAdd/ReadVariableOp+^while/lstm_cell_1139/MatMul/ReadVariableOp-^while/lstm_cell_1139/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"n
4while_lstm_cell_1139_biasadd_readvariableop_resource6while_lstm_cell_1139_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1139_matmul_1_readvariableop_resource7while_lstm_cell_1139_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1139_matmul_readvariableop_resource5while_lstm_cell_1139_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1139/BiasAdd/ReadVariableOp+while/lstm_cell_1139/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1139/MatMul/ReadVariableOp*while/lstm_cell_1139/MatMul/ReadVariableOp2\
,while/lstm_cell_1139/MatMul_1/ReadVariableOp,while/lstm_cell_1139/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
ѕ@
д
while_body_42662717
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1138_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1138_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1138_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1138_matmul_readvariableop_resource:	»H
5while_lstm_cell_1138_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1138_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1138/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1138/MatMul/ReadVariableOpҐ,while/lstm_cell_1138/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€29
7while/TensorArrayV2Read/TensorListGetItem/element_shape№
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemѕ
*while/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1138_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1138/MatMul/ReadVariableOpЁ
while/lstm_cell_1138/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/MatMul’
,while/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1138_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1138/MatMul_1/ReadVariableOp∆
while/lstm_cell_1138/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/MatMul_1ј
while/lstm_cell_1138/addAddV2%while/lstm_cell_1138/MatMul:product:0'while/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/addќ
+while/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1138_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1138/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1138/BiasAddBiasAddwhile/lstm_cell_1138/add:z:03while/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/BiasAddО
$while/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1138/split/split_dimУ
while/lstm_cell_1138/splitSplit-while/lstm_cell_1138/split/split_dim:output:0%while/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1138/splitЮ
while/lstm_cell_1138/SigmoidSigmoid#while/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/SigmoidҐ
while/lstm_cell_1138/Sigmoid_1Sigmoid#while/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1138/Sigmoid_1¶
while/lstm_cell_1138/mulMul"while/lstm_cell_1138/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/mulХ
while/lstm_cell_1138/ReluRelu#while/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/ReluЉ
while/lstm_cell_1138/mul_1Mul while/lstm_cell_1138/Sigmoid:y:0'while/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/mul_1±
while/lstm_cell_1138/add_1AddV2while/lstm_cell_1138/mul:z:0while/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/add_1Ґ
while/lstm_cell_1138/Sigmoid_2Sigmoid#while/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1138/Sigmoid_2Ф
while/lstm_cell_1138/Relu_1Reluwhile/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/Relu_1ј
while/lstm_cell_1138/mul_2Mul"while/lstm_cell_1138/Sigmoid_2:y:0)while/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1138/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3П
while/Identity_4Identitywhile/lstm_cell_1138/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1138/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1138/BiasAdd/ReadVariableOp+^while/lstm_cell_1138/MatMul/ReadVariableOp-^while/lstm_cell_1138/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"n
4while_lstm_cell_1138_biasadd_readvariableop_resource6while_lstm_cell_1138_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1138_matmul_1_readvariableop_resource7while_lstm_cell_1138_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1138_matmul_readvariableop_resource5while_lstm_cell_1138_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1138/BiasAdd/ReadVariableOp+while/lstm_cell_1138/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1138/MatMul/ReadVariableOp*while/lstm_cell_1138/MatMul/ReadVariableOp2\
,while/lstm_cell_1138/MatMul_1/ReadVariableOp,while/lstm_cell_1138/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
я
Ќ
while_cond_42659338
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_42659338___redundant_placeholder06
2while_while_cond_42659338___redundant_placeholder16
2while_while_cond_42659338___redundant_placeholder26
2while_while_cond_42659338___redundant_placeholder3
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
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:
я
Ќ
while_cond_42659178
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_42659178___redundant_placeholder06
2while_while_cond_42659178___redundant_placeholder16
2while_while_cond_42659178___redundant_placeholder26
2while_while_cond_42659178___redundant_placeholder3
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
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:
к
Љ
%backward_lstm_379_while_cond_42660176@
<backward_lstm_379_while_backward_lstm_379_while_loop_counterF
Bbackward_lstm_379_while_backward_lstm_379_while_maximum_iterations'
#backward_lstm_379_while_placeholder)
%backward_lstm_379_while_placeholder_1)
%backward_lstm_379_while_placeholder_2)
%backward_lstm_379_while_placeholder_3)
%backward_lstm_379_while_placeholder_4B
>backward_lstm_379_while_less_backward_lstm_379_strided_slice_1Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42660176___redundant_placeholder0Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42660176___redundant_placeholder1Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42660176___redundant_placeholder2Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42660176___redundant_placeholder3Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42660176___redundant_placeholder4$
 backward_lstm_379_while_identity
 
backward_lstm_379/while/LessLess#backward_lstm_379_while_placeholder>backward_lstm_379_while_less_backward_lstm_379_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_379/while/LessУ
 backward_lstm_379/while/IdentityIdentity backward_lstm_379/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_379/while/Identity"M
 backward_lstm_379_while_identity)backward_lstm_379/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
‘
¬
3__inference_forward_lstm_379_layer_call_fn_42662326
inputs_0
unknown:	»
	unknown_0:	2»
	unknown_1:	»
identityИҐStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_426582062
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
ф_
≥
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_42659423

inputs@
-lstm_cell_1139_matmul_readvariableop_resource:	»B
/lstm_cell_1139_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1139_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1139/BiasAdd/ReadVariableOpҐ$lstm_cell_1139/MatMul/ReadVariableOpҐ&lstm_cell_1139/MatMul_1/ReadVariableOpҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€22
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
B :и2
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
zeros_1/packed/1Й
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
:€€€€€€€€€22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permМ
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
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
ReverseV2/axisУ
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
	ReverseV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€27
5TensorArrayUnstack/TensorListFromTensor/element_shapeэ
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
strided_slice_2/stack_2Е
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2
strided_slice_2ї
$lstm_cell_1139/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1139_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1139/MatMul/ReadVariableOp≥
lstm_cell_1139/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/MatMulЅ
&lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1139_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1139/MatMul_1/ReadVariableOpѓ
lstm_cell_1139/MatMul_1MatMulzeros:output:0.lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/MatMul_1®
lstm_cell_1139/addAddV2lstm_cell_1139/MatMul:product:0!lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/addЇ
%lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1139_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1139/BiasAdd/ReadVariableOpµ
lstm_cell_1139/BiasAddBiasAddlstm_cell_1139/add:z:0-lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/BiasAddВ
lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1139/split/split_dimы
lstm_cell_1139/splitSplit'lstm_cell_1139/split/split_dim:output:0lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1139/splitМ
lstm_cell_1139/SigmoidSigmoidlstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/SigmoidР
lstm_cell_1139/Sigmoid_1Sigmoidlstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Sigmoid_1С
lstm_cell_1139/mulMullstm_cell_1139/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/mulГ
lstm_cell_1139/ReluRelulstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Relu§
lstm_cell_1139/mul_1Mullstm_cell_1139/Sigmoid:y:0!lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/mul_1Щ
lstm_cell_1139/add_1AddV2lstm_cell_1139/mul:z:0lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/add_1Р
lstm_cell_1139/Sigmoid_2Sigmoidlstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Sigmoid_2В
lstm_cell_1139/Relu_1Relulstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Relu_1®
lstm_cell_1139/mul_2Mullstm_cell_1139/Sigmoid_2:y:0#lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterХ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1139_matmul_readvariableop_resource/lstm_cell_1139_matmul_1_readvariableop_resource.lstm_cell_1139_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42659339*
condR
while_cond_42659338*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
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
:€€€€€€€€€22

Identityќ
NoOpNoOp&^lstm_cell_1139/BiasAdd/ReadVariableOp%^lstm_cell_1139/MatMul/ReadVariableOp'^lstm_cell_1139/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1139/BiasAdd/ReadVariableOp%lstm_cell_1139/BiasAdd/ReadVariableOp2L
$lstm_cell_1139/MatMul/ReadVariableOp$lstm_cell_1139/MatMul/ReadVariableOp2P
&lstm_cell_1139/MatMul_1/ReadVariableOp&lstm_cell_1139/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
…
•
$forward_lstm_379_while_cond_42661649>
:forward_lstm_379_while_forward_lstm_379_while_loop_counterD
@forward_lstm_379_while_forward_lstm_379_while_maximum_iterations&
"forward_lstm_379_while_placeholder(
$forward_lstm_379_while_placeholder_1(
$forward_lstm_379_while_placeholder_2(
$forward_lstm_379_while_placeholder_3(
$forward_lstm_379_while_placeholder_4@
<forward_lstm_379_while_less_forward_lstm_379_strided_slice_1X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42661649___redundant_placeholder0X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42661649___redundant_placeholder1X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42661649___redundant_placeholder2X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42661649___redundant_placeholder3X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42661649___redundant_placeholder4#
forward_lstm_379_while_identity
≈
forward_lstm_379/while/LessLess"forward_lstm_379_while_placeholder<forward_lstm_379_while_less_forward_lstm_379_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_379/while/LessР
forward_lstm_379/while/IdentityIdentityforward_lstm_379/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_379/while/Identity"K
forward_lstm_379_while_identity(forward_lstm_379/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
а
ј
3__inference_forward_lstm_379_layer_call_fn_42662348

inputs
unknown:	»
	unknown_0:	2»
	unknown_1:	»
identityИҐStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_426597882
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ч
И
L__inference_lstm_cell_1139_layer_call_and_return_conditional_losses_42658545

inputs

states
states_11
matmul_readvariableop_resource:	»3
 matmul_1_readvariableop_resource:	2».
biasadd_readvariableop_resource:	»
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	»*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_2Щ
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
?:€€€€€€€€€:€€€€€€€€€2:€€€€€€€€€2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_namestates
ѕ@
д
while_body_42662868
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1138_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1138_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1138_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1138_matmul_readvariableop_resource:	»H
5while_lstm_cell_1138_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1138_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1138/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1138/MatMul/ReadVariableOpҐ,while/lstm_cell_1138/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€29
7while/TensorArrayV2Read/TensorListGetItem/element_shape№
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemѕ
*while/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1138_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1138/MatMul/ReadVariableOpЁ
while/lstm_cell_1138/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/MatMul’
,while/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1138_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1138/MatMul_1/ReadVariableOp∆
while/lstm_cell_1138/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/MatMul_1ј
while/lstm_cell_1138/addAddV2%while/lstm_cell_1138/MatMul:product:0'while/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/addќ
+while/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1138_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1138/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1138/BiasAddBiasAddwhile/lstm_cell_1138/add:z:03while/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/BiasAddО
$while/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1138/split/split_dimУ
while/lstm_cell_1138/splitSplit-while/lstm_cell_1138/split/split_dim:output:0%while/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1138/splitЮ
while/lstm_cell_1138/SigmoidSigmoid#while/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/SigmoidҐ
while/lstm_cell_1138/Sigmoid_1Sigmoid#while/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1138/Sigmoid_1¶
while/lstm_cell_1138/mulMul"while/lstm_cell_1138/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/mulХ
while/lstm_cell_1138/ReluRelu#while/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/ReluЉ
while/lstm_cell_1138/mul_1Mul while/lstm_cell_1138/Sigmoid:y:0'while/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/mul_1±
while/lstm_cell_1138/add_1AddV2while/lstm_cell_1138/mul:z:0while/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/add_1Ґ
while/lstm_cell_1138/Sigmoid_2Sigmoid#while/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1138/Sigmoid_2Ф
while/lstm_cell_1138/Relu_1Reluwhile/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/Relu_1ј
while/lstm_cell_1138/mul_2Mul"while/lstm_cell_1138/Sigmoid_2:y:0)while/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1138/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3П
while/Identity_4Identitywhile/lstm_cell_1138/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1138/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1138/BiasAdd/ReadVariableOp+^while/lstm_cell_1138/MatMul/ReadVariableOp-^while/lstm_cell_1138/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"n
4while_lstm_cell_1138_biasadd_readvariableop_resource6while_lstm_cell_1138_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1138_matmul_1_readvariableop_resource7while_lstm_cell_1138_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1138_matmul_readvariableop_resource5while_lstm_cell_1138_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1138/BiasAdd/ReadVariableOp+while/lstm_cell_1138/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1138/MatMul/ReadVariableOp*while/lstm_cell_1138/MatMul/ReadVariableOp2\
,while/lstm_cell_1138/MatMul_1/ReadVariableOp,while/lstm_cell_1138/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
э
µ
%backward_lstm_379_while_cond_42661481@
<backward_lstm_379_while_backward_lstm_379_while_loop_counterF
Bbackward_lstm_379_while_backward_lstm_379_while_maximum_iterations'
#backward_lstm_379_while_placeholder)
%backward_lstm_379_while_placeholder_1)
%backward_lstm_379_while_placeholder_2)
%backward_lstm_379_while_placeholder_3B
>backward_lstm_379_while_less_backward_lstm_379_strided_slice_1Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42661481___redundant_placeholder0Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42661481___redundant_placeholder1Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42661481___redundant_placeholder2Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42661481___redundant_placeholder3$
 backward_lstm_379_while_identity
 
backward_lstm_379/while/LessLess#backward_lstm_379_while_placeholder>backward_lstm_379_while_less_backward_lstm_379_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_379/while/LessУ
 backward_lstm_379/while/IdentityIdentity backward_lstm_379/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_379/while/Identity"M
 backward_lstm_379_while_identity)backward_lstm_379/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:
м]
≤
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_42662801

inputs@
-lstm_cell_1138_matmul_readvariableop_resource:	»B
/lstm_cell_1138_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1138_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1138/BiasAdd/ReadVariableOpҐ$lstm_cell_1138/MatMul/ReadVariableOpҐ&lstm_cell_1138/MatMul_1/ReadVariableOpҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€22
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
B :и2
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
zeros_1/packed/1Й
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
:€€€€€€€€€22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permМ
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2Е
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2
strided_slice_2ї
$lstm_cell_1138/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1138_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1138/MatMul/ReadVariableOp≥
lstm_cell_1138/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/MatMulЅ
&lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1138_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1138/MatMul_1/ReadVariableOpѓ
lstm_cell_1138/MatMul_1MatMulzeros:output:0.lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/MatMul_1®
lstm_cell_1138/addAddV2lstm_cell_1138/MatMul:product:0!lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/addЇ
%lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1138_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1138/BiasAdd/ReadVariableOpµ
lstm_cell_1138/BiasAddBiasAddlstm_cell_1138/add:z:0-lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/BiasAddВ
lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1138/split/split_dimы
lstm_cell_1138/splitSplit'lstm_cell_1138/split/split_dim:output:0lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1138/splitМ
lstm_cell_1138/SigmoidSigmoidlstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/SigmoidР
lstm_cell_1138/Sigmoid_1Sigmoidlstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Sigmoid_1С
lstm_cell_1138/mulMullstm_cell_1138/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/mulГ
lstm_cell_1138/ReluRelulstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Relu§
lstm_cell_1138/mul_1Mullstm_cell_1138/Sigmoid:y:0!lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/mul_1Щ
lstm_cell_1138/add_1AddV2lstm_cell_1138/mul:z:0lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/add_1Р
lstm_cell_1138/Sigmoid_2Sigmoidlstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Sigmoid_2В
lstm_cell_1138/Relu_1Relulstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Relu_1®
lstm_cell_1138/mul_2Mullstm_cell_1138/Sigmoid_2:y:0#lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterХ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1138_matmul_readvariableop_resource/lstm_cell_1138_matmul_1_readvariableop_resource.lstm_cell_1138_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42662717*
condR
while_cond_42662716*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
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
:€€€€€€€€€22

Identityќ
NoOpNoOp&^lstm_cell_1138/BiasAdd/ReadVariableOp%^lstm_cell_1138/MatMul/ReadVariableOp'^lstm_cell_1138/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1138/BiasAdd/ReadVariableOp%lstm_cell_1138/BiasAdd/ReadVariableOp2L
$lstm_cell_1138/MatMul/ReadVariableOp$lstm_cell_1138/MatMul/ReadVariableOp2P
&lstm_cell_1138/MatMul_1/ReadVariableOp&lstm_cell_1138/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Њ
ъ
1__inference_lstm_cell_1138_layer_call_fn_42663625

inputs
states_0
states_1
unknown:	»
	unknown_0:	2»
	unknown_1:	»
identity

identity_1

identity_2ИҐStatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_lstm_cell_1138_layer_call_and_return_conditional_losses_426579132
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

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
?:€€€€€€€€€:€€€€€€€€€2:€€€€€€€€€2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€2
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€2
"
_user_specified_name
states/1
я
Ќ
while_cond_42662867
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_42662867___redundant_placeholder06
2while_while_cond_42662867___redundant_placeholder16
2while_while_cond_42662867___redundant_placeholder26
2while_while_cond_42662867___redundant_placeholder3
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
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:
м]
≤
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_42659788

inputs@
-lstm_cell_1138_matmul_readvariableop_resource:	»B
/lstm_cell_1138_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1138_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1138/BiasAdd/ReadVariableOpҐ$lstm_cell_1138/MatMul/ReadVariableOpҐ&lstm_cell_1138/MatMul_1/ReadVariableOpҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€22
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
B :и2
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
zeros_1/packed/1Й
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
:€€€€€€€€€22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permМ
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2Е
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2
strided_slice_2ї
$lstm_cell_1138/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1138_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1138/MatMul/ReadVariableOp≥
lstm_cell_1138/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/MatMulЅ
&lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1138_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1138/MatMul_1/ReadVariableOpѓ
lstm_cell_1138/MatMul_1MatMulzeros:output:0.lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/MatMul_1®
lstm_cell_1138/addAddV2lstm_cell_1138/MatMul:product:0!lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/addЇ
%lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1138_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1138/BiasAdd/ReadVariableOpµ
lstm_cell_1138/BiasAddBiasAddlstm_cell_1138/add:z:0-lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/BiasAddВ
lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1138/split/split_dimы
lstm_cell_1138/splitSplit'lstm_cell_1138/split/split_dim:output:0lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1138/splitМ
lstm_cell_1138/SigmoidSigmoidlstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/SigmoidР
lstm_cell_1138/Sigmoid_1Sigmoidlstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Sigmoid_1С
lstm_cell_1138/mulMullstm_cell_1138/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/mulГ
lstm_cell_1138/ReluRelulstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Relu§
lstm_cell_1138/mul_1Mullstm_cell_1138/Sigmoid:y:0!lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/mul_1Щ
lstm_cell_1138/add_1AddV2lstm_cell_1138/mul:z:0lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/add_1Р
lstm_cell_1138/Sigmoid_2Sigmoidlstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Sigmoid_2В
lstm_cell_1138/Relu_1Relulstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Relu_1®
lstm_cell_1138/mul_2Mullstm_cell_1138/Sigmoid_2:y:0#lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterХ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1138_matmul_readvariableop_resource/lstm_cell_1138_matmul_1_readvariableop_resource.lstm_cell_1138_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42659704*
condR
while_cond_42659703*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
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
:€€€€€€€€€22

Identityќ
NoOpNoOp&^lstm_cell_1138/BiasAdd/ReadVariableOp%^lstm_cell_1138/MatMul/ReadVariableOp'^lstm_cell_1138/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1138/BiasAdd/ReadVariableOp%lstm_cell_1138/BiasAdd/ReadVariableOp2L
$lstm_cell_1138/MatMul/ReadVariableOp$lstm_cell_1138/MatMul/ReadVariableOp2P
&lstm_cell_1138/MatMul_1/ReadVariableOp&lstm_cell_1138/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
З
ш
G__inference_dense_379_layer_call_and_return_conditional_losses_42660299

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
ф_
≥
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_42663608

inputs@
-lstm_cell_1139_matmul_readvariableop_resource:	»B
/lstm_cell_1139_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1139_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1139/BiasAdd/ReadVariableOpҐ$lstm_cell_1139/MatMul/ReadVariableOpҐ&lstm_cell_1139/MatMul_1/ReadVariableOpҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€22
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
B :и2
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
zeros_1/packed/1Й
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
:€€€€€€€€€22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permМ
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
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
ReverseV2/axisУ
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
	ReverseV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€27
5TensorArrayUnstack/TensorListFromTensor/element_shapeэ
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
strided_slice_2/stack_2Е
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2
strided_slice_2ї
$lstm_cell_1139/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1139_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1139/MatMul/ReadVariableOp≥
lstm_cell_1139/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/MatMulЅ
&lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1139_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1139/MatMul_1/ReadVariableOpѓ
lstm_cell_1139/MatMul_1MatMulzeros:output:0.lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/MatMul_1®
lstm_cell_1139/addAddV2lstm_cell_1139/MatMul:product:0!lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/addЇ
%lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1139_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1139/BiasAdd/ReadVariableOpµ
lstm_cell_1139/BiasAddBiasAddlstm_cell_1139/add:z:0-lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/BiasAddВ
lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1139/split/split_dimы
lstm_cell_1139/splitSplit'lstm_cell_1139/split/split_dim:output:0lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1139/splitМ
lstm_cell_1139/SigmoidSigmoidlstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/SigmoidР
lstm_cell_1139/Sigmoid_1Sigmoidlstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Sigmoid_1С
lstm_cell_1139/mulMullstm_cell_1139/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/mulГ
lstm_cell_1139/ReluRelulstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Relu§
lstm_cell_1139/mul_1Mullstm_cell_1139/Sigmoid:y:0!lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/mul_1Щ
lstm_cell_1139/add_1AddV2lstm_cell_1139/mul:z:0lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/add_1Р
lstm_cell_1139/Sigmoid_2Sigmoidlstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Sigmoid_2В
lstm_cell_1139/Relu_1Relulstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Relu_1®
lstm_cell_1139/mul_2Mullstm_cell_1139/Sigmoid_2:y:0#lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterХ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1139_matmul_readvariableop_resource/lstm_cell_1139_matmul_1_readvariableop_resource.lstm_cell_1139_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42663524*
condR
while_cond_42663523*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
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
:€€€€€€€€€22

Identityќ
NoOpNoOp&^lstm_cell_1139/BiasAdd/ReadVariableOp%^lstm_cell_1139/MatMul/ReadVariableOp'^lstm_cell_1139/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1139/BiasAdd/ReadVariableOp%lstm_cell_1139/BiasAdd/ReadVariableOp2L
$lstm_cell_1139/MatMul/ReadVariableOp$lstm_cell_1139/MatMul/ReadVariableOp2P
&lstm_cell_1139/MatMul_1/ReadVariableOp&lstm_cell_1139/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
м]
≤
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_42662952

inputs@
-lstm_cell_1138_matmul_readvariableop_resource:	»B
/lstm_cell_1138_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1138_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1138/BiasAdd/ReadVariableOpҐ$lstm_cell_1138/MatMul/ReadVariableOpҐ&lstm_cell_1138/MatMul_1/ReadVariableOpҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€22
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
B :и2
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
zeros_1/packed/1Й
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
:€€€€€€€€€22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permМ
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2Е
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2
strided_slice_2ї
$lstm_cell_1138/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1138_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1138/MatMul/ReadVariableOp≥
lstm_cell_1138/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/MatMulЅ
&lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1138_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1138/MatMul_1/ReadVariableOpѓ
lstm_cell_1138/MatMul_1MatMulzeros:output:0.lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/MatMul_1®
lstm_cell_1138/addAddV2lstm_cell_1138/MatMul:product:0!lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/addЇ
%lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1138_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1138/BiasAdd/ReadVariableOpµ
lstm_cell_1138/BiasAddBiasAddlstm_cell_1138/add:z:0-lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/BiasAddВ
lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1138/split/split_dimы
lstm_cell_1138/splitSplit'lstm_cell_1138/split/split_dim:output:0lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1138/splitМ
lstm_cell_1138/SigmoidSigmoidlstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/SigmoidР
lstm_cell_1138/Sigmoid_1Sigmoidlstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Sigmoid_1С
lstm_cell_1138/mulMullstm_cell_1138/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/mulГ
lstm_cell_1138/ReluRelulstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Relu§
lstm_cell_1138/mul_1Mullstm_cell_1138/Sigmoid:y:0!lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/mul_1Щ
lstm_cell_1138/add_1AddV2lstm_cell_1138/mul:z:0lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/add_1Р
lstm_cell_1138/Sigmoid_2Sigmoidlstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Sigmoid_2В
lstm_cell_1138/Relu_1Relulstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Relu_1®
lstm_cell_1138/mul_2Mullstm_cell_1138/Sigmoid_2:y:0#lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterХ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1138_matmul_readvariableop_resource/lstm_cell_1138_matmul_1_readvariableop_resource.lstm_cell_1138_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42662868*
condR
while_cond_42662867*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
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
:€€€€€€€€€22

Identityќ
NoOpNoOp&^lstm_cell_1138/BiasAdd/ReadVariableOp%^lstm_cell_1138/MatMul/ReadVariableOp'^lstm_cell_1138/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1138/BiasAdd/ReadVariableOp%lstm_cell_1138/BiasAdd/ReadVariableOp2L
$lstm_cell_1138/MatMul/ReadVariableOp$lstm_cell_1138/MatMul/ReadVariableOp2P
&lstm_cell_1138/MatMul_1/ReadVariableOp&lstm_cell_1138/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ёю
п
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_42661266
inputs_0Q
>forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource:	»S
@forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource:	2»N
?forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource:	»R
?backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resource:	»T
Abackward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource:	2»O
@backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource:	»
identityИҐ7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpҐ6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpҐ8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpҐbackward_lstm_379/whileҐ6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpҐ5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpҐ7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpҐforward_lstm_379/whileh
forward_lstm_379/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_379/ShapeЦ
$forward_lstm_379/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_379/strided_slice/stackЪ
&forward_lstm_379/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_379/strided_slice/stack_1Ъ
&forward_lstm_379/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_379/strided_slice/stack_2»
forward_lstm_379/strided_sliceStridedSliceforward_lstm_379/Shape:output:0-forward_lstm_379/strided_slice/stack:output:0/forward_lstm_379/strided_slice/stack_1:output:0/forward_lstm_379/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_379/strided_slice~
forward_lstm_379/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_379/zeros/mul/y∞
forward_lstm_379/zeros/mulMul'forward_lstm_379/strided_slice:output:0%forward_lstm_379/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros/mulБ
forward_lstm_379/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
forward_lstm_379/zeros/Less/yЂ
forward_lstm_379/zeros/LessLessforward_lstm_379/zeros/mul:z:0&forward_lstm_379/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros/LessД
forward_lstm_379/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_379/zeros/packed/1«
forward_lstm_379/zeros/packedPack'forward_lstm_379/strided_slice:output:0(forward_lstm_379/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_379/zeros/packedЕ
forward_lstm_379/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_379/zeros/Constє
forward_lstm_379/zerosFill&forward_lstm_379/zeros/packed:output:0%forward_lstm_379/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/zerosВ
forward_lstm_379/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_379/zeros_1/mul/yґ
forward_lstm_379/zeros_1/mulMul'forward_lstm_379/strided_slice:output:0'forward_lstm_379/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros_1/mulЕ
forward_lstm_379/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2!
forward_lstm_379/zeros_1/Less/y≥
forward_lstm_379/zeros_1/LessLess forward_lstm_379/zeros_1/mul:z:0(forward_lstm_379/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros_1/LessИ
!forward_lstm_379/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_379/zeros_1/packed/1Ќ
forward_lstm_379/zeros_1/packedPack'forward_lstm_379/strided_slice:output:0*forward_lstm_379/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_379/zeros_1/packedЙ
forward_lstm_379/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_379/zeros_1/ConstЅ
forward_lstm_379/zeros_1Fill(forward_lstm_379/zeros_1/packed:output:0'forward_lstm_379/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/zeros_1Ч
forward_lstm_379/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_379/transpose/permЅ
forward_lstm_379/transpose	Transposeinputs_0(forward_lstm_379/transpose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
forward_lstm_379/transposeВ
forward_lstm_379/Shape_1Shapeforward_lstm_379/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_379/Shape_1Ъ
&forward_lstm_379/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_379/strided_slice_1/stackЮ
(forward_lstm_379/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_1/stack_1Ю
(forward_lstm_379/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_1/stack_2‘
 forward_lstm_379/strided_slice_1StridedSlice!forward_lstm_379/Shape_1:output:0/forward_lstm_379/strided_slice_1/stack:output:01forward_lstm_379/strided_slice_1/stack_1:output:01forward_lstm_379/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_379/strided_slice_1І
,forward_lstm_379/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,forward_lstm_379/TensorArrayV2/element_shapeц
forward_lstm_379/TensorArrayV2TensorListReserve5forward_lstm_379/TensorArrayV2/element_shape:output:0)forward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_379/TensorArrayV2б
Fforward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2H
Fforward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeЉ
8forward_lstm_379/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_379/transpose:y:0Oforward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_379/TensorArrayUnstack/TensorListFromTensorЪ
&forward_lstm_379/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_379/strided_slice_2/stackЮ
(forward_lstm_379/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_2/stack_1Ю
(forward_lstm_379/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_2/stack_2л
 forward_lstm_379/strided_slice_2StridedSliceforward_lstm_379/transpose:y:0/forward_lstm_379/strided_slice_2/stack:output:01forward_lstm_379/strided_slice_2/stack_1:output:01forward_lstm_379/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2"
 forward_lstm_379/strided_slice_2о
5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOp>forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype027
5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpч
&forward_lstm_379/lstm_cell_1138/MatMulMatMul)forward_lstm_379/strided_slice_2:output:0=forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2(
&forward_lstm_379/lstm_cell_1138/MatMulф
7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOp@forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype029
7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpу
(forward_lstm_379/lstm_cell_1138/MatMul_1MatMulforward_lstm_379/zeros:output:0?forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(forward_lstm_379/lstm_cell_1138/MatMul_1м
#forward_lstm_379/lstm_cell_1138/addAddV20forward_lstm_379/lstm_cell_1138/MatMul:product:02forward_lstm_379/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2%
#forward_lstm_379/lstm_cell_1138/addн
6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOp?forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype028
6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpщ
'forward_lstm_379/lstm_cell_1138/BiasAddBiasAdd'forward_lstm_379/lstm_cell_1138/add:z:0>forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'forward_lstm_379/lstm_cell_1138/BiasAdd§
/forward_lstm_379/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/forward_lstm_379/lstm_cell_1138/split/split_dimњ
%forward_lstm_379/lstm_cell_1138/splitSplit8forward_lstm_379/lstm_cell_1138/split/split_dim:output:00forward_lstm_379/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2'
%forward_lstm_379/lstm_cell_1138/splitњ
'forward_lstm_379/lstm_cell_1138/SigmoidSigmoid.forward_lstm_379/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'forward_lstm_379/lstm_cell_1138/Sigmoid√
)forward_lstm_379/lstm_cell_1138/Sigmoid_1Sigmoid.forward_lstm_379/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_379/lstm_cell_1138/Sigmoid_1’
#forward_lstm_379/lstm_cell_1138/mulMul-forward_lstm_379/lstm_cell_1138/Sigmoid_1:y:0!forward_lstm_379/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22%
#forward_lstm_379/lstm_cell_1138/mulґ
$forward_lstm_379/lstm_cell_1138/ReluRelu.forward_lstm_379/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22&
$forward_lstm_379/lstm_cell_1138/Reluи
%forward_lstm_379/lstm_cell_1138/mul_1Mul+forward_lstm_379/lstm_cell_1138/Sigmoid:y:02forward_lstm_379/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_379/lstm_cell_1138/mul_1Ё
%forward_lstm_379/lstm_cell_1138/add_1AddV2'forward_lstm_379/lstm_cell_1138/mul:z:0)forward_lstm_379/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_379/lstm_cell_1138/add_1√
)forward_lstm_379/lstm_cell_1138/Sigmoid_2Sigmoid.forward_lstm_379/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_379/lstm_cell_1138/Sigmoid_2µ
&forward_lstm_379/lstm_cell_1138/Relu_1Relu)forward_lstm_379/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&forward_lstm_379/lstm_cell_1138/Relu_1м
%forward_lstm_379/lstm_cell_1138/mul_2Mul-forward_lstm_379/lstm_cell_1138/Sigmoid_2:y:04forward_lstm_379/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_379/lstm_cell_1138/mul_2±
.forward_lstm_379/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   20
.forward_lstm_379/TensorArrayV2_1/element_shapeь
 forward_lstm_379/TensorArrayV2_1TensorListReserve7forward_lstm_379/TensorArrayV2_1/element_shape:output:0)forward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_379/TensorArrayV2_1p
forward_lstm_379/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_379/time°
)forward_lstm_379/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2+
)forward_lstm_379/while/maximum_iterationsМ
#forward_lstm_379/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_379/while/loop_counterФ
forward_lstm_379/whileWhile,forward_lstm_379/while/loop_counter:output:02forward_lstm_379/while/maximum_iterations:output:0forward_lstm_379/time:output:0)forward_lstm_379/TensorArrayV2_1:handle:0forward_lstm_379/zeros:output:0!forward_lstm_379/zeros_1:output:0)forward_lstm_379/strided_slice_1:output:0Hforward_lstm_379/TensorArrayUnstack/TensorListFromTensor:output_handle:0>forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource@forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource?forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *0
body(R&
$forward_lstm_379_while_body_42661031*0
cond(R&
$forward_lstm_379_while_cond_42661030*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
forward_lstm_379/while„
Aforward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2C
Aforward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeµ
3forward_lstm_379/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_379/while:output:3Jforward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype025
3forward_lstm_379/TensorArrayV2Stack/TensorListStack£
&forward_lstm_379/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2(
&forward_lstm_379/strided_slice_3/stackЮ
(forward_lstm_379/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_379/strided_slice_3/stack_1Ю
(forward_lstm_379/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_3/stack_2А
 forward_lstm_379/strided_slice_3StridedSlice<forward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_379/strided_slice_3/stack:output:01forward_lstm_379/strided_slice_3/stack_1:output:01forward_lstm_379/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2"
 forward_lstm_379/strided_slice_3Ы
!forward_lstm_379/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_379/transpose_1/permт
forward_lstm_379/transpose_1	Transpose<forward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_379/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
forward_lstm_379/transpose_1И
forward_lstm_379/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_379/runtimej
backward_lstm_379/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_379/ShapeШ
%backward_lstm_379/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_379/strided_slice/stackЬ
'backward_lstm_379/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_379/strided_slice/stack_1Ь
'backward_lstm_379/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_379/strided_slice/stack_2ќ
backward_lstm_379/strided_sliceStridedSlice backward_lstm_379/Shape:output:0.backward_lstm_379/strided_slice/stack:output:00backward_lstm_379/strided_slice/stack_1:output:00backward_lstm_379/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_379/strided_sliceА
backward_lstm_379/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_379/zeros/mul/yі
backward_lstm_379/zeros/mulMul(backward_lstm_379/strided_slice:output:0&backward_lstm_379/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/zeros/mulГ
backward_lstm_379/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
backward_lstm_379/zeros/Less/yѓ
backward_lstm_379/zeros/LessLessbackward_lstm_379/zeros/mul:z:0'backward_lstm_379/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/zeros/LessЖ
 backward_lstm_379/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_379/zeros/packed/1Ћ
backward_lstm_379/zeros/packedPack(backward_lstm_379/strided_slice:output:0)backward_lstm_379/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_379/zeros/packedЗ
backward_lstm_379/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_379/zeros/Constљ
backward_lstm_379/zerosFill'backward_lstm_379/zeros/packed:output:0&backward_lstm_379/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_379/zerosД
backward_lstm_379/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_379/zeros_1/mul/yЇ
backward_lstm_379/zeros_1/mulMul(backward_lstm_379/strided_slice:output:0(backward_lstm_379/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/zeros_1/mulЗ
 backward_lstm_379/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 backward_lstm_379/zeros_1/Less/yЈ
backward_lstm_379/zeros_1/LessLess!backward_lstm_379/zeros_1/mul:z:0)backward_lstm_379/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_379/zeros_1/LessК
"backward_lstm_379/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_379/zeros_1/packed/1—
 backward_lstm_379/zeros_1/packedPack(backward_lstm_379/strided_slice:output:0+backward_lstm_379/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_379/zeros_1/packedЛ
backward_lstm_379/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_379/zeros_1/Const≈
backward_lstm_379/zeros_1Fill)backward_lstm_379/zeros_1/packed:output:0(backward_lstm_379/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_379/zeros_1Щ
 backward_lstm_379/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_379/transpose/permƒ
backward_lstm_379/transpose	Transposeinputs_0)backward_lstm_379/transpose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
backward_lstm_379/transposeЕ
backward_lstm_379/Shape_1Shapebackward_lstm_379/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_379/Shape_1Ь
'backward_lstm_379/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_379/strided_slice_1/stack†
)backward_lstm_379/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_1/stack_1†
)backward_lstm_379/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_1/stack_2Џ
!backward_lstm_379/strided_slice_1StridedSlice"backward_lstm_379/Shape_1:output:00backward_lstm_379/strided_slice_1/stack:output:02backward_lstm_379/strided_slice_1/stack_1:output:02backward_lstm_379/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_379/strided_slice_1©
-backward_lstm_379/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-backward_lstm_379/TensorArrayV2/element_shapeъ
backward_lstm_379/TensorArrayV2TensorListReserve6backward_lstm_379/TensorArrayV2/element_shape:output:0*backward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_379/TensorArrayV2О
 backward_lstm_379/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_379/ReverseV2/axisџ
backward_lstm_379/ReverseV2	ReverseV2backward_lstm_379/transpose:y:0)backward_lstm_379/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
backward_lstm_379/ReverseV2г
Gbackward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2I
Gbackward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape≈
9backward_lstm_379/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_379/ReverseV2:output:0Pbackward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_379/TensorArrayUnstack/TensorListFromTensorЬ
'backward_lstm_379/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_379/strided_slice_2/stack†
)backward_lstm_379/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_2/stack_1†
)backward_lstm_379/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_2/stack_2с
!backward_lstm_379/strided_slice_2StridedSlicebackward_lstm_379/transpose:y:00backward_lstm_379/strided_slice_2/stack:output:02backward_lstm_379/strided_slice_2/stack_1:output:02backward_lstm_379/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2#
!backward_lstm_379/strided_slice_2с
6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOp?backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype028
6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpы
'backward_lstm_379/lstm_cell_1139/MatMulMatMul*backward_lstm_379/strided_slice_2:output:0>backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'backward_lstm_379/lstm_cell_1139/MatMulч
8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOpAbackward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02:
8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpч
)backward_lstm_379/lstm_cell_1139/MatMul_1MatMul backward_lstm_379/zeros:output:0@backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)backward_lstm_379/lstm_cell_1139/MatMul_1р
$backward_lstm_379/lstm_cell_1139/addAddV21backward_lstm_379/lstm_cell_1139/MatMul:product:03backward_lstm_379/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2&
$backward_lstm_379/lstm_cell_1139/addр
7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOp@backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype029
7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpэ
(backward_lstm_379/lstm_cell_1139/BiasAddBiasAdd(backward_lstm_379/lstm_cell_1139/add:z:0?backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(backward_lstm_379/lstm_cell_1139/BiasAdd¶
0backward_lstm_379/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0backward_lstm_379/lstm_cell_1139/split/split_dim√
&backward_lstm_379/lstm_cell_1139/splitSplit9backward_lstm_379/lstm_cell_1139/split/split_dim:output:01backward_lstm_379/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2(
&backward_lstm_379/lstm_cell_1139/split¬
(backward_lstm_379/lstm_cell_1139/SigmoidSigmoid/backward_lstm_379/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(backward_lstm_379/lstm_cell_1139/Sigmoid∆
*backward_lstm_379/lstm_cell_1139/Sigmoid_1Sigmoid/backward_lstm_379/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_379/lstm_cell_1139/Sigmoid_1ў
$backward_lstm_379/lstm_cell_1139/mulMul.backward_lstm_379/lstm_cell_1139/Sigmoid_1:y:0"backward_lstm_379/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$backward_lstm_379/lstm_cell_1139/mulє
%backward_lstm_379/lstm_cell_1139/ReluRelu/backward_lstm_379/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22'
%backward_lstm_379/lstm_cell_1139/Reluм
&backward_lstm_379/lstm_cell_1139/mul_1Mul,backward_lstm_379/lstm_cell_1139/Sigmoid:y:03backward_lstm_379/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_379/lstm_cell_1139/mul_1б
&backward_lstm_379/lstm_cell_1139/add_1AddV2(backward_lstm_379/lstm_cell_1139/mul:z:0*backward_lstm_379/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_379/lstm_cell_1139/add_1∆
*backward_lstm_379/lstm_cell_1139/Sigmoid_2Sigmoid/backward_lstm_379/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_379/lstm_cell_1139/Sigmoid_2Є
'backward_lstm_379/lstm_cell_1139/Relu_1Relu*backward_lstm_379/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'backward_lstm_379/lstm_cell_1139/Relu_1р
&backward_lstm_379/lstm_cell_1139/mul_2Mul.backward_lstm_379/lstm_cell_1139/Sigmoid_2:y:05backward_lstm_379/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_379/lstm_cell_1139/mul_2≥
/backward_lstm_379/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   21
/backward_lstm_379/TensorArrayV2_1/element_shapeА
!backward_lstm_379/TensorArrayV2_1TensorListReserve8backward_lstm_379/TensorArrayV2_1/element_shape:output:0*backward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_379/TensorArrayV2_1r
backward_lstm_379/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_379/time£
*backward_lstm_379/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2,
*backward_lstm_379/while/maximum_iterationsО
$backward_lstm_379/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_379/while/loop_counter£
backward_lstm_379/whileWhile-backward_lstm_379/while/loop_counter:output:03backward_lstm_379/while/maximum_iterations:output:0backward_lstm_379/time:output:0*backward_lstm_379/TensorArrayV2_1:handle:0 backward_lstm_379/zeros:output:0"backward_lstm_379/zeros_1:output:0*backward_lstm_379/strided_slice_1:output:0Ibackward_lstm_379/TensorArrayUnstack/TensorListFromTensor:output_handle:0?backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resourceAbackward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource@backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *1
body)R'
%backward_lstm_379_while_body_42661180*1
cond)R'
%backward_lstm_379_while_cond_42661179*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
backward_lstm_379/whileў
Bbackward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2D
Bbackward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeє
4backward_lstm_379/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_379/while:output:3Kbackward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype026
4backward_lstm_379/TensorArrayV2Stack/TensorListStack•
'backward_lstm_379/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2)
'backward_lstm_379/strided_slice_3/stack†
)backward_lstm_379/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_379/strided_slice_3/stack_1†
)backward_lstm_379/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_3/stack_2Ж
!backward_lstm_379/strided_slice_3StridedSlice=backward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_379/strided_slice_3/stack:output:02backward_lstm_379/strided_slice_3/stack_1:output:02backward_lstm_379/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2#
!backward_lstm_379/strided_slice_3Э
"backward_lstm_379/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_379/transpose_1/permц
backward_lstm_379/transpose_1	Transpose=backward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_379/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
backward_lstm_379/transpose_1К
backward_lstm_379/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_379/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisƒ
concatConcatV2)forward_lstm_379/strided_slice_3:output:0*backward_lstm_379/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d2

IdentityЏ
NoOpNoOp8^backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp7^backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp9^backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp^backward_lstm_379/while7^forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp6^forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp8^forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp^forward_lstm_379/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : 2r
7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp2p
6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp2t
8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp22
backward_lstm_379/whilebackward_lstm_379/while2p
6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp2n
5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp2r
7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp20
forward_lstm_379/whileforward_lstm_379/while:g c
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
Щ
™
Esequential_379_bidirectional_379_forward_lstm_379_while_cond_42657554А
|sequential_379_bidirectional_379_forward_lstm_379_while_sequential_379_bidirectional_379_forward_lstm_379_while_loop_counterЗ
Вsequential_379_bidirectional_379_forward_lstm_379_while_sequential_379_bidirectional_379_forward_lstm_379_while_maximum_iterationsG
Csequential_379_bidirectional_379_forward_lstm_379_while_placeholderI
Esequential_379_bidirectional_379_forward_lstm_379_while_placeholder_1I
Esequential_379_bidirectional_379_forward_lstm_379_while_placeholder_2I
Esequential_379_bidirectional_379_forward_lstm_379_while_placeholder_3I
Esequential_379_bidirectional_379_forward_lstm_379_while_placeholder_4В
~sequential_379_bidirectional_379_forward_lstm_379_while_less_sequential_379_bidirectional_379_forward_lstm_379_strided_slice_1Ы
Цsequential_379_bidirectional_379_forward_lstm_379_while_sequential_379_bidirectional_379_forward_lstm_379_while_cond_42657554___redundant_placeholder0Ы
Цsequential_379_bidirectional_379_forward_lstm_379_while_sequential_379_bidirectional_379_forward_lstm_379_while_cond_42657554___redundant_placeholder1Ы
Цsequential_379_bidirectional_379_forward_lstm_379_while_sequential_379_bidirectional_379_forward_lstm_379_while_cond_42657554___redundant_placeholder2Ы
Цsequential_379_bidirectional_379_forward_lstm_379_while_sequential_379_bidirectional_379_forward_lstm_379_while_cond_42657554___redundant_placeholder3Ы
Цsequential_379_bidirectional_379_forward_lstm_379_while_sequential_379_bidirectional_379_forward_lstm_379_while_cond_42657554___redundant_placeholder4D
@sequential_379_bidirectional_379_forward_lstm_379_while_identity
к
<sequential_379/bidirectional_379/forward_lstm_379/while/LessLessCsequential_379_bidirectional_379_forward_lstm_379_while_placeholder~sequential_379_bidirectional_379_forward_lstm_379_while_less_sequential_379_bidirectional_379_forward_lstm_379_strided_slice_1*
T0*
_output_shapes
: 2>
<sequential_379/bidirectional_379/forward_lstm_379/while/Lessу
@sequential_379/bidirectional_379/forward_lstm_379/while/IdentityIdentity@sequential_379/bidirectional_379/forward_lstm_379/while/Less:z:0*
T0
*
_output_shapes
: 2B
@sequential_379/bidirectional_379/forward_lstm_379/while/Identity"Н
@sequential_379_bidirectional_379_forward_lstm_379_while_identityIsequential_379/bidirectional_379/forward_lstm_379/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
ђ&
€
while_body_42658771
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_lstm_cell_1139_42658795_0:	»2
while_lstm_cell_1139_42658797_0:	2».
while_lstm_cell_1139_42658799_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_lstm_cell_1139_42658795:	»0
while_lstm_cell_1139_42658797:	2»,
while_lstm_cell_1139_42658799:	»ИҐ,while/lstm_cell_1139/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemх
,while/lstm_cell_1139/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1139_42658795_0while_lstm_cell_1139_42658797_0while_lstm_cell_1139_42658799_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_lstm_cell_1139_layer_call_and_return_conditional_losses_426586912.
,while/lstm_cell_1139/StatefulPartitionedCallщ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder5while/lstm_cell_1139/StatefulPartitionedCall:output:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¶
while/Identity_4Identity5while/lstm_cell_1139/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4¶
while/Identity_5Identity5while/lstm_cell_1139/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5Й

while/NoOpNoOp-^while/lstm_cell_1139/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"@
while_lstm_cell_1139_42658795while_lstm_cell_1139_42658795_0"@
while_lstm_cell_1139_42658797while_lstm_cell_1139_42658797_0"@
while_lstm_cell_1139_42658799while_lstm_cell_1139_42658799_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2\
,while/lstm_cell_1139/StatefulPartitionedCall,while/lstm_cell_1139/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
еH
Я
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_42658840

inputs*
lstm_cell_1139_42658758:	»*
lstm_cell_1139_42658760:	2»&
lstm_cell_1139_42658762:	»
identityИҐ&lstm_cell_1139/StatefulPartitionedCallҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€22
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
B :и2
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
zeros_1/packed/1Й
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
:€€€€€€€€€22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
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
ReverseV2/axisК
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
	ReverseV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeэ
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
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2±
&lstm_cell_1139/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1139_42658758lstm_cell_1139_42658760lstm_cell_1139_42658762*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_lstm_cell_1139_layer_call_and_return_conditional_losses_426586912(
&lstm_cell_1139/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter–
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1139_42658758lstm_cell_1139_42658760lstm_cell_1139_42658762*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42658771*
condR
while_cond_42658770*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
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
:€€€€€€€€€22

Identity
NoOpNoOp'^lstm_cell_1139/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2P
&lstm_cell_1139/StatefulPartitionedCall&lstm_cell_1139/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ѕ@
д
while_body_42659179
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1138_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1138_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1138_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1138_matmul_readvariableop_resource:	»H
5while_lstm_cell_1138_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1138_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1138/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1138/MatMul/ReadVariableOpҐ,while/lstm_cell_1138/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€29
7while/TensorArrayV2Read/TensorListGetItem/element_shape№
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemѕ
*while/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1138_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1138/MatMul/ReadVariableOpЁ
while/lstm_cell_1138/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/MatMul’
,while/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1138_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1138/MatMul_1/ReadVariableOp∆
while/lstm_cell_1138/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/MatMul_1ј
while/lstm_cell_1138/addAddV2%while/lstm_cell_1138/MatMul:product:0'while/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/addќ
+while/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1138_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1138/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1138/BiasAddBiasAddwhile/lstm_cell_1138/add:z:03while/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/BiasAddО
$while/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1138/split/split_dimУ
while/lstm_cell_1138/splitSplit-while/lstm_cell_1138/split/split_dim:output:0%while/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1138/splitЮ
while/lstm_cell_1138/SigmoidSigmoid#while/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/SigmoidҐ
while/lstm_cell_1138/Sigmoid_1Sigmoid#while/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1138/Sigmoid_1¶
while/lstm_cell_1138/mulMul"while/lstm_cell_1138/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/mulХ
while/lstm_cell_1138/ReluRelu#while/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/ReluЉ
while/lstm_cell_1138/mul_1Mul while/lstm_cell_1138/Sigmoid:y:0'while/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/mul_1±
while/lstm_cell_1138/add_1AddV2while/lstm_cell_1138/mul:z:0while/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/add_1Ґ
while/lstm_cell_1138/Sigmoid_2Sigmoid#while/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1138/Sigmoid_2Ф
while/lstm_cell_1138/Relu_1Reluwhile/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/Relu_1ј
while/lstm_cell_1138/mul_2Mul"while/lstm_cell_1138/Sigmoid_2:y:0)while/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1138/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3П
while/Identity_4Identitywhile/lstm_cell_1138/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1138/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1138/BiasAdd/ReadVariableOp+^while/lstm_cell_1138/MatMul/ReadVariableOp-^while/lstm_cell_1138/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"n
4while_lstm_cell_1138_biasadd_readvariableop_resource6while_lstm_cell_1138_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1138_matmul_1_readvariableop_resource7while_lstm_cell_1138_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1138_matmul_readvariableop_resource5while_lstm_cell_1138_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1138/BiasAdd/ReadVariableOp+while/lstm_cell_1138/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1138/MatMul/ReadVariableOp*while/lstm_cell_1138/MatMul/ReadVariableOp2\
,while/lstm_cell_1138/MatMul_1/ReadVariableOp,while/lstm_cell_1138/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
ђ&
€
while_body_42657927
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_lstm_cell_1138_42657951_0:	»2
while_lstm_cell_1138_42657953_0:	2».
while_lstm_cell_1138_42657955_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_lstm_cell_1138_42657951:	»0
while_lstm_cell_1138_42657953:	2»,
while_lstm_cell_1138_42657955:	»ИҐ,while/lstm_cell_1138/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemх
,while/lstm_cell_1138/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1138_42657951_0while_lstm_cell_1138_42657953_0while_lstm_cell_1138_42657955_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_lstm_cell_1138_layer_call_and_return_conditional_losses_426579132.
,while/lstm_cell_1138/StatefulPartitionedCallщ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder5while/lstm_cell_1138/StatefulPartitionedCall:output:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¶
while/Identity_4Identity5while/lstm_cell_1138/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4¶
while/Identity_5Identity5while/lstm_cell_1138/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5Й

while/NoOpNoOp-^while/lstm_cell_1138/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"@
while_lstm_cell_1138_42657951while_lstm_cell_1138_42657951_0"@
while_lstm_cell_1138_42657953while_lstm_cell_1138_42657953_0"@
while_lstm_cell_1138_42657955while_lstm_cell_1138_42657955_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2\
,while/lstm_cell_1138/StatefulPartitionedCall,while/lstm_cell_1138/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
еH
Я
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_42658628

inputs*
lstm_cell_1139_42658546:	»*
lstm_cell_1139_42658548:	2»&
lstm_cell_1139_42658550:	»
identityИҐ&lstm_cell_1139/StatefulPartitionedCallҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€22
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
B :и2
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
zeros_1/packed/1Й
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
:€€€€€€€€€22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
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
ReverseV2/axisК
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
	ReverseV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeэ
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
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2±
&lstm_cell_1139/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1139_42658546lstm_cell_1139_42658548lstm_cell_1139_42658550*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_lstm_cell_1139_layer_call_and_return_conditional_losses_426585452(
&lstm_cell_1139/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter–
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1139_42658546lstm_cell_1139_42658548lstm_cell_1139_42658550*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42658559*
condR
while_cond_42658558*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
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
:€€€€€€€€€22

Identity
NoOpNoOp'^lstm_cell_1139/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2P
&lstm_cell_1139/StatefulPartitionedCall&lstm_cell_1139/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Њ
ъ
1__inference_lstm_cell_1139_layer_call_fn_42663740

inputs
states_0
states_1
unknown:	»
	unknown_0:	2»
	unknown_1:	»
identity

identity_1

identity_2ИҐStatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_lstm_cell_1139_layer_call_and_return_conditional_losses_426586912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

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
?:€€€€€€€€€:€€€€€€€€€2:€€€€€€€€€2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€2
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€2
"
_user_specified_name
states/1
в
Ѕ
4__inference_backward_lstm_379_layer_call_fn_42662985

inputs
unknown:	»
	unknown_0:	2»
	unknown_1:	»
identityИҐStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_426594232
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
а
ј
3__inference_forward_lstm_379_layer_call_fn_42662337

inputs
unknown:	»
	unknown_0:	2»
	unknown_1:	»
identityИҐStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_426592632
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
÷
√
4__inference_backward_lstm_379_layer_call_fn_42662963
inputs_0
unknown:	»
	unknown_0:	2»
	unknown_1:	»
identityИҐStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_426586282
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
я
Ќ
while_cond_42662414
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_42662414___redundant_placeholder06
2while_while_cond_42662414___redundant_placeholder16
2while_while_cond_42662414___redundant_placeholder26
2while_while_cond_42662414___redundant_placeholder3
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
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:
∆@
д
while_body_42662415
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1138_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1138_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1138_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1138_matmul_readvariableop_resource:	»H
5while_lstm_cell_1138_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1138_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1138/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1138/MatMul/ReadVariableOpҐ,while/lstm_cell_1138/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemѕ
*while/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1138_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1138/MatMul/ReadVariableOpЁ
while/lstm_cell_1138/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/MatMul’
,while/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1138_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1138/MatMul_1/ReadVariableOp∆
while/lstm_cell_1138/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/MatMul_1ј
while/lstm_cell_1138/addAddV2%while/lstm_cell_1138/MatMul:product:0'while/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/addќ
+while/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1138_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1138/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1138/BiasAddBiasAddwhile/lstm_cell_1138/add:z:03while/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/BiasAddО
$while/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1138/split/split_dimУ
while/lstm_cell_1138/splitSplit-while/lstm_cell_1138/split/split_dim:output:0%while/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1138/splitЮ
while/lstm_cell_1138/SigmoidSigmoid#while/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/SigmoidҐ
while/lstm_cell_1138/Sigmoid_1Sigmoid#while/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1138/Sigmoid_1¶
while/lstm_cell_1138/mulMul"while/lstm_cell_1138/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/mulХ
while/lstm_cell_1138/ReluRelu#while/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/ReluЉ
while/lstm_cell_1138/mul_1Mul while/lstm_cell_1138/Sigmoid:y:0'while/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/mul_1±
while/lstm_cell_1138/add_1AddV2while/lstm_cell_1138/mul:z:0while/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/add_1Ґ
while/lstm_cell_1138/Sigmoid_2Sigmoid#while/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1138/Sigmoid_2Ф
while/lstm_cell_1138/Relu_1Reluwhile/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/Relu_1ј
while/lstm_cell_1138/mul_2Mul"while/lstm_cell_1138/Sigmoid_2:y:0)while/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1138/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3П
while/Identity_4Identitywhile/lstm_cell_1138/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1138/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1138/BiasAdd/ReadVariableOp+^while/lstm_cell_1138/MatMul/ReadVariableOp-^while/lstm_cell_1138/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"n
4while_lstm_cell_1138_biasadd_readvariableop_resource6while_lstm_cell_1138_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1138_matmul_1_readvariableop_resource7while_lstm_cell_1138_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1138_matmul_readvariableop_resource5while_lstm_cell_1138_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1138/BiasAdd/ReadVariableOp+while/lstm_cell_1138/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1138/MatMul/ReadVariableOp*while/lstm_cell_1138/MatMul/ReadVariableOp2\
,while/lstm_cell_1138/MatMul_1/ReadVariableOp,while/lstm_cell_1138/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
€
К
L__inference_lstm_cell_1138_layer_call_and_return_conditional_losses_42663674

inputs
states_0
states_11
matmul_readvariableop_resource:	»3
 matmul_1_readvariableop_resource:	2».
biasadd_readvariableop_resource:	»
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	»*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_2Щ
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
?:€€€€€€€€€:€€€€€€€€€2:€€€€€€€€€2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€2
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€2
"
_user_specified_name
states/1
ђ&
€
while_body_42658137
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_lstm_cell_1138_42658161_0:	»2
while_lstm_cell_1138_42658163_0:	2».
while_lstm_cell_1138_42658165_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_lstm_cell_1138_42658161:	»0
while_lstm_cell_1138_42658163:	2»,
while_lstm_cell_1138_42658165:	»ИҐ,while/lstm_cell_1138/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemх
,while/lstm_cell_1138/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1138_42658161_0while_lstm_cell_1138_42658163_0while_lstm_cell_1138_42658165_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_lstm_cell_1138_layer_call_and_return_conditional_losses_426580592.
,while/lstm_cell_1138/StatefulPartitionedCallщ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder5while/lstm_cell_1138/StatefulPartitionedCall:output:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¶
while/Identity_4Identity5while/lstm_cell_1138/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4¶
while/Identity_5Identity5while/lstm_cell_1138/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5Й

while/NoOpNoOp-^while/lstm_cell_1138/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"@
while_lstm_cell_1138_42658161while_lstm_cell_1138_42658161_0"@
while_lstm_cell_1138_42658163while_lstm_cell_1138_42658163_0"@
while_lstm_cell_1138_42658165while_lstm_cell_1138_42658165_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2\
,while/lstm_cell_1138/StatefulPartitionedCall,while/lstm_cell_1138/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
к
Љ
%backward_lstm_379_while_cond_42662186@
<backward_lstm_379_while_backward_lstm_379_while_loop_counterF
Bbackward_lstm_379_while_backward_lstm_379_while_maximum_iterations'
#backward_lstm_379_while_placeholder)
%backward_lstm_379_while_placeholder_1)
%backward_lstm_379_while_placeholder_2)
%backward_lstm_379_while_placeholder_3)
%backward_lstm_379_while_placeholder_4B
>backward_lstm_379_while_less_backward_lstm_379_strided_slice_1Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42662186___redundant_placeholder0Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42662186___redundant_placeholder1Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42662186___redundant_placeholder2Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42662186___redundant_placeholder3Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42662186___redundant_placeholder4$
 backward_lstm_379_while_identity
 
backward_lstm_379/while/LessLess#backward_lstm_379_while_placeholder>backward_lstm_379_while_less_backward_lstm_379_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_379/while/LessУ
 backward_lstm_379/while/IdentityIdentity backward_lstm_379/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_379/while/Identity"M
 backward_lstm_379_while_identity)backward_lstm_379/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
к
Љ
%backward_lstm_379_while_cond_42660616@
<backward_lstm_379_while_backward_lstm_379_while_loop_counterF
Bbackward_lstm_379_while_backward_lstm_379_while_maximum_iterations'
#backward_lstm_379_while_placeholder)
%backward_lstm_379_while_placeholder_1)
%backward_lstm_379_while_placeholder_2)
%backward_lstm_379_while_placeholder_3)
%backward_lstm_379_while_placeholder_4B
>backward_lstm_379_while_less_backward_lstm_379_strided_slice_1Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42660616___redundant_placeholder0Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42660616___redundant_placeholder1Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42660616___redundant_placeholder2Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42660616___redundant_placeholder3Z
Vbackward_lstm_379_while_backward_lstm_379_while_cond_42660616___redundant_placeholder4$
 backward_lstm_379_while_identity
 
backward_lstm_379/while/LessLess#backward_lstm_379_while_placeholder>backward_lstm_379_while_less_backward_lstm_379_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_379/while/LessУ
 backward_lstm_379/while/IdentityIdentity backward_lstm_379/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_379/while/Identity"M
 backward_lstm_379_while_identity)backward_lstm_379/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
я
Ќ
while_cond_42663064
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_42663064___redundant_placeholder06
2while_while_cond_42663064___redundant_placeholder16
2while_while_cond_42663064___redundant_placeholder26
2while_while_cond_42663064___redundant_placeholder3
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
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:
Є
£
L__inference_sequential_379_layer_call_and_return_conditional_losses_42660864

inputs
inputs_1	-
bidirectional_379_42660845:	»-
bidirectional_379_42660847:	2»)
bidirectional_379_42660849:	»-
bidirectional_379_42660851:	»-
bidirectional_379_42660853:	2»)
bidirectional_379_42660855:	»$
dense_379_42660858:d 
dense_379_42660860:
identityИҐ)bidirectional_379/StatefulPartitionedCallҐ!dense_379/StatefulPartitionedCall 
)bidirectional_379/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_379_42660845bidirectional_379_42660847bidirectional_379_42660849bidirectional_379_42660851bidirectional_379_42660853bidirectional_379_42660855*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_426607142+
)bidirectional_379/StatefulPartitionedCallЋ
!dense_379/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_379/StatefulPartitionedCall:output:0dense_379_42660858dense_379_42660860*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dense_379_layer_call_and_return_conditional_losses_426602992#
!dense_379/StatefulPartitionedCallЕ
IdentityIdentity*dense_379/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityЮ
NoOpNoOp*^bidirectional_379/StatefulPartitionedCall"^dense_379/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€:€€€€€€€€€: : : : : : : : 2V
)bidirectional_379/StatefulPartitionedCall)bidirectional_379/StatefulPartitionedCall2F
!dense_379/StatefulPartitionedCall!dense_379/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Т
‘
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_42659434

inputs,
forward_lstm_379_42659264:	»,
forward_lstm_379_42659266:	2»(
forward_lstm_379_42659268:	»-
backward_lstm_379_42659424:	»-
backward_lstm_379_42659426:	2»)
backward_lstm_379_42659428:	»
identityИҐ)backward_lstm_379/StatefulPartitionedCallҐ(forward_lstm_379/StatefulPartitionedCallя
(forward_lstm_379/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_379_42659264forward_lstm_379_42659266forward_lstm_379_42659268*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_426592632*
(forward_lstm_379/StatefulPartitionedCallе
)backward_lstm_379/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_379_42659424backward_lstm_379_42659426backward_lstm_379_42659428*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_426594232+
)backward_lstm_379/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis‘
concatConcatV21forward_lstm_379/StatefulPartitionedCall:output:02backward_lstm_379/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d2

Identity•
NoOpNoOp*^backward_lstm_379/StatefulPartitionedCall)^forward_lstm_379/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : 2V
)backward_lstm_379/StatefulPartitionedCall)backward_lstm_379/StatefulPartitionedCall2T
(forward_lstm_379/StatefulPartitionedCall(forward_lstm_379/StatefulPartitionedCall:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Њ
ъ
1__inference_lstm_cell_1138_layer_call_fn_42663642

inputs
states_0
states_1
unknown:	»
	unknown_0:	2»
	unknown_1:	»
identity

identity_1

identity_2ИҐStatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_lstm_cell_1138_layer_call_and_return_conditional_losses_426580592
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

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
?:€€€€€€€€€:€€€€€€€€€2:€€€€€€€€€2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€2
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€2
"
_user_specified_name
states/1
ѕ@
д
while_body_42659704
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1138_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1138_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1138_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1138_matmul_readvariableop_resource:	»H
5while_lstm_cell_1138_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1138_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1138/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1138/MatMul/ReadVariableOpҐ,while/lstm_cell_1138/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€29
7while/TensorArrayV2Read/TensorListGetItem/element_shape№
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemѕ
*while/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1138_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1138/MatMul/ReadVariableOpЁ
while/lstm_cell_1138/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/MatMul’
,while/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1138_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1138/MatMul_1/ReadVariableOp∆
while/lstm_cell_1138/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/MatMul_1ј
while/lstm_cell_1138/addAddV2%while/lstm_cell_1138/MatMul:product:0'while/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/addќ
+while/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1138_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1138/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1138/BiasAddBiasAddwhile/lstm_cell_1138/add:z:03while/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1138/BiasAddО
$while/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1138/split/split_dimУ
while/lstm_cell_1138/splitSplit-while/lstm_cell_1138/split/split_dim:output:0%while/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1138/splitЮ
while/lstm_cell_1138/SigmoidSigmoid#while/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/SigmoidҐ
while/lstm_cell_1138/Sigmoid_1Sigmoid#while/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1138/Sigmoid_1¶
while/lstm_cell_1138/mulMul"while/lstm_cell_1138/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/mulХ
while/lstm_cell_1138/ReluRelu#while/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/ReluЉ
while/lstm_cell_1138/mul_1Mul while/lstm_cell_1138/Sigmoid:y:0'while/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/mul_1±
while/lstm_cell_1138/add_1AddV2while/lstm_cell_1138/mul:z:0while/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/add_1Ґ
while/lstm_cell_1138/Sigmoid_2Sigmoid#while/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1138/Sigmoid_2Ф
while/lstm_cell_1138/Relu_1Reluwhile/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/Relu_1ј
while/lstm_cell_1138/mul_2Mul"while/lstm_cell_1138/Sigmoid_2:y:0)while/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1138/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1138/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3П
while/Identity_4Identitywhile/lstm_cell_1138/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1138/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1138/BiasAdd/ReadVariableOp+^while/lstm_cell_1138/MatMul/ReadVariableOp-^while/lstm_cell_1138/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"n
4while_lstm_cell_1138_biasadd_readvariableop_resource6while_lstm_cell_1138_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1138_matmul_1_readvariableop_resource7while_lstm_cell_1138_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1138_matmul_readvariableop_resource5while_lstm_cell_1138_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1138/BiasAdd/ReadVariableOp+while/lstm_cell_1138/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1138/MatMul/ReadVariableOp*while/lstm_cell_1138/MatMul/ReadVariableOp2\
,while/lstm_cell_1138/MatMul_1/ReadVariableOp,while/lstm_cell_1138/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
∆@
д
while_body_42663218
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1139_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1139_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1139_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1139_matmul_readvariableop_resource:	»H
5while_lstm_cell_1139_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1139_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1139/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1139/MatMul/ReadVariableOpҐ,while/lstm_cell_1139/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemѕ
*while/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1139_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1139/MatMul/ReadVariableOpЁ
while/lstm_cell_1139/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/MatMul’
,while/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1139_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1139/MatMul_1/ReadVariableOp∆
while/lstm_cell_1139/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/MatMul_1ј
while/lstm_cell_1139/addAddV2%while/lstm_cell_1139/MatMul:product:0'while/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/addќ
+while/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1139_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1139/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1139/BiasAddBiasAddwhile/lstm_cell_1139/add:z:03while/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/BiasAddО
$while/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1139/split/split_dimУ
while/lstm_cell_1139/splitSplit-while/lstm_cell_1139/split/split_dim:output:0%while/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1139/splitЮ
while/lstm_cell_1139/SigmoidSigmoid#while/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/SigmoidҐ
while/lstm_cell_1139/Sigmoid_1Sigmoid#while/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1139/Sigmoid_1¶
while/lstm_cell_1139/mulMul"while/lstm_cell_1139/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/mulХ
while/lstm_cell_1139/ReluRelu#while/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/ReluЉ
while/lstm_cell_1139/mul_1Mul while/lstm_cell_1139/Sigmoid:y:0'while/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/mul_1±
while/lstm_cell_1139/add_1AddV2while/lstm_cell_1139/mul:z:0while/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/add_1Ґ
while/lstm_cell_1139/Sigmoid_2Sigmoid#while/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1139/Sigmoid_2Ф
while/lstm_cell_1139/Relu_1Reluwhile/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/Relu_1ј
while/lstm_cell_1139/mul_2Mul"while/lstm_cell_1139/Sigmoid_2:y:0)while/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1139/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3П
while/Identity_4Identitywhile/lstm_cell_1139/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1139/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1139/BiasAdd/ReadVariableOp+^while/lstm_cell_1139/MatMul/ReadVariableOp-^while/lstm_cell_1139/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"n
4while_lstm_cell_1139_biasadd_readvariableop_resource6while_lstm_cell_1139_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1139_matmul_1_readvariableop_resource7while_lstm_cell_1139_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1139_matmul_readvariableop_resource5while_lstm_cell_1139_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1139/BiasAdd/ReadVariableOp+while/lstm_cell_1139/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1139/MatMul/ReadVariableOp*while/lstm_cell_1139/MatMul/ReadVariableOp2\
,while/lstm_cell_1139/MatMul_1/ReadVariableOp,while/lstm_cell_1139/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
я
Ќ
while_cond_42663523
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_42663523___redundant_placeholder06
2while_while_cond_42663523___redundant_placeholder16
2while_while_cond_42663523___redundant_placeholder26
2while_while_cond_42663523___redundant_placeholder3
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
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:
ѕ@
д
while_body_42663524
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1139_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1139_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1139_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1139_matmul_readvariableop_resource:	»H
5while_lstm_cell_1139_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1139_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1139/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1139/MatMul/ReadVariableOpҐ,while/lstm_cell_1139/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€29
7while/TensorArrayV2Read/TensorListGetItem/element_shape№
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemѕ
*while/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1139_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1139/MatMul/ReadVariableOpЁ
while/lstm_cell_1139/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/MatMul’
,while/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1139_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1139/MatMul_1/ReadVariableOp∆
while/lstm_cell_1139/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/MatMul_1ј
while/lstm_cell_1139/addAddV2%while/lstm_cell_1139/MatMul:product:0'while/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/addќ
+while/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1139_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1139/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1139/BiasAddBiasAddwhile/lstm_cell_1139/add:z:03while/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/BiasAddО
$while/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1139/split/split_dimУ
while/lstm_cell_1139/splitSplit-while/lstm_cell_1139/split/split_dim:output:0%while/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1139/splitЮ
while/lstm_cell_1139/SigmoidSigmoid#while/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/SigmoidҐ
while/lstm_cell_1139/Sigmoid_1Sigmoid#while/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1139/Sigmoid_1¶
while/lstm_cell_1139/mulMul"while/lstm_cell_1139/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/mulХ
while/lstm_cell_1139/ReluRelu#while/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/ReluЉ
while/lstm_cell_1139/mul_1Mul while/lstm_cell_1139/Sigmoid:y:0'while/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/mul_1±
while/lstm_cell_1139/add_1AddV2while/lstm_cell_1139/mul:z:0while/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/add_1Ґ
while/lstm_cell_1139/Sigmoid_2Sigmoid#while/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1139/Sigmoid_2Ф
while/lstm_cell_1139/Relu_1Reluwhile/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/Relu_1ј
while/lstm_cell_1139/mul_2Mul"while/lstm_cell_1139/Sigmoid_2:y:0)while/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1139/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3П
while/Identity_4Identitywhile/lstm_cell_1139/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1139/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1139/BiasAdd/ReadVariableOp+^while/lstm_cell_1139/MatMul/ReadVariableOp-^while/lstm_cell_1139/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"n
4while_lstm_cell_1139_biasadd_readvariableop_resource6while_lstm_cell_1139_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1139_matmul_1_readvariableop_resource7while_lstm_cell_1139_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1139_matmul_readvariableop_resource5while_lstm_cell_1139_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1139/BiasAdd/ReadVariableOp+while/lstm_cell_1139/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1139/MatMul/ReadVariableOp*while/lstm_cell_1139/MatMul/ReadVariableOp2\
,while/lstm_cell_1139/MatMul_1/ReadVariableOp,while/lstm_cell_1139/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
∞Њ
ы
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_42661926

inputs
inputs_1	Q
>forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource:	»S
@forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource:	2»N
?forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource:	»R
?backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resource:	»T
Abackward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource:	2»O
@backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource:	»
identityИҐ7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpҐ6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpҐ8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpҐbackward_lstm_379/whileҐ6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpҐ5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpҐ7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpҐforward_lstm_379/whileЧ
%forward_lstm_379/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_379/RaggedToTensor/zerosЩ
%forward_lstm_379/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2'
%forward_lstm_379/RaggedToTensor/ConstЩ
4forward_lstm_379/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_379/RaggedToTensor/Const:output:0inputs.forward_lstm_379/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_379/RaggedToTensor/RaggedTensorToTensorƒ
;forward_lstm_379/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack»
=forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1»
=forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2©
5forward_lstm_379/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_379/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask27
5forward_lstm_379/RaggedNestedRowLengths/strided_slice»
=forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack’
?forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2A
?forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1ћ
?forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2µ
7forward_lstm_379/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask29
7forward_lstm_379/RaggedNestedRowLengths/strided_slice_1С
+forward_lstm_379/RaggedNestedRowLengths/subSub>forward_lstm_379/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_379/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2-
+forward_lstm_379/RaggedNestedRowLengths/sub§
forward_lstm_379/CastCast/forward_lstm_379/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
forward_lstm_379/CastЭ
forward_lstm_379/ShapeShape=forward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_379/ShapeЦ
$forward_lstm_379/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_379/strided_slice/stackЪ
&forward_lstm_379/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_379/strided_slice/stack_1Ъ
&forward_lstm_379/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_379/strided_slice/stack_2»
forward_lstm_379/strided_sliceStridedSliceforward_lstm_379/Shape:output:0-forward_lstm_379/strided_slice/stack:output:0/forward_lstm_379/strided_slice/stack_1:output:0/forward_lstm_379/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_379/strided_slice~
forward_lstm_379/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_379/zeros/mul/y∞
forward_lstm_379/zeros/mulMul'forward_lstm_379/strided_slice:output:0%forward_lstm_379/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros/mulБ
forward_lstm_379/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
forward_lstm_379/zeros/Less/yЂ
forward_lstm_379/zeros/LessLessforward_lstm_379/zeros/mul:z:0&forward_lstm_379/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros/LessД
forward_lstm_379/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_379/zeros/packed/1«
forward_lstm_379/zeros/packedPack'forward_lstm_379/strided_slice:output:0(forward_lstm_379/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_379/zeros/packedЕ
forward_lstm_379/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_379/zeros/Constє
forward_lstm_379/zerosFill&forward_lstm_379/zeros/packed:output:0%forward_lstm_379/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/zerosВ
forward_lstm_379/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_379/zeros_1/mul/yґ
forward_lstm_379/zeros_1/mulMul'forward_lstm_379/strided_slice:output:0'forward_lstm_379/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros_1/mulЕ
forward_lstm_379/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2!
forward_lstm_379/zeros_1/Less/y≥
forward_lstm_379/zeros_1/LessLess forward_lstm_379/zeros_1/mul:z:0(forward_lstm_379/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros_1/LessИ
!forward_lstm_379/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_379/zeros_1/packed/1Ќ
forward_lstm_379/zeros_1/packedPack'forward_lstm_379/strided_slice:output:0*forward_lstm_379/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_379/zeros_1/packedЙ
forward_lstm_379/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_379/zeros_1/ConstЅ
forward_lstm_379/zeros_1Fill(forward_lstm_379/zeros_1/packed:output:0'forward_lstm_379/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/zeros_1Ч
forward_lstm_379/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_379/transpose/permн
forward_lstm_379/transpose	Transpose=forward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_379/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
forward_lstm_379/transposeВ
forward_lstm_379/Shape_1Shapeforward_lstm_379/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_379/Shape_1Ъ
&forward_lstm_379/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_379/strided_slice_1/stackЮ
(forward_lstm_379/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_1/stack_1Ю
(forward_lstm_379/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_1/stack_2‘
 forward_lstm_379/strided_slice_1StridedSlice!forward_lstm_379/Shape_1:output:0/forward_lstm_379/strided_slice_1/stack:output:01forward_lstm_379/strided_slice_1/stack_1:output:01forward_lstm_379/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_379/strided_slice_1І
,forward_lstm_379/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,forward_lstm_379/TensorArrayV2/element_shapeц
forward_lstm_379/TensorArrayV2TensorListReserve5forward_lstm_379/TensorArrayV2/element_shape:output:0)forward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_379/TensorArrayV2б
Fforward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2H
Fforward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeЉ
8forward_lstm_379/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_379/transpose:y:0Oforward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_379/TensorArrayUnstack/TensorListFromTensorЪ
&forward_lstm_379/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_379/strided_slice_2/stackЮ
(forward_lstm_379/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_2/stack_1Ю
(forward_lstm_379/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_2/stack_2в
 forward_lstm_379/strided_slice_2StridedSliceforward_lstm_379/transpose:y:0/forward_lstm_379/strided_slice_2/stack:output:01forward_lstm_379/strided_slice_2/stack_1:output:01forward_lstm_379/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2"
 forward_lstm_379/strided_slice_2о
5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOp>forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype027
5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpч
&forward_lstm_379/lstm_cell_1138/MatMulMatMul)forward_lstm_379/strided_slice_2:output:0=forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2(
&forward_lstm_379/lstm_cell_1138/MatMulф
7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOp@forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype029
7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpу
(forward_lstm_379/lstm_cell_1138/MatMul_1MatMulforward_lstm_379/zeros:output:0?forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(forward_lstm_379/lstm_cell_1138/MatMul_1м
#forward_lstm_379/lstm_cell_1138/addAddV20forward_lstm_379/lstm_cell_1138/MatMul:product:02forward_lstm_379/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2%
#forward_lstm_379/lstm_cell_1138/addн
6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOp?forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype028
6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpщ
'forward_lstm_379/lstm_cell_1138/BiasAddBiasAdd'forward_lstm_379/lstm_cell_1138/add:z:0>forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'forward_lstm_379/lstm_cell_1138/BiasAdd§
/forward_lstm_379/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/forward_lstm_379/lstm_cell_1138/split/split_dimњ
%forward_lstm_379/lstm_cell_1138/splitSplit8forward_lstm_379/lstm_cell_1138/split/split_dim:output:00forward_lstm_379/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2'
%forward_lstm_379/lstm_cell_1138/splitњ
'forward_lstm_379/lstm_cell_1138/SigmoidSigmoid.forward_lstm_379/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'forward_lstm_379/lstm_cell_1138/Sigmoid√
)forward_lstm_379/lstm_cell_1138/Sigmoid_1Sigmoid.forward_lstm_379/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_379/lstm_cell_1138/Sigmoid_1’
#forward_lstm_379/lstm_cell_1138/mulMul-forward_lstm_379/lstm_cell_1138/Sigmoid_1:y:0!forward_lstm_379/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22%
#forward_lstm_379/lstm_cell_1138/mulґ
$forward_lstm_379/lstm_cell_1138/ReluRelu.forward_lstm_379/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22&
$forward_lstm_379/lstm_cell_1138/Reluи
%forward_lstm_379/lstm_cell_1138/mul_1Mul+forward_lstm_379/lstm_cell_1138/Sigmoid:y:02forward_lstm_379/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_379/lstm_cell_1138/mul_1Ё
%forward_lstm_379/lstm_cell_1138/add_1AddV2'forward_lstm_379/lstm_cell_1138/mul:z:0)forward_lstm_379/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_379/lstm_cell_1138/add_1√
)forward_lstm_379/lstm_cell_1138/Sigmoid_2Sigmoid.forward_lstm_379/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_379/lstm_cell_1138/Sigmoid_2µ
&forward_lstm_379/lstm_cell_1138/Relu_1Relu)forward_lstm_379/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&forward_lstm_379/lstm_cell_1138/Relu_1м
%forward_lstm_379/lstm_cell_1138/mul_2Mul-forward_lstm_379/lstm_cell_1138/Sigmoid_2:y:04forward_lstm_379/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_379/lstm_cell_1138/mul_2±
.forward_lstm_379/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   20
.forward_lstm_379/TensorArrayV2_1/element_shapeь
 forward_lstm_379/TensorArrayV2_1TensorListReserve7forward_lstm_379/TensorArrayV2_1/element_shape:output:0)forward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_379/TensorArrayV2_1p
forward_lstm_379/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_379/time§
forward_lstm_379/zeros_like	ZerosLike)forward_lstm_379/lstm_cell_1138/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/zeros_like°
)forward_lstm_379/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2+
)forward_lstm_379/while/maximum_iterationsМ
#forward_lstm_379/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_379/while/loop_counterЦ	
forward_lstm_379/whileWhile,forward_lstm_379/while/loop_counter:output:02forward_lstm_379/while/maximum_iterations:output:0forward_lstm_379/time:output:0)forward_lstm_379/TensorArrayV2_1:handle:0forward_lstm_379/zeros_like:y:0forward_lstm_379/zeros:output:0!forward_lstm_379/zeros_1:output:0)forward_lstm_379/strided_slice_1:output:0Hforward_lstm_379/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_379/Cast:y:0>forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource@forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource?forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *0
body(R&
$forward_lstm_379_while_body_42661650*0
cond(R&
$forward_lstm_379_while_cond_42661649*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
forward_lstm_379/while„
Aforward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2C
Aforward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeµ
3forward_lstm_379/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_379/while:output:3Jforward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype025
3forward_lstm_379/TensorArrayV2Stack/TensorListStack£
&forward_lstm_379/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2(
&forward_lstm_379/strided_slice_3/stackЮ
(forward_lstm_379/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_379/strided_slice_3/stack_1Ю
(forward_lstm_379/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_3/stack_2А
 forward_lstm_379/strided_slice_3StridedSlice<forward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_379/strided_slice_3/stack:output:01forward_lstm_379/strided_slice_3/stack_1:output:01forward_lstm_379/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2"
 forward_lstm_379/strided_slice_3Ы
!forward_lstm_379/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_379/transpose_1/permт
forward_lstm_379/transpose_1	Transpose<forward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_379/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
forward_lstm_379/transpose_1И
forward_lstm_379/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_379/runtimeЩ
&backward_lstm_379/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_379/RaggedToTensor/zerosЫ
&backward_lstm_379/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2(
&backward_lstm_379/RaggedToTensor/ConstЭ
5backward_lstm_379/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_379/RaggedToTensor/Const:output:0inputs/backward_lstm_379/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_379/RaggedToTensor/RaggedTensorToTensor∆
<backward_lstm_379/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack 
>backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1 
>backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2Ѓ
6backward_lstm_379/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_379/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_379/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask28
6backward_lstm_379/RaggedNestedRowLengths/strided_slice 
>backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack„
@backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2B
@backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1ќ
@backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2Ї
8backward_lstm_379/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_379/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask2:
8backward_lstm_379/RaggedNestedRowLengths/strided_slice_1Х
,backward_lstm_379/RaggedNestedRowLengths/subSub?backward_lstm_379/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_379/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2.
,backward_lstm_379/RaggedNestedRowLengths/subІ
backward_lstm_379/CastCast0backward_lstm_379/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
backward_lstm_379/Cast†
backward_lstm_379/ShapeShape>backward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_379/ShapeШ
%backward_lstm_379/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_379/strided_slice/stackЬ
'backward_lstm_379/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_379/strided_slice/stack_1Ь
'backward_lstm_379/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_379/strided_slice/stack_2ќ
backward_lstm_379/strided_sliceStridedSlice backward_lstm_379/Shape:output:0.backward_lstm_379/strided_slice/stack:output:00backward_lstm_379/strided_slice/stack_1:output:00backward_lstm_379/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_379/strided_sliceА
backward_lstm_379/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_379/zeros/mul/yі
backward_lstm_379/zeros/mulMul(backward_lstm_379/strided_slice:output:0&backward_lstm_379/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/zeros/mulГ
backward_lstm_379/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
backward_lstm_379/zeros/Less/yѓ
backward_lstm_379/zeros/LessLessbackward_lstm_379/zeros/mul:z:0'backward_lstm_379/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/zeros/LessЖ
 backward_lstm_379/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_379/zeros/packed/1Ћ
backward_lstm_379/zeros/packedPack(backward_lstm_379/strided_slice:output:0)backward_lstm_379/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_379/zeros/packedЗ
backward_lstm_379/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_379/zeros/Constљ
backward_lstm_379/zerosFill'backward_lstm_379/zeros/packed:output:0&backward_lstm_379/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_379/zerosД
backward_lstm_379/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_379/zeros_1/mul/yЇ
backward_lstm_379/zeros_1/mulMul(backward_lstm_379/strided_slice:output:0(backward_lstm_379/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/zeros_1/mulЗ
 backward_lstm_379/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 backward_lstm_379/zeros_1/Less/yЈ
backward_lstm_379/zeros_1/LessLess!backward_lstm_379/zeros_1/mul:z:0)backward_lstm_379/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_379/zeros_1/LessК
"backward_lstm_379/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_379/zeros_1/packed/1—
 backward_lstm_379/zeros_1/packedPack(backward_lstm_379/strided_slice:output:0+backward_lstm_379/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_379/zeros_1/packedЛ
backward_lstm_379/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_379/zeros_1/Const≈
backward_lstm_379/zeros_1Fill)backward_lstm_379/zeros_1/packed:output:0(backward_lstm_379/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_379/zeros_1Щ
 backward_lstm_379/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_379/transpose/permс
backward_lstm_379/transpose	Transpose>backward_lstm_379/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_379/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_379/transposeЕ
backward_lstm_379/Shape_1Shapebackward_lstm_379/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_379/Shape_1Ь
'backward_lstm_379/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_379/strided_slice_1/stack†
)backward_lstm_379/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_1/stack_1†
)backward_lstm_379/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_1/stack_2Џ
!backward_lstm_379/strided_slice_1StridedSlice"backward_lstm_379/Shape_1:output:00backward_lstm_379/strided_slice_1/stack:output:02backward_lstm_379/strided_slice_1/stack_1:output:02backward_lstm_379/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_379/strided_slice_1©
-backward_lstm_379/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-backward_lstm_379/TensorArrayV2/element_shapeъ
backward_lstm_379/TensorArrayV2TensorListReserve6backward_lstm_379/TensorArrayV2/element_shape:output:0*backward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_379/TensorArrayV2О
 backward_lstm_379/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_379/ReverseV2/axis“
backward_lstm_379/ReverseV2	ReverseV2backward_lstm_379/transpose:y:0)backward_lstm_379/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_379/ReverseV2г
Gbackward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2I
Gbackward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape≈
9backward_lstm_379/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_379/ReverseV2:output:0Pbackward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_379/TensorArrayUnstack/TensorListFromTensorЬ
'backward_lstm_379/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_379/strided_slice_2/stack†
)backward_lstm_379/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_2/stack_1†
)backward_lstm_379/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_2/stack_2и
!backward_lstm_379/strided_slice_2StridedSlicebackward_lstm_379/transpose:y:00backward_lstm_379/strided_slice_2/stack:output:02backward_lstm_379/strided_slice_2/stack_1:output:02backward_lstm_379/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2#
!backward_lstm_379/strided_slice_2с
6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOp?backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype028
6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpы
'backward_lstm_379/lstm_cell_1139/MatMulMatMul*backward_lstm_379/strided_slice_2:output:0>backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'backward_lstm_379/lstm_cell_1139/MatMulч
8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOpAbackward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02:
8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpч
)backward_lstm_379/lstm_cell_1139/MatMul_1MatMul backward_lstm_379/zeros:output:0@backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)backward_lstm_379/lstm_cell_1139/MatMul_1р
$backward_lstm_379/lstm_cell_1139/addAddV21backward_lstm_379/lstm_cell_1139/MatMul:product:03backward_lstm_379/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2&
$backward_lstm_379/lstm_cell_1139/addр
7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOp@backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype029
7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpэ
(backward_lstm_379/lstm_cell_1139/BiasAddBiasAdd(backward_lstm_379/lstm_cell_1139/add:z:0?backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(backward_lstm_379/lstm_cell_1139/BiasAdd¶
0backward_lstm_379/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0backward_lstm_379/lstm_cell_1139/split/split_dim√
&backward_lstm_379/lstm_cell_1139/splitSplit9backward_lstm_379/lstm_cell_1139/split/split_dim:output:01backward_lstm_379/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2(
&backward_lstm_379/lstm_cell_1139/split¬
(backward_lstm_379/lstm_cell_1139/SigmoidSigmoid/backward_lstm_379/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(backward_lstm_379/lstm_cell_1139/Sigmoid∆
*backward_lstm_379/lstm_cell_1139/Sigmoid_1Sigmoid/backward_lstm_379/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_379/lstm_cell_1139/Sigmoid_1ў
$backward_lstm_379/lstm_cell_1139/mulMul.backward_lstm_379/lstm_cell_1139/Sigmoid_1:y:0"backward_lstm_379/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$backward_lstm_379/lstm_cell_1139/mulє
%backward_lstm_379/lstm_cell_1139/ReluRelu/backward_lstm_379/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22'
%backward_lstm_379/lstm_cell_1139/Reluм
&backward_lstm_379/lstm_cell_1139/mul_1Mul,backward_lstm_379/lstm_cell_1139/Sigmoid:y:03backward_lstm_379/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_379/lstm_cell_1139/mul_1б
&backward_lstm_379/lstm_cell_1139/add_1AddV2(backward_lstm_379/lstm_cell_1139/mul:z:0*backward_lstm_379/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_379/lstm_cell_1139/add_1∆
*backward_lstm_379/lstm_cell_1139/Sigmoid_2Sigmoid/backward_lstm_379/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_379/lstm_cell_1139/Sigmoid_2Є
'backward_lstm_379/lstm_cell_1139/Relu_1Relu*backward_lstm_379/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'backward_lstm_379/lstm_cell_1139/Relu_1р
&backward_lstm_379/lstm_cell_1139/mul_2Mul.backward_lstm_379/lstm_cell_1139/Sigmoid_2:y:05backward_lstm_379/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_379/lstm_cell_1139/mul_2≥
/backward_lstm_379/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   21
/backward_lstm_379/TensorArrayV2_1/element_shapeА
!backward_lstm_379/TensorArrayV2_1TensorListReserve8backward_lstm_379/TensorArrayV2_1/element_shape:output:0*backward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_379/TensorArrayV2_1r
backward_lstm_379/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_379/timeФ
'backward_lstm_379/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_379/Max/reduction_indices§
backward_lstm_379/MaxMaxbackward_lstm_379/Cast:y:00backward_lstm_379/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/Maxt
backward_lstm_379/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_379/sub/yШ
backward_lstm_379/subSubbackward_lstm_379/Max:output:0 backward_lstm_379/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/subЮ
backward_lstm_379/Sub_1Subbackward_lstm_379/sub:z:0backward_lstm_379/Cast:y:0*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_379/Sub_1І
backward_lstm_379/zeros_like	ZerosLike*backward_lstm_379/lstm_cell_1139/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_379/zeros_like£
*backward_lstm_379/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2,
*backward_lstm_379/while/maximum_iterationsО
$backward_lstm_379/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_379/while/loop_counter®	
backward_lstm_379/whileWhile-backward_lstm_379/while/loop_counter:output:03backward_lstm_379/while/maximum_iterations:output:0backward_lstm_379/time:output:0*backward_lstm_379/TensorArrayV2_1:handle:0 backward_lstm_379/zeros_like:y:0 backward_lstm_379/zeros:output:0"backward_lstm_379/zeros_1:output:0*backward_lstm_379/strided_slice_1:output:0Ibackward_lstm_379/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_379/Sub_1:z:0?backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resourceAbackward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource@backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *1
body)R'
%backward_lstm_379_while_body_42661829*1
cond)R'
%backward_lstm_379_while_cond_42661828*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
backward_lstm_379/whileў
Bbackward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2D
Bbackward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeє
4backward_lstm_379/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_379/while:output:3Kbackward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype026
4backward_lstm_379/TensorArrayV2Stack/TensorListStack•
'backward_lstm_379/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2)
'backward_lstm_379/strided_slice_3/stack†
)backward_lstm_379/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_379/strided_slice_3/stack_1†
)backward_lstm_379/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_3/stack_2Ж
!backward_lstm_379/strided_slice_3StridedSlice=backward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_379/strided_slice_3/stack:output:02backward_lstm_379/strided_slice_3/stack_1:output:02backward_lstm_379/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2#
!backward_lstm_379/strided_slice_3Э
"backward_lstm_379/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_379/transpose_1/permц
backward_lstm_379/transpose_1	Transpose=backward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_379/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
backward_lstm_379/transpose_1К
backward_lstm_379/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_379/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisƒ
concatConcatV2)forward_lstm_379/strided_slice_3:output:0*backward_lstm_379/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d2

IdentityЏ
NoOpNoOp8^backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp7^backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp9^backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp^backward_lstm_379/while7^forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp6^forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp8^forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp^forward_lstm_379/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€:€€€€€€€€€: : : : : : 2r
7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp2p
6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp2t
8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp22
backward_lstm_379/whilebackward_lstm_379/while2p
6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp2n
5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp2r
7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp20
forward_lstm_379/whileforward_lstm_379/while:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
–]
і
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_42662499
inputs_0@
-lstm_cell_1138_matmul_readvariableop_resource:	»B
/lstm_cell_1138_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1138_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1138/BiasAdd/ReadVariableOpҐ$lstm_cell_1138/MatMul/ReadVariableOpҐ&lstm_cell_1138/MatMul_1/ReadVariableOpҐwhileF
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€22
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
B :и2
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
zeros_1/packed/1Й
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
:€€€€€€€€€22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЕ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2ї
$lstm_cell_1138/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1138_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1138/MatMul/ReadVariableOp≥
lstm_cell_1138/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/MatMulЅ
&lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1138_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1138/MatMul_1/ReadVariableOpѓ
lstm_cell_1138/MatMul_1MatMulzeros:output:0.lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/MatMul_1®
lstm_cell_1138/addAddV2lstm_cell_1138/MatMul:product:0!lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/addЇ
%lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1138_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1138/BiasAdd/ReadVariableOpµ
lstm_cell_1138/BiasAddBiasAddlstm_cell_1138/add:z:0-lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1138/BiasAddВ
lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1138/split/split_dimы
lstm_cell_1138/splitSplit'lstm_cell_1138/split/split_dim:output:0lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1138/splitМ
lstm_cell_1138/SigmoidSigmoidlstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/SigmoidР
lstm_cell_1138/Sigmoid_1Sigmoidlstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Sigmoid_1С
lstm_cell_1138/mulMullstm_cell_1138/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/mulГ
lstm_cell_1138/ReluRelulstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Relu§
lstm_cell_1138/mul_1Mullstm_cell_1138/Sigmoid:y:0!lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/mul_1Щ
lstm_cell_1138/add_1AddV2lstm_cell_1138/mul:z:0lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/add_1Р
lstm_cell_1138/Sigmoid_2Sigmoidlstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Sigmoid_2В
lstm_cell_1138/Relu_1Relulstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/Relu_1®
lstm_cell_1138/mul_2Mullstm_cell_1138/Sigmoid_2:y:0#lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1138/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterХ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1138_matmul_readvariableop_resource/lstm_cell_1138_matmul_1_readvariableop_resource.lstm_cell_1138_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42662415*
condR
while_cond_42662414*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
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
:€€€€€€€€€22

Identityќ
NoOpNoOp&^lstm_cell_1138/BiasAdd/ReadVariableOp%^lstm_cell_1138/MatMul/ReadVariableOp'^lstm_cell_1138/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1138/BiasAdd/ReadVariableOp%lstm_cell_1138/BiasAdd/ReadVariableOp2L
$lstm_cell_1138/MatMul/ReadVariableOp$lstm_cell_1138/MatMul/ReadVariableOp2P
&lstm_cell_1138/MatMul_1/ReadVariableOp&lstm_cell_1138/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
∆@
д
while_body_42663065
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1139_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1139_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1139_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1139_matmul_readvariableop_resource:	»H
5while_lstm_cell_1139_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1139_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1139/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1139/MatMul/ReadVariableOpҐ,while/lstm_cell_1139/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemѕ
*while/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1139_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1139/MatMul/ReadVariableOpЁ
while/lstm_cell_1139/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/MatMul’
,while/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1139_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1139/MatMul_1/ReadVariableOp∆
while/lstm_cell_1139/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/MatMul_1ј
while/lstm_cell_1139/addAddV2%while/lstm_cell_1139/MatMul:product:0'while/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/addќ
+while/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1139_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1139/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1139/BiasAddBiasAddwhile/lstm_cell_1139/add:z:03while/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1139/BiasAddО
$while/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1139/split/split_dimУ
while/lstm_cell_1139/splitSplit-while/lstm_cell_1139/split/split_dim:output:0%while/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1139/splitЮ
while/lstm_cell_1139/SigmoidSigmoid#while/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/SigmoidҐ
while/lstm_cell_1139/Sigmoid_1Sigmoid#while/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1139/Sigmoid_1¶
while/lstm_cell_1139/mulMul"while/lstm_cell_1139/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/mulХ
while/lstm_cell_1139/ReluRelu#while/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/ReluЉ
while/lstm_cell_1139/mul_1Mul while/lstm_cell_1139/Sigmoid:y:0'while/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/mul_1±
while/lstm_cell_1139/add_1AddV2while/lstm_cell_1139/mul:z:0while/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/add_1Ґ
while/lstm_cell_1139/Sigmoid_2Sigmoid#while/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1139/Sigmoid_2Ф
while/lstm_cell_1139/Relu_1Reluwhile/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/Relu_1ј
while/lstm_cell_1139/mul_2Mul"while/lstm_cell_1139/Sigmoid_2:y:0)while/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1139/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1139/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3П
while/Identity_4Identitywhile/lstm_cell_1139/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1139/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1139/BiasAdd/ReadVariableOp+^while/lstm_cell_1139/MatMul/ReadVariableOp-^while/lstm_cell_1139/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"n
4while_lstm_cell_1139_biasadd_readvariableop_resource6while_lstm_cell_1139_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1139_matmul_1_readvariableop_resource7while_lstm_cell_1139_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1139_matmul_readvariableop_resource5while_lstm_cell_1139_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1139/BiasAdd/ReadVariableOp+while/lstm_cell_1139/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1139/MatMul/ReadVariableOp*while/lstm_cell_1139/MatMul/ReadVariableOp2\
,while/lstm_cell_1139/MatMul_1/ReadVariableOp,while/lstm_cell_1139/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
Іg
н
%backward_lstm_379_while_body_42660177@
<backward_lstm_379_while_backward_lstm_379_while_loop_counterF
Bbackward_lstm_379_while_backward_lstm_379_while_maximum_iterations'
#backward_lstm_379_while_placeholder)
%backward_lstm_379_while_placeholder_1)
%backward_lstm_379_while_placeholder_2)
%backward_lstm_379_while_placeholder_3)
%backward_lstm_379_while_placeholder_4?
;backward_lstm_379_while_backward_lstm_379_strided_slice_1_0{
wbackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_379_while_less_backward_lstm_379_sub_1_0Z
Gbackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0:	»\
Ibackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0:	2»W
Hbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0:	»$
 backward_lstm_379_while_identity&
"backward_lstm_379_while_identity_1&
"backward_lstm_379_while_identity_2&
"backward_lstm_379_while_identity_3&
"backward_lstm_379_while_identity_4&
"backward_lstm_379_while_identity_5&
"backward_lstm_379_while_identity_6=
9backward_lstm_379_while_backward_lstm_379_strided_slice_1y
ubackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_379_while_less_backward_lstm_379_sub_1X
Ebackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource:	»Z
Gbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource:	2»U
Fbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource:	»ИҐ=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpҐ<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOpҐ>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpз
Ibackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2K
Ibackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeњ
;backward_lstm_379/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_379_while_placeholderRbackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02=
;backward_lstm_379/while/TensorArrayV2Read/TensorListGetItemѕ
backward_lstm_379/while/LessLess6backward_lstm_379_while_less_backward_lstm_379_sub_1_0#backward_lstm_379_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_379/while/LessЕ
<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOpGbackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02>
<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp•
-backward_lstm_379/while/lstm_cell_1139/MatMulMatMulBbackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem:item:0Dbackward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-backward_lstm_379/while/lstm_cell_1139/MatMulЛ
>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOpIbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02@
>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpО
/backward_lstm_379/while/lstm_cell_1139/MatMul_1MatMul%backward_lstm_379_while_placeholder_3Fbackward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»21
/backward_lstm_379/while/lstm_cell_1139/MatMul_1И
*backward_lstm_379/while/lstm_cell_1139/addAddV27backward_lstm_379/while/lstm_cell_1139/MatMul:product:09backward_lstm_379/while/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2,
*backward_lstm_379/while/lstm_cell_1139/addД
=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOpHbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02?
=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpХ
.backward_lstm_379/while/lstm_cell_1139/BiasAddBiasAdd.backward_lstm_379/while/lstm_cell_1139/add:z:0Ebackward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.backward_lstm_379/while/lstm_cell_1139/BiasAdd≤
6backward_lstm_379/while/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6backward_lstm_379/while/lstm_cell_1139/split/split_dimџ
,backward_lstm_379/while/lstm_cell_1139/splitSplit?backward_lstm_379/while/lstm_cell_1139/split/split_dim:output:07backward_lstm_379/while/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2.
,backward_lstm_379/while/lstm_cell_1139/split‘
.backward_lstm_379/while/lstm_cell_1139/SigmoidSigmoid5backward_lstm_379/while/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€220
.backward_lstm_379/while/lstm_cell_1139/SigmoidЎ
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_1Sigmoid5backward_lstm_379/while/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_1о
*backward_lstm_379/while/lstm_cell_1139/mulMul4backward_lstm_379/while/lstm_cell_1139/Sigmoid_1:y:0%backward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_379/while/lstm_cell_1139/mulЋ
+backward_lstm_379/while/lstm_cell_1139/ReluRelu5backward_lstm_379/while/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22-
+backward_lstm_379/while/lstm_cell_1139/ReluД
,backward_lstm_379/while/lstm_cell_1139/mul_1Mul2backward_lstm_379/while/lstm_cell_1139/Sigmoid:y:09backward_lstm_379/while/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_379/while/lstm_cell_1139/mul_1щ
,backward_lstm_379/while/lstm_cell_1139/add_1AddV2.backward_lstm_379/while/lstm_cell_1139/mul:z:00backward_lstm_379/while/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_379/while/lstm_cell_1139/add_1Ў
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_2Sigmoid5backward_lstm_379/while/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_2 
-backward_lstm_379/while/lstm_cell_1139/Relu_1Relu0backward_lstm_379/while/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-backward_lstm_379/while/lstm_cell_1139/Relu_1И
,backward_lstm_379/while/lstm_cell_1139/mul_2Mul4backward_lstm_379/while/lstm_cell_1139/Sigmoid_2:y:0;backward_lstm_379/while/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_379/while/lstm_cell_1139/mul_2ч
backward_lstm_379/while/SelectSelect backward_lstm_379/while/Less:z:00backward_lstm_379/while/lstm_cell_1139/mul_2:z:0%backward_lstm_379_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22 
backward_lstm_379/while/Selectы
 backward_lstm_379/while/Select_1Select backward_lstm_379/while/Less:z:00backward_lstm_379/while/lstm_cell_1139/mul_2:z:0%backward_lstm_379_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_379/while/Select_1ы
 backward_lstm_379/while/Select_2Select backward_lstm_379/while/Less:z:00backward_lstm_379/while/lstm_cell_1139/add_1:z:0%backward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_379/while/Select_2≥
<backward_lstm_379/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_379_while_placeholder_1#backward_lstm_379_while_placeholder'backward_lstm_379/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_379/while/TensorArrayV2Write/TensorListSetItemА
backward_lstm_379/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_379/while/add/y±
backward_lstm_379/while/addAddV2#backward_lstm_379_while_placeholder&backward_lstm_379/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/while/addД
backward_lstm_379/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_379/while/add_1/y–
backward_lstm_379/while/add_1AddV2<backward_lstm_379_while_backward_lstm_379_while_loop_counter(backward_lstm_379/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/while/add_1≥
 backward_lstm_379/while/IdentityIdentity!backward_lstm_379/while/add_1:z:0^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_379/while/IdentityЎ
"backward_lstm_379/while/Identity_1IdentityBbackward_lstm_379_while_backward_lstm_379_while_maximum_iterations^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_379/while/Identity_1µ
"backward_lstm_379/while/Identity_2Identitybackward_lstm_379/while/add:z:0^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_379/while/Identity_2в
"backward_lstm_379/while/Identity_3IdentityLbackward_lstm_379/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_379/while/Identity_3ќ
"backward_lstm_379/while/Identity_4Identity'backward_lstm_379/while/Select:output:0^backward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_379/while/Identity_4–
"backward_lstm_379/while/Identity_5Identity)backward_lstm_379/while/Select_1:output:0^backward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_379/while/Identity_5–
"backward_lstm_379/while/Identity_6Identity)backward_lstm_379/while/Select_2:output:0^backward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_379/while/Identity_6Њ
backward_lstm_379/while/NoOpNoOp>^backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp=^backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp?^backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_379/while/NoOp"x
9backward_lstm_379_while_backward_lstm_379_strided_slice_1;backward_lstm_379_while_backward_lstm_379_strided_slice_1_0"M
 backward_lstm_379_while_identity)backward_lstm_379/while/Identity:output:0"Q
"backward_lstm_379_while_identity_1+backward_lstm_379/while/Identity_1:output:0"Q
"backward_lstm_379_while_identity_2+backward_lstm_379/while/Identity_2:output:0"Q
"backward_lstm_379_while_identity_3+backward_lstm_379/while/Identity_3:output:0"Q
"backward_lstm_379_while_identity_4+backward_lstm_379/while/Identity_4:output:0"Q
"backward_lstm_379_while_identity_5+backward_lstm_379/while/Identity_5:output:0"Q
"backward_lstm_379_while_identity_6+backward_lstm_379/while/Identity_6:output:0"n
4backward_lstm_379_while_less_backward_lstm_379_sub_16backward_lstm_379_while_less_backward_lstm_379_sub_1_0"Т
Fbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resourceHbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0"Ф
Gbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resourceIbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0"Р
Ebackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resourceGbackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0"р
ubackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2~
=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp2|
<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp2А
>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:€€€€€€€€€
…
•
$forward_lstm_379_while_cond_42662007>
:forward_lstm_379_while_forward_lstm_379_while_loop_counterD
@forward_lstm_379_while_forward_lstm_379_while_maximum_iterations&
"forward_lstm_379_while_placeholder(
$forward_lstm_379_while_placeholder_1(
$forward_lstm_379_while_placeholder_2(
$forward_lstm_379_while_placeholder_3(
$forward_lstm_379_while_placeholder_4@
<forward_lstm_379_while_less_forward_lstm_379_strided_slice_1X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42662007___redundant_placeholder0X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42662007___redundant_placeholder1X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42662007___redundant_placeholder2X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42662007___redundant_placeholder3X
Tforward_lstm_379_while_forward_lstm_379_while_cond_42662007___redundant_placeholder4#
forward_lstm_379_while_identity
≈
forward_lstm_379/while/LessLess"forward_lstm_379_while_placeholder<forward_lstm_379_while_less_forward_lstm_379_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_379/while/LessР
forward_lstm_379/while/IdentityIdentityforward_lstm_379/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_379/while/Identity"K
forward_lstm_379_while_identity(forward_lstm_379/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
я
Ќ
while_cond_42663217
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_42663217___redundant_placeholder06
2while_while_cond_42663217___redundant_placeholder16
2while_while_cond_42663217___redundant_placeholder26
2while_while_cond_42663217___redundant_placeholder3
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
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:
Т
‘
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_42659836

inputs,
forward_lstm_379_42659819:	»,
forward_lstm_379_42659821:	2»(
forward_lstm_379_42659823:	»-
backward_lstm_379_42659826:	»-
backward_lstm_379_42659828:	2»)
backward_lstm_379_42659830:	»
identityИҐ)backward_lstm_379/StatefulPartitionedCallҐ(forward_lstm_379/StatefulPartitionedCallя
(forward_lstm_379/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_379_42659819forward_lstm_379_42659821forward_lstm_379_42659823*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_426597882*
(forward_lstm_379/StatefulPartitionedCallе
)backward_lstm_379/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_379_42659826backward_lstm_379_42659828backward_lstm_379_42659830*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_426596152+
)backward_lstm_379/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis‘
concatConcatV21forward_lstm_379/StatefulPartitionedCall:output:02backward_lstm_379/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d2

Identity•
NoOpNoOp*^backward_lstm_379/StatefulPartitionedCall)^forward_lstm_379/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : 2V
)backward_lstm_379/StatefulPartitionedCall)backward_lstm_379/StatefulPartitionedCall2T
(forward_lstm_379/StatefulPartitionedCall(forward_lstm_379/StatefulPartitionedCall:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
я
Ќ
while_cond_42658558
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_42658558___redundant_placeholder06
2while_while_cond_42658558___redundant_placeholder16
2while_while_cond_42658558___redundant_placeholder26
2while_while_cond_42658558___redundant_placeholder3
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
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:
ЎЯ
џ
Fsequential_379_bidirectional_379_backward_lstm_379_while_body_42657734В
~sequential_379_bidirectional_379_backward_lstm_379_while_sequential_379_bidirectional_379_backward_lstm_379_while_loop_counterЙ
Дsequential_379_bidirectional_379_backward_lstm_379_while_sequential_379_bidirectional_379_backward_lstm_379_while_maximum_iterationsH
Dsequential_379_bidirectional_379_backward_lstm_379_while_placeholderJ
Fsequential_379_bidirectional_379_backward_lstm_379_while_placeholder_1J
Fsequential_379_bidirectional_379_backward_lstm_379_while_placeholder_2J
Fsequential_379_bidirectional_379_backward_lstm_379_while_placeholder_3J
Fsequential_379_bidirectional_379_backward_lstm_379_while_placeholder_4Б
}sequential_379_bidirectional_379_backward_lstm_379_while_sequential_379_bidirectional_379_backward_lstm_379_strided_slice_1_0Њ
єsequential_379_bidirectional_379_backward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_sequential_379_bidirectional_379_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0|
xsequential_379_bidirectional_379_backward_lstm_379_while_less_sequential_379_bidirectional_379_backward_lstm_379_sub_1_0{
hsequential_379_bidirectional_379_backward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0:	»}
jsequential_379_bidirectional_379_backward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0:	2»x
isequential_379_bidirectional_379_backward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0:	»E
Asequential_379_bidirectional_379_backward_lstm_379_while_identityG
Csequential_379_bidirectional_379_backward_lstm_379_while_identity_1G
Csequential_379_bidirectional_379_backward_lstm_379_while_identity_2G
Csequential_379_bidirectional_379_backward_lstm_379_while_identity_3G
Csequential_379_bidirectional_379_backward_lstm_379_while_identity_4G
Csequential_379_bidirectional_379_backward_lstm_379_while_identity_5G
Csequential_379_bidirectional_379_backward_lstm_379_while_identity_6
{sequential_379_bidirectional_379_backward_lstm_379_while_sequential_379_bidirectional_379_backward_lstm_379_strided_slice_1Љ
Јsequential_379_bidirectional_379_backward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_sequential_379_bidirectional_379_backward_lstm_379_tensorarrayunstack_tensorlistfromtensorz
vsequential_379_bidirectional_379_backward_lstm_379_while_less_sequential_379_bidirectional_379_backward_lstm_379_sub_1y
fsequential_379_bidirectional_379_backward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource:	»{
hsequential_379_bidirectional_379_backward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource:	2»v
gsequential_379_bidirectional_379_backward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource:	»ИҐ^sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpҐ]sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOpҐ_sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp©
jsequential_379/bidirectional_379/backward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2l
jsequential_379/bidirectional_379/backward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeЖ
\sequential_379/bidirectional_379/backward_lstm_379/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemєsequential_379_bidirectional_379_backward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_sequential_379_bidirectional_379_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0Dsequential_379_bidirectional_379_backward_lstm_379_while_placeholderssequential_379/bidirectional_379/backward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02^
\sequential_379/bidirectional_379/backward_lstm_379/while/TensorArrayV2Read/TensorListGetItemф
=sequential_379/bidirectional_379/backward_lstm_379/while/LessLessxsequential_379_bidirectional_379_backward_lstm_379_while_less_sequential_379_bidirectional_379_backward_lstm_379_sub_1_0Dsequential_379_bidirectional_379_backward_lstm_379_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2?
=sequential_379/bidirectional_379/backward_lstm_379/while/Lessи
]sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOphsequential_379_bidirectional_379_backward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02_
]sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp©
Nsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMulMatMulcsequential_379/bidirectional_379/backward_lstm_379/while/TensorArrayV2Read/TensorListGetItem:item:0esequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2P
Nsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMulо
_sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOpjsequential_379_bidirectional_379_backward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02a
_sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpТ
Psequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMul_1MatMulFsequential_379_bidirectional_379_backward_lstm_379_while_placeholder_3gsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2R
Psequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMul_1М
Ksequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/addAddV2Xsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMul:product:0Zsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2M
Ksequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/addз
^sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOpisequential_379_bidirectional_379_backward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02`
^sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpЩ
Osequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/BiasAddBiasAddOsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/add:z:0fsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2Q
Osequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/BiasAddф
Wsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2Y
Wsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/split/split_dimя
Msequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/splitSplit`sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/split/split_dim:output:0Xsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2O
Msequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/splitЈ
Osequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/SigmoidSigmoidVsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22Q
Osequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/Sigmoidї
Qsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/Sigmoid_1SigmoidVsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22S
Qsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/Sigmoid_1т
Ksequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/mulMulUsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/Sigmoid_1:y:0Fsequential_379_bidirectional_379_backward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22M
Ksequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/mulЃ
Lsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/ReluReluVsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22N
Lsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/ReluИ
Msequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/mul_1MulSsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/Sigmoid:y:0Zsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22O
Msequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/mul_1э
Msequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/add_1AddV2Osequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/mul:z:0Qsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22O
Msequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/add_1ї
Qsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/Sigmoid_2SigmoidVsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22S
Qsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/Sigmoid_2≠
Nsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/Relu_1ReluQsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22P
Nsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/Relu_1М
Msequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/mul_2MulUsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/Sigmoid_2:y:0\sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22O
Msequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/mul_2Ь
?sequential_379/bidirectional_379/backward_lstm_379/while/SelectSelectAsequential_379/bidirectional_379/backward_lstm_379/while/Less:z:0Qsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/mul_2:z:0Fsequential_379_bidirectional_379_backward_lstm_379_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22A
?sequential_379/bidirectional_379/backward_lstm_379/while/Select†
Asequential_379/bidirectional_379/backward_lstm_379/while/Select_1SelectAsequential_379/bidirectional_379/backward_lstm_379/while/Less:z:0Qsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/mul_2:z:0Fsequential_379_bidirectional_379_backward_lstm_379_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22C
Asequential_379/bidirectional_379/backward_lstm_379/while/Select_1†
Asequential_379/bidirectional_379/backward_lstm_379/while/Select_2SelectAsequential_379/bidirectional_379/backward_lstm_379/while/Less:z:0Qsequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/add_1:z:0Fsequential_379_bidirectional_379_backward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22C
Asequential_379/bidirectional_379/backward_lstm_379/while/Select_2Ў
]sequential_379/bidirectional_379/backward_lstm_379/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemFsequential_379_bidirectional_379_backward_lstm_379_while_placeholder_1Dsequential_379_bidirectional_379_backward_lstm_379_while_placeholderHsequential_379/bidirectional_379/backward_lstm_379/while/Select:output:0*
_output_shapes
: *
element_dtype02_
]sequential_379/bidirectional_379/backward_lstm_379/while/TensorArrayV2Write/TensorListSetItem¬
>sequential_379/bidirectional_379/backward_lstm_379/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_379/bidirectional_379/backward_lstm_379/while/add/yµ
<sequential_379/bidirectional_379/backward_lstm_379/while/addAddV2Dsequential_379_bidirectional_379_backward_lstm_379_while_placeholderGsequential_379/bidirectional_379/backward_lstm_379/while/add/y:output:0*
T0*
_output_shapes
: 2>
<sequential_379/bidirectional_379/backward_lstm_379/while/add∆
@sequential_379/bidirectional_379/backward_lstm_379/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2B
@sequential_379/bidirectional_379/backward_lstm_379/while/add_1/yх
>sequential_379/bidirectional_379/backward_lstm_379/while/add_1AddV2~sequential_379_bidirectional_379_backward_lstm_379_while_sequential_379_bidirectional_379_backward_lstm_379_while_loop_counterIsequential_379/bidirectional_379/backward_lstm_379/while/add_1/y:output:0*
T0*
_output_shapes
: 2@
>sequential_379/bidirectional_379/backward_lstm_379/while/add_1Ј
Asequential_379/bidirectional_379/backward_lstm_379/while/IdentityIdentityBsequential_379/bidirectional_379/backward_lstm_379/while/add_1:z:0>^sequential_379/bidirectional_379/backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2C
Asequential_379/bidirectional_379/backward_lstm_379/while/Identityю
Csequential_379/bidirectional_379/backward_lstm_379/while/Identity_1IdentityДsequential_379_bidirectional_379_backward_lstm_379_while_sequential_379_bidirectional_379_backward_lstm_379_while_maximum_iterations>^sequential_379/bidirectional_379/backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2E
Csequential_379/bidirectional_379/backward_lstm_379/while/Identity_1є
Csequential_379/bidirectional_379/backward_lstm_379/while/Identity_2Identity@sequential_379/bidirectional_379/backward_lstm_379/while/add:z:0>^sequential_379/bidirectional_379/backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2E
Csequential_379/bidirectional_379/backward_lstm_379/while/Identity_2ж
Csequential_379/bidirectional_379/backward_lstm_379/while/Identity_3Identitymsequential_379/bidirectional_379/backward_lstm_379/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential_379/bidirectional_379/backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2E
Csequential_379/bidirectional_379/backward_lstm_379/while/Identity_3“
Csequential_379/bidirectional_379/backward_lstm_379/while/Identity_4IdentityHsequential_379/bidirectional_379/backward_lstm_379/while/Select:output:0>^sequential_379/bidirectional_379/backward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22E
Csequential_379/bidirectional_379/backward_lstm_379/while/Identity_4‘
Csequential_379/bidirectional_379/backward_lstm_379/while/Identity_5IdentityJsequential_379/bidirectional_379/backward_lstm_379/while/Select_1:output:0>^sequential_379/bidirectional_379/backward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22E
Csequential_379/bidirectional_379/backward_lstm_379/while/Identity_5‘
Csequential_379/bidirectional_379/backward_lstm_379/while/Identity_6IdentityJsequential_379/bidirectional_379/backward_lstm_379/while/Select_2:output:0>^sequential_379/bidirectional_379/backward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22E
Csequential_379/bidirectional_379/backward_lstm_379/while/Identity_6г
=sequential_379/bidirectional_379/backward_lstm_379/while/NoOpNoOp_^sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp^^sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp`^sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2?
=sequential_379/bidirectional_379/backward_lstm_379/while/NoOp"П
Asequential_379_bidirectional_379_backward_lstm_379_while_identityJsequential_379/bidirectional_379/backward_lstm_379/while/Identity:output:0"У
Csequential_379_bidirectional_379_backward_lstm_379_while_identity_1Lsequential_379/bidirectional_379/backward_lstm_379/while/Identity_1:output:0"У
Csequential_379_bidirectional_379_backward_lstm_379_while_identity_2Lsequential_379/bidirectional_379/backward_lstm_379/while/Identity_2:output:0"У
Csequential_379_bidirectional_379_backward_lstm_379_while_identity_3Lsequential_379/bidirectional_379/backward_lstm_379/while/Identity_3:output:0"У
Csequential_379_bidirectional_379_backward_lstm_379_while_identity_4Lsequential_379/bidirectional_379/backward_lstm_379/while/Identity_4:output:0"У
Csequential_379_bidirectional_379_backward_lstm_379_while_identity_5Lsequential_379/bidirectional_379/backward_lstm_379/while/Identity_5:output:0"У
Csequential_379_bidirectional_379_backward_lstm_379_while_identity_6Lsequential_379/bidirectional_379/backward_lstm_379/while/Identity_6:output:0"т
vsequential_379_bidirectional_379_backward_lstm_379_while_less_sequential_379_bidirectional_379_backward_lstm_379_sub_1xsequential_379_bidirectional_379_backward_lstm_379_while_less_sequential_379_bidirectional_379_backward_lstm_379_sub_1_0"‘
gsequential_379_bidirectional_379_backward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resourceisequential_379_bidirectional_379_backward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0"÷
hsequential_379_bidirectional_379_backward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resourcejsequential_379_bidirectional_379_backward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0"“
fsequential_379_bidirectional_379_backward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resourcehsequential_379_bidirectional_379_backward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0"ь
{sequential_379_bidirectional_379_backward_lstm_379_while_sequential_379_bidirectional_379_backward_lstm_379_strided_slice_1}sequential_379_bidirectional_379_backward_lstm_379_while_sequential_379_bidirectional_379_backward_lstm_379_strided_slice_1_0"ц
Јsequential_379_bidirectional_379_backward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_sequential_379_bidirectional_379_backward_lstm_379_tensorarrayunstack_tensorlistfromtensorєsequential_379_bidirectional_379_backward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_sequential_379_bidirectional_379_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2ј
^sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp^sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp2Њ
]sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp]sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp2¬
_sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp_sequential_379/bidirectional_379/backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:€€€€€€€€€
ёю
п
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_42661568
inputs_0Q
>forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource:	»S
@forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource:	2»N
?forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource:	»R
?backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resource:	»T
Abackward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource:	2»O
@backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource:	»
identityИҐ7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpҐ6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpҐ8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpҐbackward_lstm_379/whileҐ6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpҐ5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpҐ7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpҐforward_lstm_379/whileh
forward_lstm_379/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_379/ShapeЦ
$forward_lstm_379/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_379/strided_slice/stackЪ
&forward_lstm_379/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_379/strided_slice/stack_1Ъ
&forward_lstm_379/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_379/strided_slice/stack_2»
forward_lstm_379/strided_sliceStridedSliceforward_lstm_379/Shape:output:0-forward_lstm_379/strided_slice/stack:output:0/forward_lstm_379/strided_slice/stack_1:output:0/forward_lstm_379/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_379/strided_slice~
forward_lstm_379/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_379/zeros/mul/y∞
forward_lstm_379/zeros/mulMul'forward_lstm_379/strided_slice:output:0%forward_lstm_379/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros/mulБ
forward_lstm_379/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
forward_lstm_379/zeros/Less/yЂ
forward_lstm_379/zeros/LessLessforward_lstm_379/zeros/mul:z:0&forward_lstm_379/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros/LessД
forward_lstm_379/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_379/zeros/packed/1«
forward_lstm_379/zeros/packedPack'forward_lstm_379/strided_slice:output:0(forward_lstm_379/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_379/zeros/packedЕ
forward_lstm_379/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_379/zeros/Constє
forward_lstm_379/zerosFill&forward_lstm_379/zeros/packed:output:0%forward_lstm_379/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/zerosВ
forward_lstm_379/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_379/zeros_1/mul/yґ
forward_lstm_379/zeros_1/mulMul'forward_lstm_379/strided_slice:output:0'forward_lstm_379/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros_1/mulЕ
forward_lstm_379/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2!
forward_lstm_379/zeros_1/Less/y≥
forward_lstm_379/zeros_1/LessLess forward_lstm_379/zeros_1/mul:z:0(forward_lstm_379/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/zeros_1/LessИ
!forward_lstm_379/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_379/zeros_1/packed/1Ќ
forward_lstm_379/zeros_1/packedPack'forward_lstm_379/strided_slice:output:0*forward_lstm_379/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_379/zeros_1/packedЙ
forward_lstm_379/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_379/zeros_1/ConstЅ
forward_lstm_379/zeros_1Fill(forward_lstm_379/zeros_1/packed:output:0'forward_lstm_379/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_379/zeros_1Ч
forward_lstm_379/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_379/transpose/permЅ
forward_lstm_379/transpose	Transposeinputs_0(forward_lstm_379/transpose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
forward_lstm_379/transposeВ
forward_lstm_379/Shape_1Shapeforward_lstm_379/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_379/Shape_1Ъ
&forward_lstm_379/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_379/strided_slice_1/stackЮ
(forward_lstm_379/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_1/stack_1Ю
(forward_lstm_379/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_1/stack_2‘
 forward_lstm_379/strided_slice_1StridedSlice!forward_lstm_379/Shape_1:output:0/forward_lstm_379/strided_slice_1/stack:output:01forward_lstm_379/strided_slice_1/stack_1:output:01forward_lstm_379/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_379/strided_slice_1І
,forward_lstm_379/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,forward_lstm_379/TensorArrayV2/element_shapeц
forward_lstm_379/TensorArrayV2TensorListReserve5forward_lstm_379/TensorArrayV2/element_shape:output:0)forward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_379/TensorArrayV2б
Fforward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2H
Fforward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeЉ
8forward_lstm_379/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_379/transpose:y:0Oforward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_379/TensorArrayUnstack/TensorListFromTensorЪ
&forward_lstm_379/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_379/strided_slice_2/stackЮ
(forward_lstm_379/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_2/stack_1Ю
(forward_lstm_379/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_2/stack_2л
 forward_lstm_379/strided_slice_2StridedSliceforward_lstm_379/transpose:y:0/forward_lstm_379/strided_slice_2/stack:output:01forward_lstm_379/strided_slice_2/stack_1:output:01forward_lstm_379/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2"
 forward_lstm_379/strided_slice_2о
5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOp>forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype027
5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOpч
&forward_lstm_379/lstm_cell_1138/MatMulMatMul)forward_lstm_379/strided_slice_2:output:0=forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2(
&forward_lstm_379/lstm_cell_1138/MatMulф
7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOp@forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype029
7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOpу
(forward_lstm_379/lstm_cell_1138/MatMul_1MatMulforward_lstm_379/zeros:output:0?forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(forward_lstm_379/lstm_cell_1138/MatMul_1м
#forward_lstm_379/lstm_cell_1138/addAddV20forward_lstm_379/lstm_cell_1138/MatMul:product:02forward_lstm_379/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2%
#forward_lstm_379/lstm_cell_1138/addн
6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOp?forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype028
6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOpщ
'forward_lstm_379/lstm_cell_1138/BiasAddBiasAdd'forward_lstm_379/lstm_cell_1138/add:z:0>forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'forward_lstm_379/lstm_cell_1138/BiasAdd§
/forward_lstm_379/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/forward_lstm_379/lstm_cell_1138/split/split_dimњ
%forward_lstm_379/lstm_cell_1138/splitSplit8forward_lstm_379/lstm_cell_1138/split/split_dim:output:00forward_lstm_379/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2'
%forward_lstm_379/lstm_cell_1138/splitњ
'forward_lstm_379/lstm_cell_1138/SigmoidSigmoid.forward_lstm_379/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'forward_lstm_379/lstm_cell_1138/Sigmoid√
)forward_lstm_379/lstm_cell_1138/Sigmoid_1Sigmoid.forward_lstm_379/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_379/lstm_cell_1138/Sigmoid_1’
#forward_lstm_379/lstm_cell_1138/mulMul-forward_lstm_379/lstm_cell_1138/Sigmoid_1:y:0!forward_lstm_379/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22%
#forward_lstm_379/lstm_cell_1138/mulґ
$forward_lstm_379/lstm_cell_1138/ReluRelu.forward_lstm_379/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22&
$forward_lstm_379/lstm_cell_1138/Reluи
%forward_lstm_379/lstm_cell_1138/mul_1Mul+forward_lstm_379/lstm_cell_1138/Sigmoid:y:02forward_lstm_379/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_379/lstm_cell_1138/mul_1Ё
%forward_lstm_379/lstm_cell_1138/add_1AddV2'forward_lstm_379/lstm_cell_1138/mul:z:0)forward_lstm_379/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_379/lstm_cell_1138/add_1√
)forward_lstm_379/lstm_cell_1138/Sigmoid_2Sigmoid.forward_lstm_379/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_379/lstm_cell_1138/Sigmoid_2µ
&forward_lstm_379/lstm_cell_1138/Relu_1Relu)forward_lstm_379/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&forward_lstm_379/lstm_cell_1138/Relu_1м
%forward_lstm_379/lstm_cell_1138/mul_2Mul-forward_lstm_379/lstm_cell_1138/Sigmoid_2:y:04forward_lstm_379/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_379/lstm_cell_1138/mul_2±
.forward_lstm_379/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   20
.forward_lstm_379/TensorArrayV2_1/element_shapeь
 forward_lstm_379/TensorArrayV2_1TensorListReserve7forward_lstm_379/TensorArrayV2_1/element_shape:output:0)forward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_379/TensorArrayV2_1p
forward_lstm_379/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_379/time°
)forward_lstm_379/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2+
)forward_lstm_379/while/maximum_iterationsМ
#forward_lstm_379/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_379/while/loop_counterФ
forward_lstm_379/whileWhile,forward_lstm_379/while/loop_counter:output:02forward_lstm_379/while/maximum_iterations:output:0forward_lstm_379/time:output:0)forward_lstm_379/TensorArrayV2_1:handle:0forward_lstm_379/zeros:output:0!forward_lstm_379/zeros_1:output:0)forward_lstm_379/strided_slice_1:output:0Hforward_lstm_379/TensorArrayUnstack/TensorListFromTensor:output_handle:0>forward_lstm_379_lstm_cell_1138_matmul_readvariableop_resource@forward_lstm_379_lstm_cell_1138_matmul_1_readvariableop_resource?forward_lstm_379_lstm_cell_1138_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *0
body(R&
$forward_lstm_379_while_body_42661333*0
cond(R&
$forward_lstm_379_while_cond_42661332*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
forward_lstm_379/while„
Aforward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2C
Aforward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeµ
3forward_lstm_379/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_379/while:output:3Jforward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype025
3forward_lstm_379/TensorArrayV2Stack/TensorListStack£
&forward_lstm_379/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2(
&forward_lstm_379/strided_slice_3/stackЮ
(forward_lstm_379/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_379/strided_slice_3/stack_1Ю
(forward_lstm_379/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_379/strided_slice_3/stack_2А
 forward_lstm_379/strided_slice_3StridedSlice<forward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_379/strided_slice_3/stack:output:01forward_lstm_379/strided_slice_3/stack_1:output:01forward_lstm_379/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2"
 forward_lstm_379/strided_slice_3Ы
!forward_lstm_379/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_379/transpose_1/permт
forward_lstm_379/transpose_1	Transpose<forward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_379/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
forward_lstm_379/transpose_1И
forward_lstm_379/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_379/runtimej
backward_lstm_379/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_379/ShapeШ
%backward_lstm_379/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_379/strided_slice/stackЬ
'backward_lstm_379/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_379/strided_slice/stack_1Ь
'backward_lstm_379/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_379/strided_slice/stack_2ќ
backward_lstm_379/strided_sliceStridedSlice backward_lstm_379/Shape:output:0.backward_lstm_379/strided_slice/stack:output:00backward_lstm_379/strided_slice/stack_1:output:00backward_lstm_379/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_379/strided_sliceА
backward_lstm_379/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_379/zeros/mul/yі
backward_lstm_379/zeros/mulMul(backward_lstm_379/strided_slice:output:0&backward_lstm_379/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/zeros/mulГ
backward_lstm_379/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
backward_lstm_379/zeros/Less/yѓ
backward_lstm_379/zeros/LessLessbackward_lstm_379/zeros/mul:z:0'backward_lstm_379/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/zeros/LessЖ
 backward_lstm_379/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_379/zeros/packed/1Ћ
backward_lstm_379/zeros/packedPack(backward_lstm_379/strided_slice:output:0)backward_lstm_379/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_379/zeros/packedЗ
backward_lstm_379/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_379/zeros/Constљ
backward_lstm_379/zerosFill'backward_lstm_379/zeros/packed:output:0&backward_lstm_379/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_379/zerosД
backward_lstm_379/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_379/zeros_1/mul/yЇ
backward_lstm_379/zeros_1/mulMul(backward_lstm_379/strided_slice:output:0(backward_lstm_379/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/zeros_1/mulЗ
 backward_lstm_379/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 backward_lstm_379/zeros_1/Less/yЈ
backward_lstm_379/zeros_1/LessLess!backward_lstm_379/zeros_1/mul:z:0)backward_lstm_379/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_379/zeros_1/LessК
"backward_lstm_379/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_379/zeros_1/packed/1—
 backward_lstm_379/zeros_1/packedPack(backward_lstm_379/strided_slice:output:0+backward_lstm_379/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_379/zeros_1/packedЛ
backward_lstm_379/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_379/zeros_1/Const≈
backward_lstm_379/zeros_1Fill)backward_lstm_379/zeros_1/packed:output:0(backward_lstm_379/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_379/zeros_1Щ
 backward_lstm_379/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_379/transpose/permƒ
backward_lstm_379/transpose	Transposeinputs_0)backward_lstm_379/transpose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
backward_lstm_379/transposeЕ
backward_lstm_379/Shape_1Shapebackward_lstm_379/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_379/Shape_1Ь
'backward_lstm_379/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_379/strided_slice_1/stack†
)backward_lstm_379/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_1/stack_1†
)backward_lstm_379/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_1/stack_2Џ
!backward_lstm_379/strided_slice_1StridedSlice"backward_lstm_379/Shape_1:output:00backward_lstm_379/strided_slice_1/stack:output:02backward_lstm_379/strided_slice_1/stack_1:output:02backward_lstm_379/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_379/strided_slice_1©
-backward_lstm_379/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-backward_lstm_379/TensorArrayV2/element_shapeъ
backward_lstm_379/TensorArrayV2TensorListReserve6backward_lstm_379/TensorArrayV2/element_shape:output:0*backward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_379/TensorArrayV2О
 backward_lstm_379/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_379/ReverseV2/axisџ
backward_lstm_379/ReverseV2	ReverseV2backward_lstm_379/transpose:y:0)backward_lstm_379/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
backward_lstm_379/ReverseV2г
Gbackward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2I
Gbackward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape≈
9backward_lstm_379/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_379/ReverseV2:output:0Pbackward_lstm_379/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_379/TensorArrayUnstack/TensorListFromTensorЬ
'backward_lstm_379/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_379/strided_slice_2/stack†
)backward_lstm_379/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_2/stack_1†
)backward_lstm_379/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_2/stack_2с
!backward_lstm_379/strided_slice_2StridedSlicebackward_lstm_379/transpose:y:00backward_lstm_379/strided_slice_2/stack:output:02backward_lstm_379/strided_slice_2/stack_1:output:02backward_lstm_379/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2#
!backward_lstm_379/strided_slice_2с
6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOp?backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype028
6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOpы
'backward_lstm_379/lstm_cell_1139/MatMulMatMul*backward_lstm_379/strided_slice_2:output:0>backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'backward_lstm_379/lstm_cell_1139/MatMulч
8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOpAbackward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02:
8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOpч
)backward_lstm_379/lstm_cell_1139/MatMul_1MatMul backward_lstm_379/zeros:output:0@backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)backward_lstm_379/lstm_cell_1139/MatMul_1р
$backward_lstm_379/lstm_cell_1139/addAddV21backward_lstm_379/lstm_cell_1139/MatMul:product:03backward_lstm_379/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2&
$backward_lstm_379/lstm_cell_1139/addр
7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOp@backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype029
7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOpэ
(backward_lstm_379/lstm_cell_1139/BiasAddBiasAdd(backward_lstm_379/lstm_cell_1139/add:z:0?backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(backward_lstm_379/lstm_cell_1139/BiasAdd¶
0backward_lstm_379/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0backward_lstm_379/lstm_cell_1139/split/split_dim√
&backward_lstm_379/lstm_cell_1139/splitSplit9backward_lstm_379/lstm_cell_1139/split/split_dim:output:01backward_lstm_379/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2(
&backward_lstm_379/lstm_cell_1139/split¬
(backward_lstm_379/lstm_cell_1139/SigmoidSigmoid/backward_lstm_379/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(backward_lstm_379/lstm_cell_1139/Sigmoid∆
*backward_lstm_379/lstm_cell_1139/Sigmoid_1Sigmoid/backward_lstm_379/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_379/lstm_cell_1139/Sigmoid_1ў
$backward_lstm_379/lstm_cell_1139/mulMul.backward_lstm_379/lstm_cell_1139/Sigmoid_1:y:0"backward_lstm_379/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$backward_lstm_379/lstm_cell_1139/mulє
%backward_lstm_379/lstm_cell_1139/ReluRelu/backward_lstm_379/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22'
%backward_lstm_379/lstm_cell_1139/Reluм
&backward_lstm_379/lstm_cell_1139/mul_1Mul,backward_lstm_379/lstm_cell_1139/Sigmoid:y:03backward_lstm_379/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_379/lstm_cell_1139/mul_1б
&backward_lstm_379/lstm_cell_1139/add_1AddV2(backward_lstm_379/lstm_cell_1139/mul:z:0*backward_lstm_379/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_379/lstm_cell_1139/add_1∆
*backward_lstm_379/lstm_cell_1139/Sigmoid_2Sigmoid/backward_lstm_379/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_379/lstm_cell_1139/Sigmoid_2Є
'backward_lstm_379/lstm_cell_1139/Relu_1Relu*backward_lstm_379/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'backward_lstm_379/lstm_cell_1139/Relu_1р
&backward_lstm_379/lstm_cell_1139/mul_2Mul.backward_lstm_379/lstm_cell_1139/Sigmoid_2:y:05backward_lstm_379/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_379/lstm_cell_1139/mul_2≥
/backward_lstm_379/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   21
/backward_lstm_379/TensorArrayV2_1/element_shapeА
!backward_lstm_379/TensorArrayV2_1TensorListReserve8backward_lstm_379/TensorArrayV2_1/element_shape:output:0*backward_lstm_379/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_379/TensorArrayV2_1r
backward_lstm_379/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_379/time£
*backward_lstm_379/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2,
*backward_lstm_379/while/maximum_iterationsО
$backward_lstm_379/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_379/while/loop_counter£
backward_lstm_379/whileWhile-backward_lstm_379/while/loop_counter:output:03backward_lstm_379/while/maximum_iterations:output:0backward_lstm_379/time:output:0*backward_lstm_379/TensorArrayV2_1:handle:0 backward_lstm_379/zeros:output:0"backward_lstm_379/zeros_1:output:0*backward_lstm_379/strided_slice_1:output:0Ibackward_lstm_379/TensorArrayUnstack/TensorListFromTensor:output_handle:0?backward_lstm_379_lstm_cell_1139_matmul_readvariableop_resourceAbackward_lstm_379_lstm_cell_1139_matmul_1_readvariableop_resource@backward_lstm_379_lstm_cell_1139_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *1
body)R'
%backward_lstm_379_while_body_42661482*1
cond)R'
%backward_lstm_379_while_cond_42661481*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
backward_lstm_379/whileў
Bbackward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2D
Bbackward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shapeє
4backward_lstm_379/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_379/while:output:3Kbackward_lstm_379/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype026
4backward_lstm_379/TensorArrayV2Stack/TensorListStack•
'backward_lstm_379/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2)
'backward_lstm_379/strided_slice_3/stack†
)backward_lstm_379/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_379/strided_slice_3/stack_1†
)backward_lstm_379/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_379/strided_slice_3/stack_2Ж
!backward_lstm_379/strided_slice_3StridedSlice=backward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_379/strided_slice_3/stack:output:02backward_lstm_379/strided_slice_3/stack_1:output:02backward_lstm_379/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2#
!backward_lstm_379/strided_slice_3Э
"backward_lstm_379/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_379/transpose_1/permц
backward_lstm_379/transpose_1	Transpose=backward_lstm_379/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_379/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
backward_lstm_379/transpose_1К
backward_lstm_379/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_379/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisƒ
concatConcatV2)forward_lstm_379/strided_slice_3:output:0*backward_lstm_379/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d2

IdentityЏ
NoOpNoOp8^backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp7^backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp9^backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp^backward_lstm_379/while7^forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp6^forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp8^forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp^forward_lstm_379/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : 2r
7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp7backward_lstm_379/lstm_cell_1139/BiasAdd/ReadVariableOp2p
6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp6backward_lstm_379/lstm_cell_1139/MatMul/ReadVariableOp2t
8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp8backward_lstm_379/lstm_cell_1139/MatMul_1/ReadVariableOp22
backward_lstm_379/whilebackward_lstm_379/while2p
6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp6forward_lstm_379/lstm_cell_1138/BiasAdd/ReadVariableOp2n
5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp5forward_lstm_379/lstm_cell_1138/MatMul/ReadVariableOp2r
7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp7forward_lstm_379/lstm_cell_1138/MatMul_1/ReadVariableOp20
forward_lstm_379/whileforward_lstm_379/while:g c
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
ь

Ў
1__inference_sequential_379_layer_call_fn_42660818

inputs
inputs_1	
unknown:	»
	unknown_0:	2»
	unknown_1:	»
	unknown_2:	»
	unknown_3:	2»
	unknown_4:	»
	unknown_5:d
	unknown_6:
identityИҐStatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_sequential_379_layer_call_and_return_conditional_losses_426607772
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€:€€€€€€€€€: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
“Є
ё 
$__inference__traced_restore_42664072
file_prefix3
!assignvariableop_dense_379_kernel:d/
!assignvariableop_1_dense_379_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: ^
Kassignvariableop_7_bidirectional_379_forward_lstm_379_lstm_cell_1138_kernel:	»h
Uassignvariableop_8_bidirectional_379_forward_lstm_379_lstm_cell_1138_recurrent_kernel:	2»X
Iassignvariableop_9_bidirectional_379_forward_lstm_379_lstm_cell_1138_bias:	»`
Massignvariableop_10_bidirectional_379_backward_lstm_379_lstm_cell_1139_kernel:	»j
Wassignvariableop_11_bidirectional_379_backward_lstm_379_lstm_cell_1139_recurrent_kernel:	2»Z
Kassignvariableop_12_bidirectional_379_backward_lstm_379_lstm_cell_1139_bias:	»#
assignvariableop_13_total: #
assignvariableop_14_count: =
+assignvariableop_15_adam_dense_379_kernel_m:d7
)assignvariableop_16_adam_dense_379_bias_m:f
Sassignvariableop_17_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_kernel_m:	»p
]assignvariableop_18_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_recurrent_kernel_m:	2»`
Qassignvariableop_19_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_bias_m:	»g
Tassignvariableop_20_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_kernel_m:	»q
^assignvariableop_21_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_recurrent_kernel_m:	2»a
Rassignvariableop_22_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_bias_m:	»=
+assignvariableop_23_adam_dense_379_kernel_v:d7
)assignvariableop_24_adam_dense_379_bias_v:f
Sassignvariableop_25_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_kernel_v:	»p
]assignvariableop_26_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_recurrent_kernel_v:	2»`
Qassignvariableop_27_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_bias_v:	»g
Tassignvariableop_28_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_kernel_v:	»q
^assignvariableop_29_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_recurrent_kernel_v:	2»a
Rassignvariableop_30_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_bias_v:	»@
.assignvariableop_31_adam_dense_379_kernel_vhat:d:
,assignvariableop_32_adam_dense_379_bias_vhat:i
Vassignvariableop_33_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_kernel_vhat:	»s
`assignvariableop_34_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_recurrent_kernel_vhat:	2»c
Tassignvariableop_35_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_bias_vhat:	»j
Wassignvariableop_36_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_kernel_vhat:	»t
aassignvariableop_37_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_recurrent_kernel_vhat:	2»d
Uassignvariableop_38_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_bias_vhat:	»
identity_40ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9®
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*і
value™BІ(B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesё
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesц
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity†
AssignVariableOpAssignVariableOp!assignvariableop_dense_379_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¶
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_379_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2°
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3£
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4£
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ґ
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6™
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7–
AssignVariableOp_7AssignVariableOpKassignvariableop_7_bidirectional_379_forward_lstm_379_lstm_cell_1138_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Џ
AssignVariableOp_8AssignVariableOpUassignvariableop_8_bidirectional_379_forward_lstm_379_lstm_cell_1138_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ќ
AssignVariableOp_9AssignVariableOpIassignvariableop_9_bidirectional_379_forward_lstm_379_lstm_cell_1138_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10’
AssignVariableOp_10AssignVariableOpMassignvariableop_10_bidirectional_379_backward_lstm_379_lstm_cell_1139_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11я
AssignVariableOp_11AssignVariableOpWassignvariableop_11_bidirectional_379_backward_lstm_379_lstm_cell_1139_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12”
AssignVariableOp_12AssignVariableOpKassignvariableop_12_bidirectional_379_backward_lstm_379_lstm_cell_1139_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13°
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14°
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15≥
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_dense_379_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16±
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_dense_379_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17џ
AssignVariableOp_17AssignVariableOpSassignvariableop_17_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18е
AssignVariableOp_18AssignVariableOp]assignvariableop_18_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19ў
AssignVariableOp_19AssignVariableOpQassignvariableop_19_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20№
AssignVariableOp_20AssignVariableOpTassignvariableop_20_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ж
AssignVariableOp_21AssignVariableOp^assignvariableop_21_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Џ
AssignVariableOp_22AssignVariableOpRassignvariableop_22_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23≥
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_379_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24±
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_379_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25џ
AssignVariableOp_25AssignVariableOpSassignvariableop_25_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26е
AssignVariableOp_26AssignVariableOp]assignvariableop_26_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27ў
AssignVariableOp_27AssignVariableOpQassignvariableop_27_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28№
AssignVariableOp_28AssignVariableOpTassignvariableop_28_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29ж
AssignVariableOp_29AssignVariableOp^assignvariableop_29_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Џ
AssignVariableOp_30AssignVariableOpRassignvariableop_30_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31ґ
AssignVariableOp_31AssignVariableOp.assignvariableop_31_adam_dense_379_kernel_vhatIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32і
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_dense_379_bias_vhatIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33ё
AssignVariableOp_33AssignVariableOpVassignvariableop_33_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_kernel_vhatIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34и
AssignVariableOp_34AssignVariableOp`assignvariableop_34_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_recurrent_kernel_vhatIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35№
AssignVariableOp_35AssignVariableOpTassignvariableop_35_adam_bidirectional_379_forward_lstm_379_lstm_cell_1138_bias_vhatIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36я
AssignVariableOp_36AssignVariableOpWassignvariableop_36_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_kernel_vhatIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37й
AssignVariableOp_37AssignVariableOpaassignvariableop_37_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_recurrent_kernel_vhatIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ё
AssignVariableOp_38AssignVariableOpUassignvariableop_38_adam_bidirectional_379_backward_lstm_379_lstm_cell_1139_bias_vhatIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЄ
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_39f
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_40†
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
оX
Д
$forward_lstm_379_while_body_42661031>
:forward_lstm_379_while_forward_lstm_379_while_loop_counterD
@forward_lstm_379_while_forward_lstm_379_while_maximum_iterations&
"forward_lstm_379_while_placeholder(
$forward_lstm_379_while_placeholder_1(
$forward_lstm_379_while_placeholder_2(
$forward_lstm_379_while_placeholder_3=
9forward_lstm_379_while_forward_lstm_379_strided_slice_1_0y
uforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0Y
Fforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0:	»[
Hforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0:	2»V
Gforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0:	»#
forward_lstm_379_while_identity%
!forward_lstm_379_while_identity_1%
!forward_lstm_379_while_identity_2%
!forward_lstm_379_while_identity_3%
!forward_lstm_379_while_identity_4%
!forward_lstm_379_while_identity_5;
7forward_lstm_379_while_forward_lstm_379_strided_slice_1w
sforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensorW
Dforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource:	»Y
Fforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource:	2»T
Eforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource:	»ИҐ<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpҐ;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOpҐ=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpе
Hforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2J
Hforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shape¬
:forward_lstm_379/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_379_while_placeholderQforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02<
:forward_lstm_379/while/TensorArrayV2Read/TensorListGetItemВ
;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOpReadVariableOpFforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02=
;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp°
,forward_lstm_379/while/lstm_cell_1138/MatMulMatMulAforward_lstm_379/while/TensorArrayV2Read/TensorListGetItem:item:0Cforward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2.
,forward_lstm_379/while/lstm_cell_1138/MatMulИ
=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpReadVariableOpHforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02?
=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOpК
.forward_lstm_379/while/lstm_cell_1138/MatMul_1MatMul$forward_lstm_379_while_placeholder_2Eforward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.forward_lstm_379/while/lstm_cell_1138/MatMul_1Д
)forward_lstm_379/while/lstm_cell_1138/addAddV26forward_lstm_379/while/lstm_cell_1138/MatMul:product:08forward_lstm_379/while/lstm_cell_1138/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)forward_lstm_379/while/lstm_cell_1138/addБ
<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpReadVariableOpGforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02>
<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOpС
-forward_lstm_379/while/lstm_cell_1138/BiasAddBiasAdd-forward_lstm_379/while/lstm_cell_1138/add:z:0Dforward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-forward_lstm_379/while/lstm_cell_1138/BiasAdd∞
5forward_lstm_379/while/lstm_cell_1138/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5forward_lstm_379/while/lstm_cell_1138/split/split_dim„
+forward_lstm_379/while/lstm_cell_1138/splitSplit>forward_lstm_379/while/lstm_cell_1138/split/split_dim:output:06forward_lstm_379/while/lstm_cell_1138/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2-
+forward_lstm_379/while/lstm_cell_1138/split—
-forward_lstm_379/while/lstm_cell_1138/SigmoidSigmoid4forward_lstm_379/while/lstm_cell_1138/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-forward_lstm_379/while/lstm_cell_1138/Sigmoid’
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_1Sigmoid4forward_lstm_379/while/lstm_cell_1138/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_1к
)forward_lstm_379/while/lstm_cell_1138/mulMul3forward_lstm_379/while/lstm_cell_1138/Sigmoid_1:y:0$forward_lstm_379_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_379/while/lstm_cell_1138/mul»
*forward_lstm_379/while/lstm_cell_1138/ReluRelu4forward_lstm_379/while/lstm_cell_1138/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22,
*forward_lstm_379/while/lstm_cell_1138/ReluА
+forward_lstm_379/while/lstm_cell_1138/mul_1Mul1forward_lstm_379/while/lstm_cell_1138/Sigmoid:y:08forward_lstm_379/while/lstm_cell_1138/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_379/while/lstm_cell_1138/mul_1х
+forward_lstm_379/while/lstm_cell_1138/add_1AddV2-forward_lstm_379/while/lstm_cell_1138/mul:z:0/forward_lstm_379/while/lstm_cell_1138/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_379/while/lstm_cell_1138/add_1’
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_2Sigmoid4forward_lstm_379/while/lstm_cell_1138/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_379/while/lstm_cell_1138/Sigmoid_2«
,forward_lstm_379/while/lstm_cell_1138/Relu_1Relu/forward_lstm_379/while/lstm_cell_1138/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,forward_lstm_379/while/lstm_cell_1138/Relu_1Д
+forward_lstm_379/while/lstm_cell_1138/mul_2Mul3forward_lstm_379/while/lstm_cell_1138/Sigmoid_2:y:0:forward_lstm_379/while/lstm_cell_1138/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_379/while/lstm_cell_1138/mul_2Ј
;forward_lstm_379/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_379_while_placeholder_1"forward_lstm_379_while_placeholder/forward_lstm_379/while/lstm_cell_1138/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_379/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_379/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_379/while/add/y≠
forward_lstm_379/while/addAddV2"forward_lstm_379_while_placeholder%forward_lstm_379/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/while/addВ
forward_lstm_379/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_379/while/add_1/yЋ
forward_lstm_379/while/add_1AddV2:forward_lstm_379_while_forward_lstm_379_while_loop_counter'forward_lstm_379/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_379/while/add_1ѓ
forward_lstm_379/while/IdentityIdentity forward_lstm_379/while/add_1:z:0^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_379/while/Identity”
!forward_lstm_379/while/Identity_1Identity@forward_lstm_379_while_forward_lstm_379_while_maximum_iterations^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_379/while/Identity_1±
!forward_lstm_379/while/Identity_2Identityforward_lstm_379/while/add:z:0^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_379/while/Identity_2ё
!forward_lstm_379/while/Identity_3IdentityKforward_lstm_379/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_379/while/Identity_3”
!forward_lstm_379/while/Identity_4Identity/forward_lstm_379/while/lstm_cell_1138/mul_2:z:0^forward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_379/while/Identity_4”
!forward_lstm_379/while/Identity_5Identity/forward_lstm_379/while/lstm_cell_1138/add_1:z:0^forward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_379/while/Identity_5є
forward_lstm_379/while/NoOpNoOp=^forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp<^forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp>^forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_379/while/NoOp"t
7forward_lstm_379_while_forward_lstm_379_strided_slice_19forward_lstm_379_while_forward_lstm_379_strided_slice_1_0"K
forward_lstm_379_while_identity(forward_lstm_379/while/Identity:output:0"O
!forward_lstm_379_while_identity_1*forward_lstm_379/while/Identity_1:output:0"O
!forward_lstm_379_while_identity_2*forward_lstm_379/while/Identity_2:output:0"O
!forward_lstm_379_while_identity_3*forward_lstm_379/while/Identity_3:output:0"O
!forward_lstm_379_while_identity_4*forward_lstm_379/while/Identity_4:output:0"O
!forward_lstm_379_while_identity_5*forward_lstm_379/while/Identity_5:output:0"Р
Eforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resourceGforward_lstm_379_while_lstm_cell_1138_biasadd_readvariableop_resource_0"Т
Fforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resourceHforward_lstm_379_while_lstm_cell_1138_matmul_1_readvariableop_resource_0"О
Dforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resourceFforward_lstm_379_while_lstm_cell_1138_matmul_readvariableop_resource_0"м
sforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensoruforward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2|
<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp<forward_lstm_379/while/lstm_cell_1138/BiasAdd/ReadVariableOp2z
;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp;forward_lstm_379/while/lstm_cell_1138/MatMul/ReadVariableOp2~
=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp=forward_lstm_379/while/lstm_cell_1138/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: 
ѕ_
µ
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_42663302
inputs_0@
-lstm_cell_1139_matmul_readvariableop_resource:	»B
/lstm_cell_1139_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1139_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1139/BiasAdd/ReadVariableOpҐ$lstm_cell_1139/MatMul/ReadVariableOpҐ&lstm_cell_1139/MatMul_1/ReadVariableOpҐwhileF
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€22
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
B :и2
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
zeros_1/packed/1Й
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
:€€€€€€€€€22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЕ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
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
ReverseV2/axisК
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
	ReverseV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeэ
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
strided_slice_2/stack_2ь
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2
strided_slice_2ї
$lstm_cell_1139/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1139_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1139/MatMul/ReadVariableOp≥
lstm_cell_1139/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/MatMulЅ
&lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1139_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1139/MatMul_1/ReadVariableOpѓ
lstm_cell_1139/MatMul_1MatMulzeros:output:0.lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/MatMul_1®
lstm_cell_1139/addAddV2lstm_cell_1139/MatMul:product:0!lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/addЇ
%lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1139_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1139/BiasAdd/ReadVariableOpµ
lstm_cell_1139/BiasAddBiasAddlstm_cell_1139/add:z:0-lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1139/BiasAddВ
lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1139/split/split_dimы
lstm_cell_1139/splitSplit'lstm_cell_1139/split/split_dim:output:0lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1139/splitМ
lstm_cell_1139/SigmoidSigmoidlstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/SigmoidР
lstm_cell_1139/Sigmoid_1Sigmoidlstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Sigmoid_1С
lstm_cell_1139/mulMullstm_cell_1139/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/mulГ
lstm_cell_1139/ReluRelulstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Relu§
lstm_cell_1139/mul_1Mullstm_cell_1139/Sigmoid:y:0!lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/mul_1Щ
lstm_cell_1139/add_1AddV2lstm_cell_1139/mul:z:0lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/add_1Р
lstm_cell_1139/Sigmoid_2Sigmoidlstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Sigmoid_2В
lstm_cell_1139/Relu_1Relulstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/Relu_1®
lstm_cell_1139/mul_2Mullstm_cell_1139/Sigmoid_2:y:0#lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1139/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterХ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1139_matmul_readvariableop_resource/lstm_cell_1139_matmul_1_readvariableop_resource.lstm_cell_1139_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42663218*
condR
while_cond_42663217*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
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
:€€€€€€€€€22

Identityќ
NoOpNoOp&^lstm_cell_1139/BiasAdd/ReadVariableOp%^lstm_cell_1139/MatMul/ReadVariableOp'^lstm_cell_1139/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1139/BiasAdd/ReadVariableOp%lstm_cell_1139/BiasAdd/ReadVariableOp2L
$lstm_cell_1139/MatMul/ReadVariableOp$lstm_cell_1139/MatMul/ReadVariableOp2P
&lstm_cell_1139/MatMul_1/ReadVariableOp&lstm_cell_1139/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
ь

Ў
1__inference_sequential_379_layer_call_fn_42660325

inputs
inputs_1	
unknown:	»
	unknown_0:	2»
	unknown_1:	»
	unknown_2:	»
	unknown_3:	2»
	unknown_4:	»
	unknown_5:d
	unknown_6:
identityИҐStatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_sequential_379_layer_call_and_return_conditional_losses_426603062
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€:€€€€€€€€€: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
я
Ќ
while_cond_42659703
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_42659703___redundant_placeholder06
2while_while_cond_42659703___redundant_placeholder16
2while_while_cond_42659703___redundant_placeholder26
2while_while_cond_42659703___redundant_placeholder3
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
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:
ч
И
L__inference_lstm_cell_1138_layer_call_and_return_conditional_losses_42657913

inputs

states
states_11
matmul_readvariableop_resource:	»3
 matmul_1_readvariableop_resource:	2».
biasadd_readvariableop_resource:	»
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	»*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_2Щ
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
?:€€€€€€€€€:€€€€€€€€€2:€€€€€€€€€2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_namestates
Ѓ

•
4__inference_bidirectional_379_layer_call_fn_42660946

inputs
inputs_1	
unknown:	»
	unknown_0:	2»
	unknown_1:	»
	unknown_2:	»
	unknown_3:	2»
	unknown_4:	»
identityИҐStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_426602742
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€:€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Іg
н
%backward_lstm_379_while_body_42661829@
<backward_lstm_379_while_backward_lstm_379_while_loop_counterF
Bbackward_lstm_379_while_backward_lstm_379_while_maximum_iterations'
#backward_lstm_379_while_placeholder)
%backward_lstm_379_while_placeholder_1)
%backward_lstm_379_while_placeholder_2)
%backward_lstm_379_while_placeholder_3)
%backward_lstm_379_while_placeholder_4?
;backward_lstm_379_while_backward_lstm_379_strided_slice_1_0{
wbackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_379_while_less_backward_lstm_379_sub_1_0Z
Gbackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0:	»\
Ibackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0:	2»W
Hbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0:	»$
 backward_lstm_379_while_identity&
"backward_lstm_379_while_identity_1&
"backward_lstm_379_while_identity_2&
"backward_lstm_379_while_identity_3&
"backward_lstm_379_while_identity_4&
"backward_lstm_379_while_identity_5&
"backward_lstm_379_while_identity_6=
9backward_lstm_379_while_backward_lstm_379_strided_slice_1y
ubackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_379_while_less_backward_lstm_379_sub_1X
Ebackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource:	»Z
Gbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource:	2»U
Fbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource:	»ИҐ=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpҐ<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOpҐ>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpз
Ibackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2K
Ibackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shapeњ
;backward_lstm_379/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_379_while_placeholderRbackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02=
;backward_lstm_379/while/TensorArrayV2Read/TensorListGetItemѕ
backward_lstm_379/while/LessLess6backward_lstm_379_while_less_backward_lstm_379_sub_1_0#backward_lstm_379_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_379/while/LessЕ
<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOpReadVariableOpGbackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02>
<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp•
-backward_lstm_379/while/lstm_cell_1139/MatMulMatMulBbackward_lstm_379/while/TensorArrayV2Read/TensorListGetItem:item:0Dbackward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-backward_lstm_379/while/lstm_cell_1139/MatMulЛ
>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpReadVariableOpIbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02@
>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOpО
/backward_lstm_379/while/lstm_cell_1139/MatMul_1MatMul%backward_lstm_379_while_placeholder_3Fbackward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»21
/backward_lstm_379/while/lstm_cell_1139/MatMul_1И
*backward_lstm_379/while/lstm_cell_1139/addAddV27backward_lstm_379/while/lstm_cell_1139/MatMul:product:09backward_lstm_379/while/lstm_cell_1139/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2,
*backward_lstm_379/while/lstm_cell_1139/addД
=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpReadVariableOpHbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02?
=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOpХ
.backward_lstm_379/while/lstm_cell_1139/BiasAddBiasAdd.backward_lstm_379/while/lstm_cell_1139/add:z:0Ebackward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.backward_lstm_379/while/lstm_cell_1139/BiasAdd≤
6backward_lstm_379/while/lstm_cell_1139/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6backward_lstm_379/while/lstm_cell_1139/split/split_dimџ
,backward_lstm_379/while/lstm_cell_1139/splitSplit?backward_lstm_379/while/lstm_cell_1139/split/split_dim:output:07backward_lstm_379/while/lstm_cell_1139/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2.
,backward_lstm_379/while/lstm_cell_1139/split‘
.backward_lstm_379/while/lstm_cell_1139/SigmoidSigmoid5backward_lstm_379/while/lstm_cell_1139/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€220
.backward_lstm_379/while/lstm_cell_1139/SigmoidЎ
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_1Sigmoid5backward_lstm_379/while/lstm_cell_1139/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_1о
*backward_lstm_379/while/lstm_cell_1139/mulMul4backward_lstm_379/while/lstm_cell_1139/Sigmoid_1:y:0%backward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_379/while/lstm_cell_1139/mulЋ
+backward_lstm_379/while/lstm_cell_1139/ReluRelu5backward_lstm_379/while/lstm_cell_1139/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22-
+backward_lstm_379/while/lstm_cell_1139/ReluД
,backward_lstm_379/while/lstm_cell_1139/mul_1Mul2backward_lstm_379/while/lstm_cell_1139/Sigmoid:y:09backward_lstm_379/while/lstm_cell_1139/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_379/while/lstm_cell_1139/mul_1щ
,backward_lstm_379/while/lstm_cell_1139/add_1AddV2.backward_lstm_379/while/lstm_cell_1139/mul:z:00backward_lstm_379/while/lstm_cell_1139/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_379/while/lstm_cell_1139/add_1Ў
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_2Sigmoid5backward_lstm_379/while/lstm_cell_1139/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_379/while/lstm_cell_1139/Sigmoid_2 
-backward_lstm_379/while/lstm_cell_1139/Relu_1Relu0backward_lstm_379/while/lstm_cell_1139/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-backward_lstm_379/while/lstm_cell_1139/Relu_1И
,backward_lstm_379/while/lstm_cell_1139/mul_2Mul4backward_lstm_379/while/lstm_cell_1139/Sigmoid_2:y:0;backward_lstm_379/while/lstm_cell_1139/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_379/while/lstm_cell_1139/mul_2ч
backward_lstm_379/while/SelectSelect backward_lstm_379/while/Less:z:00backward_lstm_379/while/lstm_cell_1139/mul_2:z:0%backward_lstm_379_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22 
backward_lstm_379/while/Selectы
 backward_lstm_379/while/Select_1Select backward_lstm_379/while/Less:z:00backward_lstm_379/while/lstm_cell_1139/mul_2:z:0%backward_lstm_379_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_379/while/Select_1ы
 backward_lstm_379/while/Select_2Select backward_lstm_379/while/Less:z:00backward_lstm_379/while/lstm_cell_1139/add_1:z:0%backward_lstm_379_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_379/while/Select_2≥
<backward_lstm_379/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_379_while_placeholder_1#backward_lstm_379_while_placeholder'backward_lstm_379/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_379/while/TensorArrayV2Write/TensorListSetItemА
backward_lstm_379/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_379/while/add/y±
backward_lstm_379/while/addAddV2#backward_lstm_379_while_placeholder&backward_lstm_379/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/while/addД
backward_lstm_379/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_379/while/add_1/y–
backward_lstm_379/while/add_1AddV2<backward_lstm_379_while_backward_lstm_379_while_loop_counter(backward_lstm_379/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_379/while/add_1≥
 backward_lstm_379/while/IdentityIdentity!backward_lstm_379/while/add_1:z:0^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_379/while/IdentityЎ
"backward_lstm_379/while/Identity_1IdentityBbackward_lstm_379_while_backward_lstm_379_while_maximum_iterations^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_379/while/Identity_1µ
"backward_lstm_379/while/Identity_2Identitybackward_lstm_379/while/add:z:0^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_379/while/Identity_2в
"backward_lstm_379/while/Identity_3IdentityLbackward_lstm_379/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_379/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_379/while/Identity_3ќ
"backward_lstm_379/while/Identity_4Identity'backward_lstm_379/while/Select:output:0^backward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_379/while/Identity_4–
"backward_lstm_379/while/Identity_5Identity)backward_lstm_379/while/Select_1:output:0^backward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_379/while/Identity_5–
"backward_lstm_379/while/Identity_6Identity)backward_lstm_379/while/Select_2:output:0^backward_lstm_379/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_379/while/Identity_6Њ
backward_lstm_379/while/NoOpNoOp>^backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp=^backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp?^backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_379/while/NoOp"x
9backward_lstm_379_while_backward_lstm_379_strided_slice_1;backward_lstm_379_while_backward_lstm_379_strided_slice_1_0"M
 backward_lstm_379_while_identity)backward_lstm_379/while/Identity:output:0"Q
"backward_lstm_379_while_identity_1+backward_lstm_379/while/Identity_1:output:0"Q
"backward_lstm_379_while_identity_2+backward_lstm_379/while/Identity_2:output:0"Q
"backward_lstm_379_while_identity_3+backward_lstm_379/while/Identity_3:output:0"Q
"backward_lstm_379_while_identity_4+backward_lstm_379/while/Identity_4:output:0"Q
"backward_lstm_379_while_identity_5+backward_lstm_379/while/Identity_5:output:0"Q
"backward_lstm_379_while_identity_6+backward_lstm_379/while/Identity_6:output:0"n
4backward_lstm_379_while_less_backward_lstm_379_sub_16backward_lstm_379_while_less_backward_lstm_379_sub_1_0"Т
Fbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resourceHbackward_lstm_379_while_lstm_cell_1139_biasadd_readvariableop_resource_0"Ф
Gbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resourceIbackward_lstm_379_while_lstm_cell_1139_matmul_1_readvariableop_resource_0"Р
Ebackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resourceGbackward_lstm_379_while_lstm_cell_1139_matmul_readvariableop_resource_0"р
ubackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_379_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_379_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2~
=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp=backward_lstm_379/while/lstm_cell_1139/BiasAdd/ReadVariableOp2|
<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp<backward_lstm_379/while/lstm_cell_1139/MatMul/ReadVariableOp2А
>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp>backward_lstm_379/while/lstm_cell_1139/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:€€€€€€€€€
‘
¬
3__inference_forward_lstm_379_layer_call_fn_42662315
inputs_0
unknown:	»
	unknown_0:	2»
	unknown_1:	»
identityИҐStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_426579962
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
Њ
ъ
1__inference_lstm_cell_1139_layer_call_fn_42663723

inputs
states_0
states_1
unknown:	»
	unknown_0:	2»
	unknown_1:	»
identity

identity_1

identity_2ИҐStatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_lstm_cell_1139_layer_call_and_return_conditional_losses_426585452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

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
?:€€€€€€€€€:€€€€€€€€€2:€€€€€€€€€2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€2
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€2
"
_user_specified_name
states/1
я
Ќ
while_cond_42659530
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_42659530___redundant_placeholder06
2while_while_cond_42659530___redundant_placeholder16
2while_while_cond_42659530___redundant_placeholder26
2while_while_cond_42659530___redundant_placeholder3
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
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:
я
Ќ
while_cond_42658136
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_42658136___redundant_placeholder06
2while_while_cond_42658136___redundant_placeholder16
2while_while_cond_42658136___redundant_placeholder26
2while_while_cond_42658136___redundant_placeholder3
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
@: : : : :€€€€€€€€€2:€€€€€€€€€2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€2:-)
'
_output_shapes
:€€€€€€€€€2:

_output_shapes
: :

_output_shapes
:"®L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*е
serving_default—
9
args_0/
serving_default_args_0:0€€€€€€€€€
9
args_0_1-
serving_default_args_0_1:0	€€€€€€€€€=
	dense_3790
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:сї
і
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
ћ
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
ї

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
√
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
 
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
≈
%cell
&
state_spec
'regularization_losses
(	variables
)trainable_variables
*	keras_api
А__call__
+Б&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
≈
+cell
,
state_spec
-regularization_losses
.	variables
/trainable_variables
0	keras_api
В__call__
+Г&call_and_return_all_conditional_losses"
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
≠
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
": d2dense_379/kernel
:2dense_379/bias
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
≠
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
K:I	»28bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel
U:S	2»2Bbidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel
E:C»26bidirectional_379/forward_lstm_379/lstm_cell_1138/bias
L:J	»29bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel
V:T	2»2Cbidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel
F:D»27bidirectional_379/backward_lstm_379/lstm_cell_1139/bias
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
г
<
state_size

kernel
recurrent_kernel
bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses"
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
Љ
Alayer_metrics
Bnon_trainable_variables

Cstates
'regularization_losses
(	variables
Dmetrics
)trainable_variables
Elayer_regularization_losses

Flayers
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
г
G
state_size

kernel
recurrent_kernel
bias
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses"
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
Љ
Llayer_metrics
Mnon_trainable_variables

Nstates
-regularization_losses
.	variables
Ometrics
/trainable_variables
Player_regularization_losses

Qlayers
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
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
∞
Vlayer_metrics
Wnon_trainable_variables
=regularization_losses
>	variables
Xmetrics
?trainable_variables
Ylayer_regularization_losses

Zlayers
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
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
∞
[layer_metrics
\non_trainable_variables
Hregularization_losses
I	variables
]metrics
Jtrainable_variables
^layer_regularization_losses

_layers
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
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
':%d2Adam/dense_379/kernel/m
!:2Adam/dense_379/bias/m
P:N	»2?Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/m
Z:X	2»2IAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/m
J:H»2=Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/m
Q:O	»2@Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/m
[:Y	2»2JAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/m
K:I»2>Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/m
':%d2Adam/dense_379/kernel/v
!:2Adam/dense_379/bias/v
P:N	»2?Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/v
Z:X	2»2IAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/v
J:H»2=Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/v
Q:O	»2@Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/v
[:Y	2»2JAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/v
K:I»2>Adam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/v
*:(d2Adam/dense_379/kernel/vhat
$:"2Adam/dense_379/bias/vhat
S:Q	»2BAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/kernel/vhat
]:[	2»2LAdam/bidirectional_379/forward_lstm_379/lstm_cell_1138/recurrent_kernel/vhat
M:K»2@Adam/bidirectional_379/forward_lstm_379/lstm_cell_1138/bias/vhat
T:R	»2CAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/kernel/vhat
^:\	2»2MAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/recurrent_kernel/vhat
N:L»2AAdam/bidirectional_379/backward_lstm_379/lstm_cell_1139/bias/vhat
ђ2©
1__inference_sequential_379_layer_call_fn_42660325
1__inference_sequential_379_layer_call_fn_42660818ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
„B‘
#__inference__wrapped_model_42657838args_0args_0_1"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
в2я
L__inference_sequential_379_layer_call_and_return_conditional_losses_42660841
L__inference_sequential_379_layer_call_and_return_conditional_losses_42660864ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ƒ2Ѕ
4__inference_bidirectional_379_layer_call_fn_42660911
4__inference_bidirectional_379_layer_call_fn_42660928
4__inference_bidirectional_379_layer_call_fn_42660946
4__inference_bidirectional_379_layer_call_fn_42660964ж
Ё≤ў
FullArgSpecO
argsGЪD
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
defaultsЪ
p 

 

 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
∞2≠
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_42661266
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_42661568
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_42661926
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_42662284ж
Ё≤ў
FullArgSpecO
argsGЪD
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
defaultsЪ
p 

 

 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
,__inference_dense_379_layer_call_fn_42662293Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_dense_379_layer_call_and_return_conditional_losses_42662304Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
‘B—
&__inference_signature_wrapper_42660894args_0args_0_1"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѓ2ђ
3__inference_forward_lstm_379_layer_call_fn_42662315
3__inference_forward_lstm_379_layer_call_fn_42662326
3__inference_forward_lstm_379_layer_call_fn_42662337
3__inference_forward_lstm_379_layer_call_fn_42662348’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ы2Ш
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_42662499
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_42662650
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_42662801
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_42662952’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
≥2∞
4__inference_backward_lstm_379_layer_call_fn_42662963
4__inference_backward_lstm_379_layer_call_fn_42662974
4__inference_backward_lstm_379_layer_call_fn_42662985
4__inference_backward_lstm_379_layer_call_fn_42662996’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Я2Ь
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_42663149
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_42663302
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_42663455
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_42663608’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
™2І
1__inference_lstm_cell_1138_layer_call_fn_42663625
1__inference_lstm_cell_1138_layer_call_fn_42663642Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
а2Ё
L__inference_lstm_cell_1138_layer_call_and_return_conditional_losses_42663674
L__inference_lstm_cell_1138_layer_call_and_return_conditional_losses_42663706Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
™2І
1__inference_lstm_cell_1139_layer_call_fn_42663723
1__inference_lstm_cell_1139_layer_call_fn_42663740Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
а2Ё
L__inference_lstm_cell_1139_layer_call_and_return_conditional_losses_42663772
L__inference_lstm_cell_1139_layer_call_and_return_conditional_losses_42663804Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 «
#__inference__wrapped_model_42657838Я\ҐY
RҐO
MТJ4Ґ1
!ъ€€€€€€€€€€€€€€€€€€
А
`
А	RaggedTensorSpec
™ "5™2
0
	dense_379#К 
	dense_379€€€€€€€€€–
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_42663149}OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€2
Ъ –
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_42663302}OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "%Ґ"
К
0€€€€€€€€€2
Ъ “
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_42663455QҐN
GҐD
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€2
Ъ “
O__inference_backward_lstm_379_layer_call_and_return_conditional_losses_42663608QҐN
GҐD
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
p

 
™ "%Ґ"
К
0€€€€€€€€€2
Ъ ®
4__inference_backward_lstm_379_layer_call_fn_42662963pOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€2®
4__inference_backward_lstm_379_layer_call_fn_42662974pOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "К€€€€€€€€€2™
4__inference_backward_lstm_379_layer_call_fn_42662985rQҐN
GҐD
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€2™
4__inference_backward_lstm_379_layer_call_fn_42662996rQҐN
GҐD
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
p

 
™ "К€€€€€€€€€2б
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_42661266Н\ҐY
RҐO
=Ъ:
8К5
inputs/0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 

 

 

 
™ "%Ґ"
К
0€€€€€€€€€d
Ъ б
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_42661568Н\ҐY
RҐO
=Ъ:
8К5
inputs/0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p

 

 

 
™ "%Ґ"
К
0€€€€€€€€€d
Ъ с
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_42661926ЭlҐi
bҐ_
MТJ4Ґ1
!ъ€€€€€€€€€€€€€€€€€€
А
`
А	RaggedTensorSpec
p 

 

 

 
™ "%Ґ"
К
0€€€€€€€€€d
Ъ с
O__inference_bidirectional_379_layer_call_and_return_conditional_losses_42662284ЭlҐi
bҐ_
MТJ4Ґ1
!ъ€€€€€€€€€€€€€€€€€€
А
`
А	RaggedTensorSpec
p

 

 

 
™ "%Ґ"
К
0€€€€€€€€€d
Ъ є
4__inference_bidirectional_379_layer_call_fn_42660911А\ҐY
RҐO
=Ъ:
8К5
inputs/0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 

 

 

 
™ "К€€€€€€€€€dє
4__inference_bidirectional_379_layer_call_fn_42660928А\ҐY
RҐO
=Ъ:
8К5
inputs/0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p

 

 

 
™ "К€€€€€€€€€d…
4__inference_bidirectional_379_layer_call_fn_42660946РlҐi
bҐ_
MТJ4Ґ1
!ъ€€€€€€€€€€€€€€€€€€
А
`
А	RaggedTensorSpec
p 

 

 

 
™ "К€€€€€€€€€d…
4__inference_bidirectional_379_layer_call_fn_42660964РlҐi
bҐ_
MТJ4Ґ1
!ъ€€€€€€€€€€€€€€€€€€
А
`
А	RaggedTensorSpec
p

 

 

 
™ "К€€€€€€€€€dІ
G__inference_dense_379_layer_call_and_return_conditional_losses_42662304\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "%Ґ"
К
0€€€€€€€€€
Ъ 
,__inference_dense_379_layer_call_fn_42662293O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "К€€€€€€€€€ѕ
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_42662499}OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€2
Ъ ѕ
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_42662650}OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "%Ґ"
К
0€€€€€€€€€2
Ъ —
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_42662801QҐN
GҐD
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€2
Ъ —
N__inference_forward_lstm_379_layer_call_and_return_conditional_losses_42662952QҐN
GҐD
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
p

 
™ "%Ґ"
К
0€€€€€€€€€2
Ъ І
3__inference_forward_lstm_379_layer_call_fn_42662315pOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€2І
3__inference_forward_lstm_379_layer_call_fn_42662326pOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "К€€€€€€€€€2©
3__inference_forward_lstm_379_layer_call_fn_42662337rQҐN
GҐD
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€2©
3__inference_forward_lstm_379_layer_call_fn_42662348rQҐN
GҐD
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
p

 
™ "К€€€€€€€€€2ќ
L__inference_lstm_cell_1138_layer_call_and_return_conditional_losses_42663674эАҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states/0€€€€€€€€€2
"К
states/1€€€€€€€€€2
p 
™ "sҐp
iҐf
К
0/0€€€€€€€€€2
EЪB
К
0/1/0€€€€€€€€€2
К
0/1/1€€€€€€€€€2
Ъ ќ
L__inference_lstm_cell_1138_layer_call_and_return_conditional_losses_42663706эАҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states/0€€€€€€€€€2
"К
states/1€€€€€€€€€2
p
™ "sҐp
iҐf
К
0/0€€€€€€€€€2
EЪB
К
0/1/0€€€€€€€€€2
К
0/1/1€€€€€€€€€2
Ъ £
1__inference_lstm_cell_1138_layer_call_fn_42663625нАҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states/0€€€€€€€€€2
"К
states/1€€€€€€€€€2
p 
™ "cҐ`
К
0€€€€€€€€€2
AЪ>
К
1/0€€€€€€€€€2
К
1/1€€€€€€€€€2£
1__inference_lstm_cell_1138_layer_call_fn_42663642нАҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states/0€€€€€€€€€2
"К
states/1€€€€€€€€€2
p
™ "cҐ`
К
0€€€€€€€€€2
AЪ>
К
1/0€€€€€€€€€2
К
1/1€€€€€€€€€2ќ
L__inference_lstm_cell_1139_layer_call_and_return_conditional_losses_42663772эАҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states/0€€€€€€€€€2
"К
states/1€€€€€€€€€2
p 
™ "sҐp
iҐf
К
0/0€€€€€€€€€2
EЪB
К
0/1/0€€€€€€€€€2
К
0/1/1€€€€€€€€€2
Ъ ќ
L__inference_lstm_cell_1139_layer_call_and_return_conditional_losses_42663804эАҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states/0€€€€€€€€€2
"К
states/1€€€€€€€€€2
p
™ "sҐp
iҐf
К
0/0€€€€€€€€€2
EЪB
К
0/1/0€€€€€€€€€2
К
0/1/1€€€€€€€€€2
Ъ £
1__inference_lstm_cell_1139_layer_call_fn_42663723нАҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states/0€€€€€€€€€2
"К
states/1€€€€€€€€€2
p 
™ "cҐ`
К
0€€€€€€€€€2
AЪ>
К
1/0€€€€€€€€€2
К
1/1€€€€€€€€€2£
1__inference_lstm_cell_1139_layer_call_fn_42663740нАҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states/0€€€€€€€€€2
"К
states/1€€€€€€€€€2
p
™ "cҐ`
К
0€€€€€€€€€2
AЪ>
К
1/0€€€€€€€€€2
К
1/1€€€€€€€€€2и
L__inference_sequential_379_layer_call_and_return_conditional_losses_42660841ЧdҐa
ZҐW
MТJ4Ґ1
!ъ€€€€€€€€€€€€€€€€€€
А
`
А	RaggedTensorSpec
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ и
L__inference_sequential_379_layer_call_and_return_conditional_losses_42660864ЧdҐa
ZҐW
MТJ4Ґ1
!ъ€€€€€€€€€€€€€€€€€€
А
`
А	RaggedTensorSpec
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ј
1__inference_sequential_379_layer_call_fn_42660325КdҐa
ZҐW
MТJ4Ґ1
!ъ€€€€€€€€€€€€€€€€€€
А
`
А	RaggedTensorSpec
p 

 
™ "К€€€€€€€€€ј
1__inference_sequential_379_layer_call_fn_42660818КdҐa
ZҐW
MТJ4Ґ1
!ъ€€€€€€€€€€€€€€€€€€
А
`
А	RaggedTensorSpec
p

 
™ "К€€€€€€€€€”
&__inference_signature_wrapper_42660894®eҐb
Ґ 
[™X
*
args_0 К
args_0€€€€€€€€€
*
args_0_1К
args_0_1€€€€€€€€€	"5™2
0
	dense_379#К 
	dense_379€€€€€€€€€