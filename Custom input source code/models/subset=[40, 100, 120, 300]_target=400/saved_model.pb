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
dense_602/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*!
shared_namedense_602/kernel
u
$dense_602/kernel/Read/ReadVariableOpReadVariableOpdense_602/kernel*
_output_shapes

:d*
dtype0
t
dense_602/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_602/bias
m
"dense_602/bias/Read/ReadVariableOpReadVariableOpdense_602/bias*
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
8bidirectional_602/forward_lstm_602/lstm_cell_1807/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*I
shared_name:8bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel
∆
Lbidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/Read/ReadVariableOpReadVariableOp8bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel*
_output_shapes
:	»*
dtype0
б
Bbidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*S
shared_nameDBbidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel
Џ
Vbidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/Read/ReadVariableOpReadVariableOpBbidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel*
_output_shapes
:	2»*
dtype0
≈
6bidirectional_602/forward_lstm_602/lstm_cell_1807/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*G
shared_name86bidirectional_602/forward_lstm_602/lstm_cell_1807/bias
Њ
Jbidirectional_602/forward_lstm_602/lstm_cell_1807/bias/Read/ReadVariableOpReadVariableOp6bidirectional_602/forward_lstm_602/lstm_cell_1807/bias*
_output_shapes	
:»*
dtype0
ѕ
9bidirectional_602/backward_lstm_602/lstm_cell_1808/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*J
shared_name;9bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel
»
Mbidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/Read/ReadVariableOpReadVariableOp9bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel*
_output_shapes
:	»*
dtype0
г
Cbidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*T
shared_nameECbidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel
№
Wbidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/Read/ReadVariableOpReadVariableOpCbidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel*
_output_shapes
:	2»*
dtype0
«
7bidirectional_602/backward_lstm_602/lstm_cell_1808/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*H
shared_name97bidirectional_602/backward_lstm_602/lstm_cell_1808/bias
ј
Kbidirectional_602/backward_lstm_602/lstm_cell_1808/bias/Read/ReadVariableOpReadVariableOp7bidirectional_602/backward_lstm_602/lstm_cell_1808/bias*
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
Adam/dense_602/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_602/kernel/m
Г
+Adam/dense_602/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_602/kernel/m*
_output_shapes

:d*
dtype0
В
Adam/dense_602/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_602/bias/m
{
)Adam/dense_602/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_602/bias/m*
_output_shapes
:*
dtype0
џ
?Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*P
shared_nameA?Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/m
‘
SAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/m/Read/ReadVariableOpReadVariableOp?Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/m*
_output_shapes
:	»*
dtype0
п
IAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*Z
shared_nameKIAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/m
и
]Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpIAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/m*
_output_shapes
:	2»*
dtype0
”
=Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*N
shared_name?=Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/m
ћ
QAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/m/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/m*
_output_shapes	
:»*
dtype0
Ё
@Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*Q
shared_nameB@Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/m
÷
TAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/m/Read/ReadVariableOpReadVariableOp@Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/m*
_output_shapes
:	»*
dtype0
с
JAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*[
shared_nameLJAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/m
к
^Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpJAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/m*
_output_shapes
:	2»*
dtype0
’
>Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*O
shared_name@>Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/m
ќ
RAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/m/Read/ReadVariableOpReadVariableOp>Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/m*
_output_shapes	
:»*
dtype0
К
Adam/dense_602/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_602/kernel/v
Г
+Adam/dense_602/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_602/kernel/v*
_output_shapes

:d*
dtype0
В
Adam/dense_602/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_602/bias/v
{
)Adam/dense_602/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_602/bias/v*
_output_shapes
:*
dtype0
џ
?Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*P
shared_nameA?Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/v
‘
SAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/v/Read/ReadVariableOpReadVariableOp?Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/v*
_output_shapes
:	»*
dtype0
п
IAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*Z
shared_nameKIAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/v
и
]Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpIAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/v*
_output_shapes
:	2»*
dtype0
”
=Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*N
shared_name?=Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/v
ћ
QAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/v/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/v*
_output_shapes	
:»*
dtype0
Ё
@Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*Q
shared_nameB@Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/v
÷
TAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/v/Read/ReadVariableOpReadVariableOp@Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/v*
_output_shapes
:	»*
dtype0
с
JAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*[
shared_nameLJAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/v
к
^Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpJAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/v*
_output_shapes
:	2»*
dtype0
’
>Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*O
shared_name@>Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/v
ќ
RAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/v/Read/ReadVariableOpReadVariableOp>Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/v*
_output_shapes	
:»*
dtype0
Р
Adam/dense_602/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*+
shared_nameAdam/dense_602/kernel/vhat
Й
.Adam/dense_602/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_602/kernel/vhat*
_output_shapes

:d*
dtype0
И
Adam/dense_602/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/dense_602/bias/vhat
Б
,Adam/dense_602/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_602/bias/vhat*
_output_shapes
:*
dtype0
б
BAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*S
shared_nameDBAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/vhat
Џ
VAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/vhat/Read/ReadVariableOpReadVariableOpBAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/vhat*
_output_shapes
:	»*
dtype0
х
LAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*]
shared_nameNLAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/vhat
о
`Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpLAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/vhat*
_output_shapes
:	2»*
dtype0
ў
@Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*Q
shared_nameB@Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/vhat
“
TAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/vhat/Read/ReadVariableOpReadVariableOp@Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/vhat*
_output_shapes	
:»*
dtype0
г
CAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*T
shared_nameECAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/vhat
№
WAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/vhat/Read/ReadVariableOpReadVariableOpCAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/vhat*
_output_shapes
:	»*
dtype0
ч
MAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*^
shared_nameOMAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/vhat
р
aAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpMAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/vhat*
_output_shapes
:	2»*
dtype0
џ
AAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*R
shared_nameCAAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/vhat
‘
UAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/vhat/Read/ReadVariableOpReadVariableOpAAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/vhat*
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
VARIABLE_VALUEdense_602/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_602/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUE8bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEBbidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6bidirectional_602/forward_lstm_602/lstm_cell_1807/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE9bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUECbidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE7bidirectional_602/backward_lstm_602/lstm_cell_1808/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_602/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_602/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE?Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ҐЯ
VARIABLE_VALUEIAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE@Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
£†
VARIABLE_VALUEJAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE>Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_602/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_602/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE?Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ҐЯ
VARIABLE_VALUEIAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE@Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
£†
VARIABLE_VALUEJAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE>Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEAdam/dense_602/kernel/vhatUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/dense_602/bias/vhatSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUEBAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/vhatEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
®•
VARIABLE_VALUELAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/vhatEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ЬЩ
VARIABLE_VALUE@Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/vhatEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUECAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/vhatEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
©¶
VARIABLE_VALUEMAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/vhatEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ЭЪ
VARIABLE_VALUEAAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/vhatEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_0_18bidirectional_602/forward_lstm_602/lstm_cell_1807/kernelBbidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel6bidirectional_602/forward_lstm_602/lstm_cell_1807/bias9bidirectional_602/backward_lstm_602/lstm_cell_1808/kernelCbidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel7bidirectional_602/backward_lstm_602/lstm_cell_1808/biasdense_602/kerneldense_602/bias*
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
&__inference_signature_wrapper_60936684
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
І
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_602/kernel/Read/ReadVariableOp"dense_602/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpLbidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/Read/ReadVariableOpVbidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/Read/ReadVariableOpJbidirectional_602/forward_lstm_602/lstm_cell_1807/bias/Read/ReadVariableOpMbidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/Read/ReadVariableOpWbidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/Read/ReadVariableOpKbidirectional_602/backward_lstm_602/lstm_cell_1808/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_602/kernel/m/Read/ReadVariableOp)Adam/dense_602/bias/m/Read/ReadVariableOpSAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/m/Read/ReadVariableOp]Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/m/Read/ReadVariableOpQAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/m/Read/ReadVariableOpTAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/m/Read/ReadVariableOp^Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/m/Read/ReadVariableOpRAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/m/Read/ReadVariableOp+Adam/dense_602/kernel/v/Read/ReadVariableOp)Adam/dense_602/bias/v/Read/ReadVariableOpSAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/v/Read/ReadVariableOp]Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/v/Read/ReadVariableOpQAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/v/Read/ReadVariableOpTAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/v/Read/ReadVariableOp^Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/v/Read/ReadVariableOpRAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/v/Read/ReadVariableOp.Adam/dense_602/kernel/vhat/Read/ReadVariableOp,Adam/dense_602/bias/vhat/Read/ReadVariableOpVAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/vhat/Read/ReadVariableOp`Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/vhat/Read/ReadVariableOpTAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/vhat/Read/ReadVariableOpWAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/vhat/Read/ReadVariableOpaAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/vhat/Read/ReadVariableOpUAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/vhat/Read/ReadVariableOpConst*4
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
!__inference__traced_save_60939735
Ц
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_602/kerneldense_602/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate8bidirectional_602/forward_lstm_602/lstm_cell_1807/kernelBbidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel6bidirectional_602/forward_lstm_602/lstm_cell_1807/bias9bidirectional_602/backward_lstm_602/lstm_cell_1808/kernelCbidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel7bidirectional_602/backward_lstm_602/lstm_cell_1808/biastotalcountAdam/dense_602/kernel/mAdam/dense_602/bias/m?Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/mIAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/m=Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/m@Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/mJAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/m>Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/mAdam/dense_602/kernel/vAdam/dense_602/bias/v?Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/vIAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/v=Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/v@Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/vJAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/v>Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/vAdam/dense_602/kernel/vhatAdam/dense_602/bias/vhatBAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/vhatLAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/vhat@Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/vhatCAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/vhatMAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/vhatAAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/vhat*3
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
$__inference__traced_restore_60939862иЪ9
я
Ќ
while_cond_60938657
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60938657___redundant_placeholder06
2while_while_cond_60938657___redundant_placeholder16
2while_while_cond_60938657___redundant_placeholder26
2while_while_cond_60938657___redundant_placeholder3
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
э
µ
%backward_lstm_602_while_cond_60936969@
<backward_lstm_602_while_backward_lstm_602_while_loop_counterF
Bbackward_lstm_602_while_backward_lstm_602_while_maximum_iterations'
#backward_lstm_602_while_placeholder)
%backward_lstm_602_while_placeholder_1)
%backward_lstm_602_while_placeholder_2)
%backward_lstm_602_while_placeholder_3B
>backward_lstm_602_while_less_backward_lstm_602_strided_slice_1Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60936969___redundant_placeholder0Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60936969___redundant_placeholder1Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60936969___redundant_placeholder2Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60936969___redundant_placeholder3$
 backward_lstm_602_while_identity
 
backward_lstm_602/while/LessLess#backward_lstm_602_while_placeholder>backward_lstm_602_while_less_backward_lstm_602_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_602/while/LessУ
 backward_lstm_602/while/IdentityIdentity backward_lstm_602/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_602/while/Identity"M
 backward_lstm_602_while_identity)backward_lstm_602/while/Identity:output:0*(
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
ь

Ў
1__inference_sequential_602_layer_call_fn_60936608

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
L__inference_sequential_602_layer_call_and_return_conditional_losses_609365672
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
€
К
L__inference_lstm_cell_1807_layer_call_and_return_conditional_losses_60939464

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
Іg
н
%backward_lstm_602_while_body_60937977@
<backward_lstm_602_while_backward_lstm_602_while_loop_counterF
Bbackward_lstm_602_while_backward_lstm_602_while_maximum_iterations'
#backward_lstm_602_while_placeholder)
%backward_lstm_602_while_placeholder_1)
%backward_lstm_602_while_placeholder_2)
%backward_lstm_602_while_placeholder_3)
%backward_lstm_602_while_placeholder_4?
;backward_lstm_602_while_backward_lstm_602_strided_slice_1_0{
wbackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_602_while_less_backward_lstm_602_sub_1_0Z
Gbackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0:	»\
Ibackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0:	2»W
Hbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0:	»$
 backward_lstm_602_while_identity&
"backward_lstm_602_while_identity_1&
"backward_lstm_602_while_identity_2&
"backward_lstm_602_while_identity_3&
"backward_lstm_602_while_identity_4&
"backward_lstm_602_while_identity_5&
"backward_lstm_602_while_identity_6=
9backward_lstm_602_while_backward_lstm_602_strided_slice_1y
ubackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_602_while_less_backward_lstm_602_sub_1X
Ebackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource:	»Z
Gbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource:	2»U
Fbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource:	»ИҐ=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpҐ<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOpҐ>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpз
Ibackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2K
Ibackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeњ
;backward_lstm_602/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_602_while_placeholderRbackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02=
;backward_lstm_602/while/TensorArrayV2Read/TensorListGetItemѕ
backward_lstm_602/while/LessLess6backward_lstm_602_while_less_backward_lstm_602_sub_1_0#backward_lstm_602_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_602/while/LessЕ
<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOpGbackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02>
<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp•
-backward_lstm_602/while/lstm_cell_1808/MatMulMatMulBbackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem:item:0Dbackward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-backward_lstm_602/while/lstm_cell_1808/MatMulЛ
>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOpIbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02@
>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpО
/backward_lstm_602/while/lstm_cell_1808/MatMul_1MatMul%backward_lstm_602_while_placeholder_3Fbackward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»21
/backward_lstm_602/while/lstm_cell_1808/MatMul_1И
*backward_lstm_602/while/lstm_cell_1808/addAddV27backward_lstm_602/while/lstm_cell_1808/MatMul:product:09backward_lstm_602/while/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2,
*backward_lstm_602/while/lstm_cell_1808/addД
=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOpHbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02?
=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpХ
.backward_lstm_602/while/lstm_cell_1808/BiasAddBiasAdd.backward_lstm_602/while/lstm_cell_1808/add:z:0Ebackward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.backward_lstm_602/while/lstm_cell_1808/BiasAdd≤
6backward_lstm_602/while/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6backward_lstm_602/while/lstm_cell_1808/split/split_dimџ
,backward_lstm_602/while/lstm_cell_1808/splitSplit?backward_lstm_602/while/lstm_cell_1808/split/split_dim:output:07backward_lstm_602/while/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2.
,backward_lstm_602/while/lstm_cell_1808/split‘
.backward_lstm_602/while/lstm_cell_1808/SigmoidSigmoid5backward_lstm_602/while/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€220
.backward_lstm_602/while/lstm_cell_1808/SigmoidЎ
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_1Sigmoid5backward_lstm_602/while/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_1о
*backward_lstm_602/while/lstm_cell_1808/mulMul4backward_lstm_602/while/lstm_cell_1808/Sigmoid_1:y:0%backward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_602/while/lstm_cell_1808/mulЋ
+backward_lstm_602/while/lstm_cell_1808/ReluRelu5backward_lstm_602/while/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22-
+backward_lstm_602/while/lstm_cell_1808/ReluД
,backward_lstm_602/while/lstm_cell_1808/mul_1Mul2backward_lstm_602/while/lstm_cell_1808/Sigmoid:y:09backward_lstm_602/while/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_602/while/lstm_cell_1808/mul_1щ
,backward_lstm_602/while/lstm_cell_1808/add_1AddV2.backward_lstm_602/while/lstm_cell_1808/mul:z:00backward_lstm_602/while/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_602/while/lstm_cell_1808/add_1Ў
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_2Sigmoid5backward_lstm_602/while/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_2 
-backward_lstm_602/while/lstm_cell_1808/Relu_1Relu0backward_lstm_602/while/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-backward_lstm_602/while/lstm_cell_1808/Relu_1И
,backward_lstm_602/while/lstm_cell_1808/mul_2Mul4backward_lstm_602/while/lstm_cell_1808/Sigmoid_2:y:0;backward_lstm_602/while/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_602/while/lstm_cell_1808/mul_2ч
backward_lstm_602/while/SelectSelect backward_lstm_602/while/Less:z:00backward_lstm_602/while/lstm_cell_1808/mul_2:z:0%backward_lstm_602_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22 
backward_lstm_602/while/Selectы
 backward_lstm_602/while/Select_1Select backward_lstm_602/while/Less:z:00backward_lstm_602/while/lstm_cell_1808/mul_2:z:0%backward_lstm_602_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_602/while/Select_1ы
 backward_lstm_602/while/Select_2Select backward_lstm_602/while/Less:z:00backward_lstm_602/while/lstm_cell_1808/add_1:z:0%backward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_602/while/Select_2≥
<backward_lstm_602/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_602_while_placeholder_1#backward_lstm_602_while_placeholder'backward_lstm_602/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_602/while/TensorArrayV2Write/TensorListSetItemА
backward_lstm_602/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_602/while/add/y±
backward_lstm_602/while/addAddV2#backward_lstm_602_while_placeholder&backward_lstm_602/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/while/addД
backward_lstm_602/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_602/while/add_1/y–
backward_lstm_602/while/add_1AddV2<backward_lstm_602_while_backward_lstm_602_while_loop_counter(backward_lstm_602/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/while/add_1≥
 backward_lstm_602/while/IdentityIdentity!backward_lstm_602/while/add_1:z:0^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_602/while/IdentityЎ
"backward_lstm_602/while/Identity_1IdentityBbackward_lstm_602_while_backward_lstm_602_while_maximum_iterations^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_602/while/Identity_1µ
"backward_lstm_602/while/Identity_2Identitybackward_lstm_602/while/add:z:0^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_602/while/Identity_2в
"backward_lstm_602/while/Identity_3IdentityLbackward_lstm_602/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_602/while/Identity_3ќ
"backward_lstm_602/while/Identity_4Identity'backward_lstm_602/while/Select:output:0^backward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_602/while/Identity_4–
"backward_lstm_602/while/Identity_5Identity)backward_lstm_602/while/Select_1:output:0^backward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_602/while/Identity_5–
"backward_lstm_602/while/Identity_6Identity)backward_lstm_602/while/Select_2:output:0^backward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_602/while/Identity_6Њ
backward_lstm_602/while/NoOpNoOp>^backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp=^backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp?^backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_602/while/NoOp"x
9backward_lstm_602_while_backward_lstm_602_strided_slice_1;backward_lstm_602_while_backward_lstm_602_strided_slice_1_0"M
 backward_lstm_602_while_identity)backward_lstm_602/while/Identity:output:0"Q
"backward_lstm_602_while_identity_1+backward_lstm_602/while/Identity_1:output:0"Q
"backward_lstm_602_while_identity_2+backward_lstm_602/while/Identity_2:output:0"Q
"backward_lstm_602_while_identity_3+backward_lstm_602/while/Identity_3:output:0"Q
"backward_lstm_602_while_identity_4+backward_lstm_602/while/Identity_4:output:0"Q
"backward_lstm_602_while_identity_5+backward_lstm_602/while/Identity_5:output:0"Q
"backward_lstm_602_while_identity_6+backward_lstm_602/while/Identity_6:output:0"n
4backward_lstm_602_while_less_backward_lstm_602_sub_16backward_lstm_602_while_less_backward_lstm_602_sub_1_0"Т
Fbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resourceHbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0"Ф
Gbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resourceIbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0"Р
Ebackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resourceGbackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0"р
ubackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2~
=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp2|
<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp2А
>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Є
£
L__inference_sequential_602_layer_call_and_return_conditional_losses_60936567

inputs
inputs_1	-
bidirectional_602_60936548:	»-
bidirectional_602_60936550:	2»)
bidirectional_602_60936552:	»-
bidirectional_602_60936554:	»-
bidirectional_602_60936556:	2»)
bidirectional_602_60936558:	»$
dense_602_60936561:d 
dense_602_60936563:
identityИҐ)bidirectional_602/StatefulPartitionedCallҐ!dense_602/StatefulPartitionedCall 
)bidirectional_602/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_602_60936548bidirectional_602_60936550bidirectional_602_60936552bidirectional_602_60936554bidirectional_602_60936556bidirectional_602_60936558*
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
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_609365042+
)bidirectional_602/StatefulPartitionedCallЋ
!dense_602/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_602/StatefulPartitionedCall:output:0dense_602_60936561dense_602_60936563*
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
G__inference_dense_602_layer_call_and_return_conditional_losses_609360892#
!dense_602/StatefulPartitionedCallЕ
IdentityIdentity*dense_602/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityЮ
NoOpNoOp*^bidirectional_602/StatefulPartitionedCall"^dense_602/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€:€€€€€€€€€: : : : : : : : 2V
)bidirectional_602/StatefulPartitionedCall)bidirectional_602/StatefulPartitionedCall2F
!dense_602/StatefulPartitionedCall!dense_602/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
¶Z
§
%backward_lstm_602_while_body_60936970@
<backward_lstm_602_while_backward_lstm_602_while_loop_counterF
Bbackward_lstm_602_while_backward_lstm_602_while_maximum_iterations'
#backward_lstm_602_while_placeholder)
%backward_lstm_602_while_placeholder_1)
%backward_lstm_602_while_placeholder_2)
%backward_lstm_602_while_placeholder_3?
;backward_lstm_602_while_backward_lstm_602_strided_slice_1_0{
wbackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0Z
Gbackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0:	»\
Ibackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0:	2»W
Hbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0:	»$
 backward_lstm_602_while_identity&
"backward_lstm_602_while_identity_1&
"backward_lstm_602_while_identity_2&
"backward_lstm_602_while_identity_3&
"backward_lstm_602_while_identity_4&
"backward_lstm_602_while_identity_5=
9backward_lstm_602_while_backward_lstm_602_strided_slice_1y
ubackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensorX
Ebackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource:	»Z
Gbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource:	2»U
Fbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource:	»ИҐ=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpҐ<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOpҐ>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpз
Ibackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2K
Ibackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shape»
;backward_lstm_602/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_602_while_placeholderRbackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02=
;backward_lstm_602/while/TensorArrayV2Read/TensorListGetItemЕ
<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOpGbackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02>
<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp•
-backward_lstm_602/while/lstm_cell_1808/MatMulMatMulBbackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem:item:0Dbackward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-backward_lstm_602/while/lstm_cell_1808/MatMulЛ
>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOpIbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02@
>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpО
/backward_lstm_602/while/lstm_cell_1808/MatMul_1MatMul%backward_lstm_602_while_placeholder_2Fbackward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»21
/backward_lstm_602/while/lstm_cell_1808/MatMul_1И
*backward_lstm_602/while/lstm_cell_1808/addAddV27backward_lstm_602/while/lstm_cell_1808/MatMul:product:09backward_lstm_602/while/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2,
*backward_lstm_602/while/lstm_cell_1808/addД
=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOpHbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02?
=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpХ
.backward_lstm_602/while/lstm_cell_1808/BiasAddBiasAdd.backward_lstm_602/while/lstm_cell_1808/add:z:0Ebackward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.backward_lstm_602/while/lstm_cell_1808/BiasAdd≤
6backward_lstm_602/while/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6backward_lstm_602/while/lstm_cell_1808/split/split_dimџ
,backward_lstm_602/while/lstm_cell_1808/splitSplit?backward_lstm_602/while/lstm_cell_1808/split/split_dim:output:07backward_lstm_602/while/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2.
,backward_lstm_602/while/lstm_cell_1808/split‘
.backward_lstm_602/while/lstm_cell_1808/SigmoidSigmoid5backward_lstm_602/while/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€220
.backward_lstm_602/while/lstm_cell_1808/SigmoidЎ
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_1Sigmoid5backward_lstm_602/while/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_1о
*backward_lstm_602/while/lstm_cell_1808/mulMul4backward_lstm_602/while/lstm_cell_1808/Sigmoid_1:y:0%backward_lstm_602_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_602/while/lstm_cell_1808/mulЋ
+backward_lstm_602/while/lstm_cell_1808/ReluRelu5backward_lstm_602/while/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22-
+backward_lstm_602/while/lstm_cell_1808/ReluД
,backward_lstm_602/while/lstm_cell_1808/mul_1Mul2backward_lstm_602/while/lstm_cell_1808/Sigmoid:y:09backward_lstm_602/while/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_602/while/lstm_cell_1808/mul_1щ
,backward_lstm_602/while/lstm_cell_1808/add_1AddV2.backward_lstm_602/while/lstm_cell_1808/mul:z:00backward_lstm_602/while/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_602/while/lstm_cell_1808/add_1Ў
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_2Sigmoid5backward_lstm_602/while/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_2 
-backward_lstm_602/while/lstm_cell_1808/Relu_1Relu0backward_lstm_602/while/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-backward_lstm_602/while/lstm_cell_1808/Relu_1И
,backward_lstm_602/while/lstm_cell_1808/mul_2Mul4backward_lstm_602/while/lstm_cell_1808/Sigmoid_2:y:0;backward_lstm_602/while/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_602/while/lstm_cell_1808/mul_2Љ
<backward_lstm_602/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_602_while_placeholder_1#backward_lstm_602_while_placeholder0backward_lstm_602/while/lstm_cell_1808/mul_2:z:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_602/while/TensorArrayV2Write/TensorListSetItemА
backward_lstm_602/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_602/while/add/y±
backward_lstm_602/while/addAddV2#backward_lstm_602_while_placeholder&backward_lstm_602/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/while/addД
backward_lstm_602/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_602/while/add_1/y–
backward_lstm_602/while/add_1AddV2<backward_lstm_602_while_backward_lstm_602_while_loop_counter(backward_lstm_602/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/while/add_1≥
 backward_lstm_602/while/IdentityIdentity!backward_lstm_602/while/add_1:z:0^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_602/while/IdentityЎ
"backward_lstm_602/while/Identity_1IdentityBbackward_lstm_602_while_backward_lstm_602_while_maximum_iterations^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_602/while/Identity_1µ
"backward_lstm_602/while/Identity_2Identitybackward_lstm_602/while/add:z:0^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_602/while/Identity_2в
"backward_lstm_602/while/Identity_3IdentityLbackward_lstm_602/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_602/while/Identity_3„
"backward_lstm_602/while/Identity_4Identity0backward_lstm_602/while/lstm_cell_1808/mul_2:z:0^backward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_602/while/Identity_4„
"backward_lstm_602/while/Identity_5Identity0backward_lstm_602/while/lstm_cell_1808/add_1:z:0^backward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_602/while/Identity_5Њ
backward_lstm_602/while/NoOpNoOp>^backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp=^backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp?^backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_602/while/NoOp"x
9backward_lstm_602_while_backward_lstm_602_strided_slice_1;backward_lstm_602_while_backward_lstm_602_strided_slice_1_0"M
 backward_lstm_602_while_identity)backward_lstm_602/while/Identity:output:0"Q
"backward_lstm_602_while_identity_1+backward_lstm_602/while/Identity_1:output:0"Q
"backward_lstm_602_while_identity_2+backward_lstm_602/while/Identity_2:output:0"Q
"backward_lstm_602_while_identity_3+backward_lstm_602/while/Identity_3:output:0"Q
"backward_lstm_602_while_identity_4+backward_lstm_602/while/Identity_4:output:0"Q
"backward_lstm_602_while_identity_5+backward_lstm_602/while/Identity_5:output:0"Т
Fbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resourceHbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0"Ф
Gbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resourceIbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0"Р
Ebackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resourceGbackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0"р
ubackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2~
=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp2|
<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp2А
>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
%backward_lstm_602_while_body_60937619@
<backward_lstm_602_while_backward_lstm_602_while_loop_counterF
Bbackward_lstm_602_while_backward_lstm_602_while_maximum_iterations'
#backward_lstm_602_while_placeholder)
%backward_lstm_602_while_placeholder_1)
%backward_lstm_602_while_placeholder_2)
%backward_lstm_602_while_placeholder_3)
%backward_lstm_602_while_placeholder_4?
;backward_lstm_602_while_backward_lstm_602_strided_slice_1_0{
wbackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_602_while_less_backward_lstm_602_sub_1_0Z
Gbackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0:	»\
Ibackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0:	2»W
Hbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0:	»$
 backward_lstm_602_while_identity&
"backward_lstm_602_while_identity_1&
"backward_lstm_602_while_identity_2&
"backward_lstm_602_while_identity_3&
"backward_lstm_602_while_identity_4&
"backward_lstm_602_while_identity_5&
"backward_lstm_602_while_identity_6=
9backward_lstm_602_while_backward_lstm_602_strided_slice_1y
ubackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_602_while_less_backward_lstm_602_sub_1X
Ebackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource:	»Z
Gbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource:	2»U
Fbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource:	»ИҐ=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpҐ<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOpҐ>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpз
Ibackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2K
Ibackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeњ
;backward_lstm_602/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_602_while_placeholderRbackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02=
;backward_lstm_602/while/TensorArrayV2Read/TensorListGetItemѕ
backward_lstm_602/while/LessLess6backward_lstm_602_while_less_backward_lstm_602_sub_1_0#backward_lstm_602_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_602/while/LessЕ
<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOpGbackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02>
<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp•
-backward_lstm_602/while/lstm_cell_1808/MatMulMatMulBbackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem:item:0Dbackward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-backward_lstm_602/while/lstm_cell_1808/MatMulЛ
>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOpIbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02@
>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpО
/backward_lstm_602/while/lstm_cell_1808/MatMul_1MatMul%backward_lstm_602_while_placeholder_3Fbackward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»21
/backward_lstm_602/while/lstm_cell_1808/MatMul_1И
*backward_lstm_602/while/lstm_cell_1808/addAddV27backward_lstm_602/while/lstm_cell_1808/MatMul:product:09backward_lstm_602/while/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2,
*backward_lstm_602/while/lstm_cell_1808/addД
=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOpHbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02?
=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpХ
.backward_lstm_602/while/lstm_cell_1808/BiasAddBiasAdd.backward_lstm_602/while/lstm_cell_1808/add:z:0Ebackward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.backward_lstm_602/while/lstm_cell_1808/BiasAdd≤
6backward_lstm_602/while/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6backward_lstm_602/while/lstm_cell_1808/split/split_dimџ
,backward_lstm_602/while/lstm_cell_1808/splitSplit?backward_lstm_602/while/lstm_cell_1808/split/split_dim:output:07backward_lstm_602/while/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2.
,backward_lstm_602/while/lstm_cell_1808/split‘
.backward_lstm_602/while/lstm_cell_1808/SigmoidSigmoid5backward_lstm_602/while/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€220
.backward_lstm_602/while/lstm_cell_1808/SigmoidЎ
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_1Sigmoid5backward_lstm_602/while/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_1о
*backward_lstm_602/while/lstm_cell_1808/mulMul4backward_lstm_602/while/lstm_cell_1808/Sigmoid_1:y:0%backward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_602/while/lstm_cell_1808/mulЋ
+backward_lstm_602/while/lstm_cell_1808/ReluRelu5backward_lstm_602/while/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22-
+backward_lstm_602/while/lstm_cell_1808/ReluД
,backward_lstm_602/while/lstm_cell_1808/mul_1Mul2backward_lstm_602/while/lstm_cell_1808/Sigmoid:y:09backward_lstm_602/while/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_602/while/lstm_cell_1808/mul_1щ
,backward_lstm_602/while/lstm_cell_1808/add_1AddV2.backward_lstm_602/while/lstm_cell_1808/mul:z:00backward_lstm_602/while/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_602/while/lstm_cell_1808/add_1Ў
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_2Sigmoid5backward_lstm_602/while/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_2 
-backward_lstm_602/while/lstm_cell_1808/Relu_1Relu0backward_lstm_602/while/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-backward_lstm_602/while/lstm_cell_1808/Relu_1И
,backward_lstm_602/while/lstm_cell_1808/mul_2Mul4backward_lstm_602/while/lstm_cell_1808/Sigmoid_2:y:0;backward_lstm_602/while/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_602/while/lstm_cell_1808/mul_2ч
backward_lstm_602/while/SelectSelect backward_lstm_602/while/Less:z:00backward_lstm_602/while/lstm_cell_1808/mul_2:z:0%backward_lstm_602_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22 
backward_lstm_602/while/Selectы
 backward_lstm_602/while/Select_1Select backward_lstm_602/while/Less:z:00backward_lstm_602/while/lstm_cell_1808/mul_2:z:0%backward_lstm_602_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_602/while/Select_1ы
 backward_lstm_602/while/Select_2Select backward_lstm_602/while/Less:z:00backward_lstm_602/while/lstm_cell_1808/add_1:z:0%backward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_602/while/Select_2≥
<backward_lstm_602/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_602_while_placeholder_1#backward_lstm_602_while_placeholder'backward_lstm_602/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_602/while/TensorArrayV2Write/TensorListSetItemА
backward_lstm_602/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_602/while/add/y±
backward_lstm_602/while/addAddV2#backward_lstm_602_while_placeholder&backward_lstm_602/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/while/addД
backward_lstm_602/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_602/while/add_1/y–
backward_lstm_602/while/add_1AddV2<backward_lstm_602_while_backward_lstm_602_while_loop_counter(backward_lstm_602/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/while/add_1≥
 backward_lstm_602/while/IdentityIdentity!backward_lstm_602/while/add_1:z:0^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_602/while/IdentityЎ
"backward_lstm_602/while/Identity_1IdentityBbackward_lstm_602_while_backward_lstm_602_while_maximum_iterations^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_602/while/Identity_1µ
"backward_lstm_602/while/Identity_2Identitybackward_lstm_602/while/add:z:0^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_602/while/Identity_2в
"backward_lstm_602/while/Identity_3IdentityLbackward_lstm_602/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_602/while/Identity_3ќ
"backward_lstm_602/while/Identity_4Identity'backward_lstm_602/while/Select:output:0^backward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_602/while/Identity_4–
"backward_lstm_602/while/Identity_5Identity)backward_lstm_602/while/Select_1:output:0^backward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_602/while/Identity_5–
"backward_lstm_602/while/Identity_6Identity)backward_lstm_602/while/Select_2:output:0^backward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_602/while/Identity_6Њ
backward_lstm_602/while/NoOpNoOp>^backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp=^backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp?^backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_602/while/NoOp"x
9backward_lstm_602_while_backward_lstm_602_strided_slice_1;backward_lstm_602_while_backward_lstm_602_strided_slice_1_0"M
 backward_lstm_602_while_identity)backward_lstm_602/while/Identity:output:0"Q
"backward_lstm_602_while_identity_1+backward_lstm_602/while/Identity_1:output:0"Q
"backward_lstm_602_while_identity_2+backward_lstm_602/while/Identity_2:output:0"Q
"backward_lstm_602_while_identity_3+backward_lstm_602/while/Identity_3:output:0"Q
"backward_lstm_602_while_identity_4+backward_lstm_602/while/Identity_4:output:0"Q
"backward_lstm_602_while_identity_5+backward_lstm_602/while/Identity_5:output:0"Q
"backward_lstm_602_while_identity_6+backward_lstm_602/while/Identity_6:output:0"n
4backward_lstm_602_while_less_backward_lstm_602_sub_16backward_lstm_602_while_less_backward_lstm_602_sub_1_0"Т
Fbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resourceHbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0"Ф
Gbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resourceIbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0"Р
Ebackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resourceGbackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0"р
ubackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2~
=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp2|
<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp2А
>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_cond_60933716
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60933716___redundant_placeholder06
2while_while_cond_60933716___redundant_placeholder16
2while_while_cond_60933716___redundant_placeholder26
2while_while_cond_60933716___redundant_placeholder3
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
Њ
ъ
1__inference_lstm_cell_1808_layer_call_fn_60939513

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
L__inference_lstm_cell_1808_layer_call_and_return_conditional_losses_609343352
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
while_body_60934969
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1807_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1807_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1807_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1807_matmul_readvariableop_resource:	»H
5while_lstm_cell_1807_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1807_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1807/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1807/MatMul/ReadVariableOpҐ,while/lstm_cell_1807/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1807_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1807/MatMul/ReadVariableOpЁ
while/lstm_cell_1807/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/MatMul’
,while/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1807_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1807/MatMul_1/ReadVariableOp∆
while/lstm_cell_1807/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/MatMul_1ј
while/lstm_cell_1807/addAddV2%while/lstm_cell_1807/MatMul:product:0'while/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/addќ
+while/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1807_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1807/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1807/BiasAddBiasAddwhile/lstm_cell_1807/add:z:03while/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/BiasAddО
$while/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1807/split/split_dimУ
while/lstm_cell_1807/splitSplit-while/lstm_cell_1807/split/split_dim:output:0%while/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1807/splitЮ
while/lstm_cell_1807/SigmoidSigmoid#while/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/SigmoidҐ
while/lstm_cell_1807/Sigmoid_1Sigmoid#while/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1807/Sigmoid_1¶
while/lstm_cell_1807/mulMul"while/lstm_cell_1807/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/mulХ
while/lstm_cell_1807/ReluRelu#while/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/ReluЉ
while/lstm_cell_1807/mul_1Mul while/lstm_cell_1807/Sigmoid:y:0'while/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/mul_1±
while/lstm_cell_1807/add_1AddV2while/lstm_cell_1807/mul:z:0while/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/add_1Ґ
while/lstm_cell_1807/Sigmoid_2Sigmoid#while/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1807/Sigmoid_2Ф
while/lstm_cell_1807/Relu_1Reluwhile/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/Relu_1ј
while/lstm_cell_1807/mul_2Mul"while/lstm_cell_1807/Sigmoid_2:y:0)while/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1807/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1807/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1807/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1807/BiasAdd/ReadVariableOp+^while/lstm_cell_1807/MatMul/ReadVariableOp-^while/lstm_cell_1807/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1807_biasadd_readvariableop_resource6while_lstm_cell_1807_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1807_matmul_1_readvariableop_resource7while_lstm_cell_1807_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1807_matmul_readvariableop_resource5while_lstm_cell_1807_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1807/BiasAdd/ReadVariableOp+while/lstm_cell_1807/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1807/MatMul/ReadVariableOp*while/lstm_cell_1807/MatMul/ReadVariableOp2\
,while/lstm_cell_1807/MatMul_1/ReadVariableOp,while/lstm_cell_1807/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
а
ј
3__inference_forward_lstm_602_layer_call_fn_60938127

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
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_609350532
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
ЩЮ
Є
Esequential_602_bidirectional_602_forward_lstm_602_while_body_60933345А
|sequential_602_bidirectional_602_forward_lstm_602_while_sequential_602_bidirectional_602_forward_lstm_602_while_loop_counterЗ
Вsequential_602_bidirectional_602_forward_lstm_602_while_sequential_602_bidirectional_602_forward_lstm_602_while_maximum_iterationsG
Csequential_602_bidirectional_602_forward_lstm_602_while_placeholderI
Esequential_602_bidirectional_602_forward_lstm_602_while_placeholder_1I
Esequential_602_bidirectional_602_forward_lstm_602_while_placeholder_2I
Esequential_602_bidirectional_602_forward_lstm_602_while_placeholder_3I
Esequential_602_bidirectional_602_forward_lstm_602_while_placeholder_4
{sequential_602_bidirectional_602_forward_lstm_602_while_sequential_602_bidirectional_602_forward_lstm_602_strided_slice_1_0Љ
Јsequential_602_bidirectional_602_forward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_sequential_602_bidirectional_602_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0|
xsequential_602_bidirectional_602_forward_lstm_602_while_greater_sequential_602_bidirectional_602_forward_lstm_602_cast_0z
gsequential_602_bidirectional_602_forward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0:	»|
isequential_602_bidirectional_602_forward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0:	2»w
hsequential_602_bidirectional_602_forward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0:	»D
@sequential_602_bidirectional_602_forward_lstm_602_while_identityF
Bsequential_602_bidirectional_602_forward_lstm_602_while_identity_1F
Bsequential_602_bidirectional_602_forward_lstm_602_while_identity_2F
Bsequential_602_bidirectional_602_forward_lstm_602_while_identity_3F
Bsequential_602_bidirectional_602_forward_lstm_602_while_identity_4F
Bsequential_602_bidirectional_602_forward_lstm_602_while_identity_5F
Bsequential_602_bidirectional_602_forward_lstm_602_while_identity_6}
ysequential_602_bidirectional_602_forward_lstm_602_while_sequential_602_bidirectional_602_forward_lstm_602_strided_slice_1Ї
µsequential_602_bidirectional_602_forward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_sequential_602_bidirectional_602_forward_lstm_602_tensorarrayunstack_tensorlistfromtensorz
vsequential_602_bidirectional_602_forward_lstm_602_while_greater_sequential_602_bidirectional_602_forward_lstm_602_castx
esequential_602_bidirectional_602_forward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource:	»z
gsequential_602_bidirectional_602_forward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource:	2»u
fsequential_602_bidirectional_602_forward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource:	»ИҐ]sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpҐ\sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOpҐ^sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpІ
isequential_602/bidirectional_602/forward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2k
isequential_602/bidirectional_602/forward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeА
[sequential_602/bidirectional_602/forward_lstm_602/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЈsequential_602_bidirectional_602_forward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_sequential_602_bidirectional_602_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0Csequential_602_bidirectional_602_forward_lstm_602_while_placeholderrsequential_602/bidirectional_602/forward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02]
[sequential_602/bidirectional_602/forward_lstm_602/while/TensorArrayV2Read/TensorListGetItemъ
?sequential_602/bidirectional_602/forward_lstm_602/while/GreaterGreaterxsequential_602_bidirectional_602_forward_lstm_602_while_greater_sequential_602_bidirectional_602_forward_lstm_602_cast_0Csequential_602_bidirectional_602_forward_lstm_602_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2A
?sequential_602/bidirectional_602/forward_lstm_602/while/Greaterе
\sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOpgsequential_602_bidirectional_602_forward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02^
\sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp•
Msequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMulMatMulbsequential_602/bidirectional_602/forward_lstm_602/while/TensorArrayV2Read/TensorListGetItem:item:0dsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2O
Msequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMulл
^sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOpisequential_602_bidirectional_602_forward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02`
^sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpО
Osequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMul_1MatMulEsequential_602_bidirectional_602_forward_lstm_602_while_placeholder_3fsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2Q
Osequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMul_1И
Jsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/addAddV2Wsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMul:product:0Ysequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2L
Jsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/addд
]sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOphsequential_602_bidirectional_602_forward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02_
]sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpХ
Nsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/BiasAddBiasAddNsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/add:z:0esequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2P
Nsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/BiasAddт
Vsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2X
Vsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/split/split_dimџ
Lsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/splitSplit_sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/split/split_dim:output:0Wsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2N
Lsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/splitі
Nsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/SigmoidSigmoidUsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22P
Nsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/SigmoidЄ
Psequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/Sigmoid_1SigmoidUsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22R
Psequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/Sigmoid_1о
Jsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/mulMulTsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/Sigmoid_1:y:0Esequential_602_bidirectional_602_forward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22L
Jsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/mulЂ
Ksequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/ReluReluUsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22M
Ksequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/ReluД
Lsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/mul_1MulRsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/Sigmoid:y:0Ysequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22N
Lsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/mul_1щ
Lsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/add_1AddV2Nsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/mul:z:0Psequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22N
Lsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/add_1Є
Psequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/Sigmoid_2SigmoidUsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22R
Psequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/Sigmoid_2™
Msequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/Relu_1ReluPsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22O
Msequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/Relu_1И
Lsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/mul_2MulTsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/Sigmoid_2:y:0[sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22N
Lsequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/mul_2Ъ
>sequential_602/bidirectional_602/forward_lstm_602/while/SelectSelectCsequential_602/bidirectional_602/forward_lstm_602/while/Greater:z:0Psequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/mul_2:z:0Esequential_602_bidirectional_602_forward_lstm_602_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22@
>sequential_602/bidirectional_602/forward_lstm_602/while/SelectЮ
@sequential_602/bidirectional_602/forward_lstm_602/while/Select_1SelectCsequential_602/bidirectional_602/forward_lstm_602/while/Greater:z:0Psequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/mul_2:z:0Esequential_602_bidirectional_602_forward_lstm_602_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22B
@sequential_602/bidirectional_602/forward_lstm_602/while/Select_1Ю
@sequential_602/bidirectional_602/forward_lstm_602/while/Select_2SelectCsequential_602/bidirectional_602/forward_lstm_602/while/Greater:z:0Psequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/add_1:z:0Esequential_602_bidirectional_602_forward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22B
@sequential_602/bidirectional_602/forward_lstm_602/while/Select_2”
\sequential_602/bidirectional_602/forward_lstm_602/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemEsequential_602_bidirectional_602_forward_lstm_602_while_placeholder_1Csequential_602_bidirectional_602_forward_lstm_602_while_placeholderGsequential_602/bidirectional_602/forward_lstm_602/while/Select:output:0*
_output_shapes
: *
element_dtype02^
\sequential_602/bidirectional_602/forward_lstm_602/while/TensorArrayV2Write/TensorListSetItemј
=sequential_602/bidirectional_602/forward_lstm_602/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2?
=sequential_602/bidirectional_602/forward_lstm_602/while/add/y±
;sequential_602/bidirectional_602/forward_lstm_602/while/addAddV2Csequential_602_bidirectional_602_forward_lstm_602_while_placeholderFsequential_602/bidirectional_602/forward_lstm_602/while/add/y:output:0*
T0*
_output_shapes
: 2=
;sequential_602/bidirectional_602/forward_lstm_602/while/addƒ
?sequential_602/bidirectional_602/forward_lstm_602/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2A
?sequential_602/bidirectional_602/forward_lstm_602/while/add_1/yр
=sequential_602/bidirectional_602/forward_lstm_602/while/add_1AddV2|sequential_602_bidirectional_602_forward_lstm_602_while_sequential_602_bidirectional_602_forward_lstm_602_while_loop_counterHsequential_602/bidirectional_602/forward_lstm_602/while/add_1/y:output:0*
T0*
_output_shapes
: 2?
=sequential_602/bidirectional_602/forward_lstm_602/while/add_1≥
@sequential_602/bidirectional_602/forward_lstm_602/while/IdentityIdentityAsequential_602/bidirectional_602/forward_lstm_602/while/add_1:z:0=^sequential_602/bidirectional_602/forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2B
@sequential_602/bidirectional_602/forward_lstm_602/while/Identityщ
Bsequential_602/bidirectional_602/forward_lstm_602/while/Identity_1IdentityВsequential_602_bidirectional_602_forward_lstm_602_while_sequential_602_bidirectional_602_forward_lstm_602_while_maximum_iterations=^sequential_602/bidirectional_602/forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2D
Bsequential_602/bidirectional_602/forward_lstm_602/while/Identity_1µ
Bsequential_602/bidirectional_602/forward_lstm_602/while/Identity_2Identity?sequential_602/bidirectional_602/forward_lstm_602/while/add:z:0=^sequential_602/bidirectional_602/forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2D
Bsequential_602/bidirectional_602/forward_lstm_602/while/Identity_2в
Bsequential_602/bidirectional_602/forward_lstm_602/while/Identity_3Identitylsequential_602/bidirectional_602/forward_lstm_602/while/TensorArrayV2Write/TensorListSetItem:output_handle:0=^sequential_602/bidirectional_602/forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2D
Bsequential_602/bidirectional_602/forward_lstm_602/while/Identity_3ќ
Bsequential_602/bidirectional_602/forward_lstm_602/while/Identity_4IdentityGsequential_602/bidirectional_602/forward_lstm_602/while/Select:output:0=^sequential_602/bidirectional_602/forward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22D
Bsequential_602/bidirectional_602/forward_lstm_602/while/Identity_4–
Bsequential_602/bidirectional_602/forward_lstm_602/while/Identity_5IdentityIsequential_602/bidirectional_602/forward_lstm_602/while/Select_1:output:0=^sequential_602/bidirectional_602/forward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22D
Bsequential_602/bidirectional_602/forward_lstm_602/while/Identity_5–
Bsequential_602/bidirectional_602/forward_lstm_602/while/Identity_6IdentityIsequential_602/bidirectional_602/forward_lstm_602/while/Select_2:output:0=^sequential_602/bidirectional_602/forward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22D
Bsequential_602/bidirectional_602/forward_lstm_602/while/Identity_6ё
<sequential_602/bidirectional_602/forward_lstm_602/while/NoOpNoOp^^sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp]^sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp_^sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2>
<sequential_602/bidirectional_602/forward_lstm_602/while/NoOp"т
vsequential_602_bidirectional_602_forward_lstm_602_while_greater_sequential_602_bidirectional_602_forward_lstm_602_castxsequential_602_bidirectional_602_forward_lstm_602_while_greater_sequential_602_bidirectional_602_forward_lstm_602_cast_0"Н
@sequential_602_bidirectional_602_forward_lstm_602_while_identityIsequential_602/bidirectional_602/forward_lstm_602/while/Identity:output:0"С
Bsequential_602_bidirectional_602_forward_lstm_602_while_identity_1Ksequential_602/bidirectional_602/forward_lstm_602/while/Identity_1:output:0"С
Bsequential_602_bidirectional_602_forward_lstm_602_while_identity_2Ksequential_602/bidirectional_602/forward_lstm_602/while/Identity_2:output:0"С
Bsequential_602_bidirectional_602_forward_lstm_602_while_identity_3Ksequential_602/bidirectional_602/forward_lstm_602/while/Identity_3:output:0"С
Bsequential_602_bidirectional_602_forward_lstm_602_while_identity_4Ksequential_602/bidirectional_602/forward_lstm_602/while/Identity_4:output:0"С
Bsequential_602_bidirectional_602_forward_lstm_602_while_identity_5Ksequential_602/bidirectional_602/forward_lstm_602/while/Identity_5:output:0"С
Bsequential_602_bidirectional_602_forward_lstm_602_while_identity_6Ksequential_602/bidirectional_602/forward_lstm_602/while/Identity_6:output:0"“
fsequential_602_bidirectional_602_forward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resourcehsequential_602_bidirectional_602_forward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0"‘
gsequential_602_bidirectional_602_forward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resourceisequential_602_bidirectional_602_forward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0"–
esequential_602_bidirectional_602_forward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resourcegsequential_602_bidirectional_602_forward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0"ш
ysequential_602_bidirectional_602_forward_lstm_602_while_sequential_602_bidirectional_602_forward_lstm_602_strided_slice_1{sequential_602_bidirectional_602_forward_lstm_602_while_sequential_602_bidirectional_602_forward_lstm_602_strided_slice_1_0"т
µsequential_602_bidirectional_602_forward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_sequential_602_bidirectional_602_forward_lstm_602_tensorarrayunstack_tensorlistfromtensorЈsequential_602_bidirectional_602_forward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_sequential_602_bidirectional_602_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2Њ
]sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp]sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp2Љ
\sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp\sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp2ј
^sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp^sequential_602/bidirectional_602/forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
ЎЯ
џ
Fsequential_602_bidirectional_602_backward_lstm_602_while_body_60933524В
~sequential_602_bidirectional_602_backward_lstm_602_while_sequential_602_bidirectional_602_backward_lstm_602_while_loop_counterЙ
Дsequential_602_bidirectional_602_backward_lstm_602_while_sequential_602_bidirectional_602_backward_lstm_602_while_maximum_iterationsH
Dsequential_602_bidirectional_602_backward_lstm_602_while_placeholderJ
Fsequential_602_bidirectional_602_backward_lstm_602_while_placeholder_1J
Fsequential_602_bidirectional_602_backward_lstm_602_while_placeholder_2J
Fsequential_602_bidirectional_602_backward_lstm_602_while_placeholder_3J
Fsequential_602_bidirectional_602_backward_lstm_602_while_placeholder_4Б
}sequential_602_bidirectional_602_backward_lstm_602_while_sequential_602_bidirectional_602_backward_lstm_602_strided_slice_1_0Њ
єsequential_602_bidirectional_602_backward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_sequential_602_bidirectional_602_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0|
xsequential_602_bidirectional_602_backward_lstm_602_while_less_sequential_602_bidirectional_602_backward_lstm_602_sub_1_0{
hsequential_602_bidirectional_602_backward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0:	»}
jsequential_602_bidirectional_602_backward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0:	2»x
isequential_602_bidirectional_602_backward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0:	»E
Asequential_602_bidirectional_602_backward_lstm_602_while_identityG
Csequential_602_bidirectional_602_backward_lstm_602_while_identity_1G
Csequential_602_bidirectional_602_backward_lstm_602_while_identity_2G
Csequential_602_bidirectional_602_backward_lstm_602_while_identity_3G
Csequential_602_bidirectional_602_backward_lstm_602_while_identity_4G
Csequential_602_bidirectional_602_backward_lstm_602_while_identity_5G
Csequential_602_bidirectional_602_backward_lstm_602_while_identity_6
{sequential_602_bidirectional_602_backward_lstm_602_while_sequential_602_bidirectional_602_backward_lstm_602_strided_slice_1Љ
Јsequential_602_bidirectional_602_backward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_sequential_602_bidirectional_602_backward_lstm_602_tensorarrayunstack_tensorlistfromtensorz
vsequential_602_bidirectional_602_backward_lstm_602_while_less_sequential_602_bidirectional_602_backward_lstm_602_sub_1y
fsequential_602_bidirectional_602_backward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource:	»{
hsequential_602_bidirectional_602_backward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource:	2»v
gsequential_602_bidirectional_602_backward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource:	»ИҐ^sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpҐ]sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOpҐ_sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp©
jsequential_602/bidirectional_602/backward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2l
jsequential_602/bidirectional_602/backward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeЖ
\sequential_602/bidirectional_602/backward_lstm_602/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemєsequential_602_bidirectional_602_backward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_sequential_602_bidirectional_602_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0Dsequential_602_bidirectional_602_backward_lstm_602_while_placeholderssequential_602/bidirectional_602/backward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02^
\sequential_602/bidirectional_602/backward_lstm_602/while/TensorArrayV2Read/TensorListGetItemф
=sequential_602/bidirectional_602/backward_lstm_602/while/LessLessxsequential_602_bidirectional_602_backward_lstm_602_while_less_sequential_602_bidirectional_602_backward_lstm_602_sub_1_0Dsequential_602_bidirectional_602_backward_lstm_602_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2?
=sequential_602/bidirectional_602/backward_lstm_602/while/Lessи
]sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOphsequential_602_bidirectional_602_backward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02_
]sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp©
Nsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMulMatMulcsequential_602/bidirectional_602/backward_lstm_602/while/TensorArrayV2Read/TensorListGetItem:item:0esequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2P
Nsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMulо
_sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOpjsequential_602_bidirectional_602_backward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02a
_sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpТ
Psequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMul_1MatMulFsequential_602_bidirectional_602_backward_lstm_602_while_placeholder_3gsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2R
Psequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMul_1М
Ksequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/addAddV2Xsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMul:product:0Zsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2M
Ksequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/addз
^sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOpisequential_602_bidirectional_602_backward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02`
^sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpЩ
Osequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/BiasAddBiasAddOsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/add:z:0fsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2Q
Osequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/BiasAddф
Wsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2Y
Wsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/split/split_dimя
Msequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/splitSplit`sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/split/split_dim:output:0Xsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2O
Msequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/splitЈ
Osequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/SigmoidSigmoidVsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22Q
Osequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/Sigmoidї
Qsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/Sigmoid_1SigmoidVsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22S
Qsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/Sigmoid_1т
Ksequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/mulMulUsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/Sigmoid_1:y:0Fsequential_602_bidirectional_602_backward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22M
Ksequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/mulЃ
Lsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/ReluReluVsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22N
Lsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/ReluИ
Msequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/mul_1MulSsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/Sigmoid:y:0Zsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22O
Msequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/mul_1э
Msequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/add_1AddV2Osequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/mul:z:0Qsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22O
Msequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/add_1ї
Qsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/Sigmoid_2SigmoidVsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22S
Qsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/Sigmoid_2≠
Nsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/Relu_1ReluQsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22P
Nsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/Relu_1М
Msequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/mul_2MulUsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/Sigmoid_2:y:0\sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22O
Msequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/mul_2Ь
?sequential_602/bidirectional_602/backward_lstm_602/while/SelectSelectAsequential_602/bidirectional_602/backward_lstm_602/while/Less:z:0Qsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/mul_2:z:0Fsequential_602_bidirectional_602_backward_lstm_602_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22A
?sequential_602/bidirectional_602/backward_lstm_602/while/Select†
Asequential_602/bidirectional_602/backward_lstm_602/while/Select_1SelectAsequential_602/bidirectional_602/backward_lstm_602/while/Less:z:0Qsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/mul_2:z:0Fsequential_602_bidirectional_602_backward_lstm_602_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22C
Asequential_602/bidirectional_602/backward_lstm_602/while/Select_1†
Asequential_602/bidirectional_602/backward_lstm_602/while/Select_2SelectAsequential_602/bidirectional_602/backward_lstm_602/while/Less:z:0Qsequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/add_1:z:0Fsequential_602_bidirectional_602_backward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22C
Asequential_602/bidirectional_602/backward_lstm_602/while/Select_2Ў
]sequential_602/bidirectional_602/backward_lstm_602/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemFsequential_602_bidirectional_602_backward_lstm_602_while_placeholder_1Dsequential_602_bidirectional_602_backward_lstm_602_while_placeholderHsequential_602/bidirectional_602/backward_lstm_602/while/Select:output:0*
_output_shapes
: *
element_dtype02_
]sequential_602/bidirectional_602/backward_lstm_602/while/TensorArrayV2Write/TensorListSetItem¬
>sequential_602/bidirectional_602/backward_lstm_602/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_602/bidirectional_602/backward_lstm_602/while/add/yµ
<sequential_602/bidirectional_602/backward_lstm_602/while/addAddV2Dsequential_602_bidirectional_602_backward_lstm_602_while_placeholderGsequential_602/bidirectional_602/backward_lstm_602/while/add/y:output:0*
T0*
_output_shapes
: 2>
<sequential_602/bidirectional_602/backward_lstm_602/while/add∆
@sequential_602/bidirectional_602/backward_lstm_602/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2B
@sequential_602/bidirectional_602/backward_lstm_602/while/add_1/yх
>sequential_602/bidirectional_602/backward_lstm_602/while/add_1AddV2~sequential_602_bidirectional_602_backward_lstm_602_while_sequential_602_bidirectional_602_backward_lstm_602_while_loop_counterIsequential_602/bidirectional_602/backward_lstm_602/while/add_1/y:output:0*
T0*
_output_shapes
: 2@
>sequential_602/bidirectional_602/backward_lstm_602/while/add_1Ј
Asequential_602/bidirectional_602/backward_lstm_602/while/IdentityIdentityBsequential_602/bidirectional_602/backward_lstm_602/while/add_1:z:0>^sequential_602/bidirectional_602/backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2C
Asequential_602/bidirectional_602/backward_lstm_602/while/Identityю
Csequential_602/bidirectional_602/backward_lstm_602/while/Identity_1IdentityДsequential_602_bidirectional_602_backward_lstm_602_while_sequential_602_bidirectional_602_backward_lstm_602_while_maximum_iterations>^sequential_602/bidirectional_602/backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2E
Csequential_602/bidirectional_602/backward_lstm_602/while/Identity_1є
Csequential_602/bidirectional_602/backward_lstm_602/while/Identity_2Identity@sequential_602/bidirectional_602/backward_lstm_602/while/add:z:0>^sequential_602/bidirectional_602/backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2E
Csequential_602/bidirectional_602/backward_lstm_602/while/Identity_2ж
Csequential_602/bidirectional_602/backward_lstm_602/while/Identity_3Identitymsequential_602/bidirectional_602/backward_lstm_602/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential_602/bidirectional_602/backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2E
Csequential_602/bidirectional_602/backward_lstm_602/while/Identity_3“
Csequential_602/bidirectional_602/backward_lstm_602/while/Identity_4IdentityHsequential_602/bidirectional_602/backward_lstm_602/while/Select:output:0>^sequential_602/bidirectional_602/backward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22E
Csequential_602/bidirectional_602/backward_lstm_602/while/Identity_4‘
Csequential_602/bidirectional_602/backward_lstm_602/while/Identity_5IdentityJsequential_602/bidirectional_602/backward_lstm_602/while/Select_1:output:0>^sequential_602/bidirectional_602/backward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22E
Csequential_602/bidirectional_602/backward_lstm_602/while/Identity_5‘
Csequential_602/bidirectional_602/backward_lstm_602/while/Identity_6IdentityJsequential_602/bidirectional_602/backward_lstm_602/while/Select_2:output:0>^sequential_602/bidirectional_602/backward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22E
Csequential_602/bidirectional_602/backward_lstm_602/while/Identity_6г
=sequential_602/bidirectional_602/backward_lstm_602/while/NoOpNoOp_^sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp^^sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp`^sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2?
=sequential_602/bidirectional_602/backward_lstm_602/while/NoOp"П
Asequential_602_bidirectional_602_backward_lstm_602_while_identityJsequential_602/bidirectional_602/backward_lstm_602/while/Identity:output:0"У
Csequential_602_bidirectional_602_backward_lstm_602_while_identity_1Lsequential_602/bidirectional_602/backward_lstm_602/while/Identity_1:output:0"У
Csequential_602_bidirectional_602_backward_lstm_602_while_identity_2Lsequential_602/bidirectional_602/backward_lstm_602/while/Identity_2:output:0"У
Csequential_602_bidirectional_602_backward_lstm_602_while_identity_3Lsequential_602/bidirectional_602/backward_lstm_602/while/Identity_3:output:0"У
Csequential_602_bidirectional_602_backward_lstm_602_while_identity_4Lsequential_602/bidirectional_602/backward_lstm_602/while/Identity_4:output:0"У
Csequential_602_bidirectional_602_backward_lstm_602_while_identity_5Lsequential_602/bidirectional_602/backward_lstm_602/while/Identity_5:output:0"У
Csequential_602_bidirectional_602_backward_lstm_602_while_identity_6Lsequential_602/bidirectional_602/backward_lstm_602/while/Identity_6:output:0"т
vsequential_602_bidirectional_602_backward_lstm_602_while_less_sequential_602_bidirectional_602_backward_lstm_602_sub_1xsequential_602_bidirectional_602_backward_lstm_602_while_less_sequential_602_bidirectional_602_backward_lstm_602_sub_1_0"‘
gsequential_602_bidirectional_602_backward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resourceisequential_602_bidirectional_602_backward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0"÷
hsequential_602_bidirectional_602_backward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resourcejsequential_602_bidirectional_602_backward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0"“
fsequential_602_bidirectional_602_backward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resourcehsequential_602_bidirectional_602_backward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0"ь
{sequential_602_bidirectional_602_backward_lstm_602_while_sequential_602_bidirectional_602_backward_lstm_602_strided_slice_1}sequential_602_bidirectional_602_backward_lstm_602_while_sequential_602_bidirectional_602_backward_lstm_602_strided_slice_1_0"ц
Јsequential_602_bidirectional_602_backward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_sequential_602_bidirectional_602_backward_lstm_602_tensorarrayunstack_tensorlistfromtensorєsequential_602_bidirectional_602_backward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_sequential_602_bidirectional_602_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2ј
^sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp^sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp2Њ
]sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp]sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp2¬
_sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp_sequential_602/bidirectional_602/backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
жF
Ю
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_60933786

inputs*
lstm_cell_1807_60933704:	»*
lstm_cell_1807_60933706:	2»&
lstm_cell_1807_60933708:	»
identityИҐ&lstm_cell_1807/StatefulPartitionedCallҐwhileD
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
&lstm_cell_1807/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1807_60933704lstm_cell_1807_60933706lstm_cell_1807_60933708*
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
L__inference_lstm_cell_1807_layer_call_and_return_conditional_losses_609337032(
&lstm_cell_1807/StatefulPartitionedCallП
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1807_60933704lstm_cell_1807_60933706lstm_cell_1807_60933708*
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
while_body_60933717*
condR
while_cond_60933716*K
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
NoOpNoOp'^lstm_cell_1807/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2P
&lstm_cell_1807/StatefulPartitionedCall&lstm_cell_1807/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
к
Љ
%backward_lstm_602_while_cond_60935966@
<backward_lstm_602_while_backward_lstm_602_while_loop_counterF
Bbackward_lstm_602_while_backward_lstm_602_while_maximum_iterations'
#backward_lstm_602_while_placeholder)
%backward_lstm_602_while_placeholder_1)
%backward_lstm_602_while_placeholder_2)
%backward_lstm_602_while_placeholder_3)
%backward_lstm_602_while_placeholder_4B
>backward_lstm_602_while_less_backward_lstm_602_strided_slice_1Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60935966___redundant_placeholder0Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60935966___redundant_placeholder1Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60935966___redundant_placeholder2Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60935966___redundant_placeholder3Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60935966___redundant_placeholder4$
 backward_lstm_602_while_identity
 
backward_lstm_602/while/LessLess#backward_lstm_602_while_placeholder>backward_lstm_602_while_less_backward_lstm_602_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_602/while/LessУ
 backward_lstm_602/while/IdentityIdentity backward_lstm_602/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_602/while/Identity"M
 backward_lstm_602_while_identity)backward_lstm_602/while/Identity:output:0*(
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
Т
‘
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_60935224

inputs,
forward_lstm_602_60935054:	»,
forward_lstm_602_60935056:	2»(
forward_lstm_602_60935058:	»-
backward_lstm_602_60935214:	»-
backward_lstm_602_60935216:	2»)
backward_lstm_602_60935218:	»
identityИҐ)backward_lstm_602/StatefulPartitionedCallҐ(forward_lstm_602/StatefulPartitionedCallя
(forward_lstm_602/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_602_60935054forward_lstm_602_60935056forward_lstm_602_60935058*
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
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_609350532*
(forward_lstm_602/StatefulPartitionedCallе
)backward_lstm_602/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_602_60935214backward_lstm_602_60935216backward_lstm_602_60935218*
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
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_609352132+
)backward_lstm_602/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis‘
concatConcatV21forward_lstm_602/StatefulPartitionedCall:output:02backward_lstm_602/StatefulPartitionedCall:output:0concat/axis:output:0*
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
NoOpNoOp*^backward_lstm_602/StatefulPartitionedCall)^forward_lstm_602/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : 2V
)backward_lstm_602/StatefulPartitionedCall)backward_lstm_602/StatefulPartitionedCall2T
(forward_lstm_602/StatefulPartitionedCall(forward_lstm_602/StatefulPartitionedCall:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
м]
≤
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_60935053

inputs@
-lstm_cell_1807_matmul_readvariableop_resource:	»B
/lstm_cell_1807_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1807_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1807/BiasAdd/ReadVariableOpҐ$lstm_cell_1807/MatMul/ReadVariableOpҐ&lstm_cell_1807/MatMul_1/ReadVariableOpҐwhileD
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
$lstm_cell_1807/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1807_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1807/MatMul/ReadVariableOp≥
lstm_cell_1807/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/MatMulЅ
&lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1807_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1807/MatMul_1/ReadVariableOpѓ
lstm_cell_1807/MatMul_1MatMulzeros:output:0.lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/MatMul_1®
lstm_cell_1807/addAddV2lstm_cell_1807/MatMul:product:0!lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/addЇ
%lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1807_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1807/BiasAdd/ReadVariableOpµ
lstm_cell_1807/BiasAddBiasAddlstm_cell_1807/add:z:0-lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/BiasAddВ
lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1807/split/split_dimы
lstm_cell_1807/splitSplit'lstm_cell_1807/split/split_dim:output:0lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1807/splitМ
lstm_cell_1807/SigmoidSigmoidlstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/SigmoidР
lstm_cell_1807/Sigmoid_1Sigmoidlstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Sigmoid_1С
lstm_cell_1807/mulMullstm_cell_1807/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/mulГ
lstm_cell_1807/ReluRelulstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Relu§
lstm_cell_1807/mul_1Mullstm_cell_1807/Sigmoid:y:0!lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/mul_1Щ
lstm_cell_1807/add_1AddV2lstm_cell_1807/mul:z:0lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/add_1Р
lstm_cell_1807/Sigmoid_2Sigmoidlstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Sigmoid_2В
lstm_cell_1807/Relu_1Relulstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Relu_1®
lstm_cell_1807/mul_2Mullstm_cell_1807/Sigmoid_2:y:0#lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1807_matmul_readvariableop_resource/lstm_cell_1807_matmul_1_readvariableop_resource.lstm_cell_1807_biasadd_readvariableop_resource*
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
while_body_60934969*
condR
while_cond_60934968*K
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
NoOpNoOp&^lstm_cell_1807/BiasAdd/ReadVariableOp%^lstm_cell_1807/MatMul/ReadVariableOp'^lstm_cell_1807/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1807/BiasAdd/ReadVariableOp%lstm_cell_1807/BiasAdd/ReadVariableOp2L
$lstm_cell_1807/MatMul/ReadVariableOp$lstm_cell_1807/MatMul/ReadVariableOp2P
&lstm_cell_1807/MatMul_1/ReadVariableOp&lstm_cell_1807/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
я
Ќ
while_cond_60935493
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60935493___redundant_placeholder06
2while_while_cond_60935493___redundant_placeholder16
2while_while_cond_60935493___redundant_placeholder26
2while_while_cond_60935493___redundant_placeholder3
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
while_body_60938855
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1808_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1808_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1808_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1808_matmul_readvariableop_resource:	»H
5while_lstm_cell_1808_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1808_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1808/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1808/MatMul/ReadVariableOpҐ,while/lstm_cell_1808/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1808_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1808/MatMul/ReadVariableOpЁ
while/lstm_cell_1808/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/MatMul’
,while/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1808_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1808/MatMul_1/ReadVariableOp∆
while/lstm_cell_1808/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/MatMul_1ј
while/lstm_cell_1808/addAddV2%while/lstm_cell_1808/MatMul:product:0'while/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/addќ
+while/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1808_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1808/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1808/BiasAddBiasAddwhile/lstm_cell_1808/add:z:03while/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/BiasAddО
$while/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1808/split/split_dimУ
while/lstm_cell_1808/splitSplit-while/lstm_cell_1808/split/split_dim:output:0%while/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1808/splitЮ
while/lstm_cell_1808/SigmoidSigmoid#while/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/SigmoidҐ
while/lstm_cell_1808/Sigmoid_1Sigmoid#while/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1808/Sigmoid_1¶
while/lstm_cell_1808/mulMul"while/lstm_cell_1808/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/mulХ
while/lstm_cell_1808/ReluRelu#while/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/ReluЉ
while/lstm_cell_1808/mul_1Mul while/lstm_cell_1808/Sigmoid:y:0'while/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/mul_1±
while/lstm_cell_1808/add_1AddV2while/lstm_cell_1808/mul:z:0while/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/add_1Ґ
while/lstm_cell_1808/Sigmoid_2Sigmoid#while/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1808/Sigmoid_2Ф
while/lstm_cell_1808/Relu_1Reluwhile/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/Relu_1ј
while/lstm_cell_1808/mul_2Mul"while/lstm_cell_1808/Sigmoid_2:y:0)while/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1808/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1808/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1808/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1808/BiasAdd/ReadVariableOp+^while/lstm_cell_1808/MatMul/ReadVariableOp-^while/lstm_cell_1808/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1808_biasadd_readvariableop_resource6while_lstm_cell_1808_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1808_matmul_1_readvariableop_resource7while_lstm_cell_1808_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1808_matmul_readvariableop_resource5while_lstm_cell_1808_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1808/BiasAdd/ReadVariableOp+while/lstm_cell_1808/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1808/MatMul/ReadVariableOp*while/lstm_cell_1808/MatMul/ReadVariableOp2\
,while/lstm_cell_1808/MatMul_1/ReadVariableOp,while/lstm_cell_1808/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
ёю
п
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_60937358
inputs_0Q
>forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource:	»S
@forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource:	2»N
?forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource:	»R
?backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resource:	»T
Abackward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource:	2»O
@backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource:	»
identityИҐ7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpҐ6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpҐ8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpҐbackward_lstm_602/whileҐ6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpҐ5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpҐ7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpҐforward_lstm_602/whileh
forward_lstm_602/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_602/ShapeЦ
$forward_lstm_602/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_602/strided_slice/stackЪ
&forward_lstm_602/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_602/strided_slice/stack_1Ъ
&forward_lstm_602/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_602/strided_slice/stack_2»
forward_lstm_602/strided_sliceStridedSliceforward_lstm_602/Shape:output:0-forward_lstm_602/strided_slice/stack:output:0/forward_lstm_602/strided_slice/stack_1:output:0/forward_lstm_602/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_602/strided_slice~
forward_lstm_602/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_602/zeros/mul/y∞
forward_lstm_602/zeros/mulMul'forward_lstm_602/strided_slice:output:0%forward_lstm_602/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros/mulБ
forward_lstm_602/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
forward_lstm_602/zeros/Less/yЂ
forward_lstm_602/zeros/LessLessforward_lstm_602/zeros/mul:z:0&forward_lstm_602/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros/LessД
forward_lstm_602/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_602/zeros/packed/1«
forward_lstm_602/zeros/packedPack'forward_lstm_602/strided_slice:output:0(forward_lstm_602/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_602/zeros/packedЕ
forward_lstm_602/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_602/zeros/Constє
forward_lstm_602/zerosFill&forward_lstm_602/zeros/packed:output:0%forward_lstm_602/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/zerosВ
forward_lstm_602/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_602/zeros_1/mul/yґ
forward_lstm_602/zeros_1/mulMul'forward_lstm_602/strided_slice:output:0'forward_lstm_602/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros_1/mulЕ
forward_lstm_602/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2!
forward_lstm_602/zeros_1/Less/y≥
forward_lstm_602/zeros_1/LessLess forward_lstm_602/zeros_1/mul:z:0(forward_lstm_602/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros_1/LessИ
!forward_lstm_602/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_602/zeros_1/packed/1Ќ
forward_lstm_602/zeros_1/packedPack'forward_lstm_602/strided_slice:output:0*forward_lstm_602/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_602/zeros_1/packedЙ
forward_lstm_602/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_602/zeros_1/ConstЅ
forward_lstm_602/zeros_1Fill(forward_lstm_602/zeros_1/packed:output:0'forward_lstm_602/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/zeros_1Ч
forward_lstm_602/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_602/transpose/permЅ
forward_lstm_602/transpose	Transposeinputs_0(forward_lstm_602/transpose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
forward_lstm_602/transposeВ
forward_lstm_602/Shape_1Shapeforward_lstm_602/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_602/Shape_1Ъ
&forward_lstm_602/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_602/strided_slice_1/stackЮ
(forward_lstm_602/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_1/stack_1Ю
(forward_lstm_602/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_1/stack_2‘
 forward_lstm_602/strided_slice_1StridedSlice!forward_lstm_602/Shape_1:output:0/forward_lstm_602/strided_slice_1/stack:output:01forward_lstm_602/strided_slice_1/stack_1:output:01forward_lstm_602/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_602/strided_slice_1І
,forward_lstm_602/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,forward_lstm_602/TensorArrayV2/element_shapeц
forward_lstm_602/TensorArrayV2TensorListReserve5forward_lstm_602/TensorArrayV2/element_shape:output:0)forward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_602/TensorArrayV2б
Fforward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2H
Fforward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeЉ
8forward_lstm_602/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_602/transpose:y:0Oforward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_602/TensorArrayUnstack/TensorListFromTensorЪ
&forward_lstm_602/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_602/strided_slice_2/stackЮ
(forward_lstm_602/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_2/stack_1Ю
(forward_lstm_602/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_2/stack_2л
 forward_lstm_602/strided_slice_2StridedSliceforward_lstm_602/transpose:y:0/forward_lstm_602/strided_slice_2/stack:output:01forward_lstm_602/strided_slice_2/stack_1:output:01forward_lstm_602/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2"
 forward_lstm_602/strided_slice_2о
5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOp>forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype027
5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpч
&forward_lstm_602/lstm_cell_1807/MatMulMatMul)forward_lstm_602/strided_slice_2:output:0=forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2(
&forward_lstm_602/lstm_cell_1807/MatMulф
7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOp@forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype029
7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpу
(forward_lstm_602/lstm_cell_1807/MatMul_1MatMulforward_lstm_602/zeros:output:0?forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(forward_lstm_602/lstm_cell_1807/MatMul_1м
#forward_lstm_602/lstm_cell_1807/addAddV20forward_lstm_602/lstm_cell_1807/MatMul:product:02forward_lstm_602/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2%
#forward_lstm_602/lstm_cell_1807/addн
6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOp?forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype028
6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpщ
'forward_lstm_602/lstm_cell_1807/BiasAddBiasAdd'forward_lstm_602/lstm_cell_1807/add:z:0>forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'forward_lstm_602/lstm_cell_1807/BiasAdd§
/forward_lstm_602/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/forward_lstm_602/lstm_cell_1807/split/split_dimњ
%forward_lstm_602/lstm_cell_1807/splitSplit8forward_lstm_602/lstm_cell_1807/split/split_dim:output:00forward_lstm_602/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2'
%forward_lstm_602/lstm_cell_1807/splitњ
'forward_lstm_602/lstm_cell_1807/SigmoidSigmoid.forward_lstm_602/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'forward_lstm_602/lstm_cell_1807/Sigmoid√
)forward_lstm_602/lstm_cell_1807/Sigmoid_1Sigmoid.forward_lstm_602/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_602/lstm_cell_1807/Sigmoid_1’
#forward_lstm_602/lstm_cell_1807/mulMul-forward_lstm_602/lstm_cell_1807/Sigmoid_1:y:0!forward_lstm_602/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22%
#forward_lstm_602/lstm_cell_1807/mulґ
$forward_lstm_602/lstm_cell_1807/ReluRelu.forward_lstm_602/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22&
$forward_lstm_602/lstm_cell_1807/Reluи
%forward_lstm_602/lstm_cell_1807/mul_1Mul+forward_lstm_602/lstm_cell_1807/Sigmoid:y:02forward_lstm_602/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_602/lstm_cell_1807/mul_1Ё
%forward_lstm_602/lstm_cell_1807/add_1AddV2'forward_lstm_602/lstm_cell_1807/mul:z:0)forward_lstm_602/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_602/lstm_cell_1807/add_1√
)forward_lstm_602/lstm_cell_1807/Sigmoid_2Sigmoid.forward_lstm_602/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_602/lstm_cell_1807/Sigmoid_2µ
&forward_lstm_602/lstm_cell_1807/Relu_1Relu)forward_lstm_602/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&forward_lstm_602/lstm_cell_1807/Relu_1м
%forward_lstm_602/lstm_cell_1807/mul_2Mul-forward_lstm_602/lstm_cell_1807/Sigmoid_2:y:04forward_lstm_602/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_602/lstm_cell_1807/mul_2±
.forward_lstm_602/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   20
.forward_lstm_602/TensorArrayV2_1/element_shapeь
 forward_lstm_602/TensorArrayV2_1TensorListReserve7forward_lstm_602/TensorArrayV2_1/element_shape:output:0)forward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_602/TensorArrayV2_1p
forward_lstm_602/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_602/time°
)forward_lstm_602/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2+
)forward_lstm_602/while/maximum_iterationsМ
#forward_lstm_602/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_602/while/loop_counterФ
forward_lstm_602/whileWhile,forward_lstm_602/while/loop_counter:output:02forward_lstm_602/while/maximum_iterations:output:0forward_lstm_602/time:output:0)forward_lstm_602/TensorArrayV2_1:handle:0forward_lstm_602/zeros:output:0!forward_lstm_602/zeros_1:output:0)forward_lstm_602/strided_slice_1:output:0Hforward_lstm_602/TensorArrayUnstack/TensorListFromTensor:output_handle:0>forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource@forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource?forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource*
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
$forward_lstm_602_while_body_60937123*0
cond(R&
$forward_lstm_602_while_cond_60937122*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
forward_lstm_602/while„
Aforward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2C
Aforward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeµ
3forward_lstm_602/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_602/while:output:3Jforward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype025
3forward_lstm_602/TensorArrayV2Stack/TensorListStack£
&forward_lstm_602/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2(
&forward_lstm_602/strided_slice_3/stackЮ
(forward_lstm_602/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_602/strided_slice_3/stack_1Ю
(forward_lstm_602/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_3/stack_2А
 forward_lstm_602/strided_slice_3StridedSlice<forward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_602/strided_slice_3/stack:output:01forward_lstm_602/strided_slice_3/stack_1:output:01forward_lstm_602/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2"
 forward_lstm_602/strided_slice_3Ы
!forward_lstm_602/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_602/transpose_1/permт
forward_lstm_602/transpose_1	Transpose<forward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_602/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
forward_lstm_602/transpose_1И
forward_lstm_602/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_602/runtimej
backward_lstm_602/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_602/ShapeШ
%backward_lstm_602/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_602/strided_slice/stackЬ
'backward_lstm_602/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_602/strided_slice/stack_1Ь
'backward_lstm_602/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_602/strided_slice/stack_2ќ
backward_lstm_602/strided_sliceStridedSlice backward_lstm_602/Shape:output:0.backward_lstm_602/strided_slice/stack:output:00backward_lstm_602/strided_slice/stack_1:output:00backward_lstm_602/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_602/strided_sliceА
backward_lstm_602/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_602/zeros/mul/yі
backward_lstm_602/zeros/mulMul(backward_lstm_602/strided_slice:output:0&backward_lstm_602/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/zeros/mulГ
backward_lstm_602/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
backward_lstm_602/zeros/Less/yѓ
backward_lstm_602/zeros/LessLessbackward_lstm_602/zeros/mul:z:0'backward_lstm_602/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/zeros/LessЖ
 backward_lstm_602/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_602/zeros/packed/1Ћ
backward_lstm_602/zeros/packedPack(backward_lstm_602/strided_slice:output:0)backward_lstm_602/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_602/zeros/packedЗ
backward_lstm_602/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_602/zeros/Constљ
backward_lstm_602/zerosFill'backward_lstm_602/zeros/packed:output:0&backward_lstm_602/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_602/zerosД
backward_lstm_602/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_602/zeros_1/mul/yЇ
backward_lstm_602/zeros_1/mulMul(backward_lstm_602/strided_slice:output:0(backward_lstm_602/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/zeros_1/mulЗ
 backward_lstm_602/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 backward_lstm_602/zeros_1/Less/yЈ
backward_lstm_602/zeros_1/LessLess!backward_lstm_602/zeros_1/mul:z:0)backward_lstm_602/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_602/zeros_1/LessК
"backward_lstm_602/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_602/zeros_1/packed/1—
 backward_lstm_602/zeros_1/packedPack(backward_lstm_602/strided_slice:output:0+backward_lstm_602/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_602/zeros_1/packedЛ
backward_lstm_602/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_602/zeros_1/Const≈
backward_lstm_602/zeros_1Fill)backward_lstm_602/zeros_1/packed:output:0(backward_lstm_602/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_602/zeros_1Щ
 backward_lstm_602/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_602/transpose/permƒ
backward_lstm_602/transpose	Transposeinputs_0)backward_lstm_602/transpose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
backward_lstm_602/transposeЕ
backward_lstm_602/Shape_1Shapebackward_lstm_602/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_602/Shape_1Ь
'backward_lstm_602/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_602/strided_slice_1/stack†
)backward_lstm_602/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_1/stack_1†
)backward_lstm_602/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_1/stack_2Џ
!backward_lstm_602/strided_slice_1StridedSlice"backward_lstm_602/Shape_1:output:00backward_lstm_602/strided_slice_1/stack:output:02backward_lstm_602/strided_slice_1/stack_1:output:02backward_lstm_602/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_602/strided_slice_1©
-backward_lstm_602/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-backward_lstm_602/TensorArrayV2/element_shapeъ
backward_lstm_602/TensorArrayV2TensorListReserve6backward_lstm_602/TensorArrayV2/element_shape:output:0*backward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_602/TensorArrayV2О
 backward_lstm_602/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_602/ReverseV2/axisџ
backward_lstm_602/ReverseV2	ReverseV2backward_lstm_602/transpose:y:0)backward_lstm_602/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
backward_lstm_602/ReverseV2г
Gbackward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2I
Gbackward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape≈
9backward_lstm_602/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_602/ReverseV2:output:0Pbackward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_602/TensorArrayUnstack/TensorListFromTensorЬ
'backward_lstm_602/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_602/strided_slice_2/stack†
)backward_lstm_602/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_2/stack_1†
)backward_lstm_602/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_2/stack_2с
!backward_lstm_602/strided_slice_2StridedSlicebackward_lstm_602/transpose:y:00backward_lstm_602/strided_slice_2/stack:output:02backward_lstm_602/strided_slice_2/stack_1:output:02backward_lstm_602/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2#
!backward_lstm_602/strided_slice_2с
6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOp?backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype028
6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpы
'backward_lstm_602/lstm_cell_1808/MatMulMatMul*backward_lstm_602/strided_slice_2:output:0>backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'backward_lstm_602/lstm_cell_1808/MatMulч
8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOpAbackward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02:
8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpч
)backward_lstm_602/lstm_cell_1808/MatMul_1MatMul backward_lstm_602/zeros:output:0@backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)backward_lstm_602/lstm_cell_1808/MatMul_1р
$backward_lstm_602/lstm_cell_1808/addAddV21backward_lstm_602/lstm_cell_1808/MatMul:product:03backward_lstm_602/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2&
$backward_lstm_602/lstm_cell_1808/addр
7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOp@backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype029
7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpэ
(backward_lstm_602/lstm_cell_1808/BiasAddBiasAdd(backward_lstm_602/lstm_cell_1808/add:z:0?backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(backward_lstm_602/lstm_cell_1808/BiasAdd¶
0backward_lstm_602/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0backward_lstm_602/lstm_cell_1808/split/split_dim√
&backward_lstm_602/lstm_cell_1808/splitSplit9backward_lstm_602/lstm_cell_1808/split/split_dim:output:01backward_lstm_602/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2(
&backward_lstm_602/lstm_cell_1808/split¬
(backward_lstm_602/lstm_cell_1808/SigmoidSigmoid/backward_lstm_602/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(backward_lstm_602/lstm_cell_1808/Sigmoid∆
*backward_lstm_602/lstm_cell_1808/Sigmoid_1Sigmoid/backward_lstm_602/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_602/lstm_cell_1808/Sigmoid_1ў
$backward_lstm_602/lstm_cell_1808/mulMul.backward_lstm_602/lstm_cell_1808/Sigmoid_1:y:0"backward_lstm_602/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$backward_lstm_602/lstm_cell_1808/mulє
%backward_lstm_602/lstm_cell_1808/ReluRelu/backward_lstm_602/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22'
%backward_lstm_602/lstm_cell_1808/Reluм
&backward_lstm_602/lstm_cell_1808/mul_1Mul,backward_lstm_602/lstm_cell_1808/Sigmoid:y:03backward_lstm_602/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_602/lstm_cell_1808/mul_1б
&backward_lstm_602/lstm_cell_1808/add_1AddV2(backward_lstm_602/lstm_cell_1808/mul:z:0*backward_lstm_602/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_602/lstm_cell_1808/add_1∆
*backward_lstm_602/lstm_cell_1808/Sigmoid_2Sigmoid/backward_lstm_602/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_602/lstm_cell_1808/Sigmoid_2Є
'backward_lstm_602/lstm_cell_1808/Relu_1Relu*backward_lstm_602/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'backward_lstm_602/lstm_cell_1808/Relu_1р
&backward_lstm_602/lstm_cell_1808/mul_2Mul.backward_lstm_602/lstm_cell_1808/Sigmoid_2:y:05backward_lstm_602/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_602/lstm_cell_1808/mul_2≥
/backward_lstm_602/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   21
/backward_lstm_602/TensorArrayV2_1/element_shapeА
!backward_lstm_602/TensorArrayV2_1TensorListReserve8backward_lstm_602/TensorArrayV2_1/element_shape:output:0*backward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_602/TensorArrayV2_1r
backward_lstm_602/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_602/time£
*backward_lstm_602/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2,
*backward_lstm_602/while/maximum_iterationsО
$backward_lstm_602/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_602/while/loop_counter£
backward_lstm_602/whileWhile-backward_lstm_602/while/loop_counter:output:03backward_lstm_602/while/maximum_iterations:output:0backward_lstm_602/time:output:0*backward_lstm_602/TensorArrayV2_1:handle:0 backward_lstm_602/zeros:output:0"backward_lstm_602/zeros_1:output:0*backward_lstm_602/strided_slice_1:output:0Ibackward_lstm_602/TensorArrayUnstack/TensorListFromTensor:output_handle:0?backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resourceAbackward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource@backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource*
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
%backward_lstm_602_while_body_60937272*1
cond)R'
%backward_lstm_602_while_cond_60937271*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
backward_lstm_602/whileў
Bbackward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2D
Bbackward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeє
4backward_lstm_602/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_602/while:output:3Kbackward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype026
4backward_lstm_602/TensorArrayV2Stack/TensorListStack•
'backward_lstm_602/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2)
'backward_lstm_602/strided_slice_3/stack†
)backward_lstm_602/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_602/strided_slice_3/stack_1†
)backward_lstm_602/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_3/stack_2Ж
!backward_lstm_602/strided_slice_3StridedSlice=backward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_602/strided_slice_3/stack:output:02backward_lstm_602/strided_slice_3/stack_1:output:02backward_lstm_602/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2#
!backward_lstm_602/strided_slice_3Э
"backward_lstm_602/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_602/transpose_1/permц
backward_lstm_602/transpose_1	Transpose=backward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_602/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
backward_lstm_602/transpose_1К
backward_lstm_602/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_602/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisƒ
concatConcatV2)forward_lstm_602/strided_slice_3:output:0*backward_lstm_602/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp8^backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp7^backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp9^backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp^backward_lstm_602/while7^forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp6^forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp8^forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp^forward_lstm_602/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : 2r
7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp2p
6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp2t
8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp22
backward_lstm_602/whilebackward_lstm_602/while2p
6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp2n
5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp2r
7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp20
forward_lstm_602/whileforward_lstm_602/while:g c
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
ѕ@
д
while_body_60939161
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1808_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1808_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1808_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1808_matmul_readvariableop_resource:	»H
5while_lstm_cell_1808_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1808_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1808/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1808/MatMul/ReadVariableOpҐ,while/lstm_cell_1808/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1808_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1808/MatMul/ReadVariableOpЁ
while/lstm_cell_1808/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/MatMul’
,while/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1808_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1808/MatMul_1/ReadVariableOp∆
while/lstm_cell_1808/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/MatMul_1ј
while/lstm_cell_1808/addAddV2%while/lstm_cell_1808/MatMul:product:0'while/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/addќ
+while/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1808_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1808/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1808/BiasAddBiasAddwhile/lstm_cell_1808/add:z:03while/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/BiasAddО
$while/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1808/split/split_dimУ
while/lstm_cell_1808/splitSplit-while/lstm_cell_1808/split/split_dim:output:0%while/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1808/splitЮ
while/lstm_cell_1808/SigmoidSigmoid#while/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/SigmoidҐ
while/lstm_cell_1808/Sigmoid_1Sigmoid#while/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1808/Sigmoid_1¶
while/lstm_cell_1808/mulMul"while/lstm_cell_1808/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/mulХ
while/lstm_cell_1808/ReluRelu#while/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/ReluЉ
while/lstm_cell_1808/mul_1Mul while/lstm_cell_1808/Sigmoid:y:0'while/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/mul_1±
while/lstm_cell_1808/add_1AddV2while/lstm_cell_1808/mul:z:0while/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/add_1Ґ
while/lstm_cell_1808/Sigmoid_2Sigmoid#while/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1808/Sigmoid_2Ф
while/lstm_cell_1808/Relu_1Reluwhile/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/Relu_1ј
while/lstm_cell_1808/mul_2Mul"while/lstm_cell_1808/Sigmoid_2:y:0)while/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1808/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1808/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1808/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1808/BiasAdd/ReadVariableOp+^while/lstm_cell_1808/MatMul/ReadVariableOp-^while/lstm_cell_1808/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1808_biasadd_readvariableop_resource6while_lstm_cell_1808_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1808_matmul_1_readvariableop_resource7while_lstm_cell_1808_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1808_matmul_readvariableop_resource5while_lstm_cell_1808_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1808/BiasAdd/ReadVariableOp+while/lstm_cell_1808/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1808/MatMul/ReadVariableOp*while/lstm_cell_1808/MatMul/ReadVariableOp2\
,while/lstm_cell_1808/MatMul_1/ReadVariableOp,while/lstm_cell_1808/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
м]
≤
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_60938591

inputs@
-lstm_cell_1807_matmul_readvariableop_resource:	»B
/lstm_cell_1807_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1807_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1807/BiasAdd/ReadVariableOpҐ$lstm_cell_1807/MatMul/ReadVariableOpҐ&lstm_cell_1807/MatMul_1/ReadVariableOpҐwhileD
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
$lstm_cell_1807/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1807_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1807/MatMul/ReadVariableOp≥
lstm_cell_1807/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/MatMulЅ
&lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1807_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1807/MatMul_1/ReadVariableOpѓ
lstm_cell_1807/MatMul_1MatMulzeros:output:0.lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/MatMul_1®
lstm_cell_1807/addAddV2lstm_cell_1807/MatMul:product:0!lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/addЇ
%lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1807_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1807/BiasAdd/ReadVariableOpµ
lstm_cell_1807/BiasAddBiasAddlstm_cell_1807/add:z:0-lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/BiasAddВ
lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1807/split/split_dimы
lstm_cell_1807/splitSplit'lstm_cell_1807/split/split_dim:output:0lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1807/splitМ
lstm_cell_1807/SigmoidSigmoidlstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/SigmoidР
lstm_cell_1807/Sigmoid_1Sigmoidlstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Sigmoid_1С
lstm_cell_1807/mulMullstm_cell_1807/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/mulГ
lstm_cell_1807/ReluRelulstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Relu§
lstm_cell_1807/mul_1Mullstm_cell_1807/Sigmoid:y:0!lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/mul_1Щ
lstm_cell_1807/add_1AddV2lstm_cell_1807/mul:z:0lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/add_1Р
lstm_cell_1807/Sigmoid_2Sigmoidlstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Sigmoid_2В
lstm_cell_1807/Relu_1Relulstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Relu_1®
lstm_cell_1807/mul_2Mullstm_cell_1807/Sigmoid_2:y:0#lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1807_matmul_readvariableop_resource/lstm_cell_1807_matmul_1_readvariableop_resource.lstm_cell_1807_biasadd_readvariableop_resource*
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
while_body_60938507*
condR
while_cond_60938506*K
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
NoOpNoOp&^lstm_cell_1807/BiasAdd/ReadVariableOp%^lstm_cell_1807/MatMul/ReadVariableOp'^lstm_cell_1807/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1807/BiasAdd/ReadVariableOp%lstm_cell_1807/BiasAdd/ReadVariableOp2L
$lstm_cell_1807/MatMul/ReadVariableOp$lstm_cell_1807/MatMul/ReadVariableOp2P
&lstm_cell_1807/MatMul_1/ReadVariableOp&lstm_cell_1807/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
я
Ќ
while_cond_60939160
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60939160___redundant_placeholder06
2while_while_cond_60939160___redundant_placeholder16
2while_while_cond_60939160___redundant_placeholder26
2while_while_cond_60939160___redundant_placeholder3
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
зe
Ћ
$forward_lstm_602_while_body_60936228>
:forward_lstm_602_while_forward_lstm_602_while_loop_counterD
@forward_lstm_602_while_forward_lstm_602_while_maximum_iterations&
"forward_lstm_602_while_placeholder(
$forward_lstm_602_while_placeholder_1(
$forward_lstm_602_while_placeholder_2(
$forward_lstm_602_while_placeholder_3(
$forward_lstm_602_while_placeholder_4=
9forward_lstm_602_while_forward_lstm_602_strided_slice_1_0y
uforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_602_while_greater_forward_lstm_602_cast_0Y
Fforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0:	»[
Hforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0:	2»V
Gforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0:	»#
forward_lstm_602_while_identity%
!forward_lstm_602_while_identity_1%
!forward_lstm_602_while_identity_2%
!forward_lstm_602_while_identity_3%
!forward_lstm_602_while_identity_4%
!forward_lstm_602_while_identity_5%
!forward_lstm_602_while_identity_6;
7forward_lstm_602_while_forward_lstm_602_strided_slice_1w
sforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_602_while_greater_forward_lstm_602_castW
Dforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource:	»Y
Fforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource:	2»T
Eforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource:	»ИҐ<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpҐ;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOpҐ=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpе
Hforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2J
Hforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeє
:forward_lstm_602/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_602_while_placeholderQforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02<
:forward_lstm_602/while/TensorArrayV2Read/TensorListGetItem’
forward_lstm_602/while/GreaterGreater6forward_lstm_602_while_greater_forward_lstm_602_cast_0"forward_lstm_602_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2 
forward_lstm_602/while/GreaterВ
;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOpFforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02=
;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp°
,forward_lstm_602/while/lstm_cell_1807/MatMulMatMulAforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem:item:0Cforward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2.
,forward_lstm_602/while/lstm_cell_1807/MatMulИ
=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOpHforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02?
=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpК
.forward_lstm_602/while/lstm_cell_1807/MatMul_1MatMul$forward_lstm_602_while_placeholder_3Eforward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.forward_lstm_602/while/lstm_cell_1807/MatMul_1Д
)forward_lstm_602/while/lstm_cell_1807/addAddV26forward_lstm_602/while/lstm_cell_1807/MatMul:product:08forward_lstm_602/while/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)forward_lstm_602/while/lstm_cell_1807/addБ
<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOpGforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02>
<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpС
-forward_lstm_602/while/lstm_cell_1807/BiasAddBiasAdd-forward_lstm_602/while/lstm_cell_1807/add:z:0Dforward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-forward_lstm_602/while/lstm_cell_1807/BiasAdd∞
5forward_lstm_602/while/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5forward_lstm_602/while/lstm_cell_1807/split/split_dim„
+forward_lstm_602/while/lstm_cell_1807/splitSplit>forward_lstm_602/while/lstm_cell_1807/split/split_dim:output:06forward_lstm_602/while/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2-
+forward_lstm_602/while/lstm_cell_1807/split—
-forward_lstm_602/while/lstm_cell_1807/SigmoidSigmoid4forward_lstm_602/while/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-forward_lstm_602/while/lstm_cell_1807/Sigmoid’
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_1Sigmoid4forward_lstm_602/while/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_1к
)forward_lstm_602/while/lstm_cell_1807/mulMul3forward_lstm_602/while/lstm_cell_1807/Sigmoid_1:y:0$forward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_602/while/lstm_cell_1807/mul»
*forward_lstm_602/while/lstm_cell_1807/ReluRelu4forward_lstm_602/while/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22,
*forward_lstm_602/while/lstm_cell_1807/ReluА
+forward_lstm_602/while/lstm_cell_1807/mul_1Mul1forward_lstm_602/while/lstm_cell_1807/Sigmoid:y:08forward_lstm_602/while/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_602/while/lstm_cell_1807/mul_1х
+forward_lstm_602/while/lstm_cell_1807/add_1AddV2-forward_lstm_602/while/lstm_cell_1807/mul:z:0/forward_lstm_602/while/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_602/while/lstm_cell_1807/add_1’
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_2Sigmoid4forward_lstm_602/while/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_2«
,forward_lstm_602/while/lstm_cell_1807/Relu_1Relu/forward_lstm_602/while/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,forward_lstm_602/while/lstm_cell_1807/Relu_1Д
+forward_lstm_602/while/lstm_cell_1807/mul_2Mul3forward_lstm_602/while/lstm_cell_1807/Sigmoid_2:y:0:forward_lstm_602/while/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_602/while/lstm_cell_1807/mul_2х
forward_lstm_602/while/SelectSelect"forward_lstm_602/while/Greater:z:0/forward_lstm_602/while/lstm_cell_1807/mul_2:z:0$forward_lstm_602_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/while/Selectщ
forward_lstm_602/while/Select_1Select"forward_lstm_602/while/Greater:z:0/forward_lstm_602/while/lstm_cell_1807/mul_2:z:0$forward_lstm_602_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_602/while/Select_1щ
forward_lstm_602/while/Select_2Select"forward_lstm_602/while/Greater:z:0/forward_lstm_602/while/lstm_cell_1807/add_1:z:0$forward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_602/while/Select_2Ѓ
;forward_lstm_602/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_602_while_placeholder_1"forward_lstm_602_while_placeholder&forward_lstm_602/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_602/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_602/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_602/while/add/y≠
forward_lstm_602/while/addAddV2"forward_lstm_602_while_placeholder%forward_lstm_602/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/while/addВ
forward_lstm_602/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_602/while/add_1/yЋ
forward_lstm_602/while/add_1AddV2:forward_lstm_602_while_forward_lstm_602_while_loop_counter'forward_lstm_602/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/while/add_1ѓ
forward_lstm_602/while/IdentityIdentity forward_lstm_602/while/add_1:z:0^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_602/while/Identity”
!forward_lstm_602/while/Identity_1Identity@forward_lstm_602_while_forward_lstm_602_while_maximum_iterations^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_602/while/Identity_1±
!forward_lstm_602/while/Identity_2Identityforward_lstm_602/while/add:z:0^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_602/while/Identity_2ё
!forward_lstm_602/while/Identity_3IdentityKforward_lstm_602/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_602/while/Identity_3 
!forward_lstm_602/while/Identity_4Identity&forward_lstm_602/while/Select:output:0^forward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_602/while/Identity_4ћ
!forward_lstm_602/while/Identity_5Identity(forward_lstm_602/while/Select_1:output:0^forward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_602/while/Identity_5ћ
!forward_lstm_602/while/Identity_6Identity(forward_lstm_602/while/Select_2:output:0^forward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_602/while/Identity_6є
forward_lstm_602/while/NoOpNoOp=^forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp<^forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp>^forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_602/while/NoOp"t
7forward_lstm_602_while_forward_lstm_602_strided_slice_19forward_lstm_602_while_forward_lstm_602_strided_slice_1_0"n
4forward_lstm_602_while_greater_forward_lstm_602_cast6forward_lstm_602_while_greater_forward_lstm_602_cast_0"K
forward_lstm_602_while_identity(forward_lstm_602/while/Identity:output:0"O
!forward_lstm_602_while_identity_1*forward_lstm_602/while/Identity_1:output:0"O
!forward_lstm_602_while_identity_2*forward_lstm_602/while/Identity_2:output:0"O
!forward_lstm_602_while_identity_3*forward_lstm_602/while/Identity_3:output:0"O
!forward_lstm_602_while_identity_4*forward_lstm_602/while/Identity_4:output:0"O
!forward_lstm_602_while_identity_5*forward_lstm_602/while/Identity_5:output:0"O
!forward_lstm_602_while_identity_6*forward_lstm_602/while/Identity_6:output:0"Р
Eforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resourceGforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0"Т
Fforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resourceHforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0"О
Dforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resourceFforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0"м
sforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensoruforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2|
<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp2z
;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp2~
=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
в
Ѕ
4__inference_backward_lstm_602_layer_call_fn_60938786

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
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_609354052
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
4__inference_bidirectional_602_layer_call_fn_60936718
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
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_609356262
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
¶Z
§
%backward_lstm_602_while_body_60937272@
<backward_lstm_602_while_backward_lstm_602_while_loop_counterF
Bbackward_lstm_602_while_backward_lstm_602_while_maximum_iterations'
#backward_lstm_602_while_placeholder)
%backward_lstm_602_while_placeholder_1)
%backward_lstm_602_while_placeholder_2)
%backward_lstm_602_while_placeholder_3?
;backward_lstm_602_while_backward_lstm_602_strided_slice_1_0{
wbackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0Z
Gbackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0:	»\
Ibackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0:	2»W
Hbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0:	»$
 backward_lstm_602_while_identity&
"backward_lstm_602_while_identity_1&
"backward_lstm_602_while_identity_2&
"backward_lstm_602_while_identity_3&
"backward_lstm_602_while_identity_4&
"backward_lstm_602_while_identity_5=
9backward_lstm_602_while_backward_lstm_602_strided_slice_1y
ubackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensorX
Ebackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource:	»Z
Gbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource:	2»U
Fbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource:	»ИҐ=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpҐ<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOpҐ>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpз
Ibackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2K
Ibackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shape»
;backward_lstm_602/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_602_while_placeholderRbackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02=
;backward_lstm_602/while/TensorArrayV2Read/TensorListGetItemЕ
<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOpGbackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02>
<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp•
-backward_lstm_602/while/lstm_cell_1808/MatMulMatMulBbackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem:item:0Dbackward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-backward_lstm_602/while/lstm_cell_1808/MatMulЛ
>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOpIbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02@
>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpО
/backward_lstm_602/while/lstm_cell_1808/MatMul_1MatMul%backward_lstm_602_while_placeholder_2Fbackward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»21
/backward_lstm_602/while/lstm_cell_1808/MatMul_1И
*backward_lstm_602/while/lstm_cell_1808/addAddV27backward_lstm_602/while/lstm_cell_1808/MatMul:product:09backward_lstm_602/while/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2,
*backward_lstm_602/while/lstm_cell_1808/addД
=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOpHbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02?
=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpХ
.backward_lstm_602/while/lstm_cell_1808/BiasAddBiasAdd.backward_lstm_602/while/lstm_cell_1808/add:z:0Ebackward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.backward_lstm_602/while/lstm_cell_1808/BiasAdd≤
6backward_lstm_602/while/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6backward_lstm_602/while/lstm_cell_1808/split/split_dimџ
,backward_lstm_602/while/lstm_cell_1808/splitSplit?backward_lstm_602/while/lstm_cell_1808/split/split_dim:output:07backward_lstm_602/while/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2.
,backward_lstm_602/while/lstm_cell_1808/split‘
.backward_lstm_602/while/lstm_cell_1808/SigmoidSigmoid5backward_lstm_602/while/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€220
.backward_lstm_602/while/lstm_cell_1808/SigmoidЎ
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_1Sigmoid5backward_lstm_602/while/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_1о
*backward_lstm_602/while/lstm_cell_1808/mulMul4backward_lstm_602/while/lstm_cell_1808/Sigmoid_1:y:0%backward_lstm_602_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_602/while/lstm_cell_1808/mulЋ
+backward_lstm_602/while/lstm_cell_1808/ReluRelu5backward_lstm_602/while/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22-
+backward_lstm_602/while/lstm_cell_1808/ReluД
,backward_lstm_602/while/lstm_cell_1808/mul_1Mul2backward_lstm_602/while/lstm_cell_1808/Sigmoid:y:09backward_lstm_602/while/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_602/while/lstm_cell_1808/mul_1щ
,backward_lstm_602/while/lstm_cell_1808/add_1AddV2.backward_lstm_602/while/lstm_cell_1808/mul:z:00backward_lstm_602/while/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_602/while/lstm_cell_1808/add_1Ў
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_2Sigmoid5backward_lstm_602/while/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_2 
-backward_lstm_602/while/lstm_cell_1808/Relu_1Relu0backward_lstm_602/while/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-backward_lstm_602/while/lstm_cell_1808/Relu_1И
,backward_lstm_602/while/lstm_cell_1808/mul_2Mul4backward_lstm_602/while/lstm_cell_1808/Sigmoid_2:y:0;backward_lstm_602/while/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_602/while/lstm_cell_1808/mul_2Љ
<backward_lstm_602/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_602_while_placeholder_1#backward_lstm_602_while_placeholder0backward_lstm_602/while/lstm_cell_1808/mul_2:z:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_602/while/TensorArrayV2Write/TensorListSetItemА
backward_lstm_602/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_602/while/add/y±
backward_lstm_602/while/addAddV2#backward_lstm_602_while_placeholder&backward_lstm_602/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/while/addД
backward_lstm_602/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_602/while/add_1/y–
backward_lstm_602/while/add_1AddV2<backward_lstm_602_while_backward_lstm_602_while_loop_counter(backward_lstm_602/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/while/add_1≥
 backward_lstm_602/while/IdentityIdentity!backward_lstm_602/while/add_1:z:0^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_602/while/IdentityЎ
"backward_lstm_602/while/Identity_1IdentityBbackward_lstm_602_while_backward_lstm_602_while_maximum_iterations^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_602/while/Identity_1µ
"backward_lstm_602/while/Identity_2Identitybackward_lstm_602/while/add:z:0^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_602/while/Identity_2в
"backward_lstm_602/while/Identity_3IdentityLbackward_lstm_602/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_602/while/Identity_3„
"backward_lstm_602/while/Identity_4Identity0backward_lstm_602/while/lstm_cell_1808/mul_2:z:0^backward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_602/while/Identity_4„
"backward_lstm_602/while/Identity_5Identity0backward_lstm_602/while/lstm_cell_1808/add_1:z:0^backward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_602/while/Identity_5Њ
backward_lstm_602/while/NoOpNoOp>^backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp=^backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp?^backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_602/while/NoOp"x
9backward_lstm_602_while_backward_lstm_602_strided_slice_1;backward_lstm_602_while_backward_lstm_602_strided_slice_1_0"M
 backward_lstm_602_while_identity)backward_lstm_602/while/Identity:output:0"Q
"backward_lstm_602_while_identity_1+backward_lstm_602/while/Identity_1:output:0"Q
"backward_lstm_602_while_identity_2+backward_lstm_602/while/Identity_2:output:0"Q
"backward_lstm_602_while_identity_3+backward_lstm_602/while/Identity_3:output:0"Q
"backward_lstm_602_while_identity_4+backward_lstm_602/while/Identity_4:output:0"Q
"backward_lstm_602_while_identity_5+backward_lstm_602/while/Identity_5:output:0"Т
Fbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resourceHbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0"Ф
Gbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resourceIbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0"Р
Ebackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resourceGbackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0"р
ubackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2~
=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp2|
<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp2А
>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
уд
У
#__inference__wrapped_model_60933628

args_0
args_0_1	r
_sequential_602_bidirectional_602_forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource:	»t
asequential_602_bidirectional_602_forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource:	2»o
`sequential_602_bidirectional_602_forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource:	»s
`sequential_602_bidirectional_602_backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resource:	»u
bsequential_602_bidirectional_602_backward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource:	2»p
asequential_602_bidirectional_602_backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource:	»I
7sequential_602_dense_602_matmul_readvariableop_resource:dF
8sequential_602_dense_602_biasadd_readvariableop_resource:
identityИҐXsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpҐWsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpҐYsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpҐ8sequential_602/bidirectional_602/backward_lstm_602/whileҐWsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpҐVsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpҐXsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpҐ7sequential_602/bidirectional_602/forward_lstm_602/whileҐ/sequential_602/dense_602/BiasAdd/ReadVariableOpҐ.sequential_602/dense_602/MatMul/ReadVariableOpў
Fsequential_602/bidirectional_602/forward_lstm_602/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2H
Fsequential_602/bidirectional_602/forward_lstm_602/RaggedToTensor/zerosџ
Fsequential_602/bidirectional_602/forward_lstm_602/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2H
Fsequential_602/bidirectional_602/forward_lstm_602/RaggedToTensor/ConstЭ
Usequential_602/bidirectional_602/forward_lstm_602/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorOsequential_602/bidirectional_602/forward_lstm_602/RaggedToTensor/Const:output:0args_0Osequential_602/bidirectional_602/forward_lstm_602/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2W
Usequential_602/bidirectional_602/forward_lstm_602/RaggedToTensor/RaggedTensorToTensorЖ
\sequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2^
\sequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice/stackК
^sequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2`
^sequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1К
^sequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2`
^sequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2ќ
Vsequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1esequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack:output:0gsequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1:output:0gsequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask2X
Vsequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_sliceК
^sequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2`
^sequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stackЧ
`sequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2b
`sequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1О
`sequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2b
`sequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2Џ
Xsequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1gsequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack:output:0isequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0isequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask2Z
Xsequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice_1Х
Lsequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/subSub_sequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice:output:0asequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2N
Lsequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/subЗ
6sequential_602/bidirectional_602/forward_lstm_602/CastCastPsequential_602/bidirectional_602/forward_lstm_602/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€28
6sequential_602/bidirectional_602/forward_lstm_602/CastА
7sequential_602/bidirectional_602/forward_lstm_602/ShapeShape^sequential_602/bidirectional_602/forward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:29
7sequential_602/bidirectional_602/forward_lstm_602/ShapeЎ
Esequential_602/bidirectional_602/forward_lstm_602/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_602/bidirectional_602/forward_lstm_602/strided_slice/stack№
Gsequential_602/bidirectional_602/forward_lstm_602/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_602/bidirectional_602/forward_lstm_602/strided_slice/stack_1№
Gsequential_602/bidirectional_602/forward_lstm_602/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_602/bidirectional_602/forward_lstm_602/strided_slice/stack_2О
?sequential_602/bidirectional_602/forward_lstm_602/strided_sliceStridedSlice@sequential_602/bidirectional_602/forward_lstm_602/Shape:output:0Nsequential_602/bidirectional_602/forward_lstm_602/strided_slice/stack:output:0Psequential_602/bidirectional_602/forward_lstm_602/strided_slice/stack_1:output:0Psequential_602/bidirectional_602/forward_lstm_602/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?sequential_602/bidirectional_602/forward_lstm_602/strided_sliceј
=sequential_602/bidirectional_602/forward_lstm_602/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22?
=sequential_602/bidirectional_602/forward_lstm_602/zeros/mul/yі
;sequential_602/bidirectional_602/forward_lstm_602/zeros/mulMulHsequential_602/bidirectional_602/forward_lstm_602/strided_slice:output:0Fsequential_602/bidirectional_602/forward_lstm_602/zeros/mul/y:output:0*
T0*
_output_shapes
: 2=
;sequential_602/bidirectional_602/forward_lstm_602/zeros/mul√
>sequential_602/bidirectional_602/forward_lstm_602/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2@
>sequential_602/bidirectional_602/forward_lstm_602/zeros/Less/yѓ
<sequential_602/bidirectional_602/forward_lstm_602/zeros/LessLess?sequential_602/bidirectional_602/forward_lstm_602/zeros/mul:z:0Gsequential_602/bidirectional_602/forward_lstm_602/zeros/Less/y:output:0*
T0*
_output_shapes
: 2>
<sequential_602/bidirectional_602/forward_lstm_602/zeros/Less∆
@sequential_602/bidirectional_602/forward_lstm_602/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22B
@sequential_602/bidirectional_602/forward_lstm_602/zeros/packed/1Ћ
>sequential_602/bidirectional_602/forward_lstm_602/zeros/packedPackHsequential_602/bidirectional_602/forward_lstm_602/strided_slice:output:0Isequential_602/bidirectional_602/forward_lstm_602/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2@
>sequential_602/bidirectional_602/forward_lstm_602/zeros/packed«
=sequential_602/bidirectional_602/forward_lstm_602/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2?
=sequential_602/bidirectional_602/forward_lstm_602/zeros/Constљ
7sequential_602/bidirectional_602/forward_lstm_602/zerosFillGsequential_602/bidirectional_602/forward_lstm_602/zeros/packed:output:0Fsequential_602/bidirectional_602/forward_lstm_602/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€229
7sequential_602/bidirectional_602/forward_lstm_602/zerosƒ
?sequential_602/bidirectional_602/forward_lstm_602/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22A
?sequential_602/bidirectional_602/forward_lstm_602/zeros_1/mul/yЇ
=sequential_602/bidirectional_602/forward_lstm_602/zeros_1/mulMulHsequential_602/bidirectional_602/forward_lstm_602/strided_slice:output:0Hsequential_602/bidirectional_602/forward_lstm_602/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2?
=sequential_602/bidirectional_602/forward_lstm_602/zeros_1/mul«
@sequential_602/bidirectional_602/forward_lstm_602/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2B
@sequential_602/bidirectional_602/forward_lstm_602/zeros_1/Less/yЈ
>sequential_602/bidirectional_602/forward_lstm_602/zeros_1/LessLessAsequential_602/bidirectional_602/forward_lstm_602/zeros_1/mul:z:0Isequential_602/bidirectional_602/forward_lstm_602/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2@
>sequential_602/bidirectional_602/forward_lstm_602/zeros_1/Less 
Bsequential_602/bidirectional_602/forward_lstm_602/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22D
Bsequential_602/bidirectional_602/forward_lstm_602/zeros_1/packed/1—
@sequential_602/bidirectional_602/forward_lstm_602/zeros_1/packedPackHsequential_602/bidirectional_602/forward_lstm_602/strided_slice:output:0Ksequential_602/bidirectional_602/forward_lstm_602/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2B
@sequential_602/bidirectional_602/forward_lstm_602/zeros_1/packedЋ
?sequential_602/bidirectional_602/forward_lstm_602/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2A
?sequential_602/bidirectional_602/forward_lstm_602/zeros_1/Const≈
9sequential_602/bidirectional_602/forward_lstm_602/zeros_1FillIsequential_602/bidirectional_602/forward_lstm_602/zeros_1/packed:output:0Hsequential_602/bidirectional_602/forward_lstm_602/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22;
9sequential_602/bidirectional_602/forward_lstm_602/zeros_1ў
@sequential_602/bidirectional_602/forward_lstm_602/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2B
@sequential_602/bidirectional_602/forward_lstm_602/transpose/permс
;sequential_602/bidirectional_602/forward_lstm_602/transpose	Transpose^sequential_602/bidirectional_602/forward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0Isequential_602/bidirectional_602/forward_lstm_602/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2=
;sequential_602/bidirectional_602/forward_lstm_602/transposeе
9sequential_602/bidirectional_602/forward_lstm_602/Shape_1Shape?sequential_602/bidirectional_602/forward_lstm_602/transpose:y:0*
T0*
_output_shapes
:2;
9sequential_602/bidirectional_602/forward_lstm_602/Shape_1№
Gsequential_602/bidirectional_602/forward_lstm_602/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2I
Gsequential_602/bidirectional_602/forward_lstm_602/strided_slice_1/stackа
Isequential_602/bidirectional_602/forward_lstm_602/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_602/bidirectional_602/forward_lstm_602/strided_slice_1/stack_1а
Isequential_602/bidirectional_602/forward_lstm_602/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_602/bidirectional_602/forward_lstm_602/strided_slice_1/stack_2Ъ
Asequential_602/bidirectional_602/forward_lstm_602/strided_slice_1StridedSliceBsequential_602/bidirectional_602/forward_lstm_602/Shape_1:output:0Psequential_602/bidirectional_602/forward_lstm_602/strided_slice_1/stack:output:0Rsequential_602/bidirectional_602/forward_lstm_602/strided_slice_1/stack_1:output:0Rsequential_602/bidirectional_602/forward_lstm_602/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2C
Asequential_602/bidirectional_602/forward_lstm_602/strided_slice_1й
Msequential_602/bidirectional_602/forward_lstm_602/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2O
Msequential_602/bidirectional_602/forward_lstm_602/TensorArrayV2/element_shapeъ
?sequential_602/bidirectional_602/forward_lstm_602/TensorArrayV2TensorListReserveVsequential_602/bidirectional_602/forward_lstm_602/TensorArrayV2/element_shape:output:0Jsequential_602/bidirectional_602/forward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?sequential_602/bidirectional_602/forward_lstm_602/TensorArrayV2£
gsequential_602/bidirectional_602/forward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2i
gsequential_602/bidirectional_602/forward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeј
Ysequential_602/bidirectional_602/forward_lstm_602/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor?sequential_602/bidirectional_602/forward_lstm_602/transpose:y:0psequential_602/bidirectional_602/forward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02[
Ysequential_602/bidirectional_602/forward_lstm_602/TensorArrayUnstack/TensorListFromTensor№
Gsequential_602/bidirectional_602/forward_lstm_602/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2I
Gsequential_602/bidirectional_602/forward_lstm_602/strided_slice_2/stackа
Isequential_602/bidirectional_602/forward_lstm_602/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_602/bidirectional_602/forward_lstm_602/strided_slice_2/stack_1а
Isequential_602/bidirectional_602/forward_lstm_602/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_602/bidirectional_602/forward_lstm_602/strided_slice_2/stack_2®
Asequential_602/bidirectional_602/forward_lstm_602/strided_slice_2StridedSlice?sequential_602/bidirectional_602/forward_lstm_602/transpose:y:0Psequential_602/bidirectional_602/forward_lstm_602/strided_slice_2/stack:output:0Rsequential_602/bidirectional_602/forward_lstm_602/strided_slice_2/stack_1:output:0Rsequential_602/bidirectional_602/forward_lstm_602/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2C
Asequential_602/bidirectional_602/forward_lstm_602/strided_slice_2—
Vsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOp_sequential_602_bidirectional_602_forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02X
Vsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpы
Gsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMulMatMulJsequential_602/bidirectional_602/forward_lstm_602/strided_slice_2:output:0^sequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2I
Gsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMul„
Xsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOpasequential_602_bidirectional_602_forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02Z
Xsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpч
Isequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMul_1MatMul@sequential_602/bidirectional_602/forward_lstm_602/zeros:output:0`sequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2K
Isequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMul_1р
Dsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/addAddV2Qsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMul:product:0Ssequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2F
Dsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/add–
Wsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOp`sequential_602_bidirectional_602_forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02Y
Wsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpэ
Hsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/BiasAddBiasAddHsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/add:z:0_sequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2J
Hsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/BiasAddж
Psequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2R
Psequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/split/split_dim√
Fsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/splitSplitYsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/split/split_dim:output:0Qsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2H
Fsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/splitҐ
Hsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/SigmoidSigmoidOsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22J
Hsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/Sigmoid¶
Jsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/Sigmoid_1SigmoidOsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22L
Jsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/Sigmoid_1ў
Dsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/mulMulNsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/Sigmoid_1:y:0Bsequential_602/bidirectional_602/forward_lstm_602/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22F
Dsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/mulЩ
Esequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/ReluReluOsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22G
Esequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/Reluм
Fsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/mul_1MulLsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/Sigmoid:y:0Ssequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22H
Fsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/mul_1б
Fsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/add_1AddV2Hsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/mul:z:0Jsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22H
Fsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/add_1¶
Jsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/Sigmoid_2SigmoidOsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22L
Jsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/Sigmoid_2Ш
Gsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/Relu_1ReluJsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22I
Gsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/Relu_1р
Fsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/mul_2MulNsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/Sigmoid_2:y:0Usequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22H
Fsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/mul_2у
Osequential_602/bidirectional_602/forward_lstm_602/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2Q
Osequential_602/bidirectional_602/forward_lstm_602/TensorArrayV2_1/element_shapeА
Asequential_602/bidirectional_602/forward_lstm_602/TensorArrayV2_1TensorListReserveXsequential_602/bidirectional_602/forward_lstm_602/TensorArrayV2_1/element_shape:output:0Jsequential_602/bidirectional_602/forward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02C
Asequential_602/bidirectional_602/forward_lstm_602/TensorArrayV2_1≤
6sequential_602/bidirectional_602/forward_lstm_602/timeConst*
_output_shapes
: *
dtype0*
value	B : 28
6sequential_602/bidirectional_602/forward_lstm_602/timeЗ
<sequential_602/bidirectional_602/forward_lstm_602/zeros_like	ZerosLikeJsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22>
<sequential_602/bidirectional_602/forward_lstm_602/zeros_likeг
Jsequential_602/bidirectional_602/forward_lstm_602/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2L
Jsequential_602/bidirectional_602/forward_lstm_602/while/maximum_iterationsќ
Dsequential_602/bidirectional_602/forward_lstm_602/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dsequential_602/bidirectional_602/forward_lstm_602/while/loop_counter«
7sequential_602/bidirectional_602/forward_lstm_602/whileWhileMsequential_602/bidirectional_602/forward_lstm_602/while/loop_counter:output:0Ssequential_602/bidirectional_602/forward_lstm_602/while/maximum_iterations:output:0?sequential_602/bidirectional_602/forward_lstm_602/time:output:0Jsequential_602/bidirectional_602/forward_lstm_602/TensorArrayV2_1:handle:0@sequential_602/bidirectional_602/forward_lstm_602/zeros_like:y:0@sequential_602/bidirectional_602/forward_lstm_602/zeros:output:0Bsequential_602/bidirectional_602/forward_lstm_602/zeros_1:output:0Jsequential_602/bidirectional_602/forward_lstm_602/strided_slice_1:output:0isequential_602/bidirectional_602/forward_lstm_602/TensorArrayUnstack/TensorListFromTensor:output_handle:0:sequential_602/bidirectional_602/forward_lstm_602/Cast:y:0_sequential_602_bidirectional_602_forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resourceasequential_602_bidirectional_602_forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource`sequential_602_bidirectional_602_forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource*
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
Esequential_602_bidirectional_602_forward_lstm_602_while_body_60933345*Q
condIRG
Esequential_602_bidirectional_602_forward_lstm_602_while_cond_60933344*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 29
7sequential_602/bidirectional_602/forward_lstm_602/whileЩ
bsequential_602/bidirectional_602/forward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2d
bsequential_602/bidirectional_602/forward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeє
Tsequential_602/bidirectional_602/forward_lstm_602/TensorArrayV2Stack/TensorListStackTensorListStack@sequential_602/bidirectional_602/forward_lstm_602/while:output:3ksequential_602/bidirectional_602/forward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02V
Tsequential_602/bidirectional_602/forward_lstm_602/TensorArrayV2Stack/TensorListStackе
Gsequential_602/bidirectional_602/forward_lstm_602/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2I
Gsequential_602/bidirectional_602/forward_lstm_602/strided_slice_3/stackа
Isequential_602/bidirectional_602/forward_lstm_602/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2K
Isequential_602/bidirectional_602/forward_lstm_602/strided_slice_3/stack_1а
Isequential_602/bidirectional_602/forward_lstm_602/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_602/bidirectional_602/forward_lstm_602/strided_slice_3/stack_2∆
Asequential_602/bidirectional_602/forward_lstm_602/strided_slice_3StridedSlice]sequential_602/bidirectional_602/forward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0Psequential_602/bidirectional_602/forward_lstm_602/strided_slice_3/stack:output:0Rsequential_602/bidirectional_602/forward_lstm_602/strided_slice_3/stack_1:output:0Rsequential_602/bidirectional_602/forward_lstm_602/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2C
Asequential_602/bidirectional_602/forward_lstm_602/strided_slice_3Ё
Bsequential_602/bidirectional_602/forward_lstm_602/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2D
Bsequential_602/bidirectional_602/forward_lstm_602/transpose_1/permц
=sequential_602/bidirectional_602/forward_lstm_602/transpose_1	Transpose]sequential_602/bidirectional_602/forward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0Ksequential_602/bidirectional_602/forward_lstm_602/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22?
=sequential_602/bidirectional_602/forward_lstm_602/transpose_1 
9sequential_602/bidirectional_602/forward_lstm_602/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2;
9sequential_602/bidirectional_602/forward_lstm_602/runtimeџ
Gsequential_602/bidirectional_602/backward_lstm_602/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2I
Gsequential_602/bidirectional_602/backward_lstm_602/RaggedToTensor/zerosЁ
Gsequential_602/bidirectional_602/backward_lstm_602/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2I
Gsequential_602/bidirectional_602/backward_lstm_602/RaggedToTensor/Const°
Vsequential_602/bidirectional_602/backward_lstm_602/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorPsequential_602/bidirectional_602/backward_lstm_602/RaggedToTensor/Const:output:0args_0Psequential_602/bidirectional_602/backward_lstm_602/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2X
Vsequential_602/bidirectional_602/backward_lstm_602/RaggedToTensor/RaggedTensorToTensorИ
]sequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2_
]sequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice/stackМ
_sequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2a
_sequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1М
_sequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2a
_sequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2”
Wsequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1fsequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack:output:0hsequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1:output:0hsequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask2Y
Wsequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_sliceМ
_sequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2a
_sequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stackЩ
asequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2c
asequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1Р
asequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2c
asequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2я
Ysequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1hsequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack:output:0jsequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0jsequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask2[
Ysequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice_1Щ
Msequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/subSub`sequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice:output:0bsequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2O
Msequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/subК
7sequential_602/bidirectional_602/backward_lstm_602/CastCastQsequential_602/bidirectional_602/backward_lstm_602/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€29
7sequential_602/bidirectional_602/backward_lstm_602/CastГ
8sequential_602/bidirectional_602/backward_lstm_602/ShapeShape_sequential_602/bidirectional_602/backward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2:
8sequential_602/bidirectional_602/backward_lstm_602/ShapeЏ
Fsequential_602/bidirectional_602/backward_lstm_602/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fsequential_602/bidirectional_602/backward_lstm_602/strided_slice/stackё
Hsequential_602/bidirectional_602/backward_lstm_602/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_602/bidirectional_602/backward_lstm_602/strided_slice/stack_1ё
Hsequential_602/bidirectional_602/backward_lstm_602/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_602/bidirectional_602/backward_lstm_602/strided_slice/stack_2Ф
@sequential_602/bidirectional_602/backward_lstm_602/strided_sliceStridedSliceAsequential_602/bidirectional_602/backward_lstm_602/Shape:output:0Osequential_602/bidirectional_602/backward_lstm_602/strided_slice/stack:output:0Qsequential_602/bidirectional_602/backward_lstm_602/strided_slice/stack_1:output:0Qsequential_602/bidirectional_602/backward_lstm_602/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2B
@sequential_602/bidirectional_602/backward_lstm_602/strided_slice¬
>sequential_602/bidirectional_602/backward_lstm_602/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22@
>sequential_602/bidirectional_602/backward_lstm_602/zeros/mul/yЄ
<sequential_602/bidirectional_602/backward_lstm_602/zeros/mulMulIsequential_602/bidirectional_602/backward_lstm_602/strided_slice:output:0Gsequential_602/bidirectional_602/backward_lstm_602/zeros/mul/y:output:0*
T0*
_output_shapes
: 2>
<sequential_602/bidirectional_602/backward_lstm_602/zeros/mul≈
?sequential_602/bidirectional_602/backward_lstm_602/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2A
?sequential_602/bidirectional_602/backward_lstm_602/zeros/Less/y≥
=sequential_602/bidirectional_602/backward_lstm_602/zeros/LessLess@sequential_602/bidirectional_602/backward_lstm_602/zeros/mul:z:0Hsequential_602/bidirectional_602/backward_lstm_602/zeros/Less/y:output:0*
T0*
_output_shapes
: 2?
=sequential_602/bidirectional_602/backward_lstm_602/zeros/Less»
Asequential_602/bidirectional_602/backward_lstm_602/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22C
Asequential_602/bidirectional_602/backward_lstm_602/zeros/packed/1ѕ
?sequential_602/bidirectional_602/backward_lstm_602/zeros/packedPackIsequential_602/bidirectional_602/backward_lstm_602/strided_slice:output:0Jsequential_602/bidirectional_602/backward_lstm_602/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2A
?sequential_602/bidirectional_602/backward_lstm_602/zeros/packed…
>sequential_602/bidirectional_602/backward_lstm_602/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2@
>sequential_602/bidirectional_602/backward_lstm_602/zeros/ConstЅ
8sequential_602/bidirectional_602/backward_lstm_602/zerosFillHsequential_602/bidirectional_602/backward_lstm_602/zeros/packed:output:0Gsequential_602/bidirectional_602/backward_lstm_602/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22:
8sequential_602/bidirectional_602/backward_lstm_602/zeros∆
@sequential_602/bidirectional_602/backward_lstm_602/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22B
@sequential_602/bidirectional_602/backward_lstm_602/zeros_1/mul/yЊ
>sequential_602/bidirectional_602/backward_lstm_602/zeros_1/mulMulIsequential_602/bidirectional_602/backward_lstm_602/strided_slice:output:0Isequential_602/bidirectional_602/backward_lstm_602/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2@
>sequential_602/bidirectional_602/backward_lstm_602/zeros_1/mul…
Asequential_602/bidirectional_602/backward_lstm_602/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2C
Asequential_602/bidirectional_602/backward_lstm_602/zeros_1/Less/yї
?sequential_602/bidirectional_602/backward_lstm_602/zeros_1/LessLessBsequential_602/bidirectional_602/backward_lstm_602/zeros_1/mul:z:0Jsequential_602/bidirectional_602/backward_lstm_602/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2A
?sequential_602/bidirectional_602/backward_lstm_602/zeros_1/Lessћ
Csequential_602/bidirectional_602/backward_lstm_602/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22E
Csequential_602/bidirectional_602/backward_lstm_602/zeros_1/packed/1’
Asequential_602/bidirectional_602/backward_lstm_602/zeros_1/packedPackIsequential_602/bidirectional_602/backward_lstm_602/strided_slice:output:0Lsequential_602/bidirectional_602/backward_lstm_602/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2C
Asequential_602/bidirectional_602/backward_lstm_602/zeros_1/packedЌ
@sequential_602/bidirectional_602/backward_lstm_602/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2B
@sequential_602/bidirectional_602/backward_lstm_602/zeros_1/Const…
:sequential_602/bidirectional_602/backward_lstm_602/zeros_1FillJsequential_602/bidirectional_602/backward_lstm_602/zeros_1/packed:output:0Isequential_602/bidirectional_602/backward_lstm_602/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22<
:sequential_602/bidirectional_602/backward_lstm_602/zeros_1џ
Asequential_602/bidirectional_602/backward_lstm_602/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2C
Asequential_602/bidirectional_602/backward_lstm_602/transpose/permх
<sequential_602/bidirectional_602/backward_lstm_602/transpose	Transpose_sequential_602/bidirectional_602/backward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0Jsequential_602/bidirectional_602/backward_lstm_602/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2>
<sequential_602/bidirectional_602/backward_lstm_602/transposeи
:sequential_602/bidirectional_602/backward_lstm_602/Shape_1Shape@sequential_602/bidirectional_602/backward_lstm_602/transpose:y:0*
T0*
_output_shapes
:2<
:sequential_602/bidirectional_602/backward_lstm_602/Shape_1ё
Hsequential_602/bidirectional_602/backward_lstm_602/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_602/bidirectional_602/backward_lstm_602/strided_slice_1/stackв
Jsequential_602/bidirectional_602/backward_lstm_602/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_602/bidirectional_602/backward_lstm_602/strided_slice_1/stack_1в
Jsequential_602/bidirectional_602/backward_lstm_602/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_602/bidirectional_602/backward_lstm_602/strided_slice_1/stack_2†
Bsequential_602/bidirectional_602/backward_lstm_602/strided_slice_1StridedSliceCsequential_602/bidirectional_602/backward_lstm_602/Shape_1:output:0Qsequential_602/bidirectional_602/backward_lstm_602/strided_slice_1/stack:output:0Ssequential_602/bidirectional_602/backward_lstm_602/strided_slice_1/stack_1:output:0Ssequential_602/bidirectional_602/backward_lstm_602/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2D
Bsequential_602/bidirectional_602/backward_lstm_602/strided_slice_1л
Nsequential_602/bidirectional_602/backward_lstm_602/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2P
Nsequential_602/bidirectional_602/backward_lstm_602/TensorArrayV2/element_shapeю
@sequential_602/bidirectional_602/backward_lstm_602/TensorArrayV2TensorListReserveWsequential_602/bidirectional_602/backward_lstm_602/TensorArrayV2/element_shape:output:0Ksequential_602/bidirectional_602/backward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02B
@sequential_602/bidirectional_602/backward_lstm_602/TensorArrayV2–
Asequential_602/bidirectional_602/backward_lstm_602/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential_602/bidirectional_602/backward_lstm_602/ReverseV2/axis÷
<sequential_602/bidirectional_602/backward_lstm_602/ReverseV2	ReverseV2@sequential_602/bidirectional_602/backward_lstm_602/transpose:y:0Jsequential_602/bidirectional_602/backward_lstm_602/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2>
<sequential_602/bidirectional_602/backward_lstm_602/ReverseV2•
hsequential_602/bidirectional_602/backward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2j
hsequential_602/bidirectional_602/backward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape…
Zsequential_602/bidirectional_602/backward_lstm_602/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorEsequential_602/bidirectional_602/backward_lstm_602/ReverseV2:output:0qsequential_602/bidirectional_602/backward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02\
Zsequential_602/bidirectional_602/backward_lstm_602/TensorArrayUnstack/TensorListFromTensorё
Hsequential_602/bidirectional_602/backward_lstm_602/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_602/bidirectional_602/backward_lstm_602/strided_slice_2/stackв
Jsequential_602/bidirectional_602/backward_lstm_602/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_602/bidirectional_602/backward_lstm_602/strided_slice_2/stack_1в
Jsequential_602/bidirectional_602/backward_lstm_602/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_602/bidirectional_602/backward_lstm_602/strided_slice_2/stack_2Ѓ
Bsequential_602/bidirectional_602/backward_lstm_602/strided_slice_2StridedSlice@sequential_602/bidirectional_602/backward_lstm_602/transpose:y:0Qsequential_602/bidirectional_602/backward_lstm_602/strided_slice_2/stack:output:0Ssequential_602/bidirectional_602/backward_lstm_602/strided_slice_2/stack_1:output:0Ssequential_602/bidirectional_602/backward_lstm_602/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2D
Bsequential_602/bidirectional_602/backward_lstm_602/strided_slice_2‘
Wsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOp`sequential_602_bidirectional_602_backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02Y
Wsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp€
Hsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMulMatMulKsequential_602/bidirectional_602/backward_lstm_602/strided_slice_2:output:0_sequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2J
Hsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMulЏ
Ysequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOpbsequential_602_bidirectional_602_backward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02[
Ysequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpы
Jsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMul_1MatMulAsequential_602/bidirectional_602/backward_lstm_602/zeros:output:0asequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2L
Jsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMul_1ф
Esequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/addAddV2Rsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMul:product:0Tsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2G
Esequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/add”
Xsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOpasequential_602_bidirectional_602_backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02Z
Xsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpБ
Isequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/BiasAddBiasAddIsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/add:z:0`sequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2K
Isequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/BiasAddи
Qsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2S
Qsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/split/split_dim«
Gsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/splitSplitZsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/split/split_dim:output:0Rsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2I
Gsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/split•
Isequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/SigmoidSigmoidPsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22K
Isequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/Sigmoid©
Ksequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/Sigmoid_1SigmoidPsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22M
Ksequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/Sigmoid_1Ё
Esequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/mulMulOsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/Sigmoid_1:y:0Csequential_602/bidirectional_602/backward_lstm_602/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22G
Esequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/mulЬ
Fsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/ReluReluPsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22H
Fsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/Reluр
Gsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/mul_1MulMsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/Sigmoid:y:0Tsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22I
Gsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/mul_1е
Gsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/add_1AddV2Isequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/mul:z:0Ksequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22I
Gsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/add_1©
Ksequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/Sigmoid_2SigmoidPsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22M
Ksequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/Sigmoid_2Ы
Hsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/Relu_1ReluKsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22J
Hsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/Relu_1ф
Gsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/mul_2MulOsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/Sigmoid_2:y:0Vsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22I
Gsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/mul_2х
Psequential_602/bidirectional_602/backward_lstm_602/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2R
Psequential_602/bidirectional_602/backward_lstm_602/TensorArrayV2_1/element_shapeД
Bsequential_602/bidirectional_602/backward_lstm_602/TensorArrayV2_1TensorListReserveYsequential_602/bidirectional_602/backward_lstm_602/TensorArrayV2_1/element_shape:output:0Ksequential_602/bidirectional_602/backward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02D
Bsequential_602/bidirectional_602/backward_lstm_602/TensorArrayV2_1і
7sequential_602/bidirectional_602/backward_lstm_602/timeConst*
_output_shapes
: *
dtype0*
value	B : 29
7sequential_602/bidirectional_602/backward_lstm_602/time÷
Hsequential_602/bidirectional_602/backward_lstm_602/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hsequential_602/bidirectional_602/backward_lstm_602/Max/reduction_indices®
6sequential_602/bidirectional_602/backward_lstm_602/MaxMax;sequential_602/bidirectional_602/backward_lstm_602/Cast:y:0Qsequential_602/bidirectional_602/backward_lstm_602/Max/reduction_indices:output:0*
T0*
_output_shapes
: 28
6sequential_602/bidirectional_602/backward_lstm_602/Maxґ
8sequential_602/bidirectional_602/backward_lstm_602/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_602/bidirectional_602/backward_lstm_602/sub/yЬ
6sequential_602/bidirectional_602/backward_lstm_602/subSub?sequential_602/bidirectional_602/backward_lstm_602/Max:output:0Asequential_602/bidirectional_602/backward_lstm_602/sub/y:output:0*
T0*
_output_shapes
: 28
6sequential_602/bidirectional_602/backward_lstm_602/subҐ
8sequential_602/bidirectional_602/backward_lstm_602/Sub_1Sub:sequential_602/bidirectional_602/backward_lstm_602/sub:z:0;sequential_602/bidirectional_602/backward_lstm_602/Cast:y:0*
T0*#
_output_shapes
:€€€€€€€€€2:
8sequential_602/bidirectional_602/backward_lstm_602/Sub_1К
=sequential_602/bidirectional_602/backward_lstm_602/zeros_like	ZerosLikeKsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22?
=sequential_602/bidirectional_602/backward_lstm_602/zeros_likeе
Ksequential_602/bidirectional_602/backward_lstm_602/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2M
Ksequential_602/bidirectional_602/backward_lstm_602/while/maximum_iterations–
Esequential_602/bidirectional_602/backward_lstm_602/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2G
Esequential_602/bidirectional_602/backward_lstm_602/while/loop_counterў
8sequential_602/bidirectional_602/backward_lstm_602/whileWhileNsequential_602/bidirectional_602/backward_lstm_602/while/loop_counter:output:0Tsequential_602/bidirectional_602/backward_lstm_602/while/maximum_iterations:output:0@sequential_602/bidirectional_602/backward_lstm_602/time:output:0Ksequential_602/bidirectional_602/backward_lstm_602/TensorArrayV2_1:handle:0Asequential_602/bidirectional_602/backward_lstm_602/zeros_like:y:0Asequential_602/bidirectional_602/backward_lstm_602/zeros:output:0Csequential_602/bidirectional_602/backward_lstm_602/zeros_1:output:0Ksequential_602/bidirectional_602/backward_lstm_602/strided_slice_1:output:0jsequential_602/bidirectional_602/backward_lstm_602/TensorArrayUnstack/TensorListFromTensor:output_handle:0<sequential_602/bidirectional_602/backward_lstm_602/Sub_1:z:0`sequential_602_bidirectional_602_backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resourcebsequential_602_bidirectional_602_backward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resourceasequential_602_bidirectional_602_backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource*
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
Fsequential_602_bidirectional_602_backward_lstm_602_while_body_60933524*R
condJRH
Fsequential_602_bidirectional_602_backward_lstm_602_while_cond_60933523*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2:
8sequential_602/bidirectional_602/backward_lstm_602/whileЫ
csequential_602/bidirectional_602/backward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2e
csequential_602/bidirectional_602/backward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeљ
Usequential_602/bidirectional_602/backward_lstm_602/TensorArrayV2Stack/TensorListStackTensorListStackAsequential_602/bidirectional_602/backward_lstm_602/while:output:3lsequential_602/bidirectional_602/backward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02W
Usequential_602/bidirectional_602/backward_lstm_602/TensorArrayV2Stack/TensorListStackз
Hsequential_602/bidirectional_602/backward_lstm_602/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2J
Hsequential_602/bidirectional_602/backward_lstm_602/strided_slice_3/stackв
Jsequential_602/bidirectional_602/backward_lstm_602/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jsequential_602/bidirectional_602/backward_lstm_602/strided_slice_3/stack_1в
Jsequential_602/bidirectional_602/backward_lstm_602/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_602/bidirectional_602/backward_lstm_602/strided_slice_3/stack_2ћ
Bsequential_602/bidirectional_602/backward_lstm_602/strided_slice_3StridedSlice^sequential_602/bidirectional_602/backward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0Qsequential_602/bidirectional_602/backward_lstm_602/strided_slice_3/stack:output:0Ssequential_602/bidirectional_602/backward_lstm_602/strided_slice_3/stack_1:output:0Ssequential_602/bidirectional_602/backward_lstm_602/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2D
Bsequential_602/bidirectional_602/backward_lstm_602/strided_slice_3я
Csequential_602/bidirectional_602/backward_lstm_602/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2E
Csequential_602/bidirectional_602/backward_lstm_602/transpose_1/permъ
>sequential_602/bidirectional_602/backward_lstm_602/transpose_1	Transpose^sequential_602/bidirectional_602/backward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0Lsequential_602/bidirectional_602/backward_lstm_602/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22@
>sequential_602/bidirectional_602/backward_lstm_602/transpose_1ћ
:sequential_602/bidirectional_602/backward_lstm_602/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2<
:sequential_602/bidirectional_602/backward_lstm_602/runtimeЮ
,sequential_602/bidirectional_602/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_602/bidirectional_602/concat/axisй
'sequential_602/bidirectional_602/concatConcatV2Jsequential_602/bidirectional_602/forward_lstm_602/strided_slice_3:output:0Ksequential_602/bidirectional_602/backward_lstm_602/strided_slice_3:output:05sequential_602/bidirectional_602/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€d2)
'sequential_602/bidirectional_602/concatЎ
.sequential_602/dense_602/MatMul/ReadVariableOpReadVariableOp7sequential_602_dense_602_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_602/dense_602/MatMul/ReadVariableOpи
sequential_602/dense_602/MatMulMatMul0sequential_602/bidirectional_602/concat:output:06sequential_602/dense_602/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2!
sequential_602/dense_602/MatMul„
/sequential_602/dense_602/BiasAdd/ReadVariableOpReadVariableOp8sequential_602_dense_602_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_602/dense_602/BiasAdd/ReadVariableOpе
 sequential_602/dense_602/BiasAddBiasAdd)sequential_602/dense_602/MatMul:product:07sequential_602/dense_602/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2"
 sequential_602/dense_602/BiasAddђ
 sequential_602/dense_602/SigmoidSigmoid)sequential_602/dense_602/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2"
 sequential_602/dense_602/Sigmoid
IdentityIdentity$sequential_602/dense_602/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity≈
NoOpNoOpY^sequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpX^sequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpZ^sequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp9^sequential_602/bidirectional_602/backward_lstm_602/whileX^sequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpW^sequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpY^sequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp8^sequential_602/bidirectional_602/forward_lstm_602/while0^sequential_602/dense_602/BiasAdd/ReadVariableOp/^sequential_602/dense_602/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€:€€€€€€€€€: : : : : : : : 2і
Xsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpXsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp2≤
Wsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpWsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp2ґ
Ysequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpYsequential_602/bidirectional_602/backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp2t
8sequential_602/bidirectional_602/backward_lstm_602/while8sequential_602/bidirectional_602/backward_lstm_602/while2≤
Wsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpWsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp2∞
Vsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpVsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp2і
Xsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpXsequential_602/bidirectional_602/forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp2r
7sequential_602/bidirectional_602/forward_lstm_602/while7sequential_602/bidirectional_602/forward_lstm_602/while2b
/sequential_602/dense_602/BiasAdd/ReadVariableOp/sequential_602/dense_602/BiasAdd/ReadVariableOp2`
.sequential_602/dense_602/MatMul/ReadVariableOp.sequential_602/dense_602/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameargs_0:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameargs_0
Іg
н
%backward_lstm_602_while_body_60935967@
<backward_lstm_602_while_backward_lstm_602_while_loop_counterF
Bbackward_lstm_602_while_backward_lstm_602_while_maximum_iterations'
#backward_lstm_602_while_placeholder)
%backward_lstm_602_while_placeholder_1)
%backward_lstm_602_while_placeholder_2)
%backward_lstm_602_while_placeholder_3)
%backward_lstm_602_while_placeholder_4?
;backward_lstm_602_while_backward_lstm_602_strided_slice_1_0{
wbackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_602_while_less_backward_lstm_602_sub_1_0Z
Gbackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0:	»\
Ibackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0:	2»W
Hbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0:	»$
 backward_lstm_602_while_identity&
"backward_lstm_602_while_identity_1&
"backward_lstm_602_while_identity_2&
"backward_lstm_602_while_identity_3&
"backward_lstm_602_while_identity_4&
"backward_lstm_602_while_identity_5&
"backward_lstm_602_while_identity_6=
9backward_lstm_602_while_backward_lstm_602_strided_slice_1y
ubackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_602_while_less_backward_lstm_602_sub_1X
Ebackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource:	»Z
Gbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource:	2»U
Fbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource:	»ИҐ=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpҐ<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOpҐ>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpз
Ibackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2K
Ibackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeњ
;backward_lstm_602/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_602_while_placeholderRbackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02=
;backward_lstm_602/while/TensorArrayV2Read/TensorListGetItemѕ
backward_lstm_602/while/LessLess6backward_lstm_602_while_less_backward_lstm_602_sub_1_0#backward_lstm_602_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_602/while/LessЕ
<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOpGbackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02>
<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp•
-backward_lstm_602/while/lstm_cell_1808/MatMulMatMulBbackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem:item:0Dbackward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-backward_lstm_602/while/lstm_cell_1808/MatMulЛ
>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOpIbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02@
>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpО
/backward_lstm_602/while/lstm_cell_1808/MatMul_1MatMul%backward_lstm_602_while_placeholder_3Fbackward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»21
/backward_lstm_602/while/lstm_cell_1808/MatMul_1И
*backward_lstm_602/while/lstm_cell_1808/addAddV27backward_lstm_602/while/lstm_cell_1808/MatMul:product:09backward_lstm_602/while/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2,
*backward_lstm_602/while/lstm_cell_1808/addД
=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOpHbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02?
=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpХ
.backward_lstm_602/while/lstm_cell_1808/BiasAddBiasAdd.backward_lstm_602/while/lstm_cell_1808/add:z:0Ebackward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.backward_lstm_602/while/lstm_cell_1808/BiasAdd≤
6backward_lstm_602/while/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6backward_lstm_602/while/lstm_cell_1808/split/split_dimџ
,backward_lstm_602/while/lstm_cell_1808/splitSplit?backward_lstm_602/while/lstm_cell_1808/split/split_dim:output:07backward_lstm_602/while/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2.
,backward_lstm_602/while/lstm_cell_1808/split‘
.backward_lstm_602/while/lstm_cell_1808/SigmoidSigmoid5backward_lstm_602/while/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€220
.backward_lstm_602/while/lstm_cell_1808/SigmoidЎ
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_1Sigmoid5backward_lstm_602/while/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_1о
*backward_lstm_602/while/lstm_cell_1808/mulMul4backward_lstm_602/while/lstm_cell_1808/Sigmoid_1:y:0%backward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_602/while/lstm_cell_1808/mulЋ
+backward_lstm_602/while/lstm_cell_1808/ReluRelu5backward_lstm_602/while/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22-
+backward_lstm_602/while/lstm_cell_1808/ReluД
,backward_lstm_602/while/lstm_cell_1808/mul_1Mul2backward_lstm_602/while/lstm_cell_1808/Sigmoid:y:09backward_lstm_602/while/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_602/while/lstm_cell_1808/mul_1щ
,backward_lstm_602/while/lstm_cell_1808/add_1AddV2.backward_lstm_602/while/lstm_cell_1808/mul:z:00backward_lstm_602/while/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_602/while/lstm_cell_1808/add_1Ў
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_2Sigmoid5backward_lstm_602/while/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_2 
-backward_lstm_602/while/lstm_cell_1808/Relu_1Relu0backward_lstm_602/while/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-backward_lstm_602/while/lstm_cell_1808/Relu_1И
,backward_lstm_602/while/lstm_cell_1808/mul_2Mul4backward_lstm_602/while/lstm_cell_1808/Sigmoid_2:y:0;backward_lstm_602/while/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_602/while/lstm_cell_1808/mul_2ч
backward_lstm_602/while/SelectSelect backward_lstm_602/while/Less:z:00backward_lstm_602/while/lstm_cell_1808/mul_2:z:0%backward_lstm_602_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22 
backward_lstm_602/while/Selectы
 backward_lstm_602/while/Select_1Select backward_lstm_602/while/Less:z:00backward_lstm_602/while/lstm_cell_1808/mul_2:z:0%backward_lstm_602_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_602/while/Select_1ы
 backward_lstm_602/while/Select_2Select backward_lstm_602/while/Less:z:00backward_lstm_602/while/lstm_cell_1808/add_1:z:0%backward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_602/while/Select_2≥
<backward_lstm_602/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_602_while_placeholder_1#backward_lstm_602_while_placeholder'backward_lstm_602/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_602/while/TensorArrayV2Write/TensorListSetItemА
backward_lstm_602/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_602/while/add/y±
backward_lstm_602/while/addAddV2#backward_lstm_602_while_placeholder&backward_lstm_602/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/while/addД
backward_lstm_602/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_602/while/add_1/y–
backward_lstm_602/while/add_1AddV2<backward_lstm_602_while_backward_lstm_602_while_loop_counter(backward_lstm_602/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/while/add_1≥
 backward_lstm_602/while/IdentityIdentity!backward_lstm_602/while/add_1:z:0^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_602/while/IdentityЎ
"backward_lstm_602/while/Identity_1IdentityBbackward_lstm_602_while_backward_lstm_602_while_maximum_iterations^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_602/while/Identity_1µ
"backward_lstm_602/while/Identity_2Identitybackward_lstm_602/while/add:z:0^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_602/while/Identity_2в
"backward_lstm_602/while/Identity_3IdentityLbackward_lstm_602/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_602/while/Identity_3ќ
"backward_lstm_602/while/Identity_4Identity'backward_lstm_602/while/Select:output:0^backward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_602/while/Identity_4–
"backward_lstm_602/while/Identity_5Identity)backward_lstm_602/while/Select_1:output:0^backward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_602/while/Identity_5–
"backward_lstm_602/while/Identity_6Identity)backward_lstm_602/while/Select_2:output:0^backward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_602/while/Identity_6Њ
backward_lstm_602/while/NoOpNoOp>^backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp=^backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp?^backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_602/while/NoOp"x
9backward_lstm_602_while_backward_lstm_602_strided_slice_1;backward_lstm_602_while_backward_lstm_602_strided_slice_1_0"M
 backward_lstm_602_while_identity)backward_lstm_602/while/Identity:output:0"Q
"backward_lstm_602_while_identity_1+backward_lstm_602/while/Identity_1:output:0"Q
"backward_lstm_602_while_identity_2+backward_lstm_602/while/Identity_2:output:0"Q
"backward_lstm_602_while_identity_3+backward_lstm_602/while/Identity_3:output:0"Q
"backward_lstm_602_while_identity_4+backward_lstm_602/while/Identity_4:output:0"Q
"backward_lstm_602_while_identity_5+backward_lstm_602/while/Identity_5:output:0"Q
"backward_lstm_602_while_identity_6+backward_lstm_602/while/Identity_6:output:0"n
4backward_lstm_602_while_less_backward_lstm_602_sub_16backward_lstm_602_while_less_backward_lstm_602_sub_1_0"Т
Fbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resourceHbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0"Ф
Gbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resourceIbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0"Р
Ebackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resourceGbackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0"р
ubackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2~
=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp2|
<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp2А
>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
$forward_lstm_602_while_cond_60936227>
:forward_lstm_602_while_forward_lstm_602_while_loop_counterD
@forward_lstm_602_while_forward_lstm_602_while_maximum_iterations&
"forward_lstm_602_while_placeholder(
$forward_lstm_602_while_placeholder_1(
$forward_lstm_602_while_placeholder_2(
$forward_lstm_602_while_placeholder_3(
$forward_lstm_602_while_placeholder_4@
<forward_lstm_602_while_less_forward_lstm_602_strided_slice_1X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60936227___redundant_placeholder0X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60936227___redundant_placeholder1X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60936227___redundant_placeholder2X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60936227___redundant_placeholder3X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60936227___redundant_placeholder4#
forward_lstm_602_while_identity
≈
forward_lstm_602/while/LessLess"forward_lstm_602_while_placeholder<forward_lstm_602_while_less_forward_lstm_602_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_602/while/LessР
forward_lstm_602/while/IdentityIdentityforward_lstm_602/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_602/while/Identity"K
forward_lstm_602_while_identity(forward_lstm_602/while/Identity:output:0*(
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
оX
Д
$forward_lstm_602_while_body_60937123>
:forward_lstm_602_while_forward_lstm_602_while_loop_counterD
@forward_lstm_602_while_forward_lstm_602_while_maximum_iterations&
"forward_lstm_602_while_placeholder(
$forward_lstm_602_while_placeholder_1(
$forward_lstm_602_while_placeholder_2(
$forward_lstm_602_while_placeholder_3=
9forward_lstm_602_while_forward_lstm_602_strided_slice_1_0y
uforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0Y
Fforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0:	»[
Hforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0:	2»V
Gforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0:	»#
forward_lstm_602_while_identity%
!forward_lstm_602_while_identity_1%
!forward_lstm_602_while_identity_2%
!forward_lstm_602_while_identity_3%
!forward_lstm_602_while_identity_4%
!forward_lstm_602_while_identity_5;
7forward_lstm_602_while_forward_lstm_602_strided_slice_1w
sforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensorW
Dforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource:	»Y
Fforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource:	2»T
Eforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource:	»ИҐ<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpҐ;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOpҐ=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpе
Hforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2J
Hforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shape¬
:forward_lstm_602/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_602_while_placeholderQforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02<
:forward_lstm_602/while/TensorArrayV2Read/TensorListGetItemВ
;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOpFforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02=
;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp°
,forward_lstm_602/while/lstm_cell_1807/MatMulMatMulAforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem:item:0Cforward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2.
,forward_lstm_602/while/lstm_cell_1807/MatMulИ
=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOpHforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02?
=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpК
.forward_lstm_602/while/lstm_cell_1807/MatMul_1MatMul$forward_lstm_602_while_placeholder_2Eforward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.forward_lstm_602/while/lstm_cell_1807/MatMul_1Д
)forward_lstm_602/while/lstm_cell_1807/addAddV26forward_lstm_602/while/lstm_cell_1807/MatMul:product:08forward_lstm_602/while/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)forward_lstm_602/while/lstm_cell_1807/addБ
<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOpGforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02>
<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpС
-forward_lstm_602/while/lstm_cell_1807/BiasAddBiasAdd-forward_lstm_602/while/lstm_cell_1807/add:z:0Dforward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-forward_lstm_602/while/lstm_cell_1807/BiasAdd∞
5forward_lstm_602/while/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5forward_lstm_602/while/lstm_cell_1807/split/split_dim„
+forward_lstm_602/while/lstm_cell_1807/splitSplit>forward_lstm_602/while/lstm_cell_1807/split/split_dim:output:06forward_lstm_602/while/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2-
+forward_lstm_602/while/lstm_cell_1807/split—
-forward_lstm_602/while/lstm_cell_1807/SigmoidSigmoid4forward_lstm_602/while/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-forward_lstm_602/while/lstm_cell_1807/Sigmoid’
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_1Sigmoid4forward_lstm_602/while/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_1к
)forward_lstm_602/while/lstm_cell_1807/mulMul3forward_lstm_602/while/lstm_cell_1807/Sigmoid_1:y:0$forward_lstm_602_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_602/while/lstm_cell_1807/mul»
*forward_lstm_602/while/lstm_cell_1807/ReluRelu4forward_lstm_602/while/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22,
*forward_lstm_602/while/lstm_cell_1807/ReluА
+forward_lstm_602/while/lstm_cell_1807/mul_1Mul1forward_lstm_602/while/lstm_cell_1807/Sigmoid:y:08forward_lstm_602/while/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_602/while/lstm_cell_1807/mul_1х
+forward_lstm_602/while/lstm_cell_1807/add_1AddV2-forward_lstm_602/while/lstm_cell_1807/mul:z:0/forward_lstm_602/while/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_602/while/lstm_cell_1807/add_1’
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_2Sigmoid4forward_lstm_602/while/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_2«
,forward_lstm_602/while/lstm_cell_1807/Relu_1Relu/forward_lstm_602/while/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,forward_lstm_602/while/lstm_cell_1807/Relu_1Д
+forward_lstm_602/while/lstm_cell_1807/mul_2Mul3forward_lstm_602/while/lstm_cell_1807/Sigmoid_2:y:0:forward_lstm_602/while/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_602/while/lstm_cell_1807/mul_2Ј
;forward_lstm_602/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_602_while_placeholder_1"forward_lstm_602_while_placeholder/forward_lstm_602/while/lstm_cell_1807/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_602/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_602/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_602/while/add/y≠
forward_lstm_602/while/addAddV2"forward_lstm_602_while_placeholder%forward_lstm_602/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/while/addВ
forward_lstm_602/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_602/while/add_1/yЋ
forward_lstm_602/while/add_1AddV2:forward_lstm_602_while_forward_lstm_602_while_loop_counter'forward_lstm_602/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/while/add_1ѓ
forward_lstm_602/while/IdentityIdentity forward_lstm_602/while/add_1:z:0^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_602/while/Identity”
!forward_lstm_602/while/Identity_1Identity@forward_lstm_602_while_forward_lstm_602_while_maximum_iterations^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_602/while/Identity_1±
!forward_lstm_602/while/Identity_2Identityforward_lstm_602/while/add:z:0^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_602/while/Identity_2ё
!forward_lstm_602/while/Identity_3IdentityKforward_lstm_602/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_602/while/Identity_3”
!forward_lstm_602/while/Identity_4Identity/forward_lstm_602/while/lstm_cell_1807/mul_2:z:0^forward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_602/while/Identity_4”
!forward_lstm_602/while/Identity_5Identity/forward_lstm_602/while/lstm_cell_1807/add_1:z:0^forward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_602/while/Identity_5є
forward_lstm_602/while/NoOpNoOp=^forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp<^forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp>^forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_602/while/NoOp"t
7forward_lstm_602_while_forward_lstm_602_strided_slice_19forward_lstm_602_while_forward_lstm_602_strided_slice_1_0"K
forward_lstm_602_while_identity(forward_lstm_602/while/Identity:output:0"O
!forward_lstm_602_while_identity_1*forward_lstm_602/while/Identity_1:output:0"O
!forward_lstm_602_while_identity_2*forward_lstm_602/while/Identity_2:output:0"O
!forward_lstm_602_while_identity_3*forward_lstm_602/while/Identity_3:output:0"O
!forward_lstm_602_while_identity_4*forward_lstm_602/while/Identity_4:output:0"O
!forward_lstm_602_while_identity_5*forward_lstm_602/while/Identity_5:output:0"Р
Eforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resourceGforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0"Т
Fforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resourceHforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0"О
Dforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resourceFforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0"м
sforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensoruforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2|
<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp2z
;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp2~
=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_body_60933717
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_lstm_cell_1807_60933741_0:	»2
while_lstm_cell_1807_60933743_0:	2».
while_lstm_cell_1807_60933745_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_lstm_cell_1807_60933741:	»0
while_lstm_cell_1807_60933743:	2»,
while_lstm_cell_1807_60933745:	»ИҐ,while/lstm_cell_1807/StatefulPartitionedCall√
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
,while/lstm_cell_1807/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1807_60933741_0while_lstm_cell_1807_60933743_0while_lstm_cell_1807_60933745_0*
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
L__inference_lstm_cell_1807_layer_call_and_return_conditional_losses_609337032.
,while/lstm_cell_1807/StatefulPartitionedCallщ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder5while/lstm_cell_1807/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity5while/lstm_cell_1807/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4¶
while/Identity_5Identity5while/lstm_cell_1807/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5Й

while/NoOpNoOp-^while/lstm_cell_1807/StatefulPartitionedCall*"
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
while_lstm_cell_1807_60933741while_lstm_cell_1807_60933741_0"@
while_lstm_cell_1807_60933743while_lstm_cell_1807_60933743_0"@
while_lstm_cell_1807_60933745while_lstm_cell_1807_60933745_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2\
,while/lstm_cell_1807/StatefulPartitionedCall,while/lstm_cell_1807/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
ч
Щ
,__inference_dense_602_layer_call_fn_60938083

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
G__inference_dense_602_layer_call_and_return_conditional_losses_609360892
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
∞Њ
ы
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_60936504

inputs
inputs_1	Q
>forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource:	»S
@forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource:	2»N
?forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource:	»R
?backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resource:	»T
Abackward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource:	2»O
@backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource:	»
identityИҐ7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpҐ6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpҐ8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpҐbackward_lstm_602/whileҐ6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpҐ5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpҐ7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpҐforward_lstm_602/whileЧ
%forward_lstm_602/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_602/RaggedToTensor/zerosЩ
%forward_lstm_602/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2'
%forward_lstm_602/RaggedToTensor/ConstЩ
4forward_lstm_602/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_602/RaggedToTensor/Const:output:0inputs.forward_lstm_602/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_602/RaggedToTensor/RaggedTensorToTensorƒ
;forward_lstm_602/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack»
=forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1»
=forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2©
5forward_lstm_602/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_602/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask27
5forward_lstm_602/RaggedNestedRowLengths/strided_slice»
=forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack’
?forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2A
?forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1ћ
?forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2µ
7forward_lstm_602/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask29
7forward_lstm_602/RaggedNestedRowLengths/strided_slice_1С
+forward_lstm_602/RaggedNestedRowLengths/subSub>forward_lstm_602/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_602/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2-
+forward_lstm_602/RaggedNestedRowLengths/sub§
forward_lstm_602/CastCast/forward_lstm_602/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
forward_lstm_602/CastЭ
forward_lstm_602/ShapeShape=forward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_602/ShapeЦ
$forward_lstm_602/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_602/strided_slice/stackЪ
&forward_lstm_602/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_602/strided_slice/stack_1Ъ
&forward_lstm_602/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_602/strided_slice/stack_2»
forward_lstm_602/strided_sliceStridedSliceforward_lstm_602/Shape:output:0-forward_lstm_602/strided_slice/stack:output:0/forward_lstm_602/strided_slice/stack_1:output:0/forward_lstm_602/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_602/strided_slice~
forward_lstm_602/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_602/zeros/mul/y∞
forward_lstm_602/zeros/mulMul'forward_lstm_602/strided_slice:output:0%forward_lstm_602/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros/mulБ
forward_lstm_602/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
forward_lstm_602/zeros/Less/yЂ
forward_lstm_602/zeros/LessLessforward_lstm_602/zeros/mul:z:0&forward_lstm_602/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros/LessД
forward_lstm_602/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_602/zeros/packed/1«
forward_lstm_602/zeros/packedPack'forward_lstm_602/strided_slice:output:0(forward_lstm_602/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_602/zeros/packedЕ
forward_lstm_602/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_602/zeros/Constє
forward_lstm_602/zerosFill&forward_lstm_602/zeros/packed:output:0%forward_lstm_602/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/zerosВ
forward_lstm_602/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_602/zeros_1/mul/yґ
forward_lstm_602/zeros_1/mulMul'forward_lstm_602/strided_slice:output:0'forward_lstm_602/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros_1/mulЕ
forward_lstm_602/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2!
forward_lstm_602/zeros_1/Less/y≥
forward_lstm_602/zeros_1/LessLess forward_lstm_602/zeros_1/mul:z:0(forward_lstm_602/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros_1/LessИ
!forward_lstm_602/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_602/zeros_1/packed/1Ќ
forward_lstm_602/zeros_1/packedPack'forward_lstm_602/strided_slice:output:0*forward_lstm_602/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_602/zeros_1/packedЙ
forward_lstm_602/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_602/zeros_1/ConstЅ
forward_lstm_602/zeros_1Fill(forward_lstm_602/zeros_1/packed:output:0'forward_lstm_602/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/zeros_1Ч
forward_lstm_602/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_602/transpose/permн
forward_lstm_602/transpose	Transpose=forward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_602/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
forward_lstm_602/transposeВ
forward_lstm_602/Shape_1Shapeforward_lstm_602/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_602/Shape_1Ъ
&forward_lstm_602/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_602/strided_slice_1/stackЮ
(forward_lstm_602/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_1/stack_1Ю
(forward_lstm_602/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_1/stack_2‘
 forward_lstm_602/strided_slice_1StridedSlice!forward_lstm_602/Shape_1:output:0/forward_lstm_602/strided_slice_1/stack:output:01forward_lstm_602/strided_slice_1/stack_1:output:01forward_lstm_602/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_602/strided_slice_1І
,forward_lstm_602/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,forward_lstm_602/TensorArrayV2/element_shapeц
forward_lstm_602/TensorArrayV2TensorListReserve5forward_lstm_602/TensorArrayV2/element_shape:output:0)forward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_602/TensorArrayV2б
Fforward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2H
Fforward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeЉ
8forward_lstm_602/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_602/transpose:y:0Oforward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_602/TensorArrayUnstack/TensorListFromTensorЪ
&forward_lstm_602/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_602/strided_slice_2/stackЮ
(forward_lstm_602/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_2/stack_1Ю
(forward_lstm_602/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_2/stack_2в
 forward_lstm_602/strided_slice_2StridedSliceforward_lstm_602/transpose:y:0/forward_lstm_602/strided_slice_2/stack:output:01forward_lstm_602/strided_slice_2/stack_1:output:01forward_lstm_602/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2"
 forward_lstm_602/strided_slice_2о
5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOp>forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype027
5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpч
&forward_lstm_602/lstm_cell_1807/MatMulMatMul)forward_lstm_602/strided_slice_2:output:0=forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2(
&forward_lstm_602/lstm_cell_1807/MatMulф
7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOp@forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype029
7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpу
(forward_lstm_602/lstm_cell_1807/MatMul_1MatMulforward_lstm_602/zeros:output:0?forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(forward_lstm_602/lstm_cell_1807/MatMul_1м
#forward_lstm_602/lstm_cell_1807/addAddV20forward_lstm_602/lstm_cell_1807/MatMul:product:02forward_lstm_602/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2%
#forward_lstm_602/lstm_cell_1807/addн
6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOp?forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype028
6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpщ
'forward_lstm_602/lstm_cell_1807/BiasAddBiasAdd'forward_lstm_602/lstm_cell_1807/add:z:0>forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'forward_lstm_602/lstm_cell_1807/BiasAdd§
/forward_lstm_602/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/forward_lstm_602/lstm_cell_1807/split/split_dimњ
%forward_lstm_602/lstm_cell_1807/splitSplit8forward_lstm_602/lstm_cell_1807/split/split_dim:output:00forward_lstm_602/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2'
%forward_lstm_602/lstm_cell_1807/splitњ
'forward_lstm_602/lstm_cell_1807/SigmoidSigmoid.forward_lstm_602/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'forward_lstm_602/lstm_cell_1807/Sigmoid√
)forward_lstm_602/lstm_cell_1807/Sigmoid_1Sigmoid.forward_lstm_602/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_602/lstm_cell_1807/Sigmoid_1’
#forward_lstm_602/lstm_cell_1807/mulMul-forward_lstm_602/lstm_cell_1807/Sigmoid_1:y:0!forward_lstm_602/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22%
#forward_lstm_602/lstm_cell_1807/mulґ
$forward_lstm_602/lstm_cell_1807/ReluRelu.forward_lstm_602/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22&
$forward_lstm_602/lstm_cell_1807/Reluи
%forward_lstm_602/lstm_cell_1807/mul_1Mul+forward_lstm_602/lstm_cell_1807/Sigmoid:y:02forward_lstm_602/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_602/lstm_cell_1807/mul_1Ё
%forward_lstm_602/lstm_cell_1807/add_1AddV2'forward_lstm_602/lstm_cell_1807/mul:z:0)forward_lstm_602/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_602/lstm_cell_1807/add_1√
)forward_lstm_602/lstm_cell_1807/Sigmoid_2Sigmoid.forward_lstm_602/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_602/lstm_cell_1807/Sigmoid_2µ
&forward_lstm_602/lstm_cell_1807/Relu_1Relu)forward_lstm_602/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&forward_lstm_602/lstm_cell_1807/Relu_1м
%forward_lstm_602/lstm_cell_1807/mul_2Mul-forward_lstm_602/lstm_cell_1807/Sigmoid_2:y:04forward_lstm_602/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_602/lstm_cell_1807/mul_2±
.forward_lstm_602/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   20
.forward_lstm_602/TensorArrayV2_1/element_shapeь
 forward_lstm_602/TensorArrayV2_1TensorListReserve7forward_lstm_602/TensorArrayV2_1/element_shape:output:0)forward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_602/TensorArrayV2_1p
forward_lstm_602/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_602/time§
forward_lstm_602/zeros_like	ZerosLike)forward_lstm_602/lstm_cell_1807/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/zeros_like°
)forward_lstm_602/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2+
)forward_lstm_602/while/maximum_iterationsМ
#forward_lstm_602/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_602/while/loop_counterЦ	
forward_lstm_602/whileWhile,forward_lstm_602/while/loop_counter:output:02forward_lstm_602/while/maximum_iterations:output:0forward_lstm_602/time:output:0)forward_lstm_602/TensorArrayV2_1:handle:0forward_lstm_602/zeros_like:y:0forward_lstm_602/zeros:output:0!forward_lstm_602/zeros_1:output:0)forward_lstm_602/strided_slice_1:output:0Hforward_lstm_602/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_602/Cast:y:0>forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource@forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource?forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource*
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
$forward_lstm_602_while_body_60936228*0
cond(R&
$forward_lstm_602_while_cond_60936227*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
forward_lstm_602/while„
Aforward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2C
Aforward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeµ
3forward_lstm_602/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_602/while:output:3Jforward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype025
3forward_lstm_602/TensorArrayV2Stack/TensorListStack£
&forward_lstm_602/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2(
&forward_lstm_602/strided_slice_3/stackЮ
(forward_lstm_602/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_602/strided_slice_3/stack_1Ю
(forward_lstm_602/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_3/stack_2А
 forward_lstm_602/strided_slice_3StridedSlice<forward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_602/strided_slice_3/stack:output:01forward_lstm_602/strided_slice_3/stack_1:output:01forward_lstm_602/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2"
 forward_lstm_602/strided_slice_3Ы
!forward_lstm_602/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_602/transpose_1/permт
forward_lstm_602/transpose_1	Transpose<forward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_602/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
forward_lstm_602/transpose_1И
forward_lstm_602/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_602/runtimeЩ
&backward_lstm_602/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_602/RaggedToTensor/zerosЫ
&backward_lstm_602/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2(
&backward_lstm_602/RaggedToTensor/ConstЭ
5backward_lstm_602/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_602/RaggedToTensor/Const:output:0inputs/backward_lstm_602/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_602/RaggedToTensor/RaggedTensorToTensor∆
<backward_lstm_602/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack 
>backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1 
>backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2Ѓ
6backward_lstm_602/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_602/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask28
6backward_lstm_602/RaggedNestedRowLengths/strided_slice 
>backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack„
@backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2B
@backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1ќ
@backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2Ї
8backward_lstm_602/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask2:
8backward_lstm_602/RaggedNestedRowLengths/strided_slice_1Х
,backward_lstm_602/RaggedNestedRowLengths/subSub?backward_lstm_602/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_602/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2.
,backward_lstm_602/RaggedNestedRowLengths/subІ
backward_lstm_602/CastCast0backward_lstm_602/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
backward_lstm_602/Cast†
backward_lstm_602/ShapeShape>backward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_602/ShapeШ
%backward_lstm_602/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_602/strided_slice/stackЬ
'backward_lstm_602/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_602/strided_slice/stack_1Ь
'backward_lstm_602/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_602/strided_slice/stack_2ќ
backward_lstm_602/strided_sliceStridedSlice backward_lstm_602/Shape:output:0.backward_lstm_602/strided_slice/stack:output:00backward_lstm_602/strided_slice/stack_1:output:00backward_lstm_602/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_602/strided_sliceА
backward_lstm_602/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_602/zeros/mul/yі
backward_lstm_602/zeros/mulMul(backward_lstm_602/strided_slice:output:0&backward_lstm_602/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/zeros/mulГ
backward_lstm_602/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
backward_lstm_602/zeros/Less/yѓ
backward_lstm_602/zeros/LessLessbackward_lstm_602/zeros/mul:z:0'backward_lstm_602/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/zeros/LessЖ
 backward_lstm_602/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_602/zeros/packed/1Ћ
backward_lstm_602/zeros/packedPack(backward_lstm_602/strided_slice:output:0)backward_lstm_602/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_602/zeros/packedЗ
backward_lstm_602/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_602/zeros/Constљ
backward_lstm_602/zerosFill'backward_lstm_602/zeros/packed:output:0&backward_lstm_602/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_602/zerosД
backward_lstm_602/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_602/zeros_1/mul/yЇ
backward_lstm_602/zeros_1/mulMul(backward_lstm_602/strided_slice:output:0(backward_lstm_602/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/zeros_1/mulЗ
 backward_lstm_602/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 backward_lstm_602/zeros_1/Less/yЈ
backward_lstm_602/zeros_1/LessLess!backward_lstm_602/zeros_1/mul:z:0)backward_lstm_602/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_602/zeros_1/LessК
"backward_lstm_602/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_602/zeros_1/packed/1—
 backward_lstm_602/zeros_1/packedPack(backward_lstm_602/strided_slice:output:0+backward_lstm_602/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_602/zeros_1/packedЛ
backward_lstm_602/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_602/zeros_1/Const≈
backward_lstm_602/zeros_1Fill)backward_lstm_602/zeros_1/packed:output:0(backward_lstm_602/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_602/zeros_1Щ
 backward_lstm_602/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_602/transpose/permс
backward_lstm_602/transpose	Transpose>backward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_602/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_602/transposeЕ
backward_lstm_602/Shape_1Shapebackward_lstm_602/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_602/Shape_1Ь
'backward_lstm_602/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_602/strided_slice_1/stack†
)backward_lstm_602/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_1/stack_1†
)backward_lstm_602/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_1/stack_2Џ
!backward_lstm_602/strided_slice_1StridedSlice"backward_lstm_602/Shape_1:output:00backward_lstm_602/strided_slice_1/stack:output:02backward_lstm_602/strided_slice_1/stack_1:output:02backward_lstm_602/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_602/strided_slice_1©
-backward_lstm_602/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-backward_lstm_602/TensorArrayV2/element_shapeъ
backward_lstm_602/TensorArrayV2TensorListReserve6backward_lstm_602/TensorArrayV2/element_shape:output:0*backward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_602/TensorArrayV2О
 backward_lstm_602/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_602/ReverseV2/axis“
backward_lstm_602/ReverseV2	ReverseV2backward_lstm_602/transpose:y:0)backward_lstm_602/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_602/ReverseV2г
Gbackward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2I
Gbackward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape≈
9backward_lstm_602/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_602/ReverseV2:output:0Pbackward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_602/TensorArrayUnstack/TensorListFromTensorЬ
'backward_lstm_602/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_602/strided_slice_2/stack†
)backward_lstm_602/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_2/stack_1†
)backward_lstm_602/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_2/stack_2и
!backward_lstm_602/strided_slice_2StridedSlicebackward_lstm_602/transpose:y:00backward_lstm_602/strided_slice_2/stack:output:02backward_lstm_602/strided_slice_2/stack_1:output:02backward_lstm_602/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2#
!backward_lstm_602/strided_slice_2с
6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOp?backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype028
6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpы
'backward_lstm_602/lstm_cell_1808/MatMulMatMul*backward_lstm_602/strided_slice_2:output:0>backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'backward_lstm_602/lstm_cell_1808/MatMulч
8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOpAbackward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02:
8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpч
)backward_lstm_602/lstm_cell_1808/MatMul_1MatMul backward_lstm_602/zeros:output:0@backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)backward_lstm_602/lstm_cell_1808/MatMul_1р
$backward_lstm_602/lstm_cell_1808/addAddV21backward_lstm_602/lstm_cell_1808/MatMul:product:03backward_lstm_602/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2&
$backward_lstm_602/lstm_cell_1808/addр
7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOp@backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype029
7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpэ
(backward_lstm_602/lstm_cell_1808/BiasAddBiasAdd(backward_lstm_602/lstm_cell_1808/add:z:0?backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(backward_lstm_602/lstm_cell_1808/BiasAdd¶
0backward_lstm_602/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0backward_lstm_602/lstm_cell_1808/split/split_dim√
&backward_lstm_602/lstm_cell_1808/splitSplit9backward_lstm_602/lstm_cell_1808/split/split_dim:output:01backward_lstm_602/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2(
&backward_lstm_602/lstm_cell_1808/split¬
(backward_lstm_602/lstm_cell_1808/SigmoidSigmoid/backward_lstm_602/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(backward_lstm_602/lstm_cell_1808/Sigmoid∆
*backward_lstm_602/lstm_cell_1808/Sigmoid_1Sigmoid/backward_lstm_602/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_602/lstm_cell_1808/Sigmoid_1ў
$backward_lstm_602/lstm_cell_1808/mulMul.backward_lstm_602/lstm_cell_1808/Sigmoid_1:y:0"backward_lstm_602/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$backward_lstm_602/lstm_cell_1808/mulє
%backward_lstm_602/lstm_cell_1808/ReluRelu/backward_lstm_602/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22'
%backward_lstm_602/lstm_cell_1808/Reluм
&backward_lstm_602/lstm_cell_1808/mul_1Mul,backward_lstm_602/lstm_cell_1808/Sigmoid:y:03backward_lstm_602/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_602/lstm_cell_1808/mul_1б
&backward_lstm_602/lstm_cell_1808/add_1AddV2(backward_lstm_602/lstm_cell_1808/mul:z:0*backward_lstm_602/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_602/lstm_cell_1808/add_1∆
*backward_lstm_602/lstm_cell_1808/Sigmoid_2Sigmoid/backward_lstm_602/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_602/lstm_cell_1808/Sigmoid_2Є
'backward_lstm_602/lstm_cell_1808/Relu_1Relu*backward_lstm_602/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'backward_lstm_602/lstm_cell_1808/Relu_1р
&backward_lstm_602/lstm_cell_1808/mul_2Mul.backward_lstm_602/lstm_cell_1808/Sigmoid_2:y:05backward_lstm_602/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_602/lstm_cell_1808/mul_2≥
/backward_lstm_602/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   21
/backward_lstm_602/TensorArrayV2_1/element_shapeА
!backward_lstm_602/TensorArrayV2_1TensorListReserve8backward_lstm_602/TensorArrayV2_1/element_shape:output:0*backward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_602/TensorArrayV2_1r
backward_lstm_602/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_602/timeФ
'backward_lstm_602/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_602/Max/reduction_indices§
backward_lstm_602/MaxMaxbackward_lstm_602/Cast:y:00backward_lstm_602/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/Maxt
backward_lstm_602/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_602/sub/yШ
backward_lstm_602/subSubbackward_lstm_602/Max:output:0 backward_lstm_602/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/subЮ
backward_lstm_602/Sub_1Subbackward_lstm_602/sub:z:0backward_lstm_602/Cast:y:0*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_602/Sub_1І
backward_lstm_602/zeros_like	ZerosLike*backward_lstm_602/lstm_cell_1808/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_602/zeros_like£
*backward_lstm_602/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2,
*backward_lstm_602/while/maximum_iterationsО
$backward_lstm_602/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_602/while/loop_counter®	
backward_lstm_602/whileWhile-backward_lstm_602/while/loop_counter:output:03backward_lstm_602/while/maximum_iterations:output:0backward_lstm_602/time:output:0*backward_lstm_602/TensorArrayV2_1:handle:0 backward_lstm_602/zeros_like:y:0 backward_lstm_602/zeros:output:0"backward_lstm_602/zeros_1:output:0*backward_lstm_602/strided_slice_1:output:0Ibackward_lstm_602/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_602/Sub_1:z:0?backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resourceAbackward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource@backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource*
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
%backward_lstm_602_while_body_60936407*1
cond)R'
%backward_lstm_602_while_cond_60936406*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
backward_lstm_602/whileў
Bbackward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2D
Bbackward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeє
4backward_lstm_602/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_602/while:output:3Kbackward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype026
4backward_lstm_602/TensorArrayV2Stack/TensorListStack•
'backward_lstm_602/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2)
'backward_lstm_602/strided_slice_3/stack†
)backward_lstm_602/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_602/strided_slice_3/stack_1†
)backward_lstm_602/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_3/stack_2Ж
!backward_lstm_602/strided_slice_3StridedSlice=backward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_602/strided_slice_3/stack:output:02backward_lstm_602/strided_slice_3/stack_1:output:02backward_lstm_602/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2#
!backward_lstm_602/strided_slice_3Э
"backward_lstm_602/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_602/transpose_1/permц
backward_lstm_602/transpose_1	Transpose=backward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_602/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
backward_lstm_602/transpose_1К
backward_lstm_602/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_602/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisƒ
concatConcatV2)forward_lstm_602/strided_slice_3:output:0*backward_lstm_602/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp8^backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp7^backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp9^backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp^backward_lstm_602/while7^forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp6^forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp8^forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp^forward_lstm_602/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€:€€€€€€€€€: : : : : : 2r
7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp2p
6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp2t
8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp22
backward_lstm_602/whilebackward_lstm_602/while2p
6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp2n
5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp2r
7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp20
forward_lstm_602/whileforward_lstm_602/while:O K
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
while_cond_60933926
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60933926___redundant_placeholder06
2while_while_cond_60933926___redundant_placeholder16
2while_while_cond_60933926___redundant_placeholder26
2while_while_cond_60933926___redundant_placeholder3
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
ђ&
€
while_body_60934349
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_lstm_cell_1808_60934373_0:	»2
while_lstm_cell_1808_60934375_0:	2».
while_lstm_cell_1808_60934377_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_lstm_cell_1808_60934373:	»0
while_lstm_cell_1808_60934375:	2»,
while_lstm_cell_1808_60934377:	»ИҐ,while/lstm_cell_1808/StatefulPartitionedCall√
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
,while/lstm_cell_1808/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1808_60934373_0while_lstm_cell_1808_60934375_0while_lstm_cell_1808_60934377_0*
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
L__inference_lstm_cell_1808_layer_call_and_return_conditional_losses_609343352.
,while/lstm_cell_1808/StatefulPartitionedCallщ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder5while/lstm_cell_1808/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity5while/lstm_cell_1808/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4¶
while/Identity_5Identity5while/lstm_cell_1808/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5Й

while/NoOpNoOp-^while/lstm_cell_1808/StatefulPartitionedCall*"
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
while_lstm_cell_1808_60934373while_lstm_cell_1808_60934373_0"@
while_lstm_cell_1808_60934375while_lstm_cell_1808_60934375_0"@
while_lstm_cell_1808_60934377while_lstm_cell_1808_60934377_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2\
,while/lstm_cell_1808/StatefulPartitionedCall,while/lstm_cell_1808/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_60936064

inputs
inputs_1	Q
>forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource:	»S
@forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource:	2»N
?forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource:	»R
?backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resource:	»T
Abackward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource:	2»O
@backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource:	»
identityИҐ7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpҐ6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpҐ8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpҐbackward_lstm_602/whileҐ6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpҐ5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpҐ7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpҐforward_lstm_602/whileЧ
%forward_lstm_602/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_602/RaggedToTensor/zerosЩ
%forward_lstm_602/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2'
%forward_lstm_602/RaggedToTensor/ConstЩ
4forward_lstm_602/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_602/RaggedToTensor/Const:output:0inputs.forward_lstm_602/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_602/RaggedToTensor/RaggedTensorToTensorƒ
;forward_lstm_602/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack»
=forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1»
=forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2©
5forward_lstm_602/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_602/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask27
5forward_lstm_602/RaggedNestedRowLengths/strided_slice»
=forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack’
?forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2A
?forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1ћ
?forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2µ
7forward_lstm_602/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask29
7forward_lstm_602/RaggedNestedRowLengths/strided_slice_1С
+forward_lstm_602/RaggedNestedRowLengths/subSub>forward_lstm_602/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_602/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2-
+forward_lstm_602/RaggedNestedRowLengths/sub§
forward_lstm_602/CastCast/forward_lstm_602/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
forward_lstm_602/CastЭ
forward_lstm_602/ShapeShape=forward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_602/ShapeЦ
$forward_lstm_602/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_602/strided_slice/stackЪ
&forward_lstm_602/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_602/strided_slice/stack_1Ъ
&forward_lstm_602/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_602/strided_slice/stack_2»
forward_lstm_602/strided_sliceStridedSliceforward_lstm_602/Shape:output:0-forward_lstm_602/strided_slice/stack:output:0/forward_lstm_602/strided_slice/stack_1:output:0/forward_lstm_602/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_602/strided_slice~
forward_lstm_602/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_602/zeros/mul/y∞
forward_lstm_602/zeros/mulMul'forward_lstm_602/strided_slice:output:0%forward_lstm_602/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros/mulБ
forward_lstm_602/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
forward_lstm_602/zeros/Less/yЂ
forward_lstm_602/zeros/LessLessforward_lstm_602/zeros/mul:z:0&forward_lstm_602/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros/LessД
forward_lstm_602/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_602/zeros/packed/1«
forward_lstm_602/zeros/packedPack'forward_lstm_602/strided_slice:output:0(forward_lstm_602/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_602/zeros/packedЕ
forward_lstm_602/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_602/zeros/Constє
forward_lstm_602/zerosFill&forward_lstm_602/zeros/packed:output:0%forward_lstm_602/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/zerosВ
forward_lstm_602/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_602/zeros_1/mul/yґ
forward_lstm_602/zeros_1/mulMul'forward_lstm_602/strided_slice:output:0'forward_lstm_602/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros_1/mulЕ
forward_lstm_602/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2!
forward_lstm_602/zeros_1/Less/y≥
forward_lstm_602/zeros_1/LessLess forward_lstm_602/zeros_1/mul:z:0(forward_lstm_602/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros_1/LessИ
!forward_lstm_602/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_602/zeros_1/packed/1Ќ
forward_lstm_602/zeros_1/packedPack'forward_lstm_602/strided_slice:output:0*forward_lstm_602/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_602/zeros_1/packedЙ
forward_lstm_602/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_602/zeros_1/ConstЅ
forward_lstm_602/zeros_1Fill(forward_lstm_602/zeros_1/packed:output:0'forward_lstm_602/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/zeros_1Ч
forward_lstm_602/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_602/transpose/permн
forward_lstm_602/transpose	Transpose=forward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_602/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
forward_lstm_602/transposeВ
forward_lstm_602/Shape_1Shapeforward_lstm_602/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_602/Shape_1Ъ
&forward_lstm_602/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_602/strided_slice_1/stackЮ
(forward_lstm_602/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_1/stack_1Ю
(forward_lstm_602/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_1/stack_2‘
 forward_lstm_602/strided_slice_1StridedSlice!forward_lstm_602/Shape_1:output:0/forward_lstm_602/strided_slice_1/stack:output:01forward_lstm_602/strided_slice_1/stack_1:output:01forward_lstm_602/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_602/strided_slice_1І
,forward_lstm_602/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,forward_lstm_602/TensorArrayV2/element_shapeц
forward_lstm_602/TensorArrayV2TensorListReserve5forward_lstm_602/TensorArrayV2/element_shape:output:0)forward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_602/TensorArrayV2б
Fforward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2H
Fforward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeЉ
8forward_lstm_602/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_602/transpose:y:0Oforward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_602/TensorArrayUnstack/TensorListFromTensorЪ
&forward_lstm_602/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_602/strided_slice_2/stackЮ
(forward_lstm_602/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_2/stack_1Ю
(forward_lstm_602/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_2/stack_2в
 forward_lstm_602/strided_slice_2StridedSliceforward_lstm_602/transpose:y:0/forward_lstm_602/strided_slice_2/stack:output:01forward_lstm_602/strided_slice_2/stack_1:output:01forward_lstm_602/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2"
 forward_lstm_602/strided_slice_2о
5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOp>forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype027
5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpч
&forward_lstm_602/lstm_cell_1807/MatMulMatMul)forward_lstm_602/strided_slice_2:output:0=forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2(
&forward_lstm_602/lstm_cell_1807/MatMulф
7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOp@forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype029
7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpу
(forward_lstm_602/lstm_cell_1807/MatMul_1MatMulforward_lstm_602/zeros:output:0?forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(forward_lstm_602/lstm_cell_1807/MatMul_1м
#forward_lstm_602/lstm_cell_1807/addAddV20forward_lstm_602/lstm_cell_1807/MatMul:product:02forward_lstm_602/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2%
#forward_lstm_602/lstm_cell_1807/addн
6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOp?forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype028
6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpщ
'forward_lstm_602/lstm_cell_1807/BiasAddBiasAdd'forward_lstm_602/lstm_cell_1807/add:z:0>forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'forward_lstm_602/lstm_cell_1807/BiasAdd§
/forward_lstm_602/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/forward_lstm_602/lstm_cell_1807/split/split_dimњ
%forward_lstm_602/lstm_cell_1807/splitSplit8forward_lstm_602/lstm_cell_1807/split/split_dim:output:00forward_lstm_602/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2'
%forward_lstm_602/lstm_cell_1807/splitњ
'forward_lstm_602/lstm_cell_1807/SigmoidSigmoid.forward_lstm_602/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'forward_lstm_602/lstm_cell_1807/Sigmoid√
)forward_lstm_602/lstm_cell_1807/Sigmoid_1Sigmoid.forward_lstm_602/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_602/lstm_cell_1807/Sigmoid_1’
#forward_lstm_602/lstm_cell_1807/mulMul-forward_lstm_602/lstm_cell_1807/Sigmoid_1:y:0!forward_lstm_602/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22%
#forward_lstm_602/lstm_cell_1807/mulґ
$forward_lstm_602/lstm_cell_1807/ReluRelu.forward_lstm_602/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22&
$forward_lstm_602/lstm_cell_1807/Reluи
%forward_lstm_602/lstm_cell_1807/mul_1Mul+forward_lstm_602/lstm_cell_1807/Sigmoid:y:02forward_lstm_602/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_602/lstm_cell_1807/mul_1Ё
%forward_lstm_602/lstm_cell_1807/add_1AddV2'forward_lstm_602/lstm_cell_1807/mul:z:0)forward_lstm_602/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_602/lstm_cell_1807/add_1√
)forward_lstm_602/lstm_cell_1807/Sigmoid_2Sigmoid.forward_lstm_602/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_602/lstm_cell_1807/Sigmoid_2µ
&forward_lstm_602/lstm_cell_1807/Relu_1Relu)forward_lstm_602/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&forward_lstm_602/lstm_cell_1807/Relu_1м
%forward_lstm_602/lstm_cell_1807/mul_2Mul-forward_lstm_602/lstm_cell_1807/Sigmoid_2:y:04forward_lstm_602/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_602/lstm_cell_1807/mul_2±
.forward_lstm_602/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   20
.forward_lstm_602/TensorArrayV2_1/element_shapeь
 forward_lstm_602/TensorArrayV2_1TensorListReserve7forward_lstm_602/TensorArrayV2_1/element_shape:output:0)forward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_602/TensorArrayV2_1p
forward_lstm_602/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_602/time§
forward_lstm_602/zeros_like	ZerosLike)forward_lstm_602/lstm_cell_1807/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/zeros_like°
)forward_lstm_602/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2+
)forward_lstm_602/while/maximum_iterationsМ
#forward_lstm_602/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_602/while/loop_counterЦ	
forward_lstm_602/whileWhile,forward_lstm_602/while/loop_counter:output:02forward_lstm_602/while/maximum_iterations:output:0forward_lstm_602/time:output:0)forward_lstm_602/TensorArrayV2_1:handle:0forward_lstm_602/zeros_like:y:0forward_lstm_602/zeros:output:0!forward_lstm_602/zeros_1:output:0)forward_lstm_602/strided_slice_1:output:0Hforward_lstm_602/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_602/Cast:y:0>forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource@forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource?forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource*
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
$forward_lstm_602_while_body_60935788*0
cond(R&
$forward_lstm_602_while_cond_60935787*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
forward_lstm_602/while„
Aforward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2C
Aforward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeµ
3forward_lstm_602/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_602/while:output:3Jforward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype025
3forward_lstm_602/TensorArrayV2Stack/TensorListStack£
&forward_lstm_602/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2(
&forward_lstm_602/strided_slice_3/stackЮ
(forward_lstm_602/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_602/strided_slice_3/stack_1Ю
(forward_lstm_602/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_3/stack_2А
 forward_lstm_602/strided_slice_3StridedSlice<forward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_602/strided_slice_3/stack:output:01forward_lstm_602/strided_slice_3/stack_1:output:01forward_lstm_602/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2"
 forward_lstm_602/strided_slice_3Ы
!forward_lstm_602/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_602/transpose_1/permт
forward_lstm_602/transpose_1	Transpose<forward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_602/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
forward_lstm_602/transpose_1И
forward_lstm_602/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_602/runtimeЩ
&backward_lstm_602/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_602/RaggedToTensor/zerosЫ
&backward_lstm_602/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2(
&backward_lstm_602/RaggedToTensor/ConstЭ
5backward_lstm_602/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_602/RaggedToTensor/Const:output:0inputs/backward_lstm_602/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_602/RaggedToTensor/RaggedTensorToTensor∆
<backward_lstm_602/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack 
>backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1 
>backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2Ѓ
6backward_lstm_602/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_602/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask28
6backward_lstm_602/RaggedNestedRowLengths/strided_slice 
>backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack„
@backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2B
@backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1ќ
@backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2Ї
8backward_lstm_602/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask2:
8backward_lstm_602/RaggedNestedRowLengths/strided_slice_1Х
,backward_lstm_602/RaggedNestedRowLengths/subSub?backward_lstm_602/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_602/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2.
,backward_lstm_602/RaggedNestedRowLengths/subІ
backward_lstm_602/CastCast0backward_lstm_602/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
backward_lstm_602/Cast†
backward_lstm_602/ShapeShape>backward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_602/ShapeШ
%backward_lstm_602/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_602/strided_slice/stackЬ
'backward_lstm_602/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_602/strided_slice/stack_1Ь
'backward_lstm_602/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_602/strided_slice/stack_2ќ
backward_lstm_602/strided_sliceStridedSlice backward_lstm_602/Shape:output:0.backward_lstm_602/strided_slice/stack:output:00backward_lstm_602/strided_slice/stack_1:output:00backward_lstm_602/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_602/strided_sliceА
backward_lstm_602/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_602/zeros/mul/yі
backward_lstm_602/zeros/mulMul(backward_lstm_602/strided_slice:output:0&backward_lstm_602/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/zeros/mulГ
backward_lstm_602/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
backward_lstm_602/zeros/Less/yѓ
backward_lstm_602/zeros/LessLessbackward_lstm_602/zeros/mul:z:0'backward_lstm_602/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/zeros/LessЖ
 backward_lstm_602/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_602/zeros/packed/1Ћ
backward_lstm_602/zeros/packedPack(backward_lstm_602/strided_slice:output:0)backward_lstm_602/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_602/zeros/packedЗ
backward_lstm_602/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_602/zeros/Constљ
backward_lstm_602/zerosFill'backward_lstm_602/zeros/packed:output:0&backward_lstm_602/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_602/zerosД
backward_lstm_602/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_602/zeros_1/mul/yЇ
backward_lstm_602/zeros_1/mulMul(backward_lstm_602/strided_slice:output:0(backward_lstm_602/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/zeros_1/mulЗ
 backward_lstm_602/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 backward_lstm_602/zeros_1/Less/yЈ
backward_lstm_602/zeros_1/LessLess!backward_lstm_602/zeros_1/mul:z:0)backward_lstm_602/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_602/zeros_1/LessК
"backward_lstm_602/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_602/zeros_1/packed/1—
 backward_lstm_602/zeros_1/packedPack(backward_lstm_602/strided_slice:output:0+backward_lstm_602/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_602/zeros_1/packedЛ
backward_lstm_602/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_602/zeros_1/Const≈
backward_lstm_602/zeros_1Fill)backward_lstm_602/zeros_1/packed:output:0(backward_lstm_602/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_602/zeros_1Щ
 backward_lstm_602/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_602/transpose/permс
backward_lstm_602/transpose	Transpose>backward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_602/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_602/transposeЕ
backward_lstm_602/Shape_1Shapebackward_lstm_602/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_602/Shape_1Ь
'backward_lstm_602/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_602/strided_slice_1/stack†
)backward_lstm_602/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_1/stack_1†
)backward_lstm_602/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_1/stack_2Џ
!backward_lstm_602/strided_slice_1StridedSlice"backward_lstm_602/Shape_1:output:00backward_lstm_602/strided_slice_1/stack:output:02backward_lstm_602/strided_slice_1/stack_1:output:02backward_lstm_602/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_602/strided_slice_1©
-backward_lstm_602/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-backward_lstm_602/TensorArrayV2/element_shapeъ
backward_lstm_602/TensorArrayV2TensorListReserve6backward_lstm_602/TensorArrayV2/element_shape:output:0*backward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_602/TensorArrayV2О
 backward_lstm_602/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_602/ReverseV2/axis“
backward_lstm_602/ReverseV2	ReverseV2backward_lstm_602/transpose:y:0)backward_lstm_602/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_602/ReverseV2г
Gbackward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2I
Gbackward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape≈
9backward_lstm_602/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_602/ReverseV2:output:0Pbackward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_602/TensorArrayUnstack/TensorListFromTensorЬ
'backward_lstm_602/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_602/strided_slice_2/stack†
)backward_lstm_602/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_2/stack_1†
)backward_lstm_602/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_2/stack_2и
!backward_lstm_602/strided_slice_2StridedSlicebackward_lstm_602/transpose:y:00backward_lstm_602/strided_slice_2/stack:output:02backward_lstm_602/strided_slice_2/stack_1:output:02backward_lstm_602/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2#
!backward_lstm_602/strided_slice_2с
6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOp?backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype028
6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpы
'backward_lstm_602/lstm_cell_1808/MatMulMatMul*backward_lstm_602/strided_slice_2:output:0>backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'backward_lstm_602/lstm_cell_1808/MatMulч
8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOpAbackward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02:
8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpч
)backward_lstm_602/lstm_cell_1808/MatMul_1MatMul backward_lstm_602/zeros:output:0@backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)backward_lstm_602/lstm_cell_1808/MatMul_1р
$backward_lstm_602/lstm_cell_1808/addAddV21backward_lstm_602/lstm_cell_1808/MatMul:product:03backward_lstm_602/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2&
$backward_lstm_602/lstm_cell_1808/addр
7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOp@backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype029
7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpэ
(backward_lstm_602/lstm_cell_1808/BiasAddBiasAdd(backward_lstm_602/lstm_cell_1808/add:z:0?backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(backward_lstm_602/lstm_cell_1808/BiasAdd¶
0backward_lstm_602/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0backward_lstm_602/lstm_cell_1808/split/split_dim√
&backward_lstm_602/lstm_cell_1808/splitSplit9backward_lstm_602/lstm_cell_1808/split/split_dim:output:01backward_lstm_602/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2(
&backward_lstm_602/lstm_cell_1808/split¬
(backward_lstm_602/lstm_cell_1808/SigmoidSigmoid/backward_lstm_602/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(backward_lstm_602/lstm_cell_1808/Sigmoid∆
*backward_lstm_602/lstm_cell_1808/Sigmoid_1Sigmoid/backward_lstm_602/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_602/lstm_cell_1808/Sigmoid_1ў
$backward_lstm_602/lstm_cell_1808/mulMul.backward_lstm_602/lstm_cell_1808/Sigmoid_1:y:0"backward_lstm_602/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$backward_lstm_602/lstm_cell_1808/mulє
%backward_lstm_602/lstm_cell_1808/ReluRelu/backward_lstm_602/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22'
%backward_lstm_602/lstm_cell_1808/Reluм
&backward_lstm_602/lstm_cell_1808/mul_1Mul,backward_lstm_602/lstm_cell_1808/Sigmoid:y:03backward_lstm_602/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_602/lstm_cell_1808/mul_1б
&backward_lstm_602/lstm_cell_1808/add_1AddV2(backward_lstm_602/lstm_cell_1808/mul:z:0*backward_lstm_602/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_602/lstm_cell_1808/add_1∆
*backward_lstm_602/lstm_cell_1808/Sigmoid_2Sigmoid/backward_lstm_602/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_602/lstm_cell_1808/Sigmoid_2Є
'backward_lstm_602/lstm_cell_1808/Relu_1Relu*backward_lstm_602/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'backward_lstm_602/lstm_cell_1808/Relu_1р
&backward_lstm_602/lstm_cell_1808/mul_2Mul.backward_lstm_602/lstm_cell_1808/Sigmoid_2:y:05backward_lstm_602/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_602/lstm_cell_1808/mul_2≥
/backward_lstm_602/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   21
/backward_lstm_602/TensorArrayV2_1/element_shapeА
!backward_lstm_602/TensorArrayV2_1TensorListReserve8backward_lstm_602/TensorArrayV2_1/element_shape:output:0*backward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_602/TensorArrayV2_1r
backward_lstm_602/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_602/timeФ
'backward_lstm_602/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_602/Max/reduction_indices§
backward_lstm_602/MaxMaxbackward_lstm_602/Cast:y:00backward_lstm_602/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/Maxt
backward_lstm_602/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_602/sub/yШ
backward_lstm_602/subSubbackward_lstm_602/Max:output:0 backward_lstm_602/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/subЮ
backward_lstm_602/Sub_1Subbackward_lstm_602/sub:z:0backward_lstm_602/Cast:y:0*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_602/Sub_1І
backward_lstm_602/zeros_like	ZerosLike*backward_lstm_602/lstm_cell_1808/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_602/zeros_like£
*backward_lstm_602/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2,
*backward_lstm_602/while/maximum_iterationsО
$backward_lstm_602/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_602/while/loop_counter®	
backward_lstm_602/whileWhile-backward_lstm_602/while/loop_counter:output:03backward_lstm_602/while/maximum_iterations:output:0backward_lstm_602/time:output:0*backward_lstm_602/TensorArrayV2_1:handle:0 backward_lstm_602/zeros_like:y:0 backward_lstm_602/zeros:output:0"backward_lstm_602/zeros_1:output:0*backward_lstm_602/strided_slice_1:output:0Ibackward_lstm_602/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_602/Sub_1:z:0?backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resourceAbackward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource@backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource*
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
%backward_lstm_602_while_body_60935967*1
cond)R'
%backward_lstm_602_while_cond_60935966*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
backward_lstm_602/whileў
Bbackward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2D
Bbackward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeє
4backward_lstm_602/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_602/while:output:3Kbackward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype026
4backward_lstm_602/TensorArrayV2Stack/TensorListStack•
'backward_lstm_602/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2)
'backward_lstm_602/strided_slice_3/stack†
)backward_lstm_602/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_602/strided_slice_3/stack_1†
)backward_lstm_602/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_3/stack_2Ж
!backward_lstm_602/strided_slice_3StridedSlice=backward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_602/strided_slice_3/stack:output:02backward_lstm_602/strided_slice_3/stack_1:output:02backward_lstm_602/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2#
!backward_lstm_602/strided_slice_3Э
"backward_lstm_602/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_602/transpose_1/permц
backward_lstm_602/transpose_1	Transpose=backward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_602/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
backward_lstm_602/transpose_1К
backward_lstm_602/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_602/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisƒ
concatConcatV2)forward_lstm_602/strided_slice_3:output:0*backward_lstm_602/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp8^backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp7^backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp9^backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp^backward_lstm_602/while7^forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp6^forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp8^forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp^forward_lstm_602/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€:€€€€€€€€€: : : : : : 2r
7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp2p
6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp2t
8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp22
backward_lstm_602/whilebackward_lstm_602/while2p
6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp2n
5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp2r
7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp20
forward_lstm_602/whileforward_lstm_602/while:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ѕ@
д
while_body_60938658
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1807_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1807_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1807_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1807_matmul_readvariableop_resource:	»H
5while_lstm_cell_1807_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1807_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1807/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1807/MatMul/ReadVariableOpҐ,while/lstm_cell_1807/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1807_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1807/MatMul/ReadVariableOpЁ
while/lstm_cell_1807/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/MatMul’
,while/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1807_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1807/MatMul_1/ReadVariableOp∆
while/lstm_cell_1807/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/MatMul_1ј
while/lstm_cell_1807/addAddV2%while/lstm_cell_1807/MatMul:product:0'while/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/addќ
+while/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1807_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1807/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1807/BiasAddBiasAddwhile/lstm_cell_1807/add:z:03while/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/BiasAddО
$while/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1807/split/split_dimУ
while/lstm_cell_1807/splitSplit-while/lstm_cell_1807/split/split_dim:output:0%while/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1807/splitЮ
while/lstm_cell_1807/SigmoidSigmoid#while/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/SigmoidҐ
while/lstm_cell_1807/Sigmoid_1Sigmoid#while/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1807/Sigmoid_1¶
while/lstm_cell_1807/mulMul"while/lstm_cell_1807/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/mulХ
while/lstm_cell_1807/ReluRelu#while/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/ReluЉ
while/lstm_cell_1807/mul_1Mul while/lstm_cell_1807/Sigmoid:y:0'while/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/mul_1±
while/lstm_cell_1807/add_1AddV2while/lstm_cell_1807/mul:z:0while/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/add_1Ґ
while/lstm_cell_1807/Sigmoid_2Sigmoid#while/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1807/Sigmoid_2Ф
while/lstm_cell_1807/Relu_1Reluwhile/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/Relu_1ј
while/lstm_cell_1807/mul_2Mul"while/lstm_cell_1807/Sigmoid_2:y:0)while/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1807/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1807/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1807/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1807/BiasAdd/ReadVariableOp+^while/lstm_cell_1807/MatMul/ReadVariableOp-^while/lstm_cell_1807/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1807_biasadd_readvariableop_resource6while_lstm_cell_1807_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1807_matmul_1_readvariableop_resource7while_lstm_cell_1807_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1807_matmul_readvariableop_resource5while_lstm_cell_1807_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1807/BiasAdd/ReadVariableOp+while/lstm_cell_1807/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1807/MatMul/ReadVariableOp*while/lstm_cell_1807/MatMul/ReadVariableOp2\
,while/lstm_cell_1807/MatMul_1/ReadVariableOp,while/lstm_cell_1807/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
L__inference_lstm_cell_1808_layer_call_and_return_conditional_losses_60939594

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
я
Ќ
while_cond_60938355
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60938355___redundant_placeholder06
2while_while_cond_60938355___redundant_placeholder16
2while_while_cond_60938355___redundant_placeholder26
2while_while_cond_60938355___redundant_placeholder3
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
ь

Ў
1__inference_sequential_602_layer_call_fn_60936115

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
L__inference_sequential_602_layer_call_and_return_conditional_losses_609360962
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
жF
Ю
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_60933996

inputs*
lstm_cell_1807_60933914:	»*
lstm_cell_1807_60933916:	2»&
lstm_cell_1807_60933918:	»
identityИҐ&lstm_cell_1807/StatefulPartitionedCallҐwhileD
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
&lstm_cell_1807/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1807_60933914lstm_cell_1807_60933916lstm_cell_1807_60933918*
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
L__inference_lstm_cell_1807_layer_call_and_return_conditional_losses_609338492(
&lstm_cell_1807/StatefulPartitionedCallП
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1807_60933914lstm_cell_1807_60933916lstm_cell_1807_60933918*
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
while_body_60933927*
condR
while_cond_60933926*K
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
NoOpNoOp'^lstm_cell_1807/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2P
&lstm_cell_1807/StatefulPartitionedCall&lstm_cell_1807/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
З
ш
G__inference_dense_602_layer_call_and_return_conditional_losses_60936089

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
я
Ќ
while_cond_60934968
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60934968___redundant_placeholder06
2while_while_cond_60934968___redundant_placeholder16
2while_while_cond_60934968___redundant_placeholder26
2while_while_cond_60934968___redundant_placeholder3
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
ф_
≥
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_60939398

inputs@
-lstm_cell_1808_matmul_readvariableop_resource:	»B
/lstm_cell_1808_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1808_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1808/BiasAdd/ReadVariableOpҐ$lstm_cell_1808/MatMul/ReadVariableOpҐ&lstm_cell_1808/MatMul_1/ReadVariableOpҐwhileD
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
$lstm_cell_1808/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1808_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1808/MatMul/ReadVariableOp≥
lstm_cell_1808/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/MatMulЅ
&lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1808_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1808/MatMul_1/ReadVariableOpѓ
lstm_cell_1808/MatMul_1MatMulzeros:output:0.lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/MatMul_1®
lstm_cell_1808/addAddV2lstm_cell_1808/MatMul:product:0!lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/addЇ
%lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1808_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1808/BiasAdd/ReadVariableOpµ
lstm_cell_1808/BiasAddBiasAddlstm_cell_1808/add:z:0-lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/BiasAddВ
lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1808/split/split_dimы
lstm_cell_1808/splitSplit'lstm_cell_1808/split/split_dim:output:0lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1808/splitМ
lstm_cell_1808/SigmoidSigmoidlstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/SigmoidР
lstm_cell_1808/Sigmoid_1Sigmoidlstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Sigmoid_1С
lstm_cell_1808/mulMullstm_cell_1808/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/mulГ
lstm_cell_1808/ReluRelulstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Relu§
lstm_cell_1808/mul_1Mullstm_cell_1808/Sigmoid:y:0!lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/mul_1Щ
lstm_cell_1808/add_1AddV2lstm_cell_1808/mul:z:0lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/add_1Р
lstm_cell_1808/Sigmoid_2Sigmoidlstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Sigmoid_2В
lstm_cell_1808/Relu_1Relulstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Relu_1®
lstm_cell_1808/mul_2Mullstm_cell_1808/Sigmoid_2:y:0#lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1808_matmul_readvariableop_resource/lstm_cell_1808_matmul_1_readvariableop_resource.lstm_cell_1808_biasadd_readvariableop_resource*
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
while_body_60939314*
condR
while_cond_60939313*K
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
NoOpNoOp&^lstm_cell_1808/BiasAdd/ReadVariableOp%^lstm_cell_1808/MatMul/ReadVariableOp'^lstm_cell_1808/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1808/BiasAdd/ReadVariableOp%lstm_cell_1808/BiasAdd/ReadVariableOp2L
$lstm_cell_1808/MatMul/ReadVariableOp$lstm_cell_1808/MatMul/ReadVariableOp2P
&lstm_cell_1808/MatMul_1/ReadVariableOp&lstm_cell_1808/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ѕ@
д
while_body_60939314
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1808_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1808_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1808_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1808_matmul_readvariableop_resource:	»H
5while_lstm_cell_1808_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1808_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1808/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1808/MatMul/ReadVariableOpҐ,while/lstm_cell_1808/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1808_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1808/MatMul/ReadVariableOpЁ
while/lstm_cell_1808/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/MatMul’
,while/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1808_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1808/MatMul_1/ReadVariableOp∆
while/lstm_cell_1808/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/MatMul_1ј
while/lstm_cell_1808/addAddV2%while/lstm_cell_1808/MatMul:product:0'while/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/addќ
+while/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1808_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1808/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1808/BiasAddBiasAddwhile/lstm_cell_1808/add:z:03while/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/BiasAddО
$while/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1808/split/split_dimУ
while/lstm_cell_1808/splitSplit-while/lstm_cell_1808/split/split_dim:output:0%while/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1808/splitЮ
while/lstm_cell_1808/SigmoidSigmoid#while/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/SigmoidҐ
while/lstm_cell_1808/Sigmoid_1Sigmoid#while/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1808/Sigmoid_1¶
while/lstm_cell_1808/mulMul"while/lstm_cell_1808/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/mulХ
while/lstm_cell_1808/ReluRelu#while/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/ReluЉ
while/lstm_cell_1808/mul_1Mul while/lstm_cell_1808/Sigmoid:y:0'while/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/mul_1±
while/lstm_cell_1808/add_1AddV2while/lstm_cell_1808/mul:z:0while/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/add_1Ґ
while/lstm_cell_1808/Sigmoid_2Sigmoid#while/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1808/Sigmoid_2Ф
while/lstm_cell_1808/Relu_1Reluwhile/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/Relu_1ј
while/lstm_cell_1808/mul_2Mul"while/lstm_cell_1808/Sigmoid_2:y:0)while/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1808/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1808/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1808/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1808/BiasAdd/ReadVariableOp+^while/lstm_cell_1808/MatMul/ReadVariableOp-^while/lstm_cell_1808/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1808_biasadd_readvariableop_resource6while_lstm_cell_1808_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1808_matmul_1_readvariableop_resource7while_lstm_cell_1808_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1808_matmul_readvariableop_resource5while_lstm_cell_1808_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1808/BiasAdd/ReadVariableOp+while/lstm_cell_1808/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1808/MatMul/ReadVariableOp*while/lstm_cell_1808/MatMul/ReadVariableOp2\
,while/lstm_cell_1808/MatMul_1/ReadVariableOp,while/lstm_cell_1808/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_body_60933927
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_lstm_cell_1807_60933951_0:	»2
while_lstm_cell_1807_60933953_0:	2».
while_lstm_cell_1807_60933955_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_lstm_cell_1807_60933951:	»0
while_lstm_cell_1807_60933953:	2»,
while_lstm_cell_1807_60933955:	»ИҐ,while/lstm_cell_1807/StatefulPartitionedCall√
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
,while/lstm_cell_1807/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1807_60933951_0while_lstm_cell_1807_60933953_0while_lstm_cell_1807_60933955_0*
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
L__inference_lstm_cell_1807_layer_call_and_return_conditional_losses_609338492.
,while/lstm_cell_1807/StatefulPartitionedCallщ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder5while/lstm_cell_1807/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity5while/lstm_cell_1807/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4¶
while/Identity_5Identity5while/lstm_cell_1807/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5Й

while/NoOpNoOp-^while/lstm_cell_1807/StatefulPartitionedCall*"
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
while_lstm_cell_1807_60933951while_lstm_cell_1807_60933951_0"@
while_lstm_cell_1807_60933953while_lstm_cell_1807_60933953_0"@
while_lstm_cell_1807_60933955while_lstm_cell_1807_60933955_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2\
,while/lstm_cell_1807/StatefulPartitionedCall,while/lstm_cell_1807/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
%backward_lstm_602_while_body_60936407@
<backward_lstm_602_while_backward_lstm_602_while_loop_counterF
Bbackward_lstm_602_while_backward_lstm_602_while_maximum_iterations'
#backward_lstm_602_while_placeholder)
%backward_lstm_602_while_placeholder_1)
%backward_lstm_602_while_placeholder_2)
%backward_lstm_602_while_placeholder_3)
%backward_lstm_602_while_placeholder_4?
;backward_lstm_602_while_backward_lstm_602_strided_slice_1_0{
wbackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_602_while_less_backward_lstm_602_sub_1_0Z
Gbackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0:	»\
Ibackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0:	2»W
Hbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0:	»$
 backward_lstm_602_while_identity&
"backward_lstm_602_while_identity_1&
"backward_lstm_602_while_identity_2&
"backward_lstm_602_while_identity_3&
"backward_lstm_602_while_identity_4&
"backward_lstm_602_while_identity_5&
"backward_lstm_602_while_identity_6=
9backward_lstm_602_while_backward_lstm_602_strided_slice_1y
ubackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_602_while_less_backward_lstm_602_sub_1X
Ebackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource:	»Z
Gbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource:	2»U
Fbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource:	»ИҐ=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpҐ<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOpҐ>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpз
Ibackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2K
Ibackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeњ
;backward_lstm_602/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_602_while_placeholderRbackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02=
;backward_lstm_602/while/TensorArrayV2Read/TensorListGetItemѕ
backward_lstm_602/while/LessLess6backward_lstm_602_while_less_backward_lstm_602_sub_1_0#backward_lstm_602_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_602/while/LessЕ
<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOpGbackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02>
<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp•
-backward_lstm_602/while/lstm_cell_1808/MatMulMatMulBbackward_lstm_602/while/TensorArrayV2Read/TensorListGetItem:item:0Dbackward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-backward_lstm_602/while/lstm_cell_1808/MatMulЛ
>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOpIbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02@
>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOpО
/backward_lstm_602/while/lstm_cell_1808/MatMul_1MatMul%backward_lstm_602_while_placeholder_3Fbackward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»21
/backward_lstm_602/while/lstm_cell_1808/MatMul_1И
*backward_lstm_602/while/lstm_cell_1808/addAddV27backward_lstm_602/while/lstm_cell_1808/MatMul:product:09backward_lstm_602/while/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2,
*backward_lstm_602/while/lstm_cell_1808/addД
=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOpHbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02?
=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOpХ
.backward_lstm_602/while/lstm_cell_1808/BiasAddBiasAdd.backward_lstm_602/while/lstm_cell_1808/add:z:0Ebackward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.backward_lstm_602/while/lstm_cell_1808/BiasAdd≤
6backward_lstm_602/while/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6backward_lstm_602/while/lstm_cell_1808/split/split_dimџ
,backward_lstm_602/while/lstm_cell_1808/splitSplit?backward_lstm_602/while/lstm_cell_1808/split/split_dim:output:07backward_lstm_602/while/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2.
,backward_lstm_602/while/lstm_cell_1808/split‘
.backward_lstm_602/while/lstm_cell_1808/SigmoidSigmoid5backward_lstm_602/while/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€220
.backward_lstm_602/while/lstm_cell_1808/SigmoidЎ
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_1Sigmoid5backward_lstm_602/while/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_1о
*backward_lstm_602/while/lstm_cell_1808/mulMul4backward_lstm_602/while/lstm_cell_1808/Sigmoid_1:y:0%backward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_602/while/lstm_cell_1808/mulЋ
+backward_lstm_602/while/lstm_cell_1808/ReluRelu5backward_lstm_602/while/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22-
+backward_lstm_602/while/lstm_cell_1808/ReluД
,backward_lstm_602/while/lstm_cell_1808/mul_1Mul2backward_lstm_602/while/lstm_cell_1808/Sigmoid:y:09backward_lstm_602/while/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_602/while/lstm_cell_1808/mul_1щ
,backward_lstm_602/while/lstm_cell_1808/add_1AddV2.backward_lstm_602/while/lstm_cell_1808/mul:z:00backward_lstm_602/while/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_602/while/lstm_cell_1808/add_1Ў
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_2Sigmoid5backward_lstm_602/while/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_602/while/lstm_cell_1808/Sigmoid_2 
-backward_lstm_602/while/lstm_cell_1808/Relu_1Relu0backward_lstm_602/while/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-backward_lstm_602/while/lstm_cell_1808/Relu_1И
,backward_lstm_602/while/lstm_cell_1808/mul_2Mul4backward_lstm_602/while/lstm_cell_1808/Sigmoid_2:y:0;backward_lstm_602/while/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_602/while/lstm_cell_1808/mul_2ч
backward_lstm_602/while/SelectSelect backward_lstm_602/while/Less:z:00backward_lstm_602/while/lstm_cell_1808/mul_2:z:0%backward_lstm_602_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22 
backward_lstm_602/while/Selectы
 backward_lstm_602/while/Select_1Select backward_lstm_602/while/Less:z:00backward_lstm_602/while/lstm_cell_1808/mul_2:z:0%backward_lstm_602_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_602/while/Select_1ы
 backward_lstm_602/while/Select_2Select backward_lstm_602/while/Less:z:00backward_lstm_602/while/lstm_cell_1808/add_1:z:0%backward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_602/while/Select_2≥
<backward_lstm_602/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_602_while_placeholder_1#backward_lstm_602_while_placeholder'backward_lstm_602/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_602/while/TensorArrayV2Write/TensorListSetItemА
backward_lstm_602/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_602/while/add/y±
backward_lstm_602/while/addAddV2#backward_lstm_602_while_placeholder&backward_lstm_602/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/while/addД
backward_lstm_602/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_602/while/add_1/y–
backward_lstm_602/while/add_1AddV2<backward_lstm_602_while_backward_lstm_602_while_loop_counter(backward_lstm_602/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/while/add_1≥
 backward_lstm_602/while/IdentityIdentity!backward_lstm_602/while/add_1:z:0^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_602/while/IdentityЎ
"backward_lstm_602/while/Identity_1IdentityBbackward_lstm_602_while_backward_lstm_602_while_maximum_iterations^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_602/while/Identity_1µ
"backward_lstm_602/while/Identity_2Identitybackward_lstm_602/while/add:z:0^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_602/while/Identity_2в
"backward_lstm_602/while/Identity_3IdentityLbackward_lstm_602/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_602/while/Identity_3ќ
"backward_lstm_602/while/Identity_4Identity'backward_lstm_602/while/Select:output:0^backward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_602/while/Identity_4–
"backward_lstm_602/while/Identity_5Identity)backward_lstm_602/while/Select_1:output:0^backward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_602/while/Identity_5–
"backward_lstm_602/while/Identity_6Identity)backward_lstm_602/while/Select_2:output:0^backward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_602/while/Identity_6Њ
backward_lstm_602/while/NoOpNoOp>^backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp=^backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp?^backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_602/while/NoOp"x
9backward_lstm_602_while_backward_lstm_602_strided_slice_1;backward_lstm_602_while_backward_lstm_602_strided_slice_1_0"M
 backward_lstm_602_while_identity)backward_lstm_602/while/Identity:output:0"Q
"backward_lstm_602_while_identity_1+backward_lstm_602/while/Identity_1:output:0"Q
"backward_lstm_602_while_identity_2+backward_lstm_602/while/Identity_2:output:0"Q
"backward_lstm_602_while_identity_3+backward_lstm_602/while/Identity_3:output:0"Q
"backward_lstm_602_while_identity_4+backward_lstm_602/while/Identity_4:output:0"Q
"backward_lstm_602_while_identity_5+backward_lstm_602/while/Identity_5:output:0"Q
"backward_lstm_602_while_identity_6+backward_lstm_602/while/Identity_6:output:0"n
4backward_lstm_602_while_less_backward_lstm_602_sub_16backward_lstm_602_while_less_backward_lstm_602_sub_1_0"Т
Fbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resourceHbackward_lstm_602_while_lstm_cell_1808_biasadd_readvariableop_resource_0"Ф
Gbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resourceIbackward_lstm_602_while_lstm_cell_1808_matmul_1_readvariableop_resource_0"Р
Ebackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resourceGbackward_lstm_602_while_lstm_cell_1808_matmul_readvariableop_resource_0"р
ubackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2~
=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp=backward_lstm_602/while/lstm_cell_1808/BiasAdd/ReadVariableOp2|
<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp<backward_lstm_602/while/lstm_cell_1808/MatMul/ReadVariableOp2А
>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp>backward_lstm_602/while/lstm_cell_1808/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
ѕ@
д
while_body_60938507
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1807_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1807_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1807_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1807_matmul_readvariableop_resource:	»H
5while_lstm_cell_1807_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1807_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1807/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1807/MatMul/ReadVariableOpҐ,while/lstm_cell_1807/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1807_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1807/MatMul/ReadVariableOpЁ
while/lstm_cell_1807/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/MatMul’
,while/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1807_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1807/MatMul_1/ReadVariableOp∆
while/lstm_cell_1807/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/MatMul_1ј
while/lstm_cell_1807/addAddV2%while/lstm_cell_1807/MatMul:product:0'while/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/addќ
+while/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1807_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1807/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1807/BiasAddBiasAddwhile/lstm_cell_1807/add:z:03while/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/BiasAddО
$while/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1807/split/split_dimУ
while/lstm_cell_1807/splitSplit-while/lstm_cell_1807/split/split_dim:output:0%while/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1807/splitЮ
while/lstm_cell_1807/SigmoidSigmoid#while/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/SigmoidҐ
while/lstm_cell_1807/Sigmoid_1Sigmoid#while/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1807/Sigmoid_1¶
while/lstm_cell_1807/mulMul"while/lstm_cell_1807/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/mulХ
while/lstm_cell_1807/ReluRelu#while/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/ReluЉ
while/lstm_cell_1807/mul_1Mul while/lstm_cell_1807/Sigmoid:y:0'while/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/mul_1±
while/lstm_cell_1807/add_1AddV2while/lstm_cell_1807/mul:z:0while/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/add_1Ґ
while/lstm_cell_1807/Sigmoid_2Sigmoid#while/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1807/Sigmoid_2Ф
while/lstm_cell_1807/Relu_1Reluwhile/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/Relu_1ј
while/lstm_cell_1807/mul_2Mul"while/lstm_cell_1807/Sigmoid_2:y:0)while/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1807/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1807/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1807/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1807/BiasAdd/ReadVariableOp+^while/lstm_cell_1807/MatMul/ReadVariableOp-^while/lstm_cell_1807/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1807_biasadd_readvariableop_resource6while_lstm_cell_1807_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1807_matmul_1_readvariableop_resource7while_lstm_cell_1807_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1807_matmul_readvariableop_resource5while_lstm_cell_1807_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1807/BiasAdd/ReadVariableOp+while/lstm_cell_1807/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1807/MatMul/ReadVariableOp*while/lstm_cell_1807/MatMul/ReadVariableOp2\
,while/lstm_cell_1807/MatMul_1/ReadVariableOp,while/lstm_cell_1807/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
%backward_lstm_602_while_cond_60937976@
<backward_lstm_602_while_backward_lstm_602_while_loop_counterF
Bbackward_lstm_602_while_backward_lstm_602_while_maximum_iterations'
#backward_lstm_602_while_placeholder)
%backward_lstm_602_while_placeholder_1)
%backward_lstm_602_while_placeholder_2)
%backward_lstm_602_while_placeholder_3)
%backward_lstm_602_while_placeholder_4B
>backward_lstm_602_while_less_backward_lstm_602_strided_slice_1Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60937976___redundant_placeholder0Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60937976___redundant_placeholder1Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60937976___redundant_placeholder2Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60937976___redundant_placeholder3Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60937976___redundant_placeholder4$
 backward_lstm_602_while_identity
 
backward_lstm_602/while/LessLess#backward_lstm_602_while_placeholder>backward_lstm_602_while_less_backward_lstm_602_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_602/while/LessУ
 backward_lstm_602/while/IdentityIdentity backward_lstm_602/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_602/while/Identity"M
 backward_lstm_602_while_identity)backward_lstm_602/while/Identity:output:0*(
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
÷
√
4__inference_backward_lstm_602_layer_call_fn_60938753
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
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_609344182
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
∞Њ
ы
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_60937716

inputs
inputs_1	Q
>forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource:	»S
@forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource:	2»N
?forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource:	»R
?backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resource:	»T
Abackward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource:	2»O
@backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource:	»
identityИҐ7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpҐ6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpҐ8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpҐbackward_lstm_602/whileҐ6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpҐ5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpҐ7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpҐforward_lstm_602/whileЧ
%forward_lstm_602/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_602/RaggedToTensor/zerosЩ
%forward_lstm_602/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2'
%forward_lstm_602/RaggedToTensor/ConstЩ
4forward_lstm_602/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_602/RaggedToTensor/Const:output:0inputs.forward_lstm_602/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_602/RaggedToTensor/RaggedTensorToTensorƒ
;forward_lstm_602/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack»
=forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1»
=forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2©
5forward_lstm_602/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_602/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask27
5forward_lstm_602/RaggedNestedRowLengths/strided_slice»
=forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack’
?forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2A
?forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1ћ
?forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2µ
7forward_lstm_602/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask29
7forward_lstm_602/RaggedNestedRowLengths/strided_slice_1С
+forward_lstm_602/RaggedNestedRowLengths/subSub>forward_lstm_602/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_602/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2-
+forward_lstm_602/RaggedNestedRowLengths/sub§
forward_lstm_602/CastCast/forward_lstm_602/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
forward_lstm_602/CastЭ
forward_lstm_602/ShapeShape=forward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_602/ShapeЦ
$forward_lstm_602/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_602/strided_slice/stackЪ
&forward_lstm_602/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_602/strided_slice/stack_1Ъ
&forward_lstm_602/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_602/strided_slice/stack_2»
forward_lstm_602/strided_sliceStridedSliceforward_lstm_602/Shape:output:0-forward_lstm_602/strided_slice/stack:output:0/forward_lstm_602/strided_slice/stack_1:output:0/forward_lstm_602/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_602/strided_slice~
forward_lstm_602/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_602/zeros/mul/y∞
forward_lstm_602/zeros/mulMul'forward_lstm_602/strided_slice:output:0%forward_lstm_602/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros/mulБ
forward_lstm_602/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
forward_lstm_602/zeros/Less/yЂ
forward_lstm_602/zeros/LessLessforward_lstm_602/zeros/mul:z:0&forward_lstm_602/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros/LessД
forward_lstm_602/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_602/zeros/packed/1«
forward_lstm_602/zeros/packedPack'forward_lstm_602/strided_slice:output:0(forward_lstm_602/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_602/zeros/packedЕ
forward_lstm_602/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_602/zeros/Constє
forward_lstm_602/zerosFill&forward_lstm_602/zeros/packed:output:0%forward_lstm_602/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/zerosВ
forward_lstm_602/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_602/zeros_1/mul/yґ
forward_lstm_602/zeros_1/mulMul'forward_lstm_602/strided_slice:output:0'forward_lstm_602/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros_1/mulЕ
forward_lstm_602/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2!
forward_lstm_602/zeros_1/Less/y≥
forward_lstm_602/zeros_1/LessLess forward_lstm_602/zeros_1/mul:z:0(forward_lstm_602/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros_1/LessИ
!forward_lstm_602/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_602/zeros_1/packed/1Ќ
forward_lstm_602/zeros_1/packedPack'forward_lstm_602/strided_slice:output:0*forward_lstm_602/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_602/zeros_1/packedЙ
forward_lstm_602/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_602/zeros_1/ConstЅ
forward_lstm_602/zeros_1Fill(forward_lstm_602/zeros_1/packed:output:0'forward_lstm_602/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/zeros_1Ч
forward_lstm_602/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_602/transpose/permн
forward_lstm_602/transpose	Transpose=forward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_602/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
forward_lstm_602/transposeВ
forward_lstm_602/Shape_1Shapeforward_lstm_602/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_602/Shape_1Ъ
&forward_lstm_602/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_602/strided_slice_1/stackЮ
(forward_lstm_602/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_1/stack_1Ю
(forward_lstm_602/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_1/stack_2‘
 forward_lstm_602/strided_slice_1StridedSlice!forward_lstm_602/Shape_1:output:0/forward_lstm_602/strided_slice_1/stack:output:01forward_lstm_602/strided_slice_1/stack_1:output:01forward_lstm_602/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_602/strided_slice_1І
,forward_lstm_602/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,forward_lstm_602/TensorArrayV2/element_shapeц
forward_lstm_602/TensorArrayV2TensorListReserve5forward_lstm_602/TensorArrayV2/element_shape:output:0)forward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_602/TensorArrayV2б
Fforward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2H
Fforward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeЉ
8forward_lstm_602/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_602/transpose:y:0Oforward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_602/TensorArrayUnstack/TensorListFromTensorЪ
&forward_lstm_602/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_602/strided_slice_2/stackЮ
(forward_lstm_602/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_2/stack_1Ю
(forward_lstm_602/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_2/stack_2в
 forward_lstm_602/strided_slice_2StridedSliceforward_lstm_602/transpose:y:0/forward_lstm_602/strided_slice_2/stack:output:01forward_lstm_602/strided_slice_2/stack_1:output:01forward_lstm_602/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2"
 forward_lstm_602/strided_slice_2о
5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOp>forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype027
5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpч
&forward_lstm_602/lstm_cell_1807/MatMulMatMul)forward_lstm_602/strided_slice_2:output:0=forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2(
&forward_lstm_602/lstm_cell_1807/MatMulф
7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOp@forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype029
7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpу
(forward_lstm_602/lstm_cell_1807/MatMul_1MatMulforward_lstm_602/zeros:output:0?forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(forward_lstm_602/lstm_cell_1807/MatMul_1м
#forward_lstm_602/lstm_cell_1807/addAddV20forward_lstm_602/lstm_cell_1807/MatMul:product:02forward_lstm_602/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2%
#forward_lstm_602/lstm_cell_1807/addн
6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOp?forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype028
6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpщ
'forward_lstm_602/lstm_cell_1807/BiasAddBiasAdd'forward_lstm_602/lstm_cell_1807/add:z:0>forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'forward_lstm_602/lstm_cell_1807/BiasAdd§
/forward_lstm_602/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/forward_lstm_602/lstm_cell_1807/split/split_dimњ
%forward_lstm_602/lstm_cell_1807/splitSplit8forward_lstm_602/lstm_cell_1807/split/split_dim:output:00forward_lstm_602/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2'
%forward_lstm_602/lstm_cell_1807/splitњ
'forward_lstm_602/lstm_cell_1807/SigmoidSigmoid.forward_lstm_602/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'forward_lstm_602/lstm_cell_1807/Sigmoid√
)forward_lstm_602/lstm_cell_1807/Sigmoid_1Sigmoid.forward_lstm_602/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_602/lstm_cell_1807/Sigmoid_1’
#forward_lstm_602/lstm_cell_1807/mulMul-forward_lstm_602/lstm_cell_1807/Sigmoid_1:y:0!forward_lstm_602/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22%
#forward_lstm_602/lstm_cell_1807/mulґ
$forward_lstm_602/lstm_cell_1807/ReluRelu.forward_lstm_602/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22&
$forward_lstm_602/lstm_cell_1807/Reluи
%forward_lstm_602/lstm_cell_1807/mul_1Mul+forward_lstm_602/lstm_cell_1807/Sigmoid:y:02forward_lstm_602/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_602/lstm_cell_1807/mul_1Ё
%forward_lstm_602/lstm_cell_1807/add_1AddV2'forward_lstm_602/lstm_cell_1807/mul:z:0)forward_lstm_602/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_602/lstm_cell_1807/add_1√
)forward_lstm_602/lstm_cell_1807/Sigmoid_2Sigmoid.forward_lstm_602/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_602/lstm_cell_1807/Sigmoid_2µ
&forward_lstm_602/lstm_cell_1807/Relu_1Relu)forward_lstm_602/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&forward_lstm_602/lstm_cell_1807/Relu_1м
%forward_lstm_602/lstm_cell_1807/mul_2Mul-forward_lstm_602/lstm_cell_1807/Sigmoid_2:y:04forward_lstm_602/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_602/lstm_cell_1807/mul_2±
.forward_lstm_602/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   20
.forward_lstm_602/TensorArrayV2_1/element_shapeь
 forward_lstm_602/TensorArrayV2_1TensorListReserve7forward_lstm_602/TensorArrayV2_1/element_shape:output:0)forward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_602/TensorArrayV2_1p
forward_lstm_602/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_602/time§
forward_lstm_602/zeros_like	ZerosLike)forward_lstm_602/lstm_cell_1807/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/zeros_like°
)forward_lstm_602/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2+
)forward_lstm_602/while/maximum_iterationsМ
#forward_lstm_602/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_602/while/loop_counterЦ	
forward_lstm_602/whileWhile,forward_lstm_602/while/loop_counter:output:02forward_lstm_602/while/maximum_iterations:output:0forward_lstm_602/time:output:0)forward_lstm_602/TensorArrayV2_1:handle:0forward_lstm_602/zeros_like:y:0forward_lstm_602/zeros:output:0!forward_lstm_602/zeros_1:output:0)forward_lstm_602/strided_slice_1:output:0Hforward_lstm_602/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_602/Cast:y:0>forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource@forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource?forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource*
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
$forward_lstm_602_while_body_60937440*0
cond(R&
$forward_lstm_602_while_cond_60937439*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
forward_lstm_602/while„
Aforward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2C
Aforward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeµ
3forward_lstm_602/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_602/while:output:3Jforward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype025
3forward_lstm_602/TensorArrayV2Stack/TensorListStack£
&forward_lstm_602/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2(
&forward_lstm_602/strided_slice_3/stackЮ
(forward_lstm_602/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_602/strided_slice_3/stack_1Ю
(forward_lstm_602/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_3/stack_2А
 forward_lstm_602/strided_slice_3StridedSlice<forward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_602/strided_slice_3/stack:output:01forward_lstm_602/strided_slice_3/stack_1:output:01forward_lstm_602/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2"
 forward_lstm_602/strided_slice_3Ы
!forward_lstm_602/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_602/transpose_1/permт
forward_lstm_602/transpose_1	Transpose<forward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_602/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
forward_lstm_602/transpose_1И
forward_lstm_602/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_602/runtimeЩ
&backward_lstm_602/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_602/RaggedToTensor/zerosЫ
&backward_lstm_602/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2(
&backward_lstm_602/RaggedToTensor/ConstЭ
5backward_lstm_602/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_602/RaggedToTensor/Const:output:0inputs/backward_lstm_602/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_602/RaggedToTensor/RaggedTensorToTensor∆
<backward_lstm_602/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack 
>backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1 
>backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2Ѓ
6backward_lstm_602/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_602/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask28
6backward_lstm_602/RaggedNestedRowLengths/strided_slice 
>backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack„
@backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2B
@backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1ќ
@backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2Ї
8backward_lstm_602/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask2:
8backward_lstm_602/RaggedNestedRowLengths/strided_slice_1Х
,backward_lstm_602/RaggedNestedRowLengths/subSub?backward_lstm_602/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_602/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2.
,backward_lstm_602/RaggedNestedRowLengths/subІ
backward_lstm_602/CastCast0backward_lstm_602/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
backward_lstm_602/Cast†
backward_lstm_602/ShapeShape>backward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_602/ShapeШ
%backward_lstm_602/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_602/strided_slice/stackЬ
'backward_lstm_602/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_602/strided_slice/stack_1Ь
'backward_lstm_602/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_602/strided_slice/stack_2ќ
backward_lstm_602/strided_sliceStridedSlice backward_lstm_602/Shape:output:0.backward_lstm_602/strided_slice/stack:output:00backward_lstm_602/strided_slice/stack_1:output:00backward_lstm_602/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_602/strided_sliceА
backward_lstm_602/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_602/zeros/mul/yі
backward_lstm_602/zeros/mulMul(backward_lstm_602/strided_slice:output:0&backward_lstm_602/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/zeros/mulГ
backward_lstm_602/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
backward_lstm_602/zeros/Less/yѓ
backward_lstm_602/zeros/LessLessbackward_lstm_602/zeros/mul:z:0'backward_lstm_602/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/zeros/LessЖ
 backward_lstm_602/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_602/zeros/packed/1Ћ
backward_lstm_602/zeros/packedPack(backward_lstm_602/strided_slice:output:0)backward_lstm_602/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_602/zeros/packedЗ
backward_lstm_602/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_602/zeros/Constљ
backward_lstm_602/zerosFill'backward_lstm_602/zeros/packed:output:0&backward_lstm_602/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_602/zerosД
backward_lstm_602/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_602/zeros_1/mul/yЇ
backward_lstm_602/zeros_1/mulMul(backward_lstm_602/strided_slice:output:0(backward_lstm_602/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/zeros_1/mulЗ
 backward_lstm_602/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 backward_lstm_602/zeros_1/Less/yЈ
backward_lstm_602/zeros_1/LessLess!backward_lstm_602/zeros_1/mul:z:0)backward_lstm_602/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_602/zeros_1/LessК
"backward_lstm_602/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_602/zeros_1/packed/1—
 backward_lstm_602/zeros_1/packedPack(backward_lstm_602/strided_slice:output:0+backward_lstm_602/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_602/zeros_1/packedЛ
backward_lstm_602/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_602/zeros_1/Const≈
backward_lstm_602/zeros_1Fill)backward_lstm_602/zeros_1/packed:output:0(backward_lstm_602/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_602/zeros_1Щ
 backward_lstm_602/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_602/transpose/permс
backward_lstm_602/transpose	Transpose>backward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_602/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_602/transposeЕ
backward_lstm_602/Shape_1Shapebackward_lstm_602/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_602/Shape_1Ь
'backward_lstm_602/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_602/strided_slice_1/stack†
)backward_lstm_602/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_1/stack_1†
)backward_lstm_602/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_1/stack_2Џ
!backward_lstm_602/strided_slice_1StridedSlice"backward_lstm_602/Shape_1:output:00backward_lstm_602/strided_slice_1/stack:output:02backward_lstm_602/strided_slice_1/stack_1:output:02backward_lstm_602/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_602/strided_slice_1©
-backward_lstm_602/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-backward_lstm_602/TensorArrayV2/element_shapeъ
backward_lstm_602/TensorArrayV2TensorListReserve6backward_lstm_602/TensorArrayV2/element_shape:output:0*backward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_602/TensorArrayV2О
 backward_lstm_602/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_602/ReverseV2/axis“
backward_lstm_602/ReverseV2	ReverseV2backward_lstm_602/transpose:y:0)backward_lstm_602/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_602/ReverseV2г
Gbackward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2I
Gbackward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape≈
9backward_lstm_602/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_602/ReverseV2:output:0Pbackward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_602/TensorArrayUnstack/TensorListFromTensorЬ
'backward_lstm_602/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_602/strided_slice_2/stack†
)backward_lstm_602/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_2/stack_1†
)backward_lstm_602/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_2/stack_2и
!backward_lstm_602/strided_slice_2StridedSlicebackward_lstm_602/transpose:y:00backward_lstm_602/strided_slice_2/stack:output:02backward_lstm_602/strided_slice_2/stack_1:output:02backward_lstm_602/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2#
!backward_lstm_602/strided_slice_2с
6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOp?backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype028
6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpы
'backward_lstm_602/lstm_cell_1808/MatMulMatMul*backward_lstm_602/strided_slice_2:output:0>backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'backward_lstm_602/lstm_cell_1808/MatMulч
8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOpAbackward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02:
8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpч
)backward_lstm_602/lstm_cell_1808/MatMul_1MatMul backward_lstm_602/zeros:output:0@backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)backward_lstm_602/lstm_cell_1808/MatMul_1р
$backward_lstm_602/lstm_cell_1808/addAddV21backward_lstm_602/lstm_cell_1808/MatMul:product:03backward_lstm_602/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2&
$backward_lstm_602/lstm_cell_1808/addр
7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOp@backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype029
7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpэ
(backward_lstm_602/lstm_cell_1808/BiasAddBiasAdd(backward_lstm_602/lstm_cell_1808/add:z:0?backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(backward_lstm_602/lstm_cell_1808/BiasAdd¶
0backward_lstm_602/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0backward_lstm_602/lstm_cell_1808/split/split_dim√
&backward_lstm_602/lstm_cell_1808/splitSplit9backward_lstm_602/lstm_cell_1808/split/split_dim:output:01backward_lstm_602/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2(
&backward_lstm_602/lstm_cell_1808/split¬
(backward_lstm_602/lstm_cell_1808/SigmoidSigmoid/backward_lstm_602/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(backward_lstm_602/lstm_cell_1808/Sigmoid∆
*backward_lstm_602/lstm_cell_1808/Sigmoid_1Sigmoid/backward_lstm_602/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_602/lstm_cell_1808/Sigmoid_1ў
$backward_lstm_602/lstm_cell_1808/mulMul.backward_lstm_602/lstm_cell_1808/Sigmoid_1:y:0"backward_lstm_602/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$backward_lstm_602/lstm_cell_1808/mulє
%backward_lstm_602/lstm_cell_1808/ReluRelu/backward_lstm_602/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22'
%backward_lstm_602/lstm_cell_1808/Reluм
&backward_lstm_602/lstm_cell_1808/mul_1Mul,backward_lstm_602/lstm_cell_1808/Sigmoid:y:03backward_lstm_602/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_602/lstm_cell_1808/mul_1б
&backward_lstm_602/lstm_cell_1808/add_1AddV2(backward_lstm_602/lstm_cell_1808/mul:z:0*backward_lstm_602/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_602/lstm_cell_1808/add_1∆
*backward_lstm_602/lstm_cell_1808/Sigmoid_2Sigmoid/backward_lstm_602/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_602/lstm_cell_1808/Sigmoid_2Є
'backward_lstm_602/lstm_cell_1808/Relu_1Relu*backward_lstm_602/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'backward_lstm_602/lstm_cell_1808/Relu_1р
&backward_lstm_602/lstm_cell_1808/mul_2Mul.backward_lstm_602/lstm_cell_1808/Sigmoid_2:y:05backward_lstm_602/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_602/lstm_cell_1808/mul_2≥
/backward_lstm_602/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   21
/backward_lstm_602/TensorArrayV2_1/element_shapeА
!backward_lstm_602/TensorArrayV2_1TensorListReserve8backward_lstm_602/TensorArrayV2_1/element_shape:output:0*backward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_602/TensorArrayV2_1r
backward_lstm_602/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_602/timeФ
'backward_lstm_602/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_602/Max/reduction_indices§
backward_lstm_602/MaxMaxbackward_lstm_602/Cast:y:00backward_lstm_602/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/Maxt
backward_lstm_602/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_602/sub/yШ
backward_lstm_602/subSubbackward_lstm_602/Max:output:0 backward_lstm_602/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/subЮ
backward_lstm_602/Sub_1Subbackward_lstm_602/sub:z:0backward_lstm_602/Cast:y:0*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_602/Sub_1І
backward_lstm_602/zeros_like	ZerosLike*backward_lstm_602/lstm_cell_1808/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_602/zeros_like£
*backward_lstm_602/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2,
*backward_lstm_602/while/maximum_iterationsО
$backward_lstm_602/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_602/while/loop_counter®	
backward_lstm_602/whileWhile-backward_lstm_602/while/loop_counter:output:03backward_lstm_602/while/maximum_iterations:output:0backward_lstm_602/time:output:0*backward_lstm_602/TensorArrayV2_1:handle:0 backward_lstm_602/zeros_like:y:0 backward_lstm_602/zeros:output:0"backward_lstm_602/zeros_1:output:0*backward_lstm_602/strided_slice_1:output:0Ibackward_lstm_602/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_602/Sub_1:z:0?backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resourceAbackward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource@backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource*
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
%backward_lstm_602_while_body_60937619*1
cond)R'
%backward_lstm_602_while_cond_60937618*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
backward_lstm_602/whileў
Bbackward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2D
Bbackward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeє
4backward_lstm_602/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_602/while:output:3Kbackward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype026
4backward_lstm_602/TensorArrayV2Stack/TensorListStack•
'backward_lstm_602/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2)
'backward_lstm_602/strided_slice_3/stack†
)backward_lstm_602/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_602/strided_slice_3/stack_1†
)backward_lstm_602/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_3/stack_2Ж
!backward_lstm_602/strided_slice_3StridedSlice=backward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_602/strided_slice_3/stack:output:02backward_lstm_602/strided_slice_3/stack_1:output:02backward_lstm_602/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2#
!backward_lstm_602/strided_slice_3Э
"backward_lstm_602/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_602/transpose_1/permц
backward_lstm_602/transpose_1	Transpose=backward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_602/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
backward_lstm_602/transpose_1К
backward_lstm_602/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_602/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisƒ
concatConcatV2)forward_lstm_602/strided_slice_3:output:0*backward_lstm_602/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp8^backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp7^backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp9^backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp^backward_lstm_602/while7^forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp6^forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp8^forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp^forward_lstm_602/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€:€€€€€€€€€: : : : : : 2r
7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp2p
6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp2t
8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp22
backward_lstm_602/whilebackward_lstm_602/while2p
6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp2n
5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp2r
7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp20
forward_lstm_602/whileforward_lstm_602/while:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
∆@
д
while_body_60938356
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1807_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1807_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1807_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1807_matmul_readvariableop_resource:	»H
5while_lstm_cell_1807_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1807_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1807/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1807/MatMul/ReadVariableOpҐ,while/lstm_cell_1807/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1807_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1807/MatMul/ReadVariableOpЁ
while/lstm_cell_1807/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/MatMul’
,while/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1807_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1807/MatMul_1/ReadVariableOp∆
while/lstm_cell_1807/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/MatMul_1ј
while/lstm_cell_1807/addAddV2%while/lstm_cell_1807/MatMul:product:0'while/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/addќ
+while/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1807_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1807/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1807/BiasAddBiasAddwhile/lstm_cell_1807/add:z:03while/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/BiasAddО
$while/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1807/split/split_dimУ
while/lstm_cell_1807/splitSplit-while/lstm_cell_1807/split/split_dim:output:0%while/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1807/splitЮ
while/lstm_cell_1807/SigmoidSigmoid#while/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/SigmoidҐ
while/lstm_cell_1807/Sigmoid_1Sigmoid#while/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1807/Sigmoid_1¶
while/lstm_cell_1807/mulMul"while/lstm_cell_1807/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/mulХ
while/lstm_cell_1807/ReluRelu#while/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/ReluЉ
while/lstm_cell_1807/mul_1Mul while/lstm_cell_1807/Sigmoid:y:0'while/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/mul_1±
while/lstm_cell_1807/add_1AddV2while/lstm_cell_1807/mul:z:0while/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/add_1Ґ
while/lstm_cell_1807/Sigmoid_2Sigmoid#while/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1807/Sigmoid_2Ф
while/lstm_cell_1807/Relu_1Reluwhile/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/Relu_1ј
while/lstm_cell_1807/mul_2Mul"while/lstm_cell_1807/Sigmoid_2:y:0)while/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1807/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1807/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1807/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1807/BiasAdd/ReadVariableOp+^while/lstm_cell_1807/MatMul/ReadVariableOp-^while/lstm_cell_1807/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1807_biasadd_readvariableop_resource6while_lstm_cell_1807_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1807_matmul_1_readvariableop_resource7while_lstm_cell_1807_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1807_matmul_readvariableop_resource5while_lstm_cell_1807_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1807/BiasAdd/ReadVariableOp+while/lstm_cell_1807/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1807/MatMul/ReadVariableOp*while/lstm_cell_1807/MatMul/ReadVariableOp2\
,while/lstm_cell_1807/MatMul_1/ReadVariableOp,while/lstm_cell_1807/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_body_60938205
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1807_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1807_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1807_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1807_matmul_readvariableop_resource:	»H
5while_lstm_cell_1807_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1807_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1807/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1807/MatMul/ReadVariableOpҐ,while/lstm_cell_1807/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1807_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1807/MatMul/ReadVariableOpЁ
while/lstm_cell_1807/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/MatMul’
,while/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1807_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1807/MatMul_1/ReadVariableOp∆
while/lstm_cell_1807/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/MatMul_1ј
while/lstm_cell_1807/addAddV2%while/lstm_cell_1807/MatMul:product:0'while/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/addќ
+while/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1807_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1807/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1807/BiasAddBiasAddwhile/lstm_cell_1807/add:z:03while/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/BiasAddО
$while/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1807/split/split_dimУ
while/lstm_cell_1807/splitSplit-while/lstm_cell_1807/split/split_dim:output:0%while/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1807/splitЮ
while/lstm_cell_1807/SigmoidSigmoid#while/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/SigmoidҐ
while/lstm_cell_1807/Sigmoid_1Sigmoid#while/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1807/Sigmoid_1¶
while/lstm_cell_1807/mulMul"while/lstm_cell_1807/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/mulХ
while/lstm_cell_1807/ReluRelu#while/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/ReluЉ
while/lstm_cell_1807/mul_1Mul while/lstm_cell_1807/Sigmoid:y:0'while/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/mul_1±
while/lstm_cell_1807/add_1AddV2while/lstm_cell_1807/mul:z:0while/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/add_1Ґ
while/lstm_cell_1807/Sigmoid_2Sigmoid#while/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1807/Sigmoid_2Ф
while/lstm_cell_1807/Relu_1Reluwhile/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/Relu_1ј
while/lstm_cell_1807/mul_2Mul"while/lstm_cell_1807/Sigmoid_2:y:0)while/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1807/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1807/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1807/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1807/BiasAdd/ReadVariableOp+^while/lstm_cell_1807/MatMul/ReadVariableOp-^while/lstm_cell_1807/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1807_biasadd_readvariableop_resource6while_lstm_cell_1807_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1807_matmul_1_readvariableop_resource7while_lstm_cell_1807_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1807_matmul_readvariableop_resource5while_lstm_cell_1807_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1807/BiasAdd/ReadVariableOp+while/lstm_cell_1807/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1807/MatMul/ReadVariableOp*while/lstm_cell_1807/MatMul/ReadVariableOp2\
,while/lstm_cell_1807/MatMul_1/ReadVariableOp,while/lstm_cell_1807/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Є
£
L__inference_sequential_602_layer_call_and_return_conditional_losses_60936631

inputs
inputs_1	-
bidirectional_602_60936612:	»-
bidirectional_602_60936614:	2»)
bidirectional_602_60936616:	»-
bidirectional_602_60936618:	»-
bidirectional_602_60936620:	2»)
bidirectional_602_60936622:	»$
dense_602_60936625:d 
dense_602_60936627:
identityИҐ)bidirectional_602/StatefulPartitionedCallҐ!dense_602/StatefulPartitionedCall 
)bidirectional_602/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_602_60936612bidirectional_602_60936614bidirectional_602_60936616bidirectional_602_60936618bidirectional_602_60936620bidirectional_602_60936622*
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
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_609360642+
)bidirectional_602/StatefulPartitionedCallЋ
!dense_602/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_602/StatefulPartitionedCall:output:0dense_602_60936625dense_602_60936627*
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
G__inference_dense_602_layer_call_and_return_conditional_losses_609360892#
!dense_602/StatefulPartitionedCallЕ
IdentityIdentity*dense_602/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityЮ
NoOpNoOp*^bidirectional_602/StatefulPartitionedCall"^dense_602/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€:€€€€€€€€€: : : : : : : : 2V
)bidirectional_602/StatefulPartitionedCall)bidirectional_602/StatefulPartitionedCall2F
!dense_602/StatefulPartitionedCall!dense_602/StatefulPartitionedCall:O K
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
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_60935405

inputs@
-lstm_cell_1808_matmul_readvariableop_resource:	»B
/lstm_cell_1808_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1808_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1808/BiasAdd/ReadVariableOpҐ$lstm_cell_1808/MatMul/ReadVariableOpҐ&lstm_cell_1808/MatMul_1/ReadVariableOpҐwhileD
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
$lstm_cell_1808/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1808_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1808/MatMul/ReadVariableOp≥
lstm_cell_1808/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/MatMulЅ
&lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1808_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1808/MatMul_1/ReadVariableOpѓ
lstm_cell_1808/MatMul_1MatMulzeros:output:0.lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/MatMul_1®
lstm_cell_1808/addAddV2lstm_cell_1808/MatMul:product:0!lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/addЇ
%lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1808_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1808/BiasAdd/ReadVariableOpµ
lstm_cell_1808/BiasAddBiasAddlstm_cell_1808/add:z:0-lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/BiasAddВ
lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1808/split/split_dimы
lstm_cell_1808/splitSplit'lstm_cell_1808/split/split_dim:output:0lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1808/splitМ
lstm_cell_1808/SigmoidSigmoidlstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/SigmoidР
lstm_cell_1808/Sigmoid_1Sigmoidlstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Sigmoid_1С
lstm_cell_1808/mulMullstm_cell_1808/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/mulГ
lstm_cell_1808/ReluRelulstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Relu§
lstm_cell_1808/mul_1Mullstm_cell_1808/Sigmoid:y:0!lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/mul_1Щ
lstm_cell_1808/add_1AddV2lstm_cell_1808/mul:z:0lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/add_1Р
lstm_cell_1808/Sigmoid_2Sigmoidlstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Sigmoid_2В
lstm_cell_1808/Relu_1Relulstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Relu_1®
lstm_cell_1808/mul_2Mullstm_cell_1808/Sigmoid_2:y:0#lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1808_matmul_readvariableop_resource/lstm_cell_1808_matmul_1_readvariableop_resource.lstm_cell_1808_biasadd_readvariableop_resource*
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
while_body_60935321*
condR
while_cond_60935320*K
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
NoOpNoOp&^lstm_cell_1808/BiasAdd/ReadVariableOp%^lstm_cell_1808/MatMul/ReadVariableOp'^lstm_cell_1808/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1808/BiasAdd/ReadVariableOp%lstm_cell_1808/BiasAdd/ReadVariableOp2L
$lstm_cell_1808/MatMul/ReadVariableOp$lstm_cell_1808/MatMul/ReadVariableOp2P
&lstm_cell_1808/MatMul_1/ReadVariableOp&lstm_cell_1808/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
яc
ю
!__inference__traced_save_60939735
file_prefix/
+savev2_dense_602_kernel_read_readvariableop-
)savev2_dense_602_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopW
Ssavev2_bidirectional_602_forward_lstm_602_lstm_cell_1807_kernel_read_readvariableopa
]savev2_bidirectional_602_forward_lstm_602_lstm_cell_1807_recurrent_kernel_read_readvariableopU
Qsavev2_bidirectional_602_forward_lstm_602_lstm_cell_1807_bias_read_readvariableopX
Tsavev2_bidirectional_602_backward_lstm_602_lstm_cell_1808_kernel_read_readvariableopb
^savev2_bidirectional_602_backward_lstm_602_lstm_cell_1808_recurrent_kernel_read_readvariableopV
Rsavev2_bidirectional_602_backward_lstm_602_lstm_cell_1808_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_602_kernel_m_read_readvariableop4
0savev2_adam_dense_602_bias_m_read_readvariableop^
Zsavev2_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_kernel_m_read_readvariableoph
dsavev2_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_recurrent_kernel_m_read_readvariableop\
Xsavev2_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_bias_m_read_readvariableop_
[savev2_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_kernel_m_read_readvariableopi
esavev2_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_recurrent_kernel_m_read_readvariableop]
Ysavev2_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_bias_m_read_readvariableop6
2savev2_adam_dense_602_kernel_v_read_readvariableop4
0savev2_adam_dense_602_bias_v_read_readvariableop^
Zsavev2_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_kernel_v_read_readvariableoph
dsavev2_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_recurrent_kernel_v_read_readvariableop\
Xsavev2_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_bias_v_read_readvariableop_
[savev2_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_kernel_v_read_readvariableopi
esavev2_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_recurrent_kernel_v_read_readvariableop]
Ysavev2_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_bias_v_read_readvariableop9
5savev2_adam_dense_602_kernel_vhat_read_readvariableop7
3savev2_adam_dense_602_bias_vhat_read_readvariableopa
]savev2_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_kernel_vhat_read_readvariableopk
gsavev2_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_recurrent_kernel_vhat_read_readvariableop_
[savev2_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_bias_vhat_read_readvariableopb
^savev2_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_kernel_vhat_read_readvariableopl
hsavev2_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_recurrent_kernel_vhat_read_readvariableop`
\savev2_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_bias_vhat_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_602_kernel_read_readvariableop)savev2_dense_602_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopSsavev2_bidirectional_602_forward_lstm_602_lstm_cell_1807_kernel_read_readvariableop]savev2_bidirectional_602_forward_lstm_602_lstm_cell_1807_recurrent_kernel_read_readvariableopQsavev2_bidirectional_602_forward_lstm_602_lstm_cell_1807_bias_read_readvariableopTsavev2_bidirectional_602_backward_lstm_602_lstm_cell_1808_kernel_read_readvariableop^savev2_bidirectional_602_backward_lstm_602_lstm_cell_1808_recurrent_kernel_read_readvariableopRsavev2_bidirectional_602_backward_lstm_602_lstm_cell_1808_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_602_kernel_m_read_readvariableop0savev2_adam_dense_602_bias_m_read_readvariableopZsavev2_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_kernel_m_read_readvariableopdsavev2_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_recurrent_kernel_m_read_readvariableopXsavev2_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_bias_m_read_readvariableop[savev2_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_kernel_m_read_readvariableopesavev2_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_recurrent_kernel_m_read_readvariableopYsavev2_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_bias_m_read_readvariableop2savev2_adam_dense_602_kernel_v_read_readvariableop0savev2_adam_dense_602_bias_v_read_readvariableopZsavev2_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_kernel_v_read_readvariableopdsavev2_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_recurrent_kernel_v_read_readvariableopXsavev2_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_bias_v_read_readvariableop[savev2_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_kernel_v_read_readvariableopesavev2_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_recurrent_kernel_v_read_readvariableopYsavev2_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_bias_v_read_readvariableop5savev2_adam_dense_602_kernel_vhat_read_readvariableop3savev2_adam_dense_602_bias_vhat_read_readvariableop]savev2_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_kernel_vhat_read_readvariableopgsavev2_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_recurrent_kernel_vhat_read_readvariableop[savev2_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_bias_vhat_read_readvariableop^savev2_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_kernel_vhat_read_readvariableophsavev2_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_recurrent_kernel_vhat_read_readvariableop\savev2_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_bias_vhat_read_readvariableopsavev2_const"/device:CPU:0*
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
еH
Я
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_60934418

inputs*
lstm_cell_1808_60934336:	»*
lstm_cell_1808_60934338:	2»&
lstm_cell_1808_60934340:	»
identityИҐ&lstm_cell_1808/StatefulPartitionedCallҐwhileD
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
&lstm_cell_1808/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1808_60934336lstm_cell_1808_60934338lstm_cell_1808_60934340*
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
L__inference_lstm_cell_1808_layer_call_and_return_conditional_losses_609343352(
&lstm_cell_1808/StatefulPartitionedCallП
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1808_60934336lstm_cell_1808_60934338lstm_cell_1808_60934340*
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
while_body_60934349*
condR
while_cond_60934348*K
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
NoOpNoOp'^lstm_cell_1808/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2P
&lstm_cell_1808/StatefulPartitionedCall&lstm_cell_1808/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
еH
Я
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_60934630

inputs*
lstm_cell_1808_60934548:	»*
lstm_cell_1808_60934550:	2»&
lstm_cell_1808_60934552:	»
identityИҐ&lstm_cell_1808/StatefulPartitionedCallҐwhileD
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
&lstm_cell_1808/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1808_60934548lstm_cell_1808_60934550lstm_cell_1808_60934552*
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
L__inference_lstm_cell_1808_layer_call_and_return_conditional_losses_609344812(
&lstm_cell_1808/StatefulPartitionedCallП
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1808_60934548lstm_cell_1808_60934550lstm_cell_1808_60934552*
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
while_body_60934561*
condR
while_cond_60934560*K
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
NoOpNoOp'^lstm_cell_1808/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2P
&lstm_cell_1808/StatefulPartitionedCall&lstm_cell_1808/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ч
И
L__inference_lstm_cell_1808_layer_call_and_return_conditional_losses_60934481

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
ч
И
L__inference_lstm_cell_1807_layer_call_and_return_conditional_losses_60933849

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
а
ј
3__inference_forward_lstm_602_layer_call_fn_60938138

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
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_609355782
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
зe
Ћ
$forward_lstm_602_while_body_60935788>
:forward_lstm_602_while_forward_lstm_602_while_loop_counterD
@forward_lstm_602_while_forward_lstm_602_while_maximum_iterations&
"forward_lstm_602_while_placeholder(
$forward_lstm_602_while_placeholder_1(
$forward_lstm_602_while_placeholder_2(
$forward_lstm_602_while_placeholder_3(
$forward_lstm_602_while_placeholder_4=
9forward_lstm_602_while_forward_lstm_602_strided_slice_1_0y
uforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_602_while_greater_forward_lstm_602_cast_0Y
Fforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0:	»[
Hforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0:	2»V
Gforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0:	»#
forward_lstm_602_while_identity%
!forward_lstm_602_while_identity_1%
!forward_lstm_602_while_identity_2%
!forward_lstm_602_while_identity_3%
!forward_lstm_602_while_identity_4%
!forward_lstm_602_while_identity_5%
!forward_lstm_602_while_identity_6;
7forward_lstm_602_while_forward_lstm_602_strided_slice_1w
sforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_602_while_greater_forward_lstm_602_castW
Dforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource:	»Y
Fforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource:	2»T
Eforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource:	»ИҐ<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpҐ;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOpҐ=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpе
Hforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2J
Hforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeє
:forward_lstm_602/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_602_while_placeholderQforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02<
:forward_lstm_602/while/TensorArrayV2Read/TensorListGetItem’
forward_lstm_602/while/GreaterGreater6forward_lstm_602_while_greater_forward_lstm_602_cast_0"forward_lstm_602_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2 
forward_lstm_602/while/GreaterВ
;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOpFforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02=
;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp°
,forward_lstm_602/while/lstm_cell_1807/MatMulMatMulAforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem:item:0Cforward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2.
,forward_lstm_602/while/lstm_cell_1807/MatMulИ
=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOpHforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02?
=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpК
.forward_lstm_602/while/lstm_cell_1807/MatMul_1MatMul$forward_lstm_602_while_placeholder_3Eforward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.forward_lstm_602/while/lstm_cell_1807/MatMul_1Д
)forward_lstm_602/while/lstm_cell_1807/addAddV26forward_lstm_602/while/lstm_cell_1807/MatMul:product:08forward_lstm_602/while/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)forward_lstm_602/while/lstm_cell_1807/addБ
<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOpGforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02>
<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpС
-forward_lstm_602/while/lstm_cell_1807/BiasAddBiasAdd-forward_lstm_602/while/lstm_cell_1807/add:z:0Dforward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-forward_lstm_602/while/lstm_cell_1807/BiasAdd∞
5forward_lstm_602/while/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5forward_lstm_602/while/lstm_cell_1807/split/split_dim„
+forward_lstm_602/while/lstm_cell_1807/splitSplit>forward_lstm_602/while/lstm_cell_1807/split/split_dim:output:06forward_lstm_602/while/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2-
+forward_lstm_602/while/lstm_cell_1807/split—
-forward_lstm_602/while/lstm_cell_1807/SigmoidSigmoid4forward_lstm_602/while/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-forward_lstm_602/while/lstm_cell_1807/Sigmoid’
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_1Sigmoid4forward_lstm_602/while/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_1к
)forward_lstm_602/while/lstm_cell_1807/mulMul3forward_lstm_602/while/lstm_cell_1807/Sigmoid_1:y:0$forward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_602/while/lstm_cell_1807/mul»
*forward_lstm_602/while/lstm_cell_1807/ReluRelu4forward_lstm_602/while/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22,
*forward_lstm_602/while/lstm_cell_1807/ReluА
+forward_lstm_602/while/lstm_cell_1807/mul_1Mul1forward_lstm_602/while/lstm_cell_1807/Sigmoid:y:08forward_lstm_602/while/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_602/while/lstm_cell_1807/mul_1х
+forward_lstm_602/while/lstm_cell_1807/add_1AddV2-forward_lstm_602/while/lstm_cell_1807/mul:z:0/forward_lstm_602/while/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_602/while/lstm_cell_1807/add_1’
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_2Sigmoid4forward_lstm_602/while/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_2«
,forward_lstm_602/while/lstm_cell_1807/Relu_1Relu/forward_lstm_602/while/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,forward_lstm_602/while/lstm_cell_1807/Relu_1Д
+forward_lstm_602/while/lstm_cell_1807/mul_2Mul3forward_lstm_602/while/lstm_cell_1807/Sigmoid_2:y:0:forward_lstm_602/while/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_602/while/lstm_cell_1807/mul_2х
forward_lstm_602/while/SelectSelect"forward_lstm_602/while/Greater:z:0/forward_lstm_602/while/lstm_cell_1807/mul_2:z:0$forward_lstm_602_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/while/Selectщ
forward_lstm_602/while/Select_1Select"forward_lstm_602/while/Greater:z:0/forward_lstm_602/while/lstm_cell_1807/mul_2:z:0$forward_lstm_602_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_602/while/Select_1щ
forward_lstm_602/while/Select_2Select"forward_lstm_602/while/Greater:z:0/forward_lstm_602/while/lstm_cell_1807/add_1:z:0$forward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_602/while/Select_2Ѓ
;forward_lstm_602/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_602_while_placeholder_1"forward_lstm_602_while_placeholder&forward_lstm_602/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_602/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_602/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_602/while/add/y≠
forward_lstm_602/while/addAddV2"forward_lstm_602_while_placeholder%forward_lstm_602/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/while/addВ
forward_lstm_602/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_602/while/add_1/yЋ
forward_lstm_602/while/add_1AddV2:forward_lstm_602_while_forward_lstm_602_while_loop_counter'forward_lstm_602/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/while/add_1ѓ
forward_lstm_602/while/IdentityIdentity forward_lstm_602/while/add_1:z:0^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_602/while/Identity”
!forward_lstm_602/while/Identity_1Identity@forward_lstm_602_while_forward_lstm_602_while_maximum_iterations^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_602/while/Identity_1±
!forward_lstm_602/while/Identity_2Identityforward_lstm_602/while/add:z:0^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_602/while/Identity_2ё
!forward_lstm_602/while/Identity_3IdentityKforward_lstm_602/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_602/while/Identity_3 
!forward_lstm_602/while/Identity_4Identity&forward_lstm_602/while/Select:output:0^forward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_602/while/Identity_4ћ
!forward_lstm_602/while/Identity_5Identity(forward_lstm_602/while/Select_1:output:0^forward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_602/while/Identity_5ћ
!forward_lstm_602/while/Identity_6Identity(forward_lstm_602/while/Select_2:output:0^forward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_602/while/Identity_6є
forward_lstm_602/while/NoOpNoOp=^forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp<^forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp>^forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_602/while/NoOp"t
7forward_lstm_602_while_forward_lstm_602_strided_slice_19forward_lstm_602_while_forward_lstm_602_strided_slice_1_0"n
4forward_lstm_602_while_greater_forward_lstm_602_cast6forward_lstm_602_while_greater_forward_lstm_602_cast_0"K
forward_lstm_602_while_identity(forward_lstm_602/while/Identity:output:0"O
!forward_lstm_602_while_identity_1*forward_lstm_602/while/Identity_1:output:0"O
!forward_lstm_602_while_identity_2*forward_lstm_602/while/Identity_2:output:0"O
!forward_lstm_602_while_identity_3*forward_lstm_602/while/Identity_3:output:0"O
!forward_lstm_602_while_identity_4*forward_lstm_602/while/Identity_4:output:0"O
!forward_lstm_602_while_identity_5*forward_lstm_602/while/Identity_5:output:0"O
!forward_lstm_602_while_identity_6*forward_lstm_602/while/Identity_6:output:0"Р
Eforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resourceGforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0"Т
Fforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resourceHforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0"О
Dforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resourceFforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0"м
sforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensoruforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2|
<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp2z
;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp2~
=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
$forward_lstm_602_while_cond_60937797>
:forward_lstm_602_while_forward_lstm_602_while_loop_counterD
@forward_lstm_602_while_forward_lstm_602_while_maximum_iterations&
"forward_lstm_602_while_placeholder(
$forward_lstm_602_while_placeholder_1(
$forward_lstm_602_while_placeholder_2(
$forward_lstm_602_while_placeholder_3(
$forward_lstm_602_while_placeholder_4@
<forward_lstm_602_while_less_forward_lstm_602_strided_slice_1X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60937797___redundant_placeholder0X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60937797___redundant_placeholder1X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60937797___redundant_placeholder2X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60937797___redundant_placeholder3X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60937797___redundant_placeholder4#
forward_lstm_602_while_identity
≈
forward_lstm_602/while/LessLess"forward_lstm_602_while_placeholder<forward_lstm_602_while_less_forward_lstm_602_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_602/while/LessР
forward_lstm_602/while/IdentityIdentityforward_lstm_602/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_602/while/Identity"K
forward_lstm_602_while_identity(forward_lstm_602/while/Identity:output:0*(
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
–]
і
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_60938440
inputs_0@
-lstm_cell_1807_matmul_readvariableop_resource:	»B
/lstm_cell_1807_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1807_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1807/BiasAdd/ReadVariableOpҐ$lstm_cell_1807/MatMul/ReadVariableOpҐ&lstm_cell_1807/MatMul_1/ReadVariableOpҐwhileF
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
$lstm_cell_1807/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1807_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1807/MatMul/ReadVariableOp≥
lstm_cell_1807/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/MatMulЅ
&lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1807_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1807/MatMul_1/ReadVariableOpѓ
lstm_cell_1807/MatMul_1MatMulzeros:output:0.lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/MatMul_1®
lstm_cell_1807/addAddV2lstm_cell_1807/MatMul:product:0!lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/addЇ
%lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1807_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1807/BiasAdd/ReadVariableOpµ
lstm_cell_1807/BiasAddBiasAddlstm_cell_1807/add:z:0-lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/BiasAddВ
lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1807/split/split_dimы
lstm_cell_1807/splitSplit'lstm_cell_1807/split/split_dim:output:0lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1807/splitМ
lstm_cell_1807/SigmoidSigmoidlstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/SigmoidР
lstm_cell_1807/Sigmoid_1Sigmoidlstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Sigmoid_1С
lstm_cell_1807/mulMullstm_cell_1807/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/mulГ
lstm_cell_1807/ReluRelulstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Relu§
lstm_cell_1807/mul_1Mullstm_cell_1807/Sigmoid:y:0!lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/mul_1Щ
lstm_cell_1807/add_1AddV2lstm_cell_1807/mul:z:0lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/add_1Р
lstm_cell_1807/Sigmoid_2Sigmoidlstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Sigmoid_2В
lstm_cell_1807/Relu_1Relulstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Relu_1®
lstm_cell_1807/mul_2Mullstm_cell_1807/Sigmoid_2:y:0#lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1807_matmul_readvariableop_resource/lstm_cell_1807_matmul_1_readvariableop_resource.lstm_cell_1807_biasadd_readvariableop_resource*
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
while_body_60938356*
condR
while_cond_60938355*K
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
NoOpNoOp&^lstm_cell_1807/BiasAdd/ReadVariableOp%^lstm_cell_1807/MatMul/ReadVariableOp'^lstm_cell_1807/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1807/BiasAdd/ReadVariableOp%lstm_cell_1807/BiasAdd/ReadVariableOp2L
$lstm_cell_1807/MatMul/ReadVariableOp$lstm_cell_1807/MatMul/ReadVariableOp2P
&lstm_cell_1807/MatMul_1/ReadVariableOp&lstm_cell_1807/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
я
Ќ
while_cond_60938854
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60938854___redundant_placeholder06
2while_while_cond_60938854___redundant_placeholder16
2while_while_cond_60938854___redundant_placeholder26
2while_while_cond_60938854___redundant_placeholder3
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
Љ
¬
Fsequential_602_bidirectional_602_backward_lstm_602_while_cond_60933523В
~sequential_602_bidirectional_602_backward_lstm_602_while_sequential_602_bidirectional_602_backward_lstm_602_while_loop_counterЙ
Дsequential_602_bidirectional_602_backward_lstm_602_while_sequential_602_bidirectional_602_backward_lstm_602_while_maximum_iterationsH
Dsequential_602_bidirectional_602_backward_lstm_602_while_placeholderJ
Fsequential_602_bidirectional_602_backward_lstm_602_while_placeholder_1J
Fsequential_602_bidirectional_602_backward_lstm_602_while_placeholder_2J
Fsequential_602_bidirectional_602_backward_lstm_602_while_placeholder_3J
Fsequential_602_bidirectional_602_backward_lstm_602_while_placeholder_4Е
Аsequential_602_bidirectional_602_backward_lstm_602_while_less_sequential_602_bidirectional_602_backward_lstm_602_strided_slice_1Э
Шsequential_602_bidirectional_602_backward_lstm_602_while_sequential_602_bidirectional_602_backward_lstm_602_while_cond_60933523___redundant_placeholder0Э
Шsequential_602_bidirectional_602_backward_lstm_602_while_sequential_602_bidirectional_602_backward_lstm_602_while_cond_60933523___redundant_placeholder1Э
Шsequential_602_bidirectional_602_backward_lstm_602_while_sequential_602_bidirectional_602_backward_lstm_602_while_cond_60933523___redundant_placeholder2Э
Шsequential_602_bidirectional_602_backward_lstm_602_while_sequential_602_bidirectional_602_backward_lstm_602_while_cond_60933523___redundant_placeholder3Э
Шsequential_602_bidirectional_602_backward_lstm_602_while_sequential_602_bidirectional_602_backward_lstm_602_while_cond_60933523___redundant_placeholder4E
Asequential_602_bidirectional_602_backward_lstm_602_while_identity
р
=sequential_602/bidirectional_602/backward_lstm_602/while/LessLessDsequential_602_bidirectional_602_backward_lstm_602_while_placeholderАsequential_602_bidirectional_602_backward_lstm_602_while_less_sequential_602_bidirectional_602_backward_lstm_602_strided_slice_1*
T0*
_output_shapes
: 2?
=sequential_602/bidirectional_602/backward_lstm_602/while/Lessц
Asequential_602/bidirectional_602/backward_lstm_602/while/IdentityIdentityAsequential_602/bidirectional_602/backward_lstm_602/while/Less:z:0*
T0
*
_output_shapes
: 2C
Asequential_602/bidirectional_602/backward_lstm_602/while/Identity"П
Asequential_602_bidirectional_602_backward_lstm_602_while_identityJsequential_602/bidirectional_602/backward_lstm_602/while/Identity:output:0*(
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
Щ
™
Esequential_602_bidirectional_602_forward_lstm_602_while_cond_60933344А
|sequential_602_bidirectional_602_forward_lstm_602_while_sequential_602_bidirectional_602_forward_lstm_602_while_loop_counterЗ
Вsequential_602_bidirectional_602_forward_lstm_602_while_sequential_602_bidirectional_602_forward_lstm_602_while_maximum_iterationsG
Csequential_602_bidirectional_602_forward_lstm_602_while_placeholderI
Esequential_602_bidirectional_602_forward_lstm_602_while_placeholder_1I
Esequential_602_bidirectional_602_forward_lstm_602_while_placeholder_2I
Esequential_602_bidirectional_602_forward_lstm_602_while_placeholder_3I
Esequential_602_bidirectional_602_forward_lstm_602_while_placeholder_4В
~sequential_602_bidirectional_602_forward_lstm_602_while_less_sequential_602_bidirectional_602_forward_lstm_602_strided_slice_1Ы
Цsequential_602_bidirectional_602_forward_lstm_602_while_sequential_602_bidirectional_602_forward_lstm_602_while_cond_60933344___redundant_placeholder0Ы
Цsequential_602_bidirectional_602_forward_lstm_602_while_sequential_602_bidirectional_602_forward_lstm_602_while_cond_60933344___redundant_placeholder1Ы
Цsequential_602_bidirectional_602_forward_lstm_602_while_sequential_602_bidirectional_602_forward_lstm_602_while_cond_60933344___redundant_placeholder2Ы
Цsequential_602_bidirectional_602_forward_lstm_602_while_sequential_602_bidirectional_602_forward_lstm_602_while_cond_60933344___redundant_placeholder3Ы
Цsequential_602_bidirectional_602_forward_lstm_602_while_sequential_602_bidirectional_602_forward_lstm_602_while_cond_60933344___redundant_placeholder4D
@sequential_602_bidirectional_602_forward_lstm_602_while_identity
к
<sequential_602/bidirectional_602/forward_lstm_602/while/LessLessCsequential_602_bidirectional_602_forward_lstm_602_while_placeholder~sequential_602_bidirectional_602_forward_lstm_602_while_less_sequential_602_bidirectional_602_forward_lstm_602_strided_slice_1*
T0*
_output_shapes
: 2>
<sequential_602/bidirectional_602/forward_lstm_602/while/Lessу
@sequential_602/bidirectional_602/forward_lstm_602/while/IdentityIdentity@sequential_602/bidirectional_602/forward_lstm_602/while/Less:z:0*
T0
*
_output_shapes
: 2B
@sequential_602/bidirectional_602/forward_lstm_602/while/Identity"Н
@sequential_602_bidirectional_602_forward_lstm_602_while_identityIsequential_602/bidirectional_602/forward_lstm_602/while/Identity:output:0*(
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
÷
√
4__inference_backward_lstm_602_layer_call_fn_60938764
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
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_609346302
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
while_cond_60935128
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60935128___redundant_placeholder06
2while_while_cond_60935128___redundant_placeholder16
2while_while_cond_60935128___redundant_placeholder26
2while_while_cond_60935128___redundant_placeholder3
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
%backward_lstm_602_while_cond_60936406@
<backward_lstm_602_while_backward_lstm_602_while_loop_counterF
Bbackward_lstm_602_while_backward_lstm_602_while_maximum_iterations'
#backward_lstm_602_while_placeholder)
%backward_lstm_602_while_placeholder_1)
%backward_lstm_602_while_placeholder_2)
%backward_lstm_602_while_placeholder_3)
%backward_lstm_602_while_placeholder_4B
>backward_lstm_602_while_less_backward_lstm_602_strided_slice_1Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60936406___redundant_placeholder0Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60936406___redundant_placeholder1Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60936406___redundant_placeholder2Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60936406___redundant_placeholder3Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60936406___redundant_placeholder4$
 backward_lstm_602_while_identity
 
backward_lstm_602/while/LessLess#backward_lstm_602_while_placeholder>backward_lstm_602_while_less_backward_lstm_602_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_602/while/LessУ
 backward_lstm_602/while/IdentityIdentity backward_lstm_602/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_602/while/Identity"M
 backward_lstm_602_while_identity)backward_lstm_602/while/Identity:output:0*(
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
∆@
д
while_body_60939008
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1808_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1808_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1808_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1808_matmul_readvariableop_resource:	»H
5while_lstm_cell_1808_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1808_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1808/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1808/MatMul/ReadVariableOpҐ,while/lstm_cell_1808/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1808_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1808/MatMul/ReadVariableOpЁ
while/lstm_cell_1808/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/MatMul’
,while/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1808_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1808/MatMul_1/ReadVariableOp∆
while/lstm_cell_1808/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/MatMul_1ј
while/lstm_cell_1808/addAddV2%while/lstm_cell_1808/MatMul:product:0'while/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/addќ
+while/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1808_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1808/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1808/BiasAddBiasAddwhile/lstm_cell_1808/add:z:03while/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/BiasAddО
$while/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1808/split/split_dimУ
while/lstm_cell_1808/splitSplit-while/lstm_cell_1808/split/split_dim:output:0%while/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1808/splitЮ
while/lstm_cell_1808/SigmoidSigmoid#while/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/SigmoidҐ
while/lstm_cell_1808/Sigmoid_1Sigmoid#while/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1808/Sigmoid_1¶
while/lstm_cell_1808/mulMul"while/lstm_cell_1808/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/mulХ
while/lstm_cell_1808/ReluRelu#while/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/ReluЉ
while/lstm_cell_1808/mul_1Mul while/lstm_cell_1808/Sigmoid:y:0'while/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/mul_1±
while/lstm_cell_1808/add_1AddV2while/lstm_cell_1808/mul:z:0while/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/add_1Ґ
while/lstm_cell_1808/Sigmoid_2Sigmoid#while/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1808/Sigmoid_2Ф
while/lstm_cell_1808/Relu_1Reluwhile/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/Relu_1ј
while/lstm_cell_1808/mul_2Mul"while/lstm_cell_1808/Sigmoid_2:y:0)while/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1808/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1808/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1808/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1808/BiasAdd/ReadVariableOp+^while/lstm_cell_1808/MatMul/ReadVariableOp-^while/lstm_cell_1808/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1808_biasadd_readvariableop_resource6while_lstm_cell_1808_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1808_matmul_1_readvariableop_resource7while_lstm_cell_1808_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1808_matmul_readvariableop_resource5while_lstm_cell_1808_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1808/BiasAdd/ReadVariableOp+while/lstm_cell_1808/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1808/MatMul/ReadVariableOp*while/lstm_cell_1808/MatMul/ReadVariableOp2\
,while/lstm_cell_1808/MatMul_1/ReadVariableOp,while/lstm_cell_1808/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_60938074

inputs
inputs_1	Q
>forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource:	»S
@forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource:	2»N
?forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource:	»R
?backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resource:	»T
Abackward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource:	2»O
@backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource:	»
identityИҐ7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpҐ6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpҐ8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpҐbackward_lstm_602/whileҐ6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpҐ5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpҐ7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpҐforward_lstm_602/whileЧ
%forward_lstm_602/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_602/RaggedToTensor/zerosЩ
%forward_lstm_602/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2'
%forward_lstm_602/RaggedToTensor/ConstЩ
4forward_lstm_602/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_602/RaggedToTensor/Const:output:0inputs.forward_lstm_602/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_602/RaggedToTensor/RaggedTensorToTensorƒ
;forward_lstm_602/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack»
=forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1»
=forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2©
5forward_lstm_602/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_602/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask27
5forward_lstm_602/RaggedNestedRowLengths/strided_slice»
=forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack’
?forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2A
?forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1ћ
?forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2µ
7forward_lstm_602/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask29
7forward_lstm_602/RaggedNestedRowLengths/strided_slice_1С
+forward_lstm_602/RaggedNestedRowLengths/subSub>forward_lstm_602/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_602/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2-
+forward_lstm_602/RaggedNestedRowLengths/sub§
forward_lstm_602/CastCast/forward_lstm_602/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
forward_lstm_602/CastЭ
forward_lstm_602/ShapeShape=forward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_602/ShapeЦ
$forward_lstm_602/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_602/strided_slice/stackЪ
&forward_lstm_602/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_602/strided_slice/stack_1Ъ
&forward_lstm_602/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_602/strided_slice/stack_2»
forward_lstm_602/strided_sliceStridedSliceforward_lstm_602/Shape:output:0-forward_lstm_602/strided_slice/stack:output:0/forward_lstm_602/strided_slice/stack_1:output:0/forward_lstm_602/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_602/strided_slice~
forward_lstm_602/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_602/zeros/mul/y∞
forward_lstm_602/zeros/mulMul'forward_lstm_602/strided_slice:output:0%forward_lstm_602/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros/mulБ
forward_lstm_602/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
forward_lstm_602/zeros/Less/yЂ
forward_lstm_602/zeros/LessLessforward_lstm_602/zeros/mul:z:0&forward_lstm_602/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros/LessД
forward_lstm_602/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_602/zeros/packed/1«
forward_lstm_602/zeros/packedPack'forward_lstm_602/strided_slice:output:0(forward_lstm_602/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_602/zeros/packedЕ
forward_lstm_602/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_602/zeros/Constє
forward_lstm_602/zerosFill&forward_lstm_602/zeros/packed:output:0%forward_lstm_602/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/zerosВ
forward_lstm_602/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_602/zeros_1/mul/yґ
forward_lstm_602/zeros_1/mulMul'forward_lstm_602/strided_slice:output:0'forward_lstm_602/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros_1/mulЕ
forward_lstm_602/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2!
forward_lstm_602/zeros_1/Less/y≥
forward_lstm_602/zeros_1/LessLess forward_lstm_602/zeros_1/mul:z:0(forward_lstm_602/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros_1/LessИ
!forward_lstm_602/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_602/zeros_1/packed/1Ќ
forward_lstm_602/zeros_1/packedPack'forward_lstm_602/strided_slice:output:0*forward_lstm_602/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_602/zeros_1/packedЙ
forward_lstm_602/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_602/zeros_1/ConstЅ
forward_lstm_602/zeros_1Fill(forward_lstm_602/zeros_1/packed:output:0'forward_lstm_602/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/zeros_1Ч
forward_lstm_602/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_602/transpose/permн
forward_lstm_602/transpose	Transpose=forward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_602/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
forward_lstm_602/transposeВ
forward_lstm_602/Shape_1Shapeforward_lstm_602/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_602/Shape_1Ъ
&forward_lstm_602/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_602/strided_slice_1/stackЮ
(forward_lstm_602/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_1/stack_1Ю
(forward_lstm_602/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_1/stack_2‘
 forward_lstm_602/strided_slice_1StridedSlice!forward_lstm_602/Shape_1:output:0/forward_lstm_602/strided_slice_1/stack:output:01forward_lstm_602/strided_slice_1/stack_1:output:01forward_lstm_602/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_602/strided_slice_1І
,forward_lstm_602/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,forward_lstm_602/TensorArrayV2/element_shapeц
forward_lstm_602/TensorArrayV2TensorListReserve5forward_lstm_602/TensorArrayV2/element_shape:output:0)forward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_602/TensorArrayV2б
Fforward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2H
Fforward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeЉ
8forward_lstm_602/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_602/transpose:y:0Oforward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_602/TensorArrayUnstack/TensorListFromTensorЪ
&forward_lstm_602/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_602/strided_slice_2/stackЮ
(forward_lstm_602/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_2/stack_1Ю
(forward_lstm_602/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_2/stack_2в
 forward_lstm_602/strided_slice_2StridedSliceforward_lstm_602/transpose:y:0/forward_lstm_602/strided_slice_2/stack:output:01forward_lstm_602/strided_slice_2/stack_1:output:01forward_lstm_602/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2"
 forward_lstm_602/strided_slice_2о
5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOp>forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype027
5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpч
&forward_lstm_602/lstm_cell_1807/MatMulMatMul)forward_lstm_602/strided_slice_2:output:0=forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2(
&forward_lstm_602/lstm_cell_1807/MatMulф
7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOp@forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype029
7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpу
(forward_lstm_602/lstm_cell_1807/MatMul_1MatMulforward_lstm_602/zeros:output:0?forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(forward_lstm_602/lstm_cell_1807/MatMul_1м
#forward_lstm_602/lstm_cell_1807/addAddV20forward_lstm_602/lstm_cell_1807/MatMul:product:02forward_lstm_602/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2%
#forward_lstm_602/lstm_cell_1807/addн
6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOp?forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype028
6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpщ
'forward_lstm_602/lstm_cell_1807/BiasAddBiasAdd'forward_lstm_602/lstm_cell_1807/add:z:0>forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'forward_lstm_602/lstm_cell_1807/BiasAdd§
/forward_lstm_602/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/forward_lstm_602/lstm_cell_1807/split/split_dimњ
%forward_lstm_602/lstm_cell_1807/splitSplit8forward_lstm_602/lstm_cell_1807/split/split_dim:output:00forward_lstm_602/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2'
%forward_lstm_602/lstm_cell_1807/splitњ
'forward_lstm_602/lstm_cell_1807/SigmoidSigmoid.forward_lstm_602/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'forward_lstm_602/lstm_cell_1807/Sigmoid√
)forward_lstm_602/lstm_cell_1807/Sigmoid_1Sigmoid.forward_lstm_602/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_602/lstm_cell_1807/Sigmoid_1’
#forward_lstm_602/lstm_cell_1807/mulMul-forward_lstm_602/lstm_cell_1807/Sigmoid_1:y:0!forward_lstm_602/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22%
#forward_lstm_602/lstm_cell_1807/mulґ
$forward_lstm_602/lstm_cell_1807/ReluRelu.forward_lstm_602/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22&
$forward_lstm_602/lstm_cell_1807/Reluи
%forward_lstm_602/lstm_cell_1807/mul_1Mul+forward_lstm_602/lstm_cell_1807/Sigmoid:y:02forward_lstm_602/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_602/lstm_cell_1807/mul_1Ё
%forward_lstm_602/lstm_cell_1807/add_1AddV2'forward_lstm_602/lstm_cell_1807/mul:z:0)forward_lstm_602/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_602/lstm_cell_1807/add_1√
)forward_lstm_602/lstm_cell_1807/Sigmoid_2Sigmoid.forward_lstm_602/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_602/lstm_cell_1807/Sigmoid_2µ
&forward_lstm_602/lstm_cell_1807/Relu_1Relu)forward_lstm_602/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&forward_lstm_602/lstm_cell_1807/Relu_1м
%forward_lstm_602/lstm_cell_1807/mul_2Mul-forward_lstm_602/lstm_cell_1807/Sigmoid_2:y:04forward_lstm_602/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_602/lstm_cell_1807/mul_2±
.forward_lstm_602/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   20
.forward_lstm_602/TensorArrayV2_1/element_shapeь
 forward_lstm_602/TensorArrayV2_1TensorListReserve7forward_lstm_602/TensorArrayV2_1/element_shape:output:0)forward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_602/TensorArrayV2_1p
forward_lstm_602/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_602/time§
forward_lstm_602/zeros_like	ZerosLike)forward_lstm_602/lstm_cell_1807/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/zeros_like°
)forward_lstm_602/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2+
)forward_lstm_602/while/maximum_iterationsМ
#forward_lstm_602/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_602/while/loop_counterЦ	
forward_lstm_602/whileWhile,forward_lstm_602/while/loop_counter:output:02forward_lstm_602/while/maximum_iterations:output:0forward_lstm_602/time:output:0)forward_lstm_602/TensorArrayV2_1:handle:0forward_lstm_602/zeros_like:y:0forward_lstm_602/zeros:output:0!forward_lstm_602/zeros_1:output:0)forward_lstm_602/strided_slice_1:output:0Hforward_lstm_602/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_602/Cast:y:0>forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource@forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource?forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource*
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
$forward_lstm_602_while_body_60937798*0
cond(R&
$forward_lstm_602_while_cond_60937797*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
forward_lstm_602/while„
Aforward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2C
Aforward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeµ
3forward_lstm_602/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_602/while:output:3Jforward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype025
3forward_lstm_602/TensorArrayV2Stack/TensorListStack£
&forward_lstm_602/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2(
&forward_lstm_602/strided_slice_3/stackЮ
(forward_lstm_602/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_602/strided_slice_3/stack_1Ю
(forward_lstm_602/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_3/stack_2А
 forward_lstm_602/strided_slice_3StridedSlice<forward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_602/strided_slice_3/stack:output:01forward_lstm_602/strided_slice_3/stack_1:output:01forward_lstm_602/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2"
 forward_lstm_602/strided_slice_3Ы
!forward_lstm_602/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_602/transpose_1/permт
forward_lstm_602/transpose_1	Transpose<forward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_602/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
forward_lstm_602/transpose_1И
forward_lstm_602/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_602/runtimeЩ
&backward_lstm_602/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_602/RaggedToTensor/zerosЫ
&backward_lstm_602/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2(
&backward_lstm_602/RaggedToTensor/ConstЭ
5backward_lstm_602/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_602/RaggedToTensor/Const:output:0inputs/backward_lstm_602/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_602/RaggedToTensor/RaggedTensorToTensor∆
<backward_lstm_602/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack 
>backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1 
>backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2Ѓ
6backward_lstm_602/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_602/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_602/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask28
6backward_lstm_602/RaggedNestedRowLengths/strided_slice 
>backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack„
@backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2B
@backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1ќ
@backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2Ї
8backward_lstm_602/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_602/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask2:
8backward_lstm_602/RaggedNestedRowLengths/strided_slice_1Х
,backward_lstm_602/RaggedNestedRowLengths/subSub?backward_lstm_602/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_602/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2.
,backward_lstm_602/RaggedNestedRowLengths/subІ
backward_lstm_602/CastCast0backward_lstm_602/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
backward_lstm_602/Cast†
backward_lstm_602/ShapeShape>backward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_602/ShapeШ
%backward_lstm_602/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_602/strided_slice/stackЬ
'backward_lstm_602/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_602/strided_slice/stack_1Ь
'backward_lstm_602/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_602/strided_slice/stack_2ќ
backward_lstm_602/strided_sliceStridedSlice backward_lstm_602/Shape:output:0.backward_lstm_602/strided_slice/stack:output:00backward_lstm_602/strided_slice/stack_1:output:00backward_lstm_602/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_602/strided_sliceА
backward_lstm_602/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_602/zeros/mul/yі
backward_lstm_602/zeros/mulMul(backward_lstm_602/strided_slice:output:0&backward_lstm_602/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/zeros/mulГ
backward_lstm_602/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
backward_lstm_602/zeros/Less/yѓ
backward_lstm_602/zeros/LessLessbackward_lstm_602/zeros/mul:z:0'backward_lstm_602/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/zeros/LessЖ
 backward_lstm_602/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_602/zeros/packed/1Ћ
backward_lstm_602/zeros/packedPack(backward_lstm_602/strided_slice:output:0)backward_lstm_602/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_602/zeros/packedЗ
backward_lstm_602/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_602/zeros/Constљ
backward_lstm_602/zerosFill'backward_lstm_602/zeros/packed:output:0&backward_lstm_602/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_602/zerosД
backward_lstm_602/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_602/zeros_1/mul/yЇ
backward_lstm_602/zeros_1/mulMul(backward_lstm_602/strided_slice:output:0(backward_lstm_602/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/zeros_1/mulЗ
 backward_lstm_602/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 backward_lstm_602/zeros_1/Less/yЈ
backward_lstm_602/zeros_1/LessLess!backward_lstm_602/zeros_1/mul:z:0)backward_lstm_602/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_602/zeros_1/LessК
"backward_lstm_602/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_602/zeros_1/packed/1—
 backward_lstm_602/zeros_1/packedPack(backward_lstm_602/strided_slice:output:0+backward_lstm_602/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_602/zeros_1/packedЛ
backward_lstm_602/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_602/zeros_1/Const≈
backward_lstm_602/zeros_1Fill)backward_lstm_602/zeros_1/packed:output:0(backward_lstm_602/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_602/zeros_1Щ
 backward_lstm_602/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_602/transpose/permс
backward_lstm_602/transpose	Transpose>backward_lstm_602/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_602/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_602/transposeЕ
backward_lstm_602/Shape_1Shapebackward_lstm_602/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_602/Shape_1Ь
'backward_lstm_602/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_602/strided_slice_1/stack†
)backward_lstm_602/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_1/stack_1†
)backward_lstm_602/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_1/stack_2Џ
!backward_lstm_602/strided_slice_1StridedSlice"backward_lstm_602/Shape_1:output:00backward_lstm_602/strided_slice_1/stack:output:02backward_lstm_602/strided_slice_1/stack_1:output:02backward_lstm_602/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_602/strided_slice_1©
-backward_lstm_602/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-backward_lstm_602/TensorArrayV2/element_shapeъ
backward_lstm_602/TensorArrayV2TensorListReserve6backward_lstm_602/TensorArrayV2/element_shape:output:0*backward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_602/TensorArrayV2О
 backward_lstm_602/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_602/ReverseV2/axis“
backward_lstm_602/ReverseV2	ReverseV2backward_lstm_602/transpose:y:0)backward_lstm_602/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_602/ReverseV2г
Gbackward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2I
Gbackward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape≈
9backward_lstm_602/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_602/ReverseV2:output:0Pbackward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_602/TensorArrayUnstack/TensorListFromTensorЬ
'backward_lstm_602/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_602/strided_slice_2/stack†
)backward_lstm_602/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_2/stack_1†
)backward_lstm_602/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_2/stack_2и
!backward_lstm_602/strided_slice_2StridedSlicebackward_lstm_602/transpose:y:00backward_lstm_602/strided_slice_2/stack:output:02backward_lstm_602/strided_slice_2/stack_1:output:02backward_lstm_602/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2#
!backward_lstm_602/strided_slice_2с
6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOp?backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype028
6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpы
'backward_lstm_602/lstm_cell_1808/MatMulMatMul*backward_lstm_602/strided_slice_2:output:0>backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'backward_lstm_602/lstm_cell_1808/MatMulч
8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOpAbackward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02:
8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpч
)backward_lstm_602/lstm_cell_1808/MatMul_1MatMul backward_lstm_602/zeros:output:0@backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)backward_lstm_602/lstm_cell_1808/MatMul_1р
$backward_lstm_602/lstm_cell_1808/addAddV21backward_lstm_602/lstm_cell_1808/MatMul:product:03backward_lstm_602/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2&
$backward_lstm_602/lstm_cell_1808/addр
7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOp@backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype029
7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpэ
(backward_lstm_602/lstm_cell_1808/BiasAddBiasAdd(backward_lstm_602/lstm_cell_1808/add:z:0?backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(backward_lstm_602/lstm_cell_1808/BiasAdd¶
0backward_lstm_602/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0backward_lstm_602/lstm_cell_1808/split/split_dim√
&backward_lstm_602/lstm_cell_1808/splitSplit9backward_lstm_602/lstm_cell_1808/split/split_dim:output:01backward_lstm_602/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2(
&backward_lstm_602/lstm_cell_1808/split¬
(backward_lstm_602/lstm_cell_1808/SigmoidSigmoid/backward_lstm_602/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(backward_lstm_602/lstm_cell_1808/Sigmoid∆
*backward_lstm_602/lstm_cell_1808/Sigmoid_1Sigmoid/backward_lstm_602/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_602/lstm_cell_1808/Sigmoid_1ў
$backward_lstm_602/lstm_cell_1808/mulMul.backward_lstm_602/lstm_cell_1808/Sigmoid_1:y:0"backward_lstm_602/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$backward_lstm_602/lstm_cell_1808/mulє
%backward_lstm_602/lstm_cell_1808/ReluRelu/backward_lstm_602/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22'
%backward_lstm_602/lstm_cell_1808/Reluм
&backward_lstm_602/lstm_cell_1808/mul_1Mul,backward_lstm_602/lstm_cell_1808/Sigmoid:y:03backward_lstm_602/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_602/lstm_cell_1808/mul_1б
&backward_lstm_602/lstm_cell_1808/add_1AddV2(backward_lstm_602/lstm_cell_1808/mul:z:0*backward_lstm_602/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_602/lstm_cell_1808/add_1∆
*backward_lstm_602/lstm_cell_1808/Sigmoid_2Sigmoid/backward_lstm_602/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_602/lstm_cell_1808/Sigmoid_2Є
'backward_lstm_602/lstm_cell_1808/Relu_1Relu*backward_lstm_602/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'backward_lstm_602/lstm_cell_1808/Relu_1р
&backward_lstm_602/lstm_cell_1808/mul_2Mul.backward_lstm_602/lstm_cell_1808/Sigmoid_2:y:05backward_lstm_602/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_602/lstm_cell_1808/mul_2≥
/backward_lstm_602/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   21
/backward_lstm_602/TensorArrayV2_1/element_shapeА
!backward_lstm_602/TensorArrayV2_1TensorListReserve8backward_lstm_602/TensorArrayV2_1/element_shape:output:0*backward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_602/TensorArrayV2_1r
backward_lstm_602/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_602/timeФ
'backward_lstm_602/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_602/Max/reduction_indices§
backward_lstm_602/MaxMaxbackward_lstm_602/Cast:y:00backward_lstm_602/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/Maxt
backward_lstm_602/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_602/sub/yШ
backward_lstm_602/subSubbackward_lstm_602/Max:output:0 backward_lstm_602/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/subЮ
backward_lstm_602/Sub_1Subbackward_lstm_602/sub:z:0backward_lstm_602/Cast:y:0*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_602/Sub_1І
backward_lstm_602/zeros_like	ZerosLike*backward_lstm_602/lstm_cell_1808/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_602/zeros_like£
*backward_lstm_602/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2,
*backward_lstm_602/while/maximum_iterationsО
$backward_lstm_602/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_602/while/loop_counter®	
backward_lstm_602/whileWhile-backward_lstm_602/while/loop_counter:output:03backward_lstm_602/while/maximum_iterations:output:0backward_lstm_602/time:output:0*backward_lstm_602/TensorArrayV2_1:handle:0 backward_lstm_602/zeros_like:y:0 backward_lstm_602/zeros:output:0"backward_lstm_602/zeros_1:output:0*backward_lstm_602/strided_slice_1:output:0Ibackward_lstm_602/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_602/Sub_1:z:0?backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resourceAbackward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource@backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource*
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
%backward_lstm_602_while_body_60937977*1
cond)R'
%backward_lstm_602_while_cond_60937976*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
backward_lstm_602/whileў
Bbackward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2D
Bbackward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeє
4backward_lstm_602/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_602/while:output:3Kbackward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype026
4backward_lstm_602/TensorArrayV2Stack/TensorListStack•
'backward_lstm_602/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2)
'backward_lstm_602/strided_slice_3/stack†
)backward_lstm_602/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_602/strided_slice_3/stack_1†
)backward_lstm_602/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_3/stack_2Ж
!backward_lstm_602/strided_slice_3StridedSlice=backward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_602/strided_slice_3/stack:output:02backward_lstm_602/strided_slice_3/stack_1:output:02backward_lstm_602/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2#
!backward_lstm_602/strided_slice_3Э
"backward_lstm_602/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_602/transpose_1/permц
backward_lstm_602/transpose_1	Transpose=backward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_602/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
backward_lstm_602/transpose_1К
backward_lstm_602/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_602/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisƒ
concatConcatV2)forward_lstm_602/strided_slice_3:output:0*backward_lstm_602/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp8^backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp7^backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp9^backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp^backward_lstm_602/while7^forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp6^forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp8^forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp^forward_lstm_602/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€:€€€€€€€€€: : : : : : 2r
7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp2p
6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp2t
8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp22
backward_lstm_602/whilebackward_lstm_602/while2p
6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp2n
5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp2r
7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp20
forward_lstm_602/whileforward_lstm_602/while:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
л	
Щ
4__inference_bidirectional_602_layer_call_fn_60936701
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
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_609352242
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
ѕ_
µ
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_60939092
inputs_0@
-lstm_cell_1808_matmul_readvariableop_resource:	»B
/lstm_cell_1808_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1808_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1808/BiasAdd/ReadVariableOpҐ$lstm_cell_1808/MatMul/ReadVariableOpҐ&lstm_cell_1808/MatMul_1/ReadVariableOpҐwhileF
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
$lstm_cell_1808/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1808_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1808/MatMul/ReadVariableOp≥
lstm_cell_1808/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/MatMulЅ
&lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1808_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1808/MatMul_1/ReadVariableOpѓ
lstm_cell_1808/MatMul_1MatMulzeros:output:0.lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/MatMul_1®
lstm_cell_1808/addAddV2lstm_cell_1808/MatMul:product:0!lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/addЇ
%lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1808_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1808/BiasAdd/ReadVariableOpµ
lstm_cell_1808/BiasAddBiasAddlstm_cell_1808/add:z:0-lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/BiasAddВ
lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1808/split/split_dimы
lstm_cell_1808/splitSplit'lstm_cell_1808/split/split_dim:output:0lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1808/splitМ
lstm_cell_1808/SigmoidSigmoidlstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/SigmoidР
lstm_cell_1808/Sigmoid_1Sigmoidlstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Sigmoid_1С
lstm_cell_1808/mulMullstm_cell_1808/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/mulГ
lstm_cell_1808/ReluRelulstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Relu§
lstm_cell_1808/mul_1Mullstm_cell_1808/Sigmoid:y:0!lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/mul_1Щ
lstm_cell_1808/add_1AddV2lstm_cell_1808/mul:z:0lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/add_1Р
lstm_cell_1808/Sigmoid_2Sigmoidlstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Sigmoid_2В
lstm_cell_1808/Relu_1Relulstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Relu_1®
lstm_cell_1808/mul_2Mullstm_cell_1808/Sigmoid_2:y:0#lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1808_matmul_readvariableop_resource/lstm_cell_1808_matmul_1_readvariableop_resource.lstm_cell_1808_biasadd_readvariableop_resource*
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
while_body_60939008*
condR
while_cond_60939007*K
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
NoOpNoOp&^lstm_cell_1808/BiasAdd/ReadVariableOp%^lstm_cell_1808/MatMul/ReadVariableOp'^lstm_cell_1808/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1808/BiasAdd/ReadVariableOp%lstm_cell_1808/BiasAdd/ReadVariableOp2L
$lstm_cell_1808/MatMul/ReadVariableOp$lstm_cell_1808/MatMul/ReadVariableOp2P
&lstm_cell_1808/MatMul_1/ReadVariableOp&lstm_cell_1808/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
Њ
ъ
1__inference_lstm_cell_1807_layer_call_fn_60939432

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
L__inference_lstm_cell_1807_layer_call_and_return_conditional_losses_609338492
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
зe
Ћ
$forward_lstm_602_while_body_60937440>
:forward_lstm_602_while_forward_lstm_602_while_loop_counterD
@forward_lstm_602_while_forward_lstm_602_while_maximum_iterations&
"forward_lstm_602_while_placeholder(
$forward_lstm_602_while_placeholder_1(
$forward_lstm_602_while_placeholder_2(
$forward_lstm_602_while_placeholder_3(
$forward_lstm_602_while_placeholder_4=
9forward_lstm_602_while_forward_lstm_602_strided_slice_1_0y
uforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_602_while_greater_forward_lstm_602_cast_0Y
Fforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0:	»[
Hforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0:	2»V
Gforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0:	»#
forward_lstm_602_while_identity%
!forward_lstm_602_while_identity_1%
!forward_lstm_602_while_identity_2%
!forward_lstm_602_while_identity_3%
!forward_lstm_602_while_identity_4%
!forward_lstm_602_while_identity_5%
!forward_lstm_602_while_identity_6;
7forward_lstm_602_while_forward_lstm_602_strided_slice_1w
sforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_602_while_greater_forward_lstm_602_castW
Dforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource:	»Y
Fforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource:	2»T
Eforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource:	»ИҐ<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpҐ;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOpҐ=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpе
Hforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2J
Hforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeє
:forward_lstm_602/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_602_while_placeholderQforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02<
:forward_lstm_602/while/TensorArrayV2Read/TensorListGetItem’
forward_lstm_602/while/GreaterGreater6forward_lstm_602_while_greater_forward_lstm_602_cast_0"forward_lstm_602_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2 
forward_lstm_602/while/GreaterВ
;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOpFforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02=
;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp°
,forward_lstm_602/while/lstm_cell_1807/MatMulMatMulAforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem:item:0Cforward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2.
,forward_lstm_602/while/lstm_cell_1807/MatMulИ
=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOpHforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02?
=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpК
.forward_lstm_602/while/lstm_cell_1807/MatMul_1MatMul$forward_lstm_602_while_placeholder_3Eforward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.forward_lstm_602/while/lstm_cell_1807/MatMul_1Д
)forward_lstm_602/while/lstm_cell_1807/addAddV26forward_lstm_602/while/lstm_cell_1807/MatMul:product:08forward_lstm_602/while/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)forward_lstm_602/while/lstm_cell_1807/addБ
<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOpGforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02>
<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpС
-forward_lstm_602/while/lstm_cell_1807/BiasAddBiasAdd-forward_lstm_602/while/lstm_cell_1807/add:z:0Dforward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-forward_lstm_602/while/lstm_cell_1807/BiasAdd∞
5forward_lstm_602/while/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5forward_lstm_602/while/lstm_cell_1807/split/split_dim„
+forward_lstm_602/while/lstm_cell_1807/splitSplit>forward_lstm_602/while/lstm_cell_1807/split/split_dim:output:06forward_lstm_602/while/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2-
+forward_lstm_602/while/lstm_cell_1807/split—
-forward_lstm_602/while/lstm_cell_1807/SigmoidSigmoid4forward_lstm_602/while/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-forward_lstm_602/while/lstm_cell_1807/Sigmoid’
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_1Sigmoid4forward_lstm_602/while/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_1к
)forward_lstm_602/while/lstm_cell_1807/mulMul3forward_lstm_602/while/lstm_cell_1807/Sigmoid_1:y:0$forward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_602/while/lstm_cell_1807/mul»
*forward_lstm_602/while/lstm_cell_1807/ReluRelu4forward_lstm_602/while/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22,
*forward_lstm_602/while/lstm_cell_1807/ReluА
+forward_lstm_602/while/lstm_cell_1807/mul_1Mul1forward_lstm_602/while/lstm_cell_1807/Sigmoid:y:08forward_lstm_602/while/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_602/while/lstm_cell_1807/mul_1х
+forward_lstm_602/while/lstm_cell_1807/add_1AddV2-forward_lstm_602/while/lstm_cell_1807/mul:z:0/forward_lstm_602/while/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_602/while/lstm_cell_1807/add_1’
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_2Sigmoid4forward_lstm_602/while/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_2«
,forward_lstm_602/while/lstm_cell_1807/Relu_1Relu/forward_lstm_602/while/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,forward_lstm_602/while/lstm_cell_1807/Relu_1Д
+forward_lstm_602/while/lstm_cell_1807/mul_2Mul3forward_lstm_602/while/lstm_cell_1807/Sigmoid_2:y:0:forward_lstm_602/while/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_602/while/lstm_cell_1807/mul_2х
forward_lstm_602/while/SelectSelect"forward_lstm_602/while/Greater:z:0/forward_lstm_602/while/lstm_cell_1807/mul_2:z:0$forward_lstm_602_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/while/Selectщ
forward_lstm_602/while/Select_1Select"forward_lstm_602/while/Greater:z:0/forward_lstm_602/while/lstm_cell_1807/mul_2:z:0$forward_lstm_602_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_602/while/Select_1щ
forward_lstm_602/while/Select_2Select"forward_lstm_602/while/Greater:z:0/forward_lstm_602/while/lstm_cell_1807/add_1:z:0$forward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_602/while/Select_2Ѓ
;forward_lstm_602/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_602_while_placeholder_1"forward_lstm_602_while_placeholder&forward_lstm_602/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_602/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_602/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_602/while/add/y≠
forward_lstm_602/while/addAddV2"forward_lstm_602_while_placeholder%forward_lstm_602/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/while/addВ
forward_lstm_602/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_602/while/add_1/yЋ
forward_lstm_602/while/add_1AddV2:forward_lstm_602_while_forward_lstm_602_while_loop_counter'forward_lstm_602/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/while/add_1ѓ
forward_lstm_602/while/IdentityIdentity forward_lstm_602/while/add_1:z:0^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_602/while/Identity”
!forward_lstm_602/while/Identity_1Identity@forward_lstm_602_while_forward_lstm_602_while_maximum_iterations^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_602/while/Identity_1±
!forward_lstm_602/while/Identity_2Identityforward_lstm_602/while/add:z:0^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_602/while/Identity_2ё
!forward_lstm_602/while/Identity_3IdentityKforward_lstm_602/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_602/while/Identity_3 
!forward_lstm_602/while/Identity_4Identity&forward_lstm_602/while/Select:output:0^forward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_602/while/Identity_4ћ
!forward_lstm_602/while/Identity_5Identity(forward_lstm_602/while/Select_1:output:0^forward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_602/while/Identity_5ћ
!forward_lstm_602/while/Identity_6Identity(forward_lstm_602/while/Select_2:output:0^forward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_602/while/Identity_6є
forward_lstm_602/while/NoOpNoOp=^forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp<^forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp>^forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_602/while/NoOp"t
7forward_lstm_602_while_forward_lstm_602_strided_slice_19forward_lstm_602_while_forward_lstm_602_strided_slice_1_0"n
4forward_lstm_602_while_greater_forward_lstm_602_cast6forward_lstm_602_while_greater_forward_lstm_602_cast_0"K
forward_lstm_602_while_identity(forward_lstm_602/while/Identity:output:0"O
!forward_lstm_602_while_identity_1*forward_lstm_602/while/Identity_1:output:0"O
!forward_lstm_602_while_identity_2*forward_lstm_602/while/Identity_2:output:0"O
!forward_lstm_602_while_identity_3*forward_lstm_602/while/Identity_3:output:0"O
!forward_lstm_602_while_identity_4*forward_lstm_602/while/Identity_4:output:0"O
!forward_lstm_602_while_identity_5*forward_lstm_602/while/Identity_5:output:0"O
!forward_lstm_602_while_identity_6*forward_lstm_602/while/Identity_6:output:0"Р
Eforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resourceGforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0"Т
Fforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resourceHforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0"О
Dforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resourceFforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0"м
sforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensoruforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2|
<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp2z
;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp2~
=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
3__inference_forward_lstm_602_layer_call_fn_60938105
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
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_609337862
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
–]
і
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_60938289
inputs_0@
-lstm_cell_1807_matmul_readvariableop_resource:	»B
/lstm_cell_1807_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1807_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1807/BiasAdd/ReadVariableOpҐ$lstm_cell_1807/MatMul/ReadVariableOpҐ&lstm_cell_1807/MatMul_1/ReadVariableOpҐwhileF
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
$lstm_cell_1807/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1807_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1807/MatMul/ReadVariableOp≥
lstm_cell_1807/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/MatMulЅ
&lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1807_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1807/MatMul_1/ReadVariableOpѓ
lstm_cell_1807/MatMul_1MatMulzeros:output:0.lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/MatMul_1®
lstm_cell_1807/addAddV2lstm_cell_1807/MatMul:product:0!lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/addЇ
%lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1807_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1807/BiasAdd/ReadVariableOpµ
lstm_cell_1807/BiasAddBiasAddlstm_cell_1807/add:z:0-lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/BiasAddВ
lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1807/split/split_dimы
lstm_cell_1807/splitSplit'lstm_cell_1807/split/split_dim:output:0lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1807/splitМ
lstm_cell_1807/SigmoidSigmoidlstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/SigmoidР
lstm_cell_1807/Sigmoid_1Sigmoidlstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Sigmoid_1С
lstm_cell_1807/mulMullstm_cell_1807/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/mulГ
lstm_cell_1807/ReluRelulstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Relu§
lstm_cell_1807/mul_1Mullstm_cell_1807/Sigmoid:y:0!lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/mul_1Щ
lstm_cell_1807/add_1AddV2lstm_cell_1807/mul:z:0lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/add_1Р
lstm_cell_1807/Sigmoid_2Sigmoidlstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Sigmoid_2В
lstm_cell_1807/Relu_1Relulstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Relu_1®
lstm_cell_1807/mul_2Mullstm_cell_1807/Sigmoid_2:y:0#lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1807_matmul_readvariableop_resource/lstm_cell_1807_matmul_1_readvariableop_resource.lstm_cell_1807_biasadd_readvariableop_resource*
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
while_body_60938205*
condR
while_cond_60938204*K
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
NoOpNoOp&^lstm_cell_1807/BiasAdd/ReadVariableOp%^lstm_cell_1807/MatMul/ReadVariableOp'^lstm_cell_1807/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1807/BiasAdd/ReadVariableOp%lstm_cell_1807/BiasAdd/ReadVariableOp2L
$lstm_cell_1807/MatMul/ReadVariableOp$lstm_cell_1807/MatMul/ReadVariableOp2P
&lstm_cell_1807/MatMul_1/ReadVariableOp&lstm_cell_1807/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
Ѓ

•
4__inference_bidirectional_602_layer_call_fn_60936754

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
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_609365042
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
э
µ
%backward_lstm_602_while_cond_60937271@
<backward_lstm_602_while_backward_lstm_602_while_loop_counterF
Bbackward_lstm_602_while_backward_lstm_602_while_maximum_iterations'
#backward_lstm_602_while_placeholder)
%backward_lstm_602_while_placeholder_1)
%backward_lstm_602_while_placeholder_2)
%backward_lstm_602_while_placeholder_3B
>backward_lstm_602_while_less_backward_lstm_602_strided_slice_1Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60937271___redundant_placeholder0Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60937271___redundant_placeholder1Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60937271___redundant_placeholder2Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60937271___redundant_placeholder3$
 backward_lstm_602_while_identity
 
backward_lstm_602/while/LessLess#backward_lstm_602_while_placeholder>backward_lstm_602_while_less_backward_lstm_602_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_602/while/LessУ
 backward_lstm_602/while/IdentityIdentity backward_lstm_602/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_602/while/Identity"M
 backward_lstm_602_while_identity)backward_lstm_602/while/Identity:output:0*(
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
…
•
$forward_lstm_602_while_cond_60937439>
:forward_lstm_602_while_forward_lstm_602_while_loop_counterD
@forward_lstm_602_while_forward_lstm_602_while_maximum_iterations&
"forward_lstm_602_while_placeholder(
$forward_lstm_602_while_placeholder_1(
$forward_lstm_602_while_placeholder_2(
$forward_lstm_602_while_placeholder_3(
$forward_lstm_602_while_placeholder_4@
<forward_lstm_602_while_less_forward_lstm_602_strided_slice_1X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60937439___redundant_placeholder0X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60937439___redundant_placeholder1X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60937439___redundant_placeholder2X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60937439___redundant_placeholder3X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60937439___redundant_placeholder4#
forward_lstm_602_while_identity
≈
forward_lstm_602/while/LessLess"forward_lstm_602_while_placeholder<forward_lstm_602_while_less_forward_lstm_602_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_602/while/LessР
forward_lstm_602/while/IdentityIdentityforward_lstm_602/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_602/while/Identity"K
forward_lstm_602_while_identity(forward_lstm_602/while/Identity:output:0*(
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
в
Ѕ
4__inference_backward_lstm_602_layer_call_fn_60938775

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
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_609352132
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
Њ
ъ
1__inference_lstm_cell_1807_layer_call_fn_60939415

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
L__inference_lstm_cell_1807_layer_call_and_return_conditional_losses_609337032
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
оX
Д
$forward_lstm_602_while_body_60936821>
:forward_lstm_602_while_forward_lstm_602_while_loop_counterD
@forward_lstm_602_while_forward_lstm_602_while_maximum_iterations&
"forward_lstm_602_while_placeholder(
$forward_lstm_602_while_placeholder_1(
$forward_lstm_602_while_placeholder_2(
$forward_lstm_602_while_placeholder_3=
9forward_lstm_602_while_forward_lstm_602_strided_slice_1_0y
uforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0Y
Fforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0:	»[
Hforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0:	2»V
Gforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0:	»#
forward_lstm_602_while_identity%
!forward_lstm_602_while_identity_1%
!forward_lstm_602_while_identity_2%
!forward_lstm_602_while_identity_3%
!forward_lstm_602_while_identity_4%
!forward_lstm_602_while_identity_5;
7forward_lstm_602_while_forward_lstm_602_strided_slice_1w
sforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensorW
Dforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource:	»Y
Fforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource:	2»T
Eforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource:	»ИҐ<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpҐ;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOpҐ=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpе
Hforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2J
Hforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shape¬
:forward_lstm_602/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_602_while_placeholderQforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02<
:forward_lstm_602/while/TensorArrayV2Read/TensorListGetItemВ
;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOpFforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02=
;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp°
,forward_lstm_602/while/lstm_cell_1807/MatMulMatMulAforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem:item:0Cforward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2.
,forward_lstm_602/while/lstm_cell_1807/MatMulИ
=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOpHforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02?
=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpК
.forward_lstm_602/while/lstm_cell_1807/MatMul_1MatMul$forward_lstm_602_while_placeholder_2Eforward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.forward_lstm_602/while/lstm_cell_1807/MatMul_1Д
)forward_lstm_602/while/lstm_cell_1807/addAddV26forward_lstm_602/while/lstm_cell_1807/MatMul:product:08forward_lstm_602/while/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)forward_lstm_602/while/lstm_cell_1807/addБ
<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOpGforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02>
<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpС
-forward_lstm_602/while/lstm_cell_1807/BiasAddBiasAdd-forward_lstm_602/while/lstm_cell_1807/add:z:0Dforward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-forward_lstm_602/while/lstm_cell_1807/BiasAdd∞
5forward_lstm_602/while/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5forward_lstm_602/while/lstm_cell_1807/split/split_dim„
+forward_lstm_602/while/lstm_cell_1807/splitSplit>forward_lstm_602/while/lstm_cell_1807/split/split_dim:output:06forward_lstm_602/while/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2-
+forward_lstm_602/while/lstm_cell_1807/split—
-forward_lstm_602/while/lstm_cell_1807/SigmoidSigmoid4forward_lstm_602/while/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-forward_lstm_602/while/lstm_cell_1807/Sigmoid’
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_1Sigmoid4forward_lstm_602/while/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_1к
)forward_lstm_602/while/lstm_cell_1807/mulMul3forward_lstm_602/while/lstm_cell_1807/Sigmoid_1:y:0$forward_lstm_602_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_602/while/lstm_cell_1807/mul»
*forward_lstm_602/while/lstm_cell_1807/ReluRelu4forward_lstm_602/while/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22,
*forward_lstm_602/while/lstm_cell_1807/ReluА
+forward_lstm_602/while/lstm_cell_1807/mul_1Mul1forward_lstm_602/while/lstm_cell_1807/Sigmoid:y:08forward_lstm_602/while/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_602/while/lstm_cell_1807/mul_1х
+forward_lstm_602/while/lstm_cell_1807/add_1AddV2-forward_lstm_602/while/lstm_cell_1807/mul:z:0/forward_lstm_602/while/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_602/while/lstm_cell_1807/add_1’
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_2Sigmoid4forward_lstm_602/while/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_2«
,forward_lstm_602/while/lstm_cell_1807/Relu_1Relu/forward_lstm_602/while/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,forward_lstm_602/while/lstm_cell_1807/Relu_1Д
+forward_lstm_602/while/lstm_cell_1807/mul_2Mul3forward_lstm_602/while/lstm_cell_1807/Sigmoid_2:y:0:forward_lstm_602/while/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_602/while/lstm_cell_1807/mul_2Ј
;forward_lstm_602/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_602_while_placeholder_1"forward_lstm_602_while_placeholder/forward_lstm_602/while/lstm_cell_1807/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_602/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_602/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_602/while/add/y≠
forward_lstm_602/while/addAddV2"forward_lstm_602_while_placeholder%forward_lstm_602/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/while/addВ
forward_lstm_602/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_602/while/add_1/yЋ
forward_lstm_602/while/add_1AddV2:forward_lstm_602_while_forward_lstm_602_while_loop_counter'forward_lstm_602/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/while/add_1ѓ
forward_lstm_602/while/IdentityIdentity forward_lstm_602/while/add_1:z:0^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_602/while/Identity”
!forward_lstm_602/while/Identity_1Identity@forward_lstm_602_while_forward_lstm_602_while_maximum_iterations^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_602/while/Identity_1±
!forward_lstm_602/while/Identity_2Identityforward_lstm_602/while/add:z:0^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_602/while/Identity_2ё
!forward_lstm_602/while/Identity_3IdentityKforward_lstm_602/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_602/while/Identity_3”
!forward_lstm_602/while/Identity_4Identity/forward_lstm_602/while/lstm_cell_1807/mul_2:z:0^forward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_602/while/Identity_4”
!forward_lstm_602/while/Identity_5Identity/forward_lstm_602/while/lstm_cell_1807/add_1:z:0^forward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_602/while/Identity_5є
forward_lstm_602/while/NoOpNoOp=^forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp<^forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp>^forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_602/while/NoOp"t
7forward_lstm_602_while_forward_lstm_602_strided_slice_19forward_lstm_602_while_forward_lstm_602_strided_slice_1_0"K
forward_lstm_602_while_identity(forward_lstm_602/while/Identity:output:0"O
!forward_lstm_602_while_identity_1*forward_lstm_602/while/Identity_1:output:0"O
!forward_lstm_602_while_identity_2*forward_lstm_602/while/Identity_2:output:0"O
!forward_lstm_602_while_identity_3*forward_lstm_602/while/Identity_3:output:0"O
!forward_lstm_602_while_identity_4*forward_lstm_602/while/Identity_4:output:0"O
!forward_lstm_602_while_identity_5*forward_lstm_602/while/Identity_5:output:0"Р
Eforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resourceGforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0"Т
Fforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resourceHforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0"О
Dforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resourceFforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0"м
sforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensoruforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2|
<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp2z
;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp2~
=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
…
•
$forward_lstm_602_while_cond_60935787>
:forward_lstm_602_while_forward_lstm_602_while_loop_counterD
@forward_lstm_602_while_forward_lstm_602_while_maximum_iterations&
"forward_lstm_602_while_placeholder(
$forward_lstm_602_while_placeholder_1(
$forward_lstm_602_while_placeholder_2(
$forward_lstm_602_while_placeholder_3(
$forward_lstm_602_while_placeholder_4@
<forward_lstm_602_while_less_forward_lstm_602_strided_slice_1X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60935787___redundant_placeholder0X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60935787___redundant_placeholder1X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60935787___redundant_placeholder2X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60935787___redundant_placeholder3X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60935787___redundant_placeholder4#
forward_lstm_602_while_identity
≈
forward_lstm_602/while/LessLess"forward_lstm_602_while_placeholder<forward_lstm_602_while_less_forward_lstm_602_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_602/while/LessР
forward_lstm_602/while/IdentityIdentityforward_lstm_602/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_602/while/Identity"K
forward_lstm_602_while_identity(forward_lstm_602/while/Identity:output:0*(
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
я
°
$forward_lstm_602_while_cond_60937122>
:forward_lstm_602_while_forward_lstm_602_while_loop_counterD
@forward_lstm_602_while_forward_lstm_602_while_maximum_iterations&
"forward_lstm_602_while_placeholder(
$forward_lstm_602_while_placeholder_1(
$forward_lstm_602_while_placeholder_2(
$forward_lstm_602_while_placeholder_3@
<forward_lstm_602_while_less_forward_lstm_602_strided_slice_1X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60937122___redundant_placeholder0X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60937122___redundant_placeholder1X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60937122___redundant_placeholder2X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60937122___redundant_placeholder3#
forward_lstm_602_while_identity
≈
forward_lstm_602/while/LessLess"forward_lstm_602_while_placeholder<forward_lstm_602_while_less_forward_lstm_602_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_602/while/LessР
forward_lstm_602/while/IdentityIdentityforward_lstm_602/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_602/while/Identity"K
forward_lstm_602_while_identity(forward_lstm_602/while/Identity:output:0*(
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
L__inference_lstm_cell_1807_layer_call_and_return_conditional_losses_60933703

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
к
Љ
%backward_lstm_602_while_cond_60937618@
<backward_lstm_602_while_backward_lstm_602_while_loop_counterF
Bbackward_lstm_602_while_backward_lstm_602_while_maximum_iterations'
#backward_lstm_602_while_placeholder)
%backward_lstm_602_while_placeholder_1)
%backward_lstm_602_while_placeholder_2)
%backward_lstm_602_while_placeholder_3)
%backward_lstm_602_while_placeholder_4B
>backward_lstm_602_while_less_backward_lstm_602_strided_slice_1Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60937618___redundant_placeholder0Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60937618___redundant_placeholder1Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60937618___redundant_placeholder2Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60937618___redundant_placeholder3Z
Vbackward_lstm_602_while_backward_lstm_602_while_cond_60937618___redundant_placeholder4$
 backward_lstm_602_while_identity
 
backward_lstm_602/while/LessLess#backward_lstm_602_while_placeholder>backward_lstm_602_while_less_backward_lstm_602_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_602/while/LessУ
 backward_lstm_602/while/IdentityIdentity backward_lstm_602/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_602/while/Identity"M
 backward_lstm_602_while_identity)backward_lstm_602/while/Identity:output:0*(
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
ф_
≥
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_60939245

inputs@
-lstm_cell_1808_matmul_readvariableop_resource:	»B
/lstm_cell_1808_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1808_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1808/BiasAdd/ReadVariableOpҐ$lstm_cell_1808/MatMul/ReadVariableOpҐ&lstm_cell_1808/MatMul_1/ReadVariableOpҐwhileD
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
$lstm_cell_1808/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1808_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1808/MatMul/ReadVariableOp≥
lstm_cell_1808/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/MatMulЅ
&lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1808_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1808/MatMul_1/ReadVariableOpѓ
lstm_cell_1808/MatMul_1MatMulzeros:output:0.lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/MatMul_1®
lstm_cell_1808/addAddV2lstm_cell_1808/MatMul:product:0!lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/addЇ
%lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1808_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1808/BiasAdd/ReadVariableOpµ
lstm_cell_1808/BiasAddBiasAddlstm_cell_1808/add:z:0-lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/BiasAddВ
lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1808/split/split_dimы
lstm_cell_1808/splitSplit'lstm_cell_1808/split/split_dim:output:0lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1808/splitМ
lstm_cell_1808/SigmoidSigmoidlstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/SigmoidР
lstm_cell_1808/Sigmoid_1Sigmoidlstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Sigmoid_1С
lstm_cell_1808/mulMullstm_cell_1808/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/mulГ
lstm_cell_1808/ReluRelulstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Relu§
lstm_cell_1808/mul_1Mullstm_cell_1808/Sigmoid:y:0!lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/mul_1Щ
lstm_cell_1808/add_1AddV2lstm_cell_1808/mul:z:0lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/add_1Р
lstm_cell_1808/Sigmoid_2Sigmoidlstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Sigmoid_2В
lstm_cell_1808/Relu_1Relulstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Relu_1®
lstm_cell_1808/mul_2Mullstm_cell_1808/Sigmoid_2:y:0#lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1808_matmul_readvariableop_resource/lstm_cell_1808_matmul_1_readvariableop_resource.lstm_cell_1808_biasadd_readvariableop_resource*
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
while_body_60939161*
condR
while_cond_60939160*K
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
NoOpNoOp&^lstm_cell_1808/BiasAdd/ReadVariableOp%^lstm_cell_1808/MatMul/ReadVariableOp'^lstm_cell_1808/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1808/BiasAdd/ReadVariableOp%lstm_cell_1808/BiasAdd/ReadVariableOp2L
$lstm_cell_1808/MatMul/ReadVariableOp$lstm_cell_1808/MatMul/ReadVariableOp2P
&lstm_cell_1808/MatMul_1/ReadVariableOp&lstm_cell_1808/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Є
£
L__inference_sequential_602_layer_call_and_return_conditional_losses_60936096

inputs
inputs_1	-
bidirectional_602_60936065:	»-
bidirectional_602_60936067:	2»)
bidirectional_602_60936069:	»-
bidirectional_602_60936071:	»-
bidirectional_602_60936073:	2»)
bidirectional_602_60936075:	»$
dense_602_60936090:d 
dense_602_60936092:
identityИҐ)bidirectional_602/StatefulPartitionedCallҐ!dense_602/StatefulPartitionedCall 
)bidirectional_602/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_602_60936065bidirectional_602_60936067bidirectional_602_60936069bidirectional_602_60936071bidirectional_602_60936073bidirectional_602_60936075*
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
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_609360642+
)bidirectional_602/StatefulPartitionedCallЋ
!dense_602/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_602/StatefulPartitionedCall:output:0dense_602_60936090dense_602_60936092*
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
G__inference_dense_602_layer_call_and_return_conditional_losses_609360892#
!dense_602/StatefulPartitionedCallЕ
IdentityIdentity*dense_602/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityЮ
NoOpNoOp*^bidirectional_602/StatefulPartitionedCall"^dense_602/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€:€€€€€€€€€: : : : : : : : 2V
)bidirectional_602/StatefulPartitionedCall)bidirectional_602/StatefulPartitionedCall2F
!dense_602/StatefulPartitionedCall!dense_602/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
м]
≤
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_60935578

inputs@
-lstm_cell_1807_matmul_readvariableop_resource:	»B
/lstm_cell_1807_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1807_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1807/BiasAdd/ReadVariableOpҐ$lstm_cell_1807/MatMul/ReadVariableOpҐ&lstm_cell_1807/MatMul_1/ReadVariableOpҐwhileD
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
$lstm_cell_1807/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1807_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1807/MatMul/ReadVariableOp≥
lstm_cell_1807/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/MatMulЅ
&lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1807_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1807/MatMul_1/ReadVariableOpѓ
lstm_cell_1807/MatMul_1MatMulzeros:output:0.lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/MatMul_1®
lstm_cell_1807/addAddV2lstm_cell_1807/MatMul:product:0!lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/addЇ
%lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1807_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1807/BiasAdd/ReadVariableOpµ
lstm_cell_1807/BiasAddBiasAddlstm_cell_1807/add:z:0-lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/BiasAddВ
lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1807/split/split_dimы
lstm_cell_1807/splitSplit'lstm_cell_1807/split/split_dim:output:0lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1807/splitМ
lstm_cell_1807/SigmoidSigmoidlstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/SigmoidР
lstm_cell_1807/Sigmoid_1Sigmoidlstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Sigmoid_1С
lstm_cell_1807/mulMullstm_cell_1807/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/mulГ
lstm_cell_1807/ReluRelulstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Relu§
lstm_cell_1807/mul_1Mullstm_cell_1807/Sigmoid:y:0!lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/mul_1Щ
lstm_cell_1807/add_1AddV2lstm_cell_1807/mul:z:0lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/add_1Р
lstm_cell_1807/Sigmoid_2Sigmoidlstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Sigmoid_2В
lstm_cell_1807/Relu_1Relulstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Relu_1®
lstm_cell_1807/mul_2Mullstm_cell_1807/Sigmoid_2:y:0#lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1807_matmul_readvariableop_resource/lstm_cell_1807_matmul_1_readvariableop_resource.lstm_cell_1807_biasadd_readvariableop_resource*
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
while_body_60935494*
condR
while_cond_60935493*K
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
NoOpNoOp&^lstm_cell_1807/BiasAdd/ReadVariableOp%^lstm_cell_1807/MatMul/ReadVariableOp'^lstm_cell_1807/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1807/BiasAdd/ReadVariableOp%lstm_cell_1807/BiasAdd/ReadVariableOp2L
$lstm_cell_1807/MatMul/ReadVariableOp$lstm_cell_1807/MatMul/ReadVariableOp2P
&lstm_cell_1807/MatMul_1/ReadVariableOp&lstm_cell_1807/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ѕ@
д
while_body_60935494
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1807_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1807_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1807_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1807_matmul_readvariableop_resource:	»H
5while_lstm_cell_1807_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1807_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1807/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1807/MatMul/ReadVariableOpҐ,while/lstm_cell_1807/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1807_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1807/MatMul/ReadVariableOpЁ
while/lstm_cell_1807/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/MatMul’
,while/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1807_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1807/MatMul_1/ReadVariableOp∆
while/lstm_cell_1807/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/MatMul_1ј
while/lstm_cell_1807/addAddV2%while/lstm_cell_1807/MatMul:product:0'while/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/addќ
+while/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1807_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1807/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1807/BiasAddBiasAddwhile/lstm_cell_1807/add:z:03while/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1807/BiasAddО
$while/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1807/split/split_dimУ
while/lstm_cell_1807/splitSplit-while/lstm_cell_1807/split/split_dim:output:0%while/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1807/splitЮ
while/lstm_cell_1807/SigmoidSigmoid#while/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/SigmoidҐ
while/lstm_cell_1807/Sigmoid_1Sigmoid#while/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1807/Sigmoid_1¶
while/lstm_cell_1807/mulMul"while/lstm_cell_1807/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/mulХ
while/lstm_cell_1807/ReluRelu#while/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/ReluЉ
while/lstm_cell_1807/mul_1Mul while/lstm_cell_1807/Sigmoid:y:0'while/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/mul_1±
while/lstm_cell_1807/add_1AddV2while/lstm_cell_1807/mul:z:0while/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/add_1Ґ
while/lstm_cell_1807/Sigmoid_2Sigmoid#while/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1807/Sigmoid_2Ф
while/lstm_cell_1807/Relu_1Reluwhile/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/Relu_1ј
while/lstm_cell_1807/mul_2Mul"while/lstm_cell_1807/Sigmoid_2:y:0)while/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1807/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1807/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1807/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1807/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1807/BiasAdd/ReadVariableOp+^while/lstm_cell_1807/MatMul/ReadVariableOp-^while/lstm_cell_1807/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1807_biasadd_readvariableop_resource6while_lstm_cell_1807_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1807_matmul_1_readvariableop_resource7while_lstm_cell_1807_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1807_matmul_readvariableop_resource5while_lstm_cell_1807_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1807/BiasAdd/ReadVariableOp+while/lstm_cell_1807/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1807/MatMul/ReadVariableOp*while/lstm_cell_1807/MatMul/ReadVariableOp2\
,while/lstm_cell_1807/MatMul_1/ReadVariableOp,while/lstm_cell_1807/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
ёю
п
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_60937056
inputs_0Q
>forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource:	»S
@forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource:	2»N
?forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource:	»R
?backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resource:	»T
Abackward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource:	2»O
@backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource:	»
identityИҐ7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpҐ6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpҐ8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpҐbackward_lstm_602/whileҐ6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpҐ5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpҐ7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpҐforward_lstm_602/whileh
forward_lstm_602/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_602/ShapeЦ
$forward_lstm_602/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_602/strided_slice/stackЪ
&forward_lstm_602/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_602/strided_slice/stack_1Ъ
&forward_lstm_602/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_602/strided_slice/stack_2»
forward_lstm_602/strided_sliceStridedSliceforward_lstm_602/Shape:output:0-forward_lstm_602/strided_slice/stack:output:0/forward_lstm_602/strided_slice/stack_1:output:0/forward_lstm_602/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_602/strided_slice~
forward_lstm_602/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_602/zeros/mul/y∞
forward_lstm_602/zeros/mulMul'forward_lstm_602/strided_slice:output:0%forward_lstm_602/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros/mulБ
forward_lstm_602/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
forward_lstm_602/zeros/Less/yЂ
forward_lstm_602/zeros/LessLessforward_lstm_602/zeros/mul:z:0&forward_lstm_602/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros/LessД
forward_lstm_602/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_602/zeros/packed/1«
forward_lstm_602/zeros/packedPack'forward_lstm_602/strided_slice:output:0(forward_lstm_602/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_602/zeros/packedЕ
forward_lstm_602/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_602/zeros/Constє
forward_lstm_602/zerosFill&forward_lstm_602/zeros/packed:output:0%forward_lstm_602/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/zerosВ
forward_lstm_602/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_602/zeros_1/mul/yґ
forward_lstm_602/zeros_1/mulMul'forward_lstm_602/strided_slice:output:0'forward_lstm_602/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros_1/mulЕ
forward_lstm_602/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2!
forward_lstm_602/zeros_1/Less/y≥
forward_lstm_602/zeros_1/LessLess forward_lstm_602/zeros_1/mul:z:0(forward_lstm_602/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/zeros_1/LessИ
!forward_lstm_602/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_602/zeros_1/packed/1Ќ
forward_lstm_602/zeros_1/packedPack'forward_lstm_602/strided_slice:output:0*forward_lstm_602/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_602/zeros_1/packedЙ
forward_lstm_602/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_602/zeros_1/ConstЅ
forward_lstm_602/zeros_1Fill(forward_lstm_602/zeros_1/packed:output:0'forward_lstm_602/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/zeros_1Ч
forward_lstm_602/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_602/transpose/permЅ
forward_lstm_602/transpose	Transposeinputs_0(forward_lstm_602/transpose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
forward_lstm_602/transposeВ
forward_lstm_602/Shape_1Shapeforward_lstm_602/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_602/Shape_1Ъ
&forward_lstm_602/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_602/strided_slice_1/stackЮ
(forward_lstm_602/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_1/stack_1Ю
(forward_lstm_602/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_1/stack_2‘
 forward_lstm_602/strided_slice_1StridedSlice!forward_lstm_602/Shape_1:output:0/forward_lstm_602/strided_slice_1/stack:output:01forward_lstm_602/strided_slice_1/stack_1:output:01forward_lstm_602/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_602/strided_slice_1І
,forward_lstm_602/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,forward_lstm_602/TensorArrayV2/element_shapeц
forward_lstm_602/TensorArrayV2TensorListReserve5forward_lstm_602/TensorArrayV2/element_shape:output:0)forward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_602/TensorArrayV2б
Fforward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2H
Fforward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeЉ
8forward_lstm_602/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_602/transpose:y:0Oforward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_602/TensorArrayUnstack/TensorListFromTensorЪ
&forward_lstm_602/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_602/strided_slice_2/stackЮ
(forward_lstm_602/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_2/stack_1Ю
(forward_lstm_602/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_2/stack_2л
 forward_lstm_602/strided_slice_2StridedSliceforward_lstm_602/transpose:y:0/forward_lstm_602/strided_slice_2/stack:output:01forward_lstm_602/strided_slice_2/stack_1:output:01forward_lstm_602/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2"
 forward_lstm_602/strided_slice_2о
5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOp>forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype027
5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOpч
&forward_lstm_602/lstm_cell_1807/MatMulMatMul)forward_lstm_602/strided_slice_2:output:0=forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2(
&forward_lstm_602/lstm_cell_1807/MatMulф
7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOp@forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype029
7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOpу
(forward_lstm_602/lstm_cell_1807/MatMul_1MatMulforward_lstm_602/zeros:output:0?forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(forward_lstm_602/lstm_cell_1807/MatMul_1м
#forward_lstm_602/lstm_cell_1807/addAddV20forward_lstm_602/lstm_cell_1807/MatMul:product:02forward_lstm_602/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2%
#forward_lstm_602/lstm_cell_1807/addн
6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOp?forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype028
6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOpщ
'forward_lstm_602/lstm_cell_1807/BiasAddBiasAdd'forward_lstm_602/lstm_cell_1807/add:z:0>forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'forward_lstm_602/lstm_cell_1807/BiasAdd§
/forward_lstm_602/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/forward_lstm_602/lstm_cell_1807/split/split_dimњ
%forward_lstm_602/lstm_cell_1807/splitSplit8forward_lstm_602/lstm_cell_1807/split/split_dim:output:00forward_lstm_602/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2'
%forward_lstm_602/lstm_cell_1807/splitњ
'forward_lstm_602/lstm_cell_1807/SigmoidSigmoid.forward_lstm_602/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'forward_lstm_602/lstm_cell_1807/Sigmoid√
)forward_lstm_602/lstm_cell_1807/Sigmoid_1Sigmoid.forward_lstm_602/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_602/lstm_cell_1807/Sigmoid_1’
#forward_lstm_602/lstm_cell_1807/mulMul-forward_lstm_602/lstm_cell_1807/Sigmoid_1:y:0!forward_lstm_602/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22%
#forward_lstm_602/lstm_cell_1807/mulґ
$forward_lstm_602/lstm_cell_1807/ReluRelu.forward_lstm_602/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22&
$forward_lstm_602/lstm_cell_1807/Reluи
%forward_lstm_602/lstm_cell_1807/mul_1Mul+forward_lstm_602/lstm_cell_1807/Sigmoid:y:02forward_lstm_602/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_602/lstm_cell_1807/mul_1Ё
%forward_lstm_602/lstm_cell_1807/add_1AddV2'forward_lstm_602/lstm_cell_1807/mul:z:0)forward_lstm_602/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_602/lstm_cell_1807/add_1√
)forward_lstm_602/lstm_cell_1807/Sigmoid_2Sigmoid.forward_lstm_602/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_602/lstm_cell_1807/Sigmoid_2µ
&forward_lstm_602/lstm_cell_1807/Relu_1Relu)forward_lstm_602/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&forward_lstm_602/lstm_cell_1807/Relu_1м
%forward_lstm_602/lstm_cell_1807/mul_2Mul-forward_lstm_602/lstm_cell_1807/Sigmoid_2:y:04forward_lstm_602/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_602/lstm_cell_1807/mul_2±
.forward_lstm_602/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   20
.forward_lstm_602/TensorArrayV2_1/element_shapeь
 forward_lstm_602/TensorArrayV2_1TensorListReserve7forward_lstm_602/TensorArrayV2_1/element_shape:output:0)forward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_602/TensorArrayV2_1p
forward_lstm_602/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_602/time°
)forward_lstm_602/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2+
)forward_lstm_602/while/maximum_iterationsМ
#forward_lstm_602/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_602/while/loop_counterФ
forward_lstm_602/whileWhile,forward_lstm_602/while/loop_counter:output:02forward_lstm_602/while/maximum_iterations:output:0forward_lstm_602/time:output:0)forward_lstm_602/TensorArrayV2_1:handle:0forward_lstm_602/zeros:output:0!forward_lstm_602/zeros_1:output:0)forward_lstm_602/strided_slice_1:output:0Hforward_lstm_602/TensorArrayUnstack/TensorListFromTensor:output_handle:0>forward_lstm_602_lstm_cell_1807_matmul_readvariableop_resource@forward_lstm_602_lstm_cell_1807_matmul_1_readvariableop_resource?forward_lstm_602_lstm_cell_1807_biasadd_readvariableop_resource*
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
$forward_lstm_602_while_body_60936821*0
cond(R&
$forward_lstm_602_while_cond_60936820*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
forward_lstm_602/while„
Aforward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2C
Aforward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeµ
3forward_lstm_602/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_602/while:output:3Jforward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype025
3forward_lstm_602/TensorArrayV2Stack/TensorListStack£
&forward_lstm_602/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2(
&forward_lstm_602/strided_slice_3/stackЮ
(forward_lstm_602/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_602/strided_slice_3/stack_1Ю
(forward_lstm_602/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_602/strided_slice_3/stack_2А
 forward_lstm_602/strided_slice_3StridedSlice<forward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_602/strided_slice_3/stack:output:01forward_lstm_602/strided_slice_3/stack_1:output:01forward_lstm_602/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2"
 forward_lstm_602/strided_slice_3Ы
!forward_lstm_602/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_602/transpose_1/permт
forward_lstm_602/transpose_1	Transpose<forward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_602/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
forward_lstm_602/transpose_1И
forward_lstm_602/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_602/runtimej
backward_lstm_602/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_602/ShapeШ
%backward_lstm_602/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_602/strided_slice/stackЬ
'backward_lstm_602/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_602/strided_slice/stack_1Ь
'backward_lstm_602/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_602/strided_slice/stack_2ќ
backward_lstm_602/strided_sliceStridedSlice backward_lstm_602/Shape:output:0.backward_lstm_602/strided_slice/stack:output:00backward_lstm_602/strided_slice/stack_1:output:00backward_lstm_602/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_602/strided_sliceА
backward_lstm_602/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_602/zeros/mul/yі
backward_lstm_602/zeros/mulMul(backward_lstm_602/strided_slice:output:0&backward_lstm_602/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/zeros/mulГ
backward_lstm_602/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
backward_lstm_602/zeros/Less/yѓ
backward_lstm_602/zeros/LessLessbackward_lstm_602/zeros/mul:z:0'backward_lstm_602/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/zeros/LessЖ
 backward_lstm_602/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_602/zeros/packed/1Ћ
backward_lstm_602/zeros/packedPack(backward_lstm_602/strided_slice:output:0)backward_lstm_602/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_602/zeros/packedЗ
backward_lstm_602/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_602/zeros/Constљ
backward_lstm_602/zerosFill'backward_lstm_602/zeros/packed:output:0&backward_lstm_602/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_602/zerosД
backward_lstm_602/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_602/zeros_1/mul/yЇ
backward_lstm_602/zeros_1/mulMul(backward_lstm_602/strided_slice:output:0(backward_lstm_602/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_602/zeros_1/mulЗ
 backward_lstm_602/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 backward_lstm_602/zeros_1/Less/yЈ
backward_lstm_602/zeros_1/LessLess!backward_lstm_602/zeros_1/mul:z:0)backward_lstm_602/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_602/zeros_1/LessК
"backward_lstm_602/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_602/zeros_1/packed/1—
 backward_lstm_602/zeros_1/packedPack(backward_lstm_602/strided_slice:output:0+backward_lstm_602/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_602/zeros_1/packedЛ
backward_lstm_602/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_602/zeros_1/Const≈
backward_lstm_602/zeros_1Fill)backward_lstm_602/zeros_1/packed:output:0(backward_lstm_602/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_602/zeros_1Щ
 backward_lstm_602/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_602/transpose/permƒ
backward_lstm_602/transpose	Transposeinputs_0)backward_lstm_602/transpose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
backward_lstm_602/transposeЕ
backward_lstm_602/Shape_1Shapebackward_lstm_602/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_602/Shape_1Ь
'backward_lstm_602/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_602/strided_slice_1/stack†
)backward_lstm_602/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_1/stack_1†
)backward_lstm_602/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_1/stack_2Џ
!backward_lstm_602/strided_slice_1StridedSlice"backward_lstm_602/Shape_1:output:00backward_lstm_602/strided_slice_1/stack:output:02backward_lstm_602/strided_slice_1/stack_1:output:02backward_lstm_602/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_602/strided_slice_1©
-backward_lstm_602/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-backward_lstm_602/TensorArrayV2/element_shapeъ
backward_lstm_602/TensorArrayV2TensorListReserve6backward_lstm_602/TensorArrayV2/element_shape:output:0*backward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_602/TensorArrayV2О
 backward_lstm_602/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_602/ReverseV2/axisџ
backward_lstm_602/ReverseV2	ReverseV2backward_lstm_602/transpose:y:0)backward_lstm_602/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
backward_lstm_602/ReverseV2г
Gbackward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2I
Gbackward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape≈
9backward_lstm_602/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_602/ReverseV2:output:0Pbackward_lstm_602/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_602/TensorArrayUnstack/TensorListFromTensorЬ
'backward_lstm_602/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_602/strided_slice_2/stack†
)backward_lstm_602/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_2/stack_1†
)backward_lstm_602/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_2/stack_2с
!backward_lstm_602/strided_slice_2StridedSlicebackward_lstm_602/transpose:y:00backward_lstm_602/strided_slice_2/stack:output:02backward_lstm_602/strided_slice_2/stack_1:output:02backward_lstm_602/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2#
!backward_lstm_602/strided_slice_2с
6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOp?backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype028
6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOpы
'backward_lstm_602/lstm_cell_1808/MatMulMatMul*backward_lstm_602/strided_slice_2:output:0>backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'backward_lstm_602/lstm_cell_1808/MatMulч
8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOpAbackward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02:
8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOpч
)backward_lstm_602/lstm_cell_1808/MatMul_1MatMul backward_lstm_602/zeros:output:0@backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)backward_lstm_602/lstm_cell_1808/MatMul_1р
$backward_lstm_602/lstm_cell_1808/addAddV21backward_lstm_602/lstm_cell_1808/MatMul:product:03backward_lstm_602/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2&
$backward_lstm_602/lstm_cell_1808/addр
7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOp@backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype029
7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOpэ
(backward_lstm_602/lstm_cell_1808/BiasAddBiasAdd(backward_lstm_602/lstm_cell_1808/add:z:0?backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(backward_lstm_602/lstm_cell_1808/BiasAdd¶
0backward_lstm_602/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0backward_lstm_602/lstm_cell_1808/split/split_dim√
&backward_lstm_602/lstm_cell_1808/splitSplit9backward_lstm_602/lstm_cell_1808/split/split_dim:output:01backward_lstm_602/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2(
&backward_lstm_602/lstm_cell_1808/split¬
(backward_lstm_602/lstm_cell_1808/SigmoidSigmoid/backward_lstm_602/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(backward_lstm_602/lstm_cell_1808/Sigmoid∆
*backward_lstm_602/lstm_cell_1808/Sigmoid_1Sigmoid/backward_lstm_602/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_602/lstm_cell_1808/Sigmoid_1ў
$backward_lstm_602/lstm_cell_1808/mulMul.backward_lstm_602/lstm_cell_1808/Sigmoid_1:y:0"backward_lstm_602/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$backward_lstm_602/lstm_cell_1808/mulє
%backward_lstm_602/lstm_cell_1808/ReluRelu/backward_lstm_602/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22'
%backward_lstm_602/lstm_cell_1808/Reluм
&backward_lstm_602/lstm_cell_1808/mul_1Mul,backward_lstm_602/lstm_cell_1808/Sigmoid:y:03backward_lstm_602/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_602/lstm_cell_1808/mul_1б
&backward_lstm_602/lstm_cell_1808/add_1AddV2(backward_lstm_602/lstm_cell_1808/mul:z:0*backward_lstm_602/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_602/lstm_cell_1808/add_1∆
*backward_lstm_602/lstm_cell_1808/Sigmoid_2Sigmoid/backward_lstm_602/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_602/lstm_cell_1808/Sigmoid_2Є
'backward_lstm_602/lstm_cell_1808/Relu_1Relu*backward_lstm_602/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'backward_lstm_602/lstm_cell_1808/Relu_1р
&backward_lstm_602/lstm_cell_1808/mul_2Mul.backward_lstm_602/lstm_cell_1808/Sigmoid_2:y:05backward_lstm_602/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_602/lstm_cell_1808/mul_2≥
/backward_lstm_602/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   21
/backward_lstm_602/TensorArrayV2_1/element_shapeА
!backward_lstm_602/TensorArrayV2_1TensorListReserve8backward_lstm_602/TensorArrayV2_1/element_shape:output:0*backward_lstm_602/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_602/TensorArrayV2_1r
backward_lstm_602/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_602/time£
*backward_lstm_602/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2,
*backward_lstm_602/while/maximum_iterationsО
$backward_lstm_602/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_602/while/loop_counter£
backward_lstm_602/whileWhile-backward_lstm_602/while/loop_counter:output:03backward_lstm_602/while/maximum_iterations:output:0backward_lstm_602/time:output:0*backward_lstm_602/TensorArrayV2_1:handle:0 backward_lstm_602/zeros:output:0"backward_lstm_602/zeros_1:output:0*backward_lstm_602/strided_slice_1:output:0Ibackward_lstm_602/TensorArrayUnstack/TensorListFromTensor:output_handle:0?backward_lstm_602_lstm_cell_1808_matmul_readvariableop_resourceAbackward_lstm_602_lstm_cell_1808_matmul_1_readvariableop_resource@backward_lstm_602_lstm_cell_1808_biasadd_readvariableop_resource*
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
%backward_lstm_602_while_body_60936970*1
cond)R'
%backward_lstm_602_while_cond_60936969*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
backward_lstm_602/whileў
Bbackward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2D
Bbackward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shapeє
4backward_lstm_602/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_602/while:output:3Kbackward_lstm_602/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype026
4backward_lstm_602/TensorArrayV2Stack/TensorListStack•
'backward_lstm_602/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2)
'backward_lstm_602/strided_slice_3/stack†
)backward_lstm_602/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_602/strided_slice_3/stack_1†
)backward_lstm_602/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_602/strided_slice_3/stack_2Ж
!backward_lstm_602/strided_slice_3StridedSlice=backward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_602/strided_slice_3/stack:output:02backward_lstm_602/strided_slice_3/stack_1:output:02backward_lstm_602/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2#
!backward_lstm_602/strided_slice_3Э
"backward_lstm_602/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_602/transpose_1/permц
backward_lstm_602/transpose_1	Transpose=backward_lstm_602/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_602/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
backward_lstm_602/transpose_1К
backward_lstm_602/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_602/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisƒ
concatConcatV2)forward_lstm_602/strided_slice_3:output:0*backward_lstm_602/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp8^backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp7^backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp9^backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp^backward_lstm_602/while7^forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp6^forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp8^forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp^forward_lstm_602/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : 2r
7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp7backward_lstm_602/lstm_cell_1808/BiasAdd/ReadVariableOp2p
6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp6backward_lstm_602/lstm_cell_1808/MatMul/ReadVariableOp2t
8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp8backward_lstm_602/lstm_cell_1808/MatMul_1/ReadVariableOp22
backward_lstm_602/whilebackward_lstm_602/while2p
6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp6forward_lstm_602/lstm_cell_1807/BiasAdd/ReadVariableOp2n
5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp5forward_lstm_602/lstm_cell_1807/MatMul/ReadVariableOp2r
7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp7forward_lstm_602/lstm_cell_1807/MatMul_1/ReadVariableOp20
forward_lstm_602/whileforward_lstm_602/while:g c
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
 

Ќ
&__inference_signature_wrapper_60936684

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
#__inference__wrapped_model_609336282
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
м]
≤
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_60938742

inputs@
-lstm_cell_1807_matmul_readvariableop_resource:	»B
/lstm_cell_1807_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1807_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1807/BiasAdd/ReadVariableOpҐ$lstm_cell_1807/MatMul/ReadVariableOpҐ&lstm_cell_1807/MatMul_1/ReadVariableOpҐwhileD
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
$lstm_cell_1807/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1807_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1807/MatMul/ReadVariableOp≥
lstm_cell_1807/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/MatMulЅ
&lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1807_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1807/MatMul_1/ReadVariableOpѓ
lstm_cell_1807/MatMul_1MatMulzeros:output:0.lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/MatMul_1®
lstm_cell_1807/addAddV2lstm_cell_1807/MatMul:product:0!lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/addЇ
%lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1807_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1807/BiasAdd/ReadVariableOpµ
lstm_cell_1807/BiasAddBiasAddlstm_cell_1807/add:z:0-lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1807/BiasAddВ
lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1807/split/split_dimы
lstm_cell_1807/splitSplit'lstm_cell_1807/split/split_dim:output:0lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1807/splitМ
lstm_cell_1807/SigmoidSigmoidlstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/SigmoidР
lstm_cell_1807/Sigmoid_1Sigmoidlstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Sigmoid_1С
lstm_cell_1807/mulMullstm_cell_1807/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/mulГ
lstm_cell_1807/ReluRelulstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Relu§
lstm_cell_1807/mul_1Mullstm_cell_1807/Sigmoid:y:0!lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/mul_1Щ
lstm_cell_1807/add_1AddV2lstm_cell_1807/mul:z:0lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/add_1Р
lstm_cell_1807/Sigmoid_2Sigmoidlstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Sigmoid_2В
lstm_cell_1807/Relu_1Relulstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/Relu_1®
lstm_cell_1807/mul_2Mullstm_cell_1807/Sigmoid_2:y:0#lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1807/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1807_matmul_readvariableop_resource/lstm_cell_1807_matmul_1_readvariableop_resource.lstm_cell_1807_biasadd_readvariableop_resource*
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
while_body_60938658*
condR
while_cond_60938657*K
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
NoOpNoOp&^lstm_cell_1807/BiasAdd/ReadVariableOp%^lstm_cell_1807/MatMul/ReadVariableOp'^lstm_cell_1807/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1807/BiasAdd/ReadVariableOp%lstm_cell_1807/BiasAdd/ReadVariableOp2L
$lstm_cell_1807/MatMul/ReadVariableOp$lstm_cell_1807/MatMul/ReadVariableOp2P
&lstm_cell_1807/MatMul_1/ReadVariableOp&lstm_cell_1807/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
я
Ќ
while_cond_60935320
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60935320___redundant_placeholder06
2while_while_cond_60935320___redundant_placeholder16
2while_while_cond_60935320___redundant_placeholder26
2while_while_cond_60935320___redundant_placeholder3
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
ѕ_
µ
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_60938939
inputs_0@
-lstm_cell_1808_matmul_readvariableop_resource:	»B
/lstm_cell_1808_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1808_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1808/BiasAdd/ReadVariableOpҐ$lstm_cell_1808/MatMul/ReadVariableOpҐ&lstm_cell_1808/MatMul_1/ReadVariableOpҐwhileF
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
$lstm_cell_1808/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1808_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1808/MatMul/ReadVariableOp≥
lstm_cell_1808/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/MatMulЅ
&lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1808_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1808/MatMul_1/ReadVariableOpѓ
lstm_cell_1808/MatMul_1MatMulzeros:output:0.lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/MatMul_1®
lstm_cell_1808/addAddV2lstm_cell_1808/MatMul:product:0!lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/addЇ
%lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1808_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1808/BiasAdd/ReadVariableOpµ
lstm_cell_1808/BiasAddBiasAddlstm_cell_1808/add:z:0-lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/BiasAddВ
lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1808/split/split_dimы
lstm_cell_1808/splitSplit'lstm_cell_1808/split/split_dim:output:0lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1808/splitМ
lstm_cell_1808/SigmoidSigmoidlstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/SigmoidР
lstm_cell_1808/Sigmoid_1Sigmoidlstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Sigmoid_1С
lstm_cell_1808/mulMullstm_cell_1808/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/mulГ
lstm_cell_1808/ReluRelulstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Relu§
lstm_cell_1808/mul_1Mullstm_cell_1808/Sigmoid:y:0!lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/mul_1Щ
lstm_cell_1808/add_1AddV2lstm_cell_1808/mul:z:0lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/add_1Р
lstm_cell_1808/Sigmoid_2Sigmoidlstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Sigmoid_2В
lstm_cell_1808/Relu_1Relulstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Relu_1®
lstm_cell_1808/mul_2Mullstm_cell_1808/Sigmoid_2:y:0#lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1808_matmul_readvariableop_resource/lstm_cell_1808_matmul_1_readvariableop_resource.lstm_cell_1808_biasadd_readvariableop_resource*
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
while_body_60938855*
condR
while_cond_60938854*K
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
NoOpNoOp&^lstm_cell_1808/BiasAdd/ReadVariableOp%^lstm_cell_1808/MatMul/ReadVariableOp'^lstm_cell_1808/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1808/BiasAdd/ReadVariableOp%lstm_cell_1808/BiasAdd/ReadVariableOp2L
$lstm_cell_1808/MatMul/ReadVariableOp$lstm_cell_1808/MatMul/ReadVariableOp2P
&lstm_cell_1808/MatMul_1/ReadVariableOp&lstm_cell_1808/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
€
К
L__inference_lstm_cell_1807_layer_call_and_return_conditional_losses_60939496

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
я
°
$forward_lstm_602_while_cond_60936820>
:forward_lstm_602_while_forward_lstm_602_while_loop_counterD
@forward_lstm_602_while_forward_lstm_602_while_maximum_iterations&
"forward_lstm_602_while_placeholder(
$forward_lstm_602_while_placeholder_1(
$forward_lstm_602_while_placeholder_2(
$forward_lstm_602_while_placeholder_3@
<forward_lstm_602_while_less_forward_lstm_602_strided_slice_1X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60936820___redundant_placeholder0X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60936820___redundant_placeholder1X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60936820___redundant_placeholder2X
Tforward_lstm_602_while_forward_lstm_602_while_cond_60936820___redundant_placeholder3#
forward_lstm_602_while_identity
≈
forward_lstm_602/while/LessLess"forward_lstm_602_while_placeholder<forward_lstm_602_while_less_forward_lstm_602_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_602/while/LessР
forward_lstm_602/while/IdentityIdentityforward_lstm_602/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_602/while/Identity"K
forward_lstm_602_while_identity(forward_lstm_602/while/Identity:output:0*(
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
while_cond_60939313
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60939313___redundant_placeholder06
2while_while_cond_60939313___redundant_placeholder16
2while_while_cond_60939313___redundant_placeholder26
2while_while_cond_60939313___redundant_placeholder3
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
while_body_60935129
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1808_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1808_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1808_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1808_matmul_readvariableop_resource:	»H
5while_lstm_cell_1808_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1808_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1808/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1808/MatMul/ReadVariableOpҐ,while/lstm_cell_1808/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1808_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1808/MatMul/ReadVariableOpЁ
while/lstm_cell_1808/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/MatMul’
,while/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1808_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1808/MatMul_1/ReadVariableOp∆
while/lstm_cell_1808/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/MatMul_1ј
while/lstm_cell_1808/addAddV2%while/lstm_cell_1808/MatMul:product:0'while/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/addќ
+while/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1808_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1808/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1808/BiasAddBiasAddwhile/lstm_cell_1808/add:z:03while/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/BiasAddО
$while/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1808/split/split_dimУ
while/lstm_cell_1808/splitSplit-while/lstm_cell_1808/split/split_dim:output:0%while/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1808/splitЮ
while/lstm_cell_1808/SigmoidSigmoid#while/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/SigmoidҐ
while/lstm_cell_1808/Sigmoid_1Sigmoid#while/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1808/Sigmoid_1¶
while/lstm_cell_1808/mulMul"while/lstm_cell_1808/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/mulХ
while/lstm_cell_1808/ReluRelu#while/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/ReluЉ
while/lstm_cell_1808/mul_1Mul while/lstm_cell_1808/Sigmoid:y:0'while/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/mul_1±
while/lstm_cell_1808/add_1AddV2while/lstm_cell_1808/mul:z:0while/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/add_1Ґ
while/lstm_cell_1808/Sigmoid_2Sigmoid#while/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1808/Sigmoid_2Ф
while/lstm_cell_1808/Relu_1Reluwhile/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/Relu_1ј
while/lstm_cell_1808/mul_2Mul"while/lstm_cell_1808/Sigmoid_2:y:0)while/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1808/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1808/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1808/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1808/BiasAdd/ReadVariableOp+^while/lstm_cell_1808/MatMul/ReadVariableOp-^while/lstm_cell_1808/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1808_biasadd_readvariableop_resource6while_lstm_cell_1808_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1808_matmul_1_readvariableop_resource7while_lstm_cell_1808_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1808_matmul_readvariableop_resource5while_lstm_cell_1808_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1808/BiasAdd/ReadVariableOp+while/lstm_cell_1808/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1808/MatMul/ReadVariableOp*while/lstm_cell_1808/MatMul/ReadVariableOp2\
,while/lstm_cell_1808/MatMul_1/ReadVariableOp,while/lstm_cell_1808/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
ч
И
L__inference_lstm_cell_1808_layer_call_and_return_conditional_losses_60934335

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
зe
Ћ
$forward_lstm_602_while_body_60937798>
:forward_lstm_602_while_forward_lstm_602_while_loop_counterD
@forward_lstm_602_while_forward_lstm_602_while_maximum_iterations&
"forward_lstm_602_while_placeholder(
$forward_lstm_602_while_placeholder_1(
$forward_lstm_602_while_placeholder_2(
$forward_lstm_602_while_placeholder_3(
$forward_lstm_602_while_placeholder_4=
9forward_lstm_602_while_forward_lstm_602_strided_slice_1_0y
uforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_602_while_greater_forward_lstm_602_cast_0Y
Fforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0:	»[
Hforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0:	2»V
Gforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0:	»#
forward_lstm_602_while_identity%
!forward_lstm_602_while_identity_1%
!forward_lstm_602_while_identity_2%
!forward_lstm_602_while_identity_3%
!forward_lstm_602_while_identity_4%
!forward_lstm_602_while_identity_5%
!forward_lstm_602_while_identity_6;
7forward_lstm_602_while_forward_lstm_602_strided_slice_1w
sforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_602_while_greater_forward_lstm_602_castW
Dforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource:	»Y
Fforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource:	2»T
Eforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource:	»ИҐ<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpҐ;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOpҐ=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpе
Hforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2J
Hforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shapeє
:forward_lstm_602/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_602_while_placeholderQforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02<
:forward_lstm_602/while/TensorArrayV2Read/TensorListGetItem’
forward_lstm_602/while/GreaterGreater6forward_lstm_602_while_greater_forward_lstm_602_cast_0"forward_lstm_602_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2 
forward_lstm_602/while/GreaterВ
;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOpReadVariableOpFforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02=
;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp°
,forward_lstm_602/while/lstm_cell_1807/MatMulMatMulAforward_lstm_602/while/TensorArrayV2Read/TensorListGetItem:item:0Cforward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2.
,forward_lstm_602/while/lstm_cell_1807/MatMulИ
=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpReadVariableOpHforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02?
=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOpК
.forward_lstm_602/while/lstm_cell_1807/MatMul_1MatMul$forward_lstm_602_while_placeholder_3Eforward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.forward_lstm_602/while/lstm_cell_1807/MatMul_1Д
)forward_lstm_602/while/lstm_cell_1807/addAddV26forward_lstm_602/while/lstm_cell_1807/MatMul:product:08forward_lstm_602/while/lstm_cell_1807/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)forward_lstm_602/while/lstm_cell_1807/addБ
<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpReadVariableOpGforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02>
<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOpС
-forward_lstm_602/while/lstm_cell_1807/BiasAddBiasAdd-forward_lstm_602/while/lstm_cell_1807/add:z:0Dforward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-forward_lstm_602/while/lstm_cell_1807/BiasAdd∞
5forward_lstm_602/while/lstm_cell_1807/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5forward_lstm_602/while/lstm_cell_1807/split/split_dim„
+forward_lstm_602/while/lstm_cell_1807/splitSplit>forward_lstm_602/while/lstm_cell_1807/split/split_dim:output:06forward_lstm_602/while/lstm_cell_1807/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2-
+forward_lstm_602/while/lstm_cell_1807/split—
-forward_lstm_602/while/lstm_cell_1807/SigmoidSigmoid4forward_lstm_602/while/lstm_cell_1807/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-forward_lstm_602/while/lstm_cell_1807/Sigmoid’
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_1Sigmoid4forward_lstm_602/while/lstm_cell_1807/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_1к
)forward_lstm_602/while/lstm_cell_1807/mulMul3forward_lstm_602/while/lstm_cell_1807/Sigmoid_1:y:0$forward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_602/while/lstm_cell_1807/mul»
*forward_lstm_602/while/lstm_cell_1807/ReluRelu4forward_lstm_602/while/lstm_cell_1807/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22,
*forward_lstm_602/while/lstm_cell_1807/ReluА
+forward_lstm_602/while/lstm_cell_1807/mul_1Mul1forward_lstm_602/while/lstm_cell_1807/Sigmoid:y:08forward_lstm_602/while/lstm_cell_1807/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_602/while/lstm_cell_1807/mul_1х
+forward_lstm_602/while/lstm_cell_1807/add_1AddV2-forward_lstm_602/while/lstm_cell_1807/mul:z:0/forward_lstm_602/while/lstm_cell_1807/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_602/while/lstm_cell_1807/add_1’
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_2Sigmoid4forward_lstm_602/while/lstm_cell_1807/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_602/while/lstm_cell_1807/Sigmoid_2«
,forward_lstm_602/while/lstm_cell_1807/Relu_1Relu/forward_lstm_602/while/lstm_cell_1807/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,forward_lstm_602/while/lstm_cell_1807/Relu_1Д
+forward_lstm_602/while/lstm_cell_1807/mul_2Mul3forward_lstm_602/while/lstm_cell_1807/Sigmoid_2:y:0:forward_lstm_602/while/lstm_cell_1807/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_602/while/lstm_cell_1807/mul_2х
forward_lstm_602/while/SelectSelect"forward_lstm_602/while/Greater:z:0/forward_lstm_602/while/lstm_cell_1807/mul_2:z:0$forward_lstm_602_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_602/while/Selectщ
forward_lstm_602/while/Select_1Select"forward_lstm_602/while/Greater:z:0/forward_lstm_602/while/lstm_cell_1807/mul_2:z:0$forward_lstm_602_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_602/while/Select_1щ
forward_lstm_602/while/Select_2Select"forward_lstm_602/while/Greater:z:0/forward_lstm_602/while/lstm_cell_1807/add_1:z:0$forward_lstm_602_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_602/while/Select_2Ѓ
;forward_lstm_602/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_602_while_placeholder_1"forward_lstm_602_while_placeholder&forward_lstm_602/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_602/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_602/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_602/while/add/y≠
forward_lstm_602/while/addAddV2"forward_lstm_602_while_placeholder%forward_lstm_602/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/while/addВ
forward_lstm_602/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_602/while/add_1/yЋ
forward_lstm_602/while/add_1AddV2:forward_lstm_602_while_forward_lstm_602_while_loop_counter'forward_lstm_602/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_602/while/add_1ѓ
forward_lstm_602/while/IdentityIdentity forward_lstm_602/while/add_1:z:0^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_602/while/Identity”
!forward_lstm_602/while/Identity_1Identity@forward_lstm_602_while_forward_lstm_602_while_maximum_iterations^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_602/while/Identity_1±
!forward_lstm_602/while/Identity_2Identityforward_lstm_602/while/add:z:0^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_602/while/Identity_2ё
!forward_lstm_602/while/Identity_3IdentityKforward_lstm_602/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_602/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_602/while/Identity_3 
!forward_lstm_602/while/Identity_4Identity&forward_lstm_602/while/Select:output:0^forward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_602/while/Identity_4ћ
!forward_lstm_602/while/Identity_5Identity(forward_lstm_602/while/Select_1:output:0^forward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_602/while/Identity_5ћ
!forward_lstm_602/while/Identity_6Identity(forward_lstm_602/while/Select_2:output:0^forward_lstm_602/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_602/while/Identity_6є
forward_lstm_602/while/NoOpNoOp=^forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp<^forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp>^forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_602/while/NoOp"t
7forward_lstm_602_while_forward_lstm_602_strided_slice_19forward_lstm_602_while_forward_lstm_602_strided_slice_1_0"n
4forward_lstm_602_while_greater_forward_lstm_602_cast6forward_lstm_602_while_greater_forward_lstm_602_cast_0"K
forward_lstm_602_while_identity(forward_lstm_602/while/Identity:output:0"O
!forward_lstm_602_while_identity_1*forward_lstm_602/while/Identity_1:output:0"O
!forward_lstm_602_while_identity_2*forward_lstm_602/while/Identity_2:output:0"O
!forward_lstm_602_while_identity_3*forward_lstm_602/while/Identity_3:output:0"O
!forward_lstm_602_while_identity_4*forward_lstm_602/while/Identity_4:output:0"O
!forward_lstm_602_while_identity_5*forward_lstm_602/while/Identity_5:output:0"O
!forward_lstm_602_while_identity_6*forward_lstm_602/while/Identity_6:output:0"Р
Eforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resourceGforward_lstm_602_while_lstm_cell_1807_biasadd_readvariableop_resource_0"Т
Fforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resourceHforward_lstm_602_while_lstm_cell_1807_matmul_1_readvariableop_resource_0"О
Dforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resourceFforward_lstm_602_while_lstm_cell_1807_matmul_readvariableop_resource_0"м
sforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensoruforward_lstm_602_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_602_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2|
<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp<forward_lstm_602/while/lstm_cell_1807/BiasAdd/ReadVariableOp2z
;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp;forward_lstm_602/while/lstm_cell_1807/MatMul/ReadVariableOp2~
=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp=forward_lstm_602/while/lstm_cell_1807/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_body_60934561
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_lstm_cell_1808_60934585_0:	»2
while_lstm_cell_1808_60934587_0:	2».
while_lstm_cell_1808_60934589_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_lstm_cell_1808_60934585:	»0
while_lstm_cell_1808_60934587:	2»,
while_lstm_cell_1808_60934589:	»ИҐ,while/lstm_cell_1808/StatefulPartitionedCall√
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
,while/lstm_cell_1808/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1808_60934585_0while_lstm_cell_1808_60934587_0while_lstm_cell_1808_60934589_0*
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
L__inference_lstm_cell_1808_layer_call_and_return_conditional_losses_609344812.
,while/lstm_cell_1808/StatefulPartitionedCallщ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder5while/lstm_cell_1808/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity5while/lstm_cell_1808/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4¶
while/Identity_5Identity5while/lstm_cell_1808/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5Й

while/NoOpNoOp-^while/lstm_cell_1808/StatefulPartitionedCall*"
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
while_lstm_cell_1808_60934585while_lstm_cell_1808_60934585_0"@
while_lstm_cell_1808_60934587while_lstm_cell_1808_60934587_0"@
while_lstm_cell_1808_60934589while_lstm_cell_1808_60934589_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2\
,while/lstm_cell_1808/StatefulPartitionedCall,while/lstm_cell_1808/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
while_cond_60938506
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60938506___redundant_placeholder06
2while_while_cond_60938506___redundant_placeholder16
2while_while_cond_60938506___redundant_placeholder26
2while_while_cond_60938506___redundant_placeholder3
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
while_cond_60939007
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60939007___redundant_placeholder06
2while_while_cond_60939007___redundant_placeholder16
2while_while_cond_60939007___redundant_placeholder26
2while_while_cond_60939007___redundant_placeholder3
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
Ѓ

•
4__inference_bidirectional_602_layer_call_fn_60936736

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
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_609360642
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
Њ
ъ
1__inference_lstm_cell_1808_layer_call_fn_60939530

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
L__inference_lstm_cell_1808_layer_call_and_return_conditional_losses_609344812
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
Є
£
L__inference_sequential_602_layer_call_and_return_conditional_losses_60936654

inputs
inputs_1	-
bidirectional_602_60936635:	»-
bidirectional_602_60936637:	2»)
bidirectional_602_60936639:	»-
bidirectional_602_60936641:	»-
bidirectional_602_60936643:	2»)
bidirectional_602_60936645:	»$
dense_602_60936648:d 
dense_602_60936650:
identityИҐ)bidirectional_602/StatefulPartitionedCallҐ!dense_602/StatefulPartitionedCall 
)bidirectional_602/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_602_60936635bidirectional_602_60936637bidirectional_602_60936639bidirectional_602_60936641bidirectional_602_60936643bidirectional_602_60936645*
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
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_609365042+
)bidirectional_602/StatefulPartitionedCallЋ
!dense_602/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_602/StatefulPartitionedCall:output:0dense_602_60936648dense_602_60936650*
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
G__inference_dense_602_layer_call_and_return_conditional_losses_609360892#
!dense_602/StatefulPartitionedCallЕ
IdentityIdentity*dense_602/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityЮ
NoOpNoOp*^bidirectional_602/StatefulPartitionedCall"^dense_602/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€:€€€€€€€€€: : : : : : : : 2V
)bidirectional_602/StatefulPartitionedCall)bidirectional_602/StatefulPartitionedCall2F
!dense_602/StatefulPartitionedCall!dense_602/StatefulPartitionedCall:O K
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
while_cond_60934348
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60934348___redundant_placeholder06
2while_while_cond_60934348___redundant_placeholder16
2while_while_cond_60934348___redundant_placeholder26
2while_while_cond_60934348___redundant_placeholder3
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
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_60935626

inputs,
forward_lstm_602_60935609:	»,
forward_lstm_602_60935611:	2»(
forward_lstm_602_60935613:	»-
backward_lstm_602_60935616:	»-
backward_lstm_602_60935618:	2»)
backward_lstm_602_60935620:	»
identityИҐ)backward_lstm_602/StatefulPartitionedCallҐ(forward_lstm_602/StatefulPartitionedCallя
(forward_lstm_602/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_602_60935609forward_lstm_602_60935611forward_lstm_602_60935613*
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
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_609355782*
(forward_lstm_602/StatefulPartitionedCallе
)backward_lstm_602/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_602_60935616backward_lstm_602_60935618backward_lstm_602_60935620*
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
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_609354052+
)backward_lstm_602/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis‘
concatConcatV21forward_lstm_602/StatefulPartitionedCall:output:02backward_lstm_602/StatefulPartitionedCall:output:0concat/axis:output:0*
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
NoOpNoOp*^backward_lstm_602/StatefulPartitionedCall)^forward_lstm_602/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : 2V
)backward_lstm_602/StatefulPartitionedCall)backward_lstm_602/StatefulPartitionedCall2T
(forward_lstm_602/StatefulPartitionedCall(forward_lstm_602/StatefulPartitionedCall:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
€
К
L__inference_lstm_cell_1808_layer_call_and_return_conditional_losses_60939562

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
ф_
≥
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_60935213

inputs@
-lstm_cell_1808_matmul_readvariableop_resource:	»B
/lstm_cell_1808_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1808_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1808/BiasAdd/ReadVariableOpҐ$lstm_cell_1808/MatMul/ReadVariableOpҐ&lstm_cell_1808/MatMul_1/ReadVariableOpҐwhileD
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
$lstm_cell_1808/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1808_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1808/MatMul/ReadVariableOp≥
lstm_cell_1808/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/MatMulЅ
&lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1808_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1808/MatMul_1/ReadVariableOpѓ
lstm_cell_1808/MatMul_1MatMulzeros:output:0.lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/MatMul_1®
lstm_cell_1808/addAddV2lstm_cell_1808/MatMul:product:0!lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/addЇ
%lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1808_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1808/BiasAdd/ReadVariableOpµ
lstm_cell_1808/BiasAddBiasAddlstm_cell_1808/add:z:0-lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1808/BiasAddВ
lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1808/split/split_dimы
lstm_cell_1808/splitSplit'lstm_cell_1808/split/split_dim:output:0lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1808/splitМ
lstm_cell_1808/SigmoidSigmoidlstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/SigmoidР
lstm_cell_1808/Sigmoid_1Sigmoidlstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Sigmoid_1С
lstm_cell_1808/mulMullstm_cell_1808/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/mulГ
lstm_cell_1808/ReluRelulstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Relu§
lstm_cell_1808/mul_1Mullstm_cell_1808/Sigmoid:y:0!lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/mul_1Щ
lstm_cell_1808/add_1AddV2lstm_cell_1808/mul:z:0lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/add_1Р
lstm_cell_1808/Sigmoid_2Sigmoidlstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Sigmoid_2В
lstm_cell_1808/Relu_1Relulstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/Relu_1®
lstm_cell_1808/mul_2Mullstm_cell_1808/Sigmoid_2:y:0#lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1808/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1808_matmul_readvariableop_resource/lstm_cell_1808_matmul_1_readvariableop_resource.lstm_cell_1808_biasadd_readvariableop_resource*
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
while_body_60935129*
condR
while_cond_60935128*K
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
NoOpNoOp&^lstm_cell_1808/BiasAdd/ReadVariableOp%^lstm_cell_1808/MatMul/ReadVariableOp'^lstm_cell_1808/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1808/BiasAdd/ReadVariableOp%lstm_cell_1808/BiasAdd/ReadVariableOp2L
$lstm_cell_1808/MatMul/ReadVariableOp$lstm_cell_1808/MatMul/ReadVariableOp2P
&lstm_cell_1808/MatMul_1/ReadVariableOp&lstm_cell_1808/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
я
Ќ
while_cond_60934560
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60934560___redundant_placeholder06
2while_while_cond_60934560___redundant_placeholder16
2while_while_cond_60934560___redundant_placeholder26
2while_while_cond_60934560___redundant_placeholder3
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
З
ш
G__inference_dense_602_layer_call_and_return_conditional_losses_60938094

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
я
Ќ
while_cond_60938204
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60938204___redundant_placeholder06
2while_while_cond_60938204___redundant_placeholder16
2while_while_cond_60938204___redundant_placeholder26
2while_while_cond_60938204___redundant_placeholder3
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
‘
¬
3__inference_forward_lstm_602_layer_call_fn_60938116
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
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_609339962
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
ѕ@
д
while_body_60935321
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1808_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1808_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1808_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1808_matmul_readvariableop_resource:	»H
5while_lstm_cell_1808_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1808_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1808/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1808/MatMul/ReadVariableOpҐ,while/lstm_cell_1808/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1808/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1808_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1808/MatMul/ReadVariableOpЁ
while/lstm_cell_1808/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1808/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/MatMul’
,while/lstm_cell_1808/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1808_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1808/MatMul_1/ReadVariableOp∆
while/lstm_cell_1808/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1808/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/MatMul_1ј
while/lstm_cell_1808/addAddV2%while/lstm_cell_1808/MatMul:product:0'while/lstm_cell_1808/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/addќ
+while/lstm_cell_1808/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1808_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1808/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1808/BiasAddBiasAddwhile/lstm_cell_1808/add:z:03while/lstm_cell_1808/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1808/BiasAddО
$while/lstm_cell_1808/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1808/split/split_dimУ
while/lstm_cell_1808/splitSplit-while/lstm_cell_1808/split/split_dim:output:0%while/lstm_cell_1808/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1808/splitЮ
while/lstm_cell_1808/SigmoidSigmoid#while/lstm_cell_1808/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/SigmoidҐ
while/lstm_cell_1808/Sigmoid_1Sigmoid#while/lstm_cell_1808/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1808/Sigmoid_1¶
while/lstm_cell_1808/mulMul"while/lstm_cell_1808/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/mulХ
while/lstm_cell_1808/ReluRelu#while/lstm_cell_1808/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/ReluЉ
while/lstm_cell_1808/mul_1Mul while/lstm_cell_1808/Sigmoid:y:0'while/lstm_cell_1808/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/mul_1±
while/lstm_cell_1808/add_1AddV2while/lstm_cell_1808/mul:z:0while/lstm_cell_1808/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/add_1Ґ
while/lstm_cell_1808/Sigmoid_2Sigmoid#while/lstm_cell_1808/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1808/Sigmoid_2Ф
while/lstm_cell_1808/Relu_1Reluwhile/lstm_cell_1808/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/Relu_1ј
while/lstm_cell_1808/mul_2Mul"while/lstm_cell_1808/Sigmoid_2:y:0)while/lstm_cell_1808/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1808/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1808/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1808/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1808/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1808/BiasAdd/ReadVariableOp+^while/lstm_cell_1808/MatMul/ReadVariableOp-^while/lstm_cell_1808/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1808_biasadd_readvariableop_resource6while_lstm_cell_1808_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1808_matmul_1_readvariableop_resource7while_lstm_cell_1808_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1808_matmul_readvariableop_resource5while_lstm_cell_1808_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1808/BiasAdd/ReadVariableOp+while/lstm_cell_1808/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1808/MatMul/ReadVariableOp*while/lstm_cell_1808/MatMul/ReadVariableOp2\
,while/lstm_cell_1808/MatMul_1/ReadVariableOp,while/lstm_cell_1808/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
“Є
ё 
$__inference__traced_restore_60939862
file_prefix3
!assignvariableop_dense_602_kernel:d/
!assignvariableop_1_dense_602_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: ^
Kassignvariableop_7_bidirectional_602_forward_lstm_602_lstm_cell_1807_kernel:	»h
Uassignvariableop_8_bidirectional_602_forward_lstm_602_lstm_cell_1807_recurrent_kernel:	2»X
Iassignvariableop_9_bidirectional_602_forward_lstm_602_lstm_cell_1807_bias:	»`
Massignvariableop_10_bidirectional_602_backward_lstm_602_lstm_cell_1808_kernel:	»j
Wassignvariableop_11_bidirectional_602_backward_lstm_602_lstm_cell_1808_recurrent_kernel:	2»Z
Kassignvariableop_12_bidirectional_602_backward_lstm_602_lstm_cell_1808_bias:	»#
assignvariableop_13_total: #
assignvariableop_14_count: =
+assignvariableop_15_adam_dense_602_kernel_m:d7
)assignvariableop_16_adam_dense_602_bias_m:f
Sassignvariableop_17_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_kernel_m:	»p
]assignvariableop_18_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_recurrent_kernel_m:	2»`
Qassignvariableop_19_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_bias_m:	»g
Tassignvariableop_20_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_kernel_m:	»q
^assignvariableop_21_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_recurrent_kernel_m:	2»a
Rassignvariableop_22_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_bias_m:	»=
+assignvariableop_23_adam_dense_602_kernel_v:d7
)assignvariableop_24_adam_dense_602_bias_v:f
Sassignvariableop_25_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_kernel_v:	»p
]assignvariableop_26_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_recurrent_kernel_v:	2»`
Qassignvariableop_27_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_bias_v:	»g
Tassignvariableop_28_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_kernel_v:	»q
^assignvariableop_29_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_recurrent_kernel_v:	2»a
Rassignvariableop_30_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_bias_v:	»@
.assignvariableop_31_adam_dense_602_kernel_vhat:d:
,assignvariableop_32_adam_dense_602_bias_vhat:i
Vassignvariableop_33_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_kernel_vhat:	»s
`assignvariableop_34_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_recurrent_kernel_vhat:	2»c
Tassignvariableop_35_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_bias_vhat:	»j
Wassignvariableop_36_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_kernel_vhat:	»t
aassignvariableop_37_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_recurrent_kernel_vhat:	2»d
Uassignvariableop_38_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_bias_vhat:	»
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
AssignVariableOpAssignVariableOp!assignvariableop_dense_602_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¶
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_602_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOpKassignvariableop_7_bidirectional_602_forward_lstm_602_lstm_cell_1807_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Џ
AssignVariableOp_8AssignVariableOpUassignvariableop_8_bidirectional_602_forward_lstm_602_lstm_cell_1807_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ќ
AssignVariableOp_9AssignVariableOpIassignvariableop_9_bidirectional_602_forward_lstm_602_lstm_cell_1807_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10’
AssignVariableOp_10AssignVariableOpMassignvariableop_10_bidirectional_602_backward_lstm_602_lstm_cell_1808_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11я
AssignVariableOp_11AssignVariableOpWassignvariableop_11_bidirectional_602_backward_lstm_602_lstm_cell_1808_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12”
AssignVariableOp_12AssignVariableOpKassignvariableop_12_bidirectional_602_backward_lstm_602_lstm_cell_1808_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_dense_602_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16±
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_dense_602_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17џ
AssignVariableOp_17AssignVariableOpSassignvariableop_17_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18е
AssignVariableOp_18AssignVariableOp]assignvariableop_18_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19ў
AssignVariableOp_19AssignVariableOpQassignvariableop_19_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20№
AssignVariableOp_20AssignVariableOpTassignvariableop_20_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ж
AssignVariableOp_21AssignVariableOp^assignvariableop_21_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Џ
AssignVariableOp_22AssignVariableOpRassignvariableop_22_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23≥
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_602_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24±
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_602_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25џ
AssignVariableOp_25AssignVariableOpSassignvariableop_25_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26е
AssignVariableOp_26AssignVariableOp]assignvariableop_26_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27ў
AssignVariableOp_27AssignVariableOpQassignvariableop_27_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28№
AssignVariableOp_28AssignVariableOpTassignvariableop_28_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29ж
AssignVariableOp_29AssignVariableOp^assignvariableop_29_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Џ
AssignVariableOp_30AssignVariableOpRassignvariableop_30_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31ґ
AssignVariableOp_31AssignVariableOp.assignvariableop_31_adam_dense_602_kernel_vhatIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32і
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_dense_602_bias_vhatIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33ё
AssignVariableOp_33AssignVariableOpVassignvariableop_33_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_kernel_vhatIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34и
AssignVariableOp_34AssignVariableOp`assignvariableop_34_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_recurrent_kernel_vhatIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35№
AssignVariableOp_35AssignVariableOpTassignvariableop_35_adam_bidirectional_602_forward_lstm_602_lstm_cell_1807_bias_vhatIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36я
AssignVariableOp_36AssignVariableOpWassignvariableop_36_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_kernel_vhatIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37й
AssignVariableOp_37AssignVariableOpaassignvariableop_37_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_recurrent_kernel_vhatIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ё
AssignVariableOp_38AssignVariableOpUassignvariableop_38_adam_bidirectional_602_backward_lstm_602_lstm_cell_1808_bias_vhatIdentity_38:output:0"/device:CPU:0*
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
_user_specified_namefile_prefix"®L
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
	dense_6020
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
": d2dense_602/kernel
:2dense_602/bias
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
K:I	»28bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel
U:S	2»2Bbidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel
E:C»26bidirectional_602/forward_lstm_602/lstm_cell_1807/bias
L:J	»29bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel
V:T	2»2Cbidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel
F:D»27bidirectional_602/backward_lstm_602/lstm_cell_1808/bias
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
':%d2Adam/dense_602/kernel/m
!:2Adam/dense_602/bias/m
P:N	»2?Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/m
Z:X	2»2IAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/m
J:H»2=Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/m
Q:O	»2@Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/m
[:Y	2»2JAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/m
K:I»2>Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/m
':%d2Adam/dense_602/kernel/v
!:2Adam/dense_602/bias/v
P:N	»2?Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/v
Z:X	2»2IAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/v
J:H»2=Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/v
Q:O	»2@Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/v
[:Y	2»2JAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/v
K:I»2>Adam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/v
*:(d2Adam/dense_602/kernel/vhat
$:"2Adam/dense_602/bias/vhat
S:Q	»2BAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/kernel/vhat
]:[	2»2LAdam/bidirectional_602/forward_lstm_602/lstm_cell_1807/recurrent_kernel/vhat
M:K»2@Adam/bidirectional_602/forward_lstm_602/lstm_cell_1807/bias/vhat
T:R	»2CAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/kernel/vhat
^:\	2»2MAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/recurrent_kernel/vhat
N:L»2AAdam/bidirectional_602/backward_lstm_602/lstm_cell_1808/bias/vhat
ђ2©
1__inference_sequential_602_layer_call_fn_60936115
1__inference_sequential_602_layer_call_fn_60936608ј
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
#__inference__wrapped_model_60933628args_0args_0_1"Ш
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
L__inference_sequential_602_layer_call_and_return_conditional_losses_60936631
L__inference_sequential_602_layer_call_and_return_conditional_losses_60936654ј
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
4__inference_bidirectional_602_layer_call_fn_60936701
4__inference_bidirectional_602_layer_call_fn_60936718
4__inference_bidirectional_602_layer_call_fn_60936736
4__inference_bidirectional_602_layer_call_fn_60936754ж
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
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_60937056
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_60937358
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_60937716
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_60938074ж
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
,__inference_dense_602_layer_call_fn_60938083Ґ
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
G__inference_dense_602_layer_call_and_return_conditional_losses_60938094Ґ
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
&__inference_signature_wrapper_60936684args_0args_0_1"Ф
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
3__inference_forward_lstm_602_layer_call_fn_60938105
3__inference_forward_lstm_602_layer_call_fn_60938116
3__inference_forward_lstm_602_layer_call_fn_60938127
3__inference_forward_lstm_602_layer_call_fn_60938138’
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
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_60938289
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_60938440
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_60938591
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_60938742’
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
4__inference_backward_lstm_602_layer_call_fn_60938753
4__inference_backward_lstm_602_layer_call_fn_60938764
4__inference_backward_lstm_602_layer_call_fn_60938775
4__inference_backward_lstm_602_layer_call_fn_60938786’
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
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_60938939
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_60939092
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_60939245
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_60939398’
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
1__inference_lstm_cell_1807_layer_call_fn_60939415
1__inference_lstm_cell_1807_layer_call_fn_60939432Њ
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
L__inference_lstm_cell_1807_layer_call_and_return_conditional_losses_60939464
L__inference_lstm_cell_1807_layer_call_and_return_conditional_losses_60939496Њ
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
1__inference_lstm_cell_1808_layer_call_fn_60939513
1__inference_lstm_cell_1808_layer_call_fn_60939530Њ
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
L__inference_lstm_cell_1808_layer_call_and_return_conditional_losses_60939562
L__inference_lstm_cell_1808_layer_call_and_return_conditional_losses_60939594Њ
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
#__inference__wrapped_model_60933628Я\ҐY
RҐO
MТJ4Ґ1
!ъ€€€€€€€€€€€€€€€€€€
А
`
А	RaggedTensorSpec
™ "5™2
0
	dense_602#К 
	dense_602€€€€€€€€€–
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_60938939}OҐL
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
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_60939092}OҐL
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
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_60939245QҐN
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
O__inference_backward_lstm_602_layer_call_and_return_conditional_losses_60939398QҐN
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
4__inference_backward_lstm_602_layer_call_fn_60938753pOҐL
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
4__inference_backward_lstm_602_layer_call_fn_60938764pOҐL
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
4__inference_backward_lstm_602_layer_call_fn_60938775rQҐN
GҐD
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€2™
4__inference_backward_lstm_602_layer_call_fn_60938786rQҐN
GҐD
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
p

 
™ "К€€€€€€€€€2б
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_60937056Н\ҐY
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
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_60937358Н\ҐY
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
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_60937716ЭlҐi
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
O__inference_bidirectional_602_layer_call_and_return_conditional_losses_60938074ЭlҐi
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
4__inference_bidirectional_602_layer_call_fn_60936701А\ҐY
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
4__inference_bidirectional_602_layer_call_fn_60936718А\ҐY
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
4__inference_bidirectional_602_layer_call_fn_60936736РlҐi
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
4__inference_bidirectional_602_layer_call_fn_60936754РlҐi
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
G__inference_dense_602_layer_call_and_return_conditional_losses_60938094\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "%Ґ"
К
0€€€€€€€€€
Ъ 
,__inference_dense_602_layer_call_fn_60938083O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "К€€€€€€€€€ѕ
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_60938289}OҐL
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
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_60938440}OҐL
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
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_60938591QҐN
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
N__inference_forward_lstm_602_layer_call_and_return_conditional_losses_60938742QҐN
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
3__inference_forward_lstm_602_layer_call_fn_60938105pOҐL
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
3__inference_forward_lstm_602_layer_call_fn_60938116pOҐL
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
3__inference_forward_lstm_602_layer_call_fn_60938127rQҐN
GҐD
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€2©
3__inference_forward_lstm_602_layer_call_fn_60938138rQҐN
GҐD
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
p

 
™ "К€€€€€€€€€2ќ
L__inference_lstm_cell_1807_layer_call_and_return_conditional_losses_60939464эАҐ}
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
L__inference_lstm_cell_1807_layer_call_and_return_conditional_losses_60939496эАҐ}
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
1__inference_lstm_cell_1807_layer_call_fn_60939415нАҐ}
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
1__inference_lstm_cell_1807_layer_call_fn_60939432нАҐ}
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
L__inference_lstm_cell_1808_layer_call_and_return_conditional_losses_60939562эАҐ}
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
L__inference_lstm_cell_1808_layer_call_and_return_conditional_losses_60939594эАҐ}
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
1__inference_lstm_cell_1808_layer_call_fn_60939513нАҐ}
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
1__inference_lstm_cell_1808_layer_call_fn_60939530нАҐ}
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
L__inference_sequential_602_layer_call_and_return_conditional_losses_60936631ЧdҐa
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
L__inference_sequential_602_layer_call_and_return_conditional_losses_60936654ЧdҐa
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
1__inference_sequential_602_layer_call_fn_60936115КdҐa
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
1__inference_sequential_602_layer_call_fn_60936608КdҐa
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
&__inference_signature_wrapper_60936684®eҐb
Ґ 
[™X
*
args_0 К
args_0€€€€€€€€€
*
args_0_1К
args_0_1€€€€€€€€€	"5™2
0
	dense_602#К 
	dense_602€€€€€€€€€