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
dense_618/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*!
shared_namedense_618/kernel
u
$dense_618/kernel/Read/ReadVariableOpReadVariableOpdense_618/kernel*
_output_shapes

:d*
dtype0
t
dense_618/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_618/bias
m
"dense_618/bias/Read/ReadVariableOpReadVariableOpdense_618/bias*
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
8bidirectional_618/forward_lstm_618/lstm_cell_1855/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*I
shared_name:8bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel
∆
Lbidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/Read/ReadVariableOpReadVariableOp8bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel*
_output_shapes
:	»*
dtype0
б
Bbidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*S
shared_nameDBbidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel
Џ
Vbidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/Read/ReadVariableOpReadVariableOpBbidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel*
_output_shapes
:	2»*
dtype0
≈
6bidirectional_618/forward_lstm_618/lstm_cell_1855/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*G
shared_name86bidirectional_618/forward_lstm_618/lstm_cell_1855/bias
Њ
Jbidirectional_618/forward_lstm_618/lstm_cell_1855/bias/Read/ReadVariableOpReadVariableOp6bidirectional_618/forward_lstm_618/lstm_cell_1855/bias*
_output_shapes	
:»*
dtype0
ѕ
9bidirectional_618/backward_lstm_618/lstm_cell_1856/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*J
shared_name;9bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel
»
Mbidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/Read/ReadVariableOpReadVariableOp9bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel*
_output_shapes
:	»*
dtype0
г
Cbidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*T
shared_nameECbidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel
№
Wbidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/Read/ReadVariableOpReadVariableOpCbidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel*
_output_shapes
:	2»*
dtype0
«
7bidirectional_618/backward_lstm_618/lstm_cell_1856/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*H
shared_name97bidirectional_618/backward_lstm_618/lstm_cell_1856/bias
ј
Kbidirectional_618/backward_lstm_618/lstm_cell_1856/bias/Read/ReadVariableOpReadVariableOp7bidirectional_618/backward_lstm_618/lstm_cell_1856/bias*
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
Adam/dense_618/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_618/kernel/m
Г
+Adam/dense_618/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_618/kernel/m*
_output_shapes

:d*
dtype0
В
Adam/dense_618/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_618/bias/m
{
)Adam/dense_618/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_618/bias/m*
_output_shapes
:*
dtype0
џ
?Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*P
shared_nameA?Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/m
‘
SAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/m/Read/ReadVariableOpReadVariableOp?Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/m*
_output_shapes
:	»*
dtype0
п
IAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*Z
shared_nameKIAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/m
и
]Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpIAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/m*
_output_shapes
:	2»*
dtype0
”
=Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*N
shared_name?=Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/m
ћ
QAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/m/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/m*
_output_shapes	
:»*
dtype0
Ё
@Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*Q
shared_nameB@Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/m
÷
TAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/m/Read/ReadVariableOpReadVariableOp@Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/m*
_output_shapes
:	»*
dtype0
с
JAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*[
shared_nameLJAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/m
к
^Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpJAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/m*
_output_shapes
:	2»*
dtype0
’
>Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*O
shared_name@>Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/m
ќ
RAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/m/Read/ReadVariableOpReadVariableOp>Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/m*
_output_shapes	
:»*
dtype0
К
Adam/dense_618/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_618/kernel/v
Г
+Adam/dense_618/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_618/kernel/v*
_output_shapes

:d*
dtype0
В
Adam/dense_618/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_618/bias/v
{
)Adam/dense_618/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_618/bias/v*
_output_shapes
:*
dtype0
џ
?Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*P
shared_nameA?Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/v
‘
SAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/v/Read/ReadVariableOpReadVariableOp?Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/v*
_output_shapes
:	»*
dtype0
п
IAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*Z
shared_nameKIAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/v
и
]Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpIAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/v*
_output_shapes
:	2»*
dtype0
”
=Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*N
shared_name?=Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/v
ћ
QAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/v/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/v*
_output_shapes	
:»*
dtype0
Ё
@Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*Q
shared_nameB@Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/v
÷
TAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/v/Read/ReadVariableOpReadVariableOp@Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/v*
_output_shapes
:	»*
dtype0
с
JAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*[
shared_nameLJAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/v
к
^Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpJAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/v*
_output_shapes
:	2»*
dtype0
’
>Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*O
shared_name@>Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/v
ќ
RAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/v/Read/ReadVariableOpReadVariableOp>Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/v*
_output_shapes	
:»*
dtype0
Р
Adam/dense_618/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*+
shared_nameAdam/dense_618/kernel/vhat
Й
.Adam/dense_618/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_618/kernel/vhat*
_output_shapes

:d*
dtype0
И
Adam/dense_618/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/dense_618/bias/vhat
Б
,Adam/dense_618/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_618/bias/vhat*
_output_shapes
:*
dtype0
б
BAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*S
shared_nameDBAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/vhat
Џ
VAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/vhat/Read/ReadVariableOpReadVariableOpBAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/vhat*
_output_shapes
:	»*
dtype0
х
LAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*]
shared_nameNLAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/vhat
о
`Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpLAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/vhat*
_output_shapes
:	2»*
dtype0
ў
@Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*Q
shared_nameB@Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/vhat
“
TAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/vhat/Read/ReadVariableOpReadVariableOp@Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/vhat*
_output_shapes	
:»*
dtype0
г
CAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»*T
shared_nameECAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/vhat
№
WAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/vhat/Read/ReadVariableOpReadVariableOpCAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/vhat*
_output_shapes
:	»*
dtype0
ч
MAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2»*^
shared_nameOMAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/vhat
р
aAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpMAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/vhat*
_output_shapes
:	2»*
dtype0
џ
AAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*R
shared_nameCAAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/vhat
‘
UAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/vhat/Read/ReadVariableOpReadVariableOpAAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/vhat*
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
VARIABLE_VALUEdense_618/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_618/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUE8bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEBbidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6bidirectional_618/forward_lstm_618/lstm_cell_1855/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE9bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUECbidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE7bidirectional_618/backward_lstm_618/lstm_cell_1856/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_618/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_618/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE?Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ҐЯ
VARIABLE_VALUEIAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE@Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
£†
VARIABLE_VALUEJAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE>Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_618/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_618/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE?Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ҐЯ
VARIABLE_VALUEIAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE@Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
£†
VARIABLE_VALUEJAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE>Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEAdam/dense_618/kernel/vhatUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/dense_618/bias/vhatSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUEBAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/vhatEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
®•
VARIABLE_VALUELAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/vhatEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ЬЩ
VARIABLE_VALUE@Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/vhatEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUECAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/vhatEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
©¶
VARIABLE_VALUEMAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/vhatEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ЭЪ
VARIABLE_VALUEAAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/vhatEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_0_18bidirectional_618/forward_lstm_618/lstm_cell_1855/kernelBbidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel6bidirectional_618/forward_lstm_618/lstm_cell_1855/bias9bidirectional_618/backward_lstm_618/lstm_cell_1856/kernelCbidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel7bidirectional_618/backward_lstm_618/lstm_cell_1856/biasdense_618/kerneldense_618/bias*
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
&__inference_signature_wrapper_62328060
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
І
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_618/kernel/Read/ReadVariableOp"dense_618/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpLbidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/Read/ReadVariableOpVbidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/Read/ReadVariableOpJbidirectional_618/forward_lstm_618/lstm_cell_1855/bias/Read/ReadVariableOpMbidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/Read/ReadVariableOpWbidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/Read/ReadVariableOpKbidirectional_618/backward_lstm_618/lstm_cell_1856/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_618/kernel/m/Read/ReadVariableOp)Adam/dense_618/bias/m/Read/ReadVariableOpSAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/m/Read/ReadVariableOp]Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/m/Read/ReadVariableOpQAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/m/Read/ReadVariableOpTAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/m/Read/ReadVariableOp^Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/m/Read/ReadVariableOpRAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/m/Read/ReadVariableOp+Adam/dense_618/kernel/v/Read/ReadVariableOp)Adam/dense_618/bias/v/Read/ReadVariableOpSAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/v/Read/ReadVariableOp]Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/v/Read/ReadVariableOpQAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/v/Read/ReadVariableOpTAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/v/Read/ReadVariableOp^Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/v/Read/ReadVariableOpRAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/v/Read/ReadVariableOp.Adam/dense_618/kernel/vhat/Read/ReadVariableOp,Adam/dense_618/bias/vhat/Read/ReadVariableOpVAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/vhat/Read/ReadVariableOp`Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/vhat/Read/ReadVariableOpTAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/vhat/Read/ReadVariableOpWAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/vhat/Read/ReadVariableOpaAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/vhat/Read/ReadVariableOpUAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/vhat/Read/ReadVariableOpConst*4
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
!__inference__traced_save_62331111
Ц
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_618/kerneldense_618/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate8bidirectional_618/forward_lstm_618/lstm_cell_1855/kernelBbidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel6bidirectional_618/forward_lstm_618/lstm_cell_1855/bias9bidirectional_618/backward_lstm_618/lstm_cell_1856/kernelCbidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel7bidirectional_618/backward_lstm_618/lstm_cell_1856/biastotalcountAdam/dense_618/kernel/mAdam/dense_618/bias/m?Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/mIAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/m=Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/m@Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/mJAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/m>Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/mAdam/dense_618/kernel/vAdam/dense_618/bias/v?Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/vIAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/v=Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/v@Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/vJAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/v>Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/vAdam/dense_618/kernel/vhatAdam/dense_618/bias/vhatBAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/vhatLAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/vhat@Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/vhatCAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/vhatMAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/vhatAAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/vhat*3
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
$__inference__traced_restore_62331238иЪ9
÷
√
4__inference_backward_lstm_618_layer_call_fn_62330129
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
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_623257942
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
while_cond_62330536
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_62330536___redundant_placeholder06
2while_while_cond_62330536___redundant_placeholder16
2while_while_cond_62330536___redundant_placeholder26
2while_while_cond_62330536___redundant_placeholder3
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
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_62326781

inputs@
-lstm_cell_1856_matmul_readvariableop_resource:	»B
/lstm_cell_1856_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1856_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1856/BiasAdd/ReadVariableOpҐ$lstm_cell_1856/MatMul/ReadVariableOpҐ&lstm_cell_1856/MatMul_1/ReadVariableOpҐwhileD
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
$lstm_cell_1856/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1856_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1856/MatMul/ReadVariableOp≥
lstm_cell_1856/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/MatMulЅ
&lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1856_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1856/MatMul_1/ReadVariableOpѓ
lstm_cell_1856/MatMul_1MatMulzeros:output:0.lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/MatMul_1®
lstm_cell_1856/addAddV2lstm_cell_1856/MatMul:product:0!lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/addЇ
%lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1856_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1856/BiasAdd/ReadVariableOpµ
lstm_cell_1856/BiasAddBiasAddlstm_cell_1856/add:z:0-lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/BiasAddВ
lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1856/split/split_dimы
lstm_cell_1856/splitSplit'lstm_cell_1856/split/split_dim:output:0lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1856/splitМ
lstm_cell_1856/SigmoidSigmoidlstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/SigmoidР
lstm_cell_1856/Sigmoid_1Sigmoidlstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Sigmoid_1С
lstm_cell_1856/mulMullstm_cell_1856/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/mulГ
lstm_cell_1856/ReluRelulstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Relu§
lstm_cell_1856/mul_1Mullstm_cell_1856/Sigmoid:y:0!lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/mul_1Щ
lstm_cell_1856/add_1AddV2lstm_cell_1856/mul:z:0lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/add_1Р
lstm_cell_1856/Sigmoid_2Sigmoidlstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Sigmoid_2В
lstm_cell_1856/Relu_1Relulstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Relu_1®
lstm_cell_1856/mul_2Mullstm_cell_1856/Sigmoid_2:y:0#lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1856_matmul_readvariableop_resource/lstm_cell_1856_matmul_1_readvariableop_resource.lstm_cell_1856_biasadd_readvariableop_resource*
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
while_body_62326697*
condR
while_cond_62326696*K
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
NoOpNoOp&^lstm_cell_1856/BiasAdd/ReadVariableOp%^lstm_cell_1856/MatMul/ReadVariableOp'^lstm_cell_1856/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1856/BiasAdd/ReadVariableOp%lstm_cell_1856/BiasAdd/ReadVariableOp2L
$lstm_cell_1856/MatMul/ReadVariableOp$lstm_cell_1856/MatMul/ReadVariableOp2P
&lstm_cell_1856/MatMul_1/ReadVariableOp&lstm_cell_1856/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
к
Љ
%backward_lstm_618_while_cond_62329352@
<backward_lstm_618_while_backward_lstm_618_while_loop_counterF
Bbackward_lstm_618_while_backward_lstm_618_while_maximum_iterations'
#backward_lstm_618_while_placeholder)
%backward_lstm_618_while_placeholder_1)
%backward_lstm_618_while_placeholder_2)
%backward_lstm_618_while_placeholder_3)
%backward_lstm_618_while_placeholder_4B
>backward_lstm_618_while_less_backward_lstm_618_strided_slice_1Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62329352___redundant_placeholder0Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62329352___redundant_placeholder1Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62329352___redundant_placeholder2Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62329352___redundant_placeholder3Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62329352___redundant_placeholder4$
 backward_lstm_618_while_identity
 
backward_lstm_618/while/LessLess#backward_lstm_618_while_placeholder>backward_lstm_618_while_less_backward_lstm_618_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_618/while/LessУ
 backward_lstm_618/while/IdentityIdentity backward_lstm_618/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_618/while/Identity"M
 backward_lstm_618_while_identity)backward_lstm_618/while/Identity:output:0*(
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
while_cond_62325092
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_62325092___redundant_placeholder06
2while_while_cond_62325092___redundant_placeholder16
2while_while_cond_62325092___redundant_placeholder26
2while_while_cond_62325092___redundant_placeholder3
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
while_cond_62325302
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_62325302___redundant_placeholder06
2while_while_cond_62325302___redundant_placeholder16
2while_while_cond_62325302___redundant_placeholder26
2while_while_cond_62325302___redundant_placeholder3
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
L__inference_sequential_618_layer_call_and_return_conditional_losses_62327943

inputs
inputs_1	-
bidirectional_618_62327924:	»-
bidirectional_618_62327926:	2»)
bidirectional_618_62327928:	»-
bidirectional_618_62327930:	»-
bidirectional_618_62327932:	2»)
bidirectional_618_62327934:	»$
dense_618_62327937:d 
dense_618_62327939:
identityИҐ)bidirectional_618/StatefulPartitionedCallҐ!dense_618/StatefulPartitionedCall 
)bidirectional_618/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_618_62327924bidirectional_618_62327926bidirectional_618_62327928bidirectional_618_62327930bidirectional_618_62327932bidirectional_618_62327934*
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
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_623278802+
)bidirectional_618/StatefulPartitionedCallЋ
!dense_618/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_618/StatefulPartitionedCall:output:0dense_618_62327937dense_618_62327939*
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
G__inference_dense_618_layer_call_and_return_conditional_losses_623274652#
!dense_618/StatefulPartitionedCallЕ
IdentityIdentity*dense_618/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityЮ
NoOpNoOp*^bidirectional_618/StatefulPartitionedCall"^dense_618/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€:€€€€€€€€€: : : : : : : : 2V
)bidirectional_618/StatefulPartitionedCall)bidirectional_618/StatefulPartitionedCall2F
!dense_618/StatefulPartitionedCall!dense_618/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
к
Љ
%backward_lstm_618_while_cond_62327782@
<backward_lstm_618_while_backward_lstm_618_while_loop_counterF
Bbackward_lstm_618_while_backward_lstm_618_while_maximum_iterations'
#backward_lstm_618_while_placeholder)
%backward_lstm_618_while_placeholder_1)
%backward_lstm_618_while_placeholder_2)
%backward_lstm_618_while_placeholder_3)
%backward_lstm_618_while_placeholder_4B
>backward_lstm_618_while_less_backward_lstm_618_strided_slice_1Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62327782___redundant_placeholder0Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62327782___redundant_placeholder1Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62327782___redundant_placeholder2Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62327782___redundant_placeholder3Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62327782___redundant_placeholder4$
 backward_lstm_618_while_identity
 
backward_lstm_618/while/LessLess#backward_lstm_618_while_placeholder>backward_lstm_618_while_less_backward_lstm_618_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_618/while/LessУ
 backward_lstm_618/while/IdentityIdentity backward_lstm_618/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_618/while/Identity"M
 backward_lstm_618_while_identity)backward_lstm_618/while/Identity:output:0*(
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
while_cond_62325724
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_62325724___redundant_placeholder06
2while_while_cond_62325724___redundant_placeholder16
2while_while_cond_62325724___redundant_placeholder26
2while_while_cond_62325724___redundant_placeholder3
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
G__inference_dense_618_layer_call_and_return_conditional_losses_62329470

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
€
К
L__inference_lstm_cell_1856_layer_call_and_return_conditional_losses_62330970

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
L__inference_sequential_618_layer_call_and_return_conditional_losses_62328007

inputs
inputs_1	-
bidirectional_618_62327988:	»-
bidirectional_618_62327990:	2»)
bidirectional_618_62327992:	»-
bidirectional_618_62327994:	»-
bidirectional_618_62327996:	2»)
bidirectional_618_62327998:	»$
dense_618_62328001:d 
dense_618_62328003:
identityИҐ)bidirectional_618/StatefulPartitionedCallҐ!dense_618/StatefulPartitionedCall 
)bidirectional_618/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_618_62327988bidirectional_618_62327990bidirectional_618_62327992bidirectional_618_62327994bidirectional_618_62327996bidirectional_618_62327998*
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
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_623274402+
)bidirectional_618/StatefulPartitionedCallЋ
!dense_618/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_618/StatefulPartitionedCall:output:0dense_618_62328001dense_618_62328003*
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
G__inference_dense_618_layer_call_and_return_conditional_losses_623274652#
!dense_618/StatefulPartitionedCallЕ
IdentityIdentity*dense_618/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityЮ
NoOpNoOp*^bidirectional_618/StatefulPartitionedCall"^dense_618/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€:€€€€€€€€€: : : : : : : : 2V
)bidirectional_618/StatefulPartitionedCall)bidirectional_618/StatefulPartitionedCall2F
!dense_618/StatefulPartitionedCall!dense_618/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
∞Њ
ы
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_62327880

inputs
inputs_1	Q
>forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource:	»S
@forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource:	2»N
?forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource:	»R
?backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resource:	»T
Abackward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource:	2»O
@backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource:	»
identityИҐ7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpҐ6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpҐ8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpҐbackward_lstm_618/whileҐ6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpҐ5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpҐ7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpҐforward_lstm_618/whileЧ
%forward_lstm_618/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_618/RaggedToTensor/zerosЩ
%forward_lstm_618/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2'
%forward_lstm_618/RaggedToTensor/ConstЩ
4forward_lstm_618/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_618/RaggedToTensor/Const:output:0inputs.forward_lstm_618/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_618/RaggedToTensor/RaggedTensorToTensorƒ
;forward_lstm_618/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack»
=forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1»
=forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2©
5forward_lstm_618/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_618/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask27
5forward_lstm_618/RaggedNestedRowLengths/strided_slice»
=forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack’
?forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2A
?forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1ћ
?forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2µ
7forward_lstm_618/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask29
7forward_lstm_618/RaggedNestedRowLengths/strided_slice_1С
+forward_lstm_618/RaggedNestedRowLengths/subSub>forward_lstm_618/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_618/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2-
+forward_lstm_618/RaggedNestedRowLengths/sub§
forward_lstm_618/CastCast/forward_lstm_618/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
forward_lstm_618/CastЭ
forward_lstm_618/ShapeShape=forward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_618/ShapeЦ
$forward_lstm_618/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_618/strided_slice/stackЪ
&forward_lstm_618/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_618/strided_slice/stack_1Ъ
&forward_lstm_618/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_618/strided_slice/stack_2»
forward_lstm_618/strided_sliceStridedSliceforward_lstm_618/Shape:output:0-forward_lstm_618/strided_slice/stack:output:0/forward_lstm_618/strided_slice/stack_1:output:0/forward_lstm_618/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_618/strided_slice~
forward_lstm_618/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_618/zeros/mul/y∞
forward_lstm_618/zeros/mulMul'forward_lstm_618/strided_slice:output:0%forward_lstm_618/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros/mulБ
forward_lstm_618/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
forward_lstm_618/zeros/Less/yЂ
forward_lstm_618/zeros/LessLessforward_lstm_618/zeros/mul:z:0&forward_lstm_618/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros/LessД
forward_lstm_618/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_618/zeros/packed/1«
forward_lstm_618/zeros/packedPack'forward_lstm_618/strided_slice:output:0(forward_lstm_618/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_618/zeros/packedЕ
forward_lstm_618/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_618/zeros/Constє
forward_lstm_618/zerosFill&forward_lstm_618/zeros/packed:output:0%forward_lstm_618/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/zerosВ
forward_lstm_618/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_618/zeros_1/mul/yґ
forward_lstm_618/zeros_1/mulMul'forward_lstm_618/strided_slice:output:0'forward_lstm_618/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros_1/mulЕ
forward_lstm_618/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2!
forward_lstm_618/zeros_1/Less/y≥
forward_lstm_618/zeros_1/LessLess forward_lstm_618/zeros_1/mul:z:0(forward_lstm_618/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros_1/LessИ
!forward_lstm_618/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_618/zeros_1/packed/1Ќ
forward_lstm_618/zeros_1/packedPack'forward_lstm_618/strided_slice:output:0*forward_lstm_618/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_618/zeros_1/packedЙ
forward_lstm_618/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_618/zeros_1/ConstЅ
forward_lstm_618/zeros_1Fill(forward_lstm_618/zeros_1/packed:output:0'forward_lstm_618/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/zeros_1Ч
forward_lstm_618/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_618/transpose/permн
forward_lstm_618/transpose	Transpose=forward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_618/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
forward_lstm_618/transposeВ
forward_lstm_618/Shape_1Shapeforward_lstm_618/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_618/Shape_1Ъ
&forward_lstm_618/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_618/strided_slice_1/stackЮ
(forward_lstm_618/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_1/stack_1Ю
(forward_lstm_618/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_1/stack_2‘
 forward_lstm_618/strided_slice_1StridedSlice!forward_lstm_618/Shape_1:output:0/forward_lstm_618/strided_slice_1/stack:output:01forward_lstm_618/strided_slice_1/stack_1:output:01forward_lstm_618/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_618/strided_slice_1І
,forward_lstm_618/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,forward_lstm_618/TensorArrayV2/element_shapeц
forward_lstm_618/TensorArrayV2TensorListReserve5forward_lstm_618/TensorArrayV2/element_shape:output:0)forward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_618/TensorArrayV2б
Fforward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2H
Fforward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeЉ
8forward_lstm_618/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_618/transpose:y:0Oforward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_618/TensorArrayUnstack/TensorListFromTensorЪ
&forward_lstm_618/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_618/strided_slice_2/stackЮ
(forward_lstm_618/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_2/stack_1Ю
(forward_lstm_618/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_2/stack_2в
 forward_lstm_618/strided_slice_2StridedSliceforward_lstm_618/transpose:y:0/forward_lstm_618/strided_slice_2/stack:output:01forward_lstm_618/strided_slice_2/stack_1:output:01forward_lstm_618/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2"
 forward_lstm_618/strided_slice_2о
5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOp>forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype027
5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpч
&forward_lstm_618/lstm_cell_1855/MatMulMatMul)forward_lstm_618/strided_slice_2:output:0=forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2(
&forward_lstm_618/lstm_cell_1855/MatMulф
7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOp@forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype029
7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpу
(forward_lstm_618/lstm_cell_1855/MatMul_1MatMulforward_lstm_618/zeros:output:0?forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(forward_lstm_618/lstm_cell_1855/MatMul_1м
#forward_lstm_618/lstm_cell_1855/addAddV20forward_lstm_618/lstm_cell_1855/MatMul:product:02forward_lstm_618/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2%
#forward_lstm_618/lstm_cell_1855/addн
6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOp?forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype028
6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpщ
'forward_lstm_618/lstm_cell_1855/BiasAddBiasAdd'forward_lstm_618/lstm_cell_1855/add:z:0>forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'forward_lstm_618/lstm_cell_1855/BiasAdd§
/forward_lstm_618/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/forward_lstm_618/lstm_cell_1855/split/split_dimњ
%forward_lstm_618/lstm_cell_1855/splitSplit8forward_lstm_618/lstm_cell_1855/split/split_dim:output:00forward_lstm_618/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2'
%forward_lstm_618/lstm_cell_1855/splitњ
'forward_lstm_618/lstm_cell_1855/SigmoidSigmoid.forward_lstm_618/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'forward_lstm_618/lstm_cell_1855/Sigmoid√
)forward_lstm_618/lstm_cell_1855/Sigmoid_1Sigmoid.forward_lstm_618/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_618/lstm_cell_1855/Sigmoid_1’
#forward_lstm_618/lstm_cell_1855/mulMul-forward_lstm_618/lstm_cell_1855/Sigmoid_1:y:0!forward_lstm_618/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22%
#forward_lstm_618/lstm_cell_1855/mulґ
$forward_lstm_618/lstm_cell_1855/ReluRelu.forward_lstm_618/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22&
$forward_lstm_618/lstm_cell_1855/Reluи
%forward_lstm_618/lstm_cell_1855/mul_1Mul+forward_lstm_618/lstm_cell_1855/Sigmoid:y:02forward_lstm_618/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_618/lstm_cell_1855/mul_1Ё
%forward_lstm_618/lstm_cell_1855/add_1AddV2'forward_lstm_618/lstm_cell_1855/mul:z:0)forward_lstm_618/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_618/lstm_cell_1855/add_1√
)forward_lstm_618/lstm_cell_1855/Sigmoid_2Sigmoid.forward_lstm_618/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_618/lstm_cell_1855/Sigmoid_2µ
&forward_lstm_618/lstm_cell_1855/Relu_1Relu)forward_lstm_618/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&forward_lstm_618/lstm_cell_1855/Relu_1м
%forward_lstm_618/lstm_cell_1855/mul_2Mul-forward_lstm_618/lstm_cell_1855/Sigmoid_2:y:04forward_lstm_618/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_618/lstm_cell_1855/mul_2±
.forward_lstm_618/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   20
.forward_lstm_618/TensorArrayV2_1/element_shapeь
 forward_lstm_618/TensorArrayV2_1TensorListReserve7forward_lstm_618/TensorArrayV2_1/element_shape:output:0)forward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_618/TensorArrayV2_1p
forward_lstm_618/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_618/time§
forward_lstm_618/zeros_like	ZerosLike)forward_lstm_618/lstm_cell_1855/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/zeros_like°
)forward_lstm_618/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2+
)forward_lstm_618/while/maximum_iterationsМ
#forward_lstm_618/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_618/while/loop_counterЦ	
forward_lstm_618/whileWhile,forward_lstm_618/while/loop_counter:output:02forward_lstm_618/while/maximum_iterations:output:0forward_lstm_618/time:output:0)forward_lstm_618/TensorArrayV2_1:handle:0forward_lstm_618/zeros_like:y:0forward_lstm_618/zeros:output:0!forward_lstm_618/zeros_1:output:0)forward_lstm_618/strided_slice_1:output:0Hforward_lstm_618/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_618/Cast:y:0>forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource@forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource?forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource*
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
$forward_lstm_618_while_body_62327604*0
cond(R&
$forward_lstm_618_while_cond_62327603*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
forward_lstm_618/while„
Aforward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2C
Aforward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeµ
3forward_lstm_618/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_618/while:output:3Jforward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype025
3forward_lstm_618/TensorArrayV2Stack/TensorListStack£
&forward_lstm_618/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2(
&forward_lstm_618/strided_slice_3/stackЮ
(forward_lstm_618/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_618/strided_slice_3/stack_1Ю
(forward_lstm_618/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_3/stack_2А
 forward_lstm_618/strided_slice_3StridedSlice<forward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_618/strided_slice_3/stack:output:01forward_lstm_618/strided_slice_3/stack_1:output:01forward_lstm_618/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2"
 forward_lstm_618/strided_slice_3Ы
!forward_lstm_618/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_618/transpose_1/permт
forward_lstm_618/transpose_1	Transpose<forward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_618/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
forward_lstm_618/transpose_1И
forward_lstm_618/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_618/runtimeЩ
&backward_lstm_618/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_618/RaggedToTensor/zerosЫ
&backward_lstm_618/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2(
&backward_lstm_618/RaggedToTensor/ConstЭ
5backward_lstm_618/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_618/RaggedToTensor/Const:output:0inputs/backward_lstm_618/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_618/RaggedToTensor/RaggedTensorToTensor∆
<backward_lstm_618/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack 
>backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1 
>backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2Ѓ
6backward_lstm_618/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_618/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask28
6backward_lstm_618/RaggedNestedRowLengths/strided_slice 
>backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack„
@backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2B
@backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1ќ
@backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2Ї
8backward_lstm_618/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask2:
8backward_lstm_618/RaggedNestedRowLengths/strided_slice_1Х
,backward_lstm_618/RaggedNestedRowLengths/subSub?backward_lstm_618/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_618/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2.
,backward_lstm_618/RaggedNestedRowLengths/subІ
backward_lstm_618/CastCast0backward_lstm_618/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
backward_lstm_618/Cast†
backward_lstm_618/ShapeShape>backward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_618/ShapeШ
%backward_lstm_618/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_618/strided_slice/stackЬ
'backward_lstm_618/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_618/strided_slice/stack_1Ь
'backward_lstm_618/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_618/strided_slice/stack_2ќ
backward_lstm_618/strided_sliceStridedSlice backward_lstm_618/Shape:output:0.backward_lstm_618/strided_slice/stack:output:00backward_lstm_618/strided_slice/stack_1:output:00backward_lstm_618/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_618/strided_sliceА
backward_lstm_618/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_618/zeros/mul/yі
backward_lstm_618/zeros/mulMul(backward_lstm_618/strided_slice:output:0&backward_lstm_618/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/zeros/mulГ
backward_lstm_618/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
backward_lstm_618/zeros/Less/yѓ
backward_lstm_618/zeros/LessLessbackward_lstm_618/zeros/mul:z:0'backward_lstm_618/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/zeros/LessЖ
 backward_lstm_618/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_618/zeros/packed/1Ћ
backward_lstm_618/zeros/packedPack(backward_lstm_618/strided_slice:output:0)backward_lstm_618/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_618/zeros/packedЗ
backward_lstm_618/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_618/zeros/Constљ
backward_lstm_618/zerosFill'backward_lstm_618/zeros/packed:output:0&backward_lstm_618/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_618/zerosД
backward_lstm_618/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_618/zeros_1/mul/yЇ
backward_lstm_618/zeros_1/mulMul(backward_lstm_618/strided_slice:output:0(backward_lstm_618/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/zeros_1/mulЗ
 backward_lstm_618/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 backward_lstm_618/zeros_1/Less/yЈ
backward_lstm_618/zeros_1/LessLess!backward_lstm_618/zeros_1/mul:z:0)backward_lstm_618/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_618/zeros_1/LessК
"backward_lstm_618/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_618/zeros_1/packed/1—
 backward_lstm_618/zeros_1/packedPack(backward_lstm_618/strided_slice:output:0+backward_lstm_618/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_618/zeros_1/packedЛ
backward_lstm_618/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_618/zeros_1/Const≈
backward_lstm_618/zeros_1Fill)backward_lstm_618/zeros_1/packed:output:0(backward_lstm_618/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_618/zeros_1Щ
 backward_lstm_618/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_618/transpose/permс
backward_lstm_618/transpose	Transpose>backward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_618/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_618/transposeЕ
backward_lstm_618/Shape_1Shapebackward_lstm_618/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_618/Shape_1Ь
'backward_lstm_618/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_618/strided_slice_1/stack†
)backward_lstm_618/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_1/stack_1†
)backward_lstm_618/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_1/stack_2Џ
!backward_lstm_618/strided_slice_1StridedSlice"backward_lstm_618/Shape_1:output:00backward_lstm_618/strided_slice_1/stack:output:02backward_lstm_618/strided_slice_1/stack_1:output:02backward_lstm_618/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_618/strided_slice_1©
-backward_lstm_618/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-backward_lstm_618/TensorArrayV2/element_shapeъ
backward_lstm_618/TensorArrayV2TensorListReserve6backward_lstm_618/TensorArrayV2/element_shape:output:0*backward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_618/TensorArrayV2О
 backward_lstm_618/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_618/ReverseV2/axis“
backward_lstm_618/ReverseV2	ReverseV2backward_lstm_618/transpose:y:0)backward_lstm_618/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_618/ReverseV2г
Gbackward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2I
Gbackward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape≈
9backward_lstm_618/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_618/ReverseV2:output:0Pbackward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_618/TensorArrayUnstack/TensorListFromTensorЬ
'backward_lstm_618/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_618/strided_slice_2/stack†
)backward_lstm_618/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_2/stack_1†
)backward_lstm_618/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_2/stack_2и
!backward_lstm_618/strided_slice_2StridedSlicebackward_lstm_618/transpose:y:00backward_lstm_618/strided_slice_2/stack:output:02backward_lstm_618/strided_slice_2/stack_1:output:02backward_lstm_618/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2#
!backward_lstm_618/strided_slice_2с
6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOp?backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype028
6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpы
'backward_lstm_618/lstm_cell_1856/MatMulMatMul*backward_lstm_618/strided_slice_2:output:0>backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'backward_lstm_618/lstm_cell_1856/MatMulч
8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOpAbackward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02:
8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpч
)backward_lstm_618/lstm_cell_1856/MatMul_1MatMul backward_lstm_618/zeros:output:0@backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)backward_lstm_618/lstm_cell_1856/MatMul_1р
$backward_lstm_618/lstm_cell_1856/addAddV21backward_lstm_618/lstm_cell_1856/MatMul:product:03backward_lstm_618/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2&
$backward_lstm_618/lstm_cell_1856/addр
7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOp@backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype029
7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpэ
(backward_lstm_618/lstm_cell_1856/BiasAddBiasAdd(backward_lstm_618/lstm_cell_1856/add:z:0?backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(backward_lstm_618/lstm_cell_1856/BiasAdd¶
0backward_lstm_618/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0backward_lstm_618/lstm_cell_1856/split/split_dim√
&backward_lstm_618/lstm_cell_1856/splitSplit9backward_lstm_618/lstm_cell_1856/split/split_dim:output:01backward_lstm_618/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2(
&backward_lstm_618/lstm_cell_1856/split¬
(backward_lstm_618/lstm_cell_1856/SigmoidSigmoid/backward_lstm_618/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(backward_lstm_618/lstm_cell_1856/Sigmoid∆
*backward_lstm_618/lstm_cell_1856/Sigmoid_1Sigmoid/backward_lstm_618/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_618/lstm_cell_1856/Sigmoid_1ў
$backward_lstm_618/lstm_cell_1856/mulMul.backward_lstm_618/lstm_cell_1856/Sigmoid_1:y:0"backward_lstm_618/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$backward_lstm_618/lstm_cell_1856/mulє
%backward_lstm_618/lstm_cell_1856/ReluRelu/backward_lstm_618/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22'
%backward_lstm_618/lstm_cell_1856/Reluм
&backward_lstm_618/lstm_cell_1856/mul_1Mul,backward_lstm_618/lstm_cell_1856/Sigmoid:y:03backward_lstm_618/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_618/lstm_cell_1856/mul_1б
&backward_lstm_618/lstm_cell_1856/add_1AddV2(backward_lstm_618/lstm_cell_1856/mul:z:0*backward_lstm_618/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_618/lstm_cell_1856/add_1∆
*backward_lstm_618/lstm_cell_1856/Sigmoid_2Sigmoid/backward_lstm_618/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_618/lstm_cell_1856/Sigmoid_2Є
'backward_lstm_618/lstm_cell_1856/Relu_1Relu*backward_lstm_618/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'backward_lstm_618/lstm_cell_1856/Relu_1р
&backward_lstm_618/lstm_cell_1856/mul_2Mul.backward_lstm_618/lstm_cell_1856/Sigmoid_2:y:05backward_lstm_618/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_618/lstm_cell_1856/mul_2≥
/backward_lstm_618/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   21
/backward_lstm_618/TensorArrayV2_1/element_shapeА
!backward_lstm_618/TensorArrayV2_1TensorListReserve8backward_lstm_618/TensorArrayV2_1/element_shape:output:0*backward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_618/TensorArrayV2_1r
backward_lstm_618/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_618/timeФ
'backward_lstm_618/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_618/Max/reduction_indices§
backward_lstm_618/MaxMaxbackward_lstm_618/Cast:y:00backward_lstm_618/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/Maxt
backward_lstm_618/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_618/sub/yШ
backward_lstm_618/subSubbackward_lstm_618/Max:output:0 backward_lstm_618/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/subЮ
backward_lstm_618/Sub_1Subbackward_lstm_618/sub:z:0backward_lstm_618/Cast:y:0*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_618/Sub_1І
backward_lstm_618/zeros_like	ZerosLike*backward_lstm_618/lstm_cell_1856/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_618/zeros_like£
*backward_lstm_618/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2,
*backward_lstm_618/while/maximum_iterationsО
$backward_lstm_618/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_618/while/loop_counter®	
backward_lstm_618/whileWhile-backward_lstm_618/while/loop_counter:output:03backward_lstm_618/while/maximum_iterations:output:0backward_lstm_618/time:output:0*backward_lstm_618/TensorArrayV2_1:handle:0 backward_lstm_618/zeros_like:y:0 backward_lstm_618/zeros:output:0"backward_lstm_618/zeros_1:output:0*backward_lstm_618/strided_slice_1:output:0Ibackward_lstm_618/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_618/Sub_1:z:0?backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resourceAbackward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource@backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource*
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
%backward_lstm_618_while_body_62327783*1
cond)R'
%backward_lstm_618_while_cond_62327782*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
backward_lstm_618/whileў
Bbackward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2D
Bbackward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeє
4backward_lstm_618/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_618/while:output:3Kbackward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype026
4backward_lstm_618/TensorArrayV2Stack/TensorListStack•
'backward_lstm_618/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2)
'backward_lstm_618/strided_slice_3/stack†
)backward_lstm_618/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_618/strided_slice_3/stack_1†
)backward_lstm_618/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_3/stack_2Ж
!backward_lstm_618/strided_slice_3StridedSlice=backward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_618/strided_slice_3/stack:output:02backward_lstm_618/strided_slice_3/stack_1:output:02backward_lstm_618/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2#
!backward_lstm_618/strided_slice_3Э
"backward_lstm_618/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_618/transpose_1/permц
backward_lstm_618/transpose_1	Transpose=backward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_618/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
backward_lstm_618/transpose_1К
backward_lstm_618/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_618/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisƒ
concatConcatV2)forward_lstm_618/strided_slice_3:output:0*backward_lstm_618/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp8^backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp7^backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp9^backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp^backward_lstm_618/while7^forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp6^forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp8^forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp^forward_lstm_618/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€:€€€€€€€€€: : : : : : 2r
7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp2p
6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp2t
8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp22
backward_lstm_618/whilebackward_lstm_618/while2p
6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp2n
5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp2r
7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp20
forward_lstm_618/whileforward_lstm_618/while:O K
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
$forward_lstm_618_while_cond_62327603>
:forward_lstm_618_while_forward_lstm_618_while_loop_counterD
@forward_lstm_618_while_forward_lstm_618_while_maximum_iterations&
"forward_lstm_618_while_placeholder(
$forward_lstm_618_while_placeholder_1(
$forward_lstm_618_while_placeholder_2(
$forward_lstm_618_while_placeholder_3(
$forward_lstm_618_while_placeholder_4@
<forward_lstm_618_while_less_forward_lstm_618_strided_slice_1X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62327603___redundant_placeholder0X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62327603___redundant_placeholder1X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62327603___redundant_placeholder2X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62327603___redundant_placeholder3X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62327603___redundant_placeholder4#
forward_lstm_618_while_identity
≈
forward_lstm_618/while/LessLess"forward_lstm_618_while_placeholder<forward_lstm_618_while_less_forward_lstm_618_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_618/while/LessР
forward_lstm_618/while/IdentityIdentityforward_lstm_618/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_618/while/Identity"K
forward_lstm_618_while_identity(forward_lstm_618/while/Identity:output:0*(
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
Ѓ

•
4__inference_bidirectional_618_layer_call_fn_62328112

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
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_623274402
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
я
°
$forward_lstm_618_while_cond_62328196>
:forward_lstm_618_while_forward_lstm_618_while_loop_counterD
@forward_lstm_618_while_forward_lstm_618_while_maximum_iterations&
"forward_lstm_618_while_placeholder(
$forward_lstm_618_while_placeholder_1(
$forward_lstm_618_while_placeholder_2(
$forward_lstm_618_while_placeholder_3@
<forward_lstm_618_while_less_forward_lstm_618_strided_slice_1X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62328196___redundant_placeholder0X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62328196___redundant_placeholder1X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62328196___redundant_placeholder2X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62328196___redundant_placeholder3#
forward_lstm_618_while_identity
≈
forward_lstm_618/while/LessLess"forward_lstm_618_while_placeholder<forward_lstm_618_while_less_forward_lstm_618_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_618/while/LessР
forward_lstm_618/while/IdentityIdentityforward_lstm_618/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_618/while/Identity"K
forward_lstm_618_while_identity(forward_lstm_618/while/Identity:output:0*(
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
while_body_62326345
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1855_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1855_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1855_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1855_matmul_readvariableop_resource:	»H
5while_lstm_cell_1855_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1855_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1855/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1855/MatMul/ReadVariableOpҐ,while/lstm_cell_1855/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1855_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1855/MatMul/ReadVariableOpЁ
while/lstm_cell_1855/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/MatMul’
,while/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1855_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1855/MatMul_1/ReadVariableOp∆
while/lstm_cell_1855/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/MatMul_1ј
while/lstm_cell_1855/addAddV2%while/lstm_cell_1855/MatMul:product:0'while/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/addќ
+while/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1855_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1855/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1855/BiasAddBiasAddwhile/lstm_cell_1855/add:z:03while/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/BiasAddО
$while/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1855/split/split_dimУ
while/lstm_cell_1855/splitSplit-while/lstm_cell_1855/split/split_dim:output:0%while/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1855/splitЮ
while/lstm_cell_1855/SigmoidSigmoid#while/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/SigmoidҐ
while/lstm_cell_1855/Sigmoid_1Sigmoid#while/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1855/Sigmoid_1¶
while/lstm_cell_1855/mulMul"while/lstm_cell_1855/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/mulХ
while/lstm_cell_1855/ReluRelu#while/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/ReluЉ
while/lstm_cell_1855/mul_1Mul while/lstm_cell_1855/Sigmoid:y:0'while/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/mul_1±
while/lstm_cell_1855/add_1AddV2while/lstm_cell_1855/mul:z:0while/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/add_1Ґ
while/lstm_cell_1855/Sigmoid_2Sigmoid#while/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1855/Sigmoid_2Ф
while/lstm_cell_1855/Relu_1Reluwhile/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/Relu_1ј
while/lstm_cell_1855/mul_2Mul"while/lstm_cell_1855/Sigmoid_2:y:0)while/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1855/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1855/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1855/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1855/BiasAdd/ReadVariableOp+^while/lstm_cell_1855/MatMul/ReadVariableOp-^while/lstm_cell_1855/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1855_biasadd_readvariableop_resource6while_lstm_cell_1855_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1855_matmul_1_readvariableop_resource7while_lstm_cell_1855_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1855_matmul_readvariableop_resource5while_lstm_cell_1855_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1855/BiasAdd/ReadVariableOp+while/lstm_cell_1855/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1855/MatMul/ReadVariableOp*while/lstm_cell_1855/MatMul/ReadVariableOp2\
,while/lstm_cell_1855/MatMul_1/ReadVariableOp,while/lstm_cell_1855/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
зe
Ћ
$forward_lstm_618_while_body_62327164>
:forward_lstm_618_while_forward_lstm_618_while_loop_counterD
@forward_lstm_618_while_forward_lstm_618_while_maximum_iterations&
"forward_lstm_618_while_placeholder(
$forward_lstm_618_while_placeholder_1(
$forward_lstm_618_while_placeholder_2(
$forward_lstm_618_while_placeholder_3(
$forward_lstm_618_while_placeholder_4=
9forward_lstm_618_while_forward_lstm_618_strided_slice_1_0y
uforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_618_while_greater_forward_lstm_618_cast_0Y
Fforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0:	»[
Hforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0:	2»V
Gforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0:	»#
forward_lstm_618_while_identity%
!forward_lstm_618_while_identity_1%
!forward_lstm_618_while_identity_2%
!forward_lstm_618_while_identity_3%
!forward_lstm_618_while_identity_4%
!forward_lstm_618_while_identity_5%
!forward_lstm_618_while_identity_6;
7forward_lstm_618_while_forward_lstm_618_strided_slice_1w
sforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_618_while_greater_forward_lstm_618_castW
Dforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource:	»Y
Fforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource:	2»T
Eforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource:	»ИҐ<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpҐ;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOpҐ=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpе
Hforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2J
Hforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeє
:forward_lstm_618/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_618_while_placeholderQforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02<
:forward_lstm_618/while/TensorArrayV2Read/TensorListGetItem’
forward_lstm_618/while/GreaterGreater6forward_lstm_618_while_greater_forward_lstm_618_cast_0"forward_lstm_618_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2 
forward_lstm_618/while/GreaterВ
;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOpFforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02=
;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp°
,forward_lstm_618/while/lstm_cell_1855/MatMulMatMulAforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem:item:0Cforward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2.
,forward_lstm_618/while/lstm_cell_1855/MatMulИ
=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOpHforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02?
=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpК
.forward_lstm_618/while/lstm_cell_1855/MatMul_1MatMul$forward_lstm_618_while_placeholder_3Eforward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.forward_lstm_618/while/lstm_cell_1855/MatMul_1Д
)forward_lstm_618/while/lstm_cell_1855/addAddV26forward_lstm_618/while/lstm_cell_1855/MatMul:product:08forward_lstm_618/while/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)forward_lstm_618/while/lstm_cell_1855/addБ
<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOpGforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02>
<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpС
-forward_lstm_618/while/lstm_cell_1855/BiasAddBiasAdd-forward_lstm_618/while/lstm_cell_1855/add:z:0Dforward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-forward_lstm_618/while/lstm_cell_1855/BiasAdd∞
5forward_lstm_618/while/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5forward_lstm_618/while/lstm_cell_1855/split/split_dim„
+forward_lstm_618/while/lstm_cell_1855/splitSplit>forward_lstm_618/while/lstm_cell_1855/split/split_dim:output:06forward_lstm_618/while/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2-
+forward_lstm_618/while/lstm_cell_1855/split—
-forward_lstm_618/while/lstm_cell_1855/SigmoidSigmoid4forward_lstm_618/while/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-forward_lstm_618/while/lstm_cell_1855/Sigmoid’
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_1Sigmoid4forward_lstm_618/while/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_1к
)forward_lstm_618/while/lstm_cell_1855/mulMul3forward_lstm_618/while/lstm_cell_1855/Sigmoid_1:y:0$forward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_618/while/lstm_cell_1855/mul»
*forward_lstm_618/while/lstm_cell_1855/ReluRelu4forward_lstm_618/while/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22,
*forward_lstm_618/while/lstm_cell_1855/ReluА
+forward_lstm_618/while/lstm_cell_1855/mul_1Mul1forward_lstm_618/while/lstm_cell_1855/Sigmoid:y:08forward_lstm_618/while/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_618/while/lstm_cell_1855/mul_1х
+forward_lstm_618/while/lstm_cell_1855/add_1AddV2-forward_lstm_618/while/lstm_cell_1855/mul:z:0/forward_lstm_618/while/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_618/while/lstm_cell_1855/add_1’
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_2Sigmoid4forward_lstm_618/while/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_2«
,forward_lstm_618/while/lstm_cell_1855/Relu_1Relu/forward_lstm_618/while/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,forward_lstm_618/while/lstm_cell_1855/Relu_1Д
+forward_lstm_618/while/lstm_cell_1855/mul_2Mul3forward_lstm_618/while/lstm_cell_1855/Sigmoid_2:y:0:forward_lstm_618/while/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_618/while/lstm_cell_1855/mul_2х
forward_lstm_618/while/SelectSelect"forward_lstm_618/while/Greater:z:0/forward_lstm_618/while/lstm_cell_1855/mul_2:z:0$forward_lstm_618_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/while/Selectщ
forward_lstm_618/while/Select_1Select"forward_lstm_618/while/Greater:z:0/forward_lstm_618/while/lstm_cell_1855/mul_2:z:0$forward_lstm_618_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_618/while/Select_1щ
forward_lstm_618/while/Select_2Select"forward_lstm_618/while/Greater:z:0/forward_lstm_618/while/lstm_cell_1855/add_1:z:0$forward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_618/while/Select_2Ѓ
;forward_lstm_618/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_618_while_placeholder_1"forward_lstm_618_while_placeholder&forward_lstm_618/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_618/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_618/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_618/while/add/y≠
forward_lstm_618/while/addAddV2"forward_lstm_618_while_placeholder%forward_lstm_618/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/while/addВ
forward_lstm_618/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_618/while/add_1/yЋ
forward_lstm_618/while/add_1AddV2:forward_lstm_618_while_forward_lstm_618_while_loop_counter'forward_lstm_618/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/while/add_1ѓ
forward_lstm_618/while/IdentityIdentity forward_lstm_618/while/add_1:z:0^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_618/while/Identity”
!forward_lstm_618/while/Identity_1Identity@forward_lstm_618_while_forward_lstm_618_while_maximum_iterations^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_618/while/Identity_1±
!forward_lstm_618/while/Identity_2Identityforward_lstm_618/while/add:z:0^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_618/while/Identity_2ё
!forward_lstm_618/while/Identity_3IdentityKforward_lstm_618/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_618/while/Identity_3 
!forward_lstm_618/while/Identity_4Identity&forward_lstm_618/while/Select:output:0^forward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_618/while/Identity_4ћ
!forward_lstm_618/while/Identity_5Identity(forward_lstm_618/while/Select_1:output:0^forward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_618/while/Identity_5ћ
!forward_lstm_618/while/Identity_6Identity(forward_lstm_618/while/Select_2:output:0^forward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_618/while/Identity_6є
forward_lstm_618/while/NoOpNoOp=^forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp<^forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp>^forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_618/while/NoOp"t
7forward_lstm_618_while_forward_lstm_618_strided_slice_19forward_lstm_618_while_forward_lstm_618_strided_slice_1_0"n
4forward_lstm_618_while_greater_forward_lstm_618_cast6forward_lstm_618_while_greater_forward_lstm_618_cast_0"K
forward_lstm_618_while_identity(forward_lstm_618/while/Identity:output:0"O
!forward_lstm_618_while_identity_1*forward_lstm_618/while/Identity_1:output:0"O
!forward_lstm_618_while_identity_2*forward_lstm_618/while/Identity_2:output:0"O
!forward_lstm_618_while_identity_3*forward_lstm_618/while/Identity_3:output:0"O
!forward_lstm_618_while_identity_4*forward_lstm_618/while/Identity_4:output:0"O
!forward_lstm_618_while_identity_5*forward_lstm_618/while/Identity_5:output:0"O
!forward_lstm_618_while_identity_6*forward_lstm_618/while/Identity_6:output:0"Р
Eforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resourceGforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0"Т
Fforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resourceHforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0"О
Dforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resourceFforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0"м
sforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensoruforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2|
<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp2z
;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp2~
=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Fsequential_618_bidirectional_618_backward_lstm_618_while_body_62324900В
~sequential_618_bidirectional_618_backward_lstm_618_while_sequential_618_bidirectional_618_backward_lstm_618_while_loop_counterЙ
Дsequential_618_bidirectional_618_backward_lstm_618_while_sequential_618_bidirectional_618_backward_lstm_618_while_maximum_iterationsH
Dsequential_618_bidirectional_618_backward_lstm_618_while_placeholderJ
Fsequential_618_bidirectional_618_backward_lstm_618_while_placeholder_1J
Fsequential_618_bidirectional_618_backward_lstm_618_while_placeholder_2J
Fsequential_618_bidirectional_618_backward_lstm_618_while_placeholder_3J
Fsequential_618_bidirectional_618_backward_lstm_618_while_placeholder_4Б
}sequential_618_bidirectional_618_backward_lstm_618_while_sequential_618_bidirectional_618_backward_lstm_618_strided_slice_1_0Њ
єsequential_618_bidirectional_618_backward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_sequential_618_bidirectional_618_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0|
xsequential_618_bidirectional_618_backward_lstm_618_while_less_sequential_618_bidirectional_618_backward_lstm_618_sub_1_0{
hsequential_618_bidirectional_618_backward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0:	»}
jsequential_618_bidirectional_618_backward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0:	2»x
isequential_618_bidirectional_618_backward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0:	»E
Asequential_618_bidirectional_618_backward_lstm_618_while_identityG
Csequential_618_bidirectional_618_backward_lstm_618_while_identity_1G
Csequential_618_bidirectional_618_backward_lstm_618_while_identity_2G
Csequential_618_bidirectional_618_backward_lstm_618_while_identity_3G
Csequential_618_bidirectional_618_backward_lstm_618_while_identity_4G
Csequential_618_bidirectional_618_backward_lstm_618_while_identity_5G
Csequential_618_bidirectional_618_backward_lstm_618_while_identity_6
{sequential_618_bidirectional_618_backward_lstm_618_while_sequential_618_bidirectional_618_backward_lstm_618_strided_slice_1Љ
Јsequential_618_bidirectional_618_backward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_sequential_618_bidirectional_618_backward_lstm_618_tensorarrayunstack_tensorlistfromtensorz
vsequential_618_bidirectional_618_backward_lstm_618_while_less_sequential_618_bidirectional_618_backward_lstm_618_sub_1y
fsequential_618_bidirectional_618_backward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource:	»{
hsequential_618_bidirectional_618_backward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource:	2»v
gsequential_618_bidirectional_618_backward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource:	»ИҐ^sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpҐ]sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOpҐ_sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp©
jsequential_618/bidirectional_618/backward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2l
jsequential_618/bidirectional_618/backward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeЖ
\sequential_618/bidirectional_618/backward_lstm_618/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemєsequential_618_bidirectional_618_backward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_sequential_618_bidirectional_618_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0Dsequential_618_bidirectional_618_backward_lstm_618_while_placeholderssequential_618/bidirectional_618/backward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02^
\sequential_618/bidirectional_618/backward_lstm_618/while/TensorArrayV2Read/TensorListGetItemф
=sequential_618/bidirectional_618/backward_lstm_618/while/LessLessxsequential_618_bidirectional_618_backward_lstm_618_while_less_sequential_618_bidirectional_618_backward_lstm_618_sub_1_0Dsequential_618_bidirectional_618_backward_lstm_618_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2?
=sequential_618/bidirectional_618/backward_lstm_618/while/Lessи
]sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOphsequential_618_bidirectional_618_backward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02_
]sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp©
Nsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMulMatMulcsequential_618/bidirectional_618/backward_lstm_618/while/TensorArrayV2Read/TensorListGetItem:item:0esequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2P
Nsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMulо
_sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOpjsequential_618_bidirectional_618_backward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02a
_sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpТ
Psequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMul_1MatMulFsequential_618_bidirectional_618_backward_lstm_618_while_placeholder_3gsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2R
Psequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMul_1М
Ksequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/addAddV2Xsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMul:product:0Zsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2M
Ksequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/addз
^sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOpisequential_618_bidirectional_618_backward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02`
^sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpЩ
Osequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/BiasAddBiasAddOsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/add:z:0fsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2Q
Osequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/BiasAddф
Wsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2Y
Wsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/split/split_dimя
Msequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/splitSplit`sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/split/split_dim:output:0Xsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2O
Msequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/splitЈ
Osequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/SigmoidSigmoidVsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22Q
Osequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/Sigmoidї
Qsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/Sigmoid_1SigmoidVsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22S
Qsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/Sigmoid_1т
Ksequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/mulMulUsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/Sigmoid_1:y:0Fsequential_618_bidirectional_618_backward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22M
Ksequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/mulЃ
Lsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/ReluReluVsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22N
Lsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/ReluИ
Msequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/mul_1MulSsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/Sigmoid:y:0Zsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22O
Msequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/mul_1э
Msequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/add_1AddV2Osequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/mul:z:0Qsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22O
Msequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/add_1ї
Qsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/Sigmoid_2SigmoidVsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22S
Qsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/Sigmoid_2≠
Nsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/Relu_1ReluQsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22P
Nsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/Relu_1М
Msequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/mul_2MulUsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/Sigmoid_2:y:0\sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22O
Msequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/mul_2Ь
?sequential_618/bidirectional_618/backward_lstm_618/while/SelectSelectAsequential_618/bidirectional_618/backward_lstm_618/while/Less:z:0Qsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/mul_2:z:0Fsequential_618_bidirectional_618_backward_lstm_618_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22A
?sequential_618/bidirectional_618/backward_lstm_618/while/Select†
Asequential_618/bidirectional_618/backward_lstm_618/while/Select_1SelectAsequential_618/bidirectional_618/backward_lstm_618/while/Less:z:0Qsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/mul_2:z:0Fsequential_618_bidirectional_618_backward_lstm_618_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22C
Asequential_618/bidirectional_618/backward_lstm_618/while/Select_1†
Asequential_618/bidirectional_618/backward_lstm_618/while/Select_2SelectAsequential_618/bidirectional_618/backward_lstm_618/while/Less:z:0Qsequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/add_1:z:0Fsequential_618_bidirectional_618_backward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22C
Asequential_618/bidirectional_618/backward_lstm_618/while/Select_2Ў
]sequential_618/bidirectional_618/backward_lstm_618/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemFsequential_618_bidirectional_618_backward_lstm_618_while_placeholder_1Dsequential_618_bidirectional_618_backward_lstm_618_while_placeholderHsequential_618/bidirectional_618/backward_lstm_618/while/Select:output:0*
_output_shapes
: *
element_dtype02_
]sequential_618/bidirectional_618/backward_lstm_618/while/TensorArrayV2Write/TensorListSetItem¬
>sequential_618/bidirectional_618/backward_lstm_618/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_618/bidirectional_618/backward_lstm_618/while/add/yµ
<sequential_618/bidirectional_618/backward_lstm_618/while/addAddV2Dsequential_618_bidirectional_618_backward_lstm_618_while_placeholderGsequential_618/bidirectional_618/backward_lstm_618/while/add/y:output:0*
T0*
_output_shapes
: 2>
<sequential_618/bidirectional_618/backward_lstm_618/while/add∆
@sequential_618/bidirectional_618/backward_lstm_618/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2B
@sequential_618/bidirectional_618/backward_lstm_618/while/add_1/yх
>sequential_618/bidirectional_618/backward_lstm_618/while/add_1AddV2~sequential_618_bidirectional_618_backward_lstm_618_while_sequential_618_bidirectional_618_backward_lstm_618_while_loop_counterIsequential_618/bidirectional_618/backward_lstm_618/while/add_1/y:output:0*
T0*
_output_shapes
: 2@
>sequential_618/bidirectional_618/backward_lstm_618/while/add_1Ј
Asequential_618/bidirectional_618/backward_lstm_618/while/IdentityIdentityBsequential_618/bidirectional_618/backward_lstm_618/while/add_1:z:0>^sequential_618/bidirectional_618/backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2C
Asequential_618/bidirectional_618/backward_lstm_618/while/Identityю
Csequential_618/bidirectional_618/backward_lstm_618/while/Identity_1IdentityДsequential_618_bidirectional_618_backward_lstm_618_while_sequential_618_bidirectional_618_backward_lstm_618_while_maximum_iterations>^sequential_618/bidirectional_618/backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2E
Csequential_618/bidirectional_618/backward_lstm_618/while/Identity_1є
Csequential_618/bidirectional_618/backward_lstm_618/while/Identity_2Identity@sequential_618/bidirectional_618/backward_lstm_618/while/add:z:0>^sequential_618/bidirectional_618/backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2E
Csequential_618/bidirectional_618/backward_lstm_618/while/Identity_2ж
Csequential_618/bidirectional_618/backward_lstm_618/while/Identity_3Identitymsequential_618/bidirectional_618/backward_lstm_618/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential_618/bidirectional_618/backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2E
Csequential_618/bidirectional_618/backward_lstm_618/while/Identity_3“
Csequential_618/bidirectional_618/backward_lstm_618/while/Identity_4IdentityHsequential_618/bidirectional_618/backward_lstm_618/while/Select:output:0>^sequential_618/bidirectional_618/backward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22E
Csequential_618/bidirectional_618/backward_lstm_618/while/Identity_4‘
Csequential_618/bidirectional_618/backward_lstm_618/while/Identity_5IdentityJsequential_618/bidirectional_618/backward_lstm_618/while/Select_1:output:0>^sequential_618/bidirectional_618/backward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22E
Csequential_618/bidirectional_618/backward_lstm_618/while/Identity_5‘
Csequential_618/bidirectional_618/backward_lstm_618/while/Identity_6IdentityJsequential_618/bidirectional_618/backward_lstm_618/while/Select_2:output:0>^sequential_618/bidirectional_618/backward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22E
Csequential_618/bidirectional_618/backward_lstm_618/while/Identity_6г
=sequential_618/bidirectional_618/backward_lstm_618/while/NoOpNoOp_^sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp^^sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp`^sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2?
=sequential_618/bidirectional_618/backward_lstm_618/while/NoOp"П
Asequential_618_bidirectional_618_backward_lstm_618_while_identityJsequential_618/bidirectional_618/backward_lstm_618/while/Identity:output:0"У
Csequential_618_bidirectional_618_backward_lstm_618_while_identity_1Lsequential_618/bidirectional_618/backward_lstm_618/while/Identity_1:output:0"У
Csequential_618_bidirectional_618_backward_lstm_618_while_identity_2Lsequential_618/bidirectional_618/backward_lstm_618/while/Identity_2:output:0"У
Csequential_618_bidirectional_618_backward_lstm_618_while_identity_3Lsequential_618/bidirectional_618/backward_lstm_618/while/Identity_3:output:0"У
Csequential_618_bidirectional_618_backward_lstm_618_while_identity_4Lsequential_618/bidirectional_618/backward_lstm_618/while/Identity_4:output:0"У
Csequential_618_bidirectional_618_backward_lstm_618_while_identity_5Lsequential_618/bidirectional_618/backward_lstm_618/while/Identity_5:output:0"У
Csequential_618_bidirectional_618_backward_lstm_618_while_identity_6Lsequential_618/bidirectional_618/backward_lstm_618/while/Identity_6:output:0"т
vsequential_618_bidirectional_618_backward_lstm_618_while_less_sequential_618_bidirectional_618_backward_lstm_618_sub_1xsequential_618_bidirectional_618_backward_lstm_618_while_less_sequential_618_bidirectional_618_backward_lstm_618_sub_1_0"‘
gsequential_618_bidirectional_618_backward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resourceisequential_618_bidirectional_618_backward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0"÷
hsequential_618_bidirectional_618_backward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resourcejsequential_618_bidirectional_618_backward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0"“
fsequential_618_bidirectional_618_backward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resourcehsequential_618_bidirectional_618_backward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0"ь
{sequential_618_bidirectional_618_backward_lstm_618_while_sequential_618_bidirectional_618_backward_lstm_618_strided_slice_1}sequential_618_bidirectional_618_backward_lstm_618_while_sequential_618_bidirectional_618_backward_lstm_618_strided_slice_1_0"ц
Јsequential_618_bidirectional_618_backward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_sequential_618_bidirectional_618_backward_lstm_618_tensorarrayunstack_tensorlistfromtensorєsequential_618_bidirectional_618_backward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_sequential_618_bidirectional_618_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2ј
^sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp^sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp2Њ
]sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp]sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp2¬
_sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp_sequential_618/bidirectional_618/backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
4__inference_backward_lstm_618_layer_call_fn_62330162

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
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_623267812
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
ь

Ў
1__inference_sequential_618_layer_call_fn_62327984

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
L__inference_sequential_618_layer_call_and_return_conditional_losses_623279432
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
Њ
ъ
1__inference_lstm_cell_1855_layer_call_fn_62330791

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
L__inference_lstm_cell_1855_layer_call_and_return_conditional_losses_623250792
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
 

Ќ
&__inference_signature_wrapper_62328060

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
#__inference__wrapped_model_623250042
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
Є
£
L__inference_sequential_618_layer_call_and_return_conditional_losses_62327472

inputs
inputs_1	-
bidirectional_618_62327441:	»-
bidirectional_618_62327443:	2»)
bidirectional_618_62327445:	»-
bidirectional_618_62327447:	»-
bidirectional_618_62327449:	2»)
bidirectional_618_62327451:	»$
dense_618_62327466:d 
dense_618_62327468:
identityИҐ)bidirectional_618/StatefulPartitionedCallҐ!dense_618/StatefulPartitionedCall 
)bidirectional_618/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_618_62327441bidirectional_618_62327443bidirectional_618_62327445bidirectional_618_62327447bidirectional_618_62327449bidirectional_618_62327451*
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
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_623274402+
)bidirectional_618/StatefulPartitionedCallЋ
!dense_618/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_618/StatefulPartitionedCall:output:0dense_618_62327466dense_618_62327468*
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
G__inference_dense_618_layer_call_and_return_conditional_losses_623274652#
!dense_618/StatefulPartitionedCallЕ
IdentityIdentity*dense_618/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityЮ
NoOpNoOp*^bidirectional_618/StatefulPartitionedCall"^dense_618/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€:€€€€€€€€€: : : : : : : : 2V
)bidirectional_618/StatefulPartitionedCall)bidirectional_618/StatefulPartitionedCall2F
!dense_618/StatefulPartitionedCall!dense_618/StatefulPartitionedCall:O K
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
while_body_62330384
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1856_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1856_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1856_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1856_matmul_readvariableop_resource:	»H
5while_lstm_cell_1856_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1856_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1856/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1856/MatMul/ReadVariableOpҐ,while/lstm_cell_1856/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1856_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1856/MatMul/ReadVariableOpЁ
while/lstm_cell_1856/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/MatMul’
,while/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1856_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1856/MatMul_1/ReadVariableOp∆
while/lstm_cell_1856/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/MatMul_1ј
while/lstm_cell_1856/addAddV2%while/lstm_cell_1856/MatMul:product:0'while/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/addќ
+while/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1856_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1856/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1856/BiasAddBiasAddwhile/lstm_cell_1856/add:z:03while/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/BiasAddО
$while/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1856/split/split_dimУ
while/lstm_cell_1856/splitSplit-while/lstm_cell_1856/split/split_dim:output:0%while/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1856/splitЮ
while/lstm_cell_1856/SigmoidSigmoid#while/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/SigmoidҐ
while/lstm_cell_1856/Sigmoid_1Sigmoid#while/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1856/Sigmoid_1¶
while/lstm_cell_1856/mulMul"while/lstm_cell_1856/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/mulХ
while/lstm_cell_1856/ReluRelu#while/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/ReluЉ
while/lstm_cell_1856/mul_1Mul while/lstm_cell_1856/Sigmoid:y:0'while/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/mul_1±
while/lstm_cell_1856/add_1AddV2while/lstm_cell_1856/mul:z:0while/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/add_1Ґ
while/lstm_cell_1856/Sigmoid_2Sigmoid#while/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1856/Sigmoid_2Ф
while/lstm_cell_1856/Relu_1Reluwhile/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/Relu_1ј
while/lstm_cell_1856/mul_2Mul"while/lstm_cell_1856/Sigmoid_2:y:0)while/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1856/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1856/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1856/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1856/BiasAdd/ReadVariableOp+^while/lstm_cell_1856/MatMul/ReadVariableOp-^while/lstm_cell_1856/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1856_biasadd_readvariableop_resource6while_lstm_cell_1856_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1856_matmul_1_readvariableop_resource7while_lstm_cell_1856_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1856_matmul_readvariableop_resource5while_lstm_cell_1856_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1856/BiasAdd/ReadVariableOp+while/lstm_cell_1856/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1856/MatMul/ReadVariableOp*while/lstm_cell_1856/MatMul/ReadVariableOp2\
,while/lstm_cell_1856/MatMul_1/ReadVariableOp,while/lstm_cell_1856/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Т
‘
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_62326600

inputs,
forward_lstm_618_62326430:	»,
forward_lstm_618_62326432:	2»(
forward_lstm_618_62326434:	»-
backward_lstm_618_62326590:	»-
backward_lstm_618_62326592:	2»)
backward_lstm_618_62326594:	»
identityИҐ)backward_lstm_618/StatefulPartitionedCallҐ(forward_lstm_618/StatefulPartitionedCallя
(forward_lstm_618/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_618_62326430forward_lstm_618_62326432forward_lstm_618_62326434*
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
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_623264292*
(forward_lstm_618/StatefulPartitionedCallе
)backward_lstm_618/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_618_62326590backward_lstm_618_62326592backward_lstm_618_62326594*
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
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_623265892+
)backward_lstm_618/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis‘
concatConcatV21forward_lstm_618/StatefulPartitionedCall:output:02backward_lstm_618/StatefulPartitionedCall:output:0concat/axis:output:0*
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
NoOpNoOp*^backward_lstm_618/StatefulPartitionedCall)^forward_lstm_618/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : 2V
)backward_lstm_618/StatefulPartitionedCall)backward_lstm_618/StatefulPartitionedCall2T
(forward_lstm_618/StatefulPartitionedCall(forward_lstm_618/StatefulPartitionedCall:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
я
Ќ
while_cond_62326696
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_62326696___redundant_placeholder06
2while_while_cond_62326696___redundant_placeholder16
2while_while_cond_62326696___redundant_placeholder26
2while_while_cond_62326696___redundant_placeholder3
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
%backward_lstm_618_while_cond_62328345@
<backward_lstm_618_while_backward_lstm_618_while_loop_counterF
Bbackward_lstm_618_while_backward_lstm_618_while_maximum_iterations'
#backward_lstm_618_while_placeholder)
%backward_lstm_618_while_placeholder_1)
%backward_lstm_618_while_placeholder_2)
%backward_lstm_618_while_placeholder_3B
>backward_lstm_618_while_less_backward_lstm_618_strided_slice_1Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62328345___redundant_placeholder0Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62328345___redundant_placeholder1Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62328345___redundant_placeholder2Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62328345___redundant_placeholder3$
 backward_lstm_618_while_identity
 
backward_lstm_618/while/LessLess#backward_lstm_618_while_placeholder>backward_lstm_618_while_less_backward_lstm_618_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_618/while/LessУ
 backward_lstm_618/while/IdentityIdentity backward_lstm_618/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_618/while/Identity"M
 backward_lstm_618_while_identity)backward_lstm_618/while/Identity:output:0*(
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
while_body_62326870
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1855_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1855_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1855_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1855_matmul_readvariableop_resource:	»H
5while_lstm_cell_1855_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1855_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1855/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1855/MatMul/ReadVariableOpҐ,while/lstm_cell_1855/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1855_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1855/MatMul/ReadVariableOpЁ
while/lstm_cell_1855/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/MatMul’
,while/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1855_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1855/MatMul_1/ReadVariableOp∆
while/lstm_cell_1855/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/MatMul_1ј
while/lstm_cell_1855/addAddV2%while/lstm_cell_1855/MatMul:product:0'while/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/addќ
+while/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1855_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1855/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1855/BiasAddBiasAddwhile/lstm_cell_1855/add:z:03while/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/BiasAddО
$while/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1855/split/split_dimУ
while/lstm_cell_1855/splitSplit-while/lstm_cell_1855/split/split_dim:output:0%while/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1855/splitЮ
while/lstm_cell_1855/SigmoidSigmoid#while/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/SigmoidҐ
while/lstm_cell_1855/Sigmoid_1Sigmoid#while/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1855/Sigmoid_1¶
while/lstm_cell_1855/mulMul"while/lstm_cell_1855/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/mulХ
while/lstm_cell_1855/ReluRelu#while/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/ReluЉ
while/lstm_cell_1855/mul_1Mul while/lstm_cell_1855/Sigmoid:y:0'while/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/mul_1±
while/lstm_cell_1855/add_1AddV2while/lstm_cell_1855/mul:z:0while/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/add_1Ґ
while/lstm_cell_1855/Sigmoid_2Sigmoid#while/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1855/Sigmoid_2Ф
while/lstm_cell_1855/Relu_1Reluwhile/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/Relu_1ј
while/lstm_cell_1855/mul_2Mul"while/lstm_cell_1855/Sigmoid_2:y:0)while/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1855/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1855/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1855/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1855/BiasAdd/ReadVariableOp+^while/lstm_cell_1855/MatMul/ReadVariableOp-^while/lstm_cell_1855/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1855_biasadd_readvariableop_resource6while_lstm_cell_1855_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1855_matmul_1_readvariableop_resource7while_lstm_cell_1855_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1855_matmul_readvariableop_resource5while_lstm_cell_1855_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1855/BiasAdd/ReadVariableOp+while/lstm_cell_1855/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1855/MatMul/ReadVariableOp*while/lstm_cell_1855/MatMul/ReadVariableOp2\
,while/lstm_cell_1855/MatMul_1/ReadVariableOp,while/lstm_cell_1855/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_cond_62325936
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_62325936___redundant_placeholder06
2while_while_cond_62325936___redundant_placeholder16
2while_while_cond_62325936___redundant_placeholder26
2while_while_cond_62325936___redundant_placeholder3
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
1__inference_lstm_cell_1855_layer_call_fn_62330808

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
L__inference_lstm_cell_1855_layer_call_and_return_conditional_losses_623252252
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
яc
ю
!__inference__traced_save_62331111
file_prefix/
+savev2_dense_618_kernel_read_readvariableop-
)savev2_dense_618_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopW
Ssavev2_bidirectional_618_forward_lstm_618_lstm_cell_1855_kernel_read_readvariableopa
]savev2_bidirectional_618_forward_lstm_618_lstm_cell_1855_recurrent_kernel_read_readvariableopU
Qsavev2_bidirectional_618_forward_lstm_618_lstm_cell_1855_bias_read_readvariableopX
Tsavev2_bidirectional_618_backward_lstm_618_lstm_cell_1856_kernel_read_readvariableopb
^savev2_bidirectional_618_backward_lstm_618_lstm_cell_1856_recurrent_kernel_read_readvariableopV
Rsavev2_bidirectional_618_backward_lstm_618_lstm_cell_1856_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_618_kernel_m_read_readvariableop4
0savev2_adam_dense_618_bias_m_read_readvariableop^
Zsavev2_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_kernel_m_read_readvariableoph
dsavev2_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_recurrent_kernel_m_read_readvariableop\
Xsavev2_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_bias_m_read_readvariableop_
[savev2_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_kernel_m_read_readvariableopi
esavev2_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_recurrent_kernel_m_read_readvariableop]
Ysavev2_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_bias_m_read_readvariableop6
2savev2_adam_dense_618_kernel_v_read_readvariableop4
0savev2_adam_dense_618_bias_v_read_readvariableop^
Zsavev2_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_kernel_v_read_readvariableoph
dsavev2_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_recurrent_kernel_v_read_readvariableop\
Xsavev2_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_bias_v_read_readvariableop_
[savev2_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_kernel_v_read_readvariableopi
esavev2_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_recurrent_kernel_v_read_readvariableop]
Ysavev2_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_bias_v_read_readvariableop9
5savev2_adam_dense_618_kernel_vhat_read_readvariableop7
3savev2_adam_dense_618_bias_vhat_read_readvariableopa
]savev2_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_kernel_vhat_read_readvariableopk
gsavev2_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_recurrent_kernel_vhat_read_readvariableop_
[savev2_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_bias_vhat_read_readvariableopb
^savev2_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_kernel_vhat_read_readvariableopl
hsavev2_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_recurrent_kernel_vhat_read_readvariableop`
\savev2_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_bias_vhat_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_618_kernel_read_readvariableop)savev2_dense_618_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopSsavev2_bidirectional_618_forward_lstm_618_lstm_cell_1855_kernel_read_readvariableop]savev2_bidirectional_618_forward_lstm_618_lstm_cell_1855_recurrent_kernel_read_readvariableopQsavev2_bidirectional_618_forward_lstm_618_lstm_cell_1855_bias_read_readvariableopTsavev2_bidirectional_618_backward_lstm_618_lstm_cell_1856_kernel_read_readvariableop^savev2_bidirectional_618_backward_lstm_618_lstm_cell_1856_recurrent_kernel_read_readvariableopRsavev2_bidirectional_618_backward_lstm_618_lstm_cell_1856_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_618_kernel_m_read_readvariableop0savev2_adam_dense_618_bias_m_read_readvariableopZsavev2_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_kernel_m_read_readvariableopdsavev2_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_recurrent_kernel_m_read_readvariableopXsavev2_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_bias_m_read_readvariableop[savev2_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_kernel_m_read_readvariableopesavev2_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_recurrent_kernel_m_read_readvariableopYsavev2_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_bias_m_read_readvariableop2savev2_adam_dense_618_kernel_v_read_readvariableop0savev2_adam_dense_618_bias_v_read_readvariableopZsavev2_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_kernel_v_read_readvariableopdsavev2_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_recurrent_kernel_v_read_readvariableopXsavev2_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_bias_v_read_readvariableop[savev2_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_kernel_v_read_readvariableopesavev2_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_recurrent_kernel_v_read_readvariableopYsavev2_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_bias_v_read_readvariableop5savev2_adam_dense_618_kernel_vhat_read_readvariableop3savev2_adam_dense_618_bias_vhat_read_readvariableop]savev2_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_kernel_vhat_read_readvariableopgsavev2_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_recurrent_kernel_vhat_read_readvariableop[savev2_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_bias_vhat_read_readvariableop^savev2_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_kernel_vhat_read_readvariableophsavev2_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_recurrent_kernel_vhat_read_readvariableop\savev2_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_bias_vhat_read_readvariableopsavev2_const"/device:CPU:0*
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
ч
И
L__inference_lstm_cell_1856_layer_call_and_return_conditional_losses_62325711

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
while_body_62326505
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1856_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1856_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1856_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1856_matmul_readvariableop_resource:	»H
5while_lstm_cell_1856_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1856_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1856/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1856/MatMul/ReadVariableOpҐ,while/lstm_cell_1856/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1856_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1856/MatMul/ReadVariableOpЁ
while/lstm_cell_1856/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/MatMul’
,while/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1856_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1856/MatMul_1/ReadVariableOp∆
while/lstm_cell_1856/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/MatMul_1ј
while/lstm_cell_1856/addAddV2%while/lstm_cell_1856/MatMul:product:0'while/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/addќ
+while/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1856_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1856/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1856/BiasAddBiasAddwhile/lstm_cell_1856/add:z:03while/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/BiasAddО
$while/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1856/split/split_dimУ
while/lstm_cell_1856/splitSplit-while/lstm_cell_1856/split/split_dim:output:0%while/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1856/splitЮ
while/lstm_cell_1856/SigmoidSigmoid#while/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/SigmoidҐ
while/lstm_cell_1856/Sigmoid_1Sigmoid#while/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1856/Sigmoid_1¶
while/lstm_cell_1856/mulMul"while/lstm_cell_1856/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/mulХ
while/lstm_cell_1856/ReluRelu#while/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/ReluЉ
while/lstm_cell_1856/mul_1Mul while/lstm_cell_1856/Sigmoid:y:0'while/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/mul_1±
while/lstm_cell_1856/add_1AddV2while/lstm_cell_1856/mul:z:0while/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/add_1Ґ
while/lstm_cell_1856/Sigmoid_2Sigmoid#while/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1856/Sigmoid_2Ф
while/lstm_cell_1856/Relu_1Reluwhile/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/Relu_1ј
while/lstm_cell_1856/mul_2Mul"while/lstm_cell_1856/Sigmoid_2:y:0)while/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1856/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1856/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1856/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1856/BiasAdd/ReadVariableOp+^while/lstm_cell_1856/MatMul/ReadVariableOp-^while/lstm_cell_1856/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1856_biasadd_readvariableop_resource6while_lstm_cell_1856_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1856_matmul_1_readvariableop_resource7while_lstm_cell_1856_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1856_matmul_readvariableop_resource5while_lstm_cell_1856_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1856/BiasAdd/ReadVariableOp+while/lstm_cell_1856/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1856/MatMul/ReadVariableOp*while/lstm_cell_1856/MatMul/ReadVariableOp2\
,while/lstm_cell_1856/MatMul_1/ReadVariableOp,while/lstm_cell_1856/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
%backward_lstm_618_while_body_62329353@
<backward_lstm_618_while_backward_lstm_618_while_loop_counterF
Bbackward_lstm_618_while_backward_lstm_618_while_maximum_iterations'
#backward_lstm_618_while_placeholder)
%backward_lstm_618_while_placeholder_1)
%backward_lstm_618_while_placeholder_2)
%backward_lstm_618_while_placeholder_3)
%backward_lstm_618_while_placeholder_4?
;backward_lstm_618_while_backward_lstm_618_strided_slice_1_0{
wbackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_618_while_less_backward_lstm_618_sub_1_0Z
Gbackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0:	»\
Ibackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0:	2»W
Hbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0:	»$
 backward_lstm_618_while_identity&
"backward_lstm_618_while_identity_1&
"backward_lstm_618_while_identity_2&
"backward_lstm_618_while_identity_3&
"backward_lstm_618_while_identity_4&
"backward_lstm_618_while_identity_5&
"backward_lstm_618_while_identity_6=
9backward_lstm_618_while_backward_lstm_618_strided_slice_1y
ubackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_618_while_less_backward_lstm_618_sub_1X
Ebackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource:	»Z
Gbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource:	2»U
Fbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource:	»ИҐ=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpҐ<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOpҐ>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpз
Ibackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2K
Ibackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeњ
;backward_lstm_618/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_618_while_placeholderRbackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02=
;backward_lstm_618/while/TensorArrayV2Read/TensorListGetItemѕ
backward_lstm_618/while/LessLess6backward_lstm_618_while_less_backward_lstm_618_sub_1_0#backward_lstm_618_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_618/while/LessЕ
<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOpGbackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02>
<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp•
-backward_lstm_618/while/lstm_cell_1856/MatMulMatMulBbackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem:item:0Dbackward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-backward_lstm_618/while/lstm_cell_1856/MatMulЛ
>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOpIbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02@
>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpО
/backward_lstm_618/while/lstm_cell_1856/MatMul_1MatMul%backward_lstm_618_while_placeholder_3Fbackward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»21
/backward_lstm_618/while/lstm_cell_1856/MatMul_1И
*backward_lstm_618/while/lstm_cell_1856/addAddV27backward_lstm_618/while/lstm_cell_1856/MatMul:product:09backward_lstm_618/while/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2,
*backward_lstm_618/while/lstm_cell_1856/addД
=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOpHbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02?
=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpХ
.backward_lstm_618/while/lstm_cell_1856/BiasAddBiasAdd.backward_lstm_618/while/lstm_cell_1856/add:z:0Ebackward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.backward_lstm_618/while/lstm_cell_1856/BiasAdd≤
6backward_lstm_618/while/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6backward_lstm_618/while/lstm_cell_1856/split/split_dimџ
,backward_lstm_618/while/lstm_cell_1856/splitSplit?backward_lstm_618/while/lstm_cell_1856/split/split_dim:output:07backward_lstm_618/while/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2.
,backward_lstm_618/while/lstm_cell_1856/split‘
.backward_lstm_618/while/lstm_cell_1856/SigmoidSigmoid5backward_lstm_618/while/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€220
.backward_lstm_618/while/lstm_cell_1856/SigmoidЎ
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_1Sigmoid5backward_lstm_618/while/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_1о
*backward_lstm_618/while/lstm_cell_1856/mulMul4backward_lstm_618/while/lstm_cell_1856/Sigmoid_1:y:0%backward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_618/while/lstm_cell_1856/mulЋ
+backward_lstm_618/while/lstm_cell_1856/ReluRelu5backward_lstm_618/while/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22-
+backward_lstm_618/while/lstm_cell_1856/ReluД
,backward_lstm_618/while/lstm_cell_1856/mul_1Mul2backward_lstm_618/while/lstm_cell_1856/Sigmoid:y:09backward_lstm_618/while/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_618/while/lstm_cell_1856/mul_1щ
,backward_lstm_618/while/lstm_cell_1856/add_1AddV2.backward_lstm_618/while/lstm_cell_1856/mul:z:00backward_lstm_618/while/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_618/while/lstm_cell_1856/add_1Ў
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_2Sigmoid5backward_lstm_618/while/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_2 
-backward_lstm_618/while/lstm_cell_1856/Relu_1Relu0backward_lstm_618/while/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-backward_lstm_618/while/lstm_cell_1856/Relu_1И
,backward_lstm_618/while/lstm_cell_1856/mul_2Mul4backward_lstm_618/while/lstm_cell_1856/Sigmoid_2:y:0;backward_lstm_618/while/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_618/while/lstm_cell_1856/mul_2ч
backward_lstm_618/while/SelectSelect backward_lstm_618/while/Less:z:00backward_lstm_618/while/lstm_cell_1856/mul_2:z:0%backward_lstm_618_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22 
backward_lstm_618/while/Selectы
 backward_lstm_618/while/Select_1Select backward_lstm_618/while/Less:z:00backward_lstm_618/while/lstm_cell_1856/mul_2:z:0%backward_lstm_618_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_618/while/Select_1ы
 backward_lstm_618/while/Select_2Select backward_lstm_618/while/Less:z:00backward_lstm_618/while/lstm_cell_1856/add_1:z:0%backward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_618/while/Select_2≥
<backward_lstm_618/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_618_while_placeholder_1#backward_lstm_618_while_placeholder'backward_lstm_618/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_618/while/TensorArrayV2Write/TensorListSetItemА
backward_lstm_618/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_618/while/add/y±
backward_lstm_618/while/addAddV2#backward_lstm_618_while_placeholder&backward_lstm_618/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/while/addД
backward_lstm_618/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_618/while/add_1/y–
backward_lstm_618/while/add_1AddV2<backward_lstm_618_while_backward_lstm_618_while_loop_counter(backward_lstm_618/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/while/add_1≥
 backward_lstm_618/while/IdentityIdentity!backward_lstm_618/while/add_1:z:0^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_618/while/IdentityЎ
"backward_lstm_618/while/Identity_1IdentityBbackward_lstm_618_while_backward_lstm_618_while_maximum_iterations^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_618/while/Identity_1µ
"backward_lstm_618/while/Identity_2Identitybackward_lstm_618/while/add:z:0^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_618/while/Identity_2в
"backward_lstm_618/while/Identity_3IdentityLbackward_lstm_618/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_618/while/Identity_3ќ
"backward_lstm_618/while/Identity_4Identity'backward_lstm_618/while/Select:output:0^backward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_618/while/Identity_4–
"backward_lstm_618/while/Identity_5Identity)backward_lstm_618/while/Select_1:output:0^backward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_618/while/Identity_5–
"backward_lstm_618/while/Identity_6Identity)backward_lstm_618/while/Select_2:output:0^backward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_618/while/Identity_6Њ
backward_lstm_618/while/NoOpNoOp>^backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp=^backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp?^backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_618/while/NoOp"x
9backward_lstm_618_while_backward_lstm_618_strided_slice_1;backward_lstm_618_while_backward_lstm_618_strided_slice_1_0"M
 backward_lstm_618_while_identity)backward_lstm_618/while/Identity:output:0"Q
"backward_lstm_618_while_identity_1+backward_lstm_618/while/Identity_1:output:0"Q
"backward_lstm_618_while_identity_2+backward_lstm_618/while/Identity_2:output:0"Q
"backward_lstm_618_while_identity_3+backward_lstm_618/while/Identity_3:output:0"Q
"backward_lstm_618_while_identity_4+backward_lstm_618/while/Identity_4:output:0"Q
"backward_lstm_618_while_identity_5+backward_lstm_618/while/Identity_5:output:0"Q
"backward_lstm_618_while_identity_6+backward_lstm_618/while/Identity_6:output:0"n
4backward_lstm_618_while_less_backward_lstm_618_sub_16backward_lstm_618_while_less_backward_lstm_618_sub_1_0"Т
Fbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resourceHbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0"Ф
Gbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resourceIbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0"Р
Ebackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resourceGbackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0"р
ubackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2~
=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp2|
<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp2А
>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_62329967

inputs@
-lstm_cell_1855_matmul_readvariableop_resource:	»B
/lstm_cell_1855_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1855_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1855/BiasAdd/ReadVariableOpҐ$lstm_cell_1855/MatMul/ReadVariableOpҐ&lstm_cell_1855/MatMul_1/ReadVariableOpҐwhileD
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
$lstm_cell_1855/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1855_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1855/MatMul/ReadVariableOp≥
lstm_cell_1855/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/MatMulЅ
&lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1855_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1855/MatMul_1/ReadVariableOpѓ
lstm_cell_1855/MatMul_1MatMulzeros:output:0.lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/MatMul_1®
lstm_cell_1855/addAddV2lstm_cell_1855/MatMul:product:0!lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/addЇ
%lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1855_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1855/BiasAdd/ReadVariableOpµ
lstm_cell_1855/BiasAddBiasAddlstm_cell_1855/add:z:0-lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/BiasAddВ
lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1855/split/split_dimы
lstm_cell_1855/splitSplit'lstm_cell_1855/split/split_dim:output:0lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1855/splitМ
lstm_cell_1855/SigmoidSigmoidlstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/SigmoidР
lstm_cell_1855/Sigmoid_1Sigmoidlstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Sigmoid_1С
lstm_cell_1855/mulMullstm_cell_1855/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/mulГ
lstm_cell_1855/ReluRelulstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Relu§
lstm_cell_1855/mul_1Mullstm_cell_1855/Sigmoid:y:0!lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/mul_1Щ
lstm_cell_1855/add_1AddV2lstm_cell_1855/mul:z:0lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/add_1Р
lstm_cell_1855/Sigmoid_2Sigmoidlstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Sigmoid_2В
lstm_cell_1855/Relu_1Relulstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Relu_1®
lstm_cell_1855/mul_2Mullstm_cell_1855/Sigmoid_2:y:0#lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1855_matmul_readvariableop_resource/lstm_cell_1855_matmul_1_readvariableop_resource.lstm_cell_1855_biasadd_readvariableop_resource*
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
while_body_62329883*
condR
while_cond_62329882*K
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
NoOpNoOp&^lstm_cell_1855/BiasAdd/ReadVariableOp%^lstm_cell_1855/MatMul/ReadVariableOp'^lstm_cell_1855/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1855/BiasAdd/ReadVariableOp%lstm_cell_1855/BiasAdd/ReadVariableOp2L
$lstm_cell_1855/MatMul/ReadVariableOp$lstm_cell_1855/MatMul/ReadVariableOp2P
&lstm_cell_1855/MatMul_1/ReadVariableOp&lstm_cell_1855/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Њ
ъ
1__inference_lstm_cell_1856_layer_call_fn_62330906

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
L__inference_lstm_cell_1856_layer_call_and_return_conditional_losses_623258572
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
$forward_lstm_618_while_body_62328197>
:forward_lstm_618_while_forward_lstm_618_while_loop_counterD
@forward_lstm_618_while_forward_lstm_618_while_maximum_iterations&
"forward_lstm_618_while_placeholder(
$forward_lstm_618_while_placeholder_1(
$forward_lstm_618_while_placeholder_2(
$forward_lstm_618_while_placeholder_3=
9forward_lstm_618_while_forward_lstm_618_strided_slice_1_0y
uforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0Y
Fforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0:	»[
Hforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0:	2»V
Gforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0:	»#
forward_lstm_618_while_identity%
!forward_lstm_618_while_identity_1%
!forward_lstm_618_while_identity_2%
!forward_lstm_618_while_identity_3%
!forward_lstm_618_while_identity_4%
!forward_lstm_618_while_identity_5;
7forward_lstm_618_while_forward_lstm_618_strided_slice_1w
sforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensorW
Dforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource:	»Y
Fforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource:	2»T
Eforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource:	»ИҐ<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpҐ;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOpҐ=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpе
Hforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2J
Hforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shape¬
:forward_lstm_618/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_618_while_placeholderQforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02<
:forward_lstm_618/while/TensorArrayV2Read/TensorListGetItemВ
;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOpFforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02=
;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp°
,forward_lstm_618/while/lstm_cell_1855/MatMulMatMulAforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem:item:0Cforward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2.
,forward_lstm_618/while/lstm_cell_1855/MatMulИ
=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOpHforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02?
=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpК
.forward_lstm_618/while/lstm_cell_1855/MatMul_1MatMul$forward_lstm_618_while_placeholder_2Eforward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.forward_lstm_618/while/lstm_cell_1855/MatMul_1Д
)forward_lstm_618/while/lstm_cell_1855/addAddV26forward_lstm_618/while/lstm_cell_1855/MatMul:product:08forward_lstm_618/while/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)forward_lstm_618/while/lstm_cell_1855/addБ
<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOpGforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02>
<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpС
-forward_lstm_618/while/lstm_cell_1855/BiasAddBiasAdd-forward_lstm_618/while/lstm_cell_1855/add:z:0Dforward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-forward_lstm_618/while/lstm_cell_1855/BiasAdd∞
5forward_lstm_618/while/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5forward_lstm_618/while/lstm_cell_1855/split/split_dim„
+forward_lstm_618/while/lstm_cell_1855/splitSplit>forward_lstm_618/while/lstm_cell_1855/split/split_dim:output:06forward_lstm_618/while/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2-
+forward_lstm_618/while/lstm_cell_1855/split—
-forward_lstm_618/while/lstm_cell_1855/SigmoidSigmoid4forward_lstm_618/while/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-forward_lstm_618/while/lstm_cell_1855/Sigmoid’
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_1Sigmoid4forward_lstm_618/while/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_1к
)forward_lstm_618/while/lstm_cell_1855/mulMul3forward_lstm_618/while/lstm_cell_1855/Sigmoid_1:y:0$forward_lstm_618_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_618/while/lstm_cell_1855/mul»
*forward_lstm_618/while/lstm_cell_1855/ReluRelu4forward_lstm_618/while/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22,
*forward_lstm_618/while/lstm_cell_1855/ReluА
+forward_lstm_618/while/lstm_cell_1855/mul_1Mul1forward_lstm_618/while/lstm_cell_1855/Sigmoid:y:08forward_lstm_618/while/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_618/while/lstm_cell_1855/mul_1х
+forward_lstm_618/while/lstm_cell_1855/add_1AddV2-forward_lstm_618/while/lstm_cell_1855/mul:z:0/forward_lstm_618/while/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_618/while/lstm_cell_1855/add_1’
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_2Sigmoid4forward_lstm_618/while/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_2«
,forward_lstm_618/while/lstm_cell_1855/Relu_1Relu/forward_lstm_618/while/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,forward_lstm_618/while/lstm_cell_1855/Relu_1Д
+forward_lstm_618/while/lstm_cell_1855/mul_2Mul3forward_lstm_618/while/lstm_cell_1855/Sigmoid_2:y:0:forward_lstm_618/while/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_618/while/lstm_cell_1855/mul_2Ј
;forward_lstm_618/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_618_while_placeholder_1"forward_lstm_618_while_placeholder/forward_lstm_618/while/lstm_cell_1855/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_618/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_618/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_618/while/add/y≠
forward_lstm_618/while/addAddV2"forward_lstm_618_while_placeholder%forward_lstm_618/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/while/addВ
forward_lstm_618/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_618/while/add_1/yЋ
forward_lstm_618/while/add_1AddV2:forward_lstm_618_while_forward_lstm_618_while_loop_counter'forward_lstm_618/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/while/add_1ѓ
forward_lstm_618/while/IdentityIdentity forward_lstm_618/while/add_1:z:0^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_618/while/Identity”
!forward_lstm_618/while/Identity_1Identity@forward_lstm_618_while_forward_lstm_618_while_maximum_iterations^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_618/while/Identity_1±
!forward_lstm_618/while/Identity_2Identityforward_lstm_618/while/add:z:0^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_618/while/Identity_2ё
!forward_lstm_618/while/Identity_3IdentityKforward_lstm_618/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_618/while/Identity_3”
!forward_lstm_618/while/Identity_4Identity/forward_lstm_618/while/lstm_cell_1855/mul_2:z:0^forward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_618/while/Identity_4”
!forward_lstm_618/while/Identity_5Identity/forward_lstm_618/while/lstm_cell_1855/add_1:z:0^forward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_618/while/Identity_5є
forward_lstm_618/while/NoOpNoOp=^forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp<^forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp>^forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_618/while/NoOp"t
7forward_lstm_618_while_forward_lstm_618_strided_slice_19forward_lstm_618_while_forward_lstm_618_strided_slice_1_0"K
forward_lstm_618_while_identity(forward_lstm_618/while/Identity:output:0"O
!forward_lstm_618_while_identity_1*forward_lstm_618/while/Identity_1:output:0"O
!forward_lstm_618_while_identity_2*forward_lstm_618/while/Identity_2:output:0"O
!forward_lstm_618_while_identity_3*forward_lstm_618/while/Identity_3:output:0"O
!forward_lstm_618_while_identity_4*forward_lstm_618/while/Identity_4:output:0"O
!forward_lstm_618_while_identity_5*forward_lstm_618/while/Identity_5:output:0"Р
Eforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resourceGforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0"Т
Fforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resourceHforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0"О
Dforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resourceFforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0"м
sforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensoruforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2|
<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp2z
;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp2~
=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_body_62325093
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_lstm_cell_1855_62325117_0:	»2
while_lstm_cell_1855_62325119_0:	2».
while_lstm_cell_1855_62325121_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_lstm_cell_1855_62325117:	»0
while_lstm_cell_1855_62325119:	2»,
while_lstm_cell_1855_62325121:	»ИҐ,while/lstm_cell_1855/StatefulPartitionedCall√
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
,while/lstm_cell_1855/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1855_62325117_0while_lstm_cell_1855_62325119_0while_lstm_cell_1855_62325121_0*
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
L__inference_lstm_cell_1855_layer_call_and_return_conditional_losses_623250792.
,while/lstm_cell_1855/StatefulPartitionedCallщ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder5while/lstm_cell_1855/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity5while/lstm_cell_1855/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4¶
while/Identity_5Identity5while/lstm_cell_1855/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5Й

while/NoOpNoOp-^while/lstm_cell_1855/StatefulPartitionedCall*"
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
while_lstm_cell_1855_62325117while_lstm_cell_1855_62325117_0"@
while_lstm_cell_1855_62325119while_lstm_cell_1855_62325119_0"@
while_lstm_cell_1855_62325121while_lstm_cell_1855_62325121_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2\
,while/lstm_cell_1855/StatefulPartitionedCall,while/lstm_cell_1855/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
Т
‘
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_62327002

inputs,
forward_lstm_618_62326985:	»,
forward_lstm_618_62326987:	2»(
forward_lstm_618_62326989:	»-
backward_lstm_618_62326992:	»-
backward_lstm_618_62326994:	2»)
backward_lstm_618_62326996:	»
identityИҐ)backward_lstm_618/StatefulPartitionedCallҐ(forward_lstm_618/StatefulPartitionedCallя
(forward_lstm_618/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_618_62326985forward_lstm_618_62326987forward_lstm_618_62326989*
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
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_623269542*
(forward_lstm_618/StatefulPartitionedCallе
)backward_lstm_618/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_618_62326992backward_lstm_618_62326994backward_lstm_618_62326996*
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
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_623267812+
)backward_lstm_618/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis‘
concatConcatV21forward_lstm_618/StatefulPartitionedCall:output:02backward_lstm_618/StatefulPartitionedCall:output:0concat/axis:output:0*
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
NoOpNoOp*^backward_lstm_618/StatefulPartitionedCall)^forward_lstm_618/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : 2V
)backward_lstm_618/StatefulPartitionedCall)backward_lstm_618/StatefulPartitionedCall2T
(forward_lstm_618/StatefulPartitionedCall(forward_lstm_618/StatefulPartitionedCall:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ч
Щ
,__inference_dense_618_layer_call_fn_62329459

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
G__inference_dense_618_layer_call_and_return_conditional_losses_623274652
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
∆@
д
while_body_62330231
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1856_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1856_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1856_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1856_matmul_readvariableop_resource:	»H
5while_lstm_cell_1856_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1856_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1856/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1856/MatMul/ReadVariableOpҐ,while/lstm_cell_1856/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1856_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1856/MatMul/ReadVariableOpЁ
while/lstm_cell_1856/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/MatMul’
,while/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1856_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1856/MatMul_1/ReadVariableOp∆
while/lstm_cell_1856/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/MatMul_1ј
while/lstm_cell_1856/addAddV2%while/lstm_cell_1856/MatMul:product:0'while/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/addќ
+while/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1856_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1856/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1856/BiasAddBiasAddwhile/lstm_cell_1856/add:z:03while/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/BiasAddО
$while/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1856/split/split_dimУ
while/lstm_cell_1856/splitSplit-while/lstm_cell_1856/split/split_dim:output:0%while/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1856/splitЮ
while/lstm_cell_1856/SigmoidSigmoid#while/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/SigmoidҐ
while/lstm_cell_1856/Sigmoid_1Sigmoid#while/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1856/Sigmoid_1¶
while/lstm_cell_1856/mulMul"while/lstm_cell_1856/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/mulХ
while/lstm_cell_1856/ReluRelu#while/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/ReluЉ
while/lstm_cell_1856/mul_1Mul while/lstm_cell_1856/Sigmoid:y:0'while/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/mul_1±
while/lstm_cell_1856/add_1AddV2while/lstm_cell_1856/mul:z:0while/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/add_1Ґ
while/lstm_cell_1856/Sigmoid_2Sigmoid#while/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1856/Sigmoid_2Ф
while/lstm_cell_1856/Relu_1Reluwhile/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/Relu_1ј
while/lstm_cell_1856/mul_2Mul"while/lstm_cell_1856/Sigmoid_2:y:0)while/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1856/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1856/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1856/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1856/BiasAdd/ReadVariableOp+^while/lstm_cell_1856/MatMul/ReadVariableOp-^while/lstm_cell_1856/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1856_biasadd_readvariableop_resource6while_lstm_cell_1856_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1856_matmul_1_readvariableop_resource7while_lstm_cell_1856_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1856_matmul_readvariableop_resource5while_lstm_cell_1856_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1856/BiasAdd/ReadVariableOp+while/lstm_cell_1856/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1856/MatMul/ReadVariableOp*while/lstm_cell_1856/MatMul/ReadVariableOp2\
,while/lstm_cell_1856/MatMul_1/ReadVariableOp,while/lstm_cell_1856/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_62328432
inputs_0Q
>forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource:	»S
@forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource:	2»N
?forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource:	»R
?backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resource:	»T
Abackward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource:	2»O
@backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource:	»
identityИҐ7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpҐ6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpҐ8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpҐbackward_lstm_618/whileҐ6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpҐ5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpҐ7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpҐforward_lstm_618/whileh
forward_lstm_618/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_618/ShapeЦ
$forward_lstm_618/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_618/strided_slice/stackЪ
&forward_lstm_618/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_618/strided_slice/stack_1Ъ
&forward_lstm_618/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_618/strided_slice/stack_2»
forward_lstm_618/strided_sliceStridedSliceforward_lstm_618/Shape:output:0-forward_lstm_618/strided_slice/stack:output:0/forward_lstm_618/strided_slice/stack_1:output:0/forward_lstm_618/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_618/strided_slice~
forward_lstm_618/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_618/zeros/mul/y∞
forward_lstm_618/zeros/mulMul'forward_lstm_618/strided_slice:output:0%forward_lstm_618/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros/mulБ
forward_lstm_618/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
forward_lstm_618/zeros/Less/yЂ
forward_lstm_618/zeros/LessLessforward_lstm_618/zeros/mul:z:0&forward_lstm_618/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros/LessД
forward_lstm_618/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_618/zeros/packed/1«
forward_lstm_618/zeros/packedPack'forward_lstm_618/strided_slice:output:0(forward_lstm_618/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_618/zeros/packedЕ
forward_lstm_618/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_618/zeros/Constє
forward_lstm_618/zerosFill&forward_lstm_618/zeros/packed:output:0%forward_lstm_618/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/zerosВ
forward_lstm_618/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_618/zeros_1/mul/yґ
forward_lstm_618/zeros_1/mulMul'forward_lstm_618/strided_slice:output:0'forward_lstm_618/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros_1/mulЕ
forward_lstm_618/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2!
forward_lstm_618/zeros_1/Less/y≥
forward_lstm_618/zeros_1/LessLess forward_lstm_618/zeros_1/mul:z:0(forward_lstm_618/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros_1/LessИ
!forward_lstm_618/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_618/zeros_1/packed/1Ќ
forward_lstm_618/zeros_1/packedPack'forward_lstm_618/strided_slice:output:0*forward_lstm_618/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_618/zeros_1/packedЙ
forward_lstm_618/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_618/zeros_1/ConstЅ
forward_lstm_618/zeros_1Fill(forward_lstm_618/zeros_1/packed:output:0'forward_lstm_618/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/zeros_1Ч
forward_lstm_618/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_618/transpose/permЅ
forward_lstm_618/transpose	Transposeinputs_0(forward_lstm_618/transpose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
forward_lstm_618/transposeВ
forward_lstm_618/Shape_1Shapeforward_lstm_618/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_618/Shape_1Ъ
&forward_lstm_618/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_618/strided_slice_1/stackЮ
(forward_lstm_618/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_1/stack_1Ю
(forward_lstm_618/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_1/stack_2‘
 forward_lstm_618/strided_slice_1StridedSlice!forward_lstm_618/Shape_1:output:0/forward_lstm_618/strided_slice_1/stack:output:01forward_lstm_618/strided_slice_1/stack_1:output:01forward_lstm_618/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_618/strided_slice_1І
,forward_lstm_618/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,forward_lstm_618/TensorArrayV2/element_shapeц
forward_lstm_618/TensorArrayV2TensorListReserve5forward_lstm_618/TensorArrayV2/element_shape:output:0)forward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_618/TensorArrayV2б
Fforward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2H
Fforward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeЉ
8forward_lstm_618/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_618/transpose:y:0Oforward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_618/TensorArrayUnstack/TensorListFromTensorЪ
&forward_lstm_618/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_618/strided_slice_2/stackЮ
(forward_lstm_618/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_2/stack_1Ю
(forward_lstm_618/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_2/stack_2л
 forward_lstm_618/strided_slice_2StridedSliceforward_lstm_618/transpose:y:0/forward_lstm_618/strided_slice_2/stack:output:01forward_lstm_618/strided_slice_2/stack_1:output:01forward_lstm_618/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2"
 forward_lstm_618/strided_slice_2о
5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOp>forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype027
5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpч
&forward_lstm_618/lstm_cell_1855/MatMulMatMul)forward_lstm_618/strided_slice_2:output:0=forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2(
&forward_lstm_618/lstm_cell_1855/MatMulф
7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOp@forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype029
7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpу
(forward_lstm_618/lstm_cell_1855/MatMul_1MatMulforward_lstm_618/zeros:output:0?forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(forward_lstm_618/lstm_cell_1855/MatMul_1м
#forward_lstm_618/lstm_cell_1855/addAddV20forward_lstm_618/lstm_cell_1855/MatMul:product:02forward_lstm_618/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2%
#forward_lstm_618/lstm_cell_1855/addн
6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOp?forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype028
6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpщ
'forward_lstm_618/lstm_cell_1855/BiasAddBiasAdd'forward_lstm_618/lstm_cell_1855/add:z:0>forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'forward_lstm_618/lstm_cell_1855/BiasAdd§
/forward_lstm_618/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/forward_lstm_618/lstm_cell_1855/split/split_dimњ
%forward_lstm_618/lstm_cell_1855/splitSplit8forward_lstm_618/lstm_cell_1855/split/split_dim:output:00forward_lstm_618/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2'
%forward_lstm_618/lstm_cell_1855/splitњ
'forward_lstm_618/lstm_cell_1855/SigmoidSigmoid.forward_lstm_618/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'forward_lstm_618/lstm_cell_1855/Sigmoid√
)forward_lstm_618/lstm_cell_1855/Sigmoid_1Sigmoid.forward_lstm_618/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_618/lstm_cell_1855/Sigmoid_1’
#forward_lstm_618/lstm_cell_1855/mulMul-forward_lstm_618/lstm_cell_1855/Sigmoid_1:y:0!forward_lstm_618/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22%
#forward_lstm_618/lstm_cell_1855/mulґ
$forward_lstm_618/lstm_cell_1855/ReluRelu.forward_lstm_618/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22&
$forward_lstm_618/lstm_cell_1855/Reluи
%forward_lstm_618/lstm_cell_1855/mul_1Mul+forward_lstm_618/lstm_cell_1855/Sigmoid:y:02forward_lstm_618/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_618/lstm_cell_1855/mul_1Ё
%forward_lstm_618/lstm_cell_1855/add_1AddV2'forward_lstm_618/lstm_cell_1855/mul:z:0)forward_lstm_618/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_618/lstm_cell_1855/add_1√
)forward_lstm_618/lstm_cell_1855/Sigmoid_2Sigmoid.forward_lstm_618/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_618/lstm_cell_1855/Sigmoid_2µ
&forward_lstm_618/lstm_cell_1855/Relu_1Relu)forward_lstm_618/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&forward_lstm_618/lstm_cell_1855/Relu_1м
%forward_lstm_618/lstm_cell_1855/mul_2Mul-forward_lstm_618/lstm_cell_1855/Sigmoid_2:y:04forward_lstm_618/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_618/lstm_cell_1855/mul_2±
.forward_lstm_618/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   20
.forward_lstm_618/TensorArrayV2_1/element_shapeь
 forward_lstm_618/TensorArrayV2_1TensorListReserve7forward_lstm_618/TensorArrayV2_1/element_shape:output:0)forward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_618/TensorArrayV2_1p
forward_lstm_618/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_618/time°
)forward_lstm_618/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2+
)forward_lstm_618/while/maximum_iterationsМ
#forward_lstm_618/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_618/while/loop_counterФ
forward_lstm_618/whileWhile,forward_lstm_618/while/loop_counter:output:02forward_lstm_618/while/maximum_iterations:output:0forward_lstm_618/time:output:0)forward_lstm_618/TensorArrayV2_1:handle:0forward_lstm_618/zeros:output:0!forward_lstm_618/zeros_1:output:0)forward_lstm_618/strided_slice_1:output:0Hforward_lstm_618/TensorArrayUnstack/TensorListFromTensor:output_handle:0>forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource@forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource?forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource*
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
$forward_lstm_618_while_body_62328197*0
cond(R&
$forward_lstm_618_while_cond_62328196*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
forward_lstm_618/while„
Aforward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2C
Aforward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeµ
3forward_lstm_618/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_618/while:output:3Jforward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype025
3forward_lstm_618/TensorArrayV2Stack/TensorListStack£
&forward_lstm_618/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2(
&forward_lstm_618/strided_slice_3/stackЮ
(forward_lstm_618/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_618/strided_slice_3/stack_1Ю
(forward_lstm_618/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_3/stack_2А
 forward_lstm_618/strided_slice_3StridedSlice<forward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_618/strided_slice_3/stack:output:01forward_lstm_618/strided_slice_3/stack_1:output:01forward_lstm_618/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2"
 forward_lstm_618/strided_slice_3Ы
!forward_lstm_618/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_618/transpose_1/permт
forward_lstm_618/transpose_1	Transpose<forward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_618/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
forward_lstm_618/transpose_1И
forward_lstm_618/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_618/runtimej
backward_lstm_618/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_618/ShapeШ
%backward_lstm_618/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_618/strided_slice/stackЬ
'backward_lstm_618/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_618/strided_slice/stack_1Ь
'backward_lstm_618/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_618/strided_slice/stack_2ќ
backward_lstm_618/strided_sliceStridedSlice backward_lstm_618/Shape:output:0.backward_lstm_618/strided_slice/stack:output:00backward_lstm_618/strided_slice/stack_1:output:00backward_lstm_618/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_618/strided_sliceА
backward_lstm_618/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_618/zeros/mul/yі
backward_lstm_618/zeros/mulMul(backward_lstm_618/strided_slice:output:0&backward_lstm_618/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/zeros/mulГ
backward_lstm_618/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
backward_lstm_618/zeros/Less/yѓ
backward_lstm_618/zeros/LessLessbackward_lstm_618/zeros/mul:z:0'backward_lstm_618/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/zeros/LessЖ
 backward_lstm_618/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_618/zeros/packed/1Ћ
backward_lstm_618/zeros/packedPack(backward_lstm_618/strided_slice:output:0)backward_lstm_618/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_618/zeros/packedЗ
backward_lstm_618/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_618/zeros/Constљ
backward_lstm_618/zerosFill'backward_lstm_618/zeros/packed:output:0&backward_lstm_618/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_618/zerosД
backward_lstm_618/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_618/zeros_1/mul/yЇ
backward_lstm_618/zeros_1/mulMul(backward_lstm_618/strided_slice:output:0(backward_lstm_618/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/zeros_1/mulЗ
 backward_lstm_618/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 backward_lstm_618/zeros_1/Less/yЈ
backward_lstm_618/zeros_1/LessLess!backward_lstm_618/zeros_1/mul:z:0)backward_lstm_618/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_618/zeros_1/LessК
"backward_lstm_618/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_618/zeros_1/packed/1—
 backward_lstm_618/zeros_1/packedPack(backward_lstm_618/strided_slice:output:0+backward_lstm_618/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_618/zeros_1/packedЛ
backward_lstm_618/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_618/zeros_1/Const≈
backward_lstm_618/zeros_1Fill)backward_lstm_618/zeros_1/packed:output:0(backward_lstm_618/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_618/zeros_1Щ
 backward_lstm_618/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_618/transpose/permƒ
backward_lstm_618/transpose	Transposeinputs_0)backward_lstm_618/transpose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
backward_lstm_618/transposeЕ
backward_lstm_618/Shape_1Shapebackward_lstm_618/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_618/Shape_1Ь
'backward_lstm_618/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_618/strided_slice_1/stack†
)backward_lstm_618/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_1/stack_1†
)backward_lstm_618/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_1/stack_2Џ
!backward_lstm_618/strided_slice_1StridedSlice"backward_lstm_618/Shape_1:output:00backward_lstm_618/strided_slice_1/stack:output:02backward_lstm_618/strided_slice_1/stack_1:output:02backward_lstm_618/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_618/strided_slice_1©
-backward_lstm_618/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-backward_lstm_618/TensorArrayV2/element_shapeъ
backward_lstm_618/TensorArrayV2TensorListReserve6backward_lstm_618/TensorArrayV2/element_shape:output:0*backward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_618/TensorArrayV2О
 backward_lstm_618/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_618/ReverseV2/axisџ
backward_lstm_618/ReverseV2	ReverseV2backward_lstm_618/transpose:y:0)backward_lstm_618/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
backward_lstm_618/ReverseV2г
Gbackward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2I
Gbackward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape≈
9backward_lstm_618/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_618/ReverseV2:output:0Pbackward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_618/TensorArrayUnstack/TensorListFromTensorЬ
'backward_lstm_618/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_618/strided_slice_2/stack†
)backward_lstm_618/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_2/stack_1†
)backward_lstm_618/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_2/stack_2с
!backward_lstm_618/strided_slice_2StridedSlicebackward_lstm_618/transpose:y:00backward_lstm_618/strided_slice_2/stack:output:02backward_lstm_618/strided_slice_2/stack_1:output:02backward_lstm_618/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2#
!backward_lstm_618/strided_slice_2с
6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOp?backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype028
6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpы
'backward_lstm_618/lstm_cell_1856/MatMulMatMul*backward_lstm_618/strided_slice_2:output:0>backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'backward_lstm_618/lstm_cell_1856/MatMulч
8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOpAbackward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02:
8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpч
)backward_lstm_618/lstm_cell_1856/MatMul_1MatMul backward_lstm_618/zeros:output:0@backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)backward_lstm_618/lstm_cell_1856/MatMul_1р
$backward_lstm_618/lstm_cell_1856/addAddV21backward_lstm_618/lstm_cell_1856/MatMul:product:03backward_lstm_618/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2&
$backward_lstm_618/lstm_cell_1856/addр
7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOp@backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype029
7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpэ
(backward_lstm_618/lstm_cell_1856/BiasAddBiasAdd(backward_lstm_618/lstm_cell_1856/add:z:0?backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(backward_lstm_618/lstm_cell_1856/BiasAdd¶
0backward_lstm_618/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0backward_lstm_618/lstm_cell_1856/split/split_dim√
&backward_lstm_618/lstm_cell_1856/splitSplit9backward_lstm_618/lstm_cell_1856/split/split_dim:output:01backward_lstm_618/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2(
&backward_lstm_618/lstm_cell_1856/split¬
(backward_lstm_618/lstm_cell_1856/SigmoidSigmoid/backward_lstm_618/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(backward_lstm_618/lstm_cell_1856/Sigmoid∆
*backward_lstm_618/lstm_cell_1856/Sigmoid_1Sigmoid/backward_lstm_618/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_618/lstm_cell_1856/Sigmoid_1ў
$backward_lstm_618/lstm_cell_1856/mulMul.backward_lstm_618/lstm_cell_1856/Sigmoid_1:y:0"backward_lstm_618/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$backward_lstm_618/lstm_cell_1856/mulє
%backward_lstm_618/lstm_cell_1856/ReluRelu/backward_lstm_618/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22'
%backward_lstm_618/lstm_cell_1856/Reluм
&backward_lstm_618/lstm_cell_1856/mul_1Mul,backward_lstm_618/lstm_cell_1856/Sigmoid:y:03backward_lstm_618/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_618/lstm_cell_1856/mul_1б
&backward_lstm_618/lstm_cell_1856/add_1AddV2(backward_lstm_618/lstm_cell_1856/mul:z:0*backward_lstm_618/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_618/lstm_cell_1856/add_1∆
*backward_lstm_618/lstm_cell_1856/Sigmoid_2Sigmoid/backward_lstm_618/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_618/lstm_cell_1856/Sigmoid_2Є
'backward_lstm_618/lstm_cell_1856/Relu_1Relu*backward_lstm_618/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'backward_lstm_618/lstm_cell_1856/Relu_1р
&backward_lstm_618/lstm_cell_1856/mul_2Mul.backward_lstm_618/lstm_cell_1856/Sigmoid_2:y:05backward_lstm_618/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_618/lstm_cell_1856/mul_2≥
/backward_lstm_618/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   21
/backward_lstm_618/TensorArrayV2_1/element_shapeА
!backward_lstm_618/TensorArrayV2_1TensorListReserve8backward_lstm_618/TensorArrayV2_1/element_shape:output:0*backward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_618/TensorArrayV2_1r
backward_lstm_618/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_618/time£
*backward_lstm_618/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2,
*backward_lstm_618/while/maximum_iterationsО
$backward_lstm_618/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_618/while/loop_counter£
backward_lstm_618/whileWhile-backward_lstm_618/while/loop_counter:output:03backward_lstm_618/while/maximum_iterations:output:0backward_lstm_618/time:output:0*backward_lstm_618/TensorArrayV2_1:handle:0 backward_lstm_618/zeros:output:0"backward_lstm_618/zeros_1:output:0*backward_lstm_618/strided_slice_1:output:0Ibackward_lstm_618/TensorArrayUnstack/TensorListFromTensor:output_handle:0?backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resourceAbackward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource@backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource*
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
%backward_lstm_618_while_body_62328346*1
cond)R'
%backward_lstm_618_while_cond_62328345*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
backward_lstm_618/whileў
Bbackward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2D
Bbackward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeє
4backward_lstm_618/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_618/while:output:3Kbackward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype026
4backward_lstm_618/TensorArrayV2Stack/TensorListStack•
'backward_lstm_618/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2)
'backward_lstm_618/strided_slice_3/stack†
)backward_lstm_618/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_618/strided_slice_3/stack_1†
)backward_lstm_618/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_3/stack_2Ж
!backward_lstm_618/strided_slice_3StridedSlice=backward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_618/strided_slice_3/stack:output:02backward_lstm_618/strided_slice_3/stack_1:output:02backward_lstm_618/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2#
!backward_lstm_618/strided_slice_3Э
"backward_lstm_618/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_618/transpose_1/permц
backward_lstm_618/transpose_1	Transpose=backward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_618/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
backward_lstm_618/transpose_1К
backward_lstm_618/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_618/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisƒ
concatConcatV2)forward_lstm_618/strided_slice_3:output:0*backward_lstm_618/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp8^backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp7^backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp9^backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp^backward_lstm_618/while7^forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp6^forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp8^forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp^forward_lstm_618/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : 2r
7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp2p
6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp2t
8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp22
backward_lstm_618/whilebackward_lstm_618/while2p
6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp2n
5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp2r
7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp20
forward_lstm_618/whileforward_lstm_618/while:g c
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
÷
√
4__inference_backward_lstm_618_layer_call_fn_62330140
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
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_623260062
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
while_cond_62330383
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_62330383___redundant_placeholder06
2while_while_cond_62330383___redundant_placeholder16
2while_while_cond_62330383___redundant_placeholder26
2while_while_cond_62330383___redundant_placeholder3
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
while_body_62325725
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_lstm_cell_1856_62325749_0:	»2
while_lstm_cell_1856_62325751_0:	2».
while_lstm_cell_1856_62325753_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_lstm_cell_1856_62325749:	»0
while_lstm_cell_1856_62325751:	2»,
while_lstm_cell_1856_62325753:	»ИҐ,while/lstm_cell_1856/StatefulPartitionedCall√
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
,while/lstm_cell_1856/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1856_62325749_0while_lstm_cell_1856_62325751_0while_lstm_cell_1856_62325753_0*
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
L__inference_lstm_cell_1856_layer_call_and_return_conditional_losses_623257112.
,while/lstm_cell_1856/StatefulPartitionedCallщ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder5while/lstm_cell_1856/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity5while/lstm_cell_1856/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4¶
while/Identity_5Identity5while/lstm_cell_1856/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5Й

while/NoOpNoOp-^while/lstm_cell_1856/StatefulPartitionedCall*"
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
while_lstm_cell_1856_62325749while_lstm_cell_1856_62325749_0"@
while_lstm_cell_1856_62325751while_lstm_cell_1856_62325751_0"@
while_lstm_cell_1856_62325753while_lstm_cell_1856_62325753_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2\
,while/lstm_cell_1856/StatefulPartitionedCall,while/lstm_cell_1856/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
while_cond_62329882
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_62329882___redundant_placeholder06
2while_while_cond_62329882___redundant_placeholder16
2while_while_cond_62329882___redundant_placeholder26
2while_while_cond_62329882___redundant_placeholder3
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
а
ј
3__inference_forward_lstm_618_layer_call_fn_62329514

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
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_623269542
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
ѕ@
д
while_body_62330537
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1856_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1856_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1856_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1856_matmul_readvariableop_resource:	»H
5while_lstm_cell_1856_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1856_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1856/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1856/MatMul/ReadVariableOpҐ,while/lstm_cell_1856/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1856_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1856/MatMul/ReadVariableOpЁ
while/lstm_cell_1856/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/MatMul’
,while/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1856_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1856/MatMul_1/ReadVariableOp∆
while/lstm_cell_1856/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/MatMul_1ј
while/lstm_cell_1856/addAddV2%while/lstm_cell_1856/MatMul:product:0'while/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/addќ
+while/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1856_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1856/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1856/BiasAddBiasAddwhile/lstm_cell_1856/add:z:03while/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/BiasAddО
$while/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1856/split/split_dimУ
while/lstm_cell_1856/splitSplit-while/lstm_cell_1856/split/split_dim:output:0%while/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1856/splitЮ
while/lstm_cell_1856/SigmoidSigmoid#while/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/SigmoidҐ
while/lstm_cell_1856/Sigmoid_1Sigmoid#while/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1856/Sigmoid_1¶
while/lstm_cell_1856/mulMul"while/lstm_cell_1856/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/mulХ
while/lstm_cell_1856/ReluRelu#while/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/ReluЉ
while/lstm_cell_1856/mul_1Mul while/lstm_cell_1856/Sigmoid:y:0'while/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/mul_1±
while/lstm_cell_1856/add_1AddV2while/lstm_cell_1856/mul:z:0while/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/add_1Ґ
while/lstm_cell_1856/Sigmoid_2Sigmoid#while/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1856/Sigmoid_2Ф
while/lstm_cell_1856/Relu_1Reluwhile/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/Relu_1ј
while/lstm_cell_1856/mul_2Mul"while/lstm_cell_1856/Sigmoid_2:y:0)while/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1856/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1856/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1856/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1856/BiasAdd/ReadVariableOp+^while/lstm_cell_1856/MatMul/ReadVariableOp-^while/lstm_cell_1856/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1856_biasadd_readvariableop_resource6while_lstm_cell_1856_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1856_matmul_1_readvariableop_resource7while_lstm_cell_1856_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1856_matmul_readvariableop_resource5while_lstm_cell_1856_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1856/BiasAdd/ReadVariableOp+while/lstm_cell_1856/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1856/MatMul/ReadVariableOp*while/lstm_cell_1856/MatMul/ReadVariableOp2\
,while/lstm_cell_1856/MatMul_1/ReadVariableOp,while/lstm_cell_1856/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_cond_62330689
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_62330689___redundant_placeholder06
2while_while_cond_62330689___redundant_placeholder16
2while_while_cond_62330689___redundant_placeholder26
2while_while_cond_62330689___redundant_placeholder3
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
1__inference_lstm_cell_1856_layer_call_fn_62330889

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
L__inference_lstm_cell_1856_layer_call_and_return_conditional_losses_623257112
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
ёю
п
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_62328734
inputs_0Q
>forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource:	»S
@forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource:	2»N
?forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource:	»R
?backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resource:	»T
Abackward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource:	2»O
@backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource:	»
identityИҐ7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpҐ6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpҐ8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpҐbackward_lstm_618/whileҐ6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpҐ5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpҐ7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpҐforward_lstm_618/whileh
forward_lstm_618/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_618/ShapeЦ
$forward_lstm_618/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_618/strided_slice/stackЪ
&forward_lstm_618/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_618/strided_slice/stack_1Ъ
&forward_lstm_618/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_618/strided_slice/stack_2»
forward_lstm_618/strided_sliceStridedSliceforward_lstm_618/Shape:output:0-forward_lstm_618/strided_slice/stack:output:0/forward_lstm_618/strided_slice/stack_1:output:0/forward_lstm_618/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_618/strided_slice~
forward_lstm_618/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_618/zeros/mul/y∞
forward_lstm_618/zeros/mulMul'forward_lstm_618/strided_slice:output:0%forward_lstm_618/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros/mulБ
forward_lstm_618/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
forward_lstm_618/zeros/Less/yЂ
forward_lstm_618/zeros/LessLessforward_lstm_618/zeros/mul:z:0&forward_lstm_618/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros/LessД
forward_lstm_618/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_618/zeros/packed/1«
forward_lstm_618/zeros/packedPack'forward_lstm_618/strided_slice:output:0(forward_lstm_618/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_618/zeros/packedЕ
forward_lstm_618/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_618/zeros/Constє
forward_lstm_618/zerosFill&forward_lstm_618/zeros/packed:output:0%forward_lstm_618/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/zerosВ
forward_lstm_618/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_618/zeros_1/mul/yґ
forward_lstm_618/zeros_1/mulMul'forward_lstm_618/strided_slice:output:0'forward_lstm_618/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros_1/mulЕ
forward_lstm_618/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2!
forward_lstm_618/zeros_1/Less/y≥
forward_lstm_618/zeros_1/LessLess forward_lstm_618/zeros_1/mul:z:0(forward_lstm_618/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros_1/LessИ
!forward_lstm_618/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_618/zeros_1/packed/1Ќ
forward_lstm_618/zeros_1/packedPack'forward_lstm_618/strided_slice:output:0*forward_lstm_618/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_618/zeros_1/packedЙ
forward_lstm_618/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_618/zeros_1/ConstЅ
forward_lstm_618/zeros_1Fill(forward_lstm_618/zeros_1/packed:output:0'forward_lstm_618/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/zeros_1Ч
forward_lstm_618/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_618/transpose/permЅ
forward_lstm_618/transpose	Transposeinputs_0(forward_lstm_618/transpose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
forward_lstm_618/transposeВ
forward_lstm_618/Shape_1Shapeforward_lstm_618/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_618/Shape_1Ъ
&forward_lstm_618/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_618/strided_slice_1/stackЮ
(forward_lstm_618/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_1/stack_1Ю
(forward_lstm_618/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_1/stack_2‘
 forward_lstm_618/strided_slice_1StridedSlice!forward_lstm_618/Shape_1:output:0/forward_lstm_618/strided_slice_1/stack:output:01forward_lstm_618/strided_slice_1/stack_1:output:01forward_lstm_618/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_618/strided_slice_1І
,forward_lstm_618/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,forward_lstm_618/TensorArrayV2/element_shapeц
forward_lstm_618/TensorArrayV2TensorListReserve5forward_lstm_618/TensorArrayV2/element_shape:output:0)forward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_618/TensorArrayV2б
Fforward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2H
Fforward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeЉ
8forward_lstm_618/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_618/transpose:y:0Oforward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_618/TensorArrayUnstack/TensorListFromTensorЪ
&forward_lstm_618/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_618/strided_slice_2/stackЮ
(forward_lstm_618/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_2/stack_1Ю
(forward_lstm_618/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_2/stack_2л
 forward_lstm_618/strided_slice_2StridedSliceforward_lstm_618/transpose:y:0/forward_lstm_618/strided_slice_2/stack:output:01forward_lstm_618/strided_slice_2/stack_1:output:01forward_lstm_618/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2"
 forward_lstm_618/strided_slice_2о
5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOp>forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype027
5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpч
&forward_lstm_618/lstm_cell_1855/MatMulMatMul)forward_lstm_618/strided_slice_2:output:0=forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2(
&forward_lstm_618/lstm_cell_1855/MatMulф
7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOp@forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype029
7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpу
(forward_lstm_618/lstm_cell_1855/MatMul_1MatMulforward_lstm_618/zeros:output:0?forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(forward_lstm_618/lstm_cell_1855/MatMul_1м
#forward_lstm_618/lstm_cell_1855/addAddV20forward_lstm_618/lstm_cell_1855/MatMul:product:02forward_lstm_618/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2%
#forward_lstm_618/lstm_cell_1855/addн
6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOp?forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype028
6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpщ
'forward_lstm_618/lstm_cell_1855/BiasAddBiasAdd'forward_lstm_618/lstm_cell_1855/add:z:0>forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'forward_lstm_618/lstm_cell_1855/BiasAdd§
/forward_lstm_618/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/forward_lstm_618/lstm_cell_1855/split/split_dimњ
%forward_lstm_618/lstm_cell_1855/splitSplit8forward_lstm_618/lstm_cell_1855/split/split_dim:output:00forward_lstm_618/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2'
%forward_lstm_618/lstm_cell_1855/splitњ
'forward_lstm_618/lstm_cell_1855/SigmoidSigmoid.forward_lstm_618/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'forward_lstm_618/lstm_cell_1855/Sigmoid√
)forward_lstm_618/lstm_cell_1855/Sigmoid_1Sigmoid.forward_lstm_618/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_618/lstm_cell_1855/Sigmoid_1’
#forward_lstm_618/lstm_cell_1855/mulMul-forward_lstm_618/lstm_cell_1855/Sigmoid_1:y:0!forward_lstm_618/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22%
#forward_lstm_618/lstm_cell_1855/mulґ
$forward_lstm_618/lstm_cell_1855/ReluRelu.forward_lstm_618/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22&
$forward_lstm_618/lstm_cell_1855/Reluи
%forward_lstm_618/lstm_cell_1855/mul_1Mul+forward_lstm_618/lstm_cell_1855/Sigmoid:y:02forward_lstm_618/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_618/lstm_cell_1855/mul_1Ё
%forward_lstm_618/lstm_cell_1855/add_1AddV2'forward_lstm_618/lstm_cell_1855/mul:z:0)forward_lstm_618/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_618/lstm_cell_1855/add_1√
)forward_lstm_618/lstm_cell_1855/Sigmoid_2Sigmoid.forward_lstm_618/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_618/lstm_cell_1855/Sigmoid_2µ
&forward_lstm_618/lstm_cell_1855/Relu_1Relu)forward_lstm_618/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&forward_lstm_618/lstm_cell_1855/Relu_1м
%forward_lstm_618/lstm_cell_1855/mul_2Mul-forward_lstm_618/lstm_cell_1855/Sigmoid_2:y:04forward_lstm_618/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_618/lstm_cell_1855/mul_2±
.forward_lstm_618/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   20
.forward_lstm_618/TensorArrayV2_1/element_shapeь
 forward_lstm_618/TensorArrayV2_1TensorListReserve7forward_lstm_618/TensorArrayV2_1/element_shape:output:0)forward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_618/TensorArrayV2_1p
forward_lstm_618/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_618/time°
)forward_lstm_618/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2+
)forward_lstm_618/while/maximum_iterationsМ
#forward_lstm_618/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_618/while/loop_counterФ
forward_lstm_618/whileWhile,forward_lstm_618/while/loop_counter:output:02forward_lstm_618/while/maximum_iterations:output:0forward_lstm_618/time:output:0)forward_lstm_618/TensorArrayV2_1:handle:0forward_lstm_618/zeros:output:0!forward_lstm_618/zeros_1:output:0)forward_lstm_618/strided_slice_1:output:0Hforward_lstm_618/TensorArrayUnstack/TensorListFromTensor:output_handle:0>forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource@forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource?forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource*
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
$forward_lstm_618_while_body_62328499*0
cond(R&
$forward_lstm_618_while_cond_62328498*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
forward_lstm_618/while„
Aforward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2C
Aforward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeµ
3forward_lstm_618/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_618/while:output:3Jforward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype025
3forward_lstm_618/TensorArrayV2Stack/TensorListStack£
&forward_lstm_618/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2(
&forward_lstm_618/strided_slice_3/stackЮ
(forward_lstm_618/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_618/strided_slice_3/stack_1Ю
(forward_lstm_618/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_3/stack_2А
 forward_lstm_618/strided_slice_3StridedSlice<forward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_618/strided_slice_3/stack:output:01forward_lstm_618/strided_slice_3/stack_1:output:01forward_lstm_618/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2"
 forward_lstm_618/strided_slice_3Ы
!forward_lstm_618/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_618/transpose_1/permт
forward_lstm_618/transpose_1	Transpose<forward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_618/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
forward_lstm_618/transpose_1И
forward_lstm_618/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_618/runtimej
backward_lstm_618/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_618/ShapeШ
%backward_lstm_618/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_618/strided_slice/stackЬ
'backward_lstm_618/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_618/strided_slice/stack_1Ь
'backward_lstm_618/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_618/strided_slice/stack_2ќ
backward_lstm_618/strided_sliceStridedSlice backward_lstm_618/Shape:output:0.backward_lstm_618/strided_slice/stack:output:00backward_lstm_618/strided_slice/stack_1:output:00backward_lstm_618/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_618/strided_sliceА
backward_lstm_618/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_618/zeros/mul/yі
backward_lstm_618/zeros/mulMul(backward_lstm_618/strided_slice:output:0&backward_lstm_618/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/zeros/mulГ
backward_lstm_618/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
backward_lstm_618/zeros/Less/yѓ
backward_lstm_618/zeros/LessLessbackward_lstm_618/zeros/mul:z:0'backward_lstm_618/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/zeros/LessЖ
 backward_lstm_618/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_618/zeros/packed/1Ћ
backward_lstm_618/zeros/packedPack(backward_lstm_618/strided_slice:output:0)backward_lstm_618/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_618/zeros/packedЗ
backward_lstm_618/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_618/zeros/Constљ
backward_lstm_618/zerosFill'backward_lstm_618/zeros/packed:output:0&backward_lstm_618/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_618/zerosД
backward_lstm_618/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_618/zeros_1/mul/yЇ
backward_lstm_618/zeros_1/mulMul(backward_lstm_618/strided_slice:output:0(backward_lstm_618/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/zeros_1/mulЗ
 backward_lstm_618/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 backward_lstm_618/zeros_1/Less/yЈ
backward_lstm_618/zeros_1/LessLess!backward_lstm_618/zeros_1/mul:z:0)backward_lstm_618/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_618/zeros_1/LessК
"backward_lstm_618/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_618/zeros_1/packed/1—
 backward_lstm_618/zeros_1/packedPack(backward_lstm_618/strided_slice:output:0+backward_lstm_618/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_618/zeros_1/packedЛ
backward_lstm_618/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_618/zeros_1/Const≈
backward_lstm_618/zeros_1Fill)backward_lstm_618/zeros_1/packed:output:0(backward_lstm_618/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_618/zeros_1Щ
 backward_lstm_618/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_618/transpose/permƒ
backward_lstm_618/transpose	Transposeinputs_0)backward_lstm_618/transpose/perm:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
backward_lstm_618/transposeЕ
backward_lstm_618/Shape_1Shapebackward_lstm_618/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_618/Shape_1Ь
'backward_lstm_618/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_618/strided_slice_1/stack†
)backward_lstm_618/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_1/stack_1†
)backward_lstm_618/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_1/stack_2Џ
!backward_lstm_618/strided_slice_1StridedSlice"backward_lstm_618/Shape_1:output:00backward_lstm_618/strided_slice_1/stack:output:02backward_lstm_618/strided_slice_1/stack_1:output:02backward_lstm_618/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_618/strided_slice_1©
-backward_lstm_618/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-backward_lstm_618/TensorArrayV2/element_shapeъ
backward_lstm_618/TensorArrayV2TensorListReserve6backward_lstm_618/TensorArrayV2/element_shape:output:0*backward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_618/TensorArrayV2О
 backward_lstm_618/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_618/ReverseV2/axisџ
backward_lstm_618/ReverseV2	ReverseV2backward_lstm_618/transpose:y:0)backward_lstm_618/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2
backward_lstm_618/ReverseV2г
Gbackward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2I
Gbackward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape≈
9backward_lstm_618/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_618/ReverseV2:output:0Pbackward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_618/TensorArrayUnstack/TensorListFromTensorЬ
'backward_lstm_618/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_618/strided_slice_2/stack†
)backward_lstm_618/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_2/stack_1†
)backward_lstm_618/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_2/stack_2с
!backward_lstm_618/strided_slice_2StridedSlicebackward_lstm_618/transpose:y:00backward_lstm_618/strided_slice_2/stack:output:02backward_lstm_618/strided_slice_2/stack_1:output:02backward_lstm_618/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
shrink_axis_mask2#
!backward_lstm_618/strided_slice_2с
6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOp?backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype028
6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpы
'backward_lstm_618/lstm_cell_1856/MatMulMatMul*backward_lstm_618/strided_slice_2:output:0>backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'backward_lstm_618/lstm_cell_1856/MatMulч
8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOpAbackward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02:
8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpч
)backward_lstm_618/lstm_cell_1856/MatMul_1MatMul backward_lstm_618/zeros:output:0@backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)backward_lstm_618/lstm_cell_1856/MatMul_1р
$backward_lstm_618/lstm_cell_1856/addAddV21backward_lstm_618/lstm_cell_1856/MatMul:product:03backward_lstm_618/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2&
$backward_lstm_618/lstm_cell_1856/addр
7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOp@backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype029
7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpэ
(backward_lstm_618/lstm_cell_1856/BiasAddBiasAdd(backward_lstm_618/lstm_cell_1856/add:z:0?backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(backward_lstm_618/lstm_cell_1856/BiasAdd¶
0backward_lstm_618/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0backward_lstm_618/lstm_cell_1856/split/split_dim√
&backward_lstm_618/lstm_cell_1856/splitSplit9backward_lstm_618/lstm_cell_1856/split/split_dim:output:01backward_lstm_618/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2(
&backward_lstm_618/lstm_cell_1856/split¬
(backward_lstm_618/lstm_cell_1856/SigmoidSigmoid/backward_lstm_618/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(backward_lstm_618/lstm_cell_1856/Sigmoid∆
*backward_lstm_618/lstm_cell_1856/Sigmoid_1Sigmoid/backward_lstm_618/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_618/lstm_cell_1856/Sigmoid_1ў
$backward_lstm_618/lstm_cell_1856/mulMul.backward_lstm_618/lstm_cell_1856/Sigmoid_1:y:0"backward_lstm_618/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$backward_lstm_618/lstm_cell_1856/mulє
%backward_lstm_618/lstm_cell_1856/ReluRelu/backward_lstm_618/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22'
%backward_lstm_618/lstm_cell_1856/Reluм
&backward_lstm_618/lstm_cell_1856/mul_1Mul,backward_lstm_618/lstm_cell_1856/Sigmoid:y:03backward_lstm_618/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_618/lstm_cell_1856/mul_1б
&backward_lstm_618/lstm_cell_1856/add_1AddV2(backward_lstm_618/lstm_cell_1856/mul:z:0*backward_lstm_618/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_618/lstm_cell_1856/add_1∆
*backward_lstm_618/lstm_cell_1856/Sigmoid_2Sigmoid/backward_lstm_618/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_618/lstm_cell_1856/Sigmoid_2Є
'backward_lstm_618/lstm_cell_1856/Relu_1Relu*backward_lstm_618/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'backward_lstm_618/lstm_cell_1856/Relu_1р
&backward_lstm_618/lstm_cell_1856/mul_2Mul.backward_lstm_618/lstm_cell_1856/Sigmoid_2:y:05backward_lstm_618/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_618/lstm_cell_1856/mul_2≥
/backward_lstm_618/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   21
/backward_lstm_618/TensorArrayV2_1/element_shapeА
!backward_lstm_618/TensorArrayV2_1TensorListReserve8backward_lstm_618/TensorArrayV2_1/element_shape:output:0*backward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_618/TensorArrayV2_1r
backward_lstm_618/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_618/time£
*backward_lstm_618/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2,
*backward_lstm_618/while/maximum_iterationsО
$backward_lstm_618/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_618/while/loop_counter£
backward_lstm_618/whileWhile-backward_lstm_618/while/loop_counter:output:03backward_lstm_618/while/maximum_iterations:output:0backward_lstm_618/time:output:0*backward_lstm_618/TensorArrayV2_1:handle:0 backward_lstm_618/zeros:output:0"backward_lstm_618/zeros_1:output:0*backward_lstm_618/strided_slice_1:output:0Ibackward_lstm_618/TensorArrayUnstack/TensorListFromTensor:output_handle:0?backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resourceAbackward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource@backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource*
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
%backward_lstm_618_while_body_62328648*1
cond)R'
%backward_lstm_618_while_cond_62328647*K
output_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : *
parallel_iterations 2
backward_lstm_618/whileў
Bbackward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2D
Bbackward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeє
4backward_lstm_618/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_618/while:output:3Kbackward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype026
4backward_lstm_618/TensorArrayV2Stack/TensorListStack•
'backward_lstm_618/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2)
'backward_lstm_618/strided_slice_3/stack†
)backward_lstm_618/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_618/strided_slice_3/stack_1†
)backward_lstm_618/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_3/stack_2Ж
!backward_lstm_618/strided_slice_3StridedSlice=backward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_618/strided_slice_3/stack:output:02backward_lstm_618/strided_slice_3/stack_1:output:02backward_lstm_618/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2#
!backward_lstm_618/strided_slice_3Э
"backward_lstm_618/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_618/transpose_1/permц
backward_lstm_618/transpose_1	Transpose=backward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_618/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
backward_lstm_618/transpose_1К
backward_lstm_618/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_618/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisƒ
concatConcatV2)forward_lstm_618/strided_slice_3:output:0*backward_lstm_618/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp8^backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp7^backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp9^backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp^backward_lstm_618/while7^forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp6^forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp8^forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp^forward_lstm_618/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : 2r
7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp2p
6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp2t
8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp22
backward_lstm_618/whilebackward_lstm_618/while2p
6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp2n
5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp2r
7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp20
forward_lstm_618/whileforward_lstm_618/while:g c
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
я
Ќ
while_cond_62329731
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_62329731___redundant_placeholder06
2while_while_cond_62329731___redundant_placeholder16
2while_while_cond_62329731___redundant_placeholder26
2while_while_cond_62329731___redundant_placeholder3
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
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_62326954

inputs@
-lstm_cell_1855_matmul_readvariableop_resource:	»B
/lstm_cell_1855_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1855_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1855/BiasAdd/ReadVariableOpҐ$lstm_cell_1855/MatMul/ReadVariableOpҐ&lstm_cell_1855/MatMul_1/ReadVariableOpҐwhileD
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
$lstm_cell_1855/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1855_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1855/MatMul/ReadVariableOp≥
lstm_cell_1855/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/MatMulЅ
&lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1855_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1855/MatMul_1/ReadVariableOpѓ
lstm_cell_1855/MatMul_1MatMulzeros:output:0.lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/MatMul_1®
lstm_cell_1855/addAddV2lstm_cell_1855/MatMul:product:0!lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/addЇ
%lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1855_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1855/BiasAdd/ReadVariableOpµ
lstm_cell_1855/BiasAddBiasAddlstm_cell_1855/add:z:0-lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/BiasAddВ
lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1855/split/split_dimы
lstm_cell_1855/splitSplit'lstm_cell_1855/split/split_dim:output:0lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1855/splitМ
lstm_cell_1855/SigmoidSigmoidlstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/SigmoidР
lstm_cell_1855/Sigmoid_1Sigmoidlstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Sigmoid_1С
lstm_cell_1855/mulMullstm_cell_1855/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/mulГ
lstm_cell_1855/ReluRelulstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Relu§
lstm_cell_1855/mul_1Mullstm_cell_1855/Sigmoid:y:0!lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/mul_1Щ
lstm_cell_1855/add_1AddV2lstm_cell_1855/mul:z:0lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/add_1Р
lstm_cell_1855/Sigmoid_2Sigmoidlstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Sigmoid_2В
lstm_cell_1855/Relu_1Relulstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Relu_1®
lstm_cell_1855/mul_2Mullstm_cell_1855/Sigmoid_2:y:0#lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1855_matmul_readvariableop_resource/lstm_cell_1855_matmul_1_readvariableop_resource.lstm_cell_1855_biasadd_readvariableop_resource*
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
while_body_62326870*
condR
while_cond_62326869*K
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
NoOpNoOp&^lstm_cell_1855/BiasAdd/ReadVariableOp%^lstm_cell_1855/MatMul/ReadVariableOp'^lstm_cell_1855/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1855/BiasAdd/ReadVariableOp%lstm_cell_1855/BiasAdd/ReadVariableOp2L
$lstm_cell_1855/MatMul/ReadVariableOp$lstm_cell_1855/MatMul/ReadVariableOp2P
&lstm_cell_1855/MatMul_1/ReadVariableOp&lstm_cell_1855/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Іg
н
%backward_lstm_618_while_body_62328995@
<backward_lstm_618_while_backward_lstm_618_while_loop_counterF
Bbackward_lstm_618_while_backward_lstm_618_while_maximum_iterations'
#backward_lstm_618_while_placeholder)
%backward_lstm_618_while_placeholder_1)
%backward_lstm_618_while_placeholder_2)
%backward_lstm_618_while_placeholder_3)
%backward_lstm_618_while_placeholder_4?
;backward_lstm_618_while_backward_lstm_618_strided_slice_1_0{
wbackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_618_while_less_backward_lstm_618_sub_1_0Z
Gbackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0:	»\
Ibackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0:	2»W
Hbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0:	»$
 backward_lstm_618_while_identity&
"backward_lstm_618_while_identity_1&
"backward_lstm_618_while_identity_2&
"backward_lstm_618_while_identity_3&
"backward_lstm_618_while_identity_4&
"backward_lstm_618_while_identity_5&
"backward_lstm_618_while_identity_6=
9backward_lstm_618_while_backward_lstm_618_strided_slice_1y
ubackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_618_while_less_backward_lstm_618_sub_1X
Ebackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource:	»Z
Gbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource:	2»U
Fbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource:	»ИҐ=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpҐ<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOpҐ>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpз
Ibackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2K
Ibackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeњ
;backward_lstm_618/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_618_while_placeholderRbackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02=
;backward_lstm_618/while/TensorArrayV2Read/TensorListGetItemѕ
backward_lstm_618/while/LessLess6backward_lstm_618_while_less_backward_lstm_618_sub_1_0#backward_lstm_618_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_618/while/LessЕ
<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOpGbackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02>
<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp•
-backward_lstm_618/while/lstm_cell_1856/MatMulMatMulBbackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem:item:0Dbackward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-backward_lstm_618/while/lstm_cell_1856/MatMulЛ
>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOpIbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02@
>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpО
/backward_lstm_618/while/lstm_cell_1856/MatMul_1MatMul%backward_lstm_618_while_placeholder_3Fbackward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»21
/backward_lstm_618/while/lstm_cell_1856/MatMul_1И
*backward_lstm_618/while/lstm_cell_1856/addAddV27backward_lstm_618/while/lstm_cell_1856/MatMul:product:09backward_lstm_618/while/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2,
*backward_lstm_618/while/lstm_cell_1856/addД
=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOpHbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02?
=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpХ
.backward_lstm_618/while/lstm_cell_1856/BiasAddBiasAdd.backward_lstm_618/while/lstm_cell_1856/add:z:0Ebackward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.backward_lstm_618/while/lstm_cell_1856/BiasAdd≤
6backward_lstm_618/while/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6backward_lstm_618/while/lstm_cell_1856/split/split_dimџ
,backward_lstm_618/while/lstm_cell_1856/splitSplit?backward_lstm_618/while/lstm_cell_1856/split/split_dim:output:07backward_lstm_618/while/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2.
,backward_lstm_618/while/lstm_cell_1856/split‘
.backward_lstm_618/while/lstm_cell_1856/SigmoidSigmoid5backward_lstm_618/while/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€220
.backward_lstm_618/while/lstm_cell_1856/SigmoidЎ
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_1Sigmoid5backward_lstm_618/while/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_1о
*backward_lstm_618/while/lstm_cell_1856/mulMul4backward_lstm_618/while/lstm_cell_1856/Sigmoid_1:y:0%backward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_618/while/lstm_cell_1856/mulЋ
+backward_lstm_618/while/lstm_cell_1856/ReluRelu5backward_lstm_618/while/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22-
+backward_lstm_618/while/lstm_cell_1856/ReluД
,backward_lstm_618/while/lstm_cell_1856/mul_1Mul2backward_lstm_618/while/lstm_cell_1856/Sigmoid:y:09backward_lstm_618/while/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_618/while/lstm_cell_1856/mul_1щ
,backward_lstm_618/while/lstm_cell_1856/add_1AddV2.backward_lstm_618/while/lstm_cell_1856/mul:z:00backward_lstm_618/while/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_618/while/lstm_cell_1856/add_1Ў
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_2Sigmoid5backward_lstm_618/while/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_2 
-backward_lstm_618/while/lstm_cell_1856/Relu_1Relu0backward_lstm_618/while/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-backward_lstm_618/while/lstm_cell_1856/Relu_1И
,backward_lstm_618/while/lstm_cell_1856/mul_2Mul4backward_lstm_618/while/lstm_cell_1856/Sigmoid_2:y:0;backward_lstm_618/while/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_618/while/lstm_cell_1856/mul_2ч
backward_lstm_618/while/SelectSelect backward_lstm_618/while/Less:z:00backward_lstm_618/while/lstm_cell_1856/mul_2:z:0%backward_lstm_618_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22 
backward_lstm_618/while/Selectы
 backward_lstm_618/while/Select_1Select backward_lstm_618/while/Less:z:00backward_lstm_618/while/lstm_cell_1856/mul_2:z:0%backward_lstm_618_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_618/while/Select_1ы
 backward_lstm_618/while/Select_2Select backward_lstm_618/while/Less:z:00backward_lstm_618/while/lstm_cell_1856/add_1:z:0%backward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_618/while/Select_2≥
<backward_lstm_618/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_618_while_placeholder_1#backward_lstm_618_while_placeholder'backward_lstm_618/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_618/while/TensorArrayV2Write/TensorListSetItemА
backward_lstm_618/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_618/while/add/y±
backward_lstm_618/while/addAddV2#backward_lstm_618_while_placeholder&backward_lstm_618/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/while/addД
backward_lstm_618/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_618/while/add_1/y–
backward_lstm_618/while/add_1AddV2<backward_lstm_618_while_backward_lstm_618_while_loop_counter(backward_lstm_618/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/while/add_1≥
 backward_lstm_618/while/IdentityIdentity!backward_lstm_618/while/add_1:z:0^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_618/while/IdentityЎ
"backward_lstm_618/while/Identity_1IdentityBbackward_lstm_618_while_backward_lstm_618_while_maximum_iterations^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_618/while/Identity_1µ
"backward_lstm_618/while/Identity_2Identitybackward_lstm_618/while/add:z:0^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_618/while/Identity_2в
"backward_lstm_618/while/Identity_3IdentityLbackward_lstm_618/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_618/while/Identity_3ќ
"backward_lstm_618/while/Identity_4Identity'backward_lstm_618/while/Select:output:0^backward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_618/while/Identity_4–
"backward_lstm_618/while/Identity_5Identity)backward_lstm_618/while/Select_1:output:0^backward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_618/while/Identity_5–
"backward_lstm_618/while/Identity_6Identity)backward_lstm_618/while/Select_2:output:0^backward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_618/while/Identity_6Њ
backward_lstm_618/while/NoOpNoOp>^backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp=^backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp?^backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_618/while/NoOp"x
9backward_lstm_618_while_backward_lstm_618_strided_slice_1;backward_lstm_618_while_backward_lstm_618_strided_slice_1_0"M
 backward_lstm_618_while_identity)backward_lstm_618/while/Identity:output:0"Q
"backward_lstm_618_while_identity_1+backward_lstm_618/while/Identity_1:output:0"Q
"backward_lstm_618_while_identity_2+backward_lstm_618/while/Identity_2:output:0"Q
"backward_lstm_618_while_identity_3+backward_lstm_618/while/Identity_3:output:0"Q
"backward_lstm_618_while_identity_4+backward_lstm_618/while/Identity_4:output:0"Q
"backward_lstm_618_while_identity_5+backward_lstm_618/while/Identity_5:output:0"Q
"backward_lstm_618_while_identity_6+backward_lstm_618/while/Identity_6:output:0"n
4backward_lstm_618_while_less_backward_lstm_618_sub_16backward_lstm_618_while_less_backward_lstm_618_sub_1_0"Т
Fbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resourceHbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0"Ф
Gbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resourceIbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0"Р
Ebackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resourceGbackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0"р
ubackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2~
=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp2|
<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp2А
>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_62325372

inputs*
lstm_cell_1855_62325290:	»*
lstm_cell_1855_62325292:	2»&
lstm_cell_1855_62325294:	»
identityИҐ&lstm_cell_1855/StatefulPartitionedCallҐwhileD
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
&lstm_cell_1855/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1855_62325290lstm_cell_1855_62325292lstm_cell_1855_62325294*
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
L__inference_lstm_cell_1855_layer_call_and_return_conditional_losses_623252252(
&lstm_cell_1855/StatefulPartitionedCallП
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1855_62325290lstm_cell_1855_62325292lstm_cell_1855_62325294*
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
while_body_62325303*
condR
while_cond_62325302*K
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
NoOpNoOp'^lstm_cell_1855/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2P
&lstm_cell_1855/StatefulPartitionedCall&lstm_cell_1855/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
я
Ќ
while_cond_62329580
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_62329580___redundant_placeholder06
2while_while_cond_62329580___redundant_placeholder16
2while_while_cond_62329580___redundant_placeholder26
2while_while_cond_62329580___redundant_placeholder3
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
$forward_lstm_618_while_body_62328499>
:forward_lstm_618_while_forward_lstm_618_while_loop_counterD
@forward_lstm_618_while_forward_lstm_618_while_maximum_iterations&
"forward_lstm_618_while_placeholder(
$forward_lstm_618_while_placeholder_1(
$forward_lstm_618_while_placeholder_2(
$forward_lstm_618_while_placeholder_3=
9forward_lstm_618_while_forward_lstm_618_strided_slice_1_0y
uforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0Y
Fforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0:	»[
Hforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0:	2»V
Gforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0:	»#
forward_lstm_618_while_identity%
!forward_lstm_618_while_identity_1%
!forward_lstm_618_while_identity_2%
!forward_lstm_618_while_identity_3%
!forward_lstm_618_while_identity_4%
!forward_lstm_618_while_identity_5;
7forward_lstm_618_while_forward_lstm_618_strided_slice_1w
sforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensorW
Dforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource:	»Y
Fforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource:	2»T
Eforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource:	»ИҐ<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpҐ;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOpҐ=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpе
Hforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2J
Hforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shape¬
:forward_lstm_618/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_618_while_placeholderQforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02<
:forward_lstm_618/while/TensorArrayV2Read/TensorListGetItemВ
;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOpFforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02=
;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp°
,forward_lstm_618/while/lstm_cell_1855/MatMulMatMulAforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem:item:0Cforward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2.
,forward_lstm_618/while/lstm_cell_1855/MatMulИ
=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOpHforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02?
=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpК
.forward_lstm_618/while/lstm_cell_1855/MatMul_1MatMul$forward_lstm_618_while_placeholder_2Eforward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.forward_lstm_618/while/lstm_cell_1855/MatMul_1Д
)forward_lstm_618/while/lstm_cell_1855/addAddV26forward_lstm_618/while/lstm_cell_1855/MatMul:product:08forward_lstm_618/while/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)forward_lstm_618/while/lstm_cell_1855/addБ
<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOpGforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02>
<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpС
-forward_lstm_618/while/lstm_cell_1855/BiasAddBiasAdd-forward_lstm_618/while/lstm_cell_1855/add:z:0Dforward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-forward_lstm_618/while/lstm_cell_1855/BiasAdd∞
5forward_lstm_618/while/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5forward_lstm_618/while/lstm_cell_1855/split/split_dim„
+forward_lstm_618/while/lstm_cell_1855/splitSplit>forward_lstm_618/while/lstm_cell_1855/split/split_dim:output:06forward_lstm_618/while/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2-
+forward_lstm_618/while/lstm_cell_1855/split—
-forward_lstm_618/while/lstm_cell_1855/SigmoidSigmoid4forward_lstm_618/while/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-forward_lstm_618/while/lstm_cell_1855/Sigmoid’
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_1Sigmoid4forward_lstm_618/while/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_1к
)forward_lstm_618/while/lstm_cell_1855/mulMul3forward_lstm_618/while/lstm_cell_1855/Sigmoid_1:y:0$forward_lstm_618_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_618/while/lstm_cell_1855/mul»
*forward_lstm_618/while/lstm_cell_1855/ReluRelu4forward_lstm_618/while/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22,
*forward_lstm_618/while/lstm_cell_1855/ReluА
+forward_lstm_618/while/lstm_cell_1855/mul_1Mul1forward_lstm_618/while/lstm_cell_1855/Sigmoid:y:08forward_lstm_618/while/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_618/while/lstm_cell_1855/mul_1х
+forward_lstm_618/while/lstm_cell_1855/add_1AddV2-forward_lstm_618/while/lstm_cell_1855/mul:z:0/forward_lstm_618/while/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_618/while/lstm_cell_1855/add_1’
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_2Sigmoid4forward_lstm_618/while/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_2«
,forward_lstm_618/while/lstm_cell_1855/Relu_1Relu/forward_lstm_618/while/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,forward_lstm_618/while/lstm_cell_1855/Relu_1Д
+forward_lstm_618/while/lstm_cell_1855/mul_2Mul3forward_lstm_618/while/lstm_cell_1855/Sigmoid_2:y:0:forward_lstm_618/while/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_618/while/lstm_cell_1855/mul_2Ј
;forward_lstm_618/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_618_while_placeholder_1"forward_lstm_618_while_placeholder/forward_lstm_618/while/lstm_cell_1855/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_618/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_618/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_618/while/add/y≠
forward_lstm_618/while/addAddV2"forward_lstm_618_while_placeholder%forward_lstm_618/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/while/addВ
forward_lstm_618/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_618/while/add_1/yЋ
forward_lstm_618/while/add_1AddV2:forward_lstm_618_while_forward_lstm_618_while_loop_counter'forward_lstm_618/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/while/add_1ѓ
forward_lstm_618/while/IdentityIdentity forward_lstm_618/while/add_1:z:0^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_618/while/Identity”
!forward_lstm_618/while/Identity_1Identity@forward_lstm_618_while_forward_lstm_618_while_maximum_iterations^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_618/while/Identity_1±
!forward_lstm_618/while/Identity_2Identityforward_lstm_618/while/add:z:0^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_618/while/Identity_2ё
!forward_lstm_618/while/Identity_3IdentityKforward_lstm_618/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_618/while/Identity_3”
!forward_lstm_618/while/Identity_4Identity/forward_lstm_618/while/lstm_cell_1855/mul_2:z:0^forward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_618/while/Identity_4”
!forward_lstm_618/while/Identity_5Identity/forward_lstm_618/while/lstm_cell_1855/add_1:z:0^forward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_618/while/Identity_5є
forward_lstm_618/while/NoOpNoOp=^forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp<^forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp>^forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_618/while/NoOp"t
7forward_lstm_618_while_forward_lstm_618_strided_slice_19forward_lstm_618_while_forward_lstm_618_strided_slice_1_0"K
forward_lstm_618_while_identity(forward_lstm_618/while/Identity:output:0"O
!forward_lstm_618_while_identity_1*forward_lstm_618/while/Identity_1:output:0"O
!forward_lstm_618_while_identity_2*forward_lstm_618/while/Identity_2:output:0"O
!forward_lstm_618_while_identity_3*forward_lstm_618/while/Identity_3:output:0"O
!forward_lstm_618_while_identity_4*forward_lstm_618/while/Identity_4:output:0"O
!forward_lstm_618_while_identity_5*forward_lstm_618/while/Identity_5:output:0"Р
Eforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resourceGforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0"Т
Fforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resourceHforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0"О
Dforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resourceFforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0"м
sforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensoruforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2|
<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp2z
;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp2~
=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
%backward_lstm_618_while_body_62327343@
<backward_lstm_618_while_backward_lstm_618_while_loop_counterF
Bbackward_lstm_618_while_backward_lstm_618_while_maximum_iterations'
#backward_lstm_618_while_placeholder)
%backward_lstm_618_while_placeholder_1)
%backward_lstm_618_while_placeholder_2)
%backward_lstm_618_while_placeholder_3)
%backward_lstm_618_while_placeholder_4?
;backward_lstm_618_while_backward_lstm_618_strided_slice_1_0{
wbackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_618_while_less_backward_lstm_618_sub_1_0Z
Gbackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0:	»\
Ibackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0:	2»W
Hbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0:	»$
 backward_lstm_618_while_identity&
"backward_lstm_618_while_identity_1&
"backward_lstm_618_while_identity_2&
"backward_lstm_618_while_identity_3&
"backward_lstm_618_while_identity_4&
"backward_lstm_618_while_identity_5&
"backward_lstm_618_while_identity_6=
9backward_lstm_618_while_backward_lstm_618_strided_slice_1y
ubackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_618_while_less_backward_lstm_618_sub_1X
Ebackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource:	»Z
Gbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource:	2»U
Fbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource:	»ИҐ=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpҐ<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOpҐ>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpз
Ibackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2K
Ibackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeњ
;backward_lstm_618/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_618_while_placeholderRbackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02=
;backward_lstm_618/while/TensorArrayV2Read/TensorListGetItemѕ
backward_lstm_618/while/LessLess6backward_lstm_618_while_less_backward_lstm_618_sub_1_0#backward_lstm_618_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_618/while/LessЕ
<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOpGbackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02>
<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp•
-backward_lstm_618/while/lstm_cell_1856/MatMulMatMulBbackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem:item:0Dbackward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-backward_lstm_618/while/lstm_cell_1856/MatMulЛ
>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOpIbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02@
>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpО
/backward_lstm_618/while/lstm_cell_1856/MatMul_1MatMul%backward_lstm_618_while_placeholder_3Fbackward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»21
/backward_lstm_618/while/lstm_cell_1856/MatMul_1И
*backward_lstm_618/while/lstm_cell_1856/addAddV27backward_lstm_618/while/lstm_cell_1856/MatMul:product:09backward_lstm_618/while/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2,
*backward_lstm_618/while/lstm_cell_1856/addД
=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOpHbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02?
=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpХ
.backward_lstm_618/while/lstm_cell_1856/BiasAddBiasAdd.backward_lstm_618/while/lstm_cell_1856/add:z:0Ebackward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.backward_lstm_618/while/lstm_cell_1856/BiasAdd≤
6backward_lstm_618/while/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6backward_lstm_618/while/lstm_cell_1856/split/split_dimџ
,backward_lstm_618/while/lstm_cell_1856/splitSplit?backward_lstm_618/while/lstm_cell_1856/split/split_dim:output:07backward_lstm_618/while/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2.
,backward_lstm_618/while/lstm_cell_1856/split‘
.backward_lstm_618/while/lstm_cell_1856/SigmoidSigmoid5backward_lstm_618/while/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€220
.backward_lstm_618/while/lstm_cell_1856/SigmoidЎ
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_1Sigmoid5backward_lstm_618/while/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_1о
*backward_lstm_618/while/lstm_cell_1856/mulMul4backward_lstm_618/while/lstm_cell_1856/Sigmoid_1:y:0%backward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_618/while/lstm_cell_1856/mulЋ
+backward_lstm_618/while/lstm_cell_1856/ReluRelu5backward_lstm_618/while/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22-
+backward_lstm_618/while/lstm_cell_1856/ReluД
,backward_lstm_618/while/lstm_cell_1856/mul_1Mul2backward_lstm_618/while/lstm_cell_1856/Sigmoid:y:09backward_lstm_618/while/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_618/while/lstm_cell_1856/mul_1щ
,backward_lstm_618/while/lstm_cell_1856/add_1AddV2.backward_lstm_618/while/lstm_cell_1856/mul:z:00backward_lstm_618/while/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_618/while/lstm_cell_1856/add_1Ў
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_2Sigmoid5backward_lstm_618/while/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_2 
-backward_lstm_618/while/lstm_cell_1856/Relu_1Relu0backward_lstm_618/while/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-backward_lstm_618/while/lstm_cell_1856/Relu_1И
,backward_lstm_618/while/lstm_cell_1856/mul_2Mul4backward_lstm_618/while/lstm_cell_1856/Sigmoid_2:y:0;backward_lstm_618/while/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_618/while/lstm_cell_1856/mul_2ч
backward_lstm_618/while/SelectSelect backward_lstm_618/while/Less:z:00backward_lstm_618/while/lstm_cell_1856/mul_2:z:0%backward_lstm_618_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22 
backward_lstm_618/while/Selectы
 backward_lstm_618/while/Select_1Select backward_lstm_618/while/Less:z:00backward_lstm_618/while/lstm_cell_1856/mul_2:z:0%backward_lstm_618_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_618/while/Select_1ы
 backward_lstm_618/while/Select_2Select backward_lstm_618/while/Less:z:00backward_lstm_618/while/lstm_cell_1856/add_1:z:0%backward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_618/while/Select_2≥
<backward_lstm_618/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_618_while_placeholder_1#backward_lstm_618_while_placeholder'backward_lstm_618/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_618/while/TensorArrayV2Write/TensorListSetItemА
backward_lstm_618/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_618/while/add/y±
backward_lstm_618/while/addAddV2#backward_lstm_618_while_placeholder&backward_lstm_618/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/while/addД
backward_lstm_618/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_618/while/add_1/y–
backward_lstm_618/while/add_1AddV2<backward_lstm_618_while_backward_lstm_618_while_loop_counter(backward_lstm_618/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/while/add_1≥
 backward_lstm_618/while/IdentityIdentity!backward_lstm_618/while/add_1:z:0^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_618/while/IdentityЎ
"backward_lstm_618/while/Identity_1IdentityBbackward_lstm_618_while_backward_lstm_618_while_maximum_iterations^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_618/while/Identity_1µ
"backward_lstm_618/while/Identity_2Identitybackward_lstm_618/while/add:z:0^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_618/while/Identity_2в
"backward_lstm_618/while/Identity_3IdentityLbackward_lstm_618/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_618/while/Identity_3ќ
"backward_lstm_618/while/Identity_4Identity'backward_lstm_618/while/Select:output:0^backward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_618/while/Identity_4–
"backward_lstm_618/while/Identity_5Identity)backward_lstm_618/while/Select_1:output:0^backward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_618/while/Identity_5–
"backward_lstm_618/while/Identity_6Identity)backward_lstm_618/while/Select_2:output:0^backward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_618/while/Identity_6Њ
backward_lstm_618/while/NoOpNoOp>^backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp=^backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp?^backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_618/while/NoOp"x
9backward_lstm_618_while_backward_lstm_618_strided_slice_1;backward_lstm_618_while_backward_lstm_618_strided_slice_1_0"M
 backward_lstm_618_while_identity)backward_lstm_618/while/Identity:output:0"Q
"backward_lstm_618_while_identity_1+backward_lstm_618/while/Identity_1:output:0"Q
"backward_lstm_618_while_identity_2+backward_lstm_618/while/Identity_2:output:0"Q
"backward_lstm_618_while_identity_3+backward_lstm_618/while/Identity_3:output:0"Q
"backward_lstm_618_while_identity_4+backward_lstm_618/while/Identity_4:output:0"Q
"backward_lstm_618_while_identity_5+backward_lstm_618/while/Identity_5:output:0"Q
"backward_lstm_618_while_identity_6+backward_lstm_618/while/Identity_6:output:0"n
4backward_lstm_618_while_less_backward_lstm_618_sub_16backward_lstm_618_while_less_backward_lstm_618_sub_1_0"Т
Fbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resourceHbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0"Ф
Gbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resourceIbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0"Р
Ebackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resourceGbackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0"р
ubackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2~
=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp2|
<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp2А
>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
∆@
д
while_body_62329581
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1855_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1855_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1855_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1855_matmul_readvariableop_resource:	»H
5while_lstm_cell_1855_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1855_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1855/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1855/MatMul/ReadVariableOpҐ,while/lstm_cell_1855/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1855_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1855/MatMul/ReadVariableOpЁ
while/lstm_cell_1855/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/MatMul’
,while/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1855_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1855/MatMul_1/ReadVariableOp∆
while/lstm_cell_1855/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/MatMul_1ј
while/lstm_cell_1855/addAddV2%while/lstm_cell_1855/MatMul:product:0'while/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/addќ
+while/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1855_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1855/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1855/BiasAddBiasAddwhile/lstm_cell_1855/add:z:03while/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/BiasAddО
$while/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1855/split/split_dimУ
while/lstm_cell_1855/splitSplit-while/lstm_cell_1855/split/split_dim:output:0%while/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1855/splitЮ
while/lstm_cell_1855/SigmoidSigmoid#while/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/SigmoidҐ
while/lstm_cell_1855/Sigmoid_1Sigmoid#while/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1855/Sigmoid_1¶
while/lstm_cell_1855/mulMul"while/lstm_cell_1855/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/mulХ
while/lstm_cell_1855/ReluRelu#while/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/ReluЉ
while/lstm_cell_1855/mul_1Mul while/lstm_cell_1855/Sigmoid:y:0'while/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/mul_1±
while/lstm_cell_1855/add_1AddV2while/lstm_cell_1855/mul:z:0while/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/add_1Ґ
while/lstm_cell_1855/Sigmoid_2Sigmoid#while/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1855/Sigmoid_2Ф
while/lstm_cell_1855/Relu_1Reluwhile/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/Relu_1ј
while/lstm_cell_1855/mul_2Mul"while/lstm_cell_1855/Sigmoid_2:y:0)while/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1855/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1855/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1855/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1855/BiasAdd/ReadVariableOp+^while/lstm_cell_1855/MatMul/ReadVariableOp-^while/lstm_cell_1855/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1855_biasadd_readvariableop_resource6while_lstm_cell_1855_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1855_matmul_1_readvariableop_resource7while_lstm_cell_1855_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1855_matmul_readvariableop_resource5while_lstm_cell_1855_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1855/BiasAdd/ReadVariableOp+while/lstm_cell_1855/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1855/MatMul/ReadVariableOp*while/lstm_cell_1855/MatMul/ReadVariableOp2\
,while/lstm_cell_1855/MatMul_1/ReadVariableOp,while/lstm_cell_1855/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_62326006

inputs*
lstm_cell_1856_62325924:	»*
lstm_cell_1856_62325926:	2»&
lstm_cell_1856_62325928:	»
identityИҐ&lstm_cell_1856/StatefulPartitionedCallҐwhileD
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
&lstm_cell_1856/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1856_62325924lstm_cell_1856_62325926lstm_cell_1856_62325928*
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
L__inference_lstm_cell_1856_layer_call_and_return_conditional_losses_623258572(
&lstm_cell_1856/StatefulPartitionedCallП
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1856_62325924lstm_cell_1856_62325926lstm_cell_1856_62325928*
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
while_body_62325937*
condR
while_cond_62325936*K
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
NoOpNoOp'^lstm_cell_1856/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2P
&lstm_cell_1856/StatefulPartitionedCall&lstm_cell_1856/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
…
•
$forward_lstm_618_while_cond_62328815>
:forward_lstm_618_while_forward_lstm_618_while_loop_counterD
@forward_lstm_618_while_forward_lstm_618_while_maximum_iterations&
"forward_lstm_618_while_placeholder(
$forward_lstm_618_while_placeholder_1(
$forward_lstm_618_while_placeholder_2(
$forward_lstm_618_while_placeholder_3(
$forward_lstm_618_while_placeholder_4@
<forward_lstm_618_while_less_forward_lstm_618_strided_slice_1X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62328815___redundant_placeholder0X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62328815___redundant_placeholder1X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62328815___redundant_placeholder2X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62328815___redundant_placeholder3X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62328815___redundant_placeholder4#
forward_lstm_618_while_identity
≈
forward_lstm_618/while/LessLess"forward_lstm_618_while_placeholder<forward_lstm_618_while_less_forward_lstm_618_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_618/while/LessР
forward_lstm_618/while/IdentityIdentityforward_lstm_618/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_618/while/Identity"K
forward_lstm_618_while_identity(forward_lstm_618/while/Identity:output:0*(
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
м]
≤
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_62330118

inputs@
-lstm_cell_1855_matmul_readvariableop_resource:	»B
/lstm_cell_1855_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1855_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1855/BiasAdd/ReadVariableOpҐ$lstm_cell_1855/MatMul/ReadVariableOpҐ&lstm_cell_1855/MatMul_1/ReadVariableOpҐwhileD
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
$lstm_cell_1855/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1855_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1855/MatMul/ReadVariableOp≥
lstm_cell_1855/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/MatMulЅ
&lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1855_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1855/MatMul_1/ReadVariableOpѓ
lstm_cell_1855/MatMul_1MatMulzeros:output:0.lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/MatMul_1®
lstm_cell_1855/addAddV2lstm_cell_1855/MatMul:product:0!lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/addЇ
%lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1855_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1855/BiasAdd/ReadVariableOpµ
lstm_cell_1855/BiasAddBiasAddlstm_cell_1855/add:z:0-lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/BiasAddВ
lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1855/split/split_dimы
lstm_cell_1855/splitSplit'lstm_cell_1855/split/split_dim:output:0lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1855/splitМ
lstm_cell_1855/SigmoidSigmoidlstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/SigmoidР
lstm_cell_1855/Sigmoid_1Sigmoidlstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Sigmoid_1С
lstm_cell_1855/mulMullstm_cell_1855/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/mulГ
lstm_cell_1855/ReluRelulstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Relu§
lstm_cell_1855/mul_1Mullstm_cell_1855/Sigmoid:y:0!lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/mul_1Щ
lstm_cell_1855/add_1AddV2lstm_cell_1855/mul:z:0lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/add_1Р
lstm_cell_1855/Sigmoid_2Sigmoidlstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Sigmoid_2В
lstm_cell_1855/Relu_1Relulstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Relu_1®
lstm_cell_1855/mul_2Mullstm_cell_1855/Sigmoid_2:y:0#lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1855_matmul_readvariableop_resource/lstm_cell_1855_matmul_1_readvariableop_resource.lstm_cell_1855_biasadd_readvariableop_resource*
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
while_body_62330034*
condR
while_cond_62330033*K
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
NoOpNoOp&^lstm_cell_1855/BiasAdd/ReadVariableOp%^lstm_cell_1855/MatMul/ReadVariableOp'^lstm_cell_1855/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1855/BiasAdd/ReadVariableOp%lstm_cell_1855/BiasAdd/ReadVariableOp2L
$lstm_cell_1855/MatMul/ReadVariableOp$lstm_cell_1855/MatMul/ReadVariableOp2P
&lstm_cell_1855/MatMul_1/ReadVariableOp&lstm_cell_1855/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ѕ@
д
while_body_62330034
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1855_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1855_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1855_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1855_matmul_readvariableop_resource:	»H
5while_lstm_cell_1855_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1855_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1855/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1855/MatMul/ReadVariableOpҐ,while/lstm_cell_1855/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1855_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1855/MatMul/ReadVariableOpЁ
while/lstm_cell_1855/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/MatMul’
,while/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1855_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1855/MatMul_1/ReadVariableOp∆
while/lstm_cell_1855/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/MatMul_1ј
while/lstm_cell_1855/addAddV2%while/lstm_cell_1855/MatMul:product:0'while/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/addќ
+while/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1855_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1855/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1855/BiasAddBiasAddwhile/lstm_cell_1855/add:z:03while/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/BiasAddО
$while/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1855/split/split_dimУ
while/lstm_cell_1855/splitSplit-while/lstm_cell_1855/split/split_dim:output:0%while/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1855/splitЮ
while/lstm_cell_1855/SigmoidSigmoid#while/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/SigmoidҐ
while/lstm_cell_1855/Sigmoid_1Sigmoid#while/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1855/Sigmoid_1¶
while/lstm_cell_1855/mulMul"while/lstm_cell_1855/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/mulХ
while/lstm_cell_1855/ReluRelu#while/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/ReluЉ
while/lstm_cell_1855/mul_1Mul while/lstm_cell_1855/Sigmoid:y:0'while/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/mul_1±
while/lstm_cell_1855/add_1AddV2while/lstm_cell_1855/mul:z:0while/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/add_1Ґ
while/lstm_cell_1855/Sigmoid_2Sigmoid#while/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1855/Sigmoid_2Ф
while/lstm_cell_1855/Relu_1Reluwhile/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/Relu_1ј
while/lstm_cell_1855/mul_2Mul"while/lstm_cell_1855/Sigmoid_2:y:0)while/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1855/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1855/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1855/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1855/BiasAdd/ReadVariableOp+^while/lstm_cell_1855/MatMul/ReadVariableOp-^while/lstm_cell_1855/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1855_biasadd_readvariableop_resource6while_lstm_cell_1855_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1855_matmul_1_readvariableop_resource7while_lstm_cell_1855_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1855_matmul_readvariableop_resource5while_lstm_cell_1855_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1855/BiasAdd/ReadVariableOp+while/lstm_cell_1855/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1855/MatMul/ReadVariableOp*while/lstm_cell_1855/MatMul/ReadVariableOp2\
,while/lstm_cell_1855/MatMul_1/ReadVariableOp,while/lstm_cell_1855/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_62325162

inputs*
lstm_cell_1855_62325080:	»*
lstm_cell_1855_62325082:	2»&
lstm_cell_1855_62325084:	»
identityИҐ&lstm_cell_1855/StatefulPartitionedCallҐwhileD
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
&lstm_cell_1855/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1855_62325080lstm_cell_1855_62325082lstm_cell_1855_62325084*
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
L__inference_lstm_cell_1855_layer_call_and_return_conditional_losses_623250792(
&lstm_cell_1855/StatefulPartitionedCallП
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1855_62325080lstm_cell_1855_62325082lstm_cell_1855_62325084*
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
while_body_62325093*
condR
while_cond_62325092*K
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
NoOpNoOp'^lstm_cell_1855/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2P
&lstm_cell_1855/StatefulPartitionedCall&lstm_cell_1855/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
€
К
L__inference_lstm_cell_1855_layer_call_and_return_conditional_losses_62330840

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
∞Њ
ы
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_62329092

inputs
inputs_1	Q
>forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource:	»S
@forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource:	2»N
?forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource:	»R
?backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resource:	»T
Abackward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource:	2»O
@backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource:	»
identityИҐ7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpҐ6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpҐ8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpҐbackward_lstm_618/whileҐ6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpҐ5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpҐ7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpҐforward_lstm_618/whileЧ
%forward_lstm_618/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_618/RaggedToTensor/zerosЩ
%forward_lstm_618/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2'
%forward_lstm_618/RaggedToTensor/ConstЩ
4forward_lstm_618/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_618/RaggedToTensor/Const:output:0inputs.forward_lstm_618/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_618/RaggedToTensor/RaggedTensorToTensorƒ
;forward_lstm_618/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack»
=forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1»
=forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2©
5forward_lstm_618/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_618/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask27
5forward_lstm_618/RaggedNestedRowLengths/strided_slice»
=forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack’
?forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2A
?forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1ћ
?forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2µ
7forward_lstm_618/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask29
7forward_lstm_618/RaggedNestedRowLengths/strided_slice_1С
+forward_lstm_618/RaggedNestedRowLengths/subSub>forward_lstm_618/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_618/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2-
+forward_lstm_618/RaggedNestedRowLengths/sub§
forward_lstm_618/CastCast/forward_lstm_618/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
forward_lstm_618/CastЭ
forward_lstm_618/ShapeShape=forward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_618/ShapeЦ
$forward_lstm_618/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_618/strided_slice/stackЪ
&forward_lstm_618/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_618/strided_slice/stack_1Ъ
&forward_lstm_618/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_618/strided_slice/stack_2»
forward_lstm_618/strided_sliceStridedSliceforward_lstm_618/Shape:output:0-forward_lstm_618/strided_slice/stack:output:0/forward_lstm_618/strided_slice/stack_1:output:0/forward_lstm_618/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_618/strided_slice~
forward_lstm_618/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_618/zeros/mul/y∞
forward_lstm_618/zeros/mulMul'forward_lstm_618/strided_slice:output:0%forward_lstm_618/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros/mulБ
forward_lstm_618/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
forward_lstm_618/zeros/Less/yЂ
forward_lstm_618/zeros/LessLessforward_lstm_618/zeros/mul:z:0&forward_lstm_618/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros/LessД
forward_lstm_618/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_618/zeros/packed/1«
forward_lstm_618/zeros/packedPack'forward_lstm_618/strided_slice:output:0(forward_lstm_618/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_618/zeros/packedЕ
forward_lstm_618/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_618/zeros/Constє
forward_lstm_618/zerosFill&forward_lstm_618/zeros/packed:output:0%forward_lstm_618/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/zerosВ
forward_lstm_618/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_618/zeros_1/mul/yґ
forward_lstm_618/zeros_1/mulMul'forward_lstm_618/strided_slice:output:0'forward_lstm_618/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros_1/mulЕ
forward_lstm_618/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2!
forward_lstm_618/zeros_1/Less/y≥
forward_lstm_618/zeros_1/LessLess forward_lstm_618/zeros_1/mul:z:0(forward_lstm_618/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros_1/LessИ
!forward_lstm_618/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_618/zeros_1/packed/1Ќ
forward_lstm_618/zeros_1/packedPack'forward_lstm_618/strided_slice:output:0*forward_lstm_618/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_618/zeros_1/packedЙ
forward_lstm_618/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_618/zeros_1/ConstЅ
forward_lstm_618/zeros_1Fill(forward_lstm_618/zeros_1/packed:output:0'forward_lstm_618/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/zeros_1Ч
forward_lstm_618/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_618/transpose/permн
forward_lstm_618/transpose	Transpose=forward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_618/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
forward_lstm_618/transposeВ
forward_lstm_618/Shape_1Shapeforward_lstm_618/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_618/Shape_1Ъ
&forward_lstm_618/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_618/strided_slice_1/stackЮ
(forward_lstm_618/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_1/stack_1Ю
(forward_lstm_618/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_1/stack_2‘
 forward_lstm_618/strided_slice_1StridedSlice!forward_lstm_618/Shape_1:output:0/forward_lstm_618/strided_slice_1/stack:output:01forward_lstm_618/strided_slice_1/stack_1:output:01forward_lstm_618/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_618/strided_slice_1І
,forward_lstm_618/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,forward_lstm_618/TensorArrayV2/element_shapeц
forward_lstm_618/TensorArrayV2TensorListReserve5forward_lstm_618/TensorArrayV2/element_shape:output:0)forward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_618/TensorArrayV2б
Fforward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2H
Fforward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeЉ
8forward_lstm_618/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_618/transpose:y:0Oforward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_618/TensorArrayUnstack/TensorListFromTensorЪ
&forward_lstm_618/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_618/strided_slice_2/stackЮ
(forward_lstm_618/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_2/stack_1Ю
(forward_lstm_618/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_2/stack_2в
 forward_lstm_618/strided_slice_2StridedSliceforward_lstm_618/transpose:y:0/forward_lstm_618/strided_slice_2/stack:output:01forward_lstm_618/strided_slice_2/stack_1:output:01forward_lstm_618/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2"
 forward_lstm_618/strided_slice_2о
5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOp>forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype027
5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpч
&forward_lstm_618/lstm_cell_1855/MatMulMatMul)forward_lstm_618/strided_slice_2:output:0=forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2(
&forward_lstm_618/lstm_cell_1855/MatMulф
7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOp@forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype029
7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpу
(forward_lstm_618/lstm_cell_1855/MatMul_1MatMulforward_lstm_618/zeros:output:0?forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(forward_lstm_618/lstm_cell_1855/MatMul_1м
#forward_lstm_618/lstm_cell_1855/addAddV20forward_lstm_618/lstm_cell_1855/MatMul:product:02forward_lstm_618/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2%
#forward_lstm_618/lstm_cell_1855/addн
6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOp?forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype028
6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpщ
'forward_lstm_618/lstm_cell_1855/BiasAddBiasAdd'forward_lstm_618/lstm_cell_1855/add:z:0>forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'forward_lstm_618/lstm_cell_1855/BiasAdd§
/forward_lstm_618/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/forward_lstm_618/lstm_cell_1855/split/split_dimњ
%forward_lstm_618/lstm_cell_1855/splitSplit8forward_lstm_618/lstm_cell_1855/split/split_dim:output:00forward_lstm_618/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2'
%forward_lstm_618/lstm_cell_1855/splitњ
'forward_lstm_618/lstm_cell_1855/SigmoidSigmoid.forward_lstm_618/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'forward_lstm_618/lstm_cell_1855/Sigmoid√
)forward_lstm_618/lstm_cell_1855/Sigmoid_1Sigmoid.forward_lstm_618/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_618/lstm_cell_1855/Sigmoid_1’
#forward_lstm_618/lstm_cell_1855/mulMul-forward_lstm_618/lstm_cell_1855/Sigmoid_1:y:0!forward_lstm_618/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22%
#forward_lstm_618/lstm_cell_1855/mulґ
$forward_lstm_618/lstm_cell_1855/ReluRelu.forward_lstm_618/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22&
$forward_lstm_618/lstm_cell_1855/Reluи
%forward_lstm_618/lstm_cell_1855/mul_1Mul+forward_lstm_618/lstm_cell_1855/Sigmoid:y:02forward_lstm_618/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_618/lstm_cell_1855/mul_1Ё
%forward_lstm_618/lstm_cell_1855/add_1AddV2'forward_lstm_618/lstm_cell_1855/mul:z:0)forward_lstm_618/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_618/lstm_cell_1855/add_1√
)forward_lstm_618/lstm_cell_1855/Sigmoid_2Sigmoid.forward_lstm_618/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_618/lstm_cell_1855/Sigmoid_2µ
&forward_lstm_618/lstm_cell_1855/Relu_1Relu)forward_lstm_618/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&forward_lstm_618/lstm_cell_1855/Relu_1м
%forward_lstm_618/lstm_cell_1855/mul_2Mul-forward_lstm_618/lstm_cell_1855/Sigmoid_2:y:04forward_lstm_618/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_618/lstm_cell_1855/mul_2±
.forward_lstm_618/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   20
.forward_lstm_618/TensorArrayV2_1/element_shapeь
 forward_lstm_618/TensorArrayV2_1TensorListReserve7forward_lstm_618/TensorArrayV2_1/element_shape:output:0)forward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_618/TensorArrayV2_1p
forward_lstm_618/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_618/time§
forward_lstm_618/zeros_like	ZerosLike)forward_lstm_618/lstm_cell_1855/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/zeros_like°
)forward_lstm_618/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2+
)forward_lstm_618/while/maximum_iterationsМ
#forward_lstm_618/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_618/while/loop_counterЦ	
forward_lstm_618/whileWhile,forward_lstm_618/while/loop_counter:output:02forward_lstm_618/while/maximum_iterations:output:0forward_lstm_618/time:output:0)forward_lstm_618/TensorArrayV2_1:handle:0forward_lstm_618/zeros_like:y:0forward_lstm_618/zeros:output:0!forward_lstm_618/zeros_1:output:0)forward_lstm_618/strided_slice_1:output:0Hforward_lstm_618/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_618/Cast:y:0>forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource@forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource?forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource*
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
$forward_lstm_618_while_body_62328816*0
cond(R&
$forward_lstm_618_while_cond_62328815*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
forward_lstm_618/while„
Aforward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2C
Aforward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeµ
3forward_lstm_618/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_618/while:output:3Jforward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype025
3forward_lstm_618/TensorArrayV2Stack/TensorListStack£
&forward_lstm_618/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2(
&forward_lstm_618/strided_slice_3/stackЮ
(forward_lstm_618/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_618/strided_slice_3/stack_1Ю
(forward_lstm_618/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_3/stack_2А
 forward_lstm_618/strided_slice_3StridedSlice<forward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_618/strided_slice_3/stack:output:01forward_lstm_618/strided_slice_3/stack_1:output:01forward_lstm_618/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2"
 forward_lstm_618/strided_slice_3Ы
!forward_lstm_618/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_618/transpose_1/permт
forward_lstm_618/transpose_1	Transpose<forward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_618/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
forward_lstm_618/transpose_1И
forward_lstm_618/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_618/runtimeЩ
&backward_lstm_618/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_618/RaggedToTensor/zerosЫ
&backward_lstm_618/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2(
&backward_lstm_618/RaggedToTensor/ConstЭ
5backward_lstm_618/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_618/RaggedToTensor/Const:output:0inputs/backward_lstm_618/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_618/RaggedToTensor/RaggedTensorToTensor∆
<backward_lstm_618/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack 
>backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1 
>backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2Ѓ
6backward_lstm_618/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_618/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask28
6backward_lstm_618/RaggedNestedRowLengths/strided_slice 
>backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack„
@backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2B
@backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1ќ
@backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2Ї
8backward_lstm_618/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask2:
8backward_lstm_618/RaggedNestedRowLengths/strided_slice_1Х
,backward_lstm_618/RaggedNestedRowLengths/subSub?backward_lstm_618/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_618/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2.
,backward_lstm_618/RaggedNestedRowLengths/subІ
backward_lstm_618/CastCast0backward_lstm_618/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
backward_lstm_618/Cast†
backward_lstm_618/ShapeShape>backward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_618/ShapeШ
%backward_lstm_618/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_618/strided_slice/stackЬ
'backward_lstm_618/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_618/strided_slice/stack_1Ь
'backward_lstm_618/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_618/strided_slice/stack_2ќ
backward_lstm_618/strided_sliceStridedSlice backward_lstm_618/Shape:output:0.backward_lstm_618/strided_slice/stack:output:00backward_lstm_618/strided_slice/stack_1:output:00backward_lstm_618/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_618/strided_sliceА
backward_lstm_618/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_618/zeros/mul/yі
backward_lstm_618/zeros/mulMul(backward_lstm_618/strided_slice:output:0&backward_lstm_618/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/zeros/mulГ
backward_lstm_618/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
backward_lstm_618/zeros/Less/yѓ
backward_lstm_618/zeros/LessLessbackward_lstm_618/zeros/mul:z:0'backward_lstm_618/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/zeros/LessЖ
 backward_lstm_618/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_618/zeros/packed/1Ћ
backward_lstm_618/zeros/packedPack(backward_lstm_618/strided_slice:output:0)backward_lstm_618/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_618/zeros/packedЗ
backward_lstm_618/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_618/zeros/Constљ
backward_lstm_618/zerosFill'backward_lstm_618/zeros/packed:output:0&backward_lstm_618/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_618/zerosД
backward_lstm_618/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_618/zeros_1/mul/yЇ
backward_lstm_618/zeros_1/mulMul(backward_lstm_618/strided_slice:output:0(backward_lstm_618/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/zeros_1/mulЗ
 backward_lstm_618/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 backward_lstm_618/zeros_1/Less/yЈ
backward_lstm_618/zeros_1/LessLess!backward_lstm_618/zeros_1/mul:z:0)backward_lstm_618/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_618/zeros_1/LessК
"backward_lstm_618/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_618/zeros_1/packed/1—
 backward_lstm_618/zeros_1/packedPack(backward_lstm_618/strided_slice:output:0+backward_lstm_618/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_618/zeros_1/packedЛ
backward_lstm_618/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_618/zeros_1/Const≈
backward_lstm_618/zeros_1Fill)backward_lstm_618/zeros_1/packed:output:0(backward_lstm_618/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_618/zeros_1Щ
 backward_lstm_618/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_618/transpose/permс
backward_lstm_618/transpose	Transpose>backward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_618/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_618/transposeЕ
backward_lstm_618/Shape_1Shapebackward_lstm_618/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_618/Shape_1Ь
'backward_lstm_618/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_618/strided_slice_1/stack†
)backward_lstm_618/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_1/stack_1†
)backward_lstm_618/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_1/stack_2Џ
!backward_lstm_618/strided_slice_1StridedSlice"backward_lstm_618/Shape_1:output:00backward_lstm_618/strided_slice_1/stack:output:02backward_lstm_618/strided_slice_1/stack_1:output:02backward_lstm_618/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_618/strided_slice_1©
-backward_lstm_618/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-backward_lstm_618/TensorArrayV2/element_shapeъ
backward_lstm_618/TensorArrayV2TensorListReserve6backward_lstm_618/TensorArrayV2/element_shape:output:0*backward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_618/TensorArrayV2О
 backward_lstm_618/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_618/ReverseV2/axis“
backward_lstm_618/ReverseV2	ReverseV2backward_lstm_618/transpose:y:0)backward_lstm_618/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_618/ReverseV2г
Gbackward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2I
Gbackward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape≈
9backward_lstm_618/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_618/ReverseV2:output:0Pbackward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_618/TensorArrayUnstack/TensorListFromTensorЬ
'backward_lstm_618/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_618/strided_slice_2/stack†
)backward_lstm_618/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_2/stack_1†
)backward_lstm_618/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_2/stack_2и
!backward_lstm_618/strided_slice_2StridedSlicebackward_lstm_618/transpose:y:00backward_lstm_618/strided_slice_2/stack:output:02backward_lstm_618/strided_slice_2/stack_1:output:02backward_lstm_618/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2#
!backward_lstm_618/strided_slice_2с
6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOp?backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype028
6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpы
'backward_lstm_618/lstm_cell_1856/MatMulMatMul*backward_lstm_618/strided_slice_2:output:0>backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'backward_lstm_618/lstm_cell_1856/MatMulч
8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOpAbackward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02:
8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpч
)backward_lstm_618/lstm_cell_1856/MatMul_1MatMul backward_lstm_618/zeros:output:0@backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)backward_lstm_618/lstm_cell_1856/MatMul_1р
$backward_lstm_618/lstm_cell_1856/addAddV21backward_lstm_618/lstm_cell_1856/MatMul:product:03backward_lstm_618/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2&
$backward_lstm_618/lstm_cell_1856/addр
7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOp@backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype029
7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpэ
(backward_lstm_618/lstm_cell_1856/BiasAddBiasAdd(backward_lstm_618/lstm_cell_1856/add:z:0?backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(backward_lstm_618/lstm_cell_1856/BiasAdd¶
0backward_lstm_618/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0backward_lstm_618/lstm_cell_1856/split/split_dim√
&backward_lstm_618/lstm_cell_1856/splitSplit9backward_lstm_618/lstm_cell_1856/split/split_dim:output:01backward_lstm_618/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2(
&backward_lstm_618/lstm_cell_1856/split¬
(backward_lstm_618/lstm_cell_1856/SigmoidSigmoid/backward_lstm_618/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(backward_lstm_618/lstm_cell_1856/Sigmoid∆
*backward_lstm_618/lstm_cell_1856/Sigmoid_1Sigmoid/backward_lstm_618/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_618/lstm_cell_1856/Sigmoid_1ў
$backward_lstm_618/lstm_cell_1856/mulMul.backward_lstm_618/lstm_cell_1856/Sigmoid_1:y:0"backward_lstm_618/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$backward_lstm_618/lstm_cell_1856/mulє
%backward_lstm_618/lstm_cell_1856/ReluRelu/backward_lstm_618/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22'
%backward_lstm_618/lstm_cell_1856/Reluм
&backward_lstm_618/lstm_cell_1856/mul_1Mul,backward_lstm_618/lstm_cell_1856/Sigmoid:y:03backward_lstm_618/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_618/lstm_cell_1856/mul_1б
&backward_lstm_618/lstm_cell_1856/add_1AddV2(backward_lstm_618/lstm_cell_1856/mul:z:0*backward_lstm_618/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_618/lstm_cell_1856/add_1∆
*backward_lstm_618/lstm_cell_1856/Sigmoid_2Sigmoid/backward_lstm_618/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_618/lstm_cell_1856/Sigmoid_2Є
'backward_lstm_618/lstm_cell_1856/Relu_1Relu*backward_lstm_618/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'backward_lstm_618/lstm_cell_1856/Relu_1р
&backward_lstm_618/lstm_cell_1856/mul_2Mul.backward_lstm_618/lstm_cell_1856/Sigmoid_2:y:05backward_lstm_618/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_618/lstm_cell_1856/mul_2≥
/backward_lstm_618/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   21
/backward_lstm_618/TensorArrayV2_1/element_shapeА
!backward_lstm_618/TensorArrayV2_1TensorListReserve8backward_lstm_618/TensorArrayV2_1/element_shape:output:0*backward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_618/TensorArrayV2_1r
backward_lstm_618/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_618/timeФ
'backward_lstm_618/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_618/Max/reduction_indices§
backward_lstm_618/MaxMaxbackward_lstm_618/Cast:y:00backward_lstm_618/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/Maxt
backward_lstm_618/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_618/sub/yШ
backward_lstm_618/subSubbackward_lstm_618/Max:output:0 backward_lstm_618/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/subЮ
backward_lstm_618/Sub_1Subbackward_lstm_618/sub:z:0backward_lstm_618/Cast:y:0*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_618/Sub_1І
backward_lstm_618/zeros_like	ZerosLike*backward_lstm_618/lstm_cell_1856/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_618/zeros_like£
*backward_lstm_618/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2,
*backward_lstm_618/while/maximum_iterationsО
$backward_lstm_618/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_618/while/loop_counter®	
backward_lstm_618/whileWhile-backward_lstm_618/while/loop_counter:output:03backward_lstm_618/while/maximum_iterations:output:0backward_lstm_618/time:output:0*backward_lstm_618/TensorArrayV2_1:handle:0 backward_lstm_618/zeros_like:y:0 backward_lstm_618/zeros:output:0"backward_lstm_618/zeros_1:output:0*backward_lstm_618/strided_slice_1:output:0Ibackward_lstm_618/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_618/Sub_1:z:0?backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resourceAbackward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource@backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource*
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
%backward_lstm_618_while_body_62328995*1
cond)R'
%backward_lstm_618_while_cond_62328994*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
backward_lstm_618/whileў
Bbackward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2D
Bbackward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeє
4backward_lstm_618/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_618/while:output:3Kbackward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype026
4backward_lstm_618/TensorArrayV2Stack/TensorListStack•
'backward_lstm_618/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2)
'backward_lstm_618/strided_slice_3/stack†
)backward_lstm_618/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_618/strided_slice_3/stack_1†
)backward_lstm_618/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_3/stack_2Ж
!backward_lstm_618/strided_slice_3StridedSlice=backward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_618/strided_slice_3/stack:output:02backward_lstm_618/strided_slice_3/stack_1:output:02backward_lstm_618/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2#
!backward_lstm_618/strided_slice_3Э
"backward_lstm_618/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_618/transpose_1/permц
backward_lstm_618/transpose_1	Transpose=backward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_618/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
backward_lstm_618/transpose_1К
backward_lstm_618/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_618/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisƒ
concatConcatV2)forward_lstm_618/strided_slice_3:output:0*backward_lstm_618/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp8^backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp7^backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp9^backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp^backward_lstm_618/while7^forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp6^forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp8^forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp^forward_lstm_618/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€:€€€€€€€€€: : : : : : 2r
7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp2p
6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp2t
8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp22
backward_lstm_618/whilebackward_lstm_618/while2p
6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp2n
5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp2r
7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp20
forward_lstm_618/whileforward_lstm_618/while:O K
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
$forward_lstm_618_while_cond_62329173>
:forward_lstm_618_while_forward_lstm_618_while_loop_counterD
@forward_lstm_618_while_forward_lstm_618_while_maximum_iterations&
"forward_lstm_618_while_placeholder(
$forward_lstm_618_while_placeholder_1(
$forward_lstm_618_while_placeholder_2(
$forward_lstm_618_while_placeholder_3(
$forward_lstm_618_while_placeholder_4@
<forward_lstm_618_while_less_forward_lstm_618_strided_slice_1X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62329173___redundant_placeholder0X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62329173___redundant_placeholder1X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62329173___redundant_placeholder2X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62329173___redundant_placeholder3X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62329173___redundant_placeholder4#
forward_lstm_618_while_identity
≈
forward_lstm_618/while/LessLess"forward_lstm_618_while_placeholder<forward_lstm_618_while_less_forward_lstm_618_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_618/while/LessР
forward_lstm_618/while/IdentityIdentityforward_lstm_618/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_618/while/Identity"K
forward_lstm_618_while_identity(forward_lstm_618/while/Identity:output:0*(
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
л	
Щ
4__inference_bidirectional_618_layer_call_fn_62328094
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
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_623270022
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
‘
¬
3__inference_forward_lstm_618_layer_call_fn_62329492
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
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_623253722
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
в
Ѕ
4__inference_backward_lstm_618_layer_call_fn_62330151

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
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_623265892
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
Esequential_618_bidirectional_618_forward_lstm_618_while_body_62324721А
|sequential_618_bidirectional_618_forward_lstm_618_while_sequential_618_bidirectional_618_forward_lstm_618_while_loop_counterЗ
Вsequential_618_bidirectional_618_forward_lstm_618_while_sequential_618_bidirectional_618_forward_lstm_618_while_maximum_iterationsG
Csequential_618_bidirectional_618_forward_lstm_618_while_placeholderI
Esequential_618_bidirectional_618_forward_lstm_618_while_placeholder_1I
Esequential_618_bidirectional_618_forward_lstm_618_while_placeholder_2I
Esequential_618_bidirectional_618_forward_lstm_618_while_placeholder_3I
Esequential_618_bidirectional_618_forward_lstm_618_while_placeholder_4
{sequential_618_bidirectional_618_forward_lstm_618_while_sequential_618_bidirectional_618_forward_lstm_618_strided_slice_1_0Љ
Јsequential_618_bidirectional_618_forward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_sequential_618_bidirectional_618_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0|
xsequential_618_bidirectional_618_forward_lstm_618_while_greater_sequential_618_bidirectional_618_forward_lstm_618_cast_0z
gsequential_618_bidirectional_618_forward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0:	»|
isequential_618_bidirectional_618_forward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0:	2»w
hsequential_618_bidirectional_618_forward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0:	»D
@sequential_618_bidirectional_618_forward_lstm_618_while_identityF
Bsequential_618_bidirectional_618_forward_lstm_618_while_identity_1F
Bsequential_618_bidirectional_618_forward_lstm_618_while_identity_2F
Bsequential_618_bidirectional_618_forward_lstm_618_while_identity_3F
Bsequential_618_bidirectional_618_forward_lstm_618_while_identity_4F
Bsequential_618_bidirectional_618_forward_lstm_618_while_identity_5F
Bsequential_618_bidirectional_618_forward_lstm_618_while_identity_6}
ysequential_618_bidirectional_618_forward_lstm_618_while_sequential_618_bidirectional_618_forward_lstm_618_strided_slice_1Ї
µsequential_618_bidirectional_618_forward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_sequential_618_bidirectional_618_forward_lstm_618_tensorarrayunstack_tensorlistfromtensorz
vsequential_618_bidirectional_618_forward_lstm_618_while_greater_sequential_618_bidirectional_618_forward_lstm_618_castx
esequential_618_bidirectional_618_forward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource:	»z
gsequential_618_bidirectional_618_forward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource:	2»u
fsequential_618_bidirectional_618_forward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource:	»ИҐ]sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpҐ\sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOpҐ^sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpІ
isequential_618/bidirectional_618/forward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2k
isequential_618/bidirectional_618/forward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeА
[sequential_618/bidirectional_618/forward_lstm_618/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЈsequential_618_bidirectional_618_forward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_sequential_618_bidirectional_618_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0Csequential_618_bidirectional_618_forward_lstm_618_while_placeholderrsequential_618/bidirectional_618/forward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02]
[sequential_618/bidirectional_618/forward_lstm_618/while/TensorArrayV2Read/TensorListGetItemъ
?sequential_618/bidirectional_618/forward_lstm_618/while/GreaterGreaterxsequential_618_bidirectional_618_forward_lstm_618_while_greater_sequential_618_bidirectional_618_forward_lstm_618_cast_0Csequential_618_bidirectional_618_forward_lstm_618_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2A
?sequential_618/bidirectional_618/forward_lstm_618/while/Greaterе
\sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOpgsequential_618_bidirectional_618_forward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02^
\sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp•
Msequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMulMatMulbsequential_618/bidirectional_618/forward_lstm_618/while/TensorArrayV2Read/TensorListGetItem:item:0dsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2O
Msequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMulл
^sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOpisequential_618_bidirectional_618_forward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02`
^sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpО
Osequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMul_1MatMulEsequential_618_bidirectional_618_forward_lstm_618_while_placeholder_3fsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2Q
Osequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMul_1И
Jsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/addAddV2Wsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMul:product:0Ysequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2L
Jsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/addд
]sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOphsequential_618_bidirectional_618_forward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02_
]sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpХ
Nsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/BiasAddBiasAddNsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/add:z:0esequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2P
Nsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/BiasAddт
Vsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2X
Vsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/split/split_dimџ
Lsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/splitSplit_sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/split/split_dim:output:0Wsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2N
Lsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/splitі
Nsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/SigmoidSigmoidUsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22P
Nsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/SigmoidЄ
Psequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/Sigmoid_1SigmoidUsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22R
Psequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/Sigmoid_1о
Jsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/mulMulTsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/Sigmoid_1:y:0Esequential_618_bidirectional_618_forward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22L
Jsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/mulЂ
Ksequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/ReluReluUsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22M
Ksequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/ReluД
Lsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/mul_1MulRsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/Sigmoid:y:0Ysequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22N
Lsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/mul_1щ
Lsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/add_1AddV2Nsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/mul:z:0Psequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22N
Lsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/add_1Є
Psequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/Sigmoid_2SigmoidUsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22R
Psequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/Sigmoid_2™
Msequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/Relu_1ReluPsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22O
Msequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/Relu_1И
Lsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/mul_2MulTsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/Sigmoid_2:y:0[sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22N
Lsequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/mul_2Ъ
>sequential_618/bidirectional_618/forward_lstm_618/while/SelectSelectCsequential_618/bidirectional_618/forward_lstm_618/while/Greater:z:0Psequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/mul_2:z:0Esequential_618_bidirectional_618_forward_lstm_618_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22@
>sequential_618/bidirectional_618/forward_lstm_618/while/SelectЮ
@sequential_618/bidirectional_618/forward_lstm_618/while/Select_1SelectCsequential_618/bidirectional_618/forward_lstm_618/while/Greater:z:0Psequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/mul_2:z:0Esequential_618_bidirectional_618_forward_lstm_618_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22B
@sequential_618/bidirectional_618/forward_lstm_618/while/Select_1Ю
@sequential_618/bidirectional_618/forward_lstm_618/while/Select_2SelectCsequential_618/bidirectional_618/forward_lstm_618/while/Greater:z:0Psequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/add_1:z:0Esequential_618_bidirectional_618_forward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22B
@sequential_618/bidirectional_618/forward_lstm_618/while/Select_2”
\sequential_618/bidirectional_618/forward_lstm_618/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemEsequential_618_bidirectional_618_forward_lstm_618_while_placeholder_1Csequential_618_bidirectional_618_forward_lstm_618_while_placeholderGsequential_618/bidirectional_618/forward_lstm_618/while/Select:output:0*
_output_shapes
: *
element_dtype02^
\sequential_618/bidirectional_618/forward_lstm_618/while/TensorArrayV2Write/TensorListSetItemј
=sequential_618/bidirectional_618/forward_lstm_618/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2?
=sequential_618/bidirectional_618/forward_lstm_618/while/add/y±
;sequential_618/bidirectional_618/forward_lstm_618/while/addAddV2Csequential_618_bidirectional_618_forward_lstm_618_while_placeholderFsequential_618/bidirectional_618/forward_lstm_618/while/add/y:output:0*
T0*
_output_shapes
: 2=
;sequential_618/bidirectional_618/forward_lstm_618/while/addƒ
?sequential_618/bidirectional_618/forward_lstm_618/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2A
?sequential_618/bidirectional_618/forward_lstm_618/while/add_1/yр
=sequential_618/bidirectional_618/forward_lstm_618/while/add_1AddV2|sequential_618_bidirectional_618_forward_lstm_618_while_sequential_618_bidirectional_618_forward_lstm_618_while_loop_counterHsequential_618/bidirectional_618/forward_lstm_618/while/add_1/y:output:0*
T0*
_output_shapes
: 2?
=sequential_618/bidirectional_618/forward_lstm_618/while/add_1≥
@sequential_618/bidirectional_618/forward_lstm_618/while/IdentityIdentityAsequential_618/bidirectional_618/forward_lstm_618/while/add_1:z:0=^sequential_618/bidirectional_618/forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2B
@sequential_618/bidirectional_618/forward_lstm_618/while/Identityщ
Bsequential_618/bidirectional_618/forward_lstm_618/while/Identity_1IdentityВsequential_618_bidirectional_618_forward_lstm_618_while_sequential_618_bidirectional_618_forward_lstm_618_while_maximum_iterations=^sequential_618/bidirectional_618/forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2D
Bsequential_618/bidirectional_618/forward_lstm_618/while/Identity_1µ
Bsequential_618/bidirectional_618/forward_lstm_618/while/Identity_2Identity?sequential_618/bidirectional_618/forward_lstm_618/while/add:z:0=^sequential_618/bidirectional_618/forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2D
Bsequential_618/bidirectional_618/forward_lstm_618/while/Identity_2в
Bsequential_618/bidirectional_618/forward_lstm_618/while/Identity_3Identitylsequential_618/bidirectional_618/forward_lstm_618/while/TensorArrayV2Write/TensorListSetItem:output_handle:0=^sequential_618/bidirectional_618/forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2D
Bsequential_618/bidirectional_618/forward_lstm_618/while/Identity_3ќ
Bsequential_618/bidirectional_618/forward_lstm_618/while/Identity_4IdentityGsequential_618/bidirectional_618/forward_lstm_618/while/Select:output:0=^sequential_618/bidirectional_618/forward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22D
Bsequential_618/bidirectional_618/forward_lstm_618/while/Identity_4–
Bsequential_618/bidirectional_618/forward_lstm_618/while/Identity_5IdentityIsequential_618/bidirectional_618/forward_lstm_618/while/Select_1:output:0=^sequential_618/bidirectional_618/forward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22D
Bsequential_618/bidirectional_618/forward_lstm_618/while/Identity_5–
Bsequential_618/bidirectional_618/forward_lstm_618/while/Identity_6IdentityIsequential_618/bidirectional_618/forward_lstm_618/while/Select_2:output:0=^sequential_618/bidirectional_618/forward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22D
Bsequential_618/bidirectional_618/forward_lstm_618/while/Identity_6ё
<sequential_618/bidirectional_618/forward_lstm_618/while/NoOpNoOp^^sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp]^sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp_^sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2>
<sequential_618/bidirectional_618/forward_lstm_618/while/NoOp"т
vsequential_618_bidirectional_618_forward_lstm_618_while_greater_sequential_618_bidirectional_618_forward_lstm_618_castxsequential_618_bidirectional_618_forward_lstm_618_while_greater_sequential_618_bidirectional_618_forward_lstm_618_cast_0"Н
@sequential_618_bidirectional_618_forward_lstm_618_while_identityIsequential_618/bidirectional_618/forward_lstm_618/while/Identity:output:0"С
Bsequential_618_bidirectional_618_forward_lstm_618_while_identity_1Ksequential_618/bidirectional_618/forward_lstm_618/while/Identity_1:output:0"С
Bsequential_618_bidirectional_618_forward_lstm_618_while_identity_2Ksequential_618/bidirectional_618/forward_lstm_618/while/Identity_2:output:0"С
Bsequential_618_bidirectional_618_forward_lstm_618_while_identity_3Ksequential_618/bidirectional_618/forward_lstm_618/while/Identity_3:output:0"С
Bsequential_618_bidirectional_618_forward_lstm_618_while_identity_4Ksequential_618/bidirectional_618/forward_lstm_618/while/Identity_4:output:0"С
Bsequential_618_bidirectional_618_forward_lstm_618_while_identity_5Ksequential_618/bidirectional_618/forward_lstm_618/while/Identity_5:output:0"С
Bsequential_618_bidirectional_618_forward_lstm_618_while_identity_6Ksequential_618/bidirectional_618/forward_lstm_618/while/Identity_6:output:0"“
fsequential_618_bidirectional_618_forward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resourcehsequential_618_bidirectional_618_forward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0"‘
gsequential_618_bidirectional_618_forward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resourceisequential_618_bidirectional_618_forward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0"–
esequential_618_bidirectional_618_forward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resourcegsequential_618_bidirectional_618_forward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0"ш
ysequential_618_bidirectional_618_forward_lstm_618_while_sequential_618_bidirectional_618_forward_lstm_618_strided_slice_1{sequential_618_bidirectional_618_forward_lstm_618_while_sequential_618_bidirectional_618_forward_lstm_618_strided_slice_1_0"т
µsequential_618_bidirectional_618_forward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_sequential_618_bidirectional_618_forward_lstm_618_tensorarrayunstack_tensorlistfromtensorЈsequential_618_bidirectional_618_forward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_sequential_618_bidirectional_618_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2Њ
]sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp]sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp2Љ
\sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp\sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp2ј
^sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp^sequential_618/bidirectional_618/forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
З
ш
G__inference_dense_618_layer_call_and_return_conditional_losses_62327465

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
ђ&
€
while_body_62325937
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_lstm_cell_1856_62325961_0:	»2
while_lstm_cell_1856_62325963_0:	2».
while_lstm_cell_1856_62325965_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_lstm_cell_1856_62325961:	»0
while_lstm_cell_1856_62325963:	2»,
while_lstm_cell_1856_62325965:	»ИҐ,while/lstm_cell_1856/StatefulPartitionedCall√
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
,while/lstm_cell_1856/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1856_62325961_0while_lstm_cell_1856_62325963_0while_lstm_cell_1856_62325965_0*
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
L__inference_lstm_cell_1856_layer_call_and_return_conditional_losses_623258572.
,while/lstm_cell_1856/StatefulPartitionedCallщ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder5while/lstm_cell_1856/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity5while/lstm_cell_1856/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4¶
while/Identity_5Identity5while/lstm_cell_1856/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5Й

while/NoOpNoOp-^while/lstm_cell_1856/StatefulPartitionedCall*"
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
while_lstm_cell_1856_62325961while_lstm_cell_1856_62325961_0"@
while_lstm_cell_1856_62325963while_lstm_cell_1856_62325963_0"@
while_lstm_cell_1856_62325965while_lstm_cell_1856_62325965_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2\
,while/lstm_cell_1856/StatefulPartitionedCall,while/lstm_cell_1856/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_62329450

inputs
inputs_1	Q
>forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource:	»S
@forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource:	2»N
?forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource:	»R
?backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resource:	»T
Abackward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource:	2»O
@backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource:	»
identityИҐ7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpҐ6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpҐ8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpҐbackward_lstm_618/whileҐ6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpҐ5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpҐ7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpҐforward_lstm_618/whileЧ
%forward_lstm_618/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_618/RaggedToTensor/zerosЩ
%forward_lstm_618/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2'
%forward_lstm_618/RaggedToTensor/ConstЩ
4forward_lstm_618/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_618/RaggedToTensor/Const:output:0inputs.forward_lstm_618/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_618/RaggedToTensor/RaggedTensorToTensorƒ
;forward_lstm_618/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack»
=forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1»
=forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2©
5forward_lstm_618/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_618/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask27
5forward_lstm_618/RaggedNestedRowLengths/strided_slice»
=forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack’
?forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2A
?forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1ћ
?forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2µ
7forward_lstm_618/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask29
7forward_lstm_618/RaggedNestedRowLengths/strided_slice_1С
+forward_lstm_618/RaggedNestedRowLengths/subSub>forward_lstm_618/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_618/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2-
+forward_lstm_618/RaggedNestedRowLengths/sub§
forward_lstm_618/CastCast/forward_lstm_618/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
forward_lstm_618/CastЭ
forward_lstm_618/ShapeShape=forward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_618/ShapeЦ
$forward_lstm_618/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_618/strided_slice/stackЪ
&forward_lstm_618/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_618/strided_slice/stack_1Ъ
&forward_lstm_618/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_618/strided_slice/stack_2»
forward_lstm_618/strided_sliceStridedSliceforward_lstm_618/Shape:output:0-forward_lstm_618/strided_slice/stack:output:0/forward_lstm_618/strided_slice/stack_1:output:0/forward_lstm_618/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_618/strided_slice~
forward_lstm_618/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_618/zeros/mul/y∞
forward_lstm_618/zeros/mulMul'forward_lstm_618/strided_slice:output:0%forward_lstm_618/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros/mulБ
forward_lstm_618/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
forward_lstm_618/zeros/Less/yЂ
forward_lstm_618/zeros/LessLessforward_lstm_618/zeros/mul:z:0&forward_lstm_618/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros/LessД
forward_lstm_618/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_618/zeros/packed/1«
forward_lstm_618/zeros/packedPack'forward_lstm_618/strided_slice:output:0(forward_lstm_618/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_618/zeros/packedЕ
forward_lstm_618/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_618/zeros/Constє
forward_lstm_618/zerosFill&forward_lstm_618/zeros/packed:output:0%forward_lstm_618/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/zerosВ
forward_lstm_618/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_618/zeros_1/mul/yґ
forward_lstm_618/zeros_1/mulMul'forward_lstm_618/strided_slice:output:0'forward_lstm_618/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros_1/mulЕ
forward_lstm_618/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2!
forward_lstm_618/zeros_1/Less/y≥
forward_lstm_618/zeros_1/LessLess forward_lstm_618/zeros_1/mul:z:0(forward_lstm_618/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros_1/LessИ
!forward_lstm_618/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_618/zeros_1/packed/1Ќ
forward_lstm_618/zeros_1/packedPack'forward_lstm_618/strided_slice:output:0*forward_lstm_618/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_618/zeros_1/packedЙ
forward_lstm_618/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_618/zeros_1/ConstЅ
forward_lstm_618/zeros_1Fill(forward_lstm_618/zeros_1/packed:output:0'forward_lstm_618/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/zeros_1Ч
forward_lstm_618/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_618/transpose/permн
forward_lstm_618/transpose	Transpose=forward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_618/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
forward_lstm_618/transposeВ
forward_lstm_618/Shape_1Shapeforward_lstm_618/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_618/Shape_1Ъ
&forward_lstm_618/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_618/strided_slice_1/stackЮ
(forward_lstm_618/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_1/stack_1Ю
(forward_lstm_618/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_1/stack_2‘
 forward_lstm_618/strided_slice_1StridedSlice!forward_lstm_618/Shape_1:output:0/forward_lstm_618/strided_slice_1/stack:output:01forward_lstm_618/strided_slice_1/stack_1:output:01forward_lstm_618/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_618/strided_slice_1І
,forward_lstm_618/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,forward_lstm_618/TensorArrayV2/element_shapeц
forward_lstm_618/TensorArrayV2TensorListReserve5forward_lstm_618/TensorArrayV2/element_shape:output:0)forward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_618/TensorArrayV2б
Fforward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2H
Fforward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeЉ
8forward_lstm_618/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_618/transpose:y:0Oforward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_618/TensorArrayUnstack/TensorListFromTensorЪ
&forward_lstm_618/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_618/strided_slice_2/stackЮ
(forward_lstm_618/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_2/stack_1Ю
(forward_lstm_618/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_2/stack_2в
 forward_lstm_618/strided_slice_2StridedSliceforward_lstm_618/transpose:y:0/forward_lstm_618/strided_slice_2/stack:output:01forward_lstm_618/strided_slice_2/stack_1:output:01forward_lstm_618/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2"
 forward_lstm_618/strided_slice_2о
5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOp>forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype027
5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpч
&forward_lstm_618/lstm_cell_1855/MatMulMatMul)forward_lstm_618/strided_slice_2:output:0=forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2(
&forward_lstm_618/lstm_cell_1855/MatMulф
7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOp@forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype029
7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpу
(forward_lstm_618/lstm_cell_1855/MatMul_1MatMulforward_lstm_618/zeros:output:0?forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(forward_lstm_618/lstm_cell_1855/MatMul_1м
#forward_lstm_618/lstm_cell_1855/addAddV20forward_lstm_618/lstm_cell_1855/MatMul:product:02forward_lstm_618/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2%
#forward_lstm_618/lstm_cell_1855/addн
6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOp?forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype028
6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpщ
'forward_lstm_618/lstm_cell_1855/BiasAddBiasAdd'forward_lstm_618/lstm_cell_1855/add:z:0>forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'forward_lstm_618/lstm_cell_1855/BiasAdd§
/forward_lstm_618/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/forward_lstm_618/lstm_cell_1855/split/split_dimњ
%forward_lstm_618/lstm_cell_1855/splitSplit8forward_lstm_618/lstm_cell_1855/split/split_dim:output:00forward_lstm_618/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2'
%forward_lstm_618/lstm_cell_1855/splitњ
'forward_lstm_618/lstm_cell_1855/SigmoidSigmoid.forward_lstm_618/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'forward_lstm_618/lstm_cell_1855/Sigmoid√
)forward_lstm_618/lstm_cell_1855/Sigmoid_1Sigmoid.forward_lstm_618/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_618/lstm_cell_1855/Sigmoid_1’
#forward_lstm_618/lstm_cell_1855/mulMul-forward_lstm_618/lstm_cell_1855/Sigmoid_1:y:0!forward_lstm_618/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22%
#forward_lstm_618/lstm_cell_1855/mulґ
$forward_lstm_618/lstm_cell_1855/ReluRelu.forward_lstm_618/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22&
$forward_lstm_618/lstm_cell_1855/Reluи
%forward_lstm_618/lstm_cell_1855/mul_1Mul+forward_lstm_618/lstm_cell_1855/Sigmoid:y:02forward_lstm_618/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_618/lstm_cell_1855/mul_1Ё
%forward_lstm_618/lstm_cell_1855/add_1AddV2'forward_lstm_618/lstm_cell_1855/mul:z:0)forward_lstm_618/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_618/lstm_cell_1855/add_1√
)forward_lstm_618/lstm_cell_1855/Sigmoid_2Sigmoid.forward_lstm_618/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_618/lstm_cell_1855/Sigmoid_2µ
&forward_lstm_618/lstm_cell_1855/Relu_1Relu)forward_lstm_618/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&forward_lstm_618/lstm_cell_1855/Relu_1м
%forward_lstm_618/lstm_cell_1855/mul_2Mul-forward_lstm_618/lstm_cell_1855/Sigmoid_2:y:04forward_lstm_618/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_618/lstm_cell_1855/mul_2±
.forward_lstm_618/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   20
.forward_lstm_618/TensorArrayV2_1/element_shapeь
 forward_lstm_618/TensorArrayV2_1TensorListReserve7forward_lstm_618/TensorArrayV2_1/element_shape:output:0)forward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_618/TensorArrayV2_1p
forward_lstm_618/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_618/time§
forward_lstm_618/zeros_like	ZerosLike)forward_lstm_618/lstm_cell_1855/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/zeros_like°
)forward_lstm_618/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2+
)forward_lstm_618/while/maximum_iterationsМ
#forward_lstm_618/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_618/while/loop_counterЦ	
forward_lstm_618/whileWhile,forward_lstm_618/while/loop_counter:output:02forward_lstm_618/while/maximum_iterations:output:0forward_lstm_618/time:output:0)forward_lstm_618/TensorArrayV2_1:handle:0forward_lstm_618/zeros_like:y:0forward_lstm_618/zeros:output:0!forward_lstm_618/zeros_1:output:0)forward_lstm_618/strided_slice_1:output:0Hforward_lstm_618/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_618/Cast:y:0>forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource@forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource?forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource*
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
$forward_lstm_618_while_body_62329174*0
cond(R&
$forward_lstm_618_while_cond_62329173*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
forward_lstm_618/while„
Aforward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2C
Aforward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeµ
3forward_lstm_618/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_618/while:output:3Jforward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype025
3forward_lstm_618/TensorArrayV2Stack/TensorListStack£
&forward_lstm_618/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2(
&forward_lstm_618/strided_slice_3/stackЮ
(forward_lstm_618/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_618/strided_slice_3/stack_1Ю
(forward_lstm_618/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_3/stack_2А
 forward_lstm_618/strided_slice_3StridedSlice<forward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_618/strided_slice_3/stack:output:01forward_lstm_618/strided_slice_3/stack_1:output:01forward_lstm_618/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2"
 forward_lstm_618/strided_slice_3Ы
!forward_lstm_618/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_618/transpose_1/permт
forward_lstm_618/transpose_1	Transpose<forward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_618/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
forward_lstm_618/transpose_1И
forward_lstm_618/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_618/runtimeЩ
&backward_lstm_618/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_618/RaggedToTensor/zerosЫ
&backward_lstm_618/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2(
&backward_lstm_618/RaggedToTensor/ConstЭ
5backward_lstm_618/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_618/RaggedToTensor/Const:output:0inputs/backward_lstm_618/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_618/RaggedToTensor/RaggedTensorToTensor∆
<backward_lstm_618/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack 
>backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1 
>backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2Ѓ
6backward_lstm_618/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_618/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask28
6backward_lstm_618/RaggedNestedRowLengths/strided_slice 
>backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack„
@backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2B
@backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1ќ
@backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2Ї
8backward_lstm_618/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask2:
8backward_lstm_618/RaggedNestedRowLengths/strided_slice_1Х
,backward_lstm_618/RaggedNestedRowLengths/subSub?backward_lstm_618/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_618/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2.
,backward_lstm_618/RaggedNestedRowLengths/subІ
backward_lstm_618/CastCast0backward_lstm_618/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
backward_lstm_618/Cast†
backward_lstm_618/ShapeShape>backward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_618/ShapeШ
%backward_lstm_618/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_618/strided_slice/stackЬ
'backward_lstm_618/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_618/strided_slice/stack_1Ь
'backward_lstm_618/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_618/strided_slice/stack_2ќ
backward_lstm_618/strided_sliceStridedSlice backward_lstm_618/Shape:output:0.backward_lstm_618/strided_slice/stack:output:00backward_lstm_618/strided_slice/stack_1:output:00backward_lstm_618/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_618/strided_sliceА
backward_lstm_618/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_618/zeros/mul/yі
backward_lstm_618/zeros/mulMul(backward_lstm_618/strided_slice:output:0&backward_lstm_618/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/zeros/mulГ
backward_lstm_618/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
backward_lstm_618/zeros/Less/yѓ
backward_lstm_618/zeros/LessLessbackward_lstm_618/zeros/mul:z:0'backward_lstm_618/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/zeros/LessЖ
 backward_lstm_618/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_618/zeros/packed/1Ћ
backward_lstm_618/zeros/packedPack(backward_lstm_618/strided_slice:output:0)backward_lstm_618/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_618/zeros/packedЗ
backward_lstm_618/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_618/zeros/Constљ
backward_lstm_618/zerosFill'backward_lstm_618/zeros/packed:output:0&backward_lstm_618/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_618/zerosД
backward_lstm_618/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_618/zeros_1/mul/yЇ
backward_lstm_618/zeros_1/mulMul(backward_lstm_618/strided_slice:output:0(backward_lstm_618/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/zeros_1/mulЗ
 backward_lstm_618/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 backward_lstm_618/zeros_1/Less/yЈ
backward_lstm_618/zeros_1/LessLess!backward_lstm_618/zeros_1/mul:z:0)backward_lstm_618/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_618/zeros_1/LessК
"backward_lstm_618/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_618/zeros_1/packed/1—
 backward_lstm_618/zeros_1/packedPack(backward_lstm_618/strided_slice:output:0+backward_lstm_618/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_618/zeros_1/packedЛ
backward_lstm_618/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_618/zeros_1/Const≈
backward_lstm_618/zeros_1Fill)backward_lstm_618/zeros_1/packed:output:0(backward_lstm_618/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_618/zeros_1Щ
 backward_lstm_618/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_618/transpose/permс
backward_lstm_618/transpose	Transpose>backward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_618/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_618/transposeЕ
backward_lstm_618/Shape_1Shapebackward_lstm_618/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_618/Shape_1Ь
'backward_lstm_618/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_618/strided_slice_1/stack†
)backward_lstm_618/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_1/stack_1†
)backward_lstm_618/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_1/stack_2Џ
!backward_lstm_618/strided_slice_1StridedSlice"backward_lstm_618/Shape_1:output:00backward_lstm_618/strided_slice_1/stack:output:02backward_lstm_618/strided_slice_1/stack_1:output:02backward_lstm_618/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_618/strided_slice_1©
-backward_lstm_618/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-backward_lstm_618/TensorArrayV2/element_shapeъ
backward_lstm_618/TensorArrayV2TensorListReserve6backward_lstm_618/TensorArrayV2/element_shape:output:0*backward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_618/TensorArrayV2О
 backward_lstm_618/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_618/ReverseV2/axis“
backward_lstm_618/ReverseV2	ReverseV2backward_lstm_618/transpose:y:0)backward_lstm_618/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_618/ReverseV2г
Gbackward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2I
Gbackward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape≈
9backward_lstm_618/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_618/ReverseV2:output:0Pbackward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_618/TensorArrayUnstack/TensorListFromTensorЬ
'backward_lstm_618/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_618/strided_slice_2/stack†
)backward_lstm_618/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_2/stack_1†
)backward_lstm_618/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_2/stack_2и
!backward_lstm_618/strided_slice_2StridedSlicebackward_lstm_618/transpose:y:00backward_lstm_618/strided_slice_2/stack:output:02backward_lstm_618/strided_slice_2/stack_1:output:02backward_lstm_618/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2#
!backward_lstm_618/strided_slice_2с
6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOp?backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype028
6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpы
'backward_lstm_618/lstm_cell_1856/MatMulMatMul*backward_lstm_618/strided_slice_2:output:0>backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'backward_lstm_618/lstm_cell_1856/MatMulч
8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOpAbackward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02:
8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpч
)backward_lstm_618/lstm_cell_1856/MatMul_1MatMul backward_lstm_618/zeros:output:0@backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)backward_lstm_618/lstm_cell_1856/MatMul_1р
$backward_lstm_618/lstm_cell_1856/addAddV21backward_lstm_618/lstm_cell_1856/MatMul:product:03backward_lstm_618/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2&
$backward_lstm_618/lstm_cell_1856/addр
7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOp@backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype029
7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpэ
(backward_lstm_618/lstm_cell_1856/BiasAddBiasAdd(backward_lstm_618/lstm_cell_1856/add:z:0?backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(backward_lstm_618/lstm_cell_1856/BiasAdd¶
0backward_lstm_618/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0backward_lstm_618/lstm_cell_1856/split/split_dim√
&backward_lstm_618/lstm_cell_1856/splitSplit9backward_lstm_618/lstm_cell_1856/split/split_dim:output:01backward_lstm_618/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2(
&backward_lstm_618/lstm_cell_1856/split¬
(backward_lstm_618/lstm_cell_1856/SigmoidSigmoid/backward_lstm_618/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(backward_lstm_618/lstm_cell_1856/Sigmoid∆
*backward_lstm_618/lstm_cell_1856/Sigmoid_1Sigmoid/backward_lstm_618/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_618/lstm_cell_1856/Sigmoid_1ў
$backward_lstm_618/lstm_cell_1856/mulMul.backward_lstm_618/lstm_cell_1856/Sigmoid_1:y:0"backward_lstm_618/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$backward_lstm_618/lstm_cell_1856/mulє
%backward_lstm_618/lstm_cell_1856/ReluRelu/backward_lstm_618/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22'
%backward_lstm_618/lstm_cell_1856/Reluм
&backward_lstm_618/lstm_cell_1856/mul_1Mul,backward_lstm_618/lstm_cell_1856/Sigmoid:y:03backward_lstm_618/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_618/lstm_cell_1856/mul_1б
&backward_lstm_618/lstm_cell_1856/add_1AddV2(backward_lstm_618/lstm_cell_1856/mul:z:0*backward_lstm_618/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_618/lstm_cell_1856/add_1∆
*backward_lstm_618/lstm_cell_1856/Sigmoid_2Sigmoid/backward_lstm_618/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_618/lstm_cell_1856/Sigmoid_2Є
'backward_lstm_618/lstm_cell_1856/Relu_1Relu*backward_lstm_618/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'backward_lstm_618/lstm_cell_1856/Relu_1р
&backward_lstm_618/lstm_cell_1856/mul_2Mul.backward_lstm_618/lstm_cell_1856/Sigmoid_2:y:05backward_lstm_618/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_618/lstm_cell_1856/mul_2≥
/backward_lstm_618/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   21
/backward_lstm_618/TensorArrayV2_1/element_shapeА
!backward_lstm_618/TensorArrayV2_1TensorListReserve8backward_lstm_618/TensorArrayV2_1/element_shape:output:0*backward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_618/TensorArrayV2_1r
backward_lstm_618/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_618/timeФ
'backward_lstm_618/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_618/Max/reduction_indices§
backward_lstm_618/MaxMaxbackward_lstm_618/Cast:y:00backward_lstm_618/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/Maxt
backward_lstm_618/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_618/sub/yШ
backward_lstm_618/subSubbackward_lstm_618/Max:output:0 backward_lstm_618/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/subЮ
backward_lstm_618/Sub_1Subbackward_lstm_618/sub:z:0backward_lstm_618/Cast:y:0*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_618/Sub_1І
backward_lstm_618/zeros_like	ZerosLike*backward_lstm_618/lstm_cell_1856/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_618/zeros_like£
*backward_lstm_618/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2,
*backward_lstm_618/while/maximum_iterationsО
$backward_lstm_618/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_618/while/loop_counter®	
backward_lstm_618/whileWhile-backward_lstm_618/while/loop_counter:output:03backward_lstm_618/while/maximum_iterations:output:0backward_lstm_618/time:output:0*backward_lstm_618/TensorArrayV2_1:handle:0 backward_lstm_618/zeros_like:y:0 backward_lstm_618/zeros:output:0"backward_lstm_618/zeros_1:output:0*backward_lstm_618/strided_slice_1:output:0Ibackward_lstm_618/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_618/Sub_1:z:0?backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resourceAbackward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource@backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource*
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
%backward_lstm_618_while_body_62329353*1
cond)R'
%backward_lstm_618_while_cond_62329352*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
backward_lstm_618/whileў
Bbackward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2D
Bbackward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeє
4backward_lstm_618/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_618/while:output:3Kbackward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype026
4backward_lstm_618/TensorArrayV2Stack/TensorListStack•
'backward_lstm_618/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2)
'backward_lstm_618/strided_slice_3/stack†
)backward_lstm_618/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_618/strided_slice_3/stack_1†
)backward_lstm_618/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_3/stack_2Ж
!backward_lstm_618/strided_slice_3StridedSlice=backward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_618/strided_slice_3/stack:output:02backward_lstm_618/strided_slice_3/stack_1:output:02backward_lstm_618/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2#
!backward_lstm_618/strided_slice_3Э
"backward_lstm_618/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_618/transpose_1/permц
backward_lstm_618/transpose_1	Transpose=backward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_618/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
backward_lstm_618/transpose_1К
backward_lstm_618/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_618/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisƒ
concatConcatV2)forward_lstm_618/strided_slice_3:output:0*backward_lstm_618/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp8^backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp7^backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp9^backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp^backward_lstm_618/while7^forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp6^forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp8^forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp^forward_lstm_618/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€:€€€€€€€€€: : : : : : 2r
7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp2p
6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp2t
8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp22
backward_lstm_618/whilebackward_lstm_618/while2p
6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp2n
5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp2r
7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp20
forward_lstm_618/whileforward_lstm_618/while:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ч
И
L__inference_lstm_cell_1855_layer_call_and_return_conditional_losses_62325079

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
while_cond_62330230
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_62330230___redundant_placeholder06
2while_while_cond_62330230___redundant_placeholder16
2while_while_cond_62330230___redundant_placeholder26
2while_while_cond_62330230___redundant_placeholder3
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
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_62329816
inputs_0@
-lstm_cell_1855_matmul_readvariableop_resource:	»B
/lstm_cell_1855_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1855_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1855/BiasAdd/ReadVariableOpҐ$lstm_cell_1855/MatMul/ReadVariableOpҐ&lstm_cell_1855/MatMul_1/ReadVariableOpҐwhileF
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
$lstm_cell_1855/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1855_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1855/MatMul/ReadVariableOp≥
lstm_cell_1855/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/MatMulЅ
&lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1855_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1855/MatMul_1/ReadVariableOpѓ
lstm_cell_1855/MatMul_1MatMulzeros:output:0.lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/MatMul_1®
lstm_cell_1855/addAddV2lstm_cell_1855/MatMul:product:0!lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/addЇ
%lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1855_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1855/BiasAdd/ReadVariableOpµ
lstm_cell_1855/BiasAddBiasAddlstm_cell_1855/add:z:0-lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/BiasAddВ
lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1855/split/split_dimы
lstm_cell_1855/splitSplit'lstm_cell_1855/split/split_dim:output:0lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1855/splitМ
lstm_cell_1855/SigmoidSigmoidlstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/SigmoidР
lstm_cell_1855/Sigmoid_1Sigmoidlstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Sigmoid_1С
lstm_cell_1855/mulMullstm_cell_1855/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/mulГ
lstm_cell_1855/ReluRelulstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Relu§
lstm_cell_1855/mul_1Mullstm_cell_1855/Sigmoid:y:0!lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/mul_1Щ
lstm_cell_1855/add_1AddV2lstm_cell_1855/mul:z:0lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/add_1Р
lstm_cell_1855/Sigmoid_2Sigmoidlstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Sigmoid_2В
lstm_cell_1855/Relu_1Relulstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Relu_1®
lstm_cell_1855/mul_2Mullstm_cell_1855/Sigmoid_2:y:0#lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1855_matmul_readvariableop_resource/lstm_cell_1855_matmul_1_readvariableop_resource.lstm_cell_1855_biasadd_readvariableop_resource*
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
while_body_62329732*
condR
while_cond_62329731*K
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
NoOpNoOp&^lstm_cell_1855/BiasAdd/ReadVariableOp%^lstm_cell_1855/MatMul/ReadVariableOp'^lstm_cell_1855/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1855/BiasAdd/ReadVariableOp%lstm_cell_1855/BiasAdd/ReadVariableOp2L
$lstm_cell_1855/MatMul/ReadVariableOp$lstm_cell_1855/MatMul/ReadVariableOp2P
&lstm_cell_1855/MatMul_1/ReadVariableOp&lstm_cell_1855/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
–]
і
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_62329665
inputs_0@
-lstm_cell_1855_matmul_readvariableop_resource:	»B
/lstm_cell_1855_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1855_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1855/BiasAdd/ReadVariableOpҐ$lstm_cell_1855/MatMul/ReadVariableOpҐ&lstm_cell_1855/MatMul_1/ReadVariableOpҐwhileF
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
$lstm_cell_1855/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1855_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1855/MatMul/ReadVariableOp≥
lstm_cell_1855/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/MatMulЅ
&lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1855_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1855/MatMul_1/ReadVariableOpѓ
lstm_cell_1855/MatMul_1MatMulzeros:output:0.lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/MatMul_1®
lstm_cell_1855/addAddV2lstm_cell_1855/MatMul:product:0!lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/addЇ
%lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1855_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1855/BiasAdd/ReadVariableOpµ
lstm_cell_1855/BiasAddBiasAddlstm_cell_1855/add:z:0-lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/BiasAddВ
lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1855/split/split_dimы
lstm_cell_1855/splitSplit'lstm_cell_1855/split/split_dim:output:0lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1855/splitМ
lstm_cell_1855/SigmoidSigmoidlstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/SigmoidР
lstm_cell_1855/Sigmoid_1Sigmoidlstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Sigmoid_1С
lstm_cell_1855/mulMullstm_cell_1855/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/mulГ
lstm_cell_1855/ReluRelulstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Relu§
lstm_cell_1855/mul_1Mullstm_cell_1855/Sigmoid:y:0!lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/mul_1Щ
lstm_cell_1855/add_1AddV2lstm_cell_1855/mul:z:0lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/add_1Р
lstm_cell_1855/Sigmoid_2Sigmoidlstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Sigmoid_2В
lstm_cell_1855/Relu_1Relulstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Relu_1®
lstm_cell_1855/mul_2Mullstm_cell_1855/Sigmoid_2:y:0#lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1855_matmul_readvariableop_resource/lstm_cell_1855_matmul_1_readvariableop_resource.lstm_cell_1855_biasadd_readvariableop_resource*
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
while_body_62329581*
condR
while_cond_62329580*K
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
NoOpNoOp&^lstm_cell_1855/BiasAdd/ReadVariableOp%^lstm_cell_1855/MatMul/ReadVariableOp'^lstm_cell_1855/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1855/BiasAdd/ReadVariableOp%lstm_cell_1855/BiasAdd/ReadVariableOp2L
$lstm_cell_1855/MatMul/ReadVariableOp$lstm_cell_1855/MatMul/ReadVariableOp2P
&lstm_cell_1855/MatMul_1/ReadVariableOp&lstm_cell_1855/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
я
Ќ
while_cond_62326504
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_62326504___redundant_placeholder06
2while_while_cond_62326504___redundant_placeholder16
2while_while_cond_62326504___redundant_placeholder26
2while_while_cond_62326504___redundant_placeholder3
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
…
•
$forward_lstm_618_while_cond_62327163>
:forward_lstm_618_while_forward_lstm_618_while_loop_counterD
@forward_lstm_618_while_forward_lstm_618_while_maximum_iterations&
"forward_lstm_618_while_placeholder(
$forward_lstm_618_while_placeholder_1(
$forward_lstm_618_while_placeholder_2(
$forward_lstm_618_while_placeholder_3(
$forward_lstm_618_while_placeholder_4@
<forward_lstm_618_while_less_forward_lstm_618_strided_slice_1X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62327163___redundant_placeholder0X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62327163___redundant_placeholder1X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62327163___redundant_placeholder2X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62327163___redundant_placeholder3X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62327163___redundant_placeholder4#
forward_lstm_618_while_identity
≈
forward_lstm_618/while/LessLess"forward_lstm_618_while_placeholder<forward_lstm_618_while_less_forward_lstm_618_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_618/while/LessР
forward_lstm_618/while/IdentityIdentityforward_lstm_618/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_618/while/Identity"K
forward_lstm_618_while_identity(forward_lstm_618/while/Identity:output:0*(
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
уд
У
#__inference__wrapped_model_62325004

args_0
args_0_1	r
_sequential_618_bidirectional_618_forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource:	»t
asequential_618_bidirectional_618_forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource:	2»o
`sequential_618_bidirectional_618_forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource:	»s
`sequential_618_bidirectional_618_backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resource:	»u
bsequential_618_bidirectional_618_backward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource:	2»p
asequential_618_bidirectional_618_backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource:	»I
7sequential_618_dense_618_matmul_readvariableop_resource:dF
8sequential_618_dense_618_biasadd_readvariableop_resource:
identityИҐXsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpҐWsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpҐYsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpҐ8sequential_618/bidirectional_618/backward_lstm_618/whileҐWsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpҐVsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpҐXsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpҐ7sequential_618/bidirectional_618/forward_lstm_618/whileҐ/sequential_618/dense_618/BiasAdd/ReadVariableOpҐ.sequential_618/dense_618/MatMul/ReadVariableOpў
Fsequential_618/bidirectional_618/forward_lstm_618/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2H
Fsequential_618/bidirectional_618/forward_lstm_618/RaggedToTensor/zerosџ
Fsequential_618/bidirectional_618/forward_lstm_618/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2H
Fsequential_618/bidirectional_618/forward_lstm_618/RaggedToTensor/ConstЭ
Usequential_618/bidirectional_618/forward_lstm_618/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorOsequential_618/bidirectional_618/forward_lstm_618/RaggedToTensor/Const:output:0args_0Osequential_618/bidirectional_618/forward_lstm_618/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2W
Usequential_618/bidirectional_618/forward_lstm_618/RaggedToTensor/RaggedTensorToTensorЖ
\sequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2^
\sequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice/stackК
^sequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2`
^sequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1К
^sequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2`
^sequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2ќ
Vsequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1esequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack:output:0gsequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1:output:0gsequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask2X
Vsequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_sliceК
^sequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2`
^sequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stackЧ
`sequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2b
`sequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1О
`sequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2b
`sequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2Џ
Xsequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1gsequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack:output:0isequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0isequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask2Z
Xsequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice_1Х
Lsequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/subSub_sequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice:output:0asequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2N
Lsequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/subЗ
6sequential_618/bidirectional_618/forward_lstm_618/CastCastPsequential_618/bidirectional_618/forward_lstm_618/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€28
6sequential_618/bidirectional_618/forward_lstm_618/CastА
7sequential_618/bidirectional_618/forward_lstm_618/ShapeShape^sequential_618/bidirectional_618/forward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:29
7sequential_618/bidirectional_618/forward_lstm_618/ShapeЎ
Esequential_618/bidirectional_618/forward_lstm_618/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_618/bidirectional_618/forward_lstm_618/strided_slice/stack№
Gsequential_618/bidirectional_618/forward_lstm_618/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_618/bidirectional_618/forward_lstm_618/strided_slice/stack_1№
Gsequential_618/bidirectional_618/forward_lstm_618/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_618/bidirectional_618/forward_lstm_618/strided_slice/stack_2О
?sequential_618/bidirectional_618/forward_lstm_618/strided_sliceStridedSlice@sequential_618/bidirectional_618/forward_lstm_618/Shape:output:0Nsequential_618/bidirectional_618/forward_lstm_618/strided_slice/stack:output:0Psequential_618/bidirectional_618/forward_lstm_618/strided_slice/stack_1:output:0Psequential_618/bidirectional_618/forward_lstm_618/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?sequential_618/bidirectional_618/forward_lstm_618/strided_sliceј
=sequential_618/bidirectional_618/forward_lstm_618/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22?
=sequential_618/bidirectional_618/forward_lstm_618/zeros/mul/yі
;sequential_618/bidirectional_618/forward_lstm_618/zeros/mulMulHsequential_618/bidirectional_618/forward_lstm_618/strided_slice:output:0Fsequential_618/bidirectional_618/forward_lstm_618/zeros/mul/y:output:0*
T0*
_output_shapes
: 2=
;sequential_618/bidirectional_618/forward_lstm_618/zeros/mul√
>sequential_618/bidirectional_618/forward_lstm_618/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2@
>sequential_618/bidirectional_618/forward_lstm_618/zeros/Less/yѓ
<sequential_618/bidirectional_618/forward_lstm_618/zeros/LessLess?sequential_618/bidirectional_618/forward_lstm_618/zeros/mul:z:0Gsequential_618/bidirectional_618/forward_lstm_618/zeros/Less/y:output:0*
T0*
_output_shapes
: 2>
<sequential_618/bidirectional_618/forward_lstm_618/zeros/Less∆
@sequential_618/bidirectional_618/forward_lstm_618/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22B
@sequential_618/bidirectional_618/forward_lstm_618/zeros/packed/1Ћ
>sequential_618/bidirectional_618/forward_lstm_618/zeros/packedPackHsequential_618/bidirectional_618/forward_lstm_618/strided_slice:output:0Isequential_618/bidirectional_618/forward_lstm_618/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2@
>sequential_618/bidirectional_618/forward_lstm_618/zeros/packed«
=sequential_618/bidirectional_618/forward_lstm_618/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2?
=sequential_618/bidirectional_618/forward_lstm_618/zeros/Constљ
7sequential_618/bidirectional_618/forward_lstm_618/zerosFillGsequential_618/bidirectional_618/forward_lstm_618/zeros/packed:output:0Fsequential_618/bidirectional_618/forward_lstm_618/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€229
7sequential_618/bidirectional_618/forward_lstm_618/zerosƒ
?sequential_618/bidirectional_618/forward_lstm_618/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22A
?sequential_618/bidirectional_618/forward_lstm_618/zeros_1/mul/yЇ
=sequential_618/bidirectional_618/forward_lstm_618/zeros_1/mulMulHsequential_618/bidirectional_618/forward_lstm_618/strided_slice:output:0Hsequential_618/bidirectional_618/forward_lstm_618/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2?
=sequential_618/bidirectional_618/forward_lstm_618/zeros_1/mul«
@sequential_618/bidirectional_618/forward_lstm_618/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2B
@sequential_618/bidirectional_618/forward_lstm_618/zeros_1/Less/yЈ
>sequential_618/bidirectional_618/forward_lstm_618/zeros_1/LessLessAsequential_618/bidirectional_618/forward_lstm_618/zeros_1/mul:z:0Isequential_618/bidirectional_618/forward_lstm_618/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2@
>sequential_618/bidirectional_618/forward_lstm_618/zeros_1/Less 
Bsequential_618/bidirectional_618/forward_lstm_618/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22D
Bsequential_618/bidirectional_618/forward_lstm_618/zeros_1/packed/1—
@sequential_618/bidirectional_618/forward_lstm_618/zeros_1/packedPackHsequential_618/bidirectional_618/forward_lstm_618/strided_slice:output:0Ksequential_618/bidirectional_618/forward_lstm_618/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2B
@sequential_618/bidirectional_618/forward_lstm_618/zeros_1/packedЋ
?sequential_618/bidirectional_618/forward_lstm_618/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2A
?sequential_618/bidirectional_618/forward_lstm_618/zeros_1/Const≈
9sequential_618/bidirectional_618/forward_lstm_618/zeros_1FillIsequential_618/bidirectional_618/forward_lstm_618/zeros_1/packed:output:0Hsequential_618/bidirectional_618/forward_lstm_618/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22;
9sequential_618/bidirectional_618/forward_lstm_618/zeros_1ў
@sequential_618/bidirectional_618/forward_lstm_618/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2B
@sequential_618/bidirectional_618/forward_lstm_618/transpose/permс
;sequential_618/bidirectional_618/forward_lstm_618/transpose	Transpose^sequential_618/bidirectional_618/forward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0Isequential_618/bidirectional_618/forward_lstm_618/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2=
;sequential_618/bidirectional_618/forward_lstm_618/transposeе
9sequential_618/bidirectional_618/forward_lstm_618/Shape_1Shape?sequential_618/bidirectional_618/forward_lstm_618/transpose:y:0*
T0*
_output_shapes
:2;
9sequential_618/bidirectional_618/forward_lstm_618/Shape_1№
Gsequential_618/bidirectional_618/forward_lstm_618/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2I
Gsequential_618/bidirectional_618/forward_lstm_618/strided_slice_1/stackа
Isequential_618/bidirectional_618/forward_lstm_618/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_618/bidirectional_618/forward_lstm_618/strided_slice_1/stack_1а
Isequential_618/bidirectional_618/forward_lstm_618/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_618/bidirectional_618/forward_lstm_618/strided_slice_1/stack_2Ъ
Asequential_618/bidirectional_618/forward_lstm_618/strided_slice_1StridedSliceBsequential_618/bidirectional_618/forward_lstm_618/Shape_1:output:0Psequential_618/bidirectional_618/forward_lstm_618/strided_slice_1/stack:output:0Rsequential_618/bidirectional_618/forward_lstm_618/strided_slice_1/stack_1:output:0Rsequential_618/bidirectional_618/forward_lstm_618/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2C
Asequential_618/bidirectional_618/forward_lstm_618/strided_slice_1й
Msequential_618/bidirectional_618/forward_lstm_618/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2O
Msequential_618/bidirectional_618/forward_lstm_618/TensorArrayV2/element_shapeъ
?sequential_618/bidirectional_618/forward_lstm_618/TensorArrayV2TensorListReserveVsequential_618/bidirectional_618/forward_lstm_618/TensorArrayV2/element_shape:output:0Jsequential_618/bidirectional_618/forward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?sequential_618/bidirectional_618/forward_lstm_618/TensorArrayV2£
gsequential_618/bidirectional_618/forward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2i
gsequential_618/bidirectional_618/forward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeј
Ysequential_618/bidirectional_618/forward_lstm_618/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor?sequential_618/bidirectional_618/forward_lstm_618/transpose:y:0psequential_618/bidirectional_618/forward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02[
Ysequential_618/bidirectional_618/forward_lstm_618/TensorArrayUnstack/TensorListFromTensor№
Gsequential_618/bidirectional_618/forward_lstm_618/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2I
Gsequential_618/bidirectional_618/forward_lstm_618/strided_slice_2/stackа
Isequential_618/bidirectional_618/forward_lstm_618/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_618/bidirectional_618/forward_lstm_618/strided_slice_2/stack_1а
Isequential_618/bidirectional_618/forward_lstm_618/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_618/bidirectional_618/forward_lstm_618/strided_slice_2/stack_2®
Asequential_618/bidirectional_618/forward_lstm_618/strided_slice_2StridedSlice?sequential_618/bidirectional_618/forward_lstm_618/transpose:y:0Psequential_618/bidirectional_618/forward_lstm_618/strided_slice_2/stack:output:0Rsequential_618/bidirectional_618/forward_lstm_618/strided_slice_2/stack_1:output:0Rsequential_618/bidirectional_618/forward_lstm_618/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2C
Asequential_618/bidirectional_618/forward_lstm_618/strided_slice_2—
Vsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOp_sequential_618_bidirectional_618_forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02X
Vsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpы
Gsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMulMatMulJsequential_618/bidirectional_618/forward_lstm_618/strided_slice_2:output:0^sequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2I
Gsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMul„
Xsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOpasequential_618_bidirectional_618_forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02Z
Xsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpч
Isequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMul_1MatMul@sequential_618/bidirectional_618/forward_lstm_618/zeros:output:0`sequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2K
Isequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMul_1р
Dsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/addAddV2Qsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMul:product:0Ssequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2F
Dsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/add–
Wsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOp`sequential_618_bidirectional_618_forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02Y
Wsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpэ
Hsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/BiasAddBiasAddHsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/add:z:0_sequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2J
Hsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/BiasAddж
Psequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2R
Psequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/split/split_dim√
Fsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/splitSplitYsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/split/split_dim:output:0Qsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2H
Fsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/splitҐ
Hsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/SigmoidSigmoidOsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22J
Hsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/Sigmoid¶
Jsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/Sigmoid_1SigmoidOsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22L
Jsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/Sigmoid_1ў
Dsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/mulMulNsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/Sigmoid_1:y:0Bsequential_618/bidirectional_618/forward_lstm_618/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22F
Dsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/mulЩ
Esequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/ReluReluOsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22G
Esequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/Reluм
Fsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/mul_1MulLsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/Sigmoid:y:0Ssequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22H
Fsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/mul_1б
Fsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/add_1AddV2Hsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/mul:z:0Jsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22H
Fsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/add_1¶
Jsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/Sigmoid_2SigmoidOsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22L
Jsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/Sigmoid_2Ш
Gsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/Relu_1ReluJsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22I
Gsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/Relu_1р
Fsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/mul_2MulNsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/Sigmoid_2:y:0Usequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22H
Fsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/mul_2у
Osequential_618/bidirectional_618/forward_lstm_618/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2Q
Osequential_618/bidirectional_618/forward_lstm_618/TensorArrayV2_1/element_shapeА
Asequential_618/bidirectional_618/forward_lstm_618/TensorArrayV2_1TensorListReserveXsequential_618/bidirectional_618/forward_lstm_618/TensorArrayV2_1/element_shape:output:0Jsequential_618/bidirectional_618/forward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02C
Asequential_618/bidirectional_618/forward_lstm_618/TensorArrayV2_1≤
6sequential_618/bidirectional_618/forward_lstm_618/timeConst*
_output_shapes
: *
dtype0*
value	B : 28
6sequential_618/bidirectional_618/forward_lstm_618/timeЗ
<sequential_618/bidirectional_618/forward_lstm_618/zeros_like	ZerosLikeJsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22>
<sequential_618/bidirectional_618/forward_lstm_618/zeros_likeг
Jsequential_618/bidirectional_618/forward_lstm_618/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2L
Jsequential_618/bidirectional_618/forward_lstm_618/while/maximum_iterationsќ
Dsequential_618/bidirectional_618/forward_lstm_618/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dsequential_618/bidirectional_618/forward_lstm_618/while/loop_counter«
7sequential_618/bidirectional_618/forward_lstm_618/whileWhileMsequential_618/bidirectional_618/forward_lstm_618/while/loop_counter:output:0Ssequential_618/bidirectional_618/forward_lstm_618/while/maximum_iterations:output:0?sequential_618/bidirectional_618/forward_lstm_618/time:output:0Jsequential_618/bidirectional_618/forward_lstm_618/TensorArrayV2_1:handle:0@sequential_618/bidirectional_618/forward_lstm_618/zeros_like:y:0@sequential_618/bidirectional_618/forward_lstm_618/zeros:output:0Bsequential_618/bidirectional_618/forward_lstm_618/zeros_1:output:0Jsequential_618/bidirectional_618/forward_lstm_618/strided_slice_1:output:0isequential_618/bidirectional_618/forward_lstm_618/TensorArrayUnstack/TensorListFromTensor:output_handle:0:sequential_618/bidirectional_618/forward_lstm_618/Cast:y:0_sequential_618_bidirectional_618_forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resourceasequential_618_bidirectional_618_forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource`sequential_618_bidirectional_618_forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource*
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
Esequential_618_bidirectional_618_forward_lstm_618_while_body_62324721*Q
condIRG
Esequential_618_bidirectional_618_forward_lstm_618_while_cond_62324720*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 29
7sequential_618/bidirectional_618/forward_lstm_618/whileЩ
bsequential_618/bidirectional_618/forward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2d
bsequential_618/bidirectional_618/forward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeє
Tsequential_618/bidirectional_618/forward_lstm_618/TensorArrayV2Stack/TensorListStackTensorListStack@sequential_618/bidirectional_618/forward_lstm_618/while:output:3ksequential_618/bidirectional_618/forward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02V
Tsequential_618/bidirectional_618/forward_lstm_618/TensorArrayV2Stack/TensorListStackе
Gsequential_618/bidirectional_618/forward_lstm_618/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2I
Gsequential_618/bidirectional_618/forward_lstm_618/strided_slice_3/stackа
Isequential_618/bidirectional_618/forward_lstm_618/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2K
Isequential_618/bidirectional_618/forward_lstm_618/strided_slice_3/stack_1а
Isequential_618/bidirectional_618/forward_lstm_618/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_618/bidirectional_618/forward_lstm_618/strided_slice_3/stack_2∆
Asequential_618/bidirectional_618/forward_lstm_618/strided_slice_3StridedSlice]sequential_618/bidirectional_618/forward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0Psequential_618/bidirectional_618/forward_lstm_618/strided_slice_3/stack:output:0Rsequential_618/bidirectional_618/forward_lstm_618/strided_slice_3/stack_1:output:0Rsequential_618/bidirectional_618/forward_lstm_618/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2C
Asequential_618/bidirectional_618/forward_lstm_618/strided_slice_3Ё
Bsequential_618/bidirectional_618/forward_lstm_618/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2D
Bsequential_618/bidirectional_618/forward_lstm_618/transpose_1/permц
=sequential_618/bidirectional_618/forward_lstm_618/transpose_1	Transpose]sequential_618/bidirectional_618/forward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0Ksequential_618/bidirectional_618/forward_lstm_618/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22?
=sequential_618/bidirectional_618/forward_lstm_618/transpose_1 
9sequential_618/bidirectional_618/forward_lstm_618/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2;
9sequential_618/bidirectional_618/forward_lstm_618/runtimeџ
Gsequential_618/bidirectional_618/backward_lstm_618/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2I
Gsequential_618/bidirectional_618/backward_lstm_618/RaggedToTensor/zerosЁ
Gsequential_618/bidirectional_618/backward_lstm_618/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2I
Gsequential_618/bidirectional_618/backward_lstm_618/RaggedToTensor/Const°
Vsequential_618/bidirectional_618/backward_lstm_618/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorPsequential_618/bidirectional_618/backward_lstm_618/RaggedToTensor/Const:output:0args_0Psequential_618/bidirectional_618/backward_lstm_618/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2X
Vsequential_618/bidirectional_618/backward_lstm_618/RaggedToTensor/RaggedTensorToTensorИ
]sequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2_
]sequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice/stackМ
_sequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2a
_sequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1М
_sequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2a
_sequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2”
Wsequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1fsequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack:output:0hsequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1:output:0hsequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask2Y
Wsequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_sliceМ
_sequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2a
_sequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stackЩ
asequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2c
asequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1Р
asequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2c
asequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2я
Ysequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1hsequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack:output:0jsequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0jsequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask2[
Ysequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice_1Щ
Msequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/subSub`sequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice:output:0bsequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2O
Msequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/subК
7sequential_618/bidirectional_618/backward_lstm_618/CastCastQsequential_618/bidirectional_618/backward_lstm_618/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€29
7sequential_618/bidirectional_618/backward_lstm_618/CastГ
8sequential_618/bidirectional_618/backward_lstm_618/ShapeShape_sequential_618/bidirectional_618/backward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2:
8sequential_618/bidirectional_618/backward_lstm_618/ShapeЏ
Fsequential_618/bidirectional_618/backward_lstm_618/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fsequential_618/bidirectional_618/backward_lstm_618/strided_slice/stackё
Hsequential_618/bidirectional_618/backward_lstm_618/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_618/bidirectional_618/backward_lstm_618/strided_slice/stack_1ё
Hsequential_618/bidirectional_618/backward_lstm_618/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_618/bidirectional_618/backward_lstm_618/strided_slice/stack_2Ф
@sequential_618/bidirectional_618/backward_lstm_618/strided_sliceStridedSliceAsequential_618/bidirectional_618/backward_lstm_618/Shape:output:0Osequential_618/bidirectional_618/backward_lstm_618/strided_slice/stack:output:0Qsequential_618/bidirectional_618/backward_lstm_618/strided_slice/stack_1:output:0Qsequential_618/bidirectional_618/backward_lstm_618/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2B
@sequential_618/bidirectional_618/backward_lstm_618/strided_slice¬
>sequential_618/bidirectional_618/backward_lstm_618/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22@
>sequential_618/bidirectional_618/backward_lstm_618/zeros/mul/yЄ
<sequential_618/bidirectional_618/backward_lstm_618/zeros/mulMulIsequential_618/bidirectional_618/backward_lstm_618/strided_slice:output:0Gsequential_618/bidirectional_618/backward_lstm_618/zeros/mul/y:output:0*
T0*
_output_shapes
: 2>
<sequential_618/bidirectional_618/backward_lstm_618/zeros/mul≈
?sequential_618/bidirectional_618/backward_lstm_618/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2A
?sequential_618/bidirectional_618/backward_lstm_618/zeros/Less/y≥
=sequential_618/bidirectional_618/backward_lstm_618/zeros/LessLess@sequential_618/bidirectional_618/backward_lstm_618/zeros/mul:z:0Hsequential_618/bidirectional_618/backward_lstm_618/zeros/Less/y:output:0*
T0*
_output_shapes
: 2?
=sequential_618/bidirectional_618/backward_lstm_618/zeros/Less»
Asequential_618/bidirectional_618/backward_lstm_618/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22C
Asequential_618/bidirectional_618/backward_lstm_618/zeros/packed/1ѕ
?sequential_618/bidirectional_618/backward_lstm_618/zeros/packedPackIsequential_618/bidirectional_618/backward_lstm_618/strided_slice:output:0Jsequential_618/bidirectional_618/backward_lstm_618/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2A
?sequential_618/bidirectional_618/backward_lstm_618/zeros/packed…
>sequential_618/bidirectional_618/backward_lstm_618/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2@
>sequential_618/bidirectional_618/backward_lstm_618/zeros/ConstЅ
8sequential_618/bidirectional_618/backward_lstm_618/zerosFillHsequential_618/bidirectional_618/backward_lstm_618/zeros/packed:output:0Gsequential_618/bidirectional_618/backward_lstm_618/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22:
8sequential_618/bidirectional_618/backward_lstm_618/zeros∆
@sequential_618/bidirectional_618/backward_lstm_618/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22B
@sequential_618/bidirectional_618/backward_lstm_618/zeros_1/mul/yЊ
>sequential_618/bidirectional_618/backward_lstm_618/zeros_1/mulMulIsequential_618/bidirectional_618/backward_lstm_618/strided_slice:output:0Isequential_618/bidirectional_618/backward_lstm_618/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2@
>sequential_618/bidirectional_618/backward_lstm_618/zeros_1/mul…
Asequential_618/bidirectional_618/backward_lstm_618/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2C
Asequential_618/bidirectional_618/backward_lstm_618/zeros_1/Less/yї
?sequential_618/bidirectional_618/backward_lstm_618/zeros_1/LessLessBsequential_618/bidirectional_618/backward_lstm_618/zeros_1/mul:z:0Jsequential_618/bidirectional_618/backward_lstm_618/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2A
?sequential_618/bidirectional_618/backward_lstm_618/zeros_1/Lessћ
Csequential_618/bidirectional_618/backward_lstm_618/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22E
Csequential_618/bidirectional_618/backward_lstm_618/zeros_1/packed/1’
Asequential_618/bidirectional_618/backward_lstm_618/zeros_1/packedPackIsequential_618/bidirectional_618/backward_lstm_618/strided_slice:output:0Lsequential_618/bidirectional_618/backward_lstm_618/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2C
Asequential_618/bidirectional_618/backward_lstm_618/zeros_1/packedЌ
@sequential_618/bidirectional_618/backward_lstm_618/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2B
@sequential_618/bidirectional_618/backward_lstm_618/zeros_1/Const…
:sequential_618/bidirectional_618/backward_lstm_618/zeros_1FillJsequential_618/bidirectional_618/backward_lstm_618/zeros_1/packed:output:0Isequential_618/bidirectional_618/backward_lstm_618/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22<
:sequential_618/bidirectional_618/backward_lstm_618/zeros_1џ
Asequential_618/bidirectional_618/backward_lstm_618/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2C
Asequential_618/bidirectional_618/backward_lstm_618/transpose/permх
<sequential_618/bidirectional_618/backward_lstm_618/transpose	Transpose_sequential_618/bidirectional_618/backward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0Jsequential_618/bidirectional_618/backward_lstm_618/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2>
<sequential_618/bidirectional_618/backward_lstm_618/transposeи
:sequential_618/bidirectional_618/backward_lstm_618/Shape_1Shape@sequential_618/bidirectional_618/backward_lstm_618/transpose:y:0*
T0*
_output_shapes
:2<
:sequential_618/bidirectional_618/backward_lstm_618/Shape_1ё
Hsequential_618/bidirectional_618/backward_lstm_618/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_618/bidirectional_618/backward_lstm_618/strided_slice_1/stackв
Jsequential_618/bidirectional_618/backward_lstm_618/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_618/bidirectional_618/backward_lstm_618/strided_slice_1/stack_1в
Jsequential_618/bidirectional_618/backward_lstm_618/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_618/bidirectional_618/backward_lstm_618/strided_slice_1/stack_2†
Bsequential_618/bidirectional_618/backward_lstm_618/strided_slice_1StridedSliceCsequential_618/bidirectional_618/backward_lstm_618/Shape_1:output:0Qsequential_618/bidirectional_618/backward_lstm_618/strided_slice_1/stack:output:0Ssequential_618/bidirectional_618/backward_lstm_618/strided_slice_1/stack_1:output:0Ssequential_618/bidirectional_618/backward_lstm_618/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2D
Bsequential_618/bidirectional_618/backward_lstm_618/strided_slice_1л
Nsequential_618/bidirectional_618/backward_lstm_618/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2P
Nsequential_618/bidirectional_618/backward_lstm_618/TensorArrayV2/element_shapeю
@sequential_618/bidirectional_618/backward_lstm_618/TensorArrayV2TensorListReserveWsequential_618/bidirectional_618/backward_lstm_618/TensorArrayV2/element_shape:output:0Ksequential_618/bidirectional_618/backward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02B
@sequential_618/bidirectional_618/backward_lstm_618/TensorArrayV2–
Asequential_618/bidirectional_618/backward_lstm_618/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential_618/bidirectional_618/backward_lstm_618/ReverseV2/axis÷
<sequential_618/bidirectional_618/backward_lstm_618/ReverseV2	ReverseV2@sequential_618/bidirectional_618/backward_lstm_618/transpose:y:0Jsequential_618/bidirectional_618/backward_lstm_618/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2>
<sequential_618/bidirectional_618/backward_lstm_618/ReverseV2•
hsequential_618/bidirectional_618/backward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2j
hsequential_618/bidirectional_618/backward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape…
Zsequential_618/bidirectional_618/backward_lstm_618/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorEsequential_618/bidirectional_618/backward_lstm_618/ReverseV2:output:0qsequential_618/bidirectional_618/backward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02\
Zsequential_618/bidirectional_618/backward_lstm_618/TensorArrayUnstack/TensorListFromTensorё
Hsequential_618/bidirectional_618/backward_lstm_618/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_618/bidirectional_618/backward_lstm_618/strided_slice_2/stackв
Jsequential_618/bidirectional_618/backward_lstm_618/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_618/bidirectional_618/backward_lstm_618/strided_slice_2/stack_1в
Jsequential_618/bidirectional_618/backward_lstm_618/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_618/bidirectional_618/backward_lstm_618/strided_slice_2/stack_2Ѓ
Bsequential_618/bidirectional_618/backward_lstm_618/strided_slice_2StridedSlice@sequential_618/bidirectional_618/backward_lstm_618/transpose:y:0Qsequential_618/bidirectional_618/backward_lstm_618/strided_slice_2/stack:output:0Ssequential_618/bidirectional_618/backward_lstm_618/strided_slice_2/stack_1:output:0Ssequential_618/bidirectional_618/backward_lstm_618/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2D
Bsequential_618/bidirectional_618/backward_lstm_618/strided_slice_2‘
Wsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOp`sequential_618_bidirectional_618_backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02Y
Wsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp€
Hsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMulMatMulKsequential_618/bidirectional_618/backward_lstm_618/strided_slice_2:output:0_sequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2J
Hsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMulЏ
Ysequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOpbsequential_618_bidirectional_618_backward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02[
Ysequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpы
Jsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMul_1MatMulAsequential_618/bidirectional_618/backward_lstm_618/zeros:output:0asequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2L
Jsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMul_1ф
Esequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/addAddV2Rsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMul:product:0Tsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2G
Esequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/add”
Xsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOpasequential_618_bidirectional_618_backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02Z
Xsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpБ
Isequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/BiasAddBiasAddIsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/add:z:0`sequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2K
Isequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/BiasAddи
Qsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2S
Qsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/split/split_dim«
Gsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/splitSplitZsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/split/split_dim:output:0Rsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2I
Gsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/split•
Isequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/SigmoidSigmoidPsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22K
Isequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/Sigmoid©
Ksequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/Sigmoid_1SigmoidPsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22M
Ksequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/Sigmoid_1Ё
Esequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/mulMulOsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/Sigmoid_1:y:0Csequential_618/bidirectional_618/backward_lstm_618/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22G
Esequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/mulЬ
Fsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/ReluReluPsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22H
Fsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/Reluр
Gsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/mul_1MulMsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/Sigmoid:y:0Tsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22I
Gsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/mul_1е
Gsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/add_1AddV2Isequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/mul:z:0Ksequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22I
Gsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/add_1©
Ksequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/Sigmoid_2SigmoidPsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22M
Ksequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/Sigmoid_2Ы
Hsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/Relu_1ReluKsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22J
Hsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/Relu_1ф
Gsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/mul_2MulOsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/Sigmoid_2:y:0Vsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22I
Gsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/mul_2х
Psequential_618/bidirectional_618/backward_lstm_618/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2R
Psequential_618/bidirectional_618/backward_lstm_618/TensorArrayV2_1/element_shapeД
Bsequential_618/bidirectional_618/backward_lstm_618/TensorArrayV2_1TensorListReserveYsequential_618/bidirectional_618/backward_lstm_618/TensorArrayV2_1/element_shape:output:0Ksequential_618/bidirectional_618/backward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02D
Bsequential_618/bidirectional_618/backward_lstm_618/TensorArrayV2_1і
7sequential_618/bidirectional_618/backward_lstm_618/timeConst*
_output_shapes
: *
dtype0*
value	B : 29
7sequential_618/bidirectional_618/backward_lstm_618/time÷
Hsequential_618/bidirectional_618/backward_lstm_618/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hsequential_618/bidirectional_618/backward_lstm_618/Max/reduction_indices®
6sequential_618/bidirectional_618/backward_lstm_618/MaxMax;sequential_618/bidirectional_618/backward_lstm_618/Cast:y:0Qsequential_618/bidirectional_618/backward_lstm_618/Max/reduction_indices:output:0*
T0*
_output_shapes
: 28
6sequential_618/bidirectional_618/backward_lstm_618/Maxґ
8sequential_618/bidirectional_618/backward_lstm_618/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_618/bidirectional_618/backward_lstm_618/sub/yЬ
6sequential_618/bidirectional_618/backward_lstm_618/subSub?sequential_618/bidirectional_618/backward_lstm_618/Max:output:0Asequential_618/bidirectional_618/backward_lstm_618/sub/y:output:0*
T0*
_output_shapes
: 28
6sequential_618/bidirectional_618/backward_lstm_618/subҐ
8sequential_618/bidirectional_618/backward_lstm_618/Sub_1Sub:sequential_618/bidirectional_618/backward_lstm_618/sub:z:0;sequential_618/bidirectional_618/backward_lstm_618/Cast:y:0*
T0*#
_output_shapes
:€€€€€€€€€2:
8sequential_618/bidirectional_618/backward_lstm_618/Sub_1К
=sequential_618/bidirectional_618/backward_lstm_618/zeros_like	ZerosLikeKsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22?
=sequential_618/bidirectional_618/backward_lstm_618/zeros_likeе
Ksequential_618/bidirectional_618/backward_lstm_618/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2M
Ksequential_618/bidirectional_618/backward_lstm_618/while/maximum_iterations–
Esequential_618/bidirectional_618/backward_lstm_618/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2G
Esequential_618/bidirectional_618/backward_lstm_618/while/loop_counterў
8sequential_618/bidirectional_618/backward_lstm_618/whileWhileNsequential_618/bidirectional_618/backward_lstm_618/while/loop_counter:output:0Tsequential_618/bidirectional_618/backward_lstm_618/while/maximum_iterations:output:0@sequential_618/bidirectional_618/backward_lstm_618/time:output:0Ksequential_618/bidirectional_618/backward_lstm_618/TensorArrayV2_1:handle:0Asequential_618/bidirectional_618/backward_lstm_618/zeros_like:y:0Asequential_618/bidirectional_618/backward_lstm_618/zeros:output:0Csequential_618/bidirectional_618/backward_lstm_618/zeros_1:output:0Ksequential_618/bidirectional_618/backward_lstm_618/strided_slice_1:output:0jsequential_618/bidirectional_618/backward_lstm_618/TensorArrayUnstack/TensorListFromTensor:output_handle:0<sequential_618/bidirectional_618/backward_lstm_618/Sub_1:z:0`sequential_618_bidirectional_618_backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resourcebsequential_618_bidirectional_618_backward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resourceasequential_618_bidirectional_618_backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource*
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
Fsequential_618_bidirectional_618_backward_lstm_618_while_body_62324900*R
condJRH
Fsequential_618_bidirectional_618_backward_lstm_618_while_cond_62324899*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2:
8sequential_618/bidirectional_618/backward_lstm_618/whileЫ
csequential_618/bidirectional_618/backward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2e
csequential_618/bidirectional_618/backward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeљ
Usequential_618/bidirectional_618/backward_lstm_618/TensorArrayV2Stack/TensorListStackTensorListStackAsequential_618/bidirectional_618/backward_lstm_618/while:output:3lsequential_618/bidirectional_618/backward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype02W
Usequential_618/bidirectional_618/backward_lstm_618/TensorArrayV2Stack/TensorListStackз
Hsequential_618/bidirectional_618/backward_lstm_618/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2J
Hsequential_618/bidirectional_618/backward_lstm_618/strided_slice_3/stackв
Jsequential_618/bidirectional_618/backward_lstm_618/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jsequential_618/bidirectional_618/backward_lstm_618/strided_slice_3/stack_1в
Jsequential_618/bidirectional_618/backward_lstm_618/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_618/bidirectional_618/backward_lstm_618/strided_slice_3/stack_2ћ
Bsequential_618/bidirectional_618/backward_lstm_618/strided_slice_3StridedSlice^sequential_618/bidirectional_618/backward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0Qsequential_618/bidirectional_618/backward_lstm_618/strided_slice_3/stack:output:0Ssequential_618/bidirectional_618/backward_lstm_618/strided_slice_3/stack_1:output:0Ssequential_618/bidirectional_618/backward_lstm_618/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2D
Bsequential_618/bidirectional_618/backward_lstm_618/strided_slice_3я
Csequential_618/bidirectional_618/backward_lstm_618/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2E
Csequential_618/bidirectional_618/backward_lstm_618/transpose_1/permъ
>sequential_618/bidirectional_618/backward_lstm_618/transpose_1	Transpose^sequential_618/bidirectional_618/backward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0Lsequential_618/bidirectional_618/backward_lstm_618/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22@
>sequential_618/bidirectional_618/backward_lstm_618/transpose_1ћ
:sequential_618/bidirectional_618/backward_lstm_618/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2<
:sequential_618/bidirectional_618/backward_lstm_618/runtimeЮ
,sequential_618/bidirectional_618/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_618/bidirectional_618/concat/axisй
'sequential_618/bidirectional_618/concatConcatV2Jsequential_618/bidirectional_618/forward_lstm_618/strided_slice_3:output:0Ksequential_618/bidirectional_618/backward_lstm_618/strided_slice_3:output:05sequential_618/bidirectional_618/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€d2)
'sequential_618/bidirectional_618/concatЎ
.sequential_618/dense_618/MatMul/ReadVariableOpReadVariableOp7sequential_618_dense_618_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_618/dense_618/MatMul/ReadVariableOpи
sequential_618/dense_618/MatMulMatMul0sequential_618/bidirectional_618/concat:output:06sequential_618/dense_618/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2!
sequential_618/dense_618/MatMul„
/sequential_618/dense_618/BiasAdd/ReadVariableOpReadVariableOp8sequential_618_dense_618_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_618/dense_618/BiasAdd/ReadVariableOpе
 sequential_618/dense_618/BiasAddBiasAdd)sequential_618/dense_618/MatMul:product:07sequential_618/dense_618/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2"
 sequential_618/dense_618/BiasAddђ
 sequential_618/dense_618/SigmoidSigmoid)sequential_618/dense_618/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2"
 sequential_618/dense_618/Sigmoid
IdentityIdentity$sequential_618/dense_618/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity≈
NoOpNoOpY^sequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpX^sequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpZ^sequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp9^sequential_618/bidirectional_618/backward_lstm_618/whileX^sequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpW^sequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpY^sequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp8^sequential_618/bidirectional_618/forward_lstm_618/while0^sequential_618/dense_618/BiasAdd/ReadVariableOp/^sequential_618/dense_618/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€:€€€€€€€€€: : : : : : : : 2і
Xsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpXsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp2≤
Wsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpWsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp2ґ
Ysequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpYsequential_618/bidirectional_618/backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp2t
8sequential_618/bidirectional_618/backward_lstm_618/while8sequential_618/bidirectional_618/backward_lstm_618/while2≤
Wsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpWsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp2∞
Vsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpVsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp2і
Xsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpXsequential_618/bidirectional_618/forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp2r
7sequential_618/bidirectional_618/forward_lstm_618/while7sequential_618/bidirectional_618/forward_lstm_618/while2b
/sequential_618/dense_618/BiasAdd/ReadVariableOp/sequential_618/dense_618/BiasAdd/ReadVariableOp2`
.sequential_618/dense_618/MatMul/ReadVariableOp.sequential_618/dense_618/MatMul/ReadVariableOp:O K
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
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_62327440

inputs
inputs_1	Q
>forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource:	»S
@forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource:	2»N
?forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource:	»R
?backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resource:	»T
Abackward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource:	2»O
@backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource:	»
identityИҐ7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpҐ6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpҐ8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpҐbackward_lstm_618/whileҐ6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpҐ5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpҐ7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpҐforward_lstm_618/whileЧ
%forward_lstm_618/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%forward_lstm_618/RaggedToTensor/zerosЩ
%forward_lstm_618/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2'
%forward_lstm_618/RaggedToTensor/ConstЩ
4forward_lstm_618/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.forward_lstm_618/RaggedToTensor/Const:output:0inputs.forward_lstm_618/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4forward_lstm_618/RaggedToTensor/RaggedTensorToTensorƒ
;forward_lstm_618/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack»
=forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1»
=forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2©
5forward_lstm_618/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dforward_lstm_618/RaggedNestedRowLengths/strided_slice/stack:output:0Fforward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fforward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask27
5forward_lstm_618/RaggedNestedRowLengths/strided_slice»
=forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack’
?forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2A
?forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1ћ
?forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?forward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2µ
7forward_lstm_618/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fforward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hforward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hforward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask29
7forward_lstm_618/RaggedNestedRowLengths/strided_slice_1С
+forward_lstm_618/RaggedNestedRowLengths/subSub>forward_lstm_618/RaggedNestedRowLengths/strided_slice:output:0@forward_lstm_618/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2-
+forward_lstm_618/RaggedNestedRowLengths/sub§
forward_lstm_618/CastCast/forward_lstm_618/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
forward_lstm_618/CastЭ
forward_lstm_618/ShapeShape=forward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_618/ShapeЦ
$forward_lstm_618/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_618/strided_slice/stackЪ
&forward_lstm_618/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_618/strided_slice/stack_1Ъ
&forward_lstm_618/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_618/strided_slice/stack_2»
forward_lstm_618/strided_sliceStridedSliceforward_lstm_618/Shape:output:0-forward_lstm_618/strided_slice/stack:output:0/forward_lstm_618/strided_slice/stack_1:output:0/forward_lstm_618/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_618/strided_slice~
forward_lstm_618/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_618/zeros/mul/y∞
forward_lstm_618/zeros/mulMul'forward_lstm_618/strided_slice:output:0%forward_lstm_618/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros/mulБ
forward_lstm_618/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
forward_lstm_618/zeros/Less/yЂ
forward_lstm_618/zeros/LessLessforward_lstm_618/zeros/mul:z:0&forward_lstm_618/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros/LessД
forward_lstm_618/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_618/zeros/packed/1«
forward_lstm_618/zeros/packedPack'forward_lstm_618/strided_slice:output:0(forward_lstm_618/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_618/zeros/packedЕ
forward_lstm_618/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_618/zeros/Constє
forward_lstm_618/zerosFill&forward_lstm_618/zeros/packed:output:0%forward_lstm_618/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/zerosВ
forward_lstm_618/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_618/zeros_1/mul/yґ
forward_lstm_618/zeros_1/mulMul'forward_lstm_618/strided_slice:output:0'forward_lstm_618/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros_1/mulЕ
forward_lstm_618/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2!
forward_lstm_618/zeros_1/Less/y≥
forward_lstm_618/zeros_1/LessLess forward_lstm_618/zeros_1/mul:z:0(forward_lstm_618/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/zeros_1/LessИ
!forward_lstm_618/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!forward_lstm_618/zeros_1/packed/1Ќ
forward_lstm_618/zeros_1/packedPack'forward_lstm_618/strided_slice:output:0*forward_lstm_618/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_lstm_618/zeros_1/packedЙ
forward_lstm_618/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
forward_lstm_618/zeros_1/ConstЅ
forward_lstm_618/zeros_1Fill(forward_lstm_618/zeros_1/packed:output:0'forward_lstm_618/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/zeros_1Ч
forward_lstm_618/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_618/transpose/permн
forward_lstm_618/transpose	Transpose=forward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0(forward_lstm_618/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
forward_lstm_618/transposeВ
forward_lstm_618/Shape_1Shapeforward_lstm_618/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_618/Shape_1Ъ
&forward_lstm_618/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_618/strided_slice_1/stackЮ
(forward_lstm_618/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_1/stack_1Ю
(forward_lstm_618/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_1/stack_2‘
 forward_lstm_618/strided_slice_1StridedSlice!forward_lstm_618/Shape_1:output:0/forward_lstm_618/strided_slice_1/stack:output:01forward_lstm_618/strided_slice_1/stack_1:output:01forward_lstm_618/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_lstm_618/strided_slice_1І
,forward_lstm_618/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,forward_lstm_618/TensorArrayV2/element_shapeц
forward_lstm_618/TensorArrayV2TensorListReserve5forward_lstm_618/TensorArrayV2/element_shape:output:0)forward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_618/TensorArrayV2б
Fforward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2H
Fforward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeЉ
8forward_lstm_618/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_618/transpose:y:0Oforward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8forward_lstm_618/TensorArrayUnstack/TensorListFromTensorЪ
&forward_lstm_618/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_618/strided_slice_2/stackЮ
(forward_lstm_618/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_2/stack_1Ю
(forward_lstm_618/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_2/stack_2в
 forward_lstm_618/strided_slice_2StridedSliceforward_lstm_618/transpose:y:0/forward_lstm_618/strided_slice_2/stack:output:01forward_lstm_618/strided_slice_2/stack_1:output:01forward_lstm_618/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2"
 forward_lstm_618/strided_slice_2о
5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOp>forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype027
5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOpч
&forward_lstm_618/lstm_cell_1855/MatMulMatMul)forward_lstm_618/strided_slice_2:output:0=forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2(
&forward_lstm_618/lstm_cell_1855/MatMulф
7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOp@forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype029
7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOpу
(forward_lstm_618/lstm_cell_1855/MatMul_1MatMulforward_lstm_618/zeros:output:0?forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(forward_lstm_618/lstm_cell_1855/MatMul_1м
#forward_lstm_618/lstm_cell_1855/addAddV20forward_lstm_618/lstm_cell_1855/MatMul:product:02forward_lstm_618/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2%
#forward_lstm_618/lstm_cell_1855/addн
6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOp?forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype028
6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOpщ
'forward_lstm_618/lstm_cell_1855/BiasAddBiasAdd'forward_lstm_618/lstm_cell_1855/add:z:0>forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'forward_lstm_618/lstm_cell_1855/BiasAdd§
/forward_lstm_618/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/forward_lstm_618/lstm_cell_1855/split/split_dimњ
%forward_lstm_618/lstm_cell_1855/splitSplit8forward_lstm_618/lstm_cell_1855/split/split_dim:output:00forward_lstm_618/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2'
%forward_lstm_618/lstm_cell_1855/splitњ
'forward_lstm_618/lstm_cell_1855/SigmoidSigmoid.forward_lstm_618/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'forward_lstm_618/lstm_cell_1855/Sigmoid√
)forward_lstm_618/lstm_cell_1855/Sigmoid_1Sigmoid.forward_lstm_618/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_618/lstm_cell_1855/Sigmoid_1’
#forward_lstm_618/lstm_cell_1855/mulMul-forward_lstm_618/lstm_cell_1855/Sigmoid_1:y:0!forward_lstm_618/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22%
#forward_lstm_618/lstm_cell_1855/mulґ
$forward_lstm_618/lstm_cell_1855/ReluRelu.forward_lstm_618/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22&
$forward_lstm_618/lstm_cell_1855/Reluи
%forward_lstm_618/lstm_cell_1855/mul_1Mul+forward_lstm_618/lstm_cell_1855/Sigmoid:y:02forward_lstm_618/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_618/lstm_cell_1855/mul_1Ё
%forward_lstm_618/lstm_cell_1855/add_1AddV2'forward_lstm_618/lstm_cell_1855/mul:z:0)forward_lstm_618/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_618/lstm_cell_1855/add_1√
)forward_lstm_618/lstm_cell_1855/Sigmoid_2Sigmoid.forward_lstm_618/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_618/lstm_cell_1855/Sigmoid_2µ
&forward_lstm_618/lstm_cell_1855/Relu_1Relu)forward_lstm_618/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&forward_lstm_618/lstm_cell_1855/Relu_1м
%forward_lstm_618/lstm_cell_1855/mul_2Mul-forward_lstm_618/lstm_cell_1855/Sigmoid_2:y:04forward_lstm_618/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%forward_lstm_618/lstm_cell_1855/mul_2±
.forward_lstm_618/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   20
.forward_lstm_618/TensorArrayV2_1/element_shapeь
 forward_lstm_618/TensorArrayV2_1TensorListReserve7forward_lstm_618/TensorArrayV2_1/element_shape:output:0)forward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_lstm_618/TensorArrayV2_1p
forward_lstm_618/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_618/time§
forward_lstm_618/zeros_like	ZerosLike)forward_lstm_618/lstm_cell_1855/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/zeros_like°
)forward_lstm_618/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2+
)forward_lstm_618/while/maximum_iterationsМ
#forward_lstm_618/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#forward_lstm_618/while/loop_counterЦ	
forward_lstm_618/whileWhile,forward_lstm_618/while/loop_counter:output:02forward_lstm_618/while/maximum_iterations:output:0forward_lstm_618/time:output:0)forward_lstm_618/TensorArrayV2_1:handle:0forward_lstm_618/zeros_like:y:0forward_lstm_618/zeros:output:0!forward_lstm_618/zeros_1:output:0)forward_lstm_618/strided_slice_1:output:0Hforward_lstm_618/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_618/Cast:y:0>forward_lstm_618_lstm_cell_1855_matmul_readvariableop_resource@forward_lstm_618_lstm_cell_1855_matmul_1_readvariableop_resource?forward_lstm_618_lstm_cell_1855_biasadd_readvariableop_resource*
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
$forward_lstm_618_while_body_62327164*0
cond(R&
$forward_lstm_618_while_cond_62327163*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
forward_lstm_618/while„
Aforward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2C
Aforward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeµ
3forward_lstm_618/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_618/while:output:3Jforward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype025
3forward_lstm_618/TensorArrayV2Stack/TensorListStack£
&forward_lstm_618/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2(
&forward_lstm_618/strided_slice_3/stackЮ
(forward_lstm_618/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_lstm_618/strided_slice_3/stack_1Ю
(forward_lstm_618/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_lstm_618/strided_slice_3/stack_2А
 forward_lstm_618/strided_slice_3StridedSlice<forward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0/forward_lstm_618/strided_slice_3/stack:output:01forward_lstm_618/strided_slice_3/stack_1:output:01forward_lstm_618/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2"
 forward_lstm_618/strided_slice_3Ы
!forward_lstm_618/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_lstm_618/transpose_1/permт
forward_lstm_618/transpose_1	Transpose<forward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0*forward_lstm_618/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
forward_lstm_618/transpose_1И
forward_lstm_618/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_618/runtimeЩ
&backward_lstm_618/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2(
&backward_lstm_618/RaggedToTensor/zerosЫ
&backward_lstm_618/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€2(
&backward_lstm_618/RaggedToTensor/ConstЭ
5backward_lstm_618/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor/backward_lstm_618/RaggedToTensor/Const:output:0inputs/backward_lstm_618/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS27
5backward_lstm_618/RaggedToTensor/RaggedTensorToTensor∆
<backward_lstm_618/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack 
>backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1 
>backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2Ѓ
6backward_lstm_618/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Ebackward_lstm_618/RaggedNestedRowLengths/strided_slice/stack:output:0Gbackward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_1:output:0Gbackward_lstm_618/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
end_mask28
6backward_lstm_618/RaggedNestedRowLengths/strided_slice 
>backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack„
@backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2B
@backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1ќ
@backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@backward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2Ї
8backward_lstm_618/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Gbackward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack:output:0Ibackward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Ibackward_lstm_618/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask2:
8backward_lstm_618/RaggedNestedRowLengths/strided_slice_1Х
,backward_lstm_618/RaggedNestedRowLengths/subSub?backward_lstm_618/RaggedNestedRowLengths/strided_slice:output:0Abackward_lstm_618/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:€€€€€€€€€2.
,backward_lstm_618/RaggedNestedRowLengths/subІ
backward_lstm_618/CastCast0backward_lstm_618/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
backward_lstm_618/Cast†
backward_lstm_618/ShapeShape>backward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_618/ShapeШ
%backward_lstm_618/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_618/strided_slice/stackЬ
'backward_lstm_618/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_618/strided_slice/stack_1Ь
'backward_lstm_618/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_618/strided_slice/stack_2ќ
backward_lstm_618/strided_sliceStridedSlice backward_lstm_618/Shape:output:0.backward_lstm_618/strided_slice/stack:output:00backward_lstm_618/strided_slice/stack_1:output:00backward_lstm_618/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_618/strided_sliceА
backward_lstm_618/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_618/zeros/mul/yі
backward_lstm_618/zeros/mulMul(backward_lstm_618/strided_slice:output:0&backward_lstm_618/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/zeros/mulГ
backward_lstm_618/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2 
backward_lstm_618/zeros/Less/yѓ
backward_lstm_618/zeros/LessLessbackward_lstm_618/zeros/mul:z:0'backward_lstm_618/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/zeros/LessЖ
 backward_lstm_618/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_618/zeros/packed/1Ћ
backward_lstm_618/zeros/packedPack(backward_lstm_618/strided_slice:output:0)backward_lstm_618/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_618/zeros/packedЗ
backward_lstm_618/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_618/zeros/Constљ
backward_lstm_618/zerosFill'backward_lstm_618/zeros/packed:output:0&backward_lstm_618/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_618/zerosД
backward_lstm_618/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_618/zeros_1/mul/yЇ
backward_lstm_618/zeros_1/mulMul(backward_lstm_618/strided_slice:output:0(backward_lstm_618/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/zeros_1/mulЗ
 backward_lstm_618/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 backward_lstm_618/zeros_1/Less/yЈ
backward_lstm_618/zeros_1/LessLess!backward_lstm_618/zeros_1/mul:z:0)backward_lstm_618/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_lstm_618/zeros_1/LessК
"backward_lstm_618/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"backward_lstm_618/zeros_1/packed/1—
 backward_lstm_618/zeros_1/packedPack(backward_lstm_618/strided_slice:output:0+backward_lstm_618/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_lstm_618/zeros_1/packedЛ
backward_lstm_618/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2!
backward_lstm_618/zeros_1/Const≈
backward_lstm_618/zeros_1Fill)backward_lstm_618/zeros_1/packed:output:0(backward_lstm_618/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_618/zeros_1Щ
 backward_lstm_618/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_618/transpose/permс
backward_lstm_618/transpose	Transpose>backward_lstm_618/RaggedToTensor/RaggedTensorToTensor:result:0)backward_lstm_618/transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_618/transposeЕ
backward_lstm_618/Shape_1Shapebackward_lstm_618/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_618/Shape_1Ь
'backward_lstm_618/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_618/strided_slice_1/stack†
)backward_lstm_618/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_1/stack_1†
)backward_lstm_618/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_1/stack_2Џ
!backward_lstm_618/strided_slice_1StridedSlice"backward_lstm_618/Shape_1:output:00backward_lstm_618/strided_slice_1/stack:output:02backward_lstm_618/strided_slice_1/stack_1:output:02backward_lstm_618/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_lstm_618/strided_slice_1©
-backward_lstm_618/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-backward_lstm_618/TensorArrayV2/element_shapeъ
backward_lstm_618/TensorArrayV2TensorListReserve6backward_lstm_618/TensorArrayV2/element_shape:output:0*backward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_618/TensorArrayV2О
 backward_lstm_618/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2"
 backward_lstm_618/ReverseV2/axis“
backward_lstm_618/ReverseV2	ReverseV2backward_lstm_618/transpose:y:0)backward_lstm_618/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
backward_lstm_618/ReverseV2г
Gbackward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2I
Gbackward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape≈
9backward_lstm_618/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$backward_lstm_618/ReverseV2:output:0Pbackward_lstm_618/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9backward_lstm_618/TensorArrayUnstack/TensorListFromTensorЬ
'backward_lstm_618/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_618/strided_slice_2/stack†
)backward_lstm_618/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_2/stack_1†
)backward_lstm_618/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_2/stack_2и
!backward_lstm_618/strided_slice_2StridedSlicebackward_lstm_618/transpose:y:00backward_lstm_618/strided_slice_2/stack:output:02backward_lstm_618/strided_slice_2/stack_1:output:02backward_lstm_618/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask2#
!backward_lstm_618/strided_slice_2с
6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOp?backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype028
6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOpы
'backward_lstm_618/lstm_cell_1856/MatMulMatMul*backward_lstm_618/strided_slice_2:output:0>backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2)
'backward_lstm_618/lstm_cell_1856/MatMulч
8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOpAbackward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02:
8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOpч
)backward_lstm_618/lstm_cell_1856/MatMul_1MatMul backward_lstm_618/zeros:output:0@backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)backward_lstm_618/lstm_cell_1856/MatMul_1р
$backward_lstm_618/lstm_cell_1856/addAddV21backward_lstm_618/lstm_cell_1856/MatMul:product:03backward_lstm_618/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2&
$backward_lstm_618/lstm_cell_1856/addр
7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOp@backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype029
7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOpэ
(backward_lstm_618/lstm_cell_1856/BiasAddBiasAdd(backward_lstm_618/lstm_cell_1856/add:z:0?backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2*
(backward_lstm_618/lstm_cell_1856/BiasAdd¶
0backward_lstm_618/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0backward_lstm_618/lstm_cell_1856/split/split_dim√
&backward_lstm_618/lstm_cell_1856/splitSplit9backward_lstm_618/lstm_cell_1856/split/split_dim:output:01backward_lstm_618/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2(
&backward_lstm_618/lstm_cell_1856/split¬
(backward_lstm_618/lstm_cell_1856/SigmoidSigmoid/backward_lstm_618/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(backward_lstm_618/lstm_cell_1856/Sigmoid∆
*backward_lstm_618/lstm_cell_1856/Sigmoid_1Sigmoid/backward_lstm_618/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_618/lstm_cell_1856/Sigmoid_1ў
$backward_lstm_618/lstm_cell_1856/mulMul.backward_lstm_618/lstm_cell_1856/Sigmoid_1:y:0"backward_lstm_618/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$backward_lstm_618/lstm_cell_1856/mulє
%backward_lstm_618/lstm_cell_1856/ReluRelu/backward_lstm_618/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22'
%backward_lstm_618/lstm_cell_1856/Reluм
&backward_lstm_618/lstm_cell_1856/mul_1Mul,backward_lstm_618/lstm_cell_1856/Sigmoid:y:03backward_lstm_618/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_618/lstm_cell_1856/mul_1б
&backward_lstm_618/lstm_cell_1856/add_1AddV2(backward_lstm_618/lstm_cell_1856/mul:z:0*backward_lstm_618/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_618/lstm_cell_1856/add_1∆
*backward_lstm_618/lstm_cell_1856/Sigmoid_2Sigmoid/backward_lstm_618/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_618/lstm_cell_1856/Sigmoid_2Є
'backward_lstm_618/lstm_cell_1856/Relu_1Relu*backward_lstm_618/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22)
'backward_lstm_618/lstm_cell_1856/Relu_1р
&backward_lstm_618/lstm_cell_1856/mul_2Mul.backward_lstm_618/lstm_cell_1856/Sigmoid_2:y:05backward_lstm_618/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&backward_lstm_618/lstm_cell_1856/mul_2≥
/backward_lstm_618/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   21
/backward_lstm_618/TensorArrayV2_1/element_shapeА
!backward_lstm_618/TensorArrayV2_1TensorListReserve8backward_lstm_618/TensorArrayV2_1/element_shape:output:0*backward_lstm_618/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_lstm_618/TensorArrayV2_1r
backward_lstm_618/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_618/timeФ
'backward_lstm_618/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2)
'backward_lstm_618/Max/reduction_indices§
backward_lstm_618/MaxMaxbackward_lstm_618/Cast:y:00backward_lstm_618/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/Maxt
backward_lstm_618/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_618/sub/yШ
backward_lstm_618/subSubbackward_lstm_618/Max:output:0 backward_lstm_618/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/subЮ
backward_lstm_618/Sub_1Subbackward_lstm_618/sub:z:0backward_lstm_618/Cast:y:0*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_618/Sub_1І
backward_lstm_618/zeros_like	ZerosLike*backward_lstm_618/lstm_cell_1856/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
backward_lstm_618/zeros_like£
*backward_lstm_618/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2,
*backward_lstm_618/while/maximum_iterationsО
$backward_lstm_618/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$backward_lstm_618/while/loop_counter®	
backward_lstm_618/whileWhile-backward_lstm_618/while/loop_counter:output:03backward_lstm_618/while/maximum_iterations:output:0backward_lstm_618/time:output:0*backward_lstm_618/TensorArrayV2_1:handle:0 backward_lstm_618/zeros_like:y:0 backward_lstm_618/zeros:output:0"backward_lstm_618/zeros_1:output:0*backward_lstm_618/strided_slice_1:output:0Ibackward_lstm_618/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_618/Sub_1:z:0?backward_lstm_618_lstm_cell_1856_matmul_readvariableop_resourceAbackward_lstm_618_lstm_cell_1856_matmul_1_readvariableop_resource@backward_lstm_618_lstm_cell_1856_biasadd_readvariableop_resource*
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
%backward_lstm_618_while_body_62327343*1
cond)R'
%backward_lstm_618_while_cond_62327342*m
output_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : *
parallel_iterations 2
backward_lstm_618/whileў
Bbackward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2D
Bbackward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shapeє
4backward_lstm_618/TensorArrayV2Stack/TensorListStackTensorListStack backward_lstm_618/while:output:3Kbackward_lstm_618/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2*
element_dtype026
4backward_lstm_618/TensorArrayV2Stack/TensorListStack•
'backward_lstm_618/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2)
'backward_lstm_618/strided_slice_3/stack†
)backward_lstm_618/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_lstm_618/strided_slice_3/stack_1†
)backward_lstm_618/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_lstm_618/strided_slice_3/stack_2Ж
!backward_lstm_618/strided_slice_3StridedSlice=backward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:00backward_lstm_618/strided_slice_3/stack:output:02backward_lstm_618/strided_slice_3/stack_1:output:02backward_lstm_618/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€2*
shrink_axis_mask2#
!backward_lstm_618/strided_slice_3Э
"backward_lstm_618/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_lstm_618/transpose_1/permц
backward_lstm_618/transpose_1	Transpose=backward_lstm_618/TensorArrayV2Stack/TensorListStack:tensor:0+backward_lstm_618/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€22
backward_lstm_618/transpose_1К
backward_lstm_618/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_618/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisƒ
concatConcatV2)forward_lstm_618/strided_slice_3:output:0*backward_lstm_618/strided_slice_3:output:0concat/axis:output:0*
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
NoOpNoOp8^backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp7^backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp9^backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp^backward_lstm_618/while7^forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp6^forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp8^forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp^forward_lstm_618/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€:€€€€€€€€€: : : : : : 2r
7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp7backward_lstm_618/lstm_cell_1856/BiasAdd/ReadVariableOp2p
6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp6backward_lstm_618/lstm_cell_1856/MatMul/ReadVariableOp2t
8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp8backward_lstm_618/lstm_cell_1856/MatMul_1/ReadVariableOp22
backward_lstm_618/whilebackward_lstm_618/while2p
6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp6forward_lstm_618/lstm_cell_1855/BiasAdd/ReadVariableOp2n
5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp5forward_lstm_618/lstm_cell_1855/MatMul/ReadVariableOp2r
7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp7forward_lstm_618/lstm_cell_1855/MatMul_1/ReadVariableOp20
forward_lstm_618/whileforward_lstm_618/while:O K
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
while_body_62326697
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1856_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1856_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1856_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1856_matmul_readvariableop_resource:	»H
5while_lstm_cell_1856_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1856_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1856/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1856/MatMul/ReadVariableOpҐ,while/lstm_cell_1856/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1856_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1856/MatMul/ReadVariableOpЁ
while/lstm_cell_1856/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/MatMul’
,while/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1856_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1856/MatMul_1/ReadVariableOp∆
while/lstm_cell_1856/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/MatMul_1ј
while/lstm_cell_1856/addAddV2%while/lstm_cell_1856/MatMul:product:0'while/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/addќ
+while/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1856_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1856/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1856/BiasAddBiasAddwhile/lstm_cell_1856/add:z:03while/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/BiasAddО
$while/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1856/split/split_dimУ
while/lstm_cell_1856/splitSplit-while/lstm_cell_1856/split/split_dim:output:0%while/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1856/splitЮ
while/lstm_cell_1856/SigmoidSigmoid#while/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/SigmoidҐ
while/lstm_cell_1856/Sigmoid_1Sigmoid#while/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1856/Sigmoid_1¶
while/lstm_cell_1856/mulMul"while/lstm_cell_1856/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/mulХ
while/lstm_cell_1856/ReluRelu#while/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/ReluЉ
while/lstm_cell_1856/mul_1Mul while/lstm_cell_1856/Sigmoid:y:0'while/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/mul_1±
while/lstm_cell_1856/add_1AddV2while/lstm_cell_1856/mul:z:0while/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/add_1Ґ
while/lstm_cell_1856/Sigmoid_2Sigmoid#while/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1856/Sigmoid_2Ф
while/lstm_cell_1856/Relu_1Reluwhile/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/Relu_1ј
while/lstm_cell_1856/mul_2Mul"while/lstm_cell_1856/Sigmoid_2:y:0)while/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1856/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1856/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1856/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1856/BiasAdd/ReadVariableOp+^while/lstm_cell_1856/MatMul/ReadVariableOp-^while/lstm_cell_1856/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1856_biasadd_readvariableop_resource6while_lstm_cell_1856_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1856_matmul_1_readvariableop_resource7while_lstm_cell_1856_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1856_matmul_readvariableop_resource5while_lstm_cell_1856_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1856/BiasAdd/ReadVariableOp+while/lstm_cell_1856/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1856/MatMul/ReadVariableOp*while/lstm_cell_1856/MatMul/ReadVariableOp2\
,while/lstm_cell_1856/MatMul_1/ReadVariableOp,while/lstm_cell_1856/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
зe
Ћ
$forward_lstm_618_while_body_62327604>
:forward_lstm_618_while_forward_lstm_618_while_loop_counterD
@forward_lstm_618_while_forward_lstm_618_while_maximum_iterations&
"forward_lstm_618_while_placeholder(
$forward_lstm_618_while_placeholder_1(
$forward_lstm_618_while_placeholder_2(
$forward_lstm_618_while_placeholder_3(
$forward_lstm_618_while_placeholder_4=
9forward_lstm_618_while_forward_lstm_618_strided_slice_1_0y
uforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_618_while_greater_forward_lstm_618_cast_0Y
Fforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0:	»[
Hforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0:	2»V
Gforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0:	»#
forward_lstm_618_while_identity%
!forward_lstm_618_while_identity_1%
!forward_lstm_618_while_identity_2%
!forward_lstm_618_while_identity_3%
!forward_lstm_618_while_identity_4%
!forward_lstm_618_while_identity_5%
!forward_lstm_618_while_identity_6;
7forward_lstm_618_while_forward_lstm_618_strided_slice_1w
sforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_618_while_greater_forward_lstm_618_castW
Dforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource:	»Y
Fforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource:	2»T
Eforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource:	»ИҐ<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpҐ;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOpҐ=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpе
Hforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2J
Hforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeє
:forward_lstm_618/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_618_while_placeholderQforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02<
:forward_lstm_618/while/TensorArrayV2Read/TensorListGetItem’
forward_lstm_618/while/GreaterGreater6forward_lstm_618_while_greater_forward_lstm_618_cast_0"forward_lstm_618_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2 
forward_lstm_618/while/GreaterВ
;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOpFforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02=
;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp°
,forward_lstm_618/while/lstm_cell_1855/MatMulMatMulAforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem:item:0Cforward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2.
,forward_lstm_618/while/lstm_cell_1855/MatMulИ
=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOpHforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02?
=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpК
.forward_lstm_618/while/lstm_cell_1855/MatMul_1MatMul$forward_lstm_618_while_placeholder_3Eforward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.forward_lstm_618/while/lstm_cell_1855/MatMul_1Д
)forward_lstm_618/while/lstm_cell_1855/addAddV26forward_lstm_618/while/lstm_cell_1855/MatMul:product:08forward_lstm_618/while/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)forward_lstm_618/while/lstm_cell_1855/addБ
<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOpGforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02>
<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpС
-forward_lstm_618/while/lstm_cell_1855/BiasAddBiasAdd-forward_lstm_618/while/lstm_cell_1855/add:z:0Dforward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-forward_lstm_618/while/lstm_cell_1855/BiasAdd∞
5forward_lstm_618/while/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5forward_lstm_618/while/lstm_cell_1855/split/split_dim„
+forward_lstm_618/while/lstm_cell_1855/splitSplit>forward_lstm_618/while/lstm_cell_1855/split/split_dim:output:06forward_lstm_618/while/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2-
+forward_lstm_618/while/lstm_cell_1855/split—
-forward_lstm_618/while/lstm_cell_1855/SigmoidSigmoid4forward_lstm_618/while/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-forward_lstm_618/while/lstm_cell_1855/Sigmoid’
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_1Sigmoid4forward_lstm_618/while/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_1к
)forward_lstm_618/while/lstm_cell_1855/mulMul3forward_lstm_618/while/lstm_cell_1855/Sigmoid_1:y:0$forward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_618/while/lstm_cell_1855/mul»
*forward_lstm_618/while/lstm_cell_1855/ReluRelu4forward_lstm_618/while/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22,
*forward_lstm_618/while/lstm_cell_1855/ReluА
+forward_lstm_618/while/lstm_cell_1855/mul_1Mul1forward_lstm_618/while/lstm_cell_1855/Sigmoid:y:08forward_lstm_618/while/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_618/while/lstm_cell_1855/mul_1х
+forward_lstm_618/while/lstm_cell_1855/add_1AddV2-forward_lstm_618/while/lstm_cell_1855/mul:z:0/forward_lstm_618/while/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_618/while/lstm_cell_1855/add_1’
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_2Sigmoid4forward_lstm_618/while/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_2«
,forward_lstm_618/while/lstm_cell_1855/Relu_1Relu/forward_lstm_618/while/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,forward_lstm_618/while/lstm_cell_1855/Relu_1Д
+forward_lstm_618/while/lstm_cell_1855/mul_2Mul3forward_lstm_618/while/lstm_cell_1855/Sigmoid_2:y:0:forward_lstm_618/while/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_618/while/lstm_cell_1855/mul_2х
forward_lstm_618/while/SelectSelect"forward_lstm_618/while/Greater:z:0/forward_lstm_618/while/lstm_cell_1855/mul_2:z:0$forward_lstm_618_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/while/Selectщ
forward_lstm_618/while/Select_1Select"forward_lstm_618/while/Greater:z:0/forward_lstm_618/while/lstm_cell_1855/mul_2:z:0$forward_lstm_618_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_618/while/Select_1щ
forward_lstm_618/while/Select_2Select"forward_lstm_618/while/Greater:z:0/forward_lstm_618/while/lstm_cell_1855/add_1:z:0$forward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_618/while/Select_2Ѓ
;forward_lstm_618/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_618_while_placeholder_1"forward_lstm_618_while_placeholder&forward_lstm_618/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_618/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_618/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_618/while/add/y≠
forward_lstm_618/while/addAddV2"forward_lstm_618_while_placeholder%forward_lstm_618/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/while/addВ
forward_lstm_618/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_618/while/add_1/yЋ
forward_lstm_618/while/add_1AddV2:forward_lstm_618_while_forward_lstm_618_while_loop_counter'forward_lstm_618/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/while/add_1ѓ
forward_lstm_618/while/IdentityIdentity forward_lstm_618/while/add_1:z:0^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_618/while/Identity”
!forward_lstm_618/while/Identity_1Identity@forward_lstm_618_while_forward_lstm_618_while_maximum_iterations^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_618/while/Identity_1±
!forward_lstm_618/while/Identity_2Identityforward_lstm_618/while/add:z:0^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_618/while/Identity_2ё
!forward_lstm_618/while/Identity_3IdentityKforward_lstm_618/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_618/while/Identity_3 
!forward_lstm_618/while/Identity_4Identity&forward_lstm_618/while/Select:output:0^forward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_618/while/Identity_4ћ
!forward_lstm_618/while/Identity_5Identity(forward_lstm_618/while/Select_1:output:0^forward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_618/while/Identity_5ћ
!forward_lstm_618/while/Identity_6Identity(forward_lstm_618/while/Select_2:output:0^forward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_618/while/Identity_6є
forward_lstm_618/while/NoOpNoOp=^forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp<^forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp>^forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_618/while/NoOp"t
7forward_lstm_618_while_forward_lstm_618_strided_slice_19forward_lstm_618_while_forward_lstm_618_strided_slice_1_0"n
4forward_lstm_618_while_greater_forward_lstm_618_cast6forward_lstm_618_while_greater_forward_lstm_618_cast_0"K
forward_lstm_618_while_identity(forward_lstm_618/while/Identity:output:0"O
!forward_lstm_618_while_identity_1*forward_lstm_618/while/Identity_1:output:0"O
!forward_lstm_618_while_identity_2*forward_lstm_618/while/Identity_2:output:0"O
!forward_lstm_618_while_identity_3*forward_lstm_618/while/Identity_3:output:0"O
!forward_lstm_618_while_identity_4*forward_lstm_618/while/Identity_4:output:0"O
!forward_lstm_618_while_identity_5*forward_lstm_618/while/Identity_5:output:0"O
!forward_lstm_618_while_identity_6*forward_lstm_618/while/Identity_6:output:0"Р
Eforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resourceGforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0"Т
Fforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resourceHforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0"О
Dforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resourceFforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0"м
sforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensoruforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2|
<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp2z
;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp2~
=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_cond_62326344
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_62326344___redundant_placeholder06
2while_while_cond_62326344___redundant_placeholder16
2while_while_cond_62326344___redundant_placeholder26
2while_while_cond_62326344___redundant_placeholder3
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
while_cond_62326869
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_62326869___redundant_placeholder06
2while_while_cond_62326869___redundant_placeholder16
2while_while_cond_62326869___redundant_placeholder26
2while_while_cond_62326869___redundant_placeholder3
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
L__inference_lstm_cell_1856_layer_call_and_return_conditional_losses_62325857

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
л	
Щ
4__inference_bidirectional_618_layer_call_fn_62328077
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
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_623266002
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
я
Ќ
while_cond_62330033
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_62330033___redundant_placeholder06
2while_while_cond_62330033___redundant_placeholder16
2while_while_cond_62330033___redundant_placeholder26
2while_while_cond_62330033___redundant_placeholder3
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
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_62330774

inputs@
-lstm_cell_1856_matmul_readvariableop_resource:	»B
/lstm_cell_1856_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1856_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1856/BiasAdd/ReadVariableOpҐ$lstm_cell_1856/MatMul/ReadVariableOpҐ&lstm_cell_1856/MatMul_1/ReadVariableOpҐwhileD
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
$lstm_cell_1856/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1856_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1856/MatMul/ReadVariableOp≥
lstm_cell_1856/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/MatMulЅ
&lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1856_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1856/MatMul_1/ReadVariableOpѓ
lstm_cell_1856/MatMul_1MatMulzeros:output:0.lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/MatMul_1®
lstm_cell_1856/addAddV2lstm_cell_1856/MatMul:product:0!lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/addЇ
%lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1856_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1856/BiasAdd/ReadVariableOpµ
lstm_cell_1856/BiasAddBiasAddlstm_cell_1856/add:z:0-lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/BiasAddВ
lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1856/split/split_dimы
lstm_cell_1856/splitSplit'lstm_cell_1856/split/split_dim:output:0lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1856/splitМ
lstm_cell_1856/SigmoidSigmoidlstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/SigmoidР
lstm_cell_1856/Sigmoid_1Sigmoidlstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Sigmoid_1С
lstm_cell_1856/mulMullstm_cell_1856/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/mulГ
lstm_cell_1856/ReluRelulstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Relu§
lstm_cell_1856/mul_1Mullstm_cell_1856/Sigmoid:y:0!lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/mul_1Щ
lstm_cell_1856/add_1AddV2lstm_cell_1856/mul:z:0lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/add_1Р
lstm_cell_1856/Sigmoid_2Sigmoidlstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Sigmoid_2В
lstm_cell_1856/Relu_1Relulstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Relu_1®
lstm_cell_1856/mul_2Mullstm_cell_1856/Sigmoid_2:y:0#lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1856_matmul_readvariableop_resource/lstm_cell_1856_matmul_1_readvariableop_resource.lstm_cell_1856_biasadd_readvariableop_resource*
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
while_body_62330690*
condR
while_cond_62330689*K
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
NoOpNoOp&^lstm_cell_1856/BiasAdd/ReadVariableOp%^lstm_cell_1856/MatMul/ReadVariableOp'^lstm_cell_1856/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1856/BiasAdd/ReadVariableOp%lstm_cell_1856/BiasAdd/ReadVariableOp2L
$lstm_cell_1856/MatMul/ReadVariableOp$lstm_cell_1856/MatMul/ReadVariableOp2P
&lstm_cell_1856/MatMul_1/ReadVariableOp&lstm_cell_1856/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
€
К
L__inference_lstm_cell_1855_layer_call_and_return_conditional_losses_62330872

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
%backward_lstm_618_while_cond_62327342@
<backward_lstm_618_while_backward_lstm_618_while_loop_counterF
Bbackward_lstm_618_while_backward_lstm_618_while_maximum_iterations'
#backward_lstm_618_while_placeholder)
%backward_lstm_618_while_placeholder_1)
%backward_lstm_618_while_placeholder_2)
%backward_lstm_618_while_placeholder_3)
%backward_lstm_618_while_placeholder_4B
>backward_lstm_618_while_less_backward_lstm_618_strided_slice_1Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62327342___redundant_placeholder0Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62327342___redundant_placeholder1Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62327342___redundant_placeholder2Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62327342___redundant_placeholder3Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62327342___redundant_placeholder4$
 backward_lstm_618_while_identity
 
backward_lstm_618/while/LessLess#backward_lstm_618_while_placeholder>backward_lstm_618_while_less_backward_lstm_618_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_618/while/LessУ
 backward_lstm_618/while/IdentityIdentity backward_lstm_618/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_618/while/Identity"M
 backward_lstm_618_while_identity)backward_lstm_618/while/Identity:output:0*(
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
“Є
ё 
$__inference__traced_restore_62331238
file_prefix3
!assignvariableop_dense_618_kernel:d/
!assignvariableop_1_dense_618_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: ^
Kassignvariableop_7_bidirectional_618_forward_lstm_618_lstm_cell_1855_kernel:	»h
Uassignvariableop_8_bidirectional_618_forward_lstm_618_lstm_cell_1855_recurrent_kernel:	2»X
Iassignvariableop_9_bidirectional_618_forward_lstm_618_lstm_cell_1855_bias:	»`
Massignvariableop_10_bidirectional_618_backward_lstm_618_lstm_cell_1856_kernel:	»j
Wassignvariableop_11_bidirectional_618_backward_lstm_618_lstm_cell_1856_recurrent_kernel:	2»Z
Kassignvariableop_12_bidirectional_618_backward_lstm_618_lstm_cell_1856_bias:	»#
assignvariableop_13_total: #
assignvariableop_14_count: =
+assignvariableop_15_adam_dense_618_kernel_m:d7
)assignvariableop_16_adam_dense_618_bias_m:f
Sassignvariableop_17_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_kernel_m:	»p
]assignvariableop_18_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_recurrent_kernel_m:	2»`
Qassignvariableop_19_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_bias_m:	»g
Tassignvariableop_20_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_kernel_m:	»q
^assignvariableop_21_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_recurrent_kernel_m:	2»a
Rassignvariableop_22_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_bias_m:	»=
+assignvariableop_23_adam_dense_618_kernel_v:d7
)assignvariableop_24_adam_dense_618_bias_v:f
Sassignvariableop_25_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_kernel_v:	»p
]assignvariableop_26_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_recurrent_kernel_v:	2»`
Qassignvariableop_27_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_bias_v:	»g
Tassignvariableop_28_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_kernel_v:	»q
^assignvariableop_29_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_recurrent_kernel_v:	2»a
Rassignvariableop_30_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_bias_v:	»@
.assignvariableop_31_adam_dense_618_kernel_vhat:d:
,assignvariableop_32_adam_dense_618_bias_vhat:i
Vassignvariableop_33_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_kernel_vhat:	»s
`assignvariableop_34_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_recurrent_kernel_vhat:	2»c
Tassignvariableop_35_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_bias_vhat:	»j
Wassignvariableop_36_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_kernel_vhat:	»t
aassignvariableop_37_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_recurrent_kernel_vhat:	2»d
Uassignvariableop_38_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_bias_vhat:	»
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
AssignVariableOpAssignVariableOp!assignvariableop_dense_618_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¶
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_618_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOpKassignvariableop_7_bidirectional_618_forward_lstm_618_lstm_cell_1855_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Џ
AssignVariableOp_8AssignVariableOpUassignvariableop_8_bidirectional_618_forward_lstm_618_lstm_cell_1855_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ќ
AssignVariableOp_9AssignVariableOpIassignvariableop_9_bidirectional_618_forward_lstm_618_lstm_cell_1855_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10’
AssignVariableOp_10AssignVariableOpMassignvariableop_10_bidirectional_618_backward_lstm_618_lstm_cell_1856_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11я
AssignVariableOp_11AssignVariableOpWassignvariableop_11_bidirectional_618_backward_lstm_618_lstm_cell_1856_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12”
AssignVariableOp_12AssignVariableOpKassignvariableop_12_bidirectional_618_backward_lstm_618_lstm_cell_1856_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_dense_618_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16±
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_dense_618_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17џ
AssignVariableOp_17AssignVariableOpSassignvariableop_17_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18е
AssignVariableOp_18AssignVariableOp]assignvariableop_18_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19ў
AssignVariableOp_19AssignVariableOpQassignvariableop_19_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20№
AssignVariableOp_20AssignVariableOpTassignvariableop_20_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ж
AssignVariableOp_21AssignVariableOp^assignvariableop_21_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Џ
AssignVariableOp_22AssignVariableOpRassignvariableop_22_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23≥
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_618_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24±
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_618_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25џ
AssignVariableOp_25AssignVariableOpSassignvariableop_25_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26е
AssignVariableOp_26AssignVariableOp]assignvariableop_26_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27ў
AssignVariableOp_27AssignVariableOpQassignvariableop_27_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28№
AssignVariableOp_28AssignVariableOpTassignvariableop_28_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29ж
AssignVariableOp_29AssignVariableOp^assignvariableop_29_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Џ
AssignVariableOp_30AssignVariableOpRassignvariableop_30_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31ґ
AssignVariableOp_31AssignVariableOp.assignvariableop_31_adam_dense_618_kernel_vhatIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32і
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_dense_618_bias_vhatIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33ё
AssignVariableOp_33AssignVariableOpVassignvariableop_33_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_kernel_vhatIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34и
AssignVariableOp_34AssignVariableOp`assignvariableop_34_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_recurrent_kernel_vhatIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35№
AssignVariableOp_35AssignVariableOpTassignvariableop_35_adam_bidirectional_618_forward_lstm_618_lstm_cell_1855_bias_vhatIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36я
AssignVariableOp_36AssignVariableOpWassignvariableop_36_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_kernel_vhatIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37й
AssignVariableOp_37AssignVariableOpaassignvariableop_37_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_recurrent_kernel_vhatIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ё
AssignVariableOp_38AssignVariableOpUassignvariableop_38_adam_bidirectional_618_backward_lstm_618_lstm_cell_1856_bias_vhatIdentity_38:output:0"/device:CPU:0*
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
¶Z
§
%backward_lstm_618_while_body_62328648@
<backward_lstm_618_while_backward_lstm_618_while_loop_counterF
Bbackward_lstm_618_while_backward_lstm_618_while_maximum_iterations'
#backward_lstm_618_while_placeholder)
%backward_lstm_618_while_placeholder_1)
%backward_lstm_618_while_placeholder_2)
%backward_lstm_618_while_placeholder_3?
;backward_lstm_618_while_backward_lstm_618_strided_slice_1_0{
wbackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0Z
Gbackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0:	»\
Ibackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0:	2»W
Hbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0:	»$
 backward_lstm_618_while_identity&
"backward_lstm_618_while_identity_1&
"backward_lstm_618_while_identity_2&
"backward_lstm_618_while_identity_3&
"backward_lstm_618_while_identity_4&
"backward_lstm_618_while_identity_5=
9backward_lstm_618_while_backward_lstm_618_strided_slice_1y
ubackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensorX
Ebackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource:	»Z
Gbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource:	2»U
Fbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource:	»ИҐ=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpҐ<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOpҐ>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpз
Ibackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2K
Ibackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shape»
;backward_lstm_618/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_618_while_placeholderRbackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02=
;backward_lstm_618/while/TensorArrayV2Read/TensorListGetItemЕ
<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOpGbackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02>
<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp•
-backward_lstm_618/while/lstm_cell_1856/MatMulMatMulBbackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem:item:0Dbackward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-backward_lstm_618/while/lstm_cell_1856/MatMulЛ
>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOpIbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02@
>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpО
/backward_lstm_618/while/lstm_cell_1856/MatMul_1MatMul%backward_lstm_618_while_placeholder_2Fbackward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»21
/backward_lstm_618/while/lstm_cell_1856/MatMul_1И
*backward_lstm_618/while/lstm_cell_1856/addAddV27backward_lstm_618/while/lstm_cell_1856/MatMul:product:09backward_lstm_618/while/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2,
*backward_lstm_618/while/lstm_cell_1856/addД
=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOpHbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02?
=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpХ
.backward_lstm_618/while/lstm_cell_1856/BiasAddBiasAdd.backward_lstm_618/while/lstm_cell_1856/add:z:0Ebackward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.backward_lstm_618/while/lstm_cell_1856/BiasAdd≤
6backward_lstm_618/while/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6backward_lstm_618/while/lstm_cell_1856/split/split_dimџ
,backward_lstm_618/while/lstm_cell_1856/splitSplit?backward_lstm_618/while/lstm_cell_1856/split/split_dim:output:07backward_lstm_618/while/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2.
,backward_lstm_618/while/lstm_cell_1856/split‘
.backward_lstm_618/while/lstm_cell_1856/SigmoidSigmoid5backward_lstm_618/while/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€220
.backward_lstm_618/while/lstm_cell_1856/SigmoidЎ
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_1Sigmoid5backward_lstm_618/while/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_1о
*backward_lstm_618/while/lstm_cell_1856/mulMul4backward_lstm_618/while/lstm_cell_1856/Sigmoid_1:y:0%backward_lstm_618_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_618/while/lstm_cell_1856/mulЋ
+backward_lstm_618/while/lstm_cell_1856/ReluRelu5backward_lstm_618/while/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22-
+backward_lstm_618/while/lstm_cell_1856/ReluД
,backward_lstm_618/while/lstm_cell_1856/mul_1Mul2backward_lstm_618/while/lstm_cell_1856/Sigmoid:y:09backward_lstm_618/while/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_618/while/lstm_cell_1856/mul_1щ
,backward_lstm_618/while/lstm_cell_1856/add_1AddV2.backward_lstm_618/while/lstm_cell_1856/mul:z:00backward_lstm_618/while/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_618/while/lstm_cell_1856/add_1Ў
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_2Sigmoid5backward_lstm_618/while/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_2 
-backward_lstm_618/while/lstm_cell_1856/Relu_1Relu0backward_lstm_618/while/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-backward_lstm_618/while/lstm_cell_1856/Relu_1И
,backward_lstm_618/while/lstm_cell_1856/mul_2Mul4backward_lstm_618/while/lstm_cell_1856/Sigmoid_2:y:0;backward_lstm_618/while/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_618/while/lstm_cell_1856/mul_2Љ
<backward_lstm_618/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_618_while_placeholder_1#backward_lstm_618_while_placeholder0backward_lstm_618/while/lstm_cell_1856/mul_2:z:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_618/while/TensorArrayV2Write/TensorListSetItemА
backward_lstm_618/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_618/while/add/y±
backward_lstm_618/while/addAddV2#backward_lstm_618_while_placeholder&backward_lstm_618/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/while/addД
backward_lstm_618/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_618/while/add_1/y–
backward_lstm_618/while/add_1AddV2<backward_lstm_618_while_backward_lstm_618_while_loop_counter(backward_lstm_618/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/while/add_1≥
 backward_lstm_618/while/IdentityIdentity!backward_lstm_618/while/add_1:z:0^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_618/while/IdentityЎ
"backward_lstm_618/while/Identity_1IdentityBbackward_lstm_618_while_backward_lstm_618_while_maximum_iterations^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_618/while/Identity_1µ
"backward_lstm_618/while/Identity_2Identitybackward_lstm_618/while/add:z:0^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_618/while/Identity_2в
"backward_lstm_618/while/Identity_3IdentityLbackward_lstm_618/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_618/while/Identity_3„
"backward_lstm_618/while/Identity_4Identity0backward_lstm_618/while/lstm_cell_1856/mul_2:z:0^backward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_618/while/Identity_4„
"backward_lstm_618/while/Identity_5Identity0backward_lstm_618/while/lstm_cell_1856/add_1:z:0^backward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_618/while/Identity_5Њ
backward_lstm_618/while/NoOpNoOp>^backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp=^backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp?^backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_618/while/NoOp"x
9backward_lstm_618_while_backward_lstm_618_strided_slice_1;backward_lstm_618_while_backward_lstm_618_strided_slice_1_0"M
 backward_lstm_618_while_identity)backward_lstm_618/while/Identity:output:0"Q
"backward_lstm_618_while_identity_1+backward_lstm_618/while/Identity_1:output:0"Q
"backward_lstm_618_while_identity_2+backward_lstm_618/while/Identity_2:output:0"Q
"backward_lstm_618_while_identity_3+backward_lstm_618/while/Identity_3:output:0"Q
"backward_lstm_618_while_identity_4+backward_lstm_618/while/Identity_4:output:0"Q
"backward_lstm_618_while_identity_5+backward_lstm_618/while/Identity_5:output:0"Т
Fbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resourceHbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0"Ф
Gbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resourceIbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0"Р
Ebackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resourceGbackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0"р
ubackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2~
=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp2|
<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp2А
>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_62325794

inputs*
lstm_cell_1856_62325712:	»*
lstm_cell_1856_62325714:	2»&
lstm_cell_1856_62325716:	»
identityИҐ&lstm_cell_1856/StatefulPartitionedCallҐwhileD
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
&lstm_cell_1856/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1856_62325712lstm_cell_1856_62325714lstm_cell_1856_62325716*
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
L__inference_lstm_cell_1856_layer_call_and_return_conditional_losses_623257112(
&lstm_cell_1856/StatefulPartitionedCallП
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1856_62325712lstm_cell_1856_62325714lstm_cell_1856_62325716*
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
while_body_62325725*
condR
while_cond_62325724*K
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
NoOpNoOp'^lstm_cell_1856/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2P
&lstm_cell_1856/StatefulPartitionedCall&lstm_cell_1856/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Щ
™
Esequential_618_bidirectional_618_forward_lstm_618_while_cond_62324720А
|sequential_618_bidirectional_618_forward_lstm_618_while_sequential_618_bidirectional_618_forward_lstm_618_while_loop_counterЗ
Вsequential_618_bidirectional_618_forward_lstm_618_while_sequential_618_bidirectional_618_forward_lstm_618_while_maximum_iterationsG
Csequential_618_bidirectional_618_forward_lstm_618_while_placeholderI
Esequential_618_bidirectional_618_forward_lstm_618_while_placeholder_1I
Esequential_618_bidirectional_618_forward_lstm_618_while_placeholder_2I
Esequential_618_bidirectional_618_forward_lstm_618_while_placeholder_3I
Esequential_618_bidirectional_618_forward_lstm_618_while_placeholder_4В
~sequential_618_bidirectional_618_forward_lstm_618_while_less_sequential_618_bidirectional_618_forward_lstm_618_strided_slice_1Ы
Цsequential_618_bidirectional_618_forward_lstm_618_while_sequential_618_bidirectional_618_forward_lstm_618_while_cond_62324720___redundant_placeholder0Ы
Цsequential_618_bidirectional_618_forward_lstm_618_while_sequential_618_bidirectional_618_forward_lstm_618_while_cond_62324720___redundant_placeholder1Ы
Цsequential_618_bidirectional_618_forward_lstm_618_while_sequential_618_bidirectional_618_forward_lstm_618_while_cond_62324720___redundant_placeholder2Ы
Цsequential_618_bidirectional_618_forward_lstm_618_while_sequential_618_bidirectional_618_forward_lstm_618_while_cond_62324720___redundant_placeholder3Ы
Цsequential_618_bidirectional_618_forward_lstm_618_while_sequential_618_bidirectional_618_forward_lstm_618_while_cond_62324720___redundant_placeholder4D
@sequential_618_bidirectional_618_forward_lstm_618_while_identity
к
<sequential_618/bidirectional_618/forward_lstm_618/while/LessLessCsequential_618_bidirectional_618_forward_lstm_618_while_placeholder~sequential_618_bidirectional_618_forward_lstm_618_while_less_sequential_618_bidirectional_618_forward_lstm_618_strided_slice_1*
T0*
_output_shapes
: 2>
<sequential_618/bidirectional_618/forward_lstm_618/while/Lessу
@sequential_618/bidirectional_618/forward_lstm_618/while/IdentityIdentity@sequential_618/bidirectional_618/forward_lstm_618/while/Less:z:0*
T0
*
_output_shapes
: 2B
@sequential_618/bidirectional_618/forward_lstm_618/while/Identity"Н
@sequential_618_bidirectional_618_forward_lstm_618_while_identityIsequential_618/bidirectional_618/forward_lstm_618/while/Identity:output:0*(
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
ѕ_
µ
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_62330468
inputs_0@
-lstm_cell_1856_matmul_readvariableop_resource:	»B
/lstm_cell_1856_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1856_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1856/BiasAdd/ReadVariableOpҐ$lstm_cell_1856/MatMul/ReadVariableOpҐ&lstm_cell_1856/MatMul_1/ReadVariableOpҐwhileF
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
$lstm_cell_1856/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1856_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1856/MatMul/ReadVariableOp≥
lstm_cell_1856/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/MatMulЅ
&lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1856_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1856/MatMul_1/ReadVariableOpѓ
lstm_cell_1856/MatMul_1MatMulzeros:output:0.lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/MatMul_1®
lstm_cell_1856/addAddV2lstm_cell_1856/MatMul:product:0!lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/addЇ
%lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1856_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1856/BiasAdd/ReadVariableOpµ
lstm_cell_1856/BiasAddBiasAddlstm_cell_1856/add:z:0-lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/BiasAddВ
lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1856/split/split_dimы
lstm_cell_1856/splitSplit'lstm_cell_1856/split/split_dim:output:0lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1856/splitМ
lstm_cell_1856/SigmoidSigmoidlstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/SigmoidР
lstm_cell_1856/Sigmoid_1Sigmoidlstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Sigmoid_1С
lstm_cell_1856/mulMullstm_cell_1856/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/mulГ
lstm_cell_1856/ReluRelulstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Relu§
lstm_cell_1856/mul_1Mullstm_cell_1856/Sigmoid:y:0!lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/mul_1Щ
lstm_cell_1856/add_1AddV2lstm_cell_1856/mul:z:0lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/add_1Р
lstm_cell_1856/Sigmoid_2Sigmoidlstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Sigmoid_2В
lstm_cell_1856/Relu_1Relulstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Relu_1®
lstm_cell_1856/mul_2Mullstm_cell_1856/Sigmoid_2:y:0#lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1856_matmul_readvariableop_resource/lstm_cell_1856_matmul_1_readvariableop_resource.lstm_cell_1856_biasadd_readvariableop_resource*
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
while_body_62330384*
condR
while_cond_62330383*K
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
NoOpNoOp&^lstm_cell_1856/BiasAdd/ReadVariableOp%^lstm_cell_1856/MatMul/ReadVariableOp'^lstm_cell_1856/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1856/BiasAdd/ReadVariableOp%lstm_cell_1856/BiasAdd/ReadVariableOp2L
$lstm_cell_1856/MatMul/ReadVariableOp$lstm_cell_1856/MatMul/ReadVariableOp2P
&lstm_cell_1856/MatMul_1/ReadVariableOp&lstm_cell_1856/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
ф_
≥
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_62330621

inputs@
-lstm_cell_1856_matmul_readvariableop_resource:	»B
/lstm_cell_1856_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1856_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1856/BiasAdd/ReadVariableOpҐ$lstm_cell_1856/MatMul/ReadVariableOpҐ&lstm_cell_1856/MatMul_1/ReadVariableOpҐwhileD
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
$lstm_cell_1856/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1856_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1856/MatMul/ReadVariableOp≥
lstm_cell_1856/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/MatMulЅ
&lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1856_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1856/MatMul_1/ReadVariableOpѓ
lstm_cell_1856/MatMul_1MatMulzeros:output:0.lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/MatMul_1®
lstm_cell_1856/addAddV2lstm_cell_1856/MatMul:product:0!lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/addЇ
%lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1856_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1856/BiasAdd/ReadVariableOpµ
lstm_cell_1856/BiasAddBiasAddlstm_cell_1856/add:z:0-lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/BiasAddВ
lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1856/split/split_dimы
lstm_cell_1856/splitSplit'lstm_cell_1856/split/split_dim:output:0lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1856/splitМ
lstm_cell_1856/SigmoidSigmoidlstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/SigmoidР
lstm_cell_1856/Sigmoid_1Sigmoidlstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Sigmoid_1С
lstm_cell_1856/mulMullstm_cell_1856/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/mulГ
lstm_cell_1856/ReluRelulstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Relu§
lstm_cell_1856/mul_1Mullstm_cell_1856/Sigmoid:y:0!lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/mul_1Щ
lstm_cell_1856/add_1AddV2lstm_cell_1856/mul:z:0lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/add_1Р
lstm_cell_1856/Sigmoid_2Sigmoidlstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Sigmoid_2В
lstm_cell_1856/Relu_1Relulstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Relu_1®
lstm_cell_1856/mul_2Mullstm_cell_1856/Sigmoid_2:y:0#lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1856_matmul_readvariableop_resource/lstm_cell_1856_matmul_1_readvariableop_resource.lstm_cell_1856_biasadd_readvariableop_resource*
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
while_body_62330537*
condR
while_cond_62330536*K
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
NoOpNoOp&^lstm_cell_1856/BiasAdd/ReadVariableOp%^lstm_cell_1856/MatMul/ReadVariableOp'^lstm_cell_1856/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1856/BiasAdd/ReadVariableOp%lstm_cell_1856/BiasAdd/ReadVariableOp2L
$lstm_cell_1856/MatMul/ReadVariableOp$lstm_cell_1856/MatMul/ReadVariableOp2P
&lstm_cell_1856/MatMul_1/ReadVariableOp&lstm_cell_1856/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ѕ_
µ
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_62330315
inputs_0@
-lstm_cell_1856_matmul_readvariableop_resource:	»B
/lstm_cell_1856_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1856_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1856/BiasAdd/ReadVariableOpҐ$lstm_cell_1856/MatMul/ReadVariableOpҐ&lstm_cell_1856/MatMul_1/ReadVariableOpҐwhileF
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
$lstm_cell_1856/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1856_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1856/MatMul/ReadVariableOp≥
lstm_cell_1856/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/MatMulЅ
&lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1856_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1856/MatMul_1/ReadVariableOpѓ
lstm_cell_1856/MatMul_1MatMulzeros:output:0.lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/MatMul_1®
lstm_cell_1856/addAddV2lstm_cell_1856/MatMul:product:0!lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/addЇ
%lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1856_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1856/BiasAdd/ReadVariableOpµ
lstm_cell_1856/BiasAddBiasAddlstm_cell_1856/add:z:0-lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/BiasAddВ
lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1856/split/split_dimы
lstm_cell_1856/splitSplit'lstm_cell_1856/split/split_dim:output:0lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1856/splitМ
lstm_cell_1856/SigmoidSigmoidlstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/SigmoidР
lstm_cell_1856/Sigmoid_1Sigmoidlstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Sigmoid_1С
lstm_cell_1856/mulMullstm_cell_1856/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/mulГ
lstm_cell_1856/ReluRelulstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Relu§
lstm_cell_1856/mul_1Mullstm_cell_1856/Sigmoid:y:0!lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/mul_1Щ
lstm_cell_1856/add_1AddV2lstm_cell_1856/mul:z:0lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/add_1Р
lstm_cell_1856/Sigmoid_2Sigmoidlstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Sigmoid_2В
lstm_cell_1856/Relu_1Relulstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Relu_1®
lstm_cell_1856/mul_2Mullstm_cell_1856/Sigmoid_2:y:0#lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1856_matmul_readvariableop_resource/lstm_cell_1856_matmul_1_readvariableop_resource.lstm_cell_1856_biasadd_readvariableop_resource*
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
while_body_62330231*
condR
while_cond_62330230*K
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
NoOpNoOp&^lstm_cell_1856/BiasAdd/ReadVariableOp%^lstm_cell_1856/MatMul/ReadVariableOp'^lstm_cell_1856/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1856/BiasAdd/ReadVariableOp%lstm_cell_1856/BiasAdd/ReadVariableOp2L
$lstm_cell_1856/MatMul/ReadVariableOp$lstm_cell_1856/MatMul/ReadVariableOp2P
&lstm_cell_1856/MatMul_1/ReadVariableOp&lstm_cell_1856/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
∆@
д
while_body_62329732
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1855_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1855_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1855_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1855_matmul_readvariableop_resource:	»H
5while_lstm_cell_1855_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1855_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1855/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1855/MatMul/ReadVariableOpҐ,while/lstm_cell_1855/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1855_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1855/MatMul/ReadVariableOpЁ
while/lstm_cell_1855/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/MatMul’
,while/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1855_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1855/MatMul_1/ReadVariableOp∆
while/lstm_cell_1855/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/MatMul_1ј
while/lstm_cell_1855/addAddV2%while/lstm_cell_1855/MatMul:product:0'while/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/addќ
+while/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1855_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1855/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1855/BiasAddBiasAddwhile/lstm_cell_1855/add:z:03while/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/BiasAddО
$while/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1855/split/split_dimУ
while/lstm_cell_1855/splitSplit-while/lstm_cell_1855/split/split_dim:output:0%while/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1855/splitЮ
while/lstm_cell_1855/SigmoidSigmoid#while/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/SigmoidҐ
while/lstm_cell_1855/Sigmoid_1Sigmoid#while/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1855/Sigmoid_1¶
while/lstm_cell_1855/mulMul"while/lstm_cell_1855/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/mulХ
while/lstm_cell_1855/ReluRelu#while/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/ReluЉ
while/lstm_cell_1855/mul_1Mul while/lstm_cell_1855/Sigmoid:y:0'while/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/mul_1±
while/lstm_cell_1855/add_1AddV2while/lstm_cell_1855/mul:z:0while/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/add_1Ґ
while/lstm_cell_1855/Sigmoid_2Sigmoid#while/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1855/Sigmoid_2Ф
while/lstm_cell_1855/Relu_1Reluwhile/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/Relu_1ј
while/lstm_cell_1855/mul_2Mul"while/lstm_cell_1855/Sigmoid_2:y:0)while/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1855/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1855/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1855/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1855/BiasAdd/ReadVariableOp+^while/lstm_cell_1855/MatMul/ReadVariableOp-^while/lstm_cell_1855/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1855_biasadd_readvariableop_resource6while_lstm_cell_1855_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1855_matmul_1_readvariableop_resource7while_lstm_cell_1855_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1855_matmul_readvariableop_resource5while_lstm_cell_1855_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1855/BiasAdd/ReadVariableOp+while/lstm_cell_1855/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1855/MatMul/ReadVariableOp*while/lstm_cell_1855/MatMul/ReadVariableOp2\
,while/lstm_cell_1855/MatMul_1/ReadVariableOp,while/lstm_cell_1855/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Љ
¬
Fsequential_618_bidirectional_618_backward_lstm_618_while_cond_62324899В
~sequential_618_bidirectional_618_backward_lstm_618_while_sequential_618_bidirectional_618_backward_lstm_618_while_loop_counterЙ
Дsequential_618_bidirectional_618_backward_lstm_618_while_sequential_618_bidirectional_618_backward_lstm_618_while_maximum_iterationsH
Dsequential_618_bidirectional_618_backward_lstm_618_while_placeholderJ
Fsequential_618_bidirectional_618_backward_lstm_618_while_placeholder_1J
Fsequential_618_bidirectional_618_backward_lstm_618_while_placeholder_2J
Fsequential_618_bidirectional_618_backward_lstm_618_while_placeholder_3J
Fsequential_618_bidirectional_618_backward_lstm_618_while_placeholder_4Е
Аsequential_618_bidirectional_618_backward_lstm_618_while_less_sequential_618_bidirectional_618_backward_lstm_618_strided_slice_1Э
Шsequential_618_bidirectional_618_backward_lstm_618_while_sequential_618_bidirectional_618_backward_lstm_618_while_cond_62324899___redundant_placeholder0Э
Шsequential_618_bidirectional_618_backward_lstm_618_while_sequential_618_bidirectional_618_backward_lstm_618_while_cond_62324899___redundant_placeholder1Э
Шsequential_618_bidirectional_618_backward_lstm_618_while_sequential_618_bidirectional_618_backward_lstm_618_while_cond_62324899___redundant_placeholder2Э
Шsequential_618_bidirectional_618_backward_lstm_618_while_sequential_618_bidirectional_618_backward_lstm_618_while_cond_62324899___redundant_placeholder3Э
Шsequential_618_bidirectional_618_backward_lstm_618_while_sequential_618_bidirectional_618_backward_lstm_618_while_cond_62324899___redundant_placeholder4E
Asequential_618_bidirectional_618_backward_lstm_618_while_identity
р
=sequential_618/bidirectional_618/backward_lstm_618/while/LessLessDsequential_618_bidirectional_618_backward_lstm_618_while_placeholderАsequential_618_bidirectional_618_backward_lstm_618_while_less_sequential_618_bidirectional_618_backward_lstm_618_strided_slice_1*
T0*
_output_shapes
: 2?
=sequential_618/bidirectional_618/backward_lstm_618/while/Lessц
Asequential_618/bidirectional_618/backward_lstm_618/while/IdentityIdentityAsequential_618/bidirectional_618/backward_lstm_618/while/Less:z:0*
T0
*
_output_shapes
: 2C
Asequential_618/bidirectional_618/backward_lstm_618/while/Identity"П
Asequential_618_bidirectional_618_backward_lstm_618_while_identityJsequential_618/bidirectional_618/backward_lstm_618/while/Identity:output:0*(
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
ь

Ў
1__inference_sequential_618_layer_call_fn_62327491

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
L__inference_sequential_618_layer_call_and_return_conditional_losses_623274722
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
ѕ@
д
while_body_62330690
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1856_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1856_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1856_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1856_matmul_readvariableop_resource:	»H
5while_lstm_cell_1856_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1856_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1856/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1856/MatMul/ReadVariableOpҐ,while/lstm_cell_1856/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1856_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1856/MatMul/ReadVariableOpЁ
while/lstm_cell_1856/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/MatMul’
,while/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1856_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1856/MatMul_1/ReadVariableOp∆
while/lstm_cell_1856/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/MatMul_1ј
while/lstm_cell_1856/addAddV2%while/lstm_cell_1856/MatMul:product:0'while/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/addќ
+while/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1856_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1856/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1856/BiasAddBiasAddwhile/lstm_cell_1856/add:z:03while/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1856/BiasAddО
$while/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1856/split/split_dimУ
while/lstm_cell_1856/splitSplit-while/lstm_cell_1856/split/split_dim:output:0%while/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1856/splitЮ
while/lstm_cell_1856/SigmoidSigmoid#while/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/SigmoidҐ
while/lstm_cell_1856/Sigmoid_1Sigmoid#while/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1856/Sigmoid_1¶
while/lstm_cell_1856/mulMul"while/lstm_cell_1856/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/mulХ
while/lstm_cell_1856/ReluRelu#while/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/ReluЉ
while/lstm_cell_1856/mul_1Mul while/lstm_cell_1856/Sigmoid:y:0'while/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/mul_1±
while/lstm_cell_1856/add_1AddV2while/lstm_cell_1856/mul:z:0while/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/add_1Ґ
while/lstm_cell_1856/Sigmoid_2Sigmoid#while/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1856/Sigmoid_2Ф
while/lstm_cell_1856/Relu_1Reluwhile/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/Relu_1ј
while/lstm_cell_1856/mul_2Mul"while/lstm_cell_1856/Sigmoid_2:y:0)while/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1856/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1856/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1856/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1856/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1856/BiasAdd/ReadVariableOp+^while/lstm_cell_1856/MatMul/ReadVariableOp-^while/lstm_cell_1856/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1856_biasadd_readvariableop_resource6while_lstm_cell_1856_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1856_matmul_1_readvariableop_resource7while_lstm_cell_1856_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1856_matmul_readvariableop_resource5while_lstm_cell_1856_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1856/BiasAdd/ReadVariableOp+while/lstm_cell_1856/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1856/MatMul/ReadVariableOp*while/lstm_cell_1856/MatMul/ReadVariableOp2\
,while/lstm_cell_1856/MatMul_1/ReadVariableOp,while/lstm_cell_1856/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
L__inference_lstm_cell_1856_layer_call_and_return_conditional_losses_62330938

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
м]
≤
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_62326429

inputs@
-lstm_cell_1855_matmul_readvariableop_resource:	»B
/lstm_cell_1855_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1855_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1855/BiasAdd/ReadVariableOpҐ$lstm_cell_1855/MatMul/ReadVariableOpҐ&lstm_cell_1855/MatMul_1/ReadVariableOpҐwhileD
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
$lstm_cell_1855/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1855_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1855/MatMul/ReadVariableOp≥
lstm_cell_1855/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/MatMulЅ
&lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1855_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1855/MatMul_1/ReadVariableOpѓ
lstm_cell_1855/MatMul_1MatMulzeros:output:0.lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/MatMul_1®
lstm_cell_1855/addAddV2lstm_cell_1855/MatMul:product:0!lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/addЇ
%lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1855_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1855/BiasAdd/ReadVariableOpµ
lstm_cell_1855/BiasAddBiasAddlstm_cell_1855/add:z:0-lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1855/BiasAddВ
lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1855/split/split_dimы
lstm_cell_1855/splitSplit'lstm_cell_1855/split/split_dim:output:0lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1855/splitМ
lstm_cell_1855/SigmoidSigmoidlstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/SigmoidР
lstm_cell_1855/Sigmoid_1Sigmoidlstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Sigmoid_1С
lstm_cell_1855/mulMullstm_cell_1855/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/mulГ
lstm_cell_1855/ReluRelulstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Relu§
lstm_cell_1855/mul_1Mullstm_cell_1855/Sigmoid:y:0!lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/mul_1Щ
lstm_cell_1855/add_1AddV2lstm_cell_1855/mul:z:0lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/add_1Р
lstm_cell_1855/Sigmoid_2Sigmoidlstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Sigmoid_2В
lstm_cell_1855/Relu_1Relulstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/Relu_1®
lstm_cell_1855/mul_2Mullstm_cell_1855/Sigmoid_2:y:0#lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1855/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1855_matmul_readvariableop_resource/lstm_cell_1855_matmul_1_readvariableop_resource.lstm_cell_1855_biasadd_readvariableop_resource*
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
while_body_62326345*
condR
while_cond_62326344*K
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
NoOpNoOp&^lstm_cell_1855/BiasAdd/ReadVariableOp%^lstm_cell_1855/MatMul/ReadVariableOp'^lstm_cell_1855/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1855/BiasAdd/ReadVariableOp%lstm_cell_1855/BiasAdd/ReadVariableOp2L
$lstm_cell_1855/MatMul/ReadVariableOp$lstm_cell_1855/MatMul/ReadVariableOp2P
&lstm_cell_1855/MatMul_1/ReadVariableOp&lstm_cell_1855/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ђ&
€
while_body_62325303
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_lstm_cell_1855_62325327_0:	»2
while_lstm_cell_1855_62325329_0:	2».
while_lstm_cell_1855_62325331_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_lstm_cell_1855_62325327:	»0
while_lstm_cell_1855_62325329:	2»,
while_lstm_cell_1855_62325331:	»ИҐ,while/lstm_cell_1855/StatefulPartitionedCall√
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
,while/lstm_cell_1855/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1855_62325327_0while_lstm_cell_1855_62325329_0while_lstm_cell_1855_62325331_0*
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
L__inference_lstm_cell_1855_layer_call_and_return_conditional_losses_623252252.
,while/lstm_cell_1855/StatefulPartitionedCallщ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder5while/lstm_cell_1855/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity5while/lstm_cell_1855/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4¶
while/Identity_5Identity5while/lstm_cell_1855/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5Й

while/NoOpNoOp-^while/lstm_cell_1855/StatefulPartitionedCall*"
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
while_lstm_cell_1855_62325327while_lstm_cell_1855_62325327_0"@
while_lstm_cell_1855_62325329while_lstm_cell_1855_62325329_0"@
while_lstm_cell_1855_62325331while_lstm_cell_1855_62325331_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2\
,while/lstm_cell_1855/StatefulPartitionedCall,while/lstm_cell_1855/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
%backward_lstm_618_while_body_62327783@
<backward_lstm_618_while_backward_lstm_618_while_loop_counterF
Bbackward_lstm_618_while_backward_lstm_618_while_maximum_iterations'
#backward_lstm_618_while_placeholder)
%backward_lstm_618_while_placeholder_1)
%backward_lstm_618_while_placeholder_2)
%backward_lstm_618_while_placeholder_3)
%backward_lstm_618_while_placeholder_4?
;backward_lstm_618_while_backward_lstm_618_strided_slice_1_0{
wbackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0:
6backward_lstm_618_while_less_backward_lstm_618_sub_1_0Z
Gbackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0:	»\
Ibackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0:	2»W
Hbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0:	»$
 backward_lstm_618_while_identity&
"backward_lstm_618_while_identity_1&
"backward_lstm_618_while_identity_2&
"backward_lstm_618_while_identity_3&
"backward_lstm_618_while_identity_4&
"backward_lstm_618_while_identity_5&
"backward_lstm_618_while_identity_6=
9backward_lstm_618_while_backward_lstm_618_strided_slice_1y
ubackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor8
4backward_lstm_618_while_less_backward_lstm_618_sub_1X
Ebackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource:	»Z
Gbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource:	2»U
Fbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource:	»ИҐ=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpҐ<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOpҐ>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpз
Ibackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2K
Ibackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeњ
;backward_lstm_618/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_618_while_placeholderRbackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02=
;backward_lstm_618/while/TensorArrayV2Read/TensorListGetItemѕ
backward_lstm_618/while/LessLess6backward_lstm_618_while_less_backward_lstm_618_sub_1_0#backward_lstm_618_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2
backward_lstm_618/while/LessЕ
<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOpGbackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02>
<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp•
-backward_lstm_618/while/lstm_cell_1856/MatMulMatMulBbackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem:item:0Dbackward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-backward_lstm_618/while/lstm_cell_1856/MatMulЛ
>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOpIbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02@
>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpО
/backward_lstm_618/while/lstm_cell_1856/MatMul_1MatMul%backward_lstm_618_while_placeholder_3Fbackward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»21
/backward_lstm_618/while/lstm_cell_1856/MatMul_1И
*backward_lstm_618/while/lstm_cell_1856/addAddV27backward_lstm_618/while/lstm_cell_1856/MatMul:product:09backward_lstm_618/while/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2,
*backward_lstm_618/while/lstm_cell_1856/addД
=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOpHbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02?
=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpХ
.backward_lstm_618/while/lstm_cell_1856/BiasAddBiasAdd.backward_lstm_618/while/lstm_cell_1856/add:z:0Ebackward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.backward_lstm_618/while/lstm_cell_1856/BiasAdd≤
6backward_lstm_618/while/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6backward_lstm_618/while/lstm_cell_1856/split/split_dimџ
,backward_lstm_618/while/lstm_cell_1856/splitSplit?backward_lstm_618/while/lstm_cell_1856/split/split_dim:output:07backward_lstm_618/while/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2.
,backward_lstm_618/while/lstm_cell_1856/split‘
.backward_lstm_618/while/lstm_cell_1856/SigmoidSigmoid5backward_lstm_618/while/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€220
.backward_lstm_618/while/lstm_cell_1856/SigmoidЎ
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_1Sigmoid5backward_lstm_618/while/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_1о
*backward_lstm_618/while/lstm_cell_1856/mulMul4backward_lstm_618/while/lstm_cell_1856/Sigmoid_1:y:0%backward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_618/while/lstm_cell_1856/mulЋ
+backward_lstm_618/while/lstm_cell_1856/ReluRelu5backward_lstm_618/while/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22-
+backward_lstm_618/while/lstm_cell_1856/ReluД
,backward_lstm_618/while/lstm_cell_1856/mul_1Mul2backward_lstm_618/while/lstm_cell_1856/Sigmoid:y:09backward_lstm_618/while/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_618/while/lstm_cell_1856/mul_1щ
,backward_lstm_618/while/lstm_cell_1856/add_1AddV2.backward_lstm_618/while/lstm_cell_1856/mul:z:00backward_lstm_618/while/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_618/while/lstm_cell_1856/add_1Ў
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_2Sigmoid5backward_lstm_618/while/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_2 
-backward_lstm_618/while/lstm_cell_1856/Relu_1Relu0backward_lstm_618/while/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-backward_lstm_618/while/lstm_cell_1856/Relu_1И
,backward_lstm_618/while/lstm_cell_1856/mul_2Mul4backward_lstm_618/while/lstm_cell_1856/Sigmoid_2:y:0;backward_lstm_618/while/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_618/while/lstm_cell_1856/mul_2ч
backward_lstm_618/while/SelectSelect backward_lstm_618/while/Less:z:00backward_lstm_618/while/lstm_cell_1856/mul_2:z:0%backward_lstm_618_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22 
backward_lstm_618/while/Selectы
 backward_lstm_618/while/Select_1Select backward_lstm_618/while/Less:z:00backward_lstm_618/while/lstm_cell_1856/mul_2:z:0%backward_lstm_618_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_618/while/Select_1ы
 backward_lstm_618/while/Select_2Select backward_lstm_618/while/Less:z:00backward_lstm_618/while/lstm_cell_1856/add_1:z:0%backward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22"
 backward_lstm_618/while/Select_2≥
<backward_lstm_618/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_618_while_placeholder_1#backward_lstm_618_while_placeholder'backward_lstm_618/while/Select:output:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_618/while/TensorArrayV2Write/TensorListSetItemА
backward_lstm_618/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_618/while/add/y±
backward_lstm_618/while/addAddV2#backward_lstm_618_while_placeholder&backward_lstm_618/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/while/addД
backward_lstm_618/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_618/while/add_1/y–
backward_lstm_618/while/add_1AddV2<backward_lstm_618_while_backward_lstm_618_while_loop_counter(backward_lstm_618/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/while/add_1≥
 backward_lstm_618/while/IdentityIdentity!backward_lstm_618/while/add_1:z:0^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_618/while/IdentityЎ
"backward_lstm_618/while/Identity_1IdentityBbackward_lstm_618_while_backward_lstm_618_while_maximum_iterations^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_618/while/Identity_1µ
"backward_lstm_618/while/Identity_2Identitybackward_lstm_618/while/add:z:0^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_618/while/Identity_2в
"backward_lstm_618/while/Identity_3IdentityLbackward_lstm_618/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_618/while/Identity_3ќ
"backward_lstm_618/while/Identity_4Identity'backward_lstm_618/while/Select:output:0^backward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_618/while/Identity_4–
"backward_lstm_618/while/Identity_5Identity)backward_lstm_618/while/Select_1:output:0^backward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_618/while/Identity_5–
"backward_lstm_618/while/Identity_6Identity)backward_lstm_618/while/Select_2:output:0^backward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_618/while/Identity_6Њ
backward_lstm_618/while/NoOpNoOp>^backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp=^backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp?^backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_618/while/NoOp"x
9backward_lstm_618_while_backward_lstm_618_strided_slice_1;backward_lstm_618_while_backward_lstm_618_strided_slice_1_0"M
 backward_lstm_618_while_identity)backward_lstm_618/while/Identity:output:0"Q
"backward_lstm_618_while_identity_1+backward_lstm_618/while/Identity_1:output:0"Q
"backward_lstm_618_while_identity_2+backward_lstm_618/while/Identity_2:output:0"Q
"backward_lstm_618_while_identity_3+backward_lstm_618/while/Identity_3:output:0"Q
"backward_lstm_618_while_identity_4+backward_lstm_618/while/Identity_4:output:0"Q
"backward_lstm_618_while_identity_5+backward_lstm_618/while/Identity_5:output:0"Q
"backward_lstm_618_while_identity_6+backward_lstm_618/while/Identity_6:output:0"n
4backward_lstm_618_while_less_backward_lstm_618_sub_16backward_lstm_618_while_less_backward_lstm_618_sub_1_0"Т
Fbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resourceHbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0"Ф
Gbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resourceIbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0"Р
Ebackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resourceGbackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0"р
ubackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2~
=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp2|
<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp2А
>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
ф_
≥
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_62326589

inputs@
-lstm_cell_1856_matmul_readvariableop_resource:	»B
/lstm_cell_1856_matmul_1_readvariableop_resource:	2»=
.lstm_cell_1856_biasadd_readvariableop_resource:	»
identityИҐ%lstm_cell_1856/BiasAdd/ReadVariableOpҐ$lstm_cell_1856/MatMul/ReadVariableOpҐ&lstm_cell_1856/MatMul_1/ReadVariableOpҐwhileD
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
$lstm_cell_1856/MatMul/ReadVariableOpReadVariableOp-lstm_cell_1856_matmul_readvariableop_resource*
_output_shapes
:	»*
dtype02&
$lstm_cell_1856/MatMul/ReadVariableOp≥
lstm_cell_1856/MatMulMatMulstrided_slice_2:output:0,lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/MatMulЅ
&lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_1856_matmul_1_readvariableop_resource*
_output_shapes
:	2»*
dtype02(
&lstm_cell_1856/MatMul_1/ReadVariableOpѓ
lstm_cell_1856/MatMul_1MatMulzeros:output:0.lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/MatMul_1®
lstm_cell_1856/addAddV2lstm_cell_1856/MatMul:product:0!lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/addЇ
%lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_1856_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%lstm_cell_1856/BiasAdd/ReadVariableOpµ
lstm_cell_1856/BiasAddBiasAddlstm_cell_1856/add:z:0-lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
lstm_cell_1856/BiasAddВ
lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_cell_1856/split/split_dimы
lstm_cell_1856/splitSplit'lstm_cell_1856/split/split_dim:output:0lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
lstm_cell_1856/splitМ
lstm_cell_1856/SigmoidSigmoidlstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/SigmoidР
lstm_cell_1856/Sigmoid_1Sigmoidlstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Sigmoid_1С
lstm_cell_1856/mulMullstm_cell_1856/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/mulГ
lstm_cell_1856/ReluRelulstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Relu§
lstm_cell_1856/mul_1Mullstm_cell_1856/Sigmoid:y:0!lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/mul_1Щ
lstm_cell_1856/add_1AddV2lstm_cell_1856/mul:z:0lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/add_1Р
lstm_cell_1856/Sigmoid_2Sigmoidlstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Sigmoid_2В
lstm_cell_1856/Relu_1Relulstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/Relu_1®
lstm_cell_1856/mul_2Mullstm_cell_1856/Sigmoid_2:y:0#lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
lstm_cell_1856/mul_2П
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_1856_matmul_readvariableop_resource/lstm_cell_1856_matmul_1_readvariableop_resource.lstm_cell_1856_biasadd_readvariableop_resource*
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
while_body_62326505*
condR
while_cond_62326504*K
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
NoOpNoOp&^lstm_cell_1856/BiasAdd/ReadVariableOp%^lstm_cell_1856/MatMul/ReadVariableOp'^lstm_cell_1856/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_1856/BiasAdd/ReadVariableOp%lstm_cell_1856/BiasAdd/ReadVariableOp2L
$lstm_cell_1856/MatMul/ReadVariableOp$lstm_cell_1856/MatMul/ReadVariableOp2P
&lstm_cell_1856/MatMul_1/ReadVariableOp&lstm_cell_1856/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
я
°
$forward_lstm_618_while_cond_62328498>
:forward_lstm_618_while_forward_lstm_618_while_loop_counterD
@forward_lstm_618_while_forward_lstm_618_while_maximum_iterations&
"forward_lstm_618_while_placeholder(
$forward_lstm_618_while_placeholder_1(
$forward_lstm_618_while_placeholder_2(
$forward_lstm_618_while_placeholder_3@
<forward_lstm_618_while_less_forward_lstm_618_strided_slice_1X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62328498___redundant_placeholder0X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62328498___redundant_placeholder1X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62328498___redundant_placeholder2X
Tforward_lstm_618_while_forward_lstm_618_while_cond_62328498___redundant_placeholder3#
forward_lstm_618_while_identity
≈
forward_lstm_618/while/LessLess"forward_lstm_618_while_placeholder<forward_lstm_618_while_less_forward_lstm_618_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_618/while/LessР
forward_lstm_618/while/IdentityIdentityforward_lstm_618/while/Less:z:0*
T0
*
_output_shapes
: 2!
forward_lstm_618/while/Identity"K
forward_lstm_618_while_identity(forward_lstm_618/while/Identity:output:0*(
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
L__inference_sequential_618_layer_call_and_return_conditional_losses_62328030

inputs
inputs_1	-
bidirectional_618_62328011:	»-
bidirectional_618_62328013:	2»)
bidirectional_618_62328015:	»-
bidirectional_618_62328017:	»-
bidirectional_618_62328019:	2»)
bidirectional_618_62328021:	»$
dense_618_62328024:d 
dense_618_62328026:
identityИҐ)bidirectional_618/StatefulPartitionedCallҐ!dense_618/StatefulPartitionedCall 
)bidirectional_618/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_618_62328011bidirectional_618_62328013bidirectional_618_62328015bidirectional_618_62328017bidirectional_618_62328019bidirectional_618_62328021*
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
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_623278802+
)bidirectional_618/StatefulPartitionedCallЋ
!dense_618/StatefulPartitionedCallStatefulPartitionedCall2bidirectional_618/StatefulPartitionedCall:output:0dense_618_62328024dense_618_62328026*
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
G__inference_dense_618_layer_call_and_return_conditional_losses_623274652#
!dense_618/StatefulPartitionedCallЕ
IdentityIdentity*dense_618/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityЮ
NoOpNoOp*^bidirectional_618/StatefulPartitionedCall"^dense_618/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€:€€€€€€€€€: : : : : : : : 2V
)bidirectional_618/StatefulPartitionedCall)bidirectional_618/StatefulPartitionedCall2F
!dense_618/StatefulPartitionedCall!dense_618/StatefulPartitionedCall:O K
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
%backward_lstm_618_while_body_62328346@
<backward_lstm_618_while_backward_lstm_618_while_loop_counterF
Bbackward_lstm_618_while_backward_lstm_618_while_maximum_iterations'
#backward_lstm_618_while_placeholder)
%backward_lstm_618_while_placeholder_1)
%backward_lstm_618_while_placeholder_2)
%backward_lstm_618_while_placeholder_3?
;backward_lstm_618_while_backward_lstm_618_strided_slice_1_0{
wbackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0Z
Gbackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0:	»\
Ibackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0:	2»W
Hbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0:	»$
 backward_lstm_618_while_identity&
"backward_lstm_618_while_identity_1&
"backward_lstm_618_while_identity_2&
"backward_lstm_618_while_identity_3&
"backward_lstm_618_while_identity_4&
"backward_lstm_618_while_identity_5=
9backward_lstm_618_while_backward_lstm_618_strided_slice_1y
ubackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensorX
Ebackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource:	»Z
Gbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource:	2»U
Fbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource:	»ИҐ=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpҐ<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOpҐ>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpз
Ibackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€€€€€2K
Ibackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shape»
;backward_lstm_618/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwbackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0#backward_lstm_618_while_placeholderRbackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
element_dtype02=
;backward_lstm_618/while/TensorArrayV2Read/TensorListGetItemЕ
<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOpReadVariableOpGbackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02>
<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp•
-backward_lstm_618/while/lstm_cell_1856/MatMulMatMulBbackward_lstm_618/while/TensorArrayV2Read/TensorListGetItem:item:0Dbackward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-backward_lstm_618/while/lstm_cell_1856/MatMulЛ
>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpReadVariableOpIbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02@
>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOpО
/backward_lstm_618/while/lstm_cell_1856/MatMul_1MatMul%backward_lstm_618_while_placeholder_2Fbackward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»21
/backward_lstm_618/while/lstm_cell_1856/MatMul_1И
*backward_lstm_618/while/lstm_cell_1856/addAddV27backward_lstm_618/while/lstm_cell_1856/MatMul:product:09backward_lstm_618/while/lstm_cell_1856/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2,
*backward_lstm_618/while/lstm_cell_1856/addД
=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpReadVariableOpHbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02?
=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOpХ
.backward_lstm_618/while/lstm_cell_1856/BiasAddBiasAdd.backward_lstm_618/while/lstm_cell_1856/add:z:0Ebackward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.backward_lstm_618/while/lstm_cell_1856/BiasAdd≤
6backward_lstm_618/while/lstm_cell_1856/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6backward_lstm_618/while/lstm_cell_1856/split/split_dimџ
,backward_lstm_618/while/lstm_cell_1856/splitSplit?backward_lstm_618/while/lstm_cell_1856/split/split_dim:output:07backward_lstm_618/while/lstm_cell_1856/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2.
,backward_lstm_618/while/lstm_cell_1856/split‘
.backward_lstm_618/while/lstm_cell_1856/SigmoidSigmoid5backward_lstm_618/while/lstm_cell_1856/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€220
.backward_lstm_618/while/lstm_cell_1856/SigmoidЎ
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_1Sigmoid5backward_lstm_618/while/lstm_cell_1856/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_1о
*backward_lstm_618/while/lstm_cell_1856/mulMul4backward_lstm_618/while/lstm_cell_1856/Sigmoid_1:y:0%backward_lstm_618_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22,
*backward_lstm_618/while/lstm_cell_1856/mulЋ
+backward_lstm_618/while/lstm_cell_1856/ReluRelu5backward_lstm_618/while/lstm_cell_1856/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22-
+backward_lstm_618/while/lstm_cell_1856/ReluД
,backward_lstm_618/while/lstm_cell_1856/mul_1Mul2backward_lstm_618/while/lstm_cell_1856/Sigmoid:y:09backward_lstm_618/while/lstm_cell_1856/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_618/while/lstm_cell_1856/mul_1щ
,backward_lstm_618/while/lstm_cell_1856/add_1AddV2.backward_lstm_618/while/lstm_cell_1856/mul:z:00backward_lstm_618/while/lstm_cell_1856/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_618/while/lstm_cell_1856/add_1Ў
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_2Sigmoid5backward_lstm_618/while/lstm_cell_1856/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€222
0backward_lstm_618/while/lstm_cell_1856/Sigmoid_2 
-backward_lstm_618/while/lstm_cell_1856/Relu_1Relu0backward_lstm_618/while/lstm_cell_1856/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-backward_lstm_618/while/lstm_cell_1856/Relu_1И
,backward_lstm_618/while/lstm_cell_1856/mul_2Mul4backward_lstm_618/while/lstm_cell_1856/Sigmoid_2:y:0;backward_lstm_618/while/lstm_cell_1856/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,backward_lstm_618/while/lstm_cell_1856/mul_2Љ
<backward_lstm_618/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%backward_lstm_618_while_placeholder_1#backward_lstm_618_while_placeholder0backward_lstm_618/while/lstm_cell_1856/mul_2:z:0*
_output_shapes
: *
element_dtype02>
<backward_lstm_618/while/TensorArrayV2Write/TensorListSetItemА
backward_lstm_618/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_618/while/add/y±
backward_lstm_618/while/addAddV2#backward_lstm_618_while_placeholder&backward_lstm_618/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/while/addД
backward_lstm_618/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_lstm_618/while/add_1/y–
backward_lstm_618/while/add_1AddV2<backward_lstm_618_while_backward_lstm_618_while_loop_counter(backward_lstm_618/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_618/while/add_1≥
 backward_lstm_618/while/IdentityIdentity!backward_lstm_618/while/add_1:z:0^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_618/while/IdentityЎ
"backward_lstm_618/while/Identity_1IdentityBbackward_lstm_618_while_backward_lstm_618_while_maximum_iterations^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_618/while/Identity_1µ
"backward_lstm_618/while/Identity_2Identitybackward_lstm_618/while/add:z:0^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_618/while/Identity_2в
"backward_lstm_618/while/Identity_3IdentityLbackward_lstm_618/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2$
"backward_lstm_618/while/Identity_3„
"backward_lstm_618/while/Identity_4Identity0backward_lstm_618/while/lstm_cell_1856/mul_2:z:0^backward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_618/while/Identity_4„
"backward_lstm_618/while/Identity_5Identity0backward_lstm_618/while/lstm_cell_1856/add_1:z:0^backward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22$
"backward_lstm_618/while/Identity_5Њ
backward_lstm_618/while/NoOpNoOp>^backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp=^backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp?^backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_618/while/NoOp"x
9backward_lstm_618_while_backward_lstm_618_strided_slice_1;backward_lstm_618_while_backward_lstm_618_strided_slice_1_0"M
 backward_lstm_618_while_identity)backward_lstm_618/while/Identity:output:0"Q
"backward_lstm_618_while_identity_1+backward_lstm_618/while/Identity_1:output:0"Q
"backward_lstm_618_while_identity_2+backward_lstm_618/while/Identity_2:output:0"Q
"backward_lstm_618_while_identity_3+backward_lstm_618/while/Identity_3:output:0"Q
"backward_lstm_618_while_identity_4+backward_lstm_618/while/Identity_4:output:0"Q
"backward_lstm_618_while_identity_5+backward_lstm_618/while/Identity_5:output:0"Т
Fbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resourceHbackward_lstm_618_while_lstm_cell_1856_biasadd_readvariableop_resource_0"Ф
Gbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resourceIbackward_lstm_618_while_lstm_cell_1856_matmul_1_readvariableop_resource_0"Р
Ebackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resourceGbackward_lstm_618_while_lstm_cell_1856_matmul_readvariableop_resource_0"р
ubackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensorwbackward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2~
=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp=backward_lstm_618/while/lstm_cell_1856/BiasAdd/ReadVariableOp2|
<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp<backward_lstm_618/while/lstm_cell_1856/MatMul/ReadVariableOp2А
>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp>backward_lstm_618/while/lstm_cell_1856/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
3__inference_forward_lstm_618_layer_call_fn_62329503

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
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_623264292
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
э
µ
%backward_lstm_618_while_cond_62328647@
<backward_lstm_618_while_backward_lstm_618_while_loop_counterF
Bbackward_lstm_618_while_backward_lstm_618_while_maximum_iterations'
#backward_lstm_618_while_placeholder)
%backward_lstm_618_while_placeholder_1)
%backward_lstm_618_while_placeholder_2)
%backward_lstm_618_while_placeholder_3B
>backward_lstm_618_while_less_backward_lstm_618_strided_slice_1Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62328647___redundant_placeholder0Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62328647___redundant_placeholder1Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62328647___redundant_placeholder2Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62328647___redundant_placeholder3$
 backward_lstm_618_while_identity
 
backward_lstm_618/while/LessLess#backward_lstm_618_while_placeholder>backward_lstm_618_while_less_backward_lstm_618_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_618/while/LessУ
 backward_lstm_618/while/IdentityIdentity backward_lstm_618/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_618/while/Identity"M
 backward_lstm_618_while_identity)backward_lstm_618/while/Identity:output:0*(
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
%backward_lstm_618_while_cond_62328994@
<backward_lstm_618_while_backward_lstm_618_while_loop_counterF
Bbackward_lstm_618_while_backward_lstm_618_while_maximum_iterations'
#backward_lstm_618_while_placeholder)
%backward_lstm_618_while_placeholder_1)
%backward_lstm_618_while_placeholder_2)
%backward_lstm_618_while_placeholder_3)
%backward_lstm_618_while_placeholder_4B
>backward_lstm_618_while_less_backward_lstm_618_strided_slice_1Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62328994___redundant_placeholder0Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62328994___redundant_placeholder1Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62328994___redundant_placeholder2Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62328994___redundant_placeholder3Z
Vbackward_lstm_618_while_backward_lstm_618_while_cond_62328994___redundant_placeholder4$
 backward_lstm_618_while_identity
 
backward_lstm_618/while/LessLess#backward_lstm_618_while_placeholder>backward_lstm_618_while_less_backward_lstm_618_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_618/while/LessУ
 backward_lstm_618/while/IdentityIdentity backward_lstm_618/while/Less:z:0*
T0
*
_output_shapes
: 2"
 backward_lstm_618/while/Identity"M
 backward_lstm_618_while_identity)backward_lstm_618/while/Identity:output:0*(
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
$forward_lstm_618_while_body_62329174>
:forward_lstm_618_while_forward_lstm_618_while_loop_counterD
@forward_lstm_618_while_forward_lstm_618_while_maximum_iterations&
"forward_lstm_618_while_placeholder(
$forward_lstm_618_while_placeholder_1(
$forward_lstm_618_while_placeholder_2(
$forward_lstm_618_while_placeholder_3(
$forward_lstm_618_while_placeholder_4=
9forward_lstm_618_while_forward_lstm_618_strided_slice_1_0y
uforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_618_while_greater_forward_lstm_618_cast_0Y
Fforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0:	»[
Hforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0:	2»V
Gforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0:	»#
forward_lstm_618_while_identity%
!forward_lstm_618_while_identity_1%
!forward_lstm_618_while_identity_2%
!forward_lstm_618_while_identity_3%
!forward_lstm_618_while_identity_4%
!forward_lstm_618_while_identity_5%
!forward_lstm_618_while_identity_6;
7forward_lstm_618_while_forward_lstm_618_strided_slice_1w
sforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_618_while_greater_forward_lstm_618_castW
Dforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource:	»Y
Fforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource:	2»T
Eforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource:	»ИҐ<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpҐ;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOpҐ=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpе
Hforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2J
Hforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeє
:forward_lstm_618/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_618_while_placeholderQforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02<
:forward_lstm_618/while/TensorArrayV2Read/TensorListGetItem’
forward_lstm_618/while/GreaterGreater6forward_lstm_618_while_greater_forward_lstm_618_cast_0"forward_lstm_618_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2 
forward_lstm_618/while/GreaterВ
;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOpFforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02=
;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp°
,forward_lstm_618/while/lstm_cell_1855/MatMulMatMulAforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem:item:0Cforward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2.
,forward_lstm_618/while/lstm_cell_1855/MatMulИ
=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOpHforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02?
=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpК
.forward_lstm_618/while/lstm_cell_1855/MatMul_1MatMul$forward_lstm_618_while_placeholder_3Eforward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.forward_lstm_618/while/lstm_cell_1855/MatMul_1Д
)forward_lstm_618/while/lstm_cell_1855/addAddV26forward_lstm_618/while/lstm_cell_1855/MatMul:product:08forward_lstm_618/while/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)forward_lstm_618/while/lstm_cell_1855/addБ
<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOpGforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02>
<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpС
-forward_lstm_618/while/lstm_cell_1855/BiasAddBiasAdd-forward_lstm_618/while/lstm_cell_1855/add:z:0Dforward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-forward_lstm_618/while/lstm_cell_1855/BiasAdd∞
5forward_lstm_618/while/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5forward_lstm_618/while/lstm_cell_1855/split/split_dim„
+forward_lstm_618/while/lstm_cell_1855/splitSplit>forward_lstm_618/while/lstm_cell_1855/split/split_dim:output:06forward_lstm_618/while/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2-
+forward_lstm_618/while/lstm_cell_1855/split—
-forward_lstm_618/while/lstm_cell_1855/SigmoidSigmoid4forward_lstm_618/while/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-forward_lstm_618/while/lstm_cell_1855/Sigmoid’
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_1Sigmoid4forward_lstm_618/while/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_1к
)forward_lstm_618/while/lstm_cell_1855/mulMul3forward_lstm_618/while/lstm_cell_1855/Sigmoid_1:y:0$forward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_618/while/lstm_cell_1855/mul»
*forward_lstm_618/while/lstm_cell_1855/ReluRelu4forward_lstm_618/while/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22,
*forward_lstm_618/while/lstm_cell_1855/ReluА
+forward_lstm_618/while/lstm_cell_1855/mul_1Mul1forward_lstm_618/while/lstm_cell_1855/Sigmoid:y:08forward_lstm_618/while/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_618/while/lstm_cell_1855/mul_1х
+forward_lstm_618/while/lstm_cell_1855/add_1AddV2-forward_lstm_618/while/lstm_cell_1855/mul:z:0/forward_lstm_618/while/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_618/while/lstm_cell_1855/add_1’
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_2Sigmoid4forward_lstm_618/while/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_2«
,forward_lstm_618/while/lstm_cell_1855/Relu_1Relu/forward_lstm_618/while/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,forward_lstm_618/while/lstm_cell_1855/Relu_1Д
+forward_lstm_618/while/lstm_cell_1855/mul_2Mul3forward_lstm_618/while/lstm_cell_1855/Sigmoid_2:y:0:forward_lstm_618/while/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_618/while/lstm_cell_1855/mul_2х
forward_lstm_618/while/SelectSelect"forward_lstm_618/while/Greater:z:0/forward_lstm_618/while/lstm_cell_1855/mul_2:z:0$forward_lstm_618_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/while/Selectщ
forward_lstm_618/while/Select_1Select"forward_lstm_618/while/Greater:z:0/forward_lstm_618/while/lstm_cell_1855/mul_2:z:0$forward_lstm_618_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_618/while/Select_1щ
forward_lstm_618/while/Select_2Select"forward_lstm_618/while/Greater:z:0/forward_lstm_618/while/lstm_cell_1855/add_1:z:0$forward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_618/while/Select_2Ѓ
;forward_lstm_618/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_618_while_placeholder_1"forward_lstm_618_while_placeholder&forward_lstm_618/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_618/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_618/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_618/while/add/y≠
forward_lstm_618/while/addAddV2"forward_lstm_618_while_placeholder%forward_lstm_618/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/while/addВ
forward_lstm_618/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_618/while/add_1/yЋ
forward_lstm_618/while/add_1AddV2:forward_lstm_618_while_forward_lstm_618_while_loop_counter'forward_lstm_618/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/while/add_1ѓ
forward_lstm_618/while/IdentityIdentity forward_lstm_618/while/add_1:z:0^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_618/while/Identity”
!forward_lstm_618/while/Identity_1Identity@forward_lstm_618_while_forward_lstm_618_while_maximum_iterations^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_618/while/Identity_1±
!forward_lstm_618/while/Identity_2Identityforward_lstm_618/while/add:z:0^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_618/while/Identity_2ё
!forward_lstm_618/while/Identity_3IdentityKforward_lstm_618/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_618/while/Identity_3 
!forward_lstm_618/while/Identity_4Identity&forward_lstm_618/while/Select:output:0^forward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_618/while/Identity_4ћ
!forward_lstm_618/while/Identity_5Identity(forward_lstm_618/while/Select_1:output:0^forward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_618/while/Identity_5ћ
!forward_lstm_618/while/Identity_6Identity(forward_lstm_618/while/Select_2:output:0^forward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_618/while/Identity_6є
forward_lstm_618/while/NoOpNoOp=^forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp<^forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp>^forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_618/while/NoOp"t
7forward_lstm_618_while_forward_lstm_618_strided_slice_19forward_lstm_618_while_forward_lstm_618_strided_slice_1_0"n
4forward_lstm_618_while_greater_forward_lstm_618_cast6forward_lstm_618_while_greater_forward_lstm_618_cast_0"K
forward_lstm_618_while_identity(forward_lstm_618/while/Identity:output:0"O
!forward_lstm_618_while_identity_1*forward_lstm_618/while/Identity_1:output:0"O
!forward_lstm_618_while_identity_2*forward_lstm_618/while/Identity_2:output:0"O
!forward_lstm_618_while_identity_3*forward_lstm_618/while/Identity_3:output:0"O
!forward_lstm_618_while_identity_4*forward_lstm_618/while/Identity_4:output:0"O
!forward_lstm_618_while_identity_5*forward_lstm_618/while/Identity_5:output:0"O
!forward_lstm_618_while_identity_6*forward_lstm_618/while/Identity_6:output:0"Р
Eforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resourceGforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0"Т
Fforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resourceHforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0"О
Dforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resourceFforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0"м
sforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensoruforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2|
<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp2z
;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp2~
=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Ѓ

•
4__inference_bidirectional_618_layer_call_fn_62328130

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
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_623278802
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
ч
И
L__inference_lstm_cell_1855_layer_call_and_return_conditional_losses_62325225

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
while_body_62329883
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_1855_matmul_readvariableop_resource_0:	»J
7while_lstm_cell_1855_matmul_1_readvariableop_resource_0:	2»E
6while_lstm_cell_1855_biasadd_readvariableop_resource_0:	»
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_1855_matmul_readvariableop_resource:	»H
5while_lstm_cell_1855_matmul_1_readvariableop_resource:	2»C
4while_lstm_cell_1855_biasadd_readvariableop_resource:	»ИҐ+while/lstm_cell_1855/BiasAdd/ReadVariableOpҐ*while/lstm_cell_1855/MatMul/ReadVariableOpҐ,while/lstm_cell_1855/MatMul_1/ReadVariableOp√
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
*while/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_1855_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02,
*while/lstm_cell_1855/MatMul/ReadVariableOpЁ
while/lstm_cell_1855/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/MatMul’
,while/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_1855_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02.
,while/lstm_cell_1855/MatMul_1/ReadVariableOp∆
while/lstm_cell_1855/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/MatMul_1ј
while/lstm_cell_1855/addAddV2%while/lstm_cell_1855/MatMul:product:0'while/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/addќ
+while/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_1855_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02-
+while/lstm_cell_1855/BiasAdd/ReadVariableOpЌ
while/lstm_cell_1855/BiasAddBiasAddwhile/lstm_cell_1855/add:z:03while/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
while/lstm_cell_1855/BiasAddО
$while/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$while/lstm_cell_1855/split/split_dimУ
while/lstm_cell_1855/splitSplit-while/lstm_cell_1855/split/split_dim:output:0%while/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2
while/lstm_cell_1855/splitЮ
while/lstm_cell_1855/SigmoidSigmoid#while/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/SigmoidҐ
while/lstm_cell_1855/Sigmoid_1Sigmoid#while/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1855/Sigmoid_1¶
while/lstm_cell_1855/mulMul"while/lstm_cell_1855/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/mulХ
while/lstm_cell_1855/ReluRelu#while/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/ReluЉ
while/lstm_cell_1855/mul_1Mul while/lstm_cell_1855/Sigmoid:y:0'while/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/mul_1±
while/lstm_cell_1855/add_1AddV2while/lstm_cell_1855/mul:z:0while/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/add_1Ґ
while/lstm_cell_1855/Sigmoid_2Sigmoid#while/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€22 
while/lstm_cell_1855/Sigmoid_2Ф
while/lstm_cell_1855/Relu_1Reluwhile/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/Relu_1ј
while/lstm_cell_1855/mul_2Mul"while/lstm_cell_1855/Sigmoid_2:y:0)while/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22
while/lstm_cell_1855/mul_2в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1855/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_1855/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_4П
while/Identity_5Identitywhile/lstm_cell_1855/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22
while/Identity_5д

while/NoOpNoOp,^while/lstm_cell_1855/BiasAdd/ReadVariableOp+^while/lstm_cell_1855/MatMul/ReadVariableOp-^while/lstm_cell_1855/MatMul_1/ReadVariableOp*"
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
4while_lstm_cell_1855_biasadd_readvariableop_resource6while_lstm_cell_1855_biasadd_readvariableop_resource_0"p
5while_lstm_cell_1855_matmul_1_readvariableop_resource7while_lstm_cell_1855_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_1855_matmul_readvariableop_resource5while_lstm_cell_1855_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€2:€€€€€€€€€2: : : : : 2Z
+while/lstm_cell_1855/BiasAdd/ReadVariableOp+while/lstm_cell_1855/BiasAdd/ReadVariableOp2X
*while/lstm_cell_1855/MatMul/ReadVariableOp*while/lstm_cell_1855/MatMul/ReadVariableOp2\
,while/lstm_cell_1855/MatMul_1/ReadVariableOp,while/lstm_cell_1855/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
‘
¬
3__inference_forward_lstm_618_layer_call_fn_62329481
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
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_623251622
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
зe
Ћ
$forward_lstm_618_while_body_62328816>
:forward_lstm_618_while_forward_lstm_618_while_loop_counterD
@forward_lstm_618_while_forward_lstm_618_while_maximum_iterations&
"forward_lstm_618_while_placeholder(
$forward_lstm_618_while_placeholder_1(
$forward_lstm_618_while_placeholder_2(
$forward_lstm_618_while_placeholder_3(
$forward_lstm_618_while_placeholder_4=
9forward_lstm_618_while_forward_lstm_618_strided_slice_1_0y
uforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0:
6forward_lstm_618_while_greater_forward_lstm_618_cast_0Y
Fforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0:	»[
Hforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0:	2»V
Gforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0:	»#
forward_lstm_618_while_identity%
!forward_lstm_618_while_identity_1%
!forward_lstm_618_while_identity_2%
!forward_lstm_618_while_identity_3%
!forward_lstm_618_while_identity_4%
!forward_lstm_618_while_identity_5%
!forward_lstm_618_while_identity_6;
7forward_lstm_618_while_forward_lstm_618_strided_slice_1w
sforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor8
4forward_lstm_618_while_greater_forward_lstm_618_castW
Dforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource:	»Y
Fforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource:	2»T
Eforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource:	»ИҐ<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpҐ;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOpҐ=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpе
Hforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2J
Hforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shapeє
:forward_lstm_618/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemuforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0"forward_lstm_618_while_placeholderQforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype02<
:forward_lstm_618/while/TensorArrayV2Read/TensorListGetItem’
forward_lstm_618/while/GreaterGreater6forward_lstm_618_while_greater_forward_lstm_618_cast_0"forward_lstm_618_while_placeholder*
T0*#
_output_shapes
:€€€€€€€€€2 
forward_lstm_618/while/GreaterВ
;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOpReadVariableOpFforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0*
_output_shapes
:	»*
dtype02=
;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp°
,forward_lstm_618/while/lstm_cell_1855/MatMulMatMulAforward_lstm_618/while/TensorArrayV2Read/TensorListGetItem:item:0Cforward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2.
,forward_lstm_618/while/lstm_cell_1855/MatMulИ
=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpReadVariableOpHforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0*
_output_shapes
:	2»*
dtype02?
=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOpК
.forward_lstm_618/while/lstm_cell_1855/MatMul_1MatMul$forward_lstm_618_while_placeholder_3Eforward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»20
.forward_lstm_618/while/lstm_cell_1855/MatMul_1Д
)forward_lstm_618/while/lstm_cell_1855/addAddV26forward_lstm_618/while/lstm_cell_1855/MatMul:product:08forward_lstm_618/while/lstm_cell_1855/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€»2+
)forward_lstm_618/while/lstm_cell_1855/addБ
<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpReadVariableOpGforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0*
_output_shapes	
:»*
dtype02>
<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOpС
-forward_lstm_618/while/lstm_cell_1855/BiasAddBiasAdd-forward_lstm_618/while/lstm_cell_1855/add:z:0Dforward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2/
-forward_lstm_618/while/lstm_cell_1855/BiasAdd∞
5forward_lstm_618/while/lstm_cell_1855/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5forward_lstm_618/while/lstm_cell_1855/split/split_dim„
+forward_lstm_618/while/lstm_cell_1855/splitSplit>forward_lstm_618/while/lstm_cell_1855/split/split_dim:output:06forward_lstm_618/while/lstm_cell_1855/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2*
	num_split2-
+forward_lstm_618/while/lstm_cell_1855/split—
-forward_lstm_618/while/lstm_cell_1855/SigmoidSigmoid4forward_lstm_618/while/lstm_cell_1855/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€22/
-forward_lstm_618/while/lstm_cell_1855/Sigmoid’
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_1Sigmoid4forward_lstm_618/while/lstm_cell_1855/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_1к
)forward_lstm_618/while/lstm_cell_1855/mulMul3forward_lstm_618/while/lstm_cell_1855/Sigmoid_1:y:0$forward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22+
)forward_lstm_618/while/lstm_cell_1855/mul»
*forward_lstm_618/while/lstm_cell_1855/ReluRelu4forward_lstm_618/while/lstm_cell_1855/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€22,
*forward_lstm_618/while/lstm_cell_1855/ReluА
+forward_lstm_618/while/lstm_cell_1855/mul_1Mul1forward_lstm_618/while/lstm_cell_1855/Sigmoid:y:08forward_lstm_618/while/lstm_cell_1855/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_618/while/lstm_cell_1855/mul_1х
+forward_lstm_618/while/lstm_cell_1855/add_1AddV2-forward_lstm_618/while/lstm_cell_1855/mul:z:0/forward_lstm_618/while/lstm_cell_1855/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_618/while/lstm_cell_1855/add_1’
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_2Sigmoid4forward_lstm_618/while/lstm_cell_1855/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€221
/forward_lstm_618/while/lstm_cell_1855/Sigmoid_2«
,forward_lstm_618/while/lstm_cell_1855/Relu_1Relu/forward_lstm_618/while/lstm_cell_1855/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€22.
,forward_lstm_618/while/lstm_cell_1855/Relu_1Д
+forward_lstm_618/while/lstm_cell_1855/mul_2Mul3forward_lstm_618/while/lstm_cell_1855/Sigmoid_2:y:0:forward_lstm_618/while/lstm_cell_1855/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22-
+forward_lstm_618/while/lstm_cell_1855/mul_2х
forward_lstm_618/while/SelectSelect"forward_lstm_618/while/Greater:z:0/forward_lstm_618/while/lstm_cell_1855/mul_2:z:0$forward_lstm_618_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€22
forward_lstm_618/while/Selectщ
forward_lstm_618/while/Select_1Select"forward_lstm_618/while/Greater:z:0/forward_lstm_618/while/lstm_cell_1855/mul_2:z:0$forward_lstm_618_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_618/while/Select_1щ
forward_lstm_618/while/Select_2Select"forward_lstm_618/while/Greater:z:0/forward_lstm_618/while/lstm_cell_1855/add_1:z:0$forward_lstm_618_while_placeholder_4*
T0*'
_output_shapes
:€€€€€€€€€22!
forward_lstm_618/while/Select_2Ѓ
;forward_lstm_618/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$forward_lstm_618_while_placeholder_1"forward_lstm_618_while_placeholder&forward_lstm_618/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;forward_lstm_618/while/TensorArrayV2Write/TensorListSetItem~
forward_lstm_618/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_618/while/add/y≠
forward_lstm_618/while/addAddV2"forward_lstm_618_while_placeholder%forward_lstm_618/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/while/addВ
forward_lstm_618/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_lstm_618/while/add_1/yЋ
forward_lstm_618/while/add_1AddV2:forward_lstm_618_while_forward_lstm_618_while_loop_counter'forward_lstm_618/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_618/while/add_1ѓ
forward_lstm_618/while/IdentityIdentity forward_lstm_618/while/add_1:z:0^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_618/while/Identity”
!forward_lstm_618/while/Identity_1Identity@forward_lstm_618_while_forward_lstm_618_while_maximum_iterations^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_618/while/Identity_1±
!forward_lstm_618/while/Identity_2Identityforward_lstm_618/while/add:z:0^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_618/while/Identity_2ё
!forward_lstm_618/while/Identity_3IdentityKforward_lstm_618/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_618/while/NoOp*
T0*
_output_shapes
: 2#
!forward_lstm_618/while/Identity_3 
!forward_lstm_618/while/Identity_4Identity&forward_lstm_618/while/Select:output:0^forward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_618/while/Identity_4ћ
!forward_lstm_618/while/Identity_5Identity(forward_lstm_618/while/Select_1:output:0^forward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_618/while/Identity_5ћ
!forward_lstm_618/while/Identity_6Identity(forward_lstm_618/while/Select_2:output:0^forward_lstm_618/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€22#
!forward_lstm_618/while/Identity_6є
forward_lstm_618/while/NoOpNoOp=^forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp<^forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp>^forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_618/while/NoOp"t
7forward_lstm_618_while_forward_lstm_618_strided_slice_19forward_lstm_618_while_forward_lstm_618_strided_slice_1_0"n
4forward_lstm_618_while_greater_forward_lstm_618_cast6forward_lstm_618_while_greater_forward_lstm_618_cast_0"K
forward_lstm_618_while_identity(forward_lstm_618/while/Identity:output:0"O
!forward_lstm_618_while_identity_1*forward_lstm_618/while/Identity_1:output:0"O
!forward_lstm_618_while_identity_2*forward_lstm_618/while/Identity_2:output:0"O
!forward_lstm_618_while_identity_3*forward_lstm_618/while/Identity_3:output:0"O
!forward_lstm_618_while_identity_4*forward_lstm_618/while/Identity_4:output:0"O
!forward_lstm_618_while_identity_5*forward_lstm_618/while/Identity_5:output:0"O
!forward_lstm_618_while_identity_6*forward_lstm_618/while/Identity_6:output:0"Р
Eforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resourceGforward_lstm_618_while_lstm_cell_1855_biasadd_readvariableop_resource_0"Т
Fforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resourceHforward_lstm_618_while_lstm_cell_1855_matmul_1_readvariableop_resource_0"О
Dforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resourceFforward_lstm_618_while_lstm_cell_1855_matmul_readvariableop_resource_0"м
sforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensoruforward_lstm_618_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_618_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :€€€€€€€€€2:€€€€€€€€€2:€€€€€€€€€2: : :€€€€€€€€€: : : 2|
<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp<forward_lstm_618/while/lstm_cell_1855/BiasAdd/ReadVariableOp2z
;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp;forward_lstm_618/while/lstm_cell_1855/MatMul/ReadVariableOp2~
=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp=forward_lstm_618/while/lstm_cell_1855/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
:€€€€€€€€€"®L
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
	dense_6180
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
": d2dense_618/kernel
:2dense_618/bias
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
K:I	»28bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel
U:S	2»2Bbidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel
E:C»26bidirectional_618/forward_lstm_618/lstm_cell_1855/bias
L:J	»29bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel
V:T	2»2Cbidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel
F:D»27bidirectional_618/backward_lstm_618/lstm_cell_1856/bias
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
':%d2Adam/dense_618/kernel/m
!:2Adam/dense_618/bias/m
P:N	»2?Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/m
Z:X	2»2IAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/m
J:H»2=Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/m
Q:O	»2@Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/m
[:Y	2»2JAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/m
K:I»2>Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/m
':%d2Adam/dense_618/kernel/v
!:2Adam/dense_618/bias/v
P:N	»2?Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/v
Z:X	2»2IAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/v
J:H»2=Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/v
Q:O	»2@Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/v
[:Y	2»2JAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/v
K:I»2>Adam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/v
*:(d2Adam/dense_618/kernel/vhat
$:"2Adam/dense_618/bias/vhat
S:Q	»2BAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/kernel/vhat
]:[	2»2LAdam/bidirectional_618/forward_lstm_618/lstm_cell_1855/recurrent_kernel/vhat
M:K»2@Adam/bidirectional_618/forward_lstm_618/lstm_cell_1855/bias/vhat
T:R	»2CAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/kernel/vhat
^:\	2»2MAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/recurrent_kernel/vhat
N:L»2AAdam/bidirectional_618/backward_lstm_618/lstm_cell_1856/bias/vhat
ђ2©
1__inference_sequential_618_layer_call_fn_62327491
1__inference_sequential_618_layer_call_fn_62327984ј
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
#__inference__wrapped_model_62325004args_0args_0_1"Ш
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
L__inference_sequential_618_layer_call_and_return_conditional_losses_62328007
L__inference_sequential_618_layer_call_and_return_conditional_losses_62328030ј
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
4__inference_bidirectional_618_layer_call_fn_62328077
4__inference_bidirectional_618_layer_call_fn_62328094
4__inference_bidirectional_618_layer_call_fn_62328112
4__inference_bidirectional_618_layer_call_fn_62328130ж
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
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_62328432
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_62328734
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_62329092
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_62329450ж
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
,__inference_dense_618_layer_call_fn_62329459Ґ
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
G__inference_dense_618_layer_call_and_return_conditional_losses_62329470Ґ
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
&__inference_signature_wrapper_62328060args_0args_0_1"Ф
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
3__inference_forward_lstm_618_layer_call_fn_62329481
3__inference_forward_lstm_618_layer_call_fn_62329492
3__inference_forward_lstm_618_layer_call_fn_62329503
3__inference_forward_lstm_618_layer_call_fn_62329514’
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
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_62329665
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_62329816
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_62329967
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_62330118’
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
4__inference_backward_lstm_618_layer_call_fn_62330129
4__inference_backward_lstm_618_layer_call_fn_62330140
4__inference_backward_lstm_618_layer_call_fn_62330151
4__inference_backward_lstm_618_layer_call_fn_62330162’
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
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_62330315
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_62330468
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_62330621
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_62330774’
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
1__inference_lstm_cell_1855_layer_call_fn_62330791
1__inference_lstm_cell_1855_layer_call_fn_62330808Њ
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
L__inference_lstm_cell_1855_layer_call_and_return_conditional_losses_62330840
L__inference_lstm_cell_1855_layer_call_and_return_conditional_losses_62330872Њ
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
1__inference_lstm_cell_1856_layer_call_fn_62330889
1__inference_lstm_cell_1856_layer_call_fn_62330906Њ
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
L__inference_lstm_cell_1856_layer_call_and_return_conditional_losses_62330938
L__inference_lstm_cell_1856_layer_call_and_return_conditional_losses_62330970Њ
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
#__inference__wrapped_model_62325004Я\ҐY
RҐO
MТJ4Ґ1
!ъ€€€€€€€€€€€€€€€€€€
А
`
А	RaggedTensorSpec
™ "5™2
0
	dense_618#К 
	dense_618€€€€€€€€€–
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_62330315}OҐL
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
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_62330468}OҐL
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
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_62330621QҐN
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
O__inference_backward_lstm_618_layer_call_and_return_conditional_losses_62330774QҐN
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
4__inference_backward_lstm_618_layer_call_fn_62330129pOҐL
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
4__inference_backward_lstm_618_layer_call_fn_62330140pOҐL
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
4__inference_backward_lstm_618_layer_call_fn_62330151rQҐN
GҐD
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€2™
4__inference_backward_lstm_618_layer_call_fn_62330162rQҐN
GҐD
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
p

 
™ "К€€€€€€€€€2б
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_62328432Н\ҐY
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
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_62328734Н\ҐY
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
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_62329092ЭlҐi
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
O__inference_bidirectional_618_layer_call_and_return_conditional_losses_62329450ЭlҐi
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
4__inference_bidirectional_618_layer_call_fn_62328077А\ҐY
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
4__inference_bidirectional_618_layer_call_fn_62328094А\ҐY
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
4__inference_bidirectional_618_layer_call_fn_62328112РlҐi
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
4__inference_bidirectional_618_layer_call_fn_62328130РlҐi
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
G__inference_dense_618_layer_call_and_return_conditional_losses_62329470\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "%Ґ"
К
0€€€€€€€€€
Ъ 
,__inference_dense_618_layer_call_fn_62329459O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "К€€€€€€€€€ѕ
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_62329665}OҐL
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
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_62329816}OҐL
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
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_62329967QҐN
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
N__inference_forward_lstm_618_layer_call_and_return_conditional_losses_62330118QҐN
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
3__inference_forward_lstm_618_layer_call_fn_62329481pOҐL
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
3__inference_forward_lstm_618_layer_call_fn_62329492pOҐL
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
3__inference_forward_lstm_618_layer_call_fn_62329503rQҐN
GҐD
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€2©
3__inference_forward_lstm_618_layer_call_fn_62329514rQҐN
GҐD
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
p

 
™ "К€€€€€€€€€2ќ
L__inference_lstm_cell_1855_layer_call_and_return_conditional_losses_62330840эАҐ}
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
L__inference_lstm_cell_1855_layer_call_and_return_conditional_losses_62330872эАҐ}
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
1__inference_lstm_cell_1855_layer_call_fn_62330791нАҐ}
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
1__inference_lstm_cell_1855_layer_call_fn_62330808нАҐ}
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
L__inference_lstm_cell_1856_layer_call_and_return_conditional_losses_62330938эАҐ}
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
L__inference_lstm_cell_1856_layer_call_and_return_conditional_losses_62330970эАҐ}
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
1__inference_lstm_cell_1856_layer_call_fn_62330889нАҐ}
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
1__inference_lstm_cell_1856_layer_call_fn_62330906нАҐ}
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
L__inference_sequential_618_layer_call_and_return_conditional_losses_62328007ЧdҐa
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
L__inference_sequential_618_layer_call_and_return_conditional_losses_62328030ЧdҐa
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
1__inference_sequential_618_layer_call_fn_62327491КdҐa
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
1__inference_sequential_618_layer_call_fn_62327984КdҐa
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
&__inference_signature_wrapper_62328060®eҐb
Ґ 
[™X
*
args_0 К
args_0€€€€€€€€€
*
args_0_1К
args_0_1€€€€€€€€€	"5™2
0
	dense_618#К 
	dense_618€€€€€€€€€