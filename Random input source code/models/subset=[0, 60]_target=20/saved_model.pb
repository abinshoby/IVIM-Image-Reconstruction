М:
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
Ttype"serve*2.6.02v2.6.0-rc2-32-g919f693420e8мс8
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:d*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
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
П
1bidirectional_2/forward_lstm_2/lstm_cell_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*B
shared_name31bidirectional_2/forward_lstm_2/lstm_cell_7/kernel
И
Ebidirectional_2/forward_lstm_2/lstm_cell_7/kernel/Read/ReadVariableOpReadVariableOp1bidirectional_2/forward_lstm_2/lstm_cell_7/kernel*
_output_shapes
:	Ш*
dtype0
г
;bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*L
shared_name=;bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel
Ь
Obidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/Read/ReadVariableOpReadVariableOp;bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel*
_output_shapes
:	2Ш*
dtype0
З
/bidirectional_2/forward_lstm_2/lstm_cell_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*@
shared_name1/bidirectional_2/forward_lstm_2/lstm_cell_7/bias
А
Cbidirectional_2/forward_lstm_2/lstm_cell_7/bias/Read/ReadVariableOpReadVariableOp/bidirectional_2/forward_lstm_2/lstm_cell_7/bias*
_output_shapes	
:Ш*
dtype0
С
2bidirectional_2/backward_lstm_2/lstm_cell_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*C
shared_name42bidirectional_2/backward_lstm_2/lstm_cell_8/kernel
К
Fbidirectional_2/backward_lstm_2/lstm_cell_8/kernel/Read/ReadVariableOpReadVariableOp2bidirectional_2/backward_lstm_2/lstm_cell_8/kernel*
_output_shapes
:	Ш*
dtype0
е
<bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*M
shared_name><bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel
Ю
Pbidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/Read/ReadVariableOpReadVariableOp<bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel*
_output_shapes
:	2Ш*
dtype0
Й
0bidirectional_2/backward_lstm_2/lstm_cell_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*A
shared_name20bidirectional_2/backward_lstm_2/lstm_cell_8/bias
В
Dbidirectional_2/backward_lstm_2/lstm_cell_8/bias/Read/ReadVariableOpReadVariableOp0bidirectional_2/backward_lstm_2/lstm_cell_8/bias*
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

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:d*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0
Э
8Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*I
shared_name:8Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/m
Ц
LAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/m*
_output_shapes
:	Ш*
dtype0
с
BAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*S
shared_nameDBAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/m
к
VAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpBAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/m*
_output_shapes
:	2Ш*
dtype0
Х
6Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*G
shared_name86Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/m
О
JAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/m/Read/ReadVariableOpReadVariableOp6Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/m*
_output_shapes	
:Ш*
dtype0
Я
9Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*J
shared_name;9Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/m
Ш
MAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/m/Read/ReadVariableOpReadVariableOp9Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/m*
_output_shapes
:	Ш*
dtype0
у
CAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*T
shared_nameECAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/m
м
WAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpCAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/m*
_output_shapes
:	2Ш*
dtype0
Ч
7Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*H
shared_name97Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/m
Р
KAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/m/Read/ReadVariableOpReadVariableOp7Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/m*
_output_shapes	
:Ш*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:d*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0
Э
8Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*I
shared_name:8Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/v
Ц
LAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/v*
_output_shapes
:	Ш*
dtype0
с
BAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*S
shared_nameDBAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/v
к
VAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpBAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/v*
_output_shapes
:	2Ш*
dtype0
Х
6Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*G
shared_name86Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/v
О
JAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/v/Read/ReadVariableOpReadVariableOp6Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/v*
_output_shapes	
:Ш*
dtype0
Я
9Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*J
shared_name;9Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/v
Ш
MAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/v/Read/ReadVariableOpReadVariableOp9Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/v*
_output_shapes
:	Ш*
dtype0
у
CAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*T
shared_nameECAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/v
м
WAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpCAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/v*
_output_shapes
:	2Ш*
dtype0
Ч
7Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*H
shared_name97Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/v
Р
KAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/v/Read/ReadVariableOpReadVariableOp7Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/v*
_output_shapes	
:Ш*
dtype0

Adam/dense_2/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*)
shared_nameAdam/dense_2/kernel/vhat

,Adam/dense_2/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/vhat*
_output_shapes

:d*
dtype0

Adam/dense_2/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_2/bias/vhat
}
*Adam/dense_2/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/vhat*
_output_shapes
:*
dtype0
г
;Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*L
shared_name=;Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/vhat
Ь
OAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/vhat/Read/ReadVariableOpReadVariableOp;Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/vhat*
_output_shapes
:	Ш*
dtype0
ч
EAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*V
shared_nameGEAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/vhat
р
YAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpEAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/vhat*
_output_shapes
:	2Ш*
dtype0
Ы
9Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*J
shared_name;9Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/vhat
Ф
MAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/vhat/Read/ReadVariableOpReadVariableOp9Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/vhat*
_output_shapes	
:Ш*
dtype0
е
<Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*M
shared_name><Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/vhat
Ю
PAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/vhat/Read/ReadVariableOpReadVariableOp<Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/vhat*
_output_shapes
:	Ш*
dtype0
щ
FAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*W
shared_nameHFAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/vhat
т
ZAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/vhat*
_output_shapes
:	2Ш*
dtype0
Э
:Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*K
shared_name<:Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/vhat
Ц
NAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/vhat/Read/ReadVariableOpReadVariableOp:Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/vhat*
_output_shapes	
:Ш*
dtype0

NoOpNoOp
љ?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Д?
valueЊ?BЇ? B ?
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
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
mk
VARIABLE_VALUE1bidirectional_2/forward_lstm_2/lstm_cell_7/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE;bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE/bidirectional_2/forward_lstm_2/lstm_cell_7/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2bidirectional_2/backward_lstm_2/lstm_cell_8/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE<bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE0bidirectional_2/backward_lstm_2/lstm_cell_8/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
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
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEBAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE6Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE9Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUECAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE7Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEBAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE6Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE9Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUECAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE7Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense_2/kernel/vhatUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_2/bias/vhatSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE;Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/vhatEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
Ё
VARIABLE_VALUEEAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/vhatEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE9Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/vhatEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/vhatEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
Ђ
VARIABLE_VALUEFAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/vhatEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE:Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/vhatEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
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
П
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_0_11bidirectional_2/forward_lstm_2/lstm_cell_7/kernel;bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/bidirectional_2/forward_lstm_2/lstm_cell_7/bias2bidirectional_2/backward_lstm_2/lstm_cell_8/kernel<bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel0bidirectional_2/backward_lstm_2/lstm_cell_8/biasdense_2/kerneldense_2/bias*
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
GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_364032
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
э
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpEbidirectional_2/forward_lstm_2/lstm_cell_7/kernel/Read/ReadVariableOpObidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/Read/ReadVariableOpCbidirectional_2/forward_lstm_2/lstm_cell_7/bias/Read/ReadVariableOpFbidirectional_2/backward_lstm_2/lstm_cell_8/kernel/Read/ReadVariableOpPbidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/Read/ReadVariableOpDbidirectional_2/backward_lstm_2/lstm_cell_8/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOpLAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/m/Read/ReadVariableOpVAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/m/Read/ReadVariableOpJAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/m/Read/ReadVariableOpMAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/m/Read/ReadVariableOpWAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/m/Read/ReadVariableOpKAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOpLAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/v/Read/ReadVariableOpVAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/v/Read/ReadVariableOpJAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/v/Read/ReadVariableOpMAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/v/Read/ReadVariableOpWAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/v/Read/ReadVariableOpKAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/v/Read/ReadVariableOp,Adam/dense_2/kernel/vhat/Read/ReadVariableOp*Adam/dense_2/bias/vhat/Read/ReadVariableOpOAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/vhat/Read/ReadVariableOpYAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/vhat/Read/ReadVariableOpMAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/vhat/Read/ReadVariableOpPAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/vhat/Read/ReadVariableOpZAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/vhat/Read/ReadVariableOpNAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/vhat/Read/ReadVariableOpConst*4
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
GPU 2J 8 *(
f#R!
__inference__traced_save_367083
м
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate1bidirectional_2/forward_lstm_2/lstm_cell_7/kernel;bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/bidirectional_2/forward_lstm_2/lstm_cell_7/bias2bidirectional_2/backward_lstm_2/lstm_cell_8/kernel<bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel0bidirectional_2/backward_lstm_2/lstm_cell_8/biastotalcountAdam/dense_2/kernel/mAdam/dense_2/bias/m8Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/mBAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/m6Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/m9Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/mCAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/m7Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/mAdam/dense_2/kernel/vAdam/dense_2/bias/v8Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/vBAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/v6Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/v9Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/vCAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/v7Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/vAdam/dense_2/kernel/vhatAdam/dense_2/bias/vhat;Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/vhatEAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/vhat9Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/vhat<Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/vhatFAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/vhat:Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/vhat*3
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_3672107
э`
Ф
__inference__traced_save_367083
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopP
Lsavev2_bidirectional_2_forward_lstm_2_lstm_cell_7_kernel_read_readvariableopZ
Vsavev2_bidirectional_2_forward_lstm_2_lstm_cell_7_recurrent_kernel_read_readvariableopN
Jsavev2_bidirectional_2_forward_lstm_2_lstm_cell_7_bias_read_readvariableopQ
Msavev2_bidirectional_2_backward_lstm_2_lstm_cell_8_kernel_read_readvariableop[
Wsavev2_bidirectional_2_backward_lstm_2_lstm_cell_8_recurrent_kernel_read_readvariableopO
Ksavev2_bidirectional_2_backward_lstm_2_lstm_cell_8_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableopW
Ssavev2_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_kernel_m_read_readvariableopa
]savev2_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_recurrent_kernel_m_read_readvariableopU
Qsavev2_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_bias_m_read_readvariableopX
Tsavev2_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_kernel_m_read_readvariableopb
^savev2_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_recurrent_kernel_m_read_readvariableopV
Rsavev2_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableopW
Ssavev2_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_kernel_v_read_readvariableopa
]savev2_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_recurrent_kernel_v_read_readvariableopU
Qsavev2_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_bias_v_read_readvariableopX
Tsavev2_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_kernel_v_read_readvariableopb
^savev2_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_recurrent_kernel_v_read_readvariableopV
Rsavev2_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_bias_v_read_readvariableop7
3savev2_adam_dense_2_kernel_vhat_read_readvariableop5
1savev2_adam_dense_2_bias_vhat_read_readvariableopZ
Vsavev2_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_kernel_vhat_read_readvariableopd
`savev2_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_recurrent_kernel_vhat_read_readvariableopX
Tsavev2_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_bias_vhat_read_readvariableop[
Wsavev2_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_kernel_vhat_read_readvariableope
asavev2_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_recurrent_kernel_vhat_read_readvariableopY
Usavev2_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_bias_vhat_read_readvariableop
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
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopLsavev2_bidirectional_2_forward_lstm_2_lstm_cell_7_kernel_read_readvariableopVsavev2_bidirectional_2_forward_lstm_2_lstm_cell_7_recurrent_kernel_read_readvariableopJsavev2_bidirectional_2_forward_lstm_2_lstm_cell_7_bias_read_readvariableopMsavev2_bidirectional_2_backward_lstm_2_lstm_cell_8_kernel_read_readvariableopWsavev2_bidirectional_2_backward_lstm_2_lstm_cell_8_recurrent_kernel_read_readvariableopKsavev2_bidirectional_2_backward_lstm_2_lstm_cell_8_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableopSsavev2_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_kernel_m_read_readvariableop]savev2_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_recurrent_kernel_m_read_readvariableopQsavev2_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_bias_m_read_readvariableopTsavev2_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_kernel_m_read_readvariableop^savev2_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_recurrent_kernel_m_read_readvariableopRsavev2_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopSsavev2_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_kernel_v_read_readvariableop]savev2_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_recurrent_kernel_v_read_readvariableopQsavev2_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_bias_v_read_readvariableopTsavev2_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_kernel_v_read_readvariableop^savev2_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_recurrent_kernel_v_read_readvariableopRsavev2_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_bias_v_read_readvariableop3savev2_adam_dense_2_kernel_vhat_read_readvariableop1savev2_adam_dense_2_bias_vhat_read_readvariableopVsavev2_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_kernel_vhat_read_readvariableop`savev2_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_recurrent_kernel_vhat_read_readvariableopTsavev2_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_bias_vhat_read_readvariableopWsavev2_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_kernel_vhat_read_readvariableopasavev2_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_recurrent_kernel_vhat_read_readvariableopUsavev2_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_bias_vhat_read_readvariableopsavev2_const"/device:CPU:0*
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
е
У
while_cond_361274
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_361274___redundant_placeholder04
0while_while_cond_361274___redundant_placeholder14
0while_while_cond_361274___redundant_placeholder24
0while_while_cond_361274___redundant_placeholder3
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
Ы>
Ч
while_body_365855
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_7_matmul_readvariableop_resource_0:	ШG
4while_lstm_cell_7_matmul_1_readvariableop_resource_0:	2ШB
3while_lstm_cell_7_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_7_matmul_readvariableop_resource:	ШE
2while_lstm_cell_7_matmul_1_readvariableop_resource:	2Ш@
1while_lstm_cell_7_biasadd_readvariableop_resource:	ШЂ(while/lstm_cell_7/BiasAdd/ReadVariableOpЂ'while/lstm_cell_7/MatMul/ReadVariableOpЂ)while/lstm_cell_7/MatMul_1/ReadVariableOpУ
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
)while/TensorArrayV2Read/TensorListGetItemЦ
'while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02)
'while/lstm_cell_7/MatMul/ReadVariableOpд
while/lstm_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/MatMulЬ
)while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02+
)while/lstm_cell_7/MatMul_1/ReadVariableOpН
while/lstm_cell_7/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/MatMul_1Д
while/lstm_cell_7/addAddV2"while/lstm_cell_7/MatMul:product:0$while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/addХ
(while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02*
(while/lstm_cell_7/BiasAdd/ReadVariableOpС
while/lstm_cell_7/BiasAddBiasAddwhile/lstm_cell_7/add:z:00while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/BiasAdd
!while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_7/split/split_dim
while/lstm_cell_7/splitSplit*while/lstm_cell_7/split/split_dim:output:0"while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_7/split
while/lstm_cell_7/SigmoidSigmoid while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Sigmoid
while/lstm_cell_7/Sigmoid_1Sigmoid while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Sigmoid_1
while/lstm_cell_7/mulMulwhile/lstm_cell_7/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/mul
while/lstm_cell_7/ReluRelu while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/ReluА
while/lstm_cell_7/mul_1Mulwhile/lstm_cell_7/Sigmoid:y:0$while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/mul_1Ѕ
while/lstm_cell_7/add_1AddV2while/lstm_cell_7/mul:z:0while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/add_1
while/lstm_cell_7/Sigmoid_2Sigmoid while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Sigmoid_2
while/lstm_cell_7/Relu_1Reluwhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Relu_1Д
while/lstm_cell_7/mul_2Mulwhile/lstm_cell_7/Sigmoid_2:y:0&while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/mul_2п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_7/mul_2:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_7/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_7/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5л

while/NoOpNoOp)^while/lstm_cell_7/BiasAdd/ReadVariableOp(^while/lstm_cell_7/MatMul/ReadVariableOp*^while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_7_biasadd_readvariableop_resource3while_lstm_cell_7_biasadd_readvariableop_resource_0"j
2while_lstm_cell_7_matmul_1_readvariableop_resource4while_lstm_cell_7_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_7_matmul_readvariableop_resource2while_lstm_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2T
(while/lstm_cell_7/BiasAdd/ReadVariableOp(while/lstm_cell_7/BiasAdd/ReadVariableOp2R
'while/lstm_cell_7/MatMul/ReadVariableOp'while/lstm_cell_7/MatMul/ReadVariableOp2V
)while/lstm_cell_7/MatMul_1/ReadVariableOp)while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Н
0__inference_backward_lstm_2_layer_call_fn_366134

inputs
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
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
GPU 2J 8 *T
fORM
K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_3627532
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
З

!backward_lstm_2_while_cond_364619<
8backward_lstm_2_while_backward_lstm_2_while_loop_counterB
>backward_lstm_2_while_backward_lstm_2_while_maximum_iterations%
!backward_lstm_2_while_placeholder'
#backward_lstm_2_while_placeholder_1'
#backward_lstm_2_while_placeholder_2'
#backward_lstm_2_while_placeholder_3>
:backward_lstm_2_while_less_backward_lstm_2_strided_slice_1T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_364619___redundant_placeholder0T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_364619___redundant_placeholder1T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_364619___redundant_placeholder2T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_364619___redundant_placeholder3"
backward_lstm_2_while_identity
Р
backward_lstm_2/while/LessLess!backward_lstm_2_while_placeholder:backward_lstm_2_while_less_backward_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_2/while/Less
backward_lstm_2/while/IdentityIdentitybackward_lstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2 
backward_lstm_2/while/Identity"I
backward_lstm_2_while_identity'backward_lstm_2/while/Identity:output:0*(
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


!backward_lstm_2_while_cond_363314<
8backward_lstm_2_while_backward_lstm_2_while_loop_counterB
>backward_lstm_2_while_backward_lstm_2_while_maximum_iterations%
!backward_lstm_2_while_placeholder'
#backward_lstm_2_while_placeholder_1'
#backward_lstm_2_while_placeholder_2'
#backward_lstm_2_while_placeholder_3'
#backward_lstm_2_while_placeholder_4>
:backward_lstm_2_while_less_backward_lstm_2_strided_slice_1T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_363314___redundant_placeholder0T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_363314___redundant_placeholder1T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_363314___redundant_placeholder2T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_363314___redundant_placeholder3T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_363314___redundant_placeholder4"
backward_lstm_2_while_identity
Р
backward_lstm_2/while/LessLess!backward_lstm_2_while_placeholder:backward_lstm_2_while_less_backward_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_2/while/Less
backward_lstm_2/while/IdentityIdentitybackward_lstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2 
backward_lstm_2/while/Identity"I
backward_lstm_2_while_identity'backward_lstm_2/while/Identity:output:0*(
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
е
У
while_cond_365552
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_365552___redundant_placeholder04
0while_while_cond_365552___redundant_placeholder14
0while_while_cond_365552___redundant_placeholder24
0while_while_cond_365552___redundant_placeholder3
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
ђ

G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_361051

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
^

K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_362753

inputs=
*lstm_cell_8_matmul_readvariableop_resource:	Ш?
,lstm_cell_8_matmul_1_readvariableop_resource:	2Ш:
+lstm_cell_8_biasadd_readvariableop_resource:	Ш
identityЂ"lstm_cell_8/BiasAdd/ReadVariableOpЂ!lstm_cell_8/MatMul/ReadVariableOpЂ#lstm_cell_8/MatMul_1/ReadVariableOpЂwhileD
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
strided_slice_2В
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOpЊ
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/MatMulИ
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOpІ
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/MatMul_1
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/addБ
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOpЉ
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/BiasAdd|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimя
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_8/split
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Sigmoid
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Sigmoid_1
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Relu
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/mul_1
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/add_1
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Relu_1
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
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
bodyR
while_body_362669*
condR
while_cond_362668*K
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

IdentityХ
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
H

K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_361978

inputs%
lstm_cell_8_361896:	Ш%
lstm_cell_8_361898:	2Ш!
lstm_cell_8_361900:	Ш
identityЂ#lstm_cell_8/StatefulPartitionedCallЂwhileD
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
strided_slice_2
#lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_8_361896lstm_cell_8_361898lstm_cell_8_361900*
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
GPU 2J 8 *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_3618292%
#lstm_cell_8/StatefulPartitionedCall
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
while/loop_counterН
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_8_361896lstm_cell_8_361898lstm_cell_8_361900*
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
bodyR
while_body_361909*
condR
while_cond_361908*K
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

Identity|
NoOpNoOp$^lstm_cell_8/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2J
#lstm_cell_8/StatefulPartitionedCall#lstm_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
є

д
-__inference_sequential_2_layer_call_fn_363463

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
identityЂStatefulPartitionedCallб
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
GPU 2J 8 *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_3634442
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
Ю
П
0__inference_backward_lstm_2_layer_call_fn_366112
inputs_0
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
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
GPU 2J 8 *T
fORM
K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_3619782
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
І

Ё
0__inference_bidirectional_2_layer_call_fn_364102

inputs
inputs_1	
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
	unknown_2:	Ш
	unknown_3:	2Ш
	unknown_4:	Ш
identityЂStatefulPartitionedCallК
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
GPU 2J 8 *T
fORM
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_3638522
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

є
C__inference_dense_2_layer_call_and_return_conditional_losses_365442

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
е
У
while_cond_366202
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_366202___redundant_placeholder04
0while_while_cond_366202___redundant_placeholder14
0while_while_cond_366202___redundant_placeholder24
0while_while_cond_366202___redundant_placeholder3
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
Ы>
Ч
while_body_366006
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_7_matmul_readvariableop_resource_0:	ШG
4while_lstm_cell_7_matmul_1_readvariableop_resource_0:	2ШB
3while_lstm_cell_7_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_7_matmul_readvariableop_resource:	ШE
2while_lstm_cell_7_matmul_1_readvariableop_resource:	2Ш@
1while_lstm_cell_7_biasadd_readvariableop_resource:	ШЂ(while/lstm_cell_7/BiasAdd/ReadVariableOpЂ'while/lstm_cell_7/MatMul/ReadVariableOpЂ)while/lstm_cell_7/MatMul_1/ReadVariableOpУ
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
)while/TensorArrayV2Read/TensorListGetItemЦ
'while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02)
'while/lstm_cell_7/MatMul/ReadVariableOpд
while/lstm_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/MatMulЬ
)while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02+
)while/lstm_cell_7/MatMul_1/ReadVariableOpН
while/lstm_cell_7/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/MatMul_1Д
while/lstm_cell_7/addAddV2"while/lstm_cell_7/MatMul:product:0$while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/addХ
(while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02*
(while/lstm_cell_7/BiasAdd/ReadVariableOpС
while/lstm_cell_7/BiasAddBiasAddwhile/lstm_cell_7/add:z:00while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/BiasAdd
!while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_7/split/split_dim
while/lstm_cell_7/splitSplit*while/lstm_cell_7/split/split_dim:output:0"while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_7/split
while/lstm_cell_7/SigmoidSigmoid while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Sigmoid
while/lstm_cell_7/Sigmoid_1Sigmoid while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Sigmoid_1
while/lstm_cell_7/mulMulwhile/lstm_cell_7/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/mul
while/lstm_cell_7/ReluRelu while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/ReluА
while/lstm_cell_7/mul_1Mulwhile/lstm_cell_7/Sigmoid:y:0$while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/mul_1Ѕ
while/lstm_cell_7/add_1AddV2while/lstm_cell_7/mul:z:0while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/add_1
while/lstm_cell_7/Sigmoid_2Sigmoid while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Sigmoid_2
while/lstm_cell_7/Relu_1Reluwhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Relu_1Д
while/lstm_cell_7/mul_2Mulwhile/lstm_cell_7/Sigmoid_2:y:0&while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/mul_2п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_7/mul_2:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_7/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_7/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5л

while/NoOpNoOp)^while/lstm_cell_7/BiasAdd/ReadVariableOp(^while/lstm_cell_7/MatMul/ReadVariableOp*^while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_7_biasadd_readvariableop_resource3while_lstm_cell_7_biasadd_readvariableop_resource_0"j
2while_lstm_cell_7_matmul_1_readvariableop_resource4while_lstm_cell_7_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_7_matmul_readvariableop_resource2while_lstm_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2T
(while/lstm_cell_7/BiasAdd/ReadVariableOp(while/lstm_cell_7/BiasAdd/ReadVariableOp2R
'while/lstm_cell_7/MatMul/ReadVariableOp'while/lstm_cell_7/MatMul/ReadVariableOp2V
)while/lstm_cell_7/MatMul_1/ReadVariableOp)while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
ђ

G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_361683

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
ђ

G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_361829

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
є

д
-__inference_sequential_2_layer_call_fn_363956

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
identityЂStatefulPartitionedCallб
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
GPU 2J 8 *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_3639152
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
Т>
Ч
while_body_366203
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	ШG
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:	2ШB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	ШE
2while_lstm_cell_8_matmul_1_readvariableop_resource:	2Ш@
1while_lstm_cell_8_biasadd_readvariableop_resource:	ШЂ(while/lstm_cell_8/BiasAdd/ReadVariableOpЂ'while/lstm_cell_8/MatMul/ReadVariableOpЂ)while/lstm_cell_8/MatMul_1/ReadVariableOpУ
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
)while/TensorArrayV2Read/TensorListGetItemЦ
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOpд
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/MatMulЬ
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOpН
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/MatMul_1Д
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/addХ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOpС
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/BiasAdd
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_8/split
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Sigmoid
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Sigmoid_1
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/mul
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/ReluА
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/mul_1Ѕ
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/add_1
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Sigmoid_2
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Relu_1Д
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/mul_2п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5л

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
е
У
while_cond_361908
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_361908___redundant_placeholder04
0while_while_cond_361908___redundant_placeholder14
0while_while_cond_361908___redundant_placeholder24
0while_while_cond_361908___redundant_placeholder3
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
е
У
while_cond_365854
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_365854___redundant_placeholder04
0while_while_cond_365854___redundant_placeholder14
0while_while_cond_365854___redundant_placeholder24
0while_while_cond_365854___redundant_placeholder3
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
ѕ
Д
=sequential_2_bidirectional_2_forward_lstm_2_while_body_360693t
psequential_2_bidirectional_2_forward_lstm_2_while_sequential_2_bidirectional_2_forward_lstm_2_while_loop_counterz
vsequential_2_bidirectional_2_forward_lstm_2_while_sequential_2_bidirectional_2_forward_lstm_2_while_maximum_iterationsA
=sequential_2_bidirectional_2_forward_lstm_2_while_placeholderC
?sequential_2_bidirectional_2_forward_lstm_2_while_placeholder_1C
?sequential_2_bidirectional_2_forward_lstm_2_while_placeholder_2C
?sequential_2_bidirectional_2_forward_lstm_2_while_placeholder_3C
?sequential_2_bidirectional_2_forward_lstm_2_while_placeholder_4s
osequential_2_bidirectional_2_forward_lstm_2_while_sequential_2_bidirectional_2_forward_lstm_2_strided_slice_1_0А
Ћsequential_2_bidirectional_2_forward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_2_bidirectional_2_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0p
lsequential_2_bidirectional_2_forward_lstm_2_while_greater_sequential_2_bidirectional_2_forward_lstm_2_cast_0q
^sequential_2_bidirectional_2_forward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0:	Шs
`sequential_2_bidirectional_2_forward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0:	2Шn
_sequential_2_bidirectional_2_forward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0:	Ш>
:sequential_2_bidirectional_2_forward_lstm_2_while_identity@
<sequential_2_bidirectional_2_forward_lstm_2_while_identity_1@
<sequential_2_bidirectional_2_forward_lstm_2_while_identity_2@
<sequential_2_bidirectional_2_forward_lstm_2_while_identity_3@
<sequential_2_bidirectional_2_forward_lstm_2_while_identity_4@
<sequential_2_bidirectional_2_forward_lstm_2_while_identity_5@
<sequential_2_bidirectional_2_forward_lstm_2_while_identity_6q
msequential_2_bidirectional_2_forward_lstm_2_while_sequential_2_bidirectional_2_forward_lstm_2_strided_slice_1Ў
Љsequential_2_bidirectional_2_forward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_2_bidirectional_2_forward_lstm_2_tensorarrayunstack_tensorlistfromtensorn
jsequential_2_bidirectional_2_forward_lstm_2_while_greater_sequential_2_bidirectional_2_forward_lstm_2_casto
\sequential_2_bidirectional_2_forward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource:	Шq
^sequential_2_bidirectional_2_forward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource:	2Шl
]sequential_2_bidirectional_2_forward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource:	ШЂTsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOpЂSsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOpЂUsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp
csequential_2/bidirectional_2/forward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2e
csequential_2/bidirectional_2/forward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeм
Usequential_2/bidirectional_2/forward_lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЋsequential_2_bidirectional_2_forward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_2_bidirectional_2_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0=sequential_2_bidirectional_2_forward_lstm_2_while_placeholderlsequential_2/bidirectional_2/forward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02W
Usequential_2/bidirectional_2/forward_lstm_2/while/TensorArrayV2Read/TensorListGetItemм
9sequential_2/bidirectional_2/forward_lstm_2/while/GreaterGreaterlsequential_2_bidirectional_2_forward_lstm_2_while_greater_sequential_2_bidirectional_2_forward_lstm_2_cast_0=sequential_2_bidirectional_2_forward_lstm_2_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2;
9sequential_2/bidirectional_2/forward_lstm_2/while/GreaterЪ
Ssequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp^sequential_2_bidirectional_2_forward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02U
Ssequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp
Dsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMulMatMul\sequential_2/bidirectional_2/forward_lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0[sequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2F
Dsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMulа
Usequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp`sequential_2_bidirectional_2_forward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02W
Usequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpэ
Fsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMul_1MatMul?sequential_2_bidirectional_2_forward_lstm_2_while_placeholder_3]sequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2H
Fsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMul_1ф
Asequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/addAddV2Nsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMul:product:0Psequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2C
Asequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/addЩ
Tsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp_sequential_2_bidirectional_2_forward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02V
Tsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOpё
Esequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/BiasAddBiasAddEsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/add:z:0\sequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2G
Esequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/BiasAddр
Msequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2O
Msequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/split/split_dimЗ
Csequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/splitSplitVsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/split/split_dim:output:0Nsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2E
Csequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/split
Esequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/SigmoidSigmoidLsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22G
Esequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/Sigmoid
Gsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/Sigmoid_1SigmoidLsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22I
Gsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/Sigmoid_1Э
Asequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/mulMulKsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/Sigmoid_1:y:0?sequential_2_bidirectional_2_forward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22C
Asequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/mul
Bsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/ReluReluLsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22D
Bsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/Reluр
Csequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/mul_1MulIsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/Sigmoid:y:0Psequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/mul_1е
Csequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/add_1AddV2Esequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/mul:z:0Gsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/add_1
Gsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/Sigmoid_2SigmoidLsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22I
Gsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/Sigmoid_2
Dsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/Relu_1ReluGsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22F
Dsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/Relu_1ф
Csequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/mul_2MulKsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/Sigmoid_2:y:0Rsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/mul_2љ
8sequential_2/bidirectional_2/forward_lstm_2/while/SelectSelect=sequential_2/bidirectional_2/forward_lstm_2/while/Greater:z:0Gsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/mul_2:z:0?sequential_2_bidirectional_2_forward_lstm_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22:
8sequential_2/bidirectional_2/forward_lstm_2/while/Select§
:sequential_2/bidirectional_2/forward_lstm_2/while/Select_1Select=sequential_2/bidirectional_2/forward_lstm_2/while/Greater:z:0Gsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/mul_2:z:0?sequential_2_bidirectional_2_forward_lstm_2_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22<
:sequential_2/bidirectional_2/forward_lstm_2/while/Select_1§
:sequential_2/bidirectional_2/forward_lstm_2/while/Select_2Select=sequential_2/bidirectional_2/forward_lstm_2/while/Greater:z:0Gsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/add_1:z:0?sequential_2_bidirectional_2_forward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22<
:sequential_2/bidirectional_2/forward_lstm_2/while/Select_2Е
Vsequential_2/bidirectional_2/forward_lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem?sequential_2_bidirectional_2_forward_lstm_2_while_placeholder_1=sequential_2_bidirectional_2_forward_lstm_2_while_placeholderAsequential_2/bidirectional_2/forward_lstm_2/while/Select:output:0*
_output_shapes
: *
element_dtype02X
Vsequential_2/bidirectional_2/forward_lstm_2/while/TensorArrayV2Write/TensorListSetItemД
7sequential_2/bidirectional_2/forward_lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_2/bidirectional_2/forward_lstm_2/while/add/y
5sequential_2/bidirectional_2/forward_lstm_2/while/addAddV2=sequential_2_bidirectional_2_forward_lstm_2_while_placeholder@sequential_2/bidirectional_2/forward_lstm_2/while/add/y:output:0*
T0*
_output_shapes
: 27
5sequential_2/bidirectional_2/forward_lstm_2/while/addИ
9sequential_2/bidirectional_2/forward_lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2;
9sequential_2/bidirectional_2/forward_lstm_2/while/add_1/yв
7sequential_2/bidirectional_2/forward_lstm_2/while/add_1AddV2psequential_2_bidirectional_2_forward_lstm_2_while_sequential_2_bidirectional_2_forward_lstm_2_while_loop_counterBsequential_2/bidirectional_2/forward_lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 29
7sequential_2/bidirectional_2/forward_lstm_2/while/add_1
:sequential_2/bidirectional_2/forward_lstm_2/while/IdentityIdentity;sequential_2/bidirectional_2/forward_lstm_2/while/add_1:z:07^sequential_2/bidirectional_2/forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2<
:sequential_2/bidirectional_2/forward_lstm_2/while/Identityк
<sequential_2/bidirectional_2/forward_lstm_2/while/Identity_1Identityvsequential_2_bidirectional_2_forward_lstm_2_while_sequential_2_bidirectional_2_forward_lstm_2_while_maximum_iterations7^sequential_2/bidirectional_2/forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2>
<sequential_2/bidirectional_2/forward_lstm_2/while/Identity_1
<sequential_2/bidirectional_2/forward_lstm_2/while/Identity_2Identity9sequential_2/bidirectional_2/forward_lstm_2/while/add:z:07^sequential_2/bidirectional_2/forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2>
<sequential_2/bidirectional_2/forward_lstm_2/while/Identity_2Ъ
<sequential_2/bidirectional_2/forward_lstm_2/while/Identity_3Identityfsequential_2/bidirectional_2/forward_lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:07^sequential_2/bidirectional_2/forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2>
<sequential_2/bidirectional_2/forward_lstm_2/while/Identity_3Ж
<sequential_2/bidirectional_2/forward_lstm_2/while/Identity_4IdentityAsequential_2/bidirectional_2/forward_lstm_2/while/Select:output:07^sequential_2/bidirectional_2/forward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22>
<sequential_2/bidirectional_2/forward_lstm_2/while/Identity_4И
<sequential_2/bidirectional_2/forward_lstm_2/while/Identity_5IdentityCsequential_2/bidirectional_2/forward_lstm_2/while/Select_1:output:07^sequential_2/bidirectional_2/forward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22>
<sequential_2/bidirectional_2/forward_lstm_2/while/Identity_5И
<sequential_2/bidirectional_2/forward_lstm_2/while/Identity_6IdentityCsequential_2/bidirectional_2/forward_lstm_2/while/Select_2:output:07^sequential_2/bidirectional_2/forward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22>
<sequential_2/bidirectional_2/forward_lstm_2/while/Identity_6З
6sequential_2/bidirectional_2/forward_lstm_2/while/NoOpNoOpU^sequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOpT^sequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOpV^sequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 28
6sequential_2/bidirectional_2/forward_lstm_2/while/NoOp"к
jsequential_2_bidirectional_2_forward_lstm_2_while_greater_sequential_2_bidirectional_2_forward_lstm_2_castlsequential_2_bidirectional_2_forward_lstm_2_while_greater_sequential_2_bidirectional_2_forward_lstm_2_cast_0"
:sequential_2_bidirectional_2_forward_lstm_2_while_identityCsequential_2/bidirectional_2/forward_lstm_2/while/Identity:output:0"
<sequential_2_bidirectional_2_forward_lstm_2_while_identity_1Esequential_2/bidirectional_2/forward_lstm_2/while/Identity_1:output:0"
<sequential_2_bidirectional_2_forward_lstm_2_while_identity_2Esequential_2/bidirectional_2/forward_lstm_2/while/Identity_2:output:0"
<sequential_2_bidirectional_2_forward_lstm_2_while_identity_3Esequential_2/bidirectional_2/forward_lstm_2/while/Identity_3:output:0"
<sequential_2_bidirectional_2_forward_lstm_2_while_identity_4Esequential_2/bidirectional_2/forward_lstm_2/while/Identity_4:output:0"
<sequential_2_bidirectional_2_forward_lstm_2_while_identity_5Esequential_2/bidirectional_2/forward_lstm_2/while/Identity_5:output:0"
<sequential_2_bidirectional_2_forward_lstm_2_while_identity_6Esequential_2/bidirectional_2/forward_lstm_2/while/Identity_6:output:0"Р
]sequential_2_bidirectional_2_forward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_sequential_2_bidirectional_2_forward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0"Т
^sequential_2_bidirectional_2_forward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource`sequential_2_bidirectional_2_forward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0"О
\sequential_2_bidirectional_2_forward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource^sequential_2_bidirectional_2_forward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0"р
msequential_2_bidirectional_2_forward_lstm_2_while_sequential_2_bidirectional_2_forward_lstm_2_strided_slice_1osequential_2_bidirectional_2_forward_lstm_2_while_sequential_2_bidirectional_2_forward_lstm_2_strided_slice_1_0"к
Љsequential_2_bidirectional_2_forward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_2_bidirectional_2_forward_lstm_2_tensorarrayunstack_tensorlistfromtensorЋsequential_2_bidirectional_2_forward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_2_bidirectional_2_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2Ќ
Tsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOpTsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp2Њ
Ssequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOpSsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp2Ў
Usequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpUsequential_2/bidirectional_2/forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
њ[

J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_366090

inputs=
*lstm_cell_7_matmul_readvariableop_resource:	Ш?
,lstm_cell_7_matmul_1_readvariableop_resource:	2Ш:
+lstm_cell_7_biasadd_readvariableop_resource:	Ш
identityЂ"lstm_cell_7/BiasAdd/ReadVariableOpЂ!lstm_cell_7/MatMul/ReadVariableOpЂ#lstm_cell_7/MatMul_1/ReadVariableOpЂwhileD
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
strided_slice_2В
!lstm_cell_7/MatMul/ReadVariableOpReadVariableOp*lstm_cell_7_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02#
!lstm_cell_7/MatMul/ReadVariableOpЊ
lstm_cell_7/MatMulMatMulstrided_slice_2:output:0)lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/MatMulИ
#lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02%
#lstm_cell_7/MatMul_1/ReadVariableOpІ
lstm_cell_7/MatMul_1MatMulzeros:output:0+lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/MatMul_1
lstm_cell_7/addAddV2lstm_cell_7/MatMul:product:0lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/addБ
"lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02$
"lstm_cell_7/BiasAdd/ReadVariableOpЉ
lstm_cell_7/BiasAddBiasAddlstm_cell_7/add:z:0*lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/BiasAdd|
lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_7/split/split_dimя
lstm_cell_7/splitSplit$lstm_cell_7/split/split_dim:output:0lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_7/split
lstm_cell_7/SigmoidSigmoidlstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Sigmoid
lstm_cell_7/Sigmoid_1Sigmoidlstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Sigmoid_1
lstm_cell_7/mulMullstm_cell_7/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/mulz
lstm_cell_7/ReluRelulstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Relu
lstm_cell_7/mul_1Mullstm_cell_7/Sigmoid:y:0lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/mul_1
lstm_cell_7/add_1AddV2lstm_cell_7/mul:z:0lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/add_1
lstm_cell_7/Sigmoid_2Sigmoidlstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Sigmoid_2y
lstm_cell_7/Relu_1Relulstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Relu_1
lstm_cell_7/mul_2Mullstm_cell_7/Sigmoid_2:y:0 lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_7_matmul_readvariableop_resource,lstm_cell_7_matmul_1_readvariableop_resource+lstm_cell_7_biasadd_readvariableop_resource*
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
bodyR
while_body_366006*
condR
while_cond_366005*K
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

IdentityХ
NoOpNoOp#^lstm_cell_7/BiasAdd/ReadVariableOp"^lstm_cell_7/MatMul/ReadVariableOp$^lstm_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2H
"lstm_cell_7/BiasAdd/ReadVariableOp"lstm_cell_7/BiasAdd/ReadVariableOp2F
!lstm_cell_7/MatMul/ReadVariableOp!lstm_cell_7/MatMul/ReadVariableOp2J
#lstm_cell_7/MatMul_1/ReadVariableOp#lstm_cell_7/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њ[

J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_365939

inputs=
*lstm_cell_7_matmul_readvariableop_resource:	Ш?
,lstm_cell_7_matmul_1_readvariableop_resource:	2Ш:
+lstm_cell_7_biasadd_readvariableop_resource:	Ш
identityЂ"lstm_cell_7/BiasAdd/ReadVariableOpЂ!lstm_cell_7/MatMul/ReadVariableOpЂ#lstm_cell_7/MatMul_1/ReadVariableOpЂwhileD
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
strided_slice_2В
!lstm_cell_7/MatMul/ReadVariableOpReadVariableOp*lstm_cell_7_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02#
!lstm_cell_7/MatMul/ReadVariableOpЊ
lstm_cell_7/MatMulMatMulstrided_slice_2:output:0)lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/MatMulИ
#lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02%
#lstm_cell_7/MatMul_1/ReadVariableOpІ
lstm_cell_7/MatMul_1MatMulzeros:output:0+lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/MatMul_1
lstm_cell_7/addAddV2lstm_cell_7/MatMul:product:0lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/addБ
"lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02$
"lstm_cell_7/BiasAdd/ReadVariableOpЉ
lstm_cell_7/BiasAddBiasAddlstm_cell_7/add:z:0*lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/BiasAdd|
lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_7/split/split_dimя
lstm_cell_7/splitSplit$lstm_cell_7/split/split_dim:output:0lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_7/split
lstm_cell_7/SigmoidSigmoidlstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Sigmoid
lstm_cell_7/Sigmoid_1Sigmoidlstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Sigmoid_1
lstm_cell_7/mulMullstm_cell_7/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/mulz
lstm_cell_7/ReluRelulstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Relu
lstm_cell_7/mul_1Mullstm_cell_7/Sigmoid:y:0lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/mul_1
lstm_cell_7/add_1AddV2lstm_cell_7/mul:z:0lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/add_1
lstm_cell_7/Sigmoid_2Sigmoidlstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Sigmoid_2y
lstm_cell_7/Relu_1Relulstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Relu_1
lstm_cell_7/mul_2Mullstm_cell_7/Sigmoid_2:y:0 lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_7_matmul_readvariableop_resource,lstm_cell_7_matmul_1_readvariableop_resource+lstm_cell_7_biasadd_readvariableop_resource*
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
bodyR
while_body_365855*
condR
while_cond_365854*K
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

IdentityХ
NoOpNoOp#^lstm_cell_7/BiasAdd/ReadVariableOp"^lstm_cell_7/MatMul/ReadVariableOp$^lstm_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2H
"lstm_cell_7/BiasAdd/ReadVariableOp"lstm_cell_7/BiasAdd/ReadVariableOp2F
!lstm_cell_7/MatMul/ReadVariableOp!lstm_cell_7/MatMul/ReadVariableOp2J
#lstm_cell_7/MatMul_1/ReadVariableOp#lstm_cell_7/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ў`
т
 forward_lstm_2_while_body_363136:
6forward_lstm_2_while_forward_lstm_2_while_loop_counter@
<forward_lstm_2_while_forward_lstm_2_while_maximum_iterations$
 forward_lstm_2_while_placeholder&
"forward_lstm_2_while_placeholder_1&
"forward_lstm_2_while_placeholder_2&
"forward_lstm_2_while_placeholder_3&
"forward_lstm_2_while_placeholder_49
5forward_lstm_2_while_forward_lstm_2_strided_slice_1_0u
qforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_06
2forward_lstm_2_while_greater_forward_lstm_2_cast_0T
Aforward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0:	ШV
Cforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0:	2ШQ
Bforward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0:	Ш!
forward_lstm_2_while_identity#
forward_lstm_2_while_identity_1#
forward_lstm_2_while_identity_2#
forward_lstm_2_while_identity_3#
forward_lstm_2_while_identity_4#
forward_lstm_2_while_identity_5#
forward_lstm_2_while_identity_67
3forward_lstm_2_while_forward_lstm_2_strided_slice_1s
oforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor4
0forward_lstm_2_while_greater_forward_lstm_2_castR
?forward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource:	ШT
Aforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource:	2ШO
@forward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource:	ШЂ7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOpЂ6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOpЂ8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpс
Fforward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fforward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape­
8forward_lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0 forward_lstm_2_while_placeholderOforward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02:
8forward_lstm_2/while/TensorArrayV2Read/TensorListGetItemЫ
forward_lstm_2/while/GreaterGreater2forward_lstm_2_while_greater_forward_lstm_2_cast_0 forward_lstm_2_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_2/while/Greaterѓ
6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOpAforward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype028
6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp
'forward_lstm_2/while/lstm_cell_7/MatMulMatMul?forward_lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0>forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_2/while/lstm_cell_7/MatMulљ
8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOpCforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02:
8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpљ
)forward_lstm_2/while/lstm_cell_7/MatMul_1MatMul"forward_lstm_2_while_placeholder_3@forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)forward_lstm_2/while/lstm_cell_7/MatMul_1№
$forward_lstm_2/while/lstm_cell_7/addAddV21forward_lstm_2/while/lstm_cell_7/MatMul:product:03forward_lstm_2/while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$forward_lstm_2/while/lstm_cell_7/addђ
7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOpBforward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype029
7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp§
(forward_lstm_2/while/lstm_cell_7/BiasAddBiasAdd(forward_lstm_2/while/lstm_cell_7/add:z:0?forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_2/while/lstm_cell_7/BiasAddІ
0forward_lstm_2/while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0forward_lstm_2/while/lstm_cell_7/split/split_dimУ
&forward_lstm_2/while/lstm_cell_7/splitSplit9forward_lstm_2/while/lstm_cell_7/split/split_dim:output:01forward_lstm_2/while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2(
&forward_lstm_2/while/lstm_cell_7/splitТ
(forward_lstm_2/while/lstm_cell_7/SigmoidSigmoid/forward_lstm_2/while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_2/while/lstm_cell_7/SigmoidЦ
*forward_lstm_2/while/lstm_cell_7/Sigmoid_1Sigmoid/forward_lstm_2/while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_2/while/lstm_cell_7/Sigmoid_1й
$forward_lstm_2/while/lstm_cell_7/mulMul.forward_lstm_2/while/lstm_cell_7/Sigmoid_1:y:0"forward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_2/while/lstm_cell_7/mulЙ
%forward_lstm_2/while/lstm_cell_7/ReluRelu/forward_lstm_2/while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_2/while/lstm_cell_7/Reluь
&forward_lstm_2/while/lstm_cell_7/mul_1Mul,forward_lstm_2/while/lstm_cell_7/Sigmoid:y:03forward_lstm_2/while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_2/while/lstm_cell_7/mul_1с
&forward_lstm_2/while/lstm_cell_7/add_1AddV2(forward_lstm_2/while/lstm_cell_7/mul:z:0*forward_lstm_2/while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_2/while/lstm_cell_7/add_1Ц
*forward_lstm_2/while/lstm_cell_7/Sigmoid_2Sigmoid/forward_lstm_2/while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_2/while/lstm_cell_7/Sigmoid_2И
'forward_lstm_2/while/lstm_cell_7/Relu_1Relu*forward_lstm_2/while/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_2/while/lstm_cell_7/Relu_1№
&forward_lstm_2/while/lstm_cell_7/mul_2Mul.forward_lstm_2/while/lstm_cell_7/Sigmoid_2:y:05forward_lstm_2/while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_2/while/lstm_cell_7/mul_2ш
forward_lstm_2/while/SelectSelect forward_lstm_2/while/Greater:z:0*forward_lstm_2/while/lstm_cell_7/mul_2:z:0"forward_lstm_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/while/Selectь
forward_lstm_2/while/Select_1Select forward_lstm_2/while/Greater:z:0*forward_lstm_2/while/lstm_cell_7/mul_2:z:0"forward_lstm_2_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/while/Select_1ь
forward_lstm_2/while/Select_2Select forward_lstm_2/while/Greater:z:0*forward_lstm_2/while/lstm_cell_7/add_1:z:0"forward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/while/Select_2Є
9forward_lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"forward_lstm_2_while_placeholder_1 forward_lstm_2_while_placeholder$forward_lstm_2/while/Select:output:0*
_output_shapes
: *
element_dtype02;
9forward_lstm_2/while/TensorArrayV2Write/TensorListSetItemz
forward_lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_2/while/add/yЅ
forward_lstm_2/while/addAddV2 forward_lstm_2_while_placeholder#forward_lstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/while/add~
forward_lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_2/while/add_1/yС
forward_lstm_2/while/add_1AddV26forward_lstm_2_while_forward_lstm_2_while_loop_counter%forward_lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/while/add_1Ї
forward_lstm_2/while/IdentityIdentityforward_lstm_2/while/add_1:z:0^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2
forward_lstm_2/while/IdentityЩ
forward_lstm_2/while/Identity_1Identity<forward_lstm_2_while_forward_lstm_2_while_maximum_iterations^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_2/while/Identity_1Љ
forward_lstm_2/while/Identity_2Identityforward_lstm_2/while/add:z:0^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_2/while/Identity_2ж
forward_lstm_2/while/Identity_3IdentityIforward_lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_2/while/Identity_3Т
forward_lstm_2/while/Identity_4Identity$forward_lstm_2/while/Select:output:0^forward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/while/Identity_4Ф
forward_lstm_2/while/Identity_5Identity&forward_lstm_2/while/Select_1:output:0^forward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/while/Identity_5Ф
forward_lstm_2/while/Identity_6Identity&forward_lstm_2/while/Select_2:output:0^forward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/while/Identity_6І
forward_lstm_2/while/NoOpNoOp8^forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp7^forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp9^forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_2/while/NoOp"l
3forward_lstm_2_while_forward_lstm_2_strided_slice_15forward_lstm_2_while_forward_lstm_2_strided_slice_1_0"f
0forward_lstm_2_while_greater_forward_lstm_2_cast2forward_lstm_2_while_greater_forward_lstm_2_cast_0"G
forward_lstm_2_while_identity&forward_lstm_2/while/Identity:output:0"K
forward_lstm_2_while_identity_1(forward_lstm_2/while/Identity_1:output:0"K
forward_lstm_2_while_identity_2(forward_lstm_2/while/Identity_2:output:0"K
forward_lstm_2_while_identity_3(forward_lstm_2/while/Identity_3:output:0"K
forward_lstm_2_while_identity_4(forward_lstm_2/while/Identity_4:output:0"K
forward_lstm_2_while_identity_5(forward_lstm_2/while/Identity_5:output:0"K
forward_lstm_2_while_identity_6(forward_lstm_2/while/Identity_6:output:0"
@forward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resourceBforward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0"
Aforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resourceCforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0"
?forward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resourceAforward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0"ф
oforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensorqforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2r
7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp2p
6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp2t
8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
е
У
while_cond_365703
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_365703___redundant_placeholder04
0while_while_cond_365703___redundant_placeholder14
0while_while_cond_365703___redundant_placeholder24
0while_while_cond_365703___redundant_placeholder3
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
е
У
while_cond_361064
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_361064___redundant_placeholder04
0while_while_cond_361064___redundant_placeholder14
0while_while_cond_361064___redundant_placeholder24
0while_while_cond_361064___redundant_placeholder3
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
и
М
/__inference_forward_lstm_2_layer_call_fn_365475

inputs
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
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
GPU 2J 8 *S
fNRL
J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_3624012
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
ћ
ы
 forward_lstm_2_while_cond_363575:
6forward_lstm_2_while_forward_lstm_2_while_loop_counter@
<forward_lstm_2_while_forward_lstm_2_while_maximum_iterations$
 forward_lstm_2_while_placeholder&
"forward_lstm_2_while_placeholder_1&
"forward_lstm_2_while_placeholder_2&
"forward_lstm_2_while_placeholder_3&
"forward_lstm_2_while_placeholder_4<
8forward_lstm_2_while_less_forward_lstm_2_strided_slice_1R
Nforward_lstm_2_while_forward_lstm_2_while_cond_363575___redundant_placeholder0R
Nforward_lstm_2_while_forward_lstm_2_while_cond_363575___redundant_placeholder1R
Nforward_lstm_2_while_forward_lstm_2_while_cond_363575___redundant_placeholder2R
Nforward_lstm_2_while_forward_lstm_2_while_cond_363575___redundant_placeholder3R
Nforward_lstm_2_while_forward_lstm_2_while_cond_363575___redundant_placeholder4!
forward_lstm_2_while_identity
Л
forward_lstm_2/while/LessLess forward_lstm_2_while_placeholder8forward_lstm_2_while_less_forward_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_2/while/Less
forward_lstm_2/while/IdentityIdentityforward_lstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2
forward_lstm_2/while/Identity"G
forward_lstm_2_while_identity&forward_lstm_2/while/Identity:output:0*(
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
F

J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_361134

inputs%
lstm_cell_7_361052:	Ш%
lstm_cell_7_361054:	2Ш!
lstm_cell_7_361056:	Ш
identityЂ#lstm_cell_7/StatefulPartitionedCallЂwhileD
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
strided_slice_2
#lstm_cell_7/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_7_361052lstm_cell_7_361054lstm_cell_7_361056*
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
GPU 2J 8 *P
fKRI
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_3610512%
#lstm_cell_7/StatefulPartitionedCall
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
while/loop_counterН
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_7_361052lstm_cell_7_361054lstm_cell_7_361056*
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
bodyR
while_body_361065*
condR
while_cond_361064*K
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

Identity|
NoOpNoOp$^lstm_cell_7/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2J
#lstm_cell_7/StatefulPartitionedCall#lstm_cell_7/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ЊБ
З
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_363852

inputs
inputs_1	L
9forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource:	ШN
;forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource:	2ШI
:forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource:	ШM
:backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource:	ШO
<backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource:	2ШJ
;backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource:	Ш
identityЂ2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpЂ1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpЂ3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpЂbackward_lstm_2/whileЂ1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpЂ0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpЂ2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpЂforward_lstm_2/while
#forward_lstm_2/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2%
#forward_lstm_2/RaggedToTensor/zeros
#forward_lstm_2/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2%
#forward_lstm_2/RaggedToTensor/Const
2forward_lstm_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor,forward_lstm_2/RaggedToTensor/Const:output:0inputs,forward_lstm_2/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS24
2forward_lstm_2/RaggedToTensor/RaggedTensorToTensorР
9forward_lstm_2/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9forward_lstm_2/RaggedNestedRowLengths/strided_slice/stackФ
;forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1Ф
;forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2
3forward_lstm_2/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Bforward_lstm_2/RaggedNestedRowLengths/strided_slice/stack:output:0Dforward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1:output:0Dforward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask25
3forward_lstm_2/RaggedNestedRowLengths/strided_sliceФ
;forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stackб
=forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2?
=forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1Ш
=forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2Ћ
5forward_lstm_2/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Dforward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack:output:0Fforward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Fforward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask27
5forward_lstm_2/RaggedNestedRowLengths/strided_slice_1
)forward_lstm_2/RaggedNestedRowLengths/subSub<forward_lstm_2/RaggedNestedRowLengths/strided_slice:output:0>forward_lstm_2/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2+
)forward_lstm_2/RaggedNestedRowLengths/sub
forward_lstm_2/CastCast-forward_lstm_2/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_2/Cast
forward_lstm_2/ShapeShape;forward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_2/Shape
"forward_lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"forward_lstm_2/strided_slice/stack
$forward_lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_2/strided_slice/stack_1
$forward_lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_2/strided_slice/stack_2М
forward_lstm_2/strided_sliceStridedSliceforward_lstm_2/Shape:output:0+forward_lstm_2/strided_slice/stack:output:0-forward_lstm_2/strided_slice/stack_1:output:0-forward_lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_2/strided_slicez
forward_lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_2/zeros/mul/yЈ
forward_lstm_2/zeros/mulMul%forward_lstm_2/strided_slice:output:0#forward_lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros/mul}
forward_lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_2/zeros/Less/yЃ
forward_lstm_2/zeros/LessLessforward_lstm_2/zeros/mul:z:0$forward_lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros/Less
forward_lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_2/zeros/packed/1П
forward_lstm_2/zeros/packedPack%forward_lstm_2/strided_slice:output:0&forward_lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_2/zeros/packed
forward_lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_2/zeros/ConstБ
forward_lstm_2/zerosFill$forward_lstm_2/zeros/packed:output:0#forward_lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/zeros~
forward_lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_2/zeros_1/mul/yЎ
forward_lstm_2/zeros_1/mulMul%forward_lstm_2/strided_slice:output:0%forward_lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros_1/mul
forward_lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_2/zeros_1/Less/yЋ
forward_lstm_2/zeros_1/LessLessforward_lstm_2/zeros_1/mul:z:0&forward_lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros_1/Less
forward_lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_2/zeros_1/packed/1Х
forward_lstm_2/zeros_1/packedPack%forward_lstm_2/strided_slice:output:0(forward_lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_2/zeros_1/packed
forward_lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_2/zeros_1/ConstЙ
forward_lstm_2/zeros_1Fill&forward_lstm_2/zeros_1/packed:output:0%forward_lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/zeros_1
forward_lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
forward_lstm_2/transpose/permх
forward_lstm_2/transpose	Transpose;forward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0&forward_lstm_2/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_2/transpose|
forward_lstm_2/Shape_1Shapeforward_lstm_2/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_2/Shape_1
$forward_lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_2/strided_slice_1/stack
&forward_lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_1/stack_1
&forward_lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_1/stack_2Ш
forward_lstm_2/strided_slice_1StridedSliceforward_lstm_2/Shape_1:output:0-forward_lstm_2/strided_slice_1/stack:output:0/forward_lstm_2/strided_slice_1/stack_1:output:0/forward_lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_2/strided_slice_1Ѓ
*forward_lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*forward_lstm_2/TensorArrayV2/element_shapeю
forward_lstm_2/TensorArrayV2TensorListReserve3forward_lstm_2/TensorArrayV2/element_shape:output:0'forward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_2/TensorArrayV2н
Dforward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2F
Dforward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeД
6forward_lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_2/transpose:y:0Mforward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6forward_lstm_2/TensorArrayUnstack/TensorListFromTensor
$forward_lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_2/strided_slice_2/stack
&forward_lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_2/stack_1
&forward_lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_2/stack_2ж
forward_lstm_2/strided_slice_2StridedSliceforward_lstm_2/transpose:y:0-forward_lstm_2/strided_slice_2/stack:output:0/forward_lstm_2/strided_slice_2/stack_1:output:0/forward_lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2 
forward_lstm_2/strided_slice_2п
0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp9forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype022
0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpц
!forward_lstm_2/lstm_cell_7/MatMulMatMul'forward_lstm_2/strided_slice_2:output:08forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2#
!forward_lstm_2/lstm_cell_7/MatMulх
2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp;forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype024
2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpт
#forward_lstm_2/lstm_cell_7/MatMul_1MatMulforward_lstm_2/zeros:output:0:forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#forward_lstm_2/lstm_cell_7/MatMul_1и
forward_lstm_2/lstm_cell_7/addAddV2+forward_lstm_2/lstm_cell_7/MatMul:product:0-forward_lstm_2/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2 
forward_lstm_2/lstm_cell_7/addо
1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp:forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype023
1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpх
"forward_lstm_2/lstm_cell_7/BiasAddBiasAdd"forward_lstm_2/lstm_cell_7/add:z:09forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_2/lstm_cell_7/BiasAdd
*forward_lstm_2/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*forward_lstm_2/lstm_cell_7/split/split_dimЋ
 forward_lstm_2/lstm_cell_7/splitSplit3forward_lstm_2/lstm_cell_7/split/split_dim:output:0+forward_lstm_2/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2"
 forward_lstm_2/lstm_cell_7/splitА
"forward_lstm_2/lstm_cell_7/SigmoidSigmoid)forward_lstm_2/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_2/lstm_cell_7/SigmoidД
$forward_lstm_2/lstm_cell_7/Sigmoid_1Sigmoid)forward_lstm_2/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_2/lstm_cell_7/Sigmoid_1Ф
forward_lstm_2/lstm_cell_7/mulMul(forward_lstm_2/lstm_cell_7/Sigmoid_1:y:0forward_lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
forward_lstm_2/lstm_cell_7/mulЇ
forward_lstm_2/lstm_cell_7/ReluRelu)forward_lstm_2/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/lstm_cell_7/Reluд
 forward_lstm_2/lstm_cell_7/mul_1Mul&forward_lstm_2/lstm_cell_7/Sigmoid:y:0-forward_lstm_2/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_2/lstm_cell_7/mul_1Щ
 forward_lstm_2/lstm_cell_7/add_1AddV2"forward_lstm_2/lstm_cell_7/mul:z:0$forward_lstm_2/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_2/lstm_cell_7/add_1Д
$forward_lstm_2/lstm_cell_7/Sigmoid_2Sigmoid)forward_lstm_2/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_2/lstm_cell_7/Sigmoid_2І
!forward_lstm_2/lstm_cell_7/Relu_1Relu$forward_lstm_2/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_2/lstm_cell_7/Relu_1и
 forward_lstm_2/lstm_cell_7/mul_2Mul(forward_lstm_2/lstm_cell_7/Sigmoid_2:y:0/forward_lstm_2/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_2/lstm_cell_7/mul_2­
,forward_lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2.
,forward_lstm_2/TensorArrayV2_1/element_shapeє
forward_lstm_2/TensorArrayV2_1TensorListReserve5forward_lstm_2/TensorArrayV2_1/element_shape:output:0'forward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_2/TensorArrayV2_1l
forward_lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_2/time
forward_lstm_2/zeros_like	ZerosLike$forward_lstm_2/lstm_cell_7/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/zeros_like
'forward_lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'forward_lstm_2/while/maximum_iterations
!forward_lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!forward_lstm_2/while/loop_counterч
forward_lstm_2/whileWhile*forward_lstm_2/while/loop_counter:output:00forward_lstm_2/while/maximum_iterations:output:0forward_lstm_2/time:output:0'forward_lstm_2/TensorArrayV2_1:handle:0forward_lstm_2/zeros_like:y:0forward_lstm_2/zeros:output:0forward_lstm_2/zeros_1:output:0'forward_lstm_2/strided_slice_1:output:0Fforward_lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_2/Cast:y:09forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource;forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource:forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *,
body$R"
 forward_lstm_2_while_body_363576*,
cond$R"
 forward_lstm_2_while_cond_363575*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_2/whileг
?forward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2A
?forward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape­
1forward_lstm_2/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_2/while:output:3Hforward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype023
1forward_lstm_2/TensorArrayV2Stack/TensorListStack
$forward_lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2&
$forward_lstm_2/strided_slice_3/stack
&forward_lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_2/strided_slice_3/stack_1
&forward_lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_3/stack_2є
forward_lstm_2/strided_slice_3StridedSlice:forward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0-forward_lstm_2/strided_slice_3/stack:output:0/forward_lstm_2/strided_slice_3/stack_1:output:0/forward_lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2 
forward_lstm_2/strided_slice_3
forward_lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_2/transpose_1/permъ
forward_lstm_2/transpose_1	Transpose:forward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0(forward_lstm_2/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_2/transpose_1
forward_lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_2/runtime
$backward_lstm_2/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$backward_lstm_2/RaggedToTensor/zeros
$backward_lstm_2/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2&
$backward_lstm_2/RaggedToTensor/Const
3backward_lstm_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-backward_lstm_2/RaggedToTensor/Const:output:0inputs-backward_lstm_2/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3backward_lstm_2/RaggedToTensor/RaggedTensorToTensorТ
:backward_lstm_2/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:backward_lstm_2/RaggedNestedRowLengths/strided_slice/stackЦ
<backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1Ц
<backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2Є
4backward_lstm_2/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cbackward_lstm_2/RaggedNestedRowLengths/strided_slice/stack:output:0Ebackward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1:output:0Ebackward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask26
4backward_lstm_2/RaggedNestedRowLengths/strided_sliceЦ
<backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stackг
>backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2@
>backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1Ъ
>backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2А
6backward_lstm_2/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Ebackward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gbackward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gbackward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask28
6backward_lstm_2/RaggedNestedRowLengths/strided_slice_1
*backward_lstm_2/RaggedNestedRowLengths/subSub=backward_lstm_2/RaggedNestedRowLengths/strided_slice:output:0?backward_lstm_2/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2,
*backward_lstm_2/RaggedNestedRowLengths/subЁ
backward_lstm_2/CastCast.backward_lstm_2/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_2/Cast
backward_lstm_2/ShapeShape<backward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_2/Shape
#backward_lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#backward_lstm_2/strided_slice/stack
%backward_lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_2/strided_slice/stack_1
%backward_lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_2/strided_slice/stack_2Т
backward_lstm_2/strided_sliceStridedSlicebackward_lstm_2/Shape:output:0,backward_lstm_2/strided_slice/stack:output:0.backward_lstm_2/strided_slice/stack_1:output:0.backward_lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
backward_lstm_2/strided_slice|
backward_lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_2/zeros/mul/yЌ
backward_lstm_2/zeros/mulMul&backward_lstm_2/strided_slice:output:0$backward_lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros/mul
backward_lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
backward_lstm_2/zeros/Less/yЇ
backward_lstm_2/zeros/LessLessbackward_lstm_2/zeros/mul:z:0%backward_lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros/Less
backward_lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_2/zeros/packed/1У
backward_lstm_2/zeros/packedPack&backward_lstm_2/strided_slice:output:0'backward_lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_2/zeros/packed
backward_lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_2/zeros/ConstЕ
backward_lstm_2/zerosFill%backward_lstm_2/zeros/packed:output:0$backward_lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/zeros
backward_lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_2/zeros_1/mul/yВ
backward_lstm_2/zeros_1/mulMul&backward_lstm_2/strided_slice:output:0&backward_lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros_1/mul
backward_lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_2/zeros_1/Less/yЏ
backward_lstm_2/zeros_1/LessLessbackward_lstm_2/zeros_1/mul:z:0'backward_lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros_1/Less
 backward_lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_2/zeros_1/packed/1Щ
backward_lstm_2/zeros_1/packedPack&backward_lstm_2/strided_slice:output:0)backward_lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_2/zeros_1/packed
backward_lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_2/zeros_1/ConstН
backward_lstm_2/zeros_1Fill'backward_lstm_2/zeros_1/packed:output:0&backward_lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/zeros_1
backward_lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
backward_lstm_2/transpose/permщ
backward_lstm_2/transpose	Transpose<backward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0'backward_lstm_2/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_2/transpose
backward_lstm_2/Shape_1Shapebackward_lstm_2/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_2/Shape_1
%backward_lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_2/strided_slice_1/stack
'backward_lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_1/stack_1
'backward_lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_1/stack_2Ю
backward_lstm_2/strided_slice_1StridedSlice backward_lstm_2/Shape_1:output:0.backward_lstm_2/strided_slice_1/stack:output:00backward_lstm_2/strided_slice_1/stack_1:output:00backward_lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_2/strided_slice_1Ѕ
+backward_lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+backward_lstm_2/TensorArrayV2/element_shapeђ
backward_lstm_2/TensorArrayV2TensorListReserve4backward_lstm_2/TensorArrayV2/element_shape:output:0(backward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
backward_lstm_2/TensorArrayV2
backward_lstm_2/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2 
backward_lstm_2/ReverseV2/axisЪ
backward_lstm_2/ReverseV2	ReverseV2backward_lstm_2/transpose:y:0'backward_lstm_2/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_2/ReverseV2п
Ebackward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2G
Ebackward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeН
7backward_lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"backward_lstm_2/ReverseV2:output:0Nbackward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7backward_lstm_2/TensorArrayUnstack/TensorListFromTensor
%backward_lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_2/strided_slice_2/stack
'backward_lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_2/stack_1
'backward_lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_2/stack_2м
backward_lstm_2/strided_slice_2StridedSlicebackward_lstm_2/transpose:y:0.backward_lstm_2/strided_slice_2/stack:output:00backward_lstm_2/strided_slice_2/stack_1:output:00backward_lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2!
backward_lstm_2/strided_slice_2т
1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp:backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype023
1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpъ
"backward_lstm_2/lstm_cell_8/MatMulMatMul(backward_lstm_2/strided_slice_2:output:09backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"backward_lstm_2/lstm_cell_8/MatMulш
3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp<backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype025
3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpц
$backward_lstm_2/lstm_cell_8/MatMul_1MatMulbackward_lstm_2/zeros:output:0;backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$backward_lstm_2/lstm_cell_8/MatMul_1м
backward_lstm_2/lstm_cell_8/addAddV2,backward_lstm_2/lstm_cell_8/MatMul:product:0.backward_lstm_2/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2!
backward_lstm_2/lstm_cell_8/addс
2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp;backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype024
2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpщ
#backward_lstm_2/lstm_cell_8/BiasAddBiasAdd#backward_lstm_2/lstm_cell_8/add:z:0:backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_2/lstm_cell_8/BiasAdd
+backward_lstm_2/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+backward_lstm_2/lstm_cell_8/split/split_dimЏ
!backward_lstm_2/lstm_cell_8/splitSplit4backward_lstm_2/lstm_cell_8/split/split_dim:output:0,backward_lstm_2/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2#
!backward_lstm_2/lstm_cell_8/splitГ
#backward_lstm_2/lstm_cell_8/SigmoidSigmoid*backward_lstm_2/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_2/lstm_cell_8/SigmoidЗ
%backward_lstm_2/lstm_cell_8/Sigmoid_1Sigmoid*backward_lstm_2/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_2/lstm_cell_8/Sigmoid_1Ш
backward_lstm_2/lstm_cell_8/mulMul)backward_lstm_2/lstm_cell_8/Sigmoid_1:y:0 backward_lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22!
backward_lstm_2/lstm_cell_8/mulЊ
 backward_lstm_2/lstm_cell_8/ReluRelu*backward_lstm_2/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/lstm_cell_8/Reluи
!backward_lstm_2/lstm_cell_8/mul_1Mul'backward_lstm_2/lstm_cell_8/Sigmoid:y:0.backward_lstm_2/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_2/lstm_cell_8/mul_1Э
!backward_lstm_2/lstm_cell_8/add_1AddV2#backward_lstm_2/lstm_cell_8/mul:z:0%backward_lstm_2/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_2/lstm_cell_8/add_1З
%backward_lstm_2/lstm_cell_8/Sigmoid_2Sigmoid*backward_lstm_2/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_2/lstm_cell_8/Sigmoid_2Љ
"backward_lstm_2/lstm_cell_8/Relu_1Relu%backward_lstm_2/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_2/lstm_cell_8/Relu_1м
!backward_lstm_2/lstm_cell_8/mul_2Mul)backward_lstm_2/lstm_cell_8/Sigmoid_2:y:00backward_lstm_2/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_2/lstm_cell_8/mul_2Џ
-backward_lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2/
-backward_lstm_2/TensorArrayV2_1/element_shapeј
backward_lstm_2/TensorArrayV2_1TensorListReserve6backward_lstm_2/TensorArrayV2_1/element_shape:output:0(backward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_2/TensorArrayV2_1n
backward_lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_2/time
%backward_lstm_2/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2'
%backward_lstm_2/Max/reduction_indices
backward_lstm_2/MaxMaxbackward_lstm_2/Cast:y:0.backward_lstm_2/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/Maxp
backward_lstm_2/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_2/sub/y
backward_lstm_2/subSubbackward_lstm_2/Max:output:0backward_lstm_2/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/sub
backward_lstm_2/Sub_1Subbackward_lstm_2/sub:z:0backward_lstm_2/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_2/Sub_1
backward_lstm_2/zeros_like	ZerosLike%backward_lstm_2/lstm_cell_8/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/zeros_like
(backward_lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(backward_lstm_2/while/maximum_iterations
"backward_lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"backward_lstm_2/while/loop_counterљ
backward_lstm_2/whileWhile+backward_lstm_2/while/loop_counter:output:01backward_lstm_2/while/maximum_iterations:output:0backward_lstm_2/time:output:0(backward_lstm_2/TensorArrayV2_1:handle:0backward_lstm_2/zeros_like:y:0backward_lstm_2/zeros:output:0 backward_lstm_2/zeros_1:output:0(backward_lstm_2/strided_slice_1:output:0Gbackward_lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_2/Sub_1:z:0:backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource<backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource;backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *-
body%R#
!backward_lstm_2_while_body_363755*-
cond%R#
!backward_lstm_2_while_cond_363754*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_2/whileе
@backward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2B
@backward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeБ
2backward_lstm_2/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_2/while:output:3Ibackward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype024
2backward_lstm_2/TensorArrayV2Stack/TensorListStackЁ
%backward_lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2'
%backward_lstm_2/strided_slice_3/stack
'backward_lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_2/strided_slice_3/stack_1
'backward_lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_3/stack_2њ
backward_lstm_2/strided_slice_3StridedSlice;backward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0.backward_lstm_2/strided_slice_3/stack:output:00backward_lstm_2/strided_slice_3/stack_1:output:00backward_lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2!
backward_lstm_2/strided_slice_3
 backward_lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_2/transpose_1/permю
backward_lstm_2/transpose_1	Transpose;backward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0)backward_lstm_2/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_2/transpose_1
backward_lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_2/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisР
concatConcatV2'forward_lstm_2/strided_slice_3:output:0(backward_lstm_2/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

IdentityИ
NoOpNoOp3^backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp2^backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp4^backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp^backward_lstm_2/while2^forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp1^forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp3^forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp^forward_lstm_2/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2h
2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp2f
1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp2j
3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp2.
backward_lstm_2/whilebackward_lstm_2/while2f
1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp2d
0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp2h
2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp2,
forward_lstm_2/whileforward_lstm_2/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ДU
Ч
!backward_lstm_2_while_body_364318<
8backward_lstm_2_while_backward_lstm_2_while_loop_counterB
>backward_lstm_2_while_backward_lstm_2_while_maximum_iterations%
!backward_lstm_2_while_placeholder'
#backward_lstm_2_while_placeholder_1'
#backward_lstm_2_while_placeholder_2'
#backward_lstm_2_while_placeholder_3;
7backward_lstm_2_while_backward_lstm_2_strided_slice_1_0w
sbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0U
Bbackward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0:	ШW
Dbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0:	2ШR
Cbackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ш"
backward_lstm_2_while_identity$
 backward_lstm_2_while_identity_1$
 backward_lstm_2_while_identity_2$
 backward_lstm_2_while_identity_3$
 backward_lstm_2_while_identity_4$
 backward_lstm_2_while_identity_59
5backward_lstm_2_while_backward_lstm_2_strided_slice_1u
qbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensorS
@backward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource:	ШU
Bbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource:	2ШP
Abackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource:	ШЂ8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOpЂ7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOpЂ9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOpу
Gbackward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2I
Gbackward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeМ
9backward_lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0!backward_lstm_2_while_placeholderPbackward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02;
9backward_lstm_2/while/TensorArrayV2Read/TensorListGetItemі
7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOpBbackward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype029
7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp
(backward_lstm_2/while/lstm_cell_8/MatMulMatMul@backward_lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0?backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_2/while/lstm_cell_8/MatMulќ
9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOpDbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02;
9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp§
*backward_lstm_2/while/lstm_cell_8/MatMul_1MatMul#backward_lstm_2_while_placeholder_2Abackward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2,
*backward_lstm_2/while/lstm_cell_8/MatMul_1є
%backward_lstm_2/while/lstm_cell_8/addAddV22backward_lstm_2/while/lstm_cell_8/MatMul:product:04backward_lstm_2/while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%backward_lstm_2/while/lstm_cell_8/addѕ
8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOpCbackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02:
8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp
)backward_lstm_2/while/lstm_cell_8/BiasAddBiasAdd)backward_lstm_2/while/lstm_cell_8/add:z:0@backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_2/while/lstm_cell_8/BiasAddЈ
1backward_lstm_2/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1backward_lstm_2/while/lstm_cell_8/split/split_dimЧ
'backward_lstm_2/while/lstm_cell_8/splitSplit:backward_lstm_2/while/lstm_cell_8/split/split_dim:output:02backward_lstm_2/while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2)
'backward_lstm_2/while/lstm_cell_8/splitХ
)backward_lstm_2/while/lstm_cell_8/SigmoidSigmoid0backward_lstm_2/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_2/while/lstm_cell_8/SigmoidЩ
+backward_lstm_2/while/lstm_cell_8/Sigmoid_1Sigmoid0backward_lstm_2/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_2/while/lstm_cell_8/Sigmoid_1н
%backward_lstm_2/while/lstm_cell_8/mulMul/backward_lstm_2/while/lstm_cell_8/Sigmoid_1:y:0#backward_lstm_2_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_2/while/lstm_cell_8/mulМ
&backward_lstm_2/while/lstm_cell_8/ReluRelu0backward_lstm_2/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_2/while/lstm_cell_8/Relu№
'backward_lstm_2/while/lstm_cell_8/mul_1Mul-backward_lstm_2/while/lstm_cell_8/Sigmoid:y:04backward_lstm_2/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_2/while/lstm_cell_8/mul_1х
'backward_lstm_2/while/lstm_cell_8/add_1AddV2)backward_lstm_2/while/lstm_cell_8/mul:z:0+backward_lstm_2/while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_2/while/lstm_cell_8/add_1Щ
+backward_lstm_2/while/lstm_cell_8/Sigmoid_2Sigmoid0backward_lstm_2/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_2/while/lstm_cell_8/Sigmoid_2Л
(backward_lstm_2/while/lstm_cell_8/Relu_1Relu+backward_lstm_2/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_2/while/lstm_cell_8/Relu_1є
'backward_lstm_2/while/lstm_cell_8/mul_2Mul/backward_lstm_2/while/lstm_cell_8/Sigmoid_2:y:06backward_lstm_2/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_2/while/lstm_cell_8/mul_2Џ
:backward_lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#backward_lstm_2_while_placeholder_1!backward_lstm_2_while_placeholder+backward_lstm_2/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:backward_lstm_2/while/TensorArrayV2Write/TensorListSetItem|
backward_lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_2/while/add/yЉ
backward_lstm_2/while/addAddV2!backward_lstm_2_while_placeholder$backward_lstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/while/add
backward_lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_2/while/add_1/yЦ
backward_lstm_2/while/add_1AddV28backward_lstm_2_while_backward_lstm_2_while_loop_counter&backward_lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/while/add_1Ћ
backward_lstm_2/while/IdentityIdentitybackward_lstm_2/while/add_1:z:0^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2 
backward_lstm_2/while/IdentityЮ
 backward_lstm_2/while/Identity_1Identity>backward_lstm_2_while_backward_lstm_2_while_maximum_iterations^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_2/while/Identity_1­
 backward_lstm_2/while/Identity_2Identitybackward_lstm_2/while/add:z:0^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_2/while/Identity_2к
 backward_lstm_2/while/Identity_3IdentityJbackward_lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_2/while/Identity_3Ь
 backward_lstm_2/while/Identity_4Identity+backward_lstm_2/while/lstm_cell_8/mul_2:z:0^backward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/while/Identity_4Ь
 backward_lstm_2/while/Identity_5Identity+backward_lstm_2/while/lstm_cell_8/add_1:z:0^backward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/while/Identity_5Ћ
backward_lstm_2/while/NoOpNoOp9^backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp8^backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp:^backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_2/while/NoOp"p
5backward_lstm_2_while_backward_lstm_2_strided_slice_17backward_lstm_2_while_backward_lstm_2_strided_slice_1_0"I
backward_lstm_2_while_identity'backward_lstm_2/while/Identity:output:0"M
 backward_lstm_2_while_identity_1)backward_lstm_2/while/Identity_1:output:0"M
 backward_lstm_2_while_identity_2)backward_lstm_2/while/Identity_2:output:0"M
 backward_lstm_2_while_identity_3)backward_lstm_2/while/Identity_3:output:0"M
 backward_lstm_2_while_identity_4)backward_lstm_2/while/Identity_4:output:0"M
 backward_lstm_2_while_identity_5)backward_lstm_2/while/Identity_5:output:0"
Abackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resourceCbackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0"
Bbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resourceDbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0"
@backward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resourceBbackward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0"ш
qbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensorsbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2t
8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp2r
7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp2v
9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
эa

!backward_lstm_2_while_body_365325<
8backward_lstm_2_while_backward_lstm_2_while_loop_counterB
>backward_lstm_2_while_backward_lstm_2_while_maximum_iterations%
!backward_lstm_2_while_placeholder'
#backward_lstm_2_while_placeholder_1'
#backward_lstm_2_while_placeholder_2'
#backward_lstm_2_while_placeholder_3'
#backward_lstm_2_while_placeholder_4;
7backward_lstm_2_while_backward_lstm_2_strided_slice_1_0w
sbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_06
2backward_lstm_2_while_less_backward_lstm_2_sub_1_0U
Bbackward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0:	ШW
Dbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0:	2ШR
Cbackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ш"
backward_lstm_2_while_identity$
 backward_lstm_2_while_identity_1$
 backward_lstm_2_while_identity_2$
 backward_lstm_2_while_identity_3$
 backward_lstm_2_while_identity_4$
 backward_lstm_2_while_identity_5$
 backward_lstm_2_while_identity_69
5backward_lstm_2_while_backward_lstm_2_strided_slice_1u
qbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor4
0backward_lstm_2_while_less_backward_lstm_2_sub_1S
@backward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource:	ШU
Bbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource:	2ШP
Abackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource:	ШЂ8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOpЂ7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOpЂ9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOpу
Gbackward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gbackward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
9backward_lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0!backward_lstm_2_while_placeholderPbackward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02;
9backward_lstm_2/while/TensorArrayV2Read/TensorListGetItemХ
backward_lstm_2/while/LessLess2backward_lstm_2_while_less_backward_lstm_2_sub_1_0!backward_lstm_2_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_2/while/Lessі
7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOpBbackward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype029
7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp
(backward_lstm_2/while/lstm_cell_8/MatMulMatMul@backward_lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0?backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_2/while/lstm_cell_8/MatMulќ
9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOpDbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02;
9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp§
*backward_lstm_2/while/lstm_cell_8/MatMul_1MatMul#backward_lstm_2_while_placeholder_3Abackward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2,
*backward_lstm_2/while/lstm_cell_8/MatMul_1є
%backward_lstm_2/while/lstm_cell_8/addAddV22backward_lstm_2/while/lstm_cell_8/MatMul:product:04backward_lstm_2/while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%backward_lstm_2/while/lstm_cell_8/addѕ
8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOpCbackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02:
8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp
)backward_lstm_2/while/lstm_cell_8/BiasAddBiasAdd)backward_lstm_2/while/lstm_cell_8/add:z:0@backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_2/while/lstm_cell_8/BiasAddЈ
1backward_lstm_2/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1backward_lstm_2/while/lstm_cell_8/split/split_dimЧ
'backward_lstm_2/while/lstm_cell_8/splitSplit:backward_lstm_2/while/lstm_cell_8/split/split_dim:output:02backward_lstm_2/while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2)
'backward_lstm_2/while/lstm_cell_8/splitХ
)backward_lstm_2/while/lstm_cell_8/SigmoidSigmoid0backward_lstm_2/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_2/while/lstm_cell_8/SigmoidЩ
+backward_lstm_2/while/lstm_cell_8/Sigmoid_1Sigmoid0backward_lstm_2/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_2/while/lstm_cell_8/Sigmoid_1н
%backward_lstm_2/while/lstm_cell_8/mulMul/backward_lstm_2/while/lstm_cell_8/Sigmoid_1:y:0#backward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_2/while/lstm_cell_8/mulМ
&backward_lstm_2/while/lstm_cell_8/ReluRelu0backward_lstm_2/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_2/while/lstm_cell_8/Relu№
'backward_lstm_2/while/lstm_cell_8/mul_1Mul-backward_lstm_2/while/lstm_cell_8/Sigmoid:y:04backward_lstm_2/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_2/while/lstm_cell_8/mul_1х
'backward_lstm_2/while/lstm_cell_8/add_1AddV2)backward_lstm_2/while/lstm_cell_8/mul:z:0+backward_lstm_2/while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_2/while/lstm_cell_8/add_1Щ
+backward_lstm_2/while/lstm_cell_8/Sigmoid_2Sigmoid0backward_lstm_2/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_2/while/lstm_cell_8/Sigmoid_2Л
(backward_lstm_2/while/lstm_cell_8/Relu_1Relu+backward_lstm_2/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_2/while/lstm_cell_8/Relu_1є
'backward_lstm_2/while/lstm_cell_8/mul_2Mul/backward_lstm_2/while/lstm_cell_8/Sigmoid_2:y:06backward_lstm_2/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_2/while/lstm_cell_8/mul_2ъ
backward_lstm_2/while/SelectSelectbackward_lstm_2/while/Less:z:0+backward_lstm_2/while/lstm_cell_8/mul_2:z:0#backward_lstm_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/while/Selectю
backward_lstm_2/while/Select_1Selectbackward_lstm_2/while/Less:z:0+backward_lstm_2/while/lstm_cell_8/mul_2:z:0#backward_lstm_2_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_2/while/Select_1ю
backward_lstm_2/while/Select_2Selectbackward_lstm_2/while/Less:z:0+backward_lstm_2/while/lstm_cell_8/add_1:z:0#backward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_2/while/Select_2Љ
:backward_lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#backward_lstm_2_while_placeholder_1!backward_lstm_2_while_placeholder%backward_lstm_2/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:backward_lstm_2/while/TensorArrayV2Write/TensorListSetItem|
backward_lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_2/while/add/yЉ
backward_lstm_2/while/addAddV2!backward_lstm_2_while_placeholder$backward_lstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/while/add
backward_lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_2/while/add_1/yЦ
backward_lstm_2/while/add_1AddV28backward_lstm_2_while_backward_lstm_2_while_loop_counter&backward_lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/while/add_1Ћ
backward_lstm_2/while/IdentityIdentitybackward_lstm_2/while/add_1:z:0^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2 
backward_lstm_2/while/IdentityЮ
 backward_lstm_2/while/Identity_1Identity>backward_lstm_2_while_backward_lstm_2_while_maximum_iterations^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_2/while/Identity_1­
 backward_lstm_2/while/Identity_2Identitybackward_lstm_2/while/add:z:0^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_2/while/Identity_2к
 backward_lstm_2/while/Identity_3IdentityJbackward_lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_2/while/Identity_3Ц
 backward_lstm_2/while/Identity_4Identity%backward_lstm_2/while/Select:output:0^backward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/while/Identity_4Ш
 backward_lstm_2/while/Identity_5Identity'backward_lstm_2/while/Select_1:output:0^backward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/while/Identity_5Ш
 backward_lstm_2/while/Identity_6Identity'backward_lstm_2/while/Select_2:output:0^backward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/while/Identity_6Ћ
backward_lstm_2/while/NoOpNoOp9^backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp8^backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp:^backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_2/while/NoOp"p
5backward_lstm_2_while_backward_lstm_2_strided_slice_17backward_lstm_2_while_backward_lstm_2_strided_slice_1_0"I
backward_lstm_2_while_identity'backward_lstm_2/while/Identity:output:0"M
 backward_lstm_2_while_identity_1)backward_lstm_2/while/Identity_1:output:0"M
 backward_lstm_2_while_identity_2)backward_lstm_2/while/Identity_2:output:0"M
 backward_lstm_2_while_identity_3)backward_lstm_2/while/Identity_3:output:0"M
 backward_lstm_2_while_identity_4)backward_lstm_2/while/Identity_4:output:0"M
 backward_lstm_2_while_identity_5)backward_lstm_2/while/Identity_5:output:0"M
 backward_lstm_2_while_identity_6)backward_lstm_2/while/Identity_6:output:0"f
0backward_lstm_2_while_less_backward_lstm_2_sub_12backward_lstm_2_while_less_backward_lstm_2_sub_1_0"
Abackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resourceCbackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0"
Bbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resourceDbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0"
@backward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resourceBbackward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0"ш
qbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensorsbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2t
8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp2r
7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp2v
9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Ўѓ
Ћ
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_364404
inputs_0L
9forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource:	ШN
;forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource:	2ШI
:forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource:	ШM
:backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource:	ШO
<backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource:	2ШJ
;backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource:	Ш
identityЂ2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpЂ1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpЂ3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpЂbackward_lstm_2/whileЂ1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpЂ0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpЂ2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpЂforward_lstm_2/whiled
forward_lstm_2/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_2/Shape
"forward_lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"forward_lstm_2/strided_slice/stack
$forward_lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_2/strided_slice/stack_1
$forward_lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_2/strided_slice/stack_2М
forward_lstm_2/strided_sliceStridedSliceforward_lstm_2/Shape:output:0+forward_lstm_2/strided_slice/stack:output:0-forward_lstm_2/strided_slice/stack_1:output:0-forward_lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_2/strided_slicez
forward_lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_2/zeros/mul/yЈ
forward_lstm_2/zeros/mulMul%forward_lstm_2/strided_slice:output:0#forward_lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros/mul}
forward_lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_2/zeros/Less/yЃ
forward_lstm_2/zeros/LessLessforward_lstm_2/zeros/mul:z:0$forward_lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros/Less
forward_lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_2/zeros/packed/1П
forward_lstm_2/zeros/packedPack%forward_lstm_2/strided_slice:output:0&forward_lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_2/zeros/packed
forward_lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_2/zeros/ConstБ
forward_lstm_2/zerosFill$forward_lstm_2/zeros/packed:output:0#forward_lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/zeros~
forward_lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_2/zeros_1/mul/yЎ
forward_lstm_2/zeros_1/mulMul%forward_lstm_2/strided_slice:output:0%forward_lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros_1/mul
forward_lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_2/zeros_1/Less/yЋ
forward_lstm_2/zeros_1/LessLessforward_lstm_2/zeros_1/mul:z:0&forward_lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros_1/Less
forward_lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_2/zeros_1/packed/1Х
forward_lstm_2/zeros_1/packedPack%forward_lstm_2/strided_slice:output:0(forward_lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_2/zeros_1/packed
forward_lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_2/zeros_1/ConstЙ
forward_lstm_2/zeros_1Fill&forward_lstm_2/zeros_1/packed:output:0%forward_lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/zeros_1
forward_lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
forward_lstm_2/transpose/permЛ
forward_lstm_2/transpose	Transposeinputs_0&forward_lstm_2/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
forward_lstm_2/transpose|
forward_lstm_2/Shape_1Shapeforward_lstm_2/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_2/Shape_1
$forward_lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_2/strided_slice_1/stack
&forward_lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_1/stack_1
&forward_lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_1/stack_2Ш
forward_lstm_2/strided_slice_1StridedSliceforward_lstm_2/Shape_1:output:0-forward_lstm_2/strided_slice_1/stack:output:0/forward_lstm_2/strided_slice_1/stack_1:output:0/forward_lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_2/strided_slice_1Ѓ
*forward_lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*forward_lstm_2/TensorArrayV2/element_shapeю
forward_lstm_2/TensorArrayV2TensorListReserve3forward_lstm_2/TensorArrayV2/element_shape:output:0'forward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_2/TensorArrayV2н
Dforward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2F
Dforward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeД
6forward_lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_2/transpose:y:0Mforward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6forward_lstm_2/TensorArrayUnstack/TensorListFromTensor
$forward_lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_2/strided_slice_2/stack
&forward_lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_2/stack_1
&forward_lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_2/stack_2п
forward_lstm_2/strided_slice_2StridedSliceforward_lstm_2/transpose:y:0-forward_lstm_2/strided_slice_2/stack:output:0/forward_lstm_2/strided_slice_2/stack_1:output:0/forward_lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2 
forward_lstm_2/strided_slice_2п
0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp9forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype022
0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpц
!forward_lstm_2/lstm_cell_7/MatMulMatMul'forward_lstm_2/strided_slice_2:output:08forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2#
!forward_lstm_2/lstm_cell_7/MatMulх
2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp;forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype024
2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpт
#forward_lstm_2/lstm_cell_7/MatMul_1MatMulforward_lstm_2/zeros:output:0:forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#forward_lstm_2/lstm_cell_7/MatMul_1и
forward_lstm_2/lstm_cell_7/addAddV2+forward_lstm_2/lstm_cell_7/MatMul:product:0-forward_lstm_2/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2 
forward_lstm_2/lstm_cell_7/addо
1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp:forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype023
1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpх
"forward_lstm_2/lstm_cell_7/BiasAddBiasAdd"forward_lstm_2/lstm_cell_7/add:z:09forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_2/lstm_cell_7/BiasAdd
*forward_lstm_2/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*forward_lstm_2/lstm_cell_7/split/split_dimЋ
 forward_lstm_2/lstm_cell_7/splitSplit3forward_lstm_2/lstm_cell_7/split/split_dim:output:0+forward_lstm_2/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2"
 forward_lstm_2/lstm_cell_7/splitА
"forward_lstm_2/lstm_cell_7/SigmoidSigmoid)forward_lstm_2/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_2/lstm_cell_7/SigmoidД
$forward_lstm_2/lstm_cell_7/Sigmoid_1Sigmoid)forward_lstm_2/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_2/lstm_cell_7/Sigmoid_1Ф
forward_lstm_2/lstm_cell_7/mulMul(forward_lstm_2/lstm_cell_7/Sigmoid_1:y:0forward_lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
forward_lstm_2/lstm_cell_7/mulЇ
forward_lstm_2/lstm_cell_7/ReluRelu)forward_lstm_2/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/lstm_cell_7/Reluд
 forward_lstm_2/lstm_cell_7/mul_1Mul&forward_lstm_2/lstm_cell_7/Sigmoid:y:0-forward_lstm_2/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_2/lstm_cell_7/mul_1Щ
 forward_lstm_2/lstm_cell_7/add_1AddV2"forward_lstm_2/lstm_cell_7/mul:z:0$forward_lstm_2/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_2/lstm_cell_7/add_1Д
$forward_lstm_2/lstm_cell_7/Sigmoid_2Sigmoid)forward_lstm_2/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_2/lstm_cell_7/Sigmoid_2І
!forward_lstm_2/lstm_cell_7/Relu_1Relu$forward_lstm_2/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_2/lstm_cell_7/Relu_1и
 forward_lstm_2/lstm_cell_7/mul_2Mul(forward_lstm_2/lstm_cell_7/Sigmoid_2:y:0/forward_lstm_2/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_2/lstm_cell_7/mul_2­
,forward_lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2.
,forward_lstm_2/TensorArrayV2_1/element_shapeє
forward_lstm_2/TensorArrayV2_1TensorListReserve5forward_lstm_2/TensorArrayV2_1/element_shape:output:0'forward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_2/TensorArrayV2_1l
forward_lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_2/time
'forward_lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'forward_lstm_2/while/maximum_iterations
!forward_lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!forward_lstm_2/while/loop_counterщ
forward_lstm_2/whileWhile*forward_lstm_2/while/loop_counter:output:00forward_lstm_2/while/maximum_iterations:output:0forward_lstm_2/time:output:0'forward_lstm_2/TensorArrayV2_1:handle:0forward_lstm_2/zeros:output:0forward_lstm_2/zeros_1:output:0'forward_lstm_2/strided_slice_1:output:0Fforward_lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:09forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource;forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource:forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *,
body$R"
 forward_lstm_2_while_body_364169*,
cond$R"
 forward_lstm_2_while_cond_364168*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
forward_lstm_2/whileг
?forward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2A
?forward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape­
1forward_lstm_2/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_2/while:output:3Hforward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype023
1forward_lstm_2/TensorArrayV2Stack/TensorListStack
$forward_lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2&
$forward_lstm_2/strided_slice_3/stack
&forward_lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_2/strided_slice_3/stack_1
&forward_lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_3/stack_2є
forward_lstm_2/strided_slice_3StridedSlice:forward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0-forward_lstm_2/strided_slice_3/stack:output:0/forward_lstm_2/strided_slice_3/stack_1:output:0/forward_lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2 
forward_lstm_2/strided_slice_3
forward_lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_2/transpose_1/permъ
forward_lstm_2/transpose_1	Transpose:forward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0(forward_lstm_2/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_2/transpose_1
forward_lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_2/runtimef
backward_lstm_2/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_2/Shape
#backward_lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#backward_lstm_2/strided_slice/stack
%backward_lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_2/strided_slice/stack_1
%backward_lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_2/strided_slice/stack_2Т
backward_lstm_2/strided_sliceStridedSlicebackward_lstm_2/Shape:output:0,backward_lstm_2/strided_slice/stack:output:0.backward_lstm_2/strided_slice/stack_1:output:0.backward_lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
backward_lstm_2/strided_slice|
backward_lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_2/zeros/mul/yЌ
backward_lstm_2/zeros/mulMul&backward_lstm_2/strided_slice:output:0$backward_lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros/mul
backward_lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
backward_lstm_2/zeros/Less/yЇ
backward_lstm_2/zeros/LessLessbackward_lstm_2/zeros/mul:z:0%backward_lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros/Less
backward_lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_2/zeros/packed/1У
backward_lstm_2/zeros/packedPack&backward_lstm_2/strided_slice:output:0'backward_lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_2/zeros/packed
backward_lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_2/zeros/ConstЕ
backward_lstm_2/zerosFill%backward_lstm_2/zeros/packed:output:0$backward_lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/zeros
backward_lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_2/zeros_1/mul/yВ
backward_lstm_2/zeros_1/mulMul&backward_lstm_2/strided_slice:output:0&backward_lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros_1/mul
backward_lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_2/zeros_1/Less/yЏ
backward_lstm_2/zeros_1/LessLessbackward_lstm_2/zeros_1/mul:z:0'backward_lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros_1/Less
 backward_lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_2/zeros_1/packed/1Щ
backward_lstm_2/zeros_1/packedPack&backward_lstm_2/strided_slice:output:0)backward_lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_2/zeros_1/packed
backward_lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_2/zeros_1/ConstН
backward_lstm_2/zeros_1Fill'backward_lstm_2/zeros_1/packed:output:0&backward_lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/zeros_1
backward_lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
backward_lstm_2/transpose/permО
backward_lstm_2/transpose	Transposeinputs_0'backward_lstm_2/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_2/transpose
backward_lstm_2/Shape_1Shapebackward_lstm_2/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_2/Shape_1
%backward_lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_2/strided_slice_1/stack
'backward_lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_1/stack_1
'backward_lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_1/stack_2Ю
backward_lstm_2/strided_slice_1StridedSlice backward_lstm_2/Shape_1:output:0.backward_lstm_2/strided_slice_1/stack:output:00backward_lstm_2/strided_slice_1/stack_1:output:00backward_lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_2/strided_slice_1Ѕ
+backward_lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+backward_lstm_2/TensorArrayV2/element_shapeђ
backward_lstm_2/TensorArrayV2TensorListReserve4backward_lstm_2/TensorArrayV2/element_shape:output:0(backward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
backward_lstm_2/TensorArrayV2
backward_lstm_2/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2 
backward_lstm_2/ReverseV2/axisг
backward_lstm_2/ReverseV2	ReverseV2backward_lstm_2/transpose:y:0'backward_lstm_2/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_2/ReverseV2п
Ebackward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2G
Ebackward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeН
7backward_lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"backward_lstm_2/ReverseV2:output:0Nbackward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7backward_lstm_2/TensorArrayUnstack/TensorListFromTensor
%backward_lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_2/strided_slice_2/stack
'backward_lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_2/stack_1
'backward_lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_2/stack_2х
backward_lstm_2/strided_slice_2StridedSlicebackward_lstm_2/transpose:y:0.backward_lstm_2/strided_slice_2/stack:output:00backward_lstm_2/strided_slice_2/stack_1:output:00backward_lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2!
backward_lstm_2/strided_slice_2т
1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp:backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype023
1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpъ
"backward_lstm_2/lstm_cell_8/MatMulMatMul(backward_lstm_2/strided_slice_2:output:09backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"backward_lstm_2/lstm_cell_8/MatMulш
3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp<backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype025
3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpц
$backward_lstm_2/lstm_cell_8/MatMul_1MatMulbackward_lstm_2/zeros:output:0;backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$backward_lstm_2/lstm_cell_8/MatMul_1м
backward_lstm_2/lstm_cell_8/addAddV2,backward_lstm_2/lstm_cell_8/MatMul:product:0.backward_lstm_2/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2!
backward_lstm_2/lstm_cell_8/addс
2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp;backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype024
2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpщ
#backward_lstm_2/lstm_cell_8/BiasAddBiasAdd#backward_lstm_2/lstm_cell_8/add:z:0:backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_2/lstm_cell_8/BiasAdd
+backward_lstm_2/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+backward_lstm_2/lstm_cell_8/split/split_dimЏ
!backward_lstm_2/lstm_cell_8/splitSplit4backward_lstm_2/lstm_cell_8/split/split_dim:output:0,backward_lstm_2/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2#
!backward_lstm_2/lstm_cell_8/splitГ
#backward_lstm_2/lstm_cell_8/SigmoidSigmoid*backward_lstm_2/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_2/lstm_cell_8/SigmoidЗ
%backward_lstm_2/lstm_cell_8/Sigmoid_1Sigmoid*backward_lstm_2/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_2/lstm_cell_8/Sigmoid_1Ш
backward_lstm_2/lstm_cell_8/mulMul)backward_lstm_2/lstm_cell_8/Sigmoid_1:y:0 backward_lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22!
backward_lstm_2/lstm_cell_8/mulЊ
 backward_lstm_2/lstm_cell_8/ReluRelu*backward_lstm_2/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/lstm_cell_8/Reluи
!backward_lstm_2/lstm_cell_8/mul_1Mul'backward_lstm_2/lstm_cell_8/Sigmoid:y:0.backward_lstm_2/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_2/lstm_cell_8/mul_1Э
!backward_lstm_2/lstm_cell_8/add_1AddV2#backward_lstm_2/lstm_cell_8/mul:z:0%backward_lstm_2/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_2/lstm_cell_8/add_1З
%backward_lstm_2/lstm_cell_8/Sigmoid_2Sigmoid*backward_lstm_2/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_2/lstm_cell_8/Sigmoid_2Љ
"backward_lstm_2/lstm_cell_8/Relu_1Relu%backward_lstm_2/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_2/lstm_cell_8/Relu_1м
!backward_lstm_2/lstm_cell_8/mul_2Mul)backward_lstm_2/lstm_cell_8/Sigmoid_2:y:00backward_lstm_2/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_2/lstm_cell_8/mul_2Џ
-backward_lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2/
-backward_lstm_2/TensorArrayV2_1/element_shapeј
backward_lstm_2/TensorArrayV2_1TensorListReserve6backward_lstm_2/TensorArrayV2_1/element_shape:output:0(backward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_2/TensorArrayV2_1n
backward_lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_2/time
(backward_lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(backward_lstm_2/while/maximum_iterations
"backward_lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"backward_lstm_2/while/loop_counterј
backward_lstm_2/whileWhile+backward_lstm_2/while/loop_counter:output:01backward_lstm_2/while/maximum_iterations:output:0backward_lstm_2/time:output:0(backward_lstm_2/TensorArrayV2_1:handle:0backward_lstm_2/zeros:output:0 backward_lstm_2/zeros_1:output:0(backward_lstm_2/strided_slice_1:output:0Gbackward_lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0:backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource<backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource;backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *-
body%R#
!backward_lstm_2_while_body_364318*-
cond%R#
!backward_lstm_2_while_cond_364317*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
backward_lstm_2/whileе
@backward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2B
@backward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeБ
2backward_lstm_2/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_2/while:output:3Ibackward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype024
2backward_lstm_2/TensorArrayV2Stack/TensorListStackЁ
%backward_lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2'
%backward_lstm_2/strided_slice_3/stack
'backward_lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_2/strided_slice_3/stack_1
'backward_lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_3/stack_2њ
backward_lstm_2/strided_slice_3StridedSlice;backward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0.backward_lstm_2/strided_slice_3/stack:output:00backward_lstm_2/strided_slice_3/stack_1:output:00backward_lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2!
backward_lstm_2/strided_slice_3
 backward_lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_2/transpose_1/permю
backward_lstm_2/transpose_1	Transpose;backward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0)backward_lstm_2/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_2/transpose_1
backward_lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_2/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisР
concatConcatV2'forward_lstm_2/strided_slice_3:output:0(backward_lstm_2/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

IdentityИ
NoOpNoOp3^backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp2^backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp4^backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp^backward_lstm_2/while2^forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp1^forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp3^forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp^forward_lstm_2/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2h
2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp2f
1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp2j
3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp2.
backward_lstm_2/whilebackward_lstm_2/while2f
1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp2d
0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp2h
2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp2,
forward_lstm_2/whileforward_lstm_2/while:g c
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
ћ
ы
 forward_lstm_2_while_cond_363135:
6forward_lstm_2_while_forward_lstm_2_while_loop_counter@
<forward_lstm_2_while_forward_lstm_2_while_maximum_iterations$
 forward_lstm_2_while_placeholder&
"forward_lstm_2_while_placeholder_1&
"forward_lstm_2_while_placeholder_2&
"forward_lstm_2_while_placeholder_3&
"forward_lstm_2_while_placeholder_4<
8forward_lstm_2_while_less_forward_lstm_2_strided_slice_1R
Nforward_lstm_2_while_forward_lstm_2_while_cond_363135___redundant_placeholder0R
Nforward_lstm_2_while_forward_lstm_2_while_cond_363135___redundant_placeholder1R
Nforward_lstm_2_while_forward_lstm_2_while_cond_363135___redundant_placeholder2R
Nforward_lstm_2_while_forward_lstm_2_while_cond_363135___redundant_placeholder3R
Nforward_lstm_2_while_forward_lstm_2_while_cond_363135___redundant_placeholder4!
forward_lstm_2_while_identity
Л
forward_lstm_2/while/LessLess forward_lstm_2_while_placeholder8forward_lstm_2_while_less_forward_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_2/while/Less
forward_lstm_2/while/IdentityIdentityforward_lstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2
forward_lstm_2/while/Identity"G
forward_lstm_2_while_identity&forward_lstm_2/while/Identity:output:0*(
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
ћ
ы
 forward_lstm_2_while_cond_365145:
6forward_lstm_2_while_forward_lstm_2_while_loop_counter@
<forward_lstm_2_while_forward_lstm_2_while_maximum_iterations$
 forward_lstm_2_while_placeholder&
"forward_lstm_2_while_placeholder_1&
"forward_lstm_2_while_placeholder_2&
"forward_lstm_2_while_placeholder_3&
"forward_lstm_2_while_placeholder_4<
8forward_lstm_2_while_less_forward_lstm_2_strided_slice_1R
Nforward_lstm_2_while_forward_lstm_2_while_cond_365145___redundant_placeholder0R
Nforward_lstm_2_while_forward_lstm_2_while_cond_365145___redundant_placeholder1R
Nforward_lstm_2_while_forward_lstm_2_while_cond_365145___redundant_placeholder2R
Nforward_lstm_2_while_forward_lstm_2_while_cond_365145___redundant_placeholder3R
Nforward_lstm_2_while_forward_lstm_2_while_cond_365145___redundant_placeholder4!
forward_lstm_2_while_identity
Л
forward_lstm_2/while/LessLess forward_lstm_2_while_placeholder8forward_lstm_2_while_less_forward_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_2/while/Less
forward_lstm_2/while/IdentityIdentityforward_lstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2
forward_lstm_2/while/Identity"G
forward_lstm_2_while_identity&forward_lstm_2/while/Identity:output:0*(
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
о[

J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_365637
inputs_0=
*lstm_cell_7_matmul_readvariableop_resource:	Ш?
,lstm_cell_7_matmul_1_readvariableop_resource:	2Ш:
+lstm_cell_7_biasadd_readvariableop_resource:	Ш
identityЂ"lstm_cell_7/BiasAdd/ReadVariableOpЂ!lstm_cell_7/MatMul/ReadVariableOpЂ#lstm_cell_7/MatMul_1/ReadVariableOpЂwhileF
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
strided_slice_2В
!lstm_cell_7/MatMul/ReadVariableOpReadVariableOp*lstm_cell_7_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02#
!lstm_cell_7/MatMul/ReadVariableOpЊ
lstm_cell_7/MatMulMatMulstrided_slice_2:output:0)lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/MatMulИ
#lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02%
#lstm_cell_7/MatMul_1/ReadVariableOpІ
lstm_cell_7/MatMul_1MatMulzeros:output:0+lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/MatMul_1
lstm_cell_7/addAddV2lstm_cell_7/MatMul:product:0lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/addБ
"lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02$
"lstm_cell_7/BiasAdd/ReadVariableOpЉ
lstm_cell_7/BiasAddBiasAddlstm_cell_7/add:z:0*lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/BiasAdd|
lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_7/split/split_dimя
lstm_cell_7/splitSplit$lstm_cell_7/split/split_dim:output:0lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_7/split
lstm_cell_7/SigmoidSigmoidlstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Sigmoid
lstm_cell_7/Sigmoid_1Sigmoidlstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Sigmoid_1
lstm_cell_7/mulMullstm_cell_7/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/mulz
lstm_cell_7/ReluRelulstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Relu
lstm_cell_7/mul_1Mullstm_cell_7/Sigmoid:y:0lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/mul_1
lstm_cell_7/add_1AddV2lstm_cell_7/mul:z:0lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/add_1
lstm_cell_7/Sigmoid_2Sigmoidlstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Sigmoid_2y
lstm_cell_7/Relu_1Relulstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Relu_1
lstm_cell_7/mul_2Mullstm_cell_7/Sigmoid_2:y:0 lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_7_matmul_readvariableop_resource,lstm_cell_7_matmul_1_readvariableop_resource+lstm_cell_7_biasadd_readvariableop_resource*
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
bodyR
while_body_365553*
condR
while_cond_365552*K
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

IdentityХ
NoOpNoOp#^lstm_cell_7/BiasAdd/ReadVariableOp"^lstm_cell_7/MatMul/ReadVariableOp$^lstm_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2H
"lstm_cell_7/BiasAdd/ReadVariableOp"lstm_cell_7/BiasAdd/ReadVariableOp2F
!lstm_cell_7/MatMul/ReadVariableOp!lstm_cell_7/MatMul/ReadVariableOp2J
#lstm_cell_7/MatMul_1/ReadVariableOp#lstm_cell_7/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
§S
Ї
 forward_lstm_2_while_body_364471:
6forward_lstm_2_while_forward_lstm_2_while_loop_counter@
<forward_lstm_2_while_forward_lstm_2_while_maximum_iterations$
 forward_lstm_2_while_placeholder&
"forward_lstm_2_while_placeholder_1&
"forward_lstm_2_while_placeholder_2&
"forward_lstm_2_while_placeholder_39
5forward_lstm_2_while_forward_lstm_2_strided_slice_1_0u
qforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0T
Aforward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0:	ШV
Cforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0:	2ШQ
Bforward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0:	Ш!
forward_lstm_2_while_identity#
forward_lstm_2_while_identity_1#
forward_lstm_2_while_identity_2#
forward_lstm_2_while_identity_3#
forward_lstm_2_while_identity_4#
forward_lstm_2_while_identity_57
3forward_lstm_2_while_forward_lstm_2_strided_slice_1s
oforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensorR
?forward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource:	ШT
Aforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource:	2ШO
@forward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource:	ШЂ7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOpЂ6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOpЂ8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpс
Fforward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2H
Fforward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeЖ
8forward_lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0 forward_lstm_2_while_placeholderOforward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02:
8forward_lstm_2/while/TensorArrayV2Read/TensorListGetItemѓ
6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOpAforward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype028
6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp
'forward_lstm_2/while/lstm_cell_7/MatMulMatMul?forward_lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0>forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_2/while/lstm_cell_7/MatMulљ
8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOpCforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02:
8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpљ
)forward_lstm_2/while/lstm_cell_7/MatMul_1MatMul"forward_lstm_2_while_placeholder_2@forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)forward_lstm_2/while/lstm_cell_7/MatMul_1№
$forward_lstm_2/while/lstm_cell_7/addAddV21forward_lstm_2/while/lstm_cell_7/MatMul:product:03forward_lstm_2/while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$forward_lstm_2/while/lstm_cell_7/addђ
7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOpBforward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype029
7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp§
(forward_lstm_2/while/lstm_cell_7/BiasAddBiasAdd(forward_lstm_2/while/lstm_cell_7/add:z:0?forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_2/while/lstm_cell_7/BiasAddІ
0forward_lstm_2/while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0forward_lstm_2/while/lstm_cell_7/split/split_dimУ
&forward_lstm_2/while/lstm_cell_7/splitSplit9forward_lstm_2/while/lstm_cell_7/split/split_dim:output:01forward_lstm_2/while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2(
&forward_lstm_2/while/lstm_cell_7/splitТ
(forward_lstm_2/while/lstm_cell_7/SigmoidSigmoid/forward_lstm_2/while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_2/while/lstm_cell_7/SigmoidЦ
*forward_lstm_2/while/lstm_cell_7/Sigmoid_1Sigmoid/forward_lstm_2/while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_2/while/lstm_cell_7/Sigmoid_1й
$forward_lstm_2/while/lstm_cell_7/mulMul.forward_lstm_2/while/lstm_cell_7/Sigmoid_1:y:0"forward_lstm_2_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_2/while/lstm_cell_7/mulЙ
%forward_lstm_2/while/lstm_cell_7/ReluRelu/forward_lstm_2/while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_2/while/lstm_cell_7/Reluь
&forward_lstm_2/while/lstm_cell_7/mul_1Mul,forward_lstm_2/while/lstm_cell_7/Sigmoid:y:03forward_lstm_2/while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_2/while/lstm_cell_7/mul_1с
&forward_lstm_2/while/lstm_cell_7/add_1AddV2(forward_lstm_2/while/lstm_cell_7/mul:z:0*forward_lstm_2/while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_2/while/lstm_cell_7/add_1Ц
*forward_lstm_2/while/lstm_cell_7/Sigmoid_2Sigmoid/forward_lstm_2/while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_2/while/lstm_cell_7/Sigmoid_2И
'forward_lstm_2/while/lstm_cell_7/Relu_1Relu*forward_lstm_2/while/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_2/while/lstm_cell_7/Relu_1№
&forward_lstm_2/while/lstm_cell_7/mul_2Mul.forward_lstm_2/while/lstm_cell_7/Sigmoid_2:y:05forward_lstm_2/while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_2/while/lstm_cell_7/mul_2Њ
9forward_lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"forward_lstm_2_while_placeholder_1 forward_lstm_2_while_placeholder*forward_lstm_2/while/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype02;
9forward_lstm_2/while/TensorArrayV2Write/TensorListSetItemz
forward_lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_2/while/add/yЅ
forward_lstm_2/while/addAddV2 forward_lstm_2_while_placeholder#forward_lstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/while/add~
forward_lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_2/while/add_1/yС
forward_lstm_2/while/add_1AddV26forward_lstm_2_while_forward_lstm_2_while_loop_counter%forward_lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/while/add_1Ї
forward_lstm_2/while/IdentityIdentityforward_lstm_2/while/add_1:z:0^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2
forward_lstm_2/while/IdentityЩ
forward_lstm_2/while/Identity_1Identity<forward_lstm_2_while_forward_lstm_2_while_maximum_iterations^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_2/while/Identity_1Љ
forward_lstm_2/while/Identity_2Identityforward_lstm_2/while/add:z:0^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_2/while/Identity_2ж
forward_lstm_2/while/Identity_3IdentityIforward_lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_2/while/Identity_3Ш
forward_lstm_2/while/Identity_4Identity*forward_lstm_2/while/lstm_cell_7/mul_2:z:0^forward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/while/Identity_4Ш
forward_lstm_2/while/Identity_5Identity*forward_lstm_2/while/lstm_cell_7/add_1:z:0^forward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/while/Identity_5І
forward_lstm_2/while/NoOpNoOp8^forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp7^forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp9^forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_2/while/NoOp"l
3forward_lstm_2_while_forward_lstm_2_strided_slice_15forward_lstm_2_while_forward_lstm_2_strided_slice_1_0"G
forward_lstm_2_while_identity&forward_lstm_2/while/Identity:output:0"K
forward_lstm_2_while_identity_1(forward_lstm_2/while/Identity_1:output:0"K
forward_lstm_2_while_identity_2(forward_lstm_2/while/Identity_2:output:0"K
forward_lstm_2_while_identity_3(forward_lstm_2/while/Identity_3:output:0"K
forward_lstm_2_while_identity_4(forward_lstm_2/while/Identity_4:output:0"K
forward_lstm_2_while_identity_5(forward_lstm_2/while/Identity_5:output:0"
@forward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resourceBforward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0"
Aforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resourceCforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0"
?forward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resourceAforward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0"ф
oforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensorqforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2r
7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp2p
6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp2t
8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
е
У
while_cond_362316
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_362316___redundant_placeholder04
0while_while_cond_362316___redundant_placeholder14
0while_while_cond_362316___redundant_placeholder24
0while_while_cond_362316___redundant_placeholder3
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
Ы>
Ч
while_body_362477
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	ШG
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:	2ШB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	ШE
2while_lstm_cell_8_matmul_1_readvariableop_resource:	2Ш@
1while_lstm_cell_8_biasadd_readvariableop_resource:	ШЂ(while/lstm_cell_8/BiasAdd/ReadVariableOpЂ'while/lstm_cell_8/MatMul/ReadVariableOpЂ)while/lstm_cell_8/MatMul_1/ReadVariableOpУ
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
)while/TensorArrayV2Read/TensorListGetItemЦ
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOpд
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/MatMulЬ
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOpН
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/MatMul_1Д
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/addХ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOpС
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/BiasAdd
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_8/split
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Sigmoid
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Sigmoid_1
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/mul
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/ReluА
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/mul_1Ѕ
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/add_1
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Sigmoid_2
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Relu_1Д
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/mul_2п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5л

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
ф
Ї
>sequential_2_bidirectional_2_backward_lstm_2_while_cond_360871v
rsequential_2_bidirectional_2_backward_lstm_2_while_sequential_2_bidirectional_2_backward_lstm_2_while_loop_counter|
xsequential_2_bidirectional_2_backward_lstm_2_while_sequential_2_bidirectional_2_backward_lstm_2_while_maximum_iterationsB
>sequential_2_bidirectional_2_backward_lstm_2_while_placeholderD
@sequential_2_bidirectional_2_backward_lstm_2_while_placeholder_1D
@sequential_2_bidirectional_2_backward_lstm_2_while_placeholder_2D
@sequential_2_bidirectional_2_backward_lstm_2_while_placeholder_3D
@sequential_2_bidirectional_2_backward_lstm_2_while_placeholder_4x
tsequential_2_bidirectional_2_backward_lstm_2_while_less_sequential_2_bidirectional_2_backward_lstm_2_strided_slice_1
sequential_2_bidirectional_2_backward_lstm_2_while_sequential_2_bidirectional_2_backward_lstm_2_while_cond_360871___redundant_placeholder0
sequential_2_bidirectional_2_backward_lstm_2_while_sequential_2_bidirectional_2_backward_lstm_2_while_cond_360871___redundant_placeholder1
sequential_2_bidirectional_2_backward_lstm_2_while_sequential_2_bidirectional_2_backward_lstm_2_while_cond_360871___redundant_placeholder2
sequential_2_bidirectional_2_backward_lstm_2_while_sequential_2_bidirectional_2_backward_lstm_2_while_cond_360871___redundant_placeholder3
sequential_2_bidirectional_2_backward_lstm_2_while_sequential_2_bidirectional_2_backward_lstm_2_while_cond_360871___redundant_placeholder4?
;sequential_2_bidirectional_2_backward_lstm_2_while_identity
б
7sequential_2/bidirectional_2/backward_lstm_2/while/LessLess>sequential_2_bidirectional_2_backward_lstm_2_while_placeholdertsequential_2_bidirectional_2_backward_lstm_2_while_less_sequential_2_bidirectional_2_backward_lstm_2_strided_slice_1*
T0*
_output_shapes
: 29
7sequential_2/bidirectional_2/backward_lstm_2/while/Lessф
;sequential_2/bidirectional_2/backward_lstm_2/while/IdentityIdentity;sequential_2/bidirectional_2/backward_lstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2=
;sequential_2/bidirectional_2/backward_lstm_2/while/Identity"
;sequential_2_bidirectional_2_backward_lstm_2_while_identityDsequential_2/bidirectional_2/backward_lstm_2/while/Identity:output:0*(
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


!backward_lstm_2_while_cond_364966<
8backward_lstm_2_while_backward_lstm_2_while_loop_counterB
>backward_lstm_2_while_backward_lstm_2_while_maximum_iterations%
!backward_lstm_2_while_placeholder'
#backward_lstm_2_while_placeholder_1'
#backward_lstm_2_while_placeholder_2'
#backward_lstm_2_while_placeholder_3'
#backward_lstm_2_while_placeholder_4>
:backward_lstm_2_while_less_backward_lstm_2_strided_slice_1T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_364966___redundant_placeholder0T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_364966___redundant_placeholder1T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_364966___redundant_placeholder2T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_364966___redundant_placeholder3T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_364966___redundant_placeholder4"
backward_lstm_2_while_identity
Р
backward_lstm_2/while/LessLess!backward_lstm_2_while_placeholder:backward_lstm_2_while_less_backward_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_2/while/Less
backward_lstm_2/while/IdentityIdentitybackward_lstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2 
backward_lstm_2/while/Identity"I
backward_lstm_2_while_identity'backward_lstm_2/while/Identity:output:0*(
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
н]

K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_366287
inputs_0=
*lstm_cell_8_matmul_readvariableop_resource:	Ш?
,lstm_cell_8_matmul_1_readvariableop_resource:	2Ш:
+lstm_cell_8_biasadd_readvariableop_resource:	Ш
identityЂ"lstm_cell_8/BiasAdd/ReadVariableOpЂ!lstm_cell_8/MatMul/ReadVariableOpЂ#lstm_cell_8/MatMul_1/ReadVariableOpЂwhileF
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
strided_slice_2В
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOpЊ
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/MatMulИ
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOpІ
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/MatMul_1
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/addБ
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOpЉ
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/BiasAdd|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimя
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_8/split
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Sigmoid
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Sigmoid_1
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Relu
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/mul_1
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/add_1
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Relu_1
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
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
bodyR
while_body_366203*
condR
while_cond_366202*K
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

IdentityХ
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
З

!backward_lstm_2_while_cond_364317<
8backward_lstm_2_while_backward_lstm_2_while_loop_counterB
>backward_lstm_2_while_backward_lstm_2_while_maximum_iterations%
!backward_lstm_2_while_placeholder'
#backward_lstm_2_while_placeholder_1'
#backward_lstm_2_while_placeholder_2'
#backward_lstm_2_while_placeholder_3>
:backward_lstm_2_while_less_backward_lstm_2_strided_slice_1T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_364317___redundant_placeholder0T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_364317___redundant_placeholder1T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_364317___redundant_placeholder2T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_364317___redundant_placeholder3"
backward_lstm_2_while_identity
Р
backward_lstm_2/while/LessLess!backward_lstm_2_while_placeholder:backward_lstm_2_while_less_backward_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_2/while/Less
backward_lstm_2/while/IdentityIdentitybackward_lstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2 
backward_lstm_2/while/Identity"I
backward_lstm_2_while_identity'backward_lstm_2/while/Identity:output:0*(
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
е
У
while_cond_366508
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_366508___redundant_placeholder04
0while_while_cond_366508___redundant_placeholder14
0while_while_cond_366508___redundant_placeholder24
0while_while_cond_366508___redundant_placeholder3
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
њ

G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_366942

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
Т>
Ч
while_body_365553
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_7_matmul_readvariableop_resource_0:	ШG
4while_lstm_cell_7_matmul_1_readvariableop_resource_0:	2ШB
3while_lstm_cell_7_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_7_matmul_readvariableop_resource:	ШE
2while_lstm_cell_7_matmul_1_readvariableop_resource:	2Ш@
1while_lstm_cell_7_biasadd_readvariableop_resource:	ШЂ(while/lstm_cell_7/BiasAdd/ReadVariableOpЂ'while/lstm_cell_7/MatMul/ReadVariableOpЂ)while/lstm_cell_7/MatMul_1/ReadVariableOpУ
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
)while/TensorArrayV2Read/TensorListGetItemЦ
'while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02)
'while/lstm_cell_7/MatMul/ReadVariableOpд
while/lstm_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/MatMulЬ
)while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02+
)while/lstm_cell_7/MatMul_1/ReadVariableOpН
while/lstm_cell_7/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/MatMul_1Д
while/lstm_cell_7/addAddV2"while/lstm_cell_7/MatMul:product:0$while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/addХ
(while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02*
(while/lstm_cell_7/BiasAdd/ReadVariableOpС
while/lstm_cell_7/BiasAddBiasAddwhile/lstm_cell_7/add:z:00while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/BiasAdd
!while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_7/split/split_dim
while/lstm_cell_7/splitSplit*while/lstm_cell_7/split/split_dim:output:0"while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_7/split
while/lstm_cell_7/SigmoidSigmoid while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Sigmoid
while/lstm_cell_7/Sigmoid_1Sigmoid while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Sigmoid_1
while/lstm_cell_7/mulMulwhile/lstm_cell_7/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/mul
while/lstm_cell_7/ReluRelu while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/ReluА
while/lstm_cell_7/mul_1Mulwhile/lstm_cell_7/Sigmoid:y:0$while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/mul_1Ѕ
while/lstm_cell_7/add_1AddV2while/lstm_cell_7/mul:z:0while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/add_1
while/lstm_cell_7/Sigmoid_2Sigmoid while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Sigmoid_2
while/lstm_cell_7/Relu_1Reluwhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Relu_1Д
while/lstm_cell_7/mul_2Mulwhile/lstm_cell_7/Sigmoid_2:y:0&while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/mul_2п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_7/mul_2:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_7/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_7/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5л

while/NoOpNoOp)^while/lstm_cell_7/BiasAdd/ReadVariableOp(^while/lstm_cell_7/MatMul/ReadVariableOp*^while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_7_biasadd_readvariableop_resource3while_lstm_cell_7_biasadd_readvariableop_resource_0"j
2while_lstm_cell_7_matmul_1_readvariableop_resource4while_lstm_cell_7_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_7_matmul_readvariableop_resource2while_lstm_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2T
(while/lstm_cell_7/BiasAdd/ReadVariableOp(while/lstm_cell_7/BiasAdd/ReadVariableOp2R
'while/lstm_cell_7/MatMul/ReadVariableOp'while/lstm_cell_7/MatMul/ReadVariableOp2V
)while/lstm_cell_7/MatMul_1/ReadVariableOp)while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
рЕ
Є
"__inference__traced_restore_367210
file_prefix1
assignvariableop_dense_2_kernel:d-
assignvariableop_1_dense_2_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: W
Dassignvariableop_7_bidirectional_2_forward_lstm_2_lstm_cell_7_kernel:	Шa
Nassignvariableop_8_bidirectional_2_forward_lstm_2_lstm_cell_7_recurrent_kernel:	2ШQ
Bassignvariableop_9_bidirectional_2_forward_lstm_2_lstm_cell_7_bias:	ШY
Fassignvariableop_10_bidirectional_2_backward_lstm_2_lstm_cell_8_kernel:	Шc
Passignvariableop_11_bidirectional_2_backward_lstm_2_lstm_cell_8_recurrent_kernel:	2ШS
Dassignvariableop_12_bidirectional_2_backward_lstm_2_lstm_cell_8_bias:	Ш#
assignvariableop_13_total: #
assignvariableop_14_count: ;
)assignvariableop_15_adam_dense_2_kernel_m:d5
'assignvariableop_16_adam_dense_2_bias_m:_
Lassignvariableop_17_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_kernel_m:	Шi
Vassignvariableop_18_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_recurrent_kernel_m:	2ШY
Jassignvariableop_19_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_bias_m:	Ш`
Massignvariableop_20_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_kernel_m:	Шj
Wassignvariableop_21_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_recurrent_kernel_m:	2ШZ
Kassignvariableop_22_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_bias_m:	Ш;
)assignvariableop_23_adam_dense_2_kernel_v:d5
'assignvariableop_24_adam_dense_2_bias_v:_
Lassignvariableop_25_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_kernel_v:	Шi
Vassignvariableop_26_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_recurrent_kernel_v:	2ШY
Jassignvariableop_27_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_bias_v:	Ш`
Massignvariableop_28_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_kernel_v:	Шj
Wassignvariableop_29_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_recurrent_kernel_v:	2ШZ
Kassignvariableop_30_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_bias_v:	Ш>
,assignvariableop_31_adam_dense_2_kernel_vhat:d8
*assignvariableop_32_adam_dense_2_bias_vhat:b
Oassignvariableop_33_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_kernel_vhat:	Шl
Yassignvariableop_34_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_recurrent_kernel_vhat:	2Ш\
Massignvariableop_35_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_bias_vhat:	Шc
Passignvariableop_36_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_kernel_vhat:	Шm
Zassignvariableop_37_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_recurrent_kernel_vhat:	2Ш]
Nassignvariableop_38_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_bias_vhat:	Ш
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

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Є
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
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

Identity_7Щ
AssignVariableOp_7AssignVariableOpDassignvariableop_7_bidirectional_2_forward_lstm_2_lstm_cell_7_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8г
AssignVariableOp_8AssignVariableOpNassignvariableop_8_bidirectional_2_forward_lstm_2_lstm_cell_7_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ч
AssignVariableOp_9AssignVariableOpBassignvariableop_9_bidirectional_2_forward_lstm_2_lstm_cell_7_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ю
AssignVariableOp_10AssignVariableOpFassignvariableop_10_bidirectional_2_backward_lstm_2_lstm_cell_8_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11и
AssignVariableOp_11AssignVariableOpPassignvariableop_11_bidirectional_2_backward_lstm_2_lstm_cell_8_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ь
AssignVariableOp_12AssignVariableOpDassignvariableop_12_bidirectional_2_backward_lstm_2_lstm_cell_8_biasIdentity_12:output:0"/device:CPU:0*
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
Identity_15Б
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_2_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Џ
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_2_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17д
AssignVariableOp_17AssignVariableOpLassignvariableop_17_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18о
AssignVariableOp_18AssignVariableOpVassignvariableop_18_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19в
AssignVariableOp_19AssignVariableOpJassignvariableop_19_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20е
AssignVariableOp_20AssignVariableOpMassignvariableop_20_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21п
AssignVariableOp_21AssignVariableOpWassignvariableop_21_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22г
AssignVariableOp_22AssignVariableOpKassignvariableop_22_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Б
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_2_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Џ
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_2_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25д
AssignVariableOp_25AssignVariableOpLassignvariableop_25_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26о
AssignVariableOp_26AssignVariableOpVassignvariableop_26_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27в
AssignVariableOp_27AssignVariableOpJassignvariableop_27_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28е
AssignVariableOp_28AssignVariableOpMassignvariableop_28_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29п
AssignVariableOp_29AssignVariableOpWassignvariableop_29_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30г
AssignVariableOp_30AssignVariableOpKassignvariableop_30_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Д
AssignVariableOp_31AssignVariableOp,assignvariableop_31_adam_dense_2_kernel_vhatIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32В
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_dense_2_bias_vhatIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33з
AssignVariableOp_33AssignVariableOpOassignvariableop_33_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_kernel_vhatIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34с
AssignVariableOp_34AssignVariableOpYassignvariableop_34_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_recurrent_kernel_vhatIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35е
AssignVariableOp_35AssignVariableOpMassignvariableop_35_adam_bidirectional_2_forward_lstm_2_lstm_cell_7_bias_vhatIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36и
AssignVariableOp_36AssignVariableOpPassignvariableop_36_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_kernel_vhatIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37т
AssignVariableOp_37AssignVariableOpZassignvariableop_37_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_recurrent_kernel_vhatIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38ж
AssignVariableOp_38AssignVariableOpNassignvariableop_38_adam_bidirectional_2_backward_lstm_2_lstm_cell_8_bias_vhatIdentity_38:output:0"/device:CPU:0*
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
F

J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_361344

inputs%
lstm_cell_7_361262:	Ш%
lstm_cell_7_361264:	2Ш!
lstm_cell_7_361266:	Ш
identityЂ#lstm_cell_7/StatefulPartitionedCallЂwhileD
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
strided_slice_2
#lstm_cell_7/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_7_361262lstm_cell_7_361264lstm_cell_7_361266*
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
GPU 2J 8 *P
fKRI
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_3611972%
#lstm_cell_7/StatefulPartitionedCall
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
while/loop_counterН
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_7_361262lstm_cell_7_361264lstm_cell_7_361266*
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
bodyR
while_body_361275*
condR
while_cond_361274*K
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

Identity|
NoOpNoOp$^lstm_cell_7/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2J
#lstm_cell_7/StatefulPartitionedCall#lstm_cell_7/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ц

Ы
$__inference_signature_wrapper_364032

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
identityЂStatefulPartitionedCallЊ
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
GPU 2J 8 **
f%R#
!__inference__wrapped_model_3609762
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
ЊБ
З
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_365422

inputs
inputs_1	L
9forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource:	ШN
;forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource:	2ШI
:forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource:	ШM
:backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource:	ШO
<backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource:	2ШJ
;backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource:	Ш
identityЂ2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpЂ1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpЂ3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpЂbackward_lstm_2/whileЂ1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpЂ0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpЂ2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpЂforward_lstm_2/while
#forward_lstm_2/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2%
#forward_lstm_2/RaggedToTensor/zeros
#forward_lstm_2/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2%
#forward_lstm_2/RaggedToTensor/Const
2forward_lstm_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor,forward_lstm_2/RaggedToTensor/Const:output:0inputs,forward_lstm_2/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS24
2forward_lstm_2/RaggedToTensor/RaggedTensorToTensorР
9forward_lstm_2/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9forward_lstm_2/RaggedNestedRowLengths/strided_slice/stackФ
;forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1Ф
;forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2
3forward_lstm_2/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Bforward_lstm_2/RaggedNestedRowLengths/strided_slice/stack:output:0Dforward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1:output:0Dforward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask25
3forward_lstm_2/RaggedNestedRowLengths/strided_sliceФ
;forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stackб
=forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2?
=forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1Ш
=forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2Ћ
5forward_lstm_2/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Dforward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack:output:0Fforward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Fforward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask27
5forward_lstm_2/RaggedNestedRowLengths/strided_slice_1
)forward_lstm_2/RaggedNestedRowLengths/subSub<forward_lstm_2/RaggedNestedRowLengths/strided_slice:output:0>forward_lstm_2/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2+
)forward_lstm_2/RaggedNestedRowLengths/sub
forward_lstm_2/CastCast-forward_lstm_2/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_2/Cast
forward_lstm_2/ShapeShape;forward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_2/Shape
"forward_lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"forward_lstm_2/strided_slice/stack
$forward_lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_2/strided_slice/stack_1
$forward_lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_2/strided_slice/stack_2М
forward_lstm_2/strided_sliceStridedSliceforward_lstm_2/Shape:output:0+forward_lstm_2/strided_slice/stack:output:0-forward_lstm_2/strided_slice/stack_1:output:0-forward_lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_2/strided_slicez
forward_lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_2/zeros/mul/yЈ
forward_lstm_2/zeros/mulMul%forward_lstm_2/strided_slice:output:0#forward_lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros/mul}
forward_lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_2/zeros/Less/yЃ
forward_lstm_2/zeros/LessLessforward_lstm_2/zeros/mul:z:0$forward_lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros/Less
forward_lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_2/zeros/packed/1П
forward_lstm_2/zeros/packedPack%forward_lstm_2/strided_slice:output:0&forward_lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_2/zeros/packed
forward_lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_2/zeros/ConstБ
forward_lstm_2/zerosFill$forward_lstm_2/zeros/packed:output:0#forward_lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/zeros~
forward_lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_2/zeros_1/mul/yЎ
forward_lstm_2/zeros_1/mulMul%forward_lstm_2/strided_slice:output:0%forward_lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros_1/mul
forward_lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_2/zeros_1/Less/yЋ
forward_lstm_2/zeros_1/LessLessforward_lstm_2/zeros_1/mul:z:0&forward_lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros_1/Less
forward_lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_2/zeros_1/packed/1Х
forward_lstm_2/zeros_1/packedPack%forward_lstm_2/strided_slice:output:0(forward_lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_2/zeros_1/packed
forward_lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_2/zeros_1/ConstЙ
forward_lstm_2/zeros_1Fill&forward_lstm_2/zeros_1/packed:output:0%forward_lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/zeros_1
forward_lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
forward_lstm_2/transpose/permх
forward_lstm_2/transpose	Transpose;forward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0&forward_lstm_2/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_2/transpose|
forward_lstm_2/Shape_1Shapeforward_lstm_2/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_2/Shape_1
$forward_lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_2/strided_slice_1/stack
&forward_lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_1/stack_1
&forward_lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_1/stack_2Ш
forward_lstm_2/strided_slice_1StridedSliceforward_lstm_2/Shape_1:output:0-forward_lstm_2/strided_slice_1/stack:output:0/forward_lstm_2/strided_slice_1/stack_1:output:0/forward_lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_2/strided_slice_1Ѓ
*forward_lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*forward_lstm_2/TensorArrayV2/element_shapeю
forward_lstm_2/TensorArrayV2TensorListReserve3forward_lstm_2/TensorArrayV2/element_shape:output:0'forward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_2/TensorArrayV2н
Dforward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2F
Dforward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeД
6forward_lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_2/transpose:y:0Mforward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6forward_lstm_2/TensorArrayUnstack/TensorListFromTensor
$forward_lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_2/strided_slice_2/stack
&forward_lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_2/stack_1
&forward_lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_2/stack_2ж
forward_lstm_2/strided_slice_2StridedSliceforward_lstm_2/transpose:y:0-forward_lstm_2/strided_slice_2/stack:output:0/forward_lstm_2/strided_slice_2/stack_1:output:0/forward_lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2 
forward_lstm_2/strided_slice_2п
0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp9forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype022
0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpц
!forward_lstm_2/lstm_cell_7/MatMulMatMul'forward_lstm_2/strided_slice_2:output:08forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2#
!forward_lstm_2/lstm_cell_7/MatMulх
2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp;forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype024
2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpт
#forward_lstm_2/lstm_cell_7/MatMul_1MatMulforward_lstm_2/zeros:output:0:forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#forward_lstm_2/lstm_cell_7/MatMul_1и
forward_lstm_2/lstm_cell_7/addAddV2+forward_lstm_2/lstm_cell_7/MatMul:product:0-forward_lstm_2/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2 
forward_lstm_2/lstm_cell_7/addо
1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp:forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype023
1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpх
"forward_lstm_2/lstm_cell_7/BiasAddBiasAdd"forward_lstm_2/lstm_cell_7/add:z:09forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_2/lstm_cell_7/BiasAdd
*forward_lstm_2/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*forward_lstm_2/lstm_cell_7/split/split_dimЋ
 forward_lstm_2/lstm_cell_7/splitSplit3forward_lstm_2/lstm_cell_7/split/split_dim:output:0+forward_lstm_2/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2"
 forward_lstm_2/lstm_cell_7/splitА
"forward_lstm_2/lstm_cell_7/SigmoidSigmoid)forward_lstm_2/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_2/lstm_cell_7/SigmoidД
$forward_lstm_2/lstm_cell_7/Sigmoid_1Sigmoid)forward_lstm_2/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_2/lstm_cell_7/Sigmoid_1Ф
forward_lstm_2/lstm_cell_7/mulMul(forward_lstm_2/lstm_cell_7/Sigmoid_1:y:0forward_lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
forward_lstm_2/lstm_cell_7/mulЇ
forward_lstm_2/lstm_cell_7/ReluRelu)forward_lstm_2/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/lstm_cell_7/Reluд
 forward_lstm_2/lstm_cell_7/mul_1Mul&forward_lstm_2/lstm_cell_7/Sigmoid:y:0-forward_lstm_2/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_2/lstm_cell_7/mul_1Щ
 forward_lstm_2/lstm_cell_7/add_1AddV2"forward_lstm_2/lstm_cell_7/mul:z:0$forward_lstm_2/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_2/lstm_cell_7/add_1Д
$forward_lstm_2/lstm_cell_7/Sigmoid_2Sigmoid)forward_lstm_2/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_2/lstm_cell_7/Sigmoid_2І
!forward_lstm_2/lstm_cell_7/Relu_1Relu$forward_lstm_2/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_2/lstm_cell_7/Relu_1и
 forward_lstm_2/lstm_cell_7/mul_2Mul(forward_lstm_2/lstm_cell_7/Sigmoid_2:y:0/forward_lstm_2/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_2/lstm_cell_7/mul_2­
,forward_lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2.
,forward_lstm_2/TensorArrayV2_1/element_shapeє
forward_lstm_2/TensorArrayV2_1TensorListReserve5forward_lstm_2/TensorArrayV2_1/element_shape:output:0'forward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_2/TensorArrayV2_1l
forward_lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_2/time
forward_lstm_2/zeros_like	ZerosLike$forward_lstm_2/lstm_cell_7/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/zeros_like
'forward_lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'forward_lstm_2/while/maximum_iterations
!forward_lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!forward_lstm_2/while/loop_counterч
forward_lstm_2/whileWhile*forward_lstm_2/while/loop_counter:output:00forward_lstm_2/while/maximum_iterations:output:0forward_lstm_2/time:output:0'forward_lstm_2/TensorArrayV2_1:handle:0forward_lstm_2/zeros_like:y:0forward_lstm_2/zeros:output:0forward_lstm_2/zeros_1:output:0'forward_lstm_2/strided_slice_1:output:0Fforward_lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_2/Cast:y:09forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource;forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource:forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *,
body$R"
 forward_lstm_2_while_body_365146*,
cond$R"
 forward_lstm_2_while_cond_365145*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_2/whileг
?forward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2A
?forward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape­
1forward_lstm_2/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_2/while:output:3Hforward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype023
1forward_lstm_2/TensorArrayV2Stack/TensorListStack
$forward_lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2&
$forward_lstm_2/strided_slice_3/stack
&forward_lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_2/strided_slice_3/stack_1
&forward_lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_3/stack_2є
forward_lstm_2/strided_slice_3StridedSlice:forward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0-forward_lstm_2/strided_slice_3/stack:output:0/forward_lstm_2/strided_slice_3/stack_1:output:0/forward_lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2 
forward_lstm_2/strided_slice_3
forward_lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_2/transpose_1/permъ
forward_lstm_2/transpose_1	Transpose:forward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0(forward_lstm_2/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_2/transpose_1
forward_lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_2/runtime
$backward_lstm_2/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$backward_lstm_2/RaggedToTensor/zeros
$backward_lstm_2/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2&
$backward_lstm_2/RaggedToTensor/Const
3backward_lstm_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-backward_lstm_2/RaggedToTensor/Const:output:0inputs-backward_lstm_2/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3backward_lstm_2/RaggedToTensor/RaggedTensorToTensorТ
:backward_lstm_2/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:backward_lstm_2/RaggedNestedRowLengths/strided_slice/stackЦ
<backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1Ц
<backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2Є
4backward_lstm_2/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cbackward_lstm_2/RaggedNestedRowLengths/strided_slice/stack:output:0Ebackward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1:output:0Ebackward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask26
4backward_lstm_2/RaggedNestedRowLengths/strided_sliceЦ
<backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stackг
>backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2@
>backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1Ъ
>backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2А
6backward_lstm_2/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Ebackward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gbackward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gbackward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask28
6backward_lstm_2/RaggedNestedRowLengths/strided_slice_1
*backward_lstm_2/RaggedNestedRowLengths/subSub=backward_lstm_2/RaggedNestedRowLengths/strided_slice:output:0?backward_lstm_2/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2,
*backward_lstm_2/RaggedNestedRowLengths/subЁ
backward_lstm_2/CastCast.backward_lstm_2/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_2/Cast
backward_lstm_2/ShapeShape<backward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_2/Shape
#backward_lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#backward_lstm_2/strided_slice/stack
%backward_lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_2/strided_slice/stack_1
%backward_lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_2/strided_slice/stack_2Т
backward_lstm_2/strided_sliceStridedSlicebackward_lstm_2/Shape:output:0,backward_lstm_2/strided_slice/stack:output:0.backward_lstm_2/strided_slice/stack_1:output:0.backward_lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
backward_lstm_2/strided_slice|
backward_lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_2/zeros/mul/yЌ
backward_lstm_2/zeros/mulMul&backward_lstm_2/strided_slice:output:0$backward_lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros/mul
backward_lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
backward_lstm_2/zeros/Less/yЇ
backward_lstm_2/zeros/LessLessbackward_lstm_2/zeros/mul:z:0%backward_lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros/Less
backward_lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_2/zeros/packed/1У
backward_lstm_2/zeros/packedPack&backward_lstm_2/strided_slice:output:0'backward_lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_2/zeros/packed
backward_lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_2/zeros/ConstЕ
backward_lstm_2/zerosFill%backward_lstm_2/zeros/packed:output:0$backward_lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/zeros
backward_lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_2/zeros_1/mul/yВ
backward_lstm_2/zeros_1/mulMul&backward_lstm_2/strided_slice:output:0&backward_lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros_1/mul
backward_lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_2/zeros_1/Less/yЏ
backward_lstm_2/zeros_1/LessLessbackward_lstm_2/zeros_1/mul:z:0'backward_lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros_1/Less
 backward_lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_2/zeros_1/packed/1Щ
backward_lstm_2/zeros_1/packedPack&backward_lstm_2/strided_slice:output:0)backward_lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_2/zeros_1/packed
backward_lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_2/zeros_1/ConstН
backward_lstm_2/zeros_1Fill'backward_lstm_2/zeros_1/packed:output:0&backward_lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/zeros_1
backward_lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
backward_lstm_2/transpose/permщ
backward_lstm_2/transpose	Transpose<backward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0'backward_lstm_2/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_2/transpose
backward_lstm_2/Shape_1Shapebackward_lstm_2/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_2/Shape_1
%backward_lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_2/strided_slice_1/stack
'backward_lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_1/stack_1
'backward_lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_1/stack_2Ю
backward_lstm_2/strided_slice_1StridedSlice backward_lstm_2/Shape_1:output:0.backward_lstm_2/strided_slice_1/stack:output:00backward_lstm_2/strided_slice_1/stack_1:output:00backward_lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_2/strided_slice_1Ѕ
+backward_lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+backward_lstm_2/TensorArrayV2/element_shapeђ
backward_lstm_2/TensorArrayV2TensorListReserve4backward_lstm_2/TensorArrayV2/element_shape:output:0(backward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
backward_lstm_2/TensorArrayV2
backward_lstm_2/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2 
backward_lstm_2/ReverseV2/axisЪ
backward_lstm_2/ReverseV2	ReverseV2backward_lstm_2/transpose:y:0'backward_lstm_2/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_2/ReverseV2п
Ebackward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2G
Ebackward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeН
7backward_lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"backward_lstm_2/ReverseV2:output:0Nbackward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7backward_lstm_2/TensorArrayUnstack/TensorListFromTensor
%backward_lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_2/strided_slice_2/stack
'backward_lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_2/stack_1
'backward_lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_2/stack_2м
backward_lstm_2/strided_slice_2StridedSlicebackward_lstm_2/transpose:y:0.backward_lstm_2/strided_slice_2/stack:output:00backward_lstm_2/strided_slice_2/stack_1:output:00backward_lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2!
backward_lstm_2/strided_slice_2т
1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp:backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype023
1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpъ
"backward_lstm_2/lstm_cell_8/MatMulMatMul(backward_lstm_2/strided_slice_2:output:09backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"backward_lstm_2/lstm_cell_8/MatMulш
3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp<backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype025
3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpц
$backward_lstm_2/lstm_cell_8/MatMul_1MatMulbackward_lstm_2/zeros:output:0;backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$backward_lstm_2/lstm_cell_8/MatMul_1м
backward_lstm_2/lstm_cell_8/addAddV2,backward_lstm_2/lstm_cell_8/MatMul:product:0.backward_lstm_2/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2!
backward_lstm_2/lstm_cell_8/addс
2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp;backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype024
2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpщ
#backward_lstm_2/lstm_cell_8/BiasAddBiasAdd#backward_lstm_2/lstm_cell_8/add:z:0:backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_2/lstm_cell_8/BiasAdd
+backward_lstm_2/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+backward_lstm_2/lstm_cell_8/split/split_dimЏ
!backward_lstm_2/lstm_cell_8/splitSplit4backward_lstm_2/lstm_cell_8/split/split_dim:output:0,backward_lstm_2/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2#
!backward_lstm_2/lstm_cell_8/splitГ
#backward_lstm_2/lstm_cell_8/SigmoidSigmoid*backward_lstm_2/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_2/lstm_cell_8/SigmoidЗ
%backward_lstm_2/lstm_cell_8/Sigmoid_1Sigmoid*backward_lstm_2/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_2/lstm_cell_8/Sigmoid_1Ш
backward_lstm_2/lstm_cell_8/mulMul)backward_lstm_2/lstm_cell_8/Sigmoid_1:y:0 backward_lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22!
backward_lstm_2/lstm_cell_8/mulЊ
 backward_lstm_2/lstm_cell_8/ReluRelu*backward_lstm_2/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/lstm_cell_8/Reluи
!backward_lstm_2/lstm_cell_8/mul_1Mul'backward_lstm_2/lstm_cell_8/Sigmoid:y:0.backward_lstm_2/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_2/lstm_cell_8/mul_1Э
!backward_lstm_2/lstm_cell_8/add_1AddV2#backward_lstm_2/lstm_cell_8/mul:z:0%backward_lstm_2/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_2/lstm_cell_8/add_1З
%backward_lstm_2/lstm_cell_8/Sigmoid_2Sigmoid*backward_lstm_2/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_2/lstm_cell_8/Sigmoid_2Љ
"backward_lstm_2/lstm_cell_8/Relu_1Relu%backward_lstm_2/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_2/lstm_cell_8/Relu_1м
!backward_lstm_2/lstm_cell_8/mul_2Mul)backward_lstm_2/lstm_cell_8/Sigmoid_2:y:00backward_lstm_2/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_2/lstm_cell_8/mul_2Џ
-backward_lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2/
-backward_lstm_2/TensorArrayV2_1/element_shapeј
backward_lstm_2/TensorArrayV2_1TensorListReserve6backward_lstm_2/TensorArrayV2_1/element_shape:output:0(backward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_2/TensorArrayV2_1n
backward_lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_2/time
%backward_lstm_2/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2'
%backward_lstm_2/Max/reduction_indices
backward_lstm_2/MaxMaxbackward_lstm_2/Cast:y:0.backward_lstm_2/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/Maxp
backward_lstm_2/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_2/sub/y
backward_lstm_2/subSubbackward_lstm_2/Max:output:0backward_lstm_2/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/sub
backward_lstm_2/Sub_1Subbackward_lstm_2/sub:z:0backward_lstm_2/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_2/Sub_1
backward_lstm_2/zeros_like	ZerosLike%backward_lstm_2/lstm_cell_8/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/zeros_like
(backward_lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(backward_lstm_2/while/maximum_iterations
"backward_lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"backward_lstm_2/while/loop_counterљ
backward_lstm_2/whileWhile+backward_lstm_2/while/loop_counter:output:01backward_lstm_2/while/maximum_iterations:output:0backward_lstm_2/time:output:0(backward_lstm_2/TensorArrayV2_1:handle:0backward_lstm_2/zeros_like:y:0backward_lstm_2/zeros:output:0 backward_lstm_2/zeros_1:output:0(backward_lstm_2/strided_slice_1:output:0Gbackward_lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_2/Sub_1:z:0:backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource<backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource;backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *-
body%R#
!backward_lstm_2_while_body_365325*-
cond%R#
!backward_lstm_2_while_cond_365324*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_2/whileе
@backward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2B
@backward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeБ
2backward_lstm_2/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_2/while:output:3Ibackward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype024
2backward_lstm_2/TensorArrayV2Stack/TensorListStackЁ
%backward_lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2'
%backward_lstm_2/strided_slice_3/stack
'backward_lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_2/strided_slice_3/stack_1
'backward_lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_3/stack_2њ
backward_lstm_2/strided_slice_3StridedSlice;backward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0.backward_lstm_2/strided_slice_3/stack:output:00backward_lstm_2/strided_slice_3/stack_1:output:00backward_lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2!
backward_lstm_2/strided_slice_3
 backward_lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_2/transpose_1/permю
backward_lstm_2/transpose_1	Transpose;backward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0)backward_lstm_2/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_2/transpose_1
backward_lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_2/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisР
concatConcatV2'forward_lstm_2/strided_slice_3:output:0(backward_lstm_2/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

IdentityИ
NoOpNoOp3^backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp2^backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp4^backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp^backward_lstm_2/while2^forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp1^forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp3^forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp^forward_lstm_2/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2h
2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp2f
1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp2j
3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp2.
backward_lstm_2/whilebackward_lstm_2/while2f
1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp2d
0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp2h
2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp2,
forward_lstm_2/whileforward_lstm_2/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ь
О
/__inference_forward_lstm_2_layer_call_fn_365464
inputs_0
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
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
GPU 2J 8 *S
fNRL
J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_3613442
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
е
У
while_cond_361696
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_361696___redundant_placeholder04
0while_while_cond_361696___redundant_placeholder14
0while_while_cond_361696___redundant_placeholder24
0while_while_cond_361696___redundant_placeholder3
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
Т>
Ч
while_body_366356
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	ШG
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:	2ШB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	ШE
2while_lstm_cell_8_matmul_1_readvariableop_resource:	2Ш@
1while_lstm_cell_8_biasadd_readvariableop_resource:	ШЂ(while/lstm_cell_8/BiasAdd/ReadVariableOpЂ'while/lstm_cell_8/MatMul/ReadVariableOpЂ)while/lstm_cell_8/MatMul_1/ReadVariableOpУ
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
)while/TensorArrayV2Read/TensorListGetItemЦ
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOpд
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/MatMulЬ
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOpН
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/MatMul_1Д
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/addХ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOpС
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/BiasAdd
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_8/split
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Sigmoid
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Sigmoid_1
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/mul
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/ReluА
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/mul_1Ѕ
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/add_1
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Sigmoid_2
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Relu_1Д
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/mul_2п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5л

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Ўѓ
Ћ
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_364706
inputs_0L
9forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource:	ШN
;forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource:	2ШI
:forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource:	ШM
:backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource:	ШO
<backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource:	2ШJ
;backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource:	Ш
identityЂ2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpЂ1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpЂ3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpЂbackward_lstm_2/whileЂ1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpЂ0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpЂ2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpЂforward_lstm_2/whiled
forward_lstm_2/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_2/Shape
"forward_lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"forward_lstm_2/strided_slice/stack
$forward_lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_2/strided_slice/stack_1
$forward_lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_2/strided_slice/stack_2М
forward_lstm_2/strided_sliceStridedSliceforward_lstm_2/Shape:output:0+forward_lstm_2/strided_slice/stack:output:0-forward_lstm_2/strided_slice/stack_1:output:0-forward_lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_2/strided_slicez
forward_lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_2/zeros/mul/yЈ
forward_lstm_2/zeros/mulMul%forward_lstm_2/strided_slice:output:0#forward_lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros/mul}
forward_lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_2/zeros/Less/yЃ
forward_lstm_2/zeros/LessLessforward_lstm_2/zeros/mul:z:0$forward_lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros/Less
forward_lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_2/zeros/packed/1П
forward_lstm_2/zeros/packedPack%forward_lstm_2/strided_slice:output:0&forward_lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_2/zeros/packed
forward_lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_2/zeros/ConstБ
forward_lstm_2/zerosFill$forward_lstm_2/zeros/packed:output:0#forward_lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/zeros~
forward_lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_2/zeros_1/mul/yЎ
forward_lstm_2/zeros_1/mulMul%forward_lstm_2/strided_slice:output:0%forward_lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros_1/mul
forward_lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_2/zeros_1/Less/yЋ
forward_lstm_2/zeros_1/LessLessforward_lstm_2/zeros_1/mul:z:0&forward_lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros_1/Less
forward_lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_2/zeros_1/packed/1Х
forward_lstm_2/zeros_1/packedPack%forward_lstm_2/strided_slice:output:0(forward_lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_2/zeros_1/packed
forward_lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_2/zeros_1/ConstЙ
forward_lstm_2/zeros_1Fill&forward_lstm_2/zeros_1/packed:output:0%forward_lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/zeros_1
forward_lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
forward_lstm_2/transpose/permЛ
forward_lstm_2/transpose	Transposeinputs_0&forward_lstm_2/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
forward_lstm_2/transpose|
forward_lstm_2/Shape_1Shapeforward_lstm_2/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_2/Shape_1
$forward_lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_2/strided_slice_1/stack
&forward_lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_1/stack_1
&forward_lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_1/stack_2Ш
forward_lstm_2/strided_slice_1StridedSliceforward_lstm_2/Shape_1:output:0-forward_lstm_2/strided_slice_1/stack:output:0/forward_lstm_2/strided_slice_1/stack_1:output:0/forward_lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_2/strided_slice_1Ѓ
*forward_lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*forward_lstm_2/TensorArrayV2/element_shapeю
forward_lstm_2/TensorArrayV2TensorListReserve3forward_lstm_2/TensorArrayV2/element_shape:output:0'forward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_2/TensorArrayV2н
Dforward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2F
Dforward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeД
6forward_lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_2/transpose:y:0Mforward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6forward_lstm_2/TensorArrayUnstack/TensorListFromTensor
$forward_lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_2/strided_slice_2/stack
&forward_lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_2/stack_1
&forward_lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_2/stack_2п
forward_lstm_2/strided_slice_2StridedSliceforward_lstm_2/transpose:y:0-forward_lstm_2/strided_slice_2/stack:output:0/forward_lstm_2/strided_slice_2/stack_1:output:0/forward_lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2 
forward_lstm_2/strided_slice_2п
0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp9forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype022
0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpц
!forward_lstm_2/lstm_cell_7/MatMulMatMul'forward_lstm_2/strided_slice_2:output:08forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2#
!forward_lstm_2/lstm_cell_7/MatMulх
2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp;forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype024
2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpт
#forward_lstm_2/lstm_cell_7/MatMul_1MatMulforward_lstm_2/zeros:output:0:forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#forward_lstm_2/lstm_cell_7/MatMul_1и
forward_lstm_2/lstm_cell_7/addAddV2+forward_lstm_2/lstm_cell_7/MatMul:product:0-forward_lstm_2/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2 
forward_lstm_2/lstm_cell_7/addо
1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp:forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype023
1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpх
"forward_lstm_2/lstm_cell_7/BiasAddBiasAdd"forward_lstm_2/lstm_cell_7/add:z:09forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_2/lstm_cell_7/BiasAdd
*forward_lstm_2/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*forward_lstm_2/lstm_cell_7/split/split_dimЋ
 forward_lstm_2/lstm_cell_7/splitSplit3forward_lstm_2/lstm_cell_7/split/split_dim:output:0+forward_lstm_2/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2"
 forward_lstm_2/lstm_cell_7/splitА
"forward_lstm_2/lstm_cell_7/SigmoidSigmoid)forward_lstm_2/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_2/lstm_cell_7/SigmoidД
$forward_lstm_2/lstm_cell_7/Sigmoid_1Sigmoid)forward_lstm_2/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_2/lstm_cell_7/Sigmoid_1Ф
forward_lstm_2/lstm_cell_7/mulMul(forward_lstm_2/lstm_cell_7/Sigmoid_1:y:0forward_lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
forward_lstm_2/lstm_cell_7/mulЇ
forward_lstm_2/lstm_cell_7/ReluRelu)forward_lstm_2/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/lstm_cell_7/Reluд
 forward_lstm_2/lstm_cell_7/mul_1Mul&forward_lstm_2/lstm_cell_7/Sigmoid:y:0-forward_lstm_2/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_2/lstm_cell_7/mul_1Щ
 forward_lstm_2/lstm_cell_7/add_1AddV2"forward_lstm_2/lstm_cell_7/mul:z:0$forward_lstm_2/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_2/lstm_cell_7/add_1Д
$forward_lstm_2/lstm_cell_7/Sigmoid_2Sigmoid)forward_lstm_2/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_2/lstm_cell_7/Sigmoid_2І
!forward_lstm_2/lstm_cell_7/Relu_1Relu$forward_lstm_2/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_2/lstm_cell_7/Relu_1и
 forward_lstm_2/lstm_cell_7/mul_2Mul(forward_lstm_2/lstm_cell_7/Sigmoid_2:y:0/forward_lstm_2/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_2/lstm_cell_7/mul_2­
,forward_lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2.
,forward_lstm_2/TensorArrayV2_1/element_shapeє
forward_lstm_2/TensorArrayV2_1TensorListReserve5forward_lstm_2/TensorArrayV2_1/element_shape:output:0'forward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_2/TensorArrayV2_1l
forward_lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_2/time
'forward_lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'forward_lstm_2/while/maximum_iterations
!forward_lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!forward_lstm_2/while/loop_counterщ
forward_lstm_2/whileWhile*forward_lstm_2/while/loop_counter:output:00forward_lstm_2/while/maximum_iterations:output:0forward_lstm_2/time:output:0'forward_lstm_2/TensorArrayV2_1:handle:0forward_lstm_2/zeros:output:0forward_lstm_2/zeros_1:output:0'forward_lstm_2/strided_slice_1:output:0Fforward_lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:09forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource;forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource:forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *,
body$R"
 forward_lstm_2_while_body_364471*,
cond$R"
 forward_lstm_2_while_cond_364470*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
forward_lstm_2/whileг
?forward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2A
?forward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape­
1forward_lstm_2/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_2/while:output:3Hforward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype023
1forward_lstm_2/TensorArrayV2Stack/TensorListStack
$forward_lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2&
$forward_lstm_2/strided_slice_3/stack
&forward_lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_2/strided_slice_3/stack_1
&forward_lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_3/stack_2є
forward_lstm_2/strided_slice_3StridedSlice:forward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0-forward_lstm_2/strided_slice_3/stack:output:0/forward_lstm_2/strided_slice_3/stack_1:output:0/forward_lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2 
forward_lstm_2/strided_slice_3
forward_lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_2/transpose_1/permъ
forward_lstm_2/transpose_1	Transpose:forward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0(forward_lstm_2/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_2/transpose_1
forward_lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_2/runtimef
backward_lstm_2/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_2/Shape
#backward_lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#backward_lstm_2/strided_slice/stack
%backward_lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_2/strided_slice/stack_1
%backward_lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_2/strided_slice/stack_2Т
backward_lstm_2/strided_sliceStridedSlicebackward_lstm_2/Shape:output:0,backward_lstm_2/strided_slice/stack:output:0.backward_lstm_2/strided_slice/stack_1:output:0.backward_lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
backward_lstm_2/strided_slice|
backward_lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_2/zeros/mul/yЌ
backward_lstm_2/zeros/mulMul&backward_lstm_2/strided_slice:output:0$backward_lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros/mul
backward_lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
backward_lstm_2/zeros/Less/yЇ
backward_lstm_2/zeros/LessLessbackward_lstm_2/zeros/mul:z:0%backward_lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros/Less
backward_lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_2/zeros/packed/1У
backward_lstm_2/zeros/packedPack&backward_lstm_2/strided_slice:output:0'backward_lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_2/zeros/packed
backward_lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_2/zeros/ConstЕ
backward_lstm_2/zerosFill%backward_lstm_2/zeros/packed:output:0$backward_lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/zeros
backward_lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_2/zeros_1/mul/yВ
backward_lstm_2/zeros_1/mulMul&backward_lstm_2/strided_slice:output:0&backward_lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros_1/mul
backward_lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_2/zeros_1/Less/yЏ
backward_lstm_2/zeros_1/LessLessbackward_lstm_2/zeros_1/mul:z:0'backward_lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros_1/Less
 backward_lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_2/zeros_1/packed/1Щ
backward_lstm_2/zeros_1/packedPack&backward_lstm_2/strided_slice:output:0)backward_lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_2/zeros_1/packed
backward_lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_2/zeros_1/ConstН
backward_lstm_2/zeros_1Fill'backward_lstm_2/zeros_1/packed:output:0&backward_lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/zeros_1
backward_lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
backward_lstm_2/transpose/permО
backward_lstm_2/transpose	Transposeinputs_0'backward_lstm_2/transpose/perm:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_2/transpose
backward_lstm_2/Shape_1Shapebackward_lstm_2/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_2/Shape_1
%backward_lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_2/strided_slice_1/stack
'backward_lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_1/stack_1
'backward_lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_1/stack_2Ю
backward_lstm_2/strided_slice_1StridedSlice backward_lstm_2/Shape_1:output:0.backward_lstm_2/strided_slice_1/stack:output:00backward_lstm_2/strided_slice_1/stack_1:output:00backward_lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_2/strided_slice_1Ѕ
+backward_lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+backward_lstm_2/TensorArrayV2/element_shapeђ
backward_lstm_2/TensorArrayV2TensorListReserve4backward_lstm_2/TensorArrayV2/element_shape:output:0(backward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
backward_lstm_2/TensorArrayV2
backward_lstm_2/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2 
backward_lstm_2/ReverseV2/axisг
backward_lstm_2/ReverseV2	ReverseV2backward_lstm_2/transpose:y:0'backward_lstm_2/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
backward_lstm_2/ReverseV2п
Ebackward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2G
Ebackward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeН
7backward_lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"backward_lstm_2/ReverseV2:output:0Nbackward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7backward_lstm_2/TensorArrayUnstack/TensorListFromTensor
%backward_lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_2/strided_slice_2/stack
'backward_lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_2/stack_1
'backward_lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_2/stack_2х
backward_lstm_2/strided_slice_2StridedSlicebackward_lstm_2/transpose:y:0.backward_lstm_2/strided_slice_2/stack:output:00backward_lstm_2/strided_slice_2/stack_1:output:00backward_lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shrink_axis_mask2!
backward_lstm_2/strided_slice_2т
1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp:backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype023
1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpъ
"backward_lstm_2/lstm_cell_8/MatMulMatMul(backward_lstm_2/strided_slice_2:output:09backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"backward_lstm_2/lstm_cell_8/MatMulш
3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp<backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype025
3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpц
$backward_lstm_2/lstm_cell_8/MatMul_1MatMulbackward_lstm_2/zeros:output:0;backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$backward_lstm_2/lstm_cell_8/MatMul_1м
backward_lstm_2/lstm_cell_8/addAddV2,backward_lstm_2/lstm_cell_8/MatMul:product:0.backward_lstm_2/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2!
backward_lstm_2/lstm_cell_8/addс
2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp;backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype024
2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpщ
#backward_lstm_2/lstm_cell_8/BiasAddBiasAdd#backward_lstm_2/lstm_cell_8/add:z:0:backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_2/lstm_cell_8/BiasAdd
+backward_lstm_2/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+backward_lstm_2/lstm_cell_8/split/split_dimЏ
!backward_lstm_2/lstm_cell_8/splitSplit4backward_lstm_2/lstm_cell_8/split/split_dim:output:0,backward_lstm_2/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2#
!backward_lstm_2/lstm_cell_8/splitГ
#backward_lstm_2/lstm_cell_8/SigmoidSigmoid*backward_lstm_2/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_2/lstm_cell_8/SigmoidЗ
%backward_lstm_2/lstm_cell_8/Sigmoid_1Sigmoid*backward_lstm_2/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_2/lstm_cell_8/Sigmoid_1Ш
backward_lstm_2/lstm_cell_8/mulMul)backward_lstm_2/lstm_cell_8/Sigmoid_1:y:0 backward_lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22!
backward_lstm_2/lstm_cell_8/mulЊ
 backward_lstm_2/lstm_cell_8/ReluRelu*backward_lstm_2/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/lstm_cell_8/Reluи
!backward_lstm_2/lstm_cell_8/mul_1Mul'backward_lstm_2/lstm_cell_8/Sigmoid:y:0.backward_lstm_2/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_2/lstm_cell_8/mul_1Э
!backward_lstm_2/lstm_cell_8/add_1AddV2#backward_lstm_2/lstm_cell_8/mul:z:0%backward_lstm_2/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_2/lstm_cell_8/add_1З
%backward_lstm_2/lstm_cell_8/Sigmoid_2Sigmoid*backward_lstm_2/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_2/lstm_cell_8/Sigmoid_2Љ
"backward_lstm_2/lstm_cell_8/Relu_1Relu%backward_lstm_2/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_2/lstm_cell_8/Relu_1м
!backward_lstm_2/lstm_cell_8/mul_2Mul)backward_lstm_2/lstm_cell_8/Sigmoid_2:y:00backward_lstm_2/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_2/lstm_cell_8/mul_2Џ
-backward_lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2/
-backward_lstm_2/TensorArrayV2_1/element_shapeј
backward_lstm_2/TensorArrayV2_1TensorListReserve6backward_lstm_2/TensorArrayV2_1/element_shape:output:0(backward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_2/TensorArrayV2_1n
backward_lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_2/time
(backward_lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(backward_lstm_2/while/maximum_iterations
"backward_lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"backward_lstm_2/while/loop_counterј
backward_lstm_2/whileWhile+backward_lstm_2/while/loop_counter:output:01backward_lstm_2/while/maximum_iterations:output:0backward_lstm_2/time:output:0(backward_lstm_2/TensorArrayV2_1:handle:0backward_lstm_2/zeros:output:0 backward_lstm_2/zeros_1:output:0(backward_lstm_2/strided_slice_1:output:0Gbackward_lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0:backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource<backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource;backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *-
body%R#
!backward_lstm_2_while_body_364620*-
cond%R#
!backward_lstm_2_while_cond_364619*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
backward_lstm_2/whileе
@backward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2B
@backward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeБ
2backward_lstm_2/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_2/while:output:3Ibackward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype024
2backward_lstm_2/TensorArrayV2Stack/TensorListStackЁ
%backward_lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2'
%backward_lstm_2/strided_slice_3/stack
'backward_lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_2/strided_slice_3/stack_1
'backward_lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_3/stack_2њ
backward_lstm_2/strided_slice_3StridedSlice;backward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0.backward_lstm_2/strided_slice_3/stack:output:00backward_lstm_2/strided_slice_3/stack_1:output:00backward_lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2!
backward_lstm_2/strided_slice_3
 backward_lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_2/transpose_1/permю
backward_lstm_2/transpose_1	Transpose;backward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0)backward_lstm_2/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_2/transpose_1
backward_lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_2/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisР
concatConcatV2'forward_lstm_2/strided_slice_3:output:0(backward_lstm_2/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

IdentityИ
NoOpNoOp3^backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp2^backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp4^backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp^backward_lstm_2/while2^forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp1^forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp3^forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp^forward_lstm_2/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2h
2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp2f
1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp2j
3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp2.
backward_lstm_2/whilebackward_lstm_2/while2f
1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp2d
0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp2h
2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp2,
forward_lstm_2/whileforward_lstm_2/while:g c
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
БФ

!__inference__wrapped_model_360976

args_0
args_0_1	i
Vsequential_2_bidirectional_2_forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource:	Шk
Xsequential_2_bidirectional_2_forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource:	2Шf
Wsequential_2_bidirectional_2_forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource:	Шj
Wsequential_2_bidirectional_2_backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource:	Шl
Ysequential_2_bidirectional_2_backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource:	2Шg
Xsequential_2_bidirectional_2_backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource:	ШE
3sequential_2_dense_2_matmul_readvariableop_resource:dB
4sequential_2_dense_2_biasadd_readvariableop_resource:
identityЂOsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpЂNsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpЂPsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpЂ2sequential_2/bidirectional_2/backward_lstm_2/whileЂNsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpЂMsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpЂOsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpЂ1sequential_2/bidirectional_2/forward_lstm_2/whileЂ+sequential_2/dense_2/BiasAdd/ReadVariableOpЂ*sequential_2/dense_2/MatMul/ReadVariableOpЭ
@sequential_2/bidirectional_2/forward_lstm_2/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2B
@sequential_2/bidirectional_2/forward_lstm_2/RaggedToTensor/zerosЯ
@sequential_2/bidirectional_2/forward_lstm_2/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2B
@sequential_2/bidirectional_2/forward_lstm_2/RaggedToTensor/Const
Osequential_2/bidirectional_2/forward_lstm_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorIsequential_2/bidirectional_2/forward_lstm_2/RaggedToTensor/Const:output:0args_0Isequential_2/bidirectional_2/forward_lstm_2/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2Q
Osequential_2/bidirectional_2/forward_lstm_2/RaggedToTensor/RaggedTensorToTensorњ
Vsequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2X
Vsequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice/stackў
Xsequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Z
Xsequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1ў
Xsequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Z
Xsequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2А
Psequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1_sequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack:output:0asequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1:output:0asequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask2R
Psequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_sliceў
Xsequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Z
Xsequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack
Zsequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2\
Zsequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1
Zsequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2\
Zsequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2М
Rsequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1asequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack:output:0csequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0csequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2T
Rsequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice_1§
Fsequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/subSubYsequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice:output:0[sequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2H
Fsequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/subѕ
0sequential_2/bidirectional_2/forward_lstm_2/CastCastJsequential_2/bidirectional_2/forward_lstm_2/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ22
0sequential_2/bidirectional_2/forward_lstm_2/Castю
1sequential_2/bidirectional_2/forward_lstm_2/ShapeShapeXsequential_2/bidirectional_2/forward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:23
1sequential_2/bidirectional_2/forward_lstm_2/ShapeЬ
?sequential_2/bidirectional_2/forward_lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?sequential_2/bidirectional_2/forward_lstm_2/strided_slice/stackа
Asequential_2/bidirectional_2/forward_lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential_2/bidirectional_2/forward_lstm_2/strided_slice/stack_1а
Asequential_2/bidirectional_2/forward_lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential_2/bidirectional_2/forward_lstm_2/strided_slice/stack_2ъ
9sequential_2/bidirectional_2/forward_lstm_2/strided_sliceStridedSlice:sequential_2/bidirectional_2/forward_lstm_2/Shape:output:0Hsequential_2/bidirectional_2/forward_lstm_2/strided_slice/stack:output:0Jsequential_2/bidirectional_2/forward_lstm_2/strided_slice/stack_1:output:0Jsequential_2/bidirectional_2/forward_lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9sequential_2/bidirectional_2/forward_lstm_2/strided_sliceД
7sequential_2/bidirectional_2/forward_lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :229
7sequential_2/bidirectional_2/forward_lstm_2/zeros/mul/y
5sequential_2/bidirectional_2/forward_lstm_2/zeros/mulMulBsequential_2/bidirectional_2/forward_lstm_2/strided_slice:output:0@sequential_2/bidirectional_2/forward_lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 27
5sequential_2/bidirectional_2/forward_lstm_2/zeros/mulЗ
8sequential_2/bidirectional_2/forward_lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2:
8sequential_2/bidirectional_2/forward_lstm_2/zeros/Less/y
6sequential_2/bidirectional_2/forward_lstm_2/zeros/LessLess9sequential_2/bidirectional_2/forward_lstm_2/zeros/mul:z:0Asequential_2/bidirectional_2/forward_lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 28
6sequential_2/bidirectional_2/forward_lstm_2/zeros/LessК
:sequential_2/bidirectional_2/forward_lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22<
:sequential_2/bidirectional_2/forward_lstm_2/zeros/packed/1Г
8sequential_2/bidirectional_2/forward_lstm_2/zeros/packedPackBsequential_2/bidirectional_2/forward_lstm_2/strided_slice:output:0Csequential_2/bidirectional_2/forward_lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2:
8sequential_2/bidirectional_2/forward_lstm_2/zeros/packedЛ
7sequential_2/bidirectional_2/forward_lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        29
7sequential_2/bidirectional_2/forward_lstm_2/zeros/ConstЅ
1sequential_2/bidirectional_2/forward_lstm_2/zerosFillAsequential_2/bidirectional_2/forward_lstm_2/zeros/packed:output:0@sequential_2/bidirectional_2/forward_lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ223
1sequential_2/bidirectional_2/forward_lstm_2/zerosИ
9sequential_2/bidirectional_2/forward_lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22;
9sequential_2/bidirectional_2/forward_lstm_2/zeros_1/mul/yЂ
7sequential_2/bidirectional_2/forward_lstm_2/zeros_1/mulMulBsequential_2/bidirectional_2/forward_lstm_2/strided_slice:output:0Bsequential_2/bidirectional_2/forward_lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 29
7sequential_2/bidirectional_2/forward_lstm_2/zeros_1/mulЛ
:sequential_2/bidirectional_2/forward_lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2<
:sequential_2/bidirectional_2/forward_lstm_2/zeros_1/Less/y
8sequential_2/bidirectional_2/forward_lstm_2/zeros_1/LessLess;sequential_2/bidirectional_2/forward_lstm_2/zeros_1/mul:z:0Csequential_2/bidirectional_2/forward_lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2:
8sequential_2/bidirectional_2/forward_lstm_2/zeros_1/LessО
<sequential_2/bidirectional_2/forward_lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22>
<sequential_2/bidirectional_2/forward_lstm_2/zeros_1/packed/1Й
:sequential_2/bidirectional_2/forward_lstm_2/zeros_1/packedPackBsequential_2/bidirectional_2/forward_lstm_2/strided_slice:output:0Esequential_2/bidirectional_2/forward_lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2<
:sequential_2/bidirectional_2/forward_lstm_2/zeros_1/packedП
9sequential_2/bidirectional_2/forward_lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2;
9sequential_2/bidirectional_2/forward_lstm_2/zeros_1/Const­
3sequential_2/bidirectional_2/forward_lstm_2/zeros_1FillCsequential_2/bidirectional_2/forward_lstm_2/zeros_1/packed:output:0Bsequential_2/bidirectional_2/forward_lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ225
3sequential_2/bidirectional_2/forward_lstm_2/zeros_1Э
:sequential_2/bidirectional_2/forward_lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2<
:sequential_2/bidirectional_2/forward_lstm_2/transpose/permй
5sequential_2/bidirectional_2/forward_lstm_2/transpose	TransposeXsequential_2/bidirectional_2/forward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0Csequential_2/bidirectional_2/forward_lstm_2/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ27
5sequential_2/bidirectional_2/forward_lstm_2/transposeг
3sequential_2/bidirectional_2/forward_lstm_2/Shape_1Shape9sequential_2/bidirectional_2/forward_lstm_2/transpose:y:0*
T0*
_output_shapes
:25
3sequential_2/bidirectional_2/forward_lstm_2/Shape_1а
Asequential_2/bidirectional_2/forward_lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential_2/bidirectional_2/forward_lstm_2/strided_slice_1/stackд
Csequential_2/bidirectional_2/forward_lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_2/bidirectional_2/forward_lstm_2/strided_slice_1/stack_1д
Csequential_2/bidirectional_2/forward_lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_2/bidirectional_2/forward_lstm_2/strided_slice_1/stack_2і
;sequential_2/bidirectional_2/forward_lstm_2/strided_slice_1StridedSlice<sequential_2/bidirectional_2/forward_lstm_2/Shape_1:output:0Jsequential_2/bidirectional_2/forward_lstm_2/strided_slice_1/stack:output:0Lsequential_2/bidirectional_2/forward_lstm_2/strided_slice_1/stack_1:output:0Lsequential_2/bidirectional_2/forward_lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;sequential_2/bidirectional_2/forward_lstm_2/strided_slice_1н
Gsequential_2/bidirectional_2/forward_lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2I
Gsequential_2/bidirectional_2/forward_lstm_2/TensorArrayV2/element_shapeт
9sequential_2/bidirectional_2/forward_lstm_2/TensorArrayV2TensorListReservePsequential_2/bidirectional_2/forward_lstm_2/TensorArrayV2/element_shape:output:0Dsequential_2/bidirectional_2/forward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9sequential_2/bidirectional_2/forward_lstm_2/TensorArrayV2
asequential_2/bidirectional_2/forward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2c
asequential_2/bidirectional_2/forward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeЈ
Ssequential_2/bidirectional_2/forward_lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor9sequential_2/bidirectional_2/forward_lstm_2/transpose:y:0jsequential_2/bidirectional_2/forward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02U
Ssequential_2/bidirectional_2/forward_lstm_2/TensorArrayUnstack/TensorListFromTensorа
Asequential_2/bidirectional_2/forward_lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential_2/bidirectional_2/forward_lstm_2/strided_slice_2/stackд
Csequential_2/bidirectional_2/forward_lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_2/bidirectional_2/forward_lstm_2/strided_slice_2/stack_1д
Csequential_2/bidirectional_2/forward_lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_2/bidirectional_2/forward_lstm_2/strided_slice_2/stack_2
;sequential_2/bidirectional_2/forward_lstm_2/strided_slice_2StridedSlice9sequential_2/bidirectional_2/forward_lstm_2/transpose:y:0Jsequential_2/bidirectional_2/forward_lstm_2/strided_slice_2/stack:output:0Lsequential_2/bidirectional_2/forward_lstm_2/strided_slice_2/stack_1:output:0Lsequential_2/bidirectional_2/forward_lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2=
;sequential_2/bidirectional_2/forward_lstm_2/strided_slice_2Ж
Msequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpReadVariableOpVsequential_2_bidirectional_2_forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02O
Msequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpк
>sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMulMatMulDsequential_2/bidirectional_2/forward_lstm_2/strided_slice_2:output:0Usequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2@
>sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMulМ
Osequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOpXsequential_2_bidirectional_2_forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02Q
Osequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpж
@sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMul_1MatMul:sequential_2/bidirectional_2/forward_lstm_2/zeros:output:0Wsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2B
@sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMul_1Ь
;sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/addAddV2Hsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMul:product:0Jsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2=
;sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/addЕ
Nsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOpWsequential_2_bidirectional_2_forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02P
Nsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpй
?sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/BiasAddBiasAdd?sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/add:z:0Vsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2A
?sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/BiasAddд
Gsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2I
Gsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/split/split_dim
=sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/splitSplitPsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/split/split_dim:output:0Hsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2?
=sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/split
?sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/SigmoidSigmoidFsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22A
?sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/Sigmoid
Asequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/Sigmoid_1SigmoidFsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22C
Asequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/Sigmoid_1И
;sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/mulMulEsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/Sigmoid_1:y:0<sequential_2/bidirectional_2/forward_lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22=
;sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/mulў
<sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/ReluReluFsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22>
<sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/ReluШ
=sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/mul_1MulCsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/Sigmoid:y:0Jsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22?
=sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/mul_1Н
=sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/add_1AddV2?sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/mul:z:0Asequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22?
=sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/add_1
Asequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/Sigmoid_2SigmoidFsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22C
Asequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/Sigmoid_2§
>sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/Relu_1ReluAsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22@
>sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/Relu_1Ь
=sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/mul_2MulEsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/Sigmoid_2:y:0Lsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22?
=sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/mul_2ч
Isequential_2/bidirectional_2/forward_lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2K
Isequential_2/bidirectional_2/forward_lstm_2/TensorArrayV2_1/element_shapeш
;sequential_2/bidirectional_2/forward_lstm_2/TensorArrayV2_1TensorListReserveRsequential_2/bidirectional_2/forward_lstm_2/TensorArrayV2_1/element_shape:output:0Dsequential_2/bidirectional_2/forward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_2/bidirectional_2/forward_lstm_2/TensorArrayV2_1І
0sequential_2/bidirectional_2/forward_lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_2/bidirectional_2/forward_lstm_2/timeђ
6sequential_2/bidirectional_2/forward_lstm_2/zeros_like	ZerosLikeAsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ228
6sequential_2/bidirectional_2/forward_lstm_2/zeros_likeз
Dsequential_2/bidirectional_2/forward_lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2F
Dsequential_2/bidirectional_2/forward_lstm_2/while/maximum_iterationsТ
>sequential_2/bidirectional_2/forward_lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2@
>sequential_2/bidirectional_2/forward_lstm_2/while/loop_counterд
1sequential_2/bidirectional_2/forward_lstm_2/whileWhileGsequential_2/bidirectional_2/forward_lstm_2/while/loop_counter:output:0Msequential_2/bidirectional_2/forward_lstm_2/while/maximum_iterations:output:09sequential_2/bidirectional_2/forward_lstm_2/time:output:0Dsequential_2/bidirectional_2/forward_lstm_2/TensorArrayV2_1:handle:0:sequential_2/bidirectional_2/forward_lstm_2/zeros_like:y:0:sequential_2/bidirectional_2/forward_lstm_2/zeros:output:0<sequential_2/bidirectional_2/forward_lstm_2/zeros_1:output:0Dsequential_2/bidirectional_2/forward_lstm_2/strided_slice_1:output:0csequential_2/bidirectional_2/forward_lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:04sequential_2/bidirectional_2/forward_lstm_2/Cast:y:0Vsequential_2_bidirectional_2_forward_lstm_2_lstm_cell_7_matmul_readvariableop_resourceXsequential_2_bidirectional_2_forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resourceWsequential_2_bidirectional_2_forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *I
bodyAR?
=sequential_2_bidirectional_2_forward_lstm_2_while_body_360693*I
condAR?
=sequential_2_bidirectional_2_forward_lstm_2_while_cond_360692*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 23
1sequential_2/bidirectional_2/forward_lstm_2/while
\sequential_2/bidirectional_2/forward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2^
\sequential_2/bidirectional_2/forward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeЁ
Nsequential_2/bidirectional_2/forward_lstm_2/TensorArrayV2Stack/TensorListStackTensorListStack:sequential_2/bidirectional_2/forward_lstm_2/while:output:3esequential_2/bidirectional_2/forward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02P
Nsequential_2/bidirectional_2/forward_lstm_2/TensorArrayV2Stack/TensorListStackй
Asequential_2/bidirectional_2/forward_lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2C
Asequential_2/bidirectional_2/forward_lstm_2/strided_slice_3/stackд
Csequential_2/bidirectional_2/forward_lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Csequential_2/bidirectional_2/forward_lstm_2/strided_slice_3/stack_1д
Csequential_2/bidirectional_2/forward_lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_2/bidirectional_2/forward_lstm_2/strided_slice_3/stack_2Ђ
;sequential_2/bidirectional_2/forward_lstm_2/strided_slice_3StridedSliceWsequential_2/bidirectional_2/forward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0Jsequential_2/bidirectional_2/forward_lstm_2/strided_slice_3/stack:output:0Lsequential_2/bidirectional_2/forward_lstm_2/strided_slice_3/stack_1:output:0Lsequential_2/bidirectional_2/forward_lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2=
;sequential_2/bidirectional_2/forward_lstm_2/strided_slice_3б
<sequential_2/bidirectional_2/forward_lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2>
<sequential_2/bidirectional_2/forward_lstm_2/transpose_1/permо
7sequential_2/bidirectional_2/forward_lstm_2/transpose_1	TransposeWsequential_2/bidirectional_2/forward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0Esequential_2/bidirectional_2/forward_lstm_2/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ229
7sequential_2/bidirectional_2/forward_lstm_2/transpose_1О
3sequential_2/bidirectional_2/forward_lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    25
3sequential_2/bidirectional_2/forward_lstm_2/runtimeЯ
Asequential_2/bidirectional_2/backward_lstm_2/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2C
Asequential_2/bidirectional_2/backward_lstm_2/RaggedToTensor/zerosб
Asequential_2/bidirectional_2/backward_lstm_2/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2C
Asequential_2/bidirectional_2/backward_lstm_2/RaggedToTensor/Const
Psequential_2/bidirectional_2/backward_lstm_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorJsequential_2/bidirectional_2/backward_lstm_2/RaggedToTensor/Const:output:0args_0Jsequential_2/bidirectional_2/backward_lstm_2/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2R
Psequential_2/bidirectional_2/backward_lstm_2/RaggedToTensor/RaggedTensorToTensorќ
Wsequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2Y
Wsequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack
Ysequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2[
Ysequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1
Ysequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2[
Ysequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2Е
Qsequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1`sequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack:output:0bsequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1:output:0bsequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask2S
Qsequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice
Ysequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2[
Ysequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack
[sequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2]
[sequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1
[sequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2]
[sequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2С
Ssequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1bsequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack:output:0dsequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0dsequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask2U
Ssequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice_1
Gsequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/subSubZsequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice:output:0\sequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2I
Gsequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/subј
1sequential_2/bidirectional_2/backward_lstm_2/CastCastKsequential_2/bidirectional_2/backward_lstm_2/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ23
1sequential_2/bidirectional_2/backward_lstm_2/Castё
2sequential_2/bidirectional_2/backward_lstm_2/ShapeShapeYsequential_2/bidirectional_2/backward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:24
2sequential_2/bidirectional_2/backward_lstm_2/ShapeЮ
@sequential_2/bidirectional_2/backward_lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_2/bidirectional_2/backward_lstm_2/strided_slice/stackв
Bsequential_2/bidirectional_2/backward_lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/bidirectional_2/backward_lstm_2/strided_slice/stack_1в
Bsequential_2/bidirectional_2/backward_lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/bidirectional_2/backward_lstm_2/strided_slice/stack_2№
:sequential_2/bidirectional_2/backward_lstm_2/strided_sliceStridedSlice;sequential_2/bidirectional_2/backward_lstm_2/Shape:output:0Isequential_2/bidirectional_2/backward_lstm_2/strided_slice/stack:output:0Ksequential_2/bidirectional_2/backward_lstm_2/strided_slice/stack_1:output:0Ksequential_2/bidirectional_2/backward_lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_2/bidirectional_2/backward_lstm_2/strided_sliceЖ
8sequential_2/bidirectional_2/backward_lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22:
8sequential_2/bidirectional_2/backward_lstm_2/zeros/mul/y 
6sequential_2/bidirectional_2/backward_lstm_2/zeros/mulMulCsequential_2/bidirectional_2/backward_lstm_2/strided_slice:output:0Asequential_2/bidirectional_2/backward_lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 28
6sequential_2/bidirectional_2/backward_lstm_2/zeros/mulЙ
9sequential_2/bidirectional_2/backward_lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2;
9sequential_2/bidirectional_2/backward_lstm_2/zeros/Less/y
7sequential_2/bidirectional_2/backward_lstm_2/zeros/LessLess:sequential_2/bidirectional_2/backward_lstm_2/zeros/mul:z:0Bsequential_2/bidirectional_2/backward_lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 29
7sequential_2/bidirectional_2/backward_lstm_2/zeros/LessМ
;sequential_2/bidirectional_2/backward_lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22=
;sequential_2/bidirectional_2/backward_lstm_2/zeros/packed/1З
9sequential_2/bidirectional_2/backward_lstm_2/zeros/packedPackCsequential_2/bidirectional_2/backward_lstm_2/strided_slice:output:0Dsequential_2/bidirectional_2/backward_lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2;
9sequential_2/bidirectional_2/backward_lstm_2/zeros/packedН
8sequential_2/bidirectional_2/backward_lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2:
8sequential_2/bidirectional_2/backward_lstm_2/zeros/ConstЉ
2sequential_2/bidirectional_2/backward_lstm_2/zerosFillBsequential_2/bidirectional_2/backward_lstm_2/zeros/packed:output:0Asequential_2/bidirectional_2/backward_lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ224
2sequential_2/bidirectional_2/backward_lstm_2/zerosК
:sequential_2/bidirectional_2/backward_lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22<
:sequential_2/bidirectional_2/backward_lstm_2/zeros_1/mul/yІ
8sequential_2/bidirectional_2/backward_lstm_2/zeros_1/mulMulCsequential_2/bidirectional_2/backward_lstm_2/strided_slice:output:0Csequential_2/bidirectional_2/backward_lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2:
8sequential_2/bidirectional_2/backward_lstm_2/zeros_1/mulН
;sequential_2/bidirectional_2/backward_lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2=
;sequential_2/bidirectional_2/backward_lstm_2/zeros_1/Less/yЃ
9sequential_2/bidirectional_2/backward_lstm_2/zeros_1/LessLess<sequential_2/bidirectional_2/backward_lstm_2/zeros_1/mul:z:0Dsequential_2/bidirectional_2/backward_lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2;
9sequential_2/bidirectional_2/backward_lstm_2/zeros_1/LessР
=sequential_2/bidirectional_2/backward_lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22?
=sequential_2/bidirectional_2/backward_lstm_2/zeros_1/packed/1Н
;sequential_2/bidirectional_2/backward_lstm_2/zeros_1/packedPackCsequential_2/bidirectional_2/backward_lstm_2/strided_slice:output:0Fsequential_2/bidirectional_2/backward_lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2=
;sequential_2/bidirectional_2/backward_lstm_2/zeros_1/packedС
:sequential_2/bidirectional_2/backward_lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2<
:sequential_2/bidirectional_2/backward_lstm_2/zeros_1/ConstБ
4sequential_2/bidirectional_2/backward_lstm_2/zeros_1FillDsequential_2/bidirectional_2/backward_lstm_2/zeros_1/packed:output:0Csequential_2/bidirectional_2/backward_lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ226
4sequential_2/bidirectional_2/backward_lstm_2/zeros_1Я
;sequential_2/bidirectional_2/backward_lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2=
;sequential_2/bidirectional_2/backward_lstm_2/transpose/permн
6sequential_2/bidirectional_2/backward_lstm_2/transpose	TransposeYsequential_2/bidirectional_2/backward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0Dsequential_2/bidirectional_2/backward_lstm_2/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ28
6sequential_2/bidirectional_2/backward_lstm_2/transposeж
4sequential_2/bidirectional_2/backward_lstm_2/Shape_1Shape:sequential_2/bidirectional_2/backward_lstm_2/transpose:y:0*
T0*
_output_shapes
:26
4sequential_2/bidirectional_2/backward_lstm_2/Shape_1в
Bsequential_2/bidirectional_2/backward_lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_2/bidirectional_2/backward_lstm_2/strided_slice_1/stackж
Dsequential_2/bidirectional_2/backward_lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/bidirectional_2/backward_lstm_2/strided_slice_1/stack_1ж
Dsequential_2/bidirectional_2/backward_lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/bidirectional_2/backward_lstm_2/strided_slice_1/stack_2ќ
<sequential_2/bidirectional_2/backward_lstm_2/strided_slice_1StridedSlice=sequential_2/bidirectional_2/backward_lstm_2/Shape_1:output:0Ksequential_2/bidirectional_2/backward_lstm_2/strided_slice_1/stack:output:0Msequential_2/bidirectional_2/backward_lstm_2/strided_slice_1/stack_1:output:0Msequential_2/bidirectional_2/backward_lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_2/bidirectional_2/backward_lstm_2/strided_slice_1п
Hsequential_2/bidirectional_2/backward_lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2J
Hsequential_2/bidirectional_2/backward_lstm_2/TensorArrayV2/element_shapeц
:sequential_2/bidirectional_2/backward_lstm_2/TensorArrayV2TensorListReserveQsequential_2/bidirectional_2/backward_lstm_2/TensorArrayV2/element_shape:output:0Esequential_2/bidirectional_2/backward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02<
:sequential_2/bidirectional_2/backward_lstm_2/TensorArrayV2Ф
;sequential_2/bidirectional_2/backward_lstm_2/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2=
;sequential_2/bidirectional_2/backward_lstm_2/ReverseV2/axisО
6sequential_2/bidirectional_2/backward_lstm_2/ReverseV2	ReverseV2:sequential_2/bidirectional_2/backward_lstm_2/transpose:y:0Dsequential_2/bidirectional_2/backward_lstm_2/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ28
6sequential_2/bidirectional_2/backward_lstm_2/ReverseV2
bsequential_2/bidirectional_2/backward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2d
bsequential_2/bidirectional_2/backward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeБ
Tsequential_2/bidirectional_2/backward_lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor?sequential_2/bidirectional_2/backward_lstm_2/ReverseV2:output:0ksequential_2/bidirectional_2/backward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02V
Tsequential_2/bidirectional_2/backward_lstm_2/TensorArrayUnstack/TensorListFromTensorв
Bsequential_2/bidirectional_2/backward_lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_2/bidirectional_2/backward_lstm_2/strided_slice_2/stackж
Dsequential_2/bidirectional_2/backward_lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/bidirectional_2/backward_lstm_2/strided_slice_2/stack_1ж
Dsequential_2/bidirectional_2/backward_lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/bidirectional_2/backward_lstm_2/strided_slice_2/stack_2
<sequential_2/bidirectional_2/backward_lstm_2/strided_slice_2StridedSlice:sequential_2/bidirectional_2/backward_lstm_2/transpose:y:0Ksequential_2/bidirectional_2/backward_lstm_2/strided_slice_2/stack:output:0Msequential_2/bidirectional_2/backward_lstm_2/strided_slice_2/stack_1:output:0Msequential_2/bidirectional_2/backward_lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2>
<sequential_2/bidirectional_2/backward_lstm_2/strided_slice_2Й
Nsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpReadVariableOpWsequential_2_bidirectional_2_backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02P
Nsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpо
?sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMulMatMulEsequential_2/bidirectional_2/backward_lstm_2/strided_slice_2:output:0Vsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2A
?sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMulП
Psequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOpYsequential_2_bidirectional_2_backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02R
Psequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpк
Asequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMul_1MatMul;sequential_2/bidirectional_2/backward_lstm_2/zeros:output:0Xsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2C
Asequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMul_1а
<sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/addAddV2Isequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMul:product:0Ksequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2>
<sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/addИ
Osequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOpXsequential_2_bidirectional_2_backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02Q
Osequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpн
@sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/BiasAddBiasAdd@sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/add:z:0Wsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2B
@sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/BiasAddж
Hsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2J
Hsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/split/split_dimЃ
>sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/splitSplitQsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/split/split_dim:output:0Isequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2@
>sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/split
@sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/SigmoidSigmoidGsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22B
@sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/Sigmoid
Bsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/Sigmoid_1SigmoidGsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22D
Bsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/Sigmoid_1М
<sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/mulMulFsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/Sigmoid_1:y:0=sequential_2/bidirectional_2/backward_lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22>
<sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/mul
=sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/ReluReluGsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22?
=sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/ReluЬ
>sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/mul_1MulDsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/Sigmoid:y:0Ksequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22@
>sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/mul_1С
>sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/add_1AddV2@sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/mul:z:0Bsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22@
>sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/add_1
Bsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/Sigmoid_2SigmoidGsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22D
Bsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/Sigmoid_2
?sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/Relu_1ReluBsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22A
?sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/Relu_1а
>sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/mul_2MulFsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/Sigmoid_2:y:0Msequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22@
>sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/mul_2щ
Jsequential_2/bidirectional_2/backward_lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2L
Jsequential_2/bidirectional_2/backward_lstm_2/TensorArrayV2_1/element_shapeь
<sequential_2/bidirectional_2/backward_lstm_2/TensorArrayV2_1TensorListReserveSsequential_2/bidirectional_2/backward_lstm_2/TensorArrayV2_1/element_shape:output:0Esequential_2/bidirectional_2/backward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_2/bidirectional_2/backward_lstm_2/TensorArrayV2_1Ј
1sequential_2/bidirectional_2/backward_lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential_2/bidirectional_2/backward_lstm_2/timeЪ
Bsequential_2/bidirectional_2/backward_lstm_2/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bsequential_2/bidirectional_2/backward_lstm_2/Max/reduction_indices
0sequential_2/bidirectional_2/backward_lstm_2/MaxMax5sequential_2/bidirectional_2/backward_lstm_2/Cast:y:0Ksequential_2/bidirectional_2/backward_lstm_2/Max/reduction_indices:output:0*
T0*
_output_shapes
: 22
0sequential_2/bidirectional_2/backward_lstm_2/MaxЊ
2sequential_2/bidirectional_2/backward_lstm_2/sub/yConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_2/bidirectional_2/backward_lstm_2/sub/y
0sequential_2/bidirectional_2/backward_lstm_2/subSub9sequential_2/bidirectional_2/backward_lstm_2/Max:output:0;sequential_2/bidirectional_2/backward_lstm_2/sub/y:output:0*
T0*
_output_shapes
: 22
0sequential_2/bidirectional_2/backward_lstm_2/sub
2sequential_2/bidirectional_2/backward_lstm_2/Sub_1Sub4sequential_2/bidirectional_2/backward_lstm_2/sub:z:05sequential_2/bidirectional_2/backward_lstm_2/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ24
2sequential_2/bidirectional_2/backward_lstm_2/Sub_1ѕ
7sequential_2/bidirectional_2/backward_lstm_2/zeros_like	ZerosLikeBsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ229
7sequential_2/bidirectional_2/backward_lstm_2/zeros_likeй
Esequential_2/bidirectional_2/backward_lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2G
Esequential_2/bidirectional_2/backward_lstm_2/while/maximum_iterationsФ
?sequential_2/bidirectional_2/backward_lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2A
?sequential_2/bidirectional_2/backward_lstm_2/while/loop_counterц
2sequential_2/bidirectional_2/backward_lstm_2/whileWhileHsequential_2/bidirectional_2/backward_lstm_2/while/loop_counter:output:0Nsequential_2/bidirectional_2/backward_lstm_2/while/maximum_iterations:output:0:sequential_2/bidirectional_2/backward_lstm_2/time:output:0Esequential_2/bidirectional_2/backward_lstm_2/TensorArrayV2_1:handle:0;sequential_2/bidirectional_2/backward_lstm_2/zeros_like:y:0;sequential_2/bidirectional_2/backward_lstm_2/zeros:output:0=sequential_2/bidirectional_2/backward_lstm_2/zeros_1:output:0Esequential_2/bidirectional_2/backward_lstm_2/strided_slice_1:output:0dsequential_2/bidirectional_2/backward_lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:06sequential_2/bidirectional_2/backward_lstm_2/Sub_1:z:0Wsequential_2_bidirectional_2_backward_lstm_2_lstm_cell_8_matmul_readvariableop_resourceYsequential_2_bidirectional_2_backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resourceXsequential_2_bidirectional_2_backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *J
bodyBR@
>sequential_2_bidirectional_2_backward_lstm_2_while_body_360872*J
condBR@
>sequential_2_bidirectional_2_backward_lstm_2_while_cond_360871*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 24
2sequential_2/bidirectional_2/backward_lstm_2/while
]sequential_2/bidirectional_2/backward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2_
]sequential_2/bidirectional_2/backward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeЅ
Osequential_2/bidirectional_2/backward_lstm_2/TensorArrayV2Stack/TensorListStackTensorListStack;sequential_2/bidirectional_2/backward_lstm_2/while:output:3fsequential_2/bidirectional_2/backward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02Q
Osequential_2/bidirectional_2/backward_lstm_2/TensorArrayV2Stack/TensorListStackл
Bsequential_2/bidirectional_2/backward_lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2D
Bsequential_2/bidirectional_2/backward_lstm_2/strided_slice_3/stackж
Dsequential_2/bidirectional_2/backward_lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2F
Dsequential_2/bidirectional_2/backward_lstm_2/strided_slice_3/stack_1ж
Dsequential_2/bidirectional_2/backward_lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/bidirectional_2/backward_lstm_2/strided_slice_3/stack_2Ј
<sequential_2/bidirectional_2/backward_lstm_2/strided_slice_3StridedSliceXsequential_2/bidirectional_2/backward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0Ksequential_2/bidirectional_2/backward_lstm_2/strided_slice_3/stack:output:0Msequential_2/bidirectional_2/backward_lstm_2/strided_slice_3/stack_1:output:0Msequential_2/bidirectional_2/backward_lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2>
<sequential_2/bidirectional_2/backward_lstm_2/strided_slice_3г
=sequential_2/bidirectional_2/backward_lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2?
=sequential_2/bidirectional_2/backward_lstm_2/transpose_1/permт
8sequential_2/bidirectional_2/backward_lstm_2/transpose_1	TransposeXsequential_2/bidirectional_2/backward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0Fsequential_2/bidirectional_2/backward_lstm_2/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22:
8sequential_2/bidirectional_2/backward_lstm_2/transpose_1Р
4sequential_2/bidirectional_2/backward_lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    26
4sequential_2/bidirectional_2/backward_lstm_2/runtime
(sequential_2/bidirectional_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_2/bidirectional_2/concat/axisб
#sequential_2/bidirectional_2/concatConcatV2Dsequential_2/bidirectional_2/forward_lstm_2/strided_slice_3:output:0Esequential_2/bidirectional_2/backward_lstm_2/strided_slice_3:output:01sequential_2/bidirectional_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2%
#sequential_2/bidirectional_2/concatЬ
*sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_2_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02,
*sequential_2/dense_2/MatMul/ReadVariableOpи
sequential_2/dense_2/MatMulMatMul,sequential_2/bidirectional_2/concat:output:02sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_2/dense_2/MatMulЫ
+sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_2/dense_2/BiasAdd/ReadVariableOpе
sequential_2/dense_2/BiasAddBiasAdd%sequential_2/dense_2/MatMul:product:03sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_2/dense_2/BiasAdd 
sequential_2/dense_2/SigmoidSigmoid%sequential_2/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_2/dense_2/Sigmoid{
IdentityIdentity sequential_2/dense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityћ
NoOpNoOpP^sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpO^sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpQ^sequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp3^sequential_2/bidirectional_2/backward_lstm_2/whileO^sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpN^sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpP^sequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp2^sequential_2/bidirectional_2/forward_lstm_2/while,^sequential_2/dense_2/BiasAdd/ReadVariableOp+^sequential_2/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2Ђ
Osequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpOsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp2 
Nsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpNsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp2Є
Psequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpPsequential_2/bidirectional_2/backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp2h
2sequential_2/bidirectional_2/backward_lstm_2/while2sequential_2/bidirectional_2/backward_lstm_2/while2 
Nsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpNsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp2
Msequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpMsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp2Ђ
Osequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpOsequential_2/bidirectional_2/forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp2f
1sequential_2/bidirectional_2/forward_lstm_2/while1sequential_2/bidirectional_2/forward_lstm_2/while2Z
+sequential_2/dense_2/BiasAdd/ReadVariableOp+sequential_2/dense_2/BiasAdd/ReadVariableOp2X
*sequential_2/dense_2/MatMul/ReadVariableOp*sequential_2/dense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameargs_0:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameargs_0
эa

!backward_lstm_2_while_body_364967<
8backward_lstm_2_while_backward_lstm_2_while_loop_counterB
>backward_lstm_2_while_backward_lstm_2_while_maximum_iterations%
!backward_lstm_2_while_placeholder'
#backward_lstm_2_while_placeholder_1'
#backward_lstm_2_while_placeholder_2'
#backward_lstm_2_while_placeholder_3'
#backward_lstm_2_while_placeholder_4;
7backward_lstm_2_while_backward_lstm_2_strided_slice_1_0w
sbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_06
2backward_lstm_2_while_less_backward_lstm_2_sub_1_0U
Bbackward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0:	ШW
Dbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0:	2ШR
Cbackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ш"
backward_lstm_2_while_identity$
 backward_lstm_2_while_identity_1$
 backward_lstm_2_while_identity_2$
 backward_lstm_2_while_identity_3$
 backward_lstm_2_while_identity_4$
 backward_lstm_2_while_identity_5$
 backward_lstm_2_while_identity_69
5backward_lstm_2_while_backward_lstm_2_strided_slice_1u
qbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor4
0backward_lstm_2_while_less_backward_lstm_2_sub_1S
@backward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource:	ШU
Bbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource:	2ШP
Abackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource:	ШЂ8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOpЂ7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOpЂ9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOpу
Gbackward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gbackward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
9backward_lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0!backward_lstm_2_while_placeholderPbackward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02;
9backward_lstm_2/while/TensorArrayV2Read/TensorListGetItemХ
backward_lstm_2/while/LessLess2backward_lstm_2_while_less_backward_lstm_2_sub_1_0!backward_lstm_2_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_2/while/Lessі
7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOpBbackward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype029
7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp
(backward_lstm_2/while/lstm_cell_8/MatMulMatMul@backward_lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0?backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_2/while/lstm_cell_8/MatMulќ
9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOpDbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02;
9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp§
*backward_lstm_2/while/lstm_cell_8/MatMul_1MatMul#backward_lstm_2_while_placeholder_3Abackward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2,
*backward_lstm_2/while/lstm_cell_8/MatMul_1є
%backward_lstm_2/while/lstm_cell_8/addAddV22backward_lstm_2/while/lstm_cell_8/MatMul:product:04backward_lstm_2/while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%backward_lstm_2/while/lstm_cell_8/addѕ
8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOpCbackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02:
8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp
)backward_lstm_2/while/lstm_cell_8/BiasAddBiasAdd)backward_lstm_2/while/lstm_cell_8/add:z:0@backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_2/while/lstm_cell_8/BiasAddЈ
1backward_lstm_2/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1backward_lstm_2/while/lstm_cell_8/split/split_dimЧ
'backward_lstm_2/while/lstm_cell_8/splitSplit:backward_lstm_2/while/lstm_cell_8/split/split_dim:output:02backward_lstm_2/while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2)
'backward_lstm_2/while/lstm_cell_8/splitХ
)backward_lstm_2/while/lstm_cell_8/SigmoidSigmoid0backward_lstm_2/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_2/while/lstm_cell_8/SigmoidЩ
+backward_lstm_2/while/lstm_cell_8/Sigmoid_1Sigmoid0backward_lstm_2/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_2/while/lstm_cell_8/Sigmoid_1н
%backward_lstm_2/while/lstm_cell_8/mulMul/backward_lstm_2/while/lstm_cell_8/Sigmoid_1:y:0#backward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_2/while/lstm_cell_8/mulМ
&backward_lstm_2/while/lstm_cell_8/ReluRelu0backward_lstm_2/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_2/while/lstm_cell_8/Relu№
'backward_lstm_2/while/lstm_cell_8/mul_1Mul-backward_lstm_2/while/lstm_cell_8/Sigmoid:y:04backward_lstm_2/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_2/while/lstm_cell_8/mul_1х
'backward_lstm_2/while/lstm_cell_8/add_1AddV2)backward_lstm_2/while/lstm_cell_8/mul:z:0+backward_lstm_2/while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_2/while/lstm_cell_8/add_1Щ
+backward_lstm_2/while/lstm_cell_8/Sigmoid_2Sigmoid0backward_lstm_2/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_2/while/lstm_cell_8/Sigmoid_2Л
(backward_lstm_2/while/lstm_cell_8/Relu_1Relu+backward_lstm_2/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_2/while/lstm_cell_8/Relu_1є
'backward_lstm_2/while/lstm_cell_8/mul_2Mul/backward_lstm_2/while/lstm_cell_8/Sigmoid_2:y:06backward_lstm_2/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_2/while/lstm_cell_8/mul_2ъ
backward_lstm_2/while/SelectSelectbackward_lstm_2/while/Less:z:0+backward_lstm_2/while/lstm_cell_8/mul_2:z:0#backward_lstm_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/while/Selectю
backward_lstm_2/while/Select_1Selectbackward_lstm_2/while/Less:z:0+backward_lstm_2/while/lstm_cell_8/mul_2:z:0#backward_lstm_2_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_2/while/Select_1ю
backward_lstm_2/while/Select_2Selectbackward_lstm_2/while/Less:z:0+backward_lstm_2/while/lstm_cell_8/add_1:z:0#backward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_2/while/Select_2Љ
:backward_lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#backward_lstm_2_while_placeholder_1!backward_lstm_2_while_placeholder%backward_lstm_2/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:backward_lstm_2/while/TensorArrayV2Write/TensorListSetItem|
backward_lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_2/while/add/yЉ
backward_lstm_2/while/addAddV2!backward_lstm_2_while_placeholder$backward_lstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/while/add
backward_lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_2/while/add_1/yЦ
backward_lstm_2/while/add_1AddV28backward_lstm_2_while_backward_lstm_2_while_loop_counter&backward_lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/while/add_1Ћ
backward_lstm_2/while/IdentityIdentitybackward_lstm_2/while/add_1:z:0^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2 
backward_lstm_2/while/IdentityЮ
 backward_lstm_2/while/Identity_1Identity>backward_lstm_2_while_backward_lstm_2_while_maximum_iterations^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_2/while/Identity_1­
 backward_lstm_2/while/Identity_2Identitybackward_lstm_2/while/add:z:0^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_2/while/Identity_2к
 backward_lstm_2/while/Identity_3IdentityJbackward_lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_2/while/Identity_3Ц
 backward_lstm_2/while/Identity_4Identity%backward_lstm_2/while/Select:output:0^backward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/while/Identity_4Ш
 backward_lstm_2/while/Identity_5Identity'backward_lstm_2/while/Select_1:output:0^backward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/while/Identity_5Ш
 backward_lstm_2/while/Identity_6Identity'backward_lstm_2/while/Select_2:output:0^backward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/while/Identity_6Ћ
backward_lstm_2/while/NoOpNoOp9^backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp8^backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp:^backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_2/while/NoOp"p
5backward_lstm_2_while_backward_lstm_2_strided_slice_17backward_lstm_2_while_backward_lstm_2_strided_slice_1_0"I
backward_lstm_2_while_identity'backward_lstm_2/while/Identity:output:0"M
 backward_lstm_2_while_identity_1)backward_lstm_2/while/Identity_1:output:0"M
 backward_lstm_2_while_identity_2)backward_lstm_2/while/Identity_2:output:0"M
 backward_lstm_2_while_identity_3)backward_lstm_2/while/Identity_3:output:0"M
 backward_lstm_2_while_identity_4)backward_lstm_2/while/Identity_4:output:0"M
 backward_lstm_2_while_identity_5)backward_lstm_2/while/Identity_5:output:0"M
 backward_lstm_2_while_identity_6)backward_lstm_2/while/Identity_6:output:0"f
0backward_lstm_2_while_less_backward_lstm_2_sub_12backward_lstm_2_while_less_backward_lstm_2_sub_1_0"
Abackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resourceCbackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0"
Bbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resourceDbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0"
@backward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resourceBbackward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0"ш
qbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensorsbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2t
8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp2r
7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp2v
9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Ю
П
0__inference_backward_lstm_2_layer_call_fn_366101
inputs_0
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
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
GPU 2J 8 *T
fORM
K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_3617662
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
ЊБ
З
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_363412

inputs
inputs_1	L
9forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource:	ШN
;forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource:	2ШI
:forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource:	ШM
:backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource:	ШO
<backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource:	2ШJ
;backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource:	Ш
identityЂ2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpЂ1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpЂ3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpЂbackward_lstm_2/whileЂ1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpЂ0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpЂ2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpЂforward_lstm_2/while
#forward_lstm_2/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2%
#forward_lstm_2/RaggedToTensor/zeros
#forward_lstm_2/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2%
#forward_lstm_2/RaggedToTensor/Const
2forward_lstm_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor,forward_lstm_2/RaggedToTensor/Const:output:0inputs,forward_lstm_2/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS24
2forward_lstm_2/RaggedToTensor/RaggedTensorToTensorР
9forward_lstm_2/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9forward_lstm_2/RaggedNestedRowLengths/strided_slice/stackФ
;forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1Ф
;forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2
3forward_lstm_2/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Bforward_lstm_2/RaggedNestedRowLengths/strided_slice/stack:output:0Dforward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1:output:0Dforward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask25
3forward_lstm_2/RaggedNestedRowLengths/strided_sliceФ
;forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stackб
=forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2?
=forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1Ш
=forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2Ћ
5forward_lstm_2/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Dforward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack:output:0Fforward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Fforward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask27
5forward_lstm_2/RaggedNestedRowLengths/strided_slice_1
)forward_lstm_2/RaggedNestedRowLengths/subSub<forward_lstm_2/RaggedNestedRowLengths/strided_slice:output:0>forward_lstm_2/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2+
)forward_lstm_2/RaggedNestedRowLengths/sub
forward_lstm_2/CastCast-forward_lstm_2/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_2/Cast
forward_lstm_2/ShapeShape;forward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_2/Shape
"forward_lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"forward_lstm_2/strided_slice/stack
$forward_lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_2/strided_slice/stack_1
$forward_lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_2/strided_slice/stack_2М
forward_lstm_2/strided_sliceStridedSliceforward_lstm_2/Shape:output:0+forward_lstm_2/strided_slice/stack:output:0-forward_lstm_2/strided_slice/stack_1:output:0-forward_lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_2/strided_slicez
forward_lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_2/zeros/mul/yЈ
forward_lstm_2/zeros/mulMul%forward_lstm_2/strided_slice:output:0#forward_lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros/mul}
forward_lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_2/zeros/Less/yЃ
forward_lstm_2/zeros/LessLessforward_lstm_2/zeros/mul:z:0$forward_lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros/Less
forward_lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_2/zeros/packed/1П
forward_lstm_2/zeros/packedPack%forward_lstm_2/strided_slice:output:0&forward_lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_2/zeros/packed
forward_lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_2/zeros/ConstБ
forward_lstm_2/zerosFill$forward_lstm_2/zeros/packed:output:0#forward_lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/zeros~
forward_lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_2/zeros_1/mul/yЎ
forward_lstm_2/zeros_1/mulMul%forward_lstm_2/strided_slice:output:0%forward_lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros_1/mul
forward_lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_2/zeros_1/Less/yЋ
forward_lstm_2/zeros_1/LessLessforward_lstm_2/zeros_1/mul:z:0&forward_lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros_1/Less
forward_lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_2/zeros_1/packed/1Х
forward_lstm_2/zeros_1/packedPack%forward_lstm_2/strided_slice:output:0(forward_lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_2/zeros_1/packed
forward_lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_2/zeros_1/ConstЙ
forward_lstm_2/zeros_1Fill&forward_lstm_2/zeros_1/packed:output:0%forward_lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/zeros_1
forward_lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
forward_lstm_2/transpose/permх
forward_lstm_2/transpose	Transpose;forward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0&forward_lstm_2/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_2/transpose|
forward_lstm_2/Shape_1Shapeforward_lstm_2/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_2/Shape_1
$forward_lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_2/strided_slice_1/stack
&forward_lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_1/stack_1
&forward_lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_1/stack_2Ш
forward_lstm_2/strided_slice_1StridedSliceforward_lstm_2/Shape_1:output:0-forward_lstm_2/strided_slice_1/stack:output:0/forward_lstm_2/strided_slice_1/stack_1:output:0/forward_lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_2/strided_slice_1Ѓ
*forward_lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*forward_lstm_2/TensorArrayV2/element_shapeю
forward_lstm_2/TensorArrayV2TensorListReserve3forward_lstm_2/TensorArrayV2/element_shape:output:0'forward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_2/TensorArrayV2н
Dforward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2F
Dforward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeД
6forward_lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_2/transpose:y:0Mforward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6forward_lstm_2/TensorArrayUnstack/TensorListFromTensor
$forward_lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_2/strided_slice_2/stack
&forward_lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_2/stack_1
&forward_lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_2/stack_2ж
forward_lstm_2/strided_slice_2StridedSliceforward_lstm_2/transpose:y:0-forward_lstm_2/strided_slice_2/stack:output:0/forward_lstm_2/strided_slice_2/stack_1:output:0/forward_lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2 
forward_lstm_2/strided_slice_2п
0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp9forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype022
0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpц
!forward_lstm_2/lstm_cell_7/MatMulMatMul'forward_lstm_2/strided_slice_2:output:08forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2#
!forward_lstm_2/lstm_cell_7/MatMulх
2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp;forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype024
2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpт
#forward_lstm_2/lstm_cell_7/MatMul_1MatMulforward_lstm_2/zeros:output:0:forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#forward_lstm_2/lstm_cell_7/MatMul_1и
forward_lstm_2/lstm_cell_7/addAddV2+forward_lstm_2/lstm_cell_7/MatMul:product:0-forward_lstm_2/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2 
forward_lstm_2/lstm_cell_7/addо
1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp:forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype023
1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpх
"forward_lstm_2/lstm_cell_7/BiasAddBiasAdd"forward_lstm_2/lstm_cell_7/add:z:09forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_2/lstm_cell_7/BiasAdd
*forward_lstm_2/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*forward_lstm_2/lstm_cell_7/split/split_dimЋ
 forward_lstm_2/lstm_cell_7/splitSplit3forward_lstm_2/lstm_cell_7/split/split_dim:output:0+forward_lstm_2/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2"
 forward_lstm_2/lstm_cell_7/splitА
"forward_lstm_2/lstm_cell_7/SigmoidSigmoid)forward_lstm_2/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_2/lstm_cell_7/SigmoidД
$forward_lstm_2/lstm_cell_7/Sigmoid_1Sigmoid)forward_lstm_2/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_2/lstm_cell_7/Sigmoid_1Ф
forward_lstm_2/lstm_cell_7/mulMul(forward_lstm_2/lstm_cell_7/Sigmoid_1:y:0forward_lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
forward_lstm_2/lstm_cell_7/mulЇ
forward_lstm_2/lstm_cell_7/ReluRelu)forward_lstm_2/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/lstm_cell_7/Reluд
 forward_lstm_2/lstm_cell_7/mul_1Mul&forward_lstm_2/lstm_cell_7/Sigmoid:y:0-forward_lstm_2/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_2/lstm_cell_7/mul_1Щ
 forward_lstm_2/lstm_cell_7/add_1AddV2"forward_lstm_2/lstm_cell_7/mul:z:0$forward_lstm_2/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_2/lstm_cell_7/add_1Д
$forward_lstm_2/lstm_cell_7/Sigmoid_2Sigmoid)forward_lstm_2/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_2/lstm_cell_7/Sigmoid_2І
!forward_lstm_2/lstm_cell_7/Relu_1Relu$forward_lstm_2/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_2/lstm_cell_7/Relu_1и
 forward_lstm_2/lstm_cell_7/mul_2Mul(forward_lstm_2/lstm_cell_7/Sigmoid_2:y:0/forward_lstm_2/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_2/lstm_cell_7/mul_2­
,forward_lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2.
,forward_lstm_2/TensorArrayV2_1/element_shapeє
forward_lstm_2/TensorArrayV2_1TensorListReserve5forward_lstm_2/TensorArrayV2_1/element_shape:output:0'forward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_2/TensorArrayV2_1l
forward_lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_2/time
forward_lstm_2/zeros_like	ZerosLike$forward_lstm_2/lstm_cell_7/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/zeros_like
'forward_lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'forward_lstm_2/while/maximum_iterations
!forward_lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!forward_lstm_2/while/loop_counterч
forward_lstm_2/whileWhile*forward_lstm_2/while/loop_counter:output:00forward_lstm_2/while/maximum_iterations:output:0forward_lstm_2/time:output:0'forward_lstm_2/TensorArrayV2_1:handle:0forward_lstm_2/zeros_like:y:0forward_lstm_2/zeros:output:0forward_lstm_2/zeros_1:output:0'forward_lstm_2/strided_slice_1:output:0Fforward_lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_2/Cast:y:09forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource;forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource:forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *,
body$R"
 forward_lstm_2_while_body_363136*,
cond$R"
 forward_lstm_2_while_cond_363135*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_2/whileг
?forward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2A
?forward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape­
1forward_lstm_2/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_2/while:output:3Hforward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype023
1forward_lstm_2/TensorArrayV2Stack/TensorListStack
$forward_lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2&
$forward_lstm_2/strided_slice_3/stack
&forward_lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_2/strided_slice_3/stack_1
&forward_lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_3/stack_2є
forward_lstm_2/strided_slice_3StridedSlice:forward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0-forward_lstm_2/strided_slice_3/stack:output:0/forward_lstm_2/strided_slice_3/stack_1:output:0/forward_lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2 
forward_lstm_2/strided_slice_3
forward_lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_2/transpose_1/permъ
forward_lstm_2/transpose_1	Transpose:forward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0(forward_lstm_2/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_2/transpose_1
forward_lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_2/runtime
$backward_lstm_2/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$backward_lstm_2/RaggedToTensor/zeros
$backward_lstm_2/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2&
$backward_lstm_2/RaggedToTensor/Const
3backward_lstm_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-backward_lstm_2/RaggedToTensor/Const:output:0inputs-backward_lstm_2/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3backward_lstm_2/RaggedToTensor/RaggedTensorToTensorТ
:backward_lstm_2/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:backward_lstm_2/RaggedNestedRowLengths/strided_slice/stackЦ
<backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1Ц
<backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2Є
4backward_lstm_2/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cbackward_lstm_2/RaggedNestedRowLengths/strided_slice/stack:output:0Ebackward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1:output:0Ebackward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask26
4backward_lstm_2/RaggedNestedRowLengths/strided_sliceЦ
<backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stackг
>backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2@
>backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1Ъ
>backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2А
6backward_lstm_2/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Ebackward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gbackward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gbackward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask28
6backward_lstm_2/RaggedNestedRowLengths/strided_slice_1
*backward_lstm_2/RaggedNestedRowLengths/subSub=backward_lstm_2/RaggedNestedRowLengths/strided_slice:output:0?backward_lstm_2/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2,
*backward_lstm_2/RaggedNestedRowLengths/subЁ
backward_lstm_2/CastCast.backward_lstm_2/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_2/Cast
backward_lstm_2/ShapeShape<backward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_2/Shape
#backward_lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#backward_lstm_2/strided_slice/stack
%backward_lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_2/strided_slice/stack_1
%backward_lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_2/strided_slice/stack_2Т
backward_lstm_2/strided_sliceStridedSlicebackward_lstm_2/Shape:output:0,backward_lstm_2/strided_slice/stack:output:0.backward_lstm_2/strided_slice/stack_1:output:0.backward_lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
backward_lstm_2/strided_slice|
backward_lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_2/zeros/mul/yЌ
backward_lstm_2/zeros/mulMul&backward_lstm_2/strided_slice:output:0$backward_lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros/mul
backward_lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
backward_lstm_2/zeros/Less/yЇ
backward_lstm_2/zeros/LessLessbackward_lstm_2/zeros/mul:z:0%backward_lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros/Less
backward_lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_2/zeros/packed/1У
backward_lstm_2/zeros/packedPack&backward_lstm_2/strided_slice:output:0'backward_lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_2/zeros/packed
backward_lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_2/zeros/ConstЕ
backward_lstm_2/zerosFill%backward_lstm_2/zeros/packed:output:0$backward_lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/zeros
backward_lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_2/zeros_1/mul/yВ
backward_lstm_2/zeros_1/mulMul&backward_lstm_2/strided_slice:output:0&backward_lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros_1/mul
backward_lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_2/zeros_1/Less/yЏ
backward_lstm_2/zeros_1/LessLessbackward_lstm_2/zeros_1/mul:z:0'backward_lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros_1/Less
 backward_lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_2/zeros_1/packed/1Щ
backward_lstm_2/zeros_1/packedPack&backward_lstm_2/strided_slice:output:0)backward_lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_2/zeros_1/packed
backward_lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_2/zeros_1/ConstН
backward_lstm_2/zeros_1Fill'backward_lstm_2/zeros_1/packed:output:0&backward_lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/zeros_1
backward_lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
backward_lstm_2/transpose/permщ
backward_lstm_2/transpose	Transpose<backward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0'backward_lstm_2/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_2/transpose
backward_lstm_2/Shape_1Shapebackward_lstm_2/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_2/Shape_1
%backward_lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_2/strided_slice_1/stack
'backward_lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_1/stack_1
'backward_lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_1/stack_2Ю
backward_lstm_2/strided_slice_1StridedSlice backward_lstm_2/Shape_1:output:0.backward_lstm_2/strided_slice_1/stack:output:00backward_lstm_2/strided_slice_1/stack_1:output:00backward_lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_2/strided_slice_1Ѕ
+backward_lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+backward_lstm_2/TensorArrayV2/element_shapeђ
backward_lstm_2/TensorArrayV2TensorListReserve4backward_lstm_2/TensorArrayV2/element_shape:output:0(backward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
backward_lstm_2/TensorArrayV2
backward_lstm_2/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2 
backward_lstm_2/ReverseV2/axisЪ
backward_lstm_2/ReverseV2	ReverseV2backward_lstm_2/transpose:y:0'backward_lstm_2/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_2/ReverseV2п
Ebackward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2G
Ebackward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeН
7backward_lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"backward_lstm_2/ReverseV2:output:0Nbackward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7backward_lstm_2/TensorArrayUnstack/TensorListFromTensor
%backward_lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_2/strided_slice_2/stack
'backward_lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_2/stack_1
'backward_lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_2/stack_2м
backward_lstm_2/strided_slice_2StridedSlicebackward_lstm_2/transpose:y:0.backward_lstm_2/strided_slice_2/stack:output:00backward_lstm_2/strided_slice_2/stack_1:output:00backward_lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2!
backward_lstm_2/strided_slice_2т
1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp:backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype023
1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpъ
"backward_lstm_2/lstm_cell_8/MatMulMatMul(backward_lstm_2/strided_slice_2:output:09backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"backward_lstm_2/lstm_cell_8/MatMulш
3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp<backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype025
3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpц
$backward_lstm_2/lstm_cell_8/MatMul_1MatMulbackward_lstm_2/zeros:output:0;backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$backward_lstm_2/lstm_cell_8/MatMul_1м
backward_lstm_2/lstm_cell_8/addAddV2,backward_lstm_2/lstm_cell_8/MatMul:product:0.backward_lstm_2/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2!
backward_lstm_2/lstm_cell_8/addс
2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp;backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype024
2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpщ
#backward_lstm_2/lstm_cell_8/BiasAddBiasAdd#backward_lstm_2/lstm_cell_8/add:z:0:backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_2/lstm_cell_8/BiasAdd
+backward_lstm_2/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+backward_lstm_2/lstm_cell_8/split/split_dimЏ
!backward_lstm_2/lstm_cell_8/splitSplit4backward_lstm_2/lstm_cell_8/split/split_dim:output:0,backward_lstm_2/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2#
!backward_lstm_2/lstm_cell_8/splitГ
#backward_lstm_2/lstm_cell_8/SigmoidSigmoid*backward_lstm_2/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_2/lstm_cell_8/SigmoidЗ
%backward_lstm_2/lstm_cell_8/Sigmoid_1Sigmoid*backward_lstm_2/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_2/lstm_cell_8/Sigmoid_1Ш
backward_lstm_2/lstm_cell_8/mulMul)backward_lstm_2/lstm_cell_8/Sigmoid_1:y:0 backward_lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22!
backward_lstm_2/lstm_cell_8/mulЊ
 backward_lstm_2/lstm_cell_8/ReluRelu*backward_lstm_2/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/lstm_cell_8/Reluи
!backward_lstm_2/lstm_cell_8/mul_1Mul'backward_lstm_2/lstm_cell_8/Sigmoid:y:0.backward_lstm_2/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_2/lstm_cell_8/mul_1Э
!backward_lstm_2/lstm_cell_8/add_1AddV2#backward_lstm_2/lstm_cell_8/mul:z:0%backward_lstm_2/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_2/lstm_cell_8/add_1З
%backward_lstm_2/lstm_cell_8/Sigmoid_2Sigmoid*backward_lstm_2/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_2/lstm_cell_8/Sigmoid_2Љ
"backward_lstm_2/lstm_cell_8/Relu_1Relu%backward_lstm_2/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_2/lstm_cell_8/Relu_1м
!backward_lstm_2/lstm_cell_8/mul_2Mul)backward_lstm_2/lstm_cell_8/Sigmoid_2:y:00backward_lstm_2/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_2/lstm_cell_8/mul_2Џ
-backward_lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2/
-backward_lstm_2/TensorArrayV2_1/element_shapeј
backward_lstm_2/TensorArrayV2_1TensorListReserve6backward_lstm_2/TensorArrayV2_1/element_shape:output:0(backward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_2/TensorArrayV2_1n
backward_lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_2/time
%backward_lstm_2/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2'
%backward_lstm_2/Max/reduction_indices
backward_lstm_2/MaxMaxbackward_lstm_2/Cast:y:0.backward_lstm_2/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/Maxp
backward_lstm_2/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_2/sub/y
backward_lstm_2/subSubbackward_lstm_2/Max:output:0backward_lstm_2/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/sub
backward_lstm_2/Sub_1Subbackward_lstm_2/sub:z:0backward_lstm_2/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_2/Sub_1
backward_lstm_2/zeros_like	ZerosLike%backward_lstm_2/lstm_cell_8/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/zeros_like
(backward_lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(backward_lstm_2/while/maximum_iterations
"backward_lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"backward_lstm_2/while/loop_counterљ
backward_lstm_2/whileWhile+backward_lstm_2/while/loop_counter:output:01backward_lstm_2/while/maximum_iterations:output:0backward_lstm_2/time:output:0(backward_lstm_2/TensorArrayV2_1:handle:0backward_lstm_2/zeros_like:y:0backward_lstm_2/zeros:output:0 backward_lstm_2/zeros_1:output:0(backward_lstm_2/strided_slice_1:output:0Gbackward_lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_2/Sub_1:z:0:backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource<backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource;backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *-
body%R#
!backward_lstm_2_while_body_363315*-
cond%R#
!backward_lstm_2_while_cond_363314*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_2/whileе
@backward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2B
@backward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeБ
2backward_lstm_2/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_2/while:output:3Ibackward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype024
2backward_lstm_2/TensorArrayV2Stack/TensorListStackЁ
%backward_lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2'
%backward_lstm_2/strided_slice_3/stack
'backward_lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_2/strided_slice_3/stack_1
'backward_lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_3/stack_2њ
backward_lstm_2/strided_slice_3StridedSlice;backward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0.backward_lstm_2/strided_slice_3/stack:output:00backward_lstm_2/strided_slice_3/stack_1:output:00backward_lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2!
backward_lstm_2/strided_slice_3
 backward_lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_2/transpose_1/permю
backward_lstm_2/transpose_1	Transpose;backward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0)backward_lstm_2/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_2/transpose_1
backward_lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_2/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisР
concatConcatV2'forward_lstm_2/strided_slice_3:output:0(backward_lstm_2/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

IdentityИ
NoOpNoOp3^backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp2^backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp4^backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp^backward_lstm_2/while2^forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp1^forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp3^forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp^forward_lstm_2/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2h
2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp2f
1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp2j
3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp2.
backward_lstm_2/whilebackward_lstm_2/while2f
1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp2d
0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp2h
2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp2,
forward_lstm_2/whileforward_lstm_2/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
њ

G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_366812

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
а
ћ
H__inference_sequential_2_layer_call_and_return_conditional_losses_364002

inputs
inputs_1	)
bidirectional_2_363983:	Ш)
bidirectional_2_363985:	2Ш%
bidirectional_2_363987:	Ш)
bidirectional_2_363989:	Ш)
bidirectional_2_363991:	2Ш%
bidirectional_2_363993:	Ш 
dense_2_363996:d
dense_2_363998:
identityЂ'bidirectional_2/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЊ
'bidirectional_2/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_2_363983bidirectional_2_363985bidirectional_2_363987bidirectional_2_363989bidirectional_2_363991bidirectional_2_363993*
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
GPU 2J 8 *T
fORM
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_3638522)
'bidirectional_2/StatefulPartitionedCallЙ
dense_2/StatefulPartitionedCallStatefulPartitionedCall0bidirectional_2/StatefulPartitionedCall:output:0dense_2_363996dense_2_363998*
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
GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3634372!
dense_2/StatefulPartitionedCall
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp(^bidirectional_2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2R
'bidirectional_2/StatefulPartitionedCall'bidirectional_2/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
§S
Ї
 forward_lstm_2_while_body_364169:
6forward_lstm_2_while_forward_lstm_2_while_loop_counter@
<forward_lstm_2_while_forward_lstm_2_while_maximum_iterations$
 forward_lstm_2_while_placeholder&
"forward_lstm_2_while_placeholder_1&
"forward_lstm_2_while_placeholder_2&
"forward_lstm_2_while_placeholder_39
5forward_lstm_2_while_forward_lstm_2_strided_slice_1_0u
qforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0T
Aforward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0:	ШV
Cforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0:	2ШQ
Bforward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0:	Ш!
forward_lstm_2_while_identity#
forward_lstm_2_while_identity_1#
forward_lstm_2_while_identity_2#
forward_lstm_2_while_identity_3#
forward_lstm_2_while_identity_4#
forward_lstm_2_while_identity_57
3forward_lstm_2_while_forward_lstm_2_strided_slice_1s
oforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensorR
?forward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource:	ШT
Aforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource:	2ШO
@forward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource:	ШЂ7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOpЂ6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOpЂ8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpс
Fforward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2H
Fforward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeЖ
8forward_lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0 forward_lstm_2_while_placeholderOforward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02:
8forward_lstm_2/while/TensorArrayV2Read/TensorListGetItemѓ
6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOpAforward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype028
6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp
'forward_lstm_2/while/lstm_cell_7/MatMulMatMul?forward_lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0>forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_2/while/lstm_cell_7/MatMulљ
8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOpCforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02:
8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpљ
)forward_lstm_2/while/lstm_cell_7/MatMul_1MatMul"forward_lstm_2_while_placeholder_2@forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)forward_lstm_2/while/lstm_cell_7/MatMul_1№
$forward_lstm_2/while/lstm_cell_7/addAddV21forward_lstm_2/while/lstm_cell_7/MatMul:product:03forward_lstm_2/while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$forward_lstm_2/while/lstm_cell_7/addђ
7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOpBforward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype029
7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp§
(forward_lstm_2/while/lstm_cell_7/BiasAddBiasAdd(forward_lstm_2/while/lstm_cell_7/add:z:0?forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_2/while/lstm_cell_7/BiasAddІ
0forward_lstm_2/while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0forward_lstm_2/while/lstm_cell_7/split/split_dimУ
&forward_lstm_2/while/lstm_cell_7/splitSplit9forward_lstm_2/while/lstm_cell_7/split/split_dim:output:01forward_lstm_2/while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2(
&forward_lstm_2/while/lstm_cell_7/splitТ
(forward_lstm_2/while/lstm_cell_7/SigmoidSigmoid/forward_lstm_2/while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_2/while/lstm_cell_7/SigmoidЦ
*forward_lstm_2/while/lstm_cell_7/Sigmoid_1Sigmoid/forward_lstm_2/while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_2/while/lstm_cell_7/Sigmoid_1й
$forward_lstm_2/while/lstm_cell_7/mulMul.forward_lstm_2/while/lstm_cell_7/Sigmoid_1:y:0"forward_lstm_2_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_2/while/lstm_cell_7/mulЙ
%forward_lstm_2/while/lstm_cell_7/ReluRelu/forward_lstm_2/while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_2/while/lstm_cell_7/Reluь
&forward_lstm_2/while/lstm_cell_7/mul_1Mul,forward_lstm_2/while/lstm_cell_7/Sigmoid:y:03forward_lstm_2/while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_2/while/lstm_cell_7/mul_1с
&forward_lstm_2/while/lstm_cell_7/add_1AddV2(forward_lstm_2/while/lstm_cell_7/mul:z:0*forward_lstm_2/while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_2/while/lstm_cell_7/add_1Ц
*forward_lstm_2/while/lstm_cell_7/Sigmoid_2Sigmoid/forward_lstm_2/while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_2/while/lstm_cell_7/Sigmoid_2И
'forward_lstm_2/while/lstm_cell_7/Relu_1Relu*forward_lstm_2/while/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_2/while/lstm_cell_7/Relu_1№
&forward_lstm_2/while/lstm_cell_7/mul_2Mul.forward_lstm_2/while/lstm_cell_7/Sigmoid_2:y:05forward_lstm_2/while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_2/while/lstm_cell_7/mul_2Њ
9forward_lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"forward_lstm_2_while_placeholder_1 forward_lstm_2_while_placeholder*forward_lstm_2/while/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype02;
9forward_lstm_2/while/TensorArrayV2Write/TensorListSetItemz
forward_lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_2/while/add/yЅ
forward_lstm_2/while/addAddV2 forward_lstm_2_while_placeholder#forward_lstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/while/add~
forward_lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_2/while/add_1/yС
forward_lstm_2/while/add_1AddV26forward_lstm_2_while_forward_lstm_2_while_loop_counter%forward_lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/while/add_1Ї
forward_lstm_2/while/IdentityIdentityforward_lstm_2/while/add_1:z:0^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2
forward_lstm_2/while/IdentityЩ
forward_lstm_2/while/Identity_1Identity<forward_lstm_2_while_forward_lstm_2_while_maximum_iterations^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_2/while/Identity_1Љ
forward_lstm_2/while/Identity_2Identityforward_lstm_2/while/add:z:0^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_2/while/Identity_2ж
forward_lstm_2/while/Identity_3IdentityIforward_lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_2/while/Identity_3Ш
forward_lstm_2/while/Identity_4Identity*forward_lstm_2/while/lstm_cell_7/mul_2:z:0^forward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/while/Identity_4Ш
forward_lstm_2/while/Identity_5Identity*forward_lstm_2/while/lstm_cell_7/add_1:z:0^forward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/while/Identity_5І
forward_lstm_2/while/NoOpNoOp8^forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp7^forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp9^forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_2/while/NoOp"l
3forward_lstm_2_while_forward_lstm_2_strided_slice_15forward_lstm_2_while_forward_lstm_2_strided_slice_1_0"G
forward_lstm_2_while_identity&forward_lstm_2/while/Identity:output:0"K
forward_lstm_2_while_identity_1(forward_lstm_2/while/Identity_1:output:0"K
forward_lstm_2_while_identity_2(forward_lstm_2/while/Identity_2:output:0"K
forward_lstm_2_while_identity_3(forward_lstm_2/while/Identity_3:output:0"K
forward_lstm_2_while_identity_4(forward_lstm_2/while/Identity_4:output:0"K
forward_lstm_2_while_identity_5(forward_lstm_2/while/Identity_5:output:0"
@forward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resourceBforward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0"
Aforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resourceCforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0"
?forward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resourceAforward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0"ф
oforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensorqforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2r
7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp2p
6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp2t
8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Ы>
Ч
while_body_366509
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	ШG
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:	2ШB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	ШE
2while_lstm_cell_8_matmul_1_readvariableop_resource:	2Ш@
1while_lstm_cell_8_biasadd_readvariableop_resource:	ШЂ(while/lstm_cell_8/BiasAdd/ReadVariableOpЂ'while/lstm_cell_8/MatMul/ReadVariableOpЂ)while/lstm_cell_8/MatMul_1/ReadVariableOpУ
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
)while/TensorArrayV2Read/TensorListGetItemЦ
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOpд
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/MatMulЬ
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOpН
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/MatMul_1Д
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/addХ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOpС
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/BiasAdd
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_8/split
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Sigmoid
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Sigmoid_1
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/mul
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/ReluА
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/mul_1Ѕ
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/add_1
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Sigmoid_2
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Relu_1Д
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/mul_2п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5л

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
К
Д
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_362572

inputs(
forward_lstm_2_362402:	Ш(
forward_lstm_2_362404:	2Ш$
forward_lstm_2_362406:	Ш)
backward_lstm_2_362562:	Ш)
backward_lstm_2_362564:	2Ш%
backward_lstm_2_362566:	Ш
identityЂ'backward_lstm_2/StatefulPartitionedCallЂ&forward_lstm_2/StatefulPartitionedCallЫ
&forward_lstm_2/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_2_362402forward_lstm_2_362404forward_lstm_2_362406*
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
GPU 2J 8 *S
fNRL
J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_3624012(
&forward_lstm_2/StatefulPartitionedCallб
'backward_lstm_2/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_2_362562backward_lstm_2_362564backward_lstm_2_362566*
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
GPU 2J 8 *T
fORM
K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_3625612)
'backward_lstm_2/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisа
concatConcatV2/forward_lstm_2/StatefulPartitionedCall:output:00backward_lstm_2/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

IdentityЁ
NoOpNoOp(^backward_lstm_2/StatefulPartitionedCall'^forward_lstm_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2R
'backward_lstm_2/StatefulPartitionedCall'backward_lstm_2/StatefulPartitionedCall2P
&forward_lstm_2/StatefulPartitionedCall&forward_lstm_2/StatefulPartitionedCall:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

я
 forward_lstm_2_while_cond_364168:
6forward_lstm_2_while_forward_lstm_2_while_loop_counter@
<forward_lstm_2_while_forward_lstm_2_while_maximum_iterations$
 forward_lstm_2_while_placeholder&
"forward_lstm_2_while_placeholder_1&
"forward_lstm_2_while_placeholder_2&
"forward_lstm_2_while_placeholder_3<
8forward_lstm_2_while_less_forward_lstm_2_strided_slice_1R
Nforward_lstm_2_while_forward_lstm_2_while_cond_364168___redundant_placeholder0R
Nforward_lstm_2_while_forward_lstm_2_while_cond_364168___redundant_placeholder1R
Nforward_lstm_2_while_forward_lstm_2_while_cond_364168___redundant_placeholder2R
Nforward_lstm_2_while_forward_lstm_2_while_cond_364168___redundant_placeholder3!
forward_lstm_2_while_identity
Л
forward_lstm_2/while/LessLess forward_lstm_2_while_placeholder8forward_lstm_2_while_less_forward_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_2/while/Less
forward_lstm_2/while/IdentityIdentityforward_lstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2
forward_lstm_2/while/Identity"G
forward_lstm_2_while_identity&forward_lstm_2/while/Identity:output:0*(
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
у	

0__inference_bidirectional_2_layer_call_fn_364066
inputs_0
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
	unknown_2:	Ш
	unknown_3:	2Ш
	unknown_4:	Ш
identityЂStatefulPartitionedCallБ
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
GPU 2J 8 *T
fORM
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_3629742
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
Ы>
Ч
while_body_366662
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	ШG
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:	2ШB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	ШE
2while_lstm_cell_8_matmul_1_readvariableop_resource:	2Ш@
1while_lstm_cell_8_biasadd_readvariableop_resource:	ШЂ(while/lstm_cell_8/BiasAdd/ReadVariableOpЂ'while/lstm_cell_8/MatMul/ReadVariableOpЂ)while/lstm_cell_8/MatMul_1/ReadVariableOpУ
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
)while/TensorArrayV2Read/TensorListGetItemЦ
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOpд
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/MatMulЬ
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOpН
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/MatMul_1Д
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/addХ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOpС
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/BiasAdd
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_8/split
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Sigmoid
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Sigmoid_1
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/mul
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/ReluА
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/mul_1Ѕ
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/add_1
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Sigmoid_2
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Relu_1Д
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/mul_2п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5л

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Н
0__inference_backward_lstm_2_layer_call_fn_366123

inputs
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
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
GPU 2J 8 *T
fORM
K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_3625612
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
Д
ѕ
,__inference_lstm_cell_8_layer_call_fn_366878

inputs
states_0
states_1
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identity

identity_1

identity_2ЂStatefulPartitionedCallТ
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
GPU 2J 8 *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_3618292
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
П%
м
while_body_361065
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_7_361089_0:	Ш-
while_lstm_cell_7_361091_0:	2Ш)
while_lstm_cell_7_361093_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_7_361089:	Ш+
while_lstm_cell_7_361091:	2Ш'
while_lstm_cell_7_361093:	ШЂ)while/lstm_cell_7/StatefulPartitionedCallУ
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
)while/TensorArrayV2Read/TensorListGetItemл
)while/lstm_cell_7/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_7_361089_0while_lstm_cell_7_361091_0while_lstm_cell_7_361093_0*
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
GPU 2J 8 *P
fKRI
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_3610512+
)while/lstm_cell_7/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_7/StatefulPartitionedCall:output:0*
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
while/Identity_3Ѓ
while/Identity_4Identity2while/lstm_cell_7/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѓ
while/Identity_5Identity2while/lstm_cell_7/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp*^while/lstm_cell_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_7_361089while_lstm_cell_7_361089_0"6
while_lstm_cell_7_361091while_lstm_cell_7_361091_0"6
while_lstm_cell_7_361093while_lstm_cell_7_361093_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2V
)while/lstm_cell_7/StatefulPartitionedCall)while/lstm_cell_7/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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


!backward_lstm_2_while_cond_363754<
8backward_lstm_2_while_backward_lstm_2_while_loop_counterB
>backward_lstm_2_while_backward_lstm_2_while_maximum_iterations%
!backward_lstm_2_while_placeholder'
#backward_lstm_2_while_placeholder_1'
#backward_lstm_2_while_placeholder_2'
#backward_lstm_2_while_placeholder_3'
#backward_lstm_2_while_placeholder_4>
:backward_lstm_2_while_less_backward_lstm_2_strided_slice_1T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_363754___redundant_placeholder0T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_363754___redundant_placeholder1T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_363754___redundant_placeholder2T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_363754___redundant_placeholder3T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_363754___redundant_placeholder4"
backward_lstm_2_while_identity
Р
backward_lstm_2/while/LessLess!backward_lstm_2_while_placeholder:backward_lstm_2_while_less_backward_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_2/while/Less
backward_lstm_2/while/IdentityIdentitybackward_lstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2 
backward_lstm_2/while/Identity"I
backward_lstm_2_while_identity'backward_lstm_2/while/Identity:output:0*(
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
Ў`
т
 forward_lstm_2_while_body_365146:
6forward_lstm_2_while_forward_lstm_2_while_loop_counter@
<forward_lstm_2_while_forward_lstm_2_while_maximum_iterations$
 forward_lstm_2_while_placeholder&
"forward_lstm_2_while_placeholder_1&
"forward_lstm_2_while_placeholder_2&
"forward_lstm_2_while_placeholder_3&
"forward_lstm_2_while_placeholder_49
5forward_lstm_2_while_forward_lstm_2_strided_slice_1_0u
qforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_06
2forward_lstm_2_while_greater_forward_lstm_2_cast_0T
Aforward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0:	ШV
Cforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0:	2ШQ
Bforward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0:	Ш!
forward_lstm_2_while_identity#
forward_lstm_2_while_identity_1#
forward_lstm_2_while_identity_2#
forward_lstm_2_while_identity_3#
forward_lstm_2_while_identity_4#
forward_lstm_2_while_identity_5#
forward_lstm_2_while_identity_67
3forward_lstm_2_while_forward_lstm_2_strided_slice_1s
oforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor4
0forward_lstm_2_while_greater_forward_lstm_2_castR
?forward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource:	ШT
Aforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource:	2ШO
@forward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource:	ШЂ7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOpЂ6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOpЂ8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpс
Fforward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fforward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape­
8forward_lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0 forward_lstm_2_while_placeholderOforward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02:
8forward_lstm_2/while/TensorArrayV2Read/TensorListGetItemЫ
forward_lstm_2/while/GreaterGreater2forward_lstm_2_while_greater_forward_lstm_2_cast_0 forward_lstm_2_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_2/while/Greaterѓ
6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOpAforward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype028
6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp
'forward_lstm_2/while/lstm_cell_7/MatMulMatMul?forward_lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0>forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_2/while/lstm_cell_7/MatMulљ
8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOpCforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02:
8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpљ
)forward_lstm_2/while/lstm_cell_7/MatMul_1MatMul"forward_lstm_2_while_placeholder_3@forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)forward_lstm_2/while/lstm_cell_7/MatMul_1№
$forward_lstm_2/while/lstm_cell_7/addAddV21forward_lstm_2/while/lstm_cell_7/MatMul:product:03forward_lstm_2/while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$forward_lstm_2/while/lstm_cell_7/addђ
7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOpBforward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype029
7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp§
(forward_lstm_2/while/lstm_cell_7/BiasAddBiasAdd(forward_lstm_2/while/lstm_cell_7/add:z:0?forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_2/while/lstm_cell_7/BiasAddІ
0forward_lstm_2/while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0forward_lstm_2/while/lstm_cell_7/split/split_dimУ
&forward_lstm_2/while/lstm_cell_7/splitSplit9forward_lstm_2/while/lstm_cell_7/split/split_dim:output:01forward_lstm_2/while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2(
&forward_lstm_2/while/lstm_cell_7/splitТ
(forward_lstm_2/while/lstm_cell_7/SigmoidSigmoid/forward_lstm_2/while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_2/while/lstm_cell_7/SigmoidЦ
*forward_lstm_2/while/lstm_cell_7/Sigmoid_1Sigmoid/forward_lstm_2/while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_2/while/lstm_cell_7/Sigmoid_1й
$forward_lstm_2/while/lstm_cell_7/mulMul.forward_lstm_2/while/lstm_cell_7/Sigmoid_1:y:0"forward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_2/while/lstm_cell_7/mulЙ
%forward_lstm_2/while/lstm_cell_7/ReluRelu/forward_lstm_2/while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_2/while/lstm_cell_7/Reluь
&forward_lstm_2/while/lstm_cell_7/mul_1Mul,forward_lstm_2/while/lstm_cell_7/Sigmoid:y:03forward_lstm_2/while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_2/while/lstm_cell_7/mul_1с
&forward_lstm_2/while/lstm_cell_7/add_1AddV2(forward_lstm_2/while/lstm_cell_7/mul:z:0*forward_lstm_2/while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_2/while/lstm_cell_7/add_1Ц
*forward_lstm_2/while/lstm_cell_7/Sigmoid_2Sigmoid/forward_lstm_2/while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_2/while/lstm_cell_7/Sigmoid_2И
'forward_lstm_2/while/lstm_cell_7/Relu_1Relu*forward_lstm_2/while/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_2/while/lstm_cell_7/Relu_1№
&forward_lstm_2/while/lstm_cell_7/mul_2Mul.forward_lstm_2/while/lstm_cell_7/Sigmoid_2:y:05forward_lstm_2/while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_2/while/lstm_cell_7/mul_2ш
forward_lstm_2/while/SelectSelect forward_lstm_2/while/Greater:z:0*forward_lstm_2/while/lstm_cell_7/mul_2:z:0"forward_lstm_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/while/Selectь
forward_lstm_2/while/Select_1Select forward_lstm_2/while/Greater:z:0*forward_lstm_2/while/lstm_cell_7/mul_2:z:0"forward_lstm_2_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/while/Select_1ь
forward_lstm_2/while/Select_2Select forward_lstm_2/while/Greater:z:0*forward_lstm_2/while/lstm_cell_7/add_1:z:0"forward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/while/Select_2Є
9forward_lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"forward_lstm_2_while_placeholder_1 forward_lstm_2_while_placeholder$forward_lstm_2/while/Select:output:0*
_output_shapes
: *
element_dtype02;
9forward_lstm_2/while/TensorArrayV2Write/TensorListSetItemz
forward_lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_2/while/add/yЅ
forward_lstm_2/while/addAddV2 forward_lstm_2_while_placeholder#forward_lstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/while/add~
forward_lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_2/while/add_1/yС
forward_lstm_2/while/add_1AddV26forward_lstm_2_while_forward_lstm_2_while_loop_counter%forward_lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/while/add_1Ї
forward_lstm_2/while/IdentityIdentityforward_lstm_2/while/add_1:z:0^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2
forward_lstm_2/while/IdentityЩ
forward_lstm_2/while/Identity_1Identity<forward_lstm_2_while_forward_lstm_2_while_maximum_iterations^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_2/while/Identity_1Љ
forward_lstm_2/while/Identity_2Identityforward_lstm_2/while/add:z:0^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_2/while/Identity_2ж
forward_lstm_2/while/Identity_3IdentityIforward_lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_2/while/Identity_3Т
forward_lstm_2/while/Identity_4Identity$forward_lstm_2/while/Select:output:0^forward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/while/Identity_4Ф
forward_lstm_2/while/Identity_5Identity&forward_lstm_2/while/Select_1:output:0^forward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/while/Identity_5Ф
forward_lstm_2/while/Identity_6Identity&forward_lstm_2/while/Select_2:output:0^forward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/while/Identity_6І
forward_lstm_2/while/NoOpNoOp8^forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp7^forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp9^forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_2/while/NoOp"l
3forward_lstm_2_while_forward_lstm_2_strided_slice_15forward_lstm_2_while_forward_lstm_2_strided_slice_1_0"f
0forward_lstm_2_while_greater_forward_lstm_2_cast2forward_lstm_2_while_greater_forward_lstm_2_cast_0"G
forward_lstm_2_while_identity&forward_lstm_2/while/Identity:output:0"K
forward_lstm_2_while_identity_1(forward_lstm_2/while/Identity_1:output:0"K
forward_lstm_2_while_identity_2(forward_lstm_2/while/Identity_2:output:0"K
forward_lstm_2_while_identity_3(forward_lstm_2/while/Identity_3:output:0"K
forward_lstm_2_while_identity_4(forward_lstm_2/while/Identity_4:output:0"K
forward_lstm_2_while_identity_5(forward_lstm_2/while/Identity_5:output:0"K
forward_lstm_2_while_identity_6(forward_lstm_2/while/Identity_6:output:0"
@forward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resourceBforward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0"
Aforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resourceCforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0"
?forward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resourceAforward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0"ф
oforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensorqforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2r
7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp2p
6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp2t
8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
П%
м
while_body_361697
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_8_361721_0:	Ш-
while_lstm_cell_8_361723_0:	2Ш)
while_lstm_cell_8_361725_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_8_361721:	Ш+
while_lstm_cell_8_361723:	2Ш'
while_lstm_cell_8_361725:	ШЂ)while/lstm_cell_8/StatefulPartitionedCallУ
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
)while/TensorArrayV2Read/TensorListGetItemл
)while/lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_8_361721_0while_lstm_cell_8_361723_0while_lstm_cell_8_361725_0*
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
GPU 2J 8 *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_3616832+
)while/lstm_cell_8/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_8/StatefulPartitionedCall:output:0*
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
while/Identity_3Ѓ
while/Identity_4Identity2while/lstm_cell_8/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѓ
while/Identity_5Identity2while/lstm_cell_8/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp*^while/lstm_cell_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_8_361721while_lstm_cell_8_361721_0"6
while_lstm_cell_8_361723while_lstm_cell_8_361723_0"6
while_lstm_cell_8_361725while_lstm_cell_8_361725_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2V
)while/lstm_cell_8/StatefulPartitionedCall)while/lstm_cell_8/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
П%
м
while_body_361275
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_7_361299_0:	Ш-
while_lstm_cell_7_361301_0:	2Ш)
while_lstm_cell_7_361303_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_7_361299:	Ш+
while_lstm_cell_7_361301:	2Ш'
while_lstm_cell_7_361303:	ШЂ)while/lstm_cell_7/StatefulPartitionedCallУ
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
)while/TensorArrayV2Read/TensorListGetItemл
)while/lstm_cell_7/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_7_361299_0while_lstm_cell_7_361301_0while_lstm_cell_7_361303_0*
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
GPU 2J 8 *P
fKRI
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_3611972+
)while/lstm_cell_7/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_7/StatefulPartitionedCall:output:0*
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
while/Identity_3Ѓ
while/Identity_4Identity2while/lstm_cell_7/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѓ
while/Identity_5Identity2while/lstm_cell_7/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp*^while/lstm_cell_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_7_361299while_lstm_cell_7_361299_0"6
while_lstm_cell_7_361301while_lstm_cell_7_361301_0"6
while_lstm_cell_7_361303while_lstm_cell_7_361303_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2V
)while/lstm_cell_7/StatefulPartitionedCall)while/lstm_cell_7/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
Ў`
т
 forward_lstm_2_while_body_363576:
6forward_lstm_2_while_forward_lstm_2_while_loop_counter@
<forward_lstm_2_while_forward_lstm_2_while_maximum_iterations$
 forward_lstm_2_while_placeholder&
"forward_lstm_2_while_placeholder_1&
"forward_lstm_2_while_placeholder_2&
"forward_lstm_2_while_placeholder_3&
"forward_lstm_2_while_placeholder_49
5forward_lstm_2_while_forward_lstm_2_strided_slice_1_0u
qforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_06
2forward_lstm_2_while_greater_forward_lstm_2_cast_0T
Aforward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0:	ШV
Cforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0:	2ШQ
Bforward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0:	Ш!
forward_lstm_2_while_identity#
forward_lstm_2_while_identity_1#
forward_lstm_2_while_identity_2#
forward_lstm_2_while_identity_3#
forward_lstm_2_while_identity_4#
forward_lstm_2_while_identity_5#
forward_lstm_2_while_identity_67
3forward_lstm_2_while_forward_lstm_2_strided_slice_1s
oforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor4
0forward_lstm_2_while_greater_forward_lstm_2_castR
?forward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource:	ШT
Aforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource:	2ШO
@forward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource:	ШЂ7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOpЂ6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOpЂ8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpс
Fforward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fforward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape­
8forward_lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0 forward_lstm_2_while_placeholderOforward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02:
8forward_lstm_2/while/TensorArrayV2Read/TensorListGetItemЫ
forward_lstm_2/while/GreaterGreater2forward_lstm_2_while_greater_forward_lstm_2_cast_0 forward_lstm_2_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_2/while/Greaterѓ
6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOpAforward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype028
6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp
'forward_lstm_2/while/lstm_cell_7/MatMulMatMul?forward_lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0>forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_2/while/lstm_cell_7/MatMulљ
8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOpCforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02:
8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpљ
)forward_lstm_2/while/lstm_cell_7/MatMul_1MatMul"forward_lstm_2_while_placeholder_3@forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)forward_lstm_2/while/lstm_cell_7/MatMul_1№
$forward_lstm_2/while/lstm_cell_7/addAddV21forward_lstm_2/while/lstm_cell_7/MatMul:product:03forward_lstm_2/while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$forward_lstm_2/while/lstm_cell_7/addђ
7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOpBforward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype029
7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp§
(forward_lstm_2/while/lstm_cell_7/BiasAddBiasAdd(forward_lstm_2/while/lstm_cell_7/add:z:0?forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_2/while/lstm_cell_7/BiasAddІ
0forward_lstm_2/while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0forward_lstm_2/while/lstm_cell_7/split/split_dimУ
&forward_lstm_2/while/lstm_cell_7/splitSplit9forward_lstm_2/while/lstm_cell_7/split/split_dim:output:01forward_lstm_2/while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2(
&forward_lstm_2/while/lstm_cell_7/splitТ
(forward_lstm_2/while/lstm_cell_7/SigmoidSigmoid/forward_lstm_2/while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_2/while/lstm_cell_7/SigmoidЦ
*forward_lstm_2/while/lstm_cell_7/Sigmoid_1Sigmoid/forward_lstm_2/while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_2/while/lstm_cell_7/Sigmoid_1й
$forward_lstm_2/while/lstm_cell_7/mulMul.forward_lstm_2/while/lstm_cell_7/Sigmoid_1:y:0"forward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_2/while/lstm_cell_7/mulЙ
%forward_lstm_2/while/lstm_cell_7/ReluRelu/forward_lstm_2/while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_2/while/lstm_cell_7/Reluь
&forward_lstm_2/while/lstm_cell_7/mul_1Mul,forward_lstm_2/while/lstm_cell_7/Sigmoid:y:03forward_lstm_2/while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_2/while/lstm_cell_7/mul_1с
&forward_lstm_2/while/lstm_cell_7/add_1AddV2(forward_lstm_2/while/lstm_cell_7/mul:z:0*forward_lstm_2/while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_2/while/lstm_cell_7/add_1Ц
*forward_lstm_2/while/lstm_cell_7/Sigmoid_2Sigmoid/forward_lstm_2/while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_2/while/lstm_cell_7/Sigmoid_2И
'forward_lstm_2/while/lstm_cell_7/Relu_1Relu*forward_lstm_2/while/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_2/while/lstm_cell_7/Relu_1№
&forward_lstm_2/while/lstm_cell_7/mul_2Mul.forward_lstm_2/while/lstm_cell_7/Sigmoid_2:y:05forward_lstm_2/while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_2/while/lstm_cell_7/mul_2ш
forward_lstm_2/while/SelectSelect forward_lstm_2/while/Greater:z:0*forward_lstm_2/while/lstm_cell_7/mul_2:z:0"forward_lstm_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/while/Selectь
forward_lstm_2/while/Select_1Select forward_lstm_2/while/Greater:z:0*forward_lstm_2/while/lstm_cell_7/mul_2:z:0"forward_lstm_2_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/while/Select_1ь
forward_lstm_2/while/Select_2Select forward_lstm_2/while/Greater:z:0*forward_lstm_2/while/lstm_cell_7/add_1:z:0"forward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/while/Select_2Є
9forward_lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"forward_lstm_2_while_placeholder_1 forward_lstm_2_while_placeholder$forward_lstm_2/while/Select:output:0*
_output_shapes
: *
element_dtype02;
9forward_lstm_2/while/TensorArrayV2Write/TensorListSetItemz
forward_lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_2/while/add/yЅ
forward_lstm_2/while/addAddV2 forward_lstm_2_while_placeholder#forward_lstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/while/add~
forward_lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_2/while/add_1/yС
forward_lstm_2/while/add_1AddV26forward_lstm_2_while_forward_lstm_2_while_loop_counter%forward_lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/while/add_1Ї
forward_lstm_2/while/IdentityIdentityforward_lstm_2/while/add_1:z:0^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2
forward_lstm_2/while/IdentityЩ
forward_lstm_2/while/Identity_1Identity<forward_lstm_2_while_forward_lstm_2_while_maximum_iterations^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_2/while/Identity_1Љ
forward_lstm_2/while/Identity_2Identityforward_lstm_2/while/add:z:0^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_2/while/Identity_2ж
forward_lstm_2/while/Identity_3IdentityIforward_lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_2/while/Identity_3Т
forward_lstm_2/while/Identity_4Identity$forward_lstm_2/while/Select:output:0^forward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/while/Identity_4Ф
forward_lstm_2/while/Identity_5Identity&forward_lstm_2/while/Select_1:output:0^forward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/while/Identity_5Ф
forward_lstm_2/while/Identity_6Identity&forward_lstm_2/while/Select_2:output:0^forward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/while/Identity_6І
forward_lstm_2/while/NoOpNoOp8^forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp7^forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp9^forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_2/while/NoOp"l
3forward_lstm_2_while_forward_lstm_2_strided_slice_15forward_lstm_2_while_forward_lstm_2_strided_slice_1_0"f
0forward_lstm_2_while_greater_forward_lstm_2_cast2forward_lstm_2_while_greater_forward_lstm_2_cast_0"G
forward_lstm_2_while_identity&forward_lstm_2/while/Identity:output:0"K
forward_lstm_2_while_identity_1(forward_lstm_2/while/Identity_1:output:0"K
forward_lstm_2_while_identity_2(forward_lstm_2/while/Identity_2:output:0"K
forward_lstm_2_while_identity_3(forward_lstm_2/while/Identity_3:output:0"K
forward_lstm_2_while_identity_4(forward_lstm_2/while/Identity_4:output:0"K
forward_lstm_2_while_identity_5(forward_lstm_2/while/Identity_5:output:0"K
forward_lstm_2_while_identity_6(forward_lstm_2/while/Identity_6:output:0"
@forward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resourceBforward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0"
Aforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resourceCforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0"
?forward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resourceAforward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0"ф
oforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensorqforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2r
7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp2p
6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp2t
8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Д
ѕ
,__inference_lstm_cell_7_layer_call_fn_366763

inputs
states_0
states_1
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identity

identity_1

identity_2ЂStatefulPartitionedCallТ
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
GPU 2J 8 *P
fKRI
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_3610512
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
њ[

J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_362926

inputs=
*lstm_cell_7_matmul_readvariableop_resource:	Ш?
,lstm_cell_7_matmul_1_readvariableop_resource:	2Ш:
+lstm_cell_7_biasadd_readvariableop_resource:	Ш
identityЂ"lstm_cell_7/BiasAdd/ReadVariableOpЂ!lstm_cell_7/MatMul/ReadVariableOpЂ#lstm_cell_7/MatMul_1/ReadVariableOpЂwhileD
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
strided_slice_2В
!lstm_cell_7/MatMul/ReadVariableOpReadVariableOp*lstm_cell_7_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02#
!lstm_cell_7/MatMul/ReadVariableOpЊ
lstm_cell_7/MatMulMatMulstrided_slice_2:output:0)lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/MatMulИ
#lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02%
#lstm_cell_7/MatMul_1/ReadVariableOpІ
lstm_cell_7/MatMul_1MatMulzeros:output:0+lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/MatMul_1
lstm_cell_7/addAddV2lstm_cell_7/MatMul:product:0lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/addБ
"lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02$
"lstm_cell_7/BiasAdd/ReadVariableOpЉ
lstm_cell_7/BiasAddBiasAddlstm_cell_7/add:z:0*lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/BiasAdd|
lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_7/split/split_dimя
lstm_cell_7/splitSplit$lstm_cell_7/split/split_dim:output:0lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_7/split
lstm_cell_7/SigmoidSigmoidlstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Sigmoid
lstm_cell_7/Sigmoid_1Sigmoidlstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Sigmoid_1
lstm_cell_7/mulMullstm_cell_7/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/mulz
lstm_cell_7/ReluRelulstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Relu
lstm_cell_7/mul_1Mullstm_cell_7/Sigmoid:y:0lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/mul_1
lstm_cell_7/add_1AddV2lstm_cell_7/mul:z:0lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/add_1
lstm_cell_7/Sigmoid_2Sigmoidlstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Sigmoid_2y
lstm_cell_7/Relu_1Relulstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Relu_1
lstm_cell_7/mul_2Mullstm_cell_7/Sigmoid_2:y:0 lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_7_matmul_readvariableop_resource,lstm_cell_7_matmul_1_readvariableop_resource+lstm_cell_7_biasadd_readvariableop_resource*
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
bodyR
while_body_362842*
condR
while_cond_362841*K
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

IdentityХ
NoOpNoOp#^lstm_cell_7/BiasAdd/ReadVariableOp"^lstm_cell_7/MatMul/ReadVariableOp$^lstm_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2H
"lstm_cell_7/BiasAdd/ReadVariableOp"lstm_cell_7/BiasAdd/ReadVariableOp2F
!lstm_cell_7/MatMul/ReadVariableOp!lstm_cell_7/MatMul/ReadVariableOp2J
#lstm_cell_7/MatMul_1/ReadVariableOp#lstm_cell_7/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


!backward_lstm_2_while_cond_365324<
8backward_lstm_2_while_backward_lstm_2_while_loop_counterB
>backward_lstm_2_while_backward_lstm_2_while_maximum_iterations%
!backward_lstm_2_while_placeholder'
#backward_lstm_2_while_placeholder_1'
#backward_lstm_2_while_placeholder_2'
#backward_lstm_2_while_placeholder_3'
#backward_lstm_2_while_placeholder_4>
:backward_lstm_2_while_less_backward_lstm_2_strided_slice_1T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_365324___redundant_placeholder0T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_365324___redundant_placeholder1T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_365324___redundant_placeholder2T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_365324___redundant_placeholder3T
Pbackward_lstm_2_while_backward_lstm_2_while_cond_365324___redundant_placeholder4"
backward_lstm_2_while_identity
Р
backward_lstm_2/while/LessLess!backward_lstm_2_while_placeholder:backward_lstm_2_while_less_backward_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_2/while/Less
backward_lstm_2/while/IdentityIdentitybackward_lstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2 
backward_lstm_2/while/Identity"I
backward_lstm_2_while_identity'backward_lstm_2/while/Identity:output:0*(
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
Ы>
Ч
while_body_362669
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	ШG
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:	2ШB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	ШE
2while_lstm_cell_8_matmul_1_readvariableop_resource:	2Ш@
1while_lstm_cell_8_biasadd_readvariableop_resource:	ШЂ(while/lstm_cell_8/BiasAdd/ReadVariableOpЂ'while/lstm_cell_8/MatMul/ReadVariableOpЂ)while/lstm_cell_8/MatMul_1/ReadVariableOpУ
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
)while/TensorArrayV2Read/TensorListGetItemЦ
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOpд
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/MatMulЬ
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOpН
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/MatMul_1Д
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/addХ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOpС
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_8/BiasAdd
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_8/split
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Sigmoid
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Sigmoid_1
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/mul
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/ReluА
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/mul_1Ѕ
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/add_1
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Sigmoid_2
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/Relu_1Д
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_8/mul_2п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5л

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Д
ѕ
,__inference_lstm_cell_7_layer_call_fn_366780

inputs
states_0
states_1
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identity

identity_1

identity_2ЂStatefulPartitionedCallТ
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
GPU 2J 8 *P
fKRI
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_3611972
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
H

K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_361766

inputs%
lstm_cell_8_361684:	Ш%
lstm_cell_8_361686:	2Ш!
lstm_cell_8_361688:	Ш
identityЂ#lstm_cell_8/StatefulPartitionedCallЂwhileD
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
strided_slice_2
#lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_8_361684lstm_cell_8_361686lstm_cell_8_361688*
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
GPU 2J 8 *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_3616832%
#lstm_cell_8/StatefulPartitionedCall
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
while/loop_counterН
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_8_361684lstm_cell_8_361686lstm_cell_8_361688*
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
bodyR
while_body_361697*
condR
while_cond_361696*K
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

Identity|
NoOpNoOp$^lstm_cell_8/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2J
#lstm_cell_8/StatefulPartitionedCall#lstm_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њ

G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_366844

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
ћ
ы
 forward_lstm_2_while_cond_364787:
6forward_lstm_2_while_forward_lstm_2_while_loop_counter@
<forward_lstm_2_while_forward_lstm_2_while_maximum_iterations$
 forward_lstm_2_while_placeholder&
"forward_lstm_2_while_placeholder_1&
"forward_lstm_2_while_placeholder_2&
"forward_lstm_2_while_placeholder_3&
"forward_lstm_2_while_placeholder_4<
8forward_lstm_2_while_less_forward_lstm_2_strided_slice_1R
Nforward_lstm_2_while_forward_lstm_2_while_cond_364787___redundant_placeholder0R
Nforward_lstm_2_while_forward_lstm_2_while_cond_364787___redundant_placeholder1R
Nforward_lstm_2_while_forward_lstm_2_while_cond_364787___redundant_placeholder2R
Nforward_lstm_2_while_forward_lstm_2_while_cond_364787___redundant_placeholder3R
Nforward_lstm_2_while_forward_lstm_2_while_cond_364787___redundant_placeholder4!
forward_lstm_2_while_identity
Л
forward_lstm_2/while/LessLess forward_lstm_2_while_placeholder8forward_lstm_2_while_less_forward_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_2/while/Less
forward_lstm_2/while/IdentityIdentityforward_lstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2
forward_lstm_2/while/Identity"G
forward_lstm_2_while_identity&forward_lstm_2/while/Identity:output:0*(
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
њ

G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_366910

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
У

=sequential_2_bidirectional_2_forward_lstm_2_while_cond_360692t
psequential_2_bidirectional_2_forward_lstm_2_while_sequential_2_bidirectional_2_forward_lstm_2_while_loop_counterz
vsequential_2_bidirectional_2_forward_lstm_2_while_sequential_2_bidirectional_2_forward_lstm_2_while_maximum_iterationsA
=sequential_2_bidirectional_2_forward_lstm_2_while_placeholderC
?sequential_2_bidirectional_2_forward_lstm_2_while_placeholder_1C
?sequential_2_bidirectional_2_forward_lstm_2_while_placeholder_2C
?sequential_2_bidirectional_2_forward_lstm_2_while_placeholder_3C
?sequential_2_bidirectional_2_forward_lstm_2_while_placeholder_4v
rsequential_2_bidirectional_2_forward_lstm_2_while_less_sequential_2_bidirectional_2_forward_lstm_2_strided_slice_1
sequential_2_bidirectional_2_forward_lstm_2_while_sequential_2_bidirectional_2_forward_lstm_2_while_cond_360692___redundant_placeholder0
sequential_2_bidirectional_2_forward_lstm_2_while_sequential_2_bidirectional_2_forward_lstm_2_while_cond_360692___redundant_placeholder1
sequential_2_bidirectional_2_forward_lstm_2_while_sequential_2_bidirectional_2_forward_lstm_2_while_cond_360692___redundant_placeholder2
sequential_2_bidirectional_2_forward_lstm_2_while_sequential_2_bidirectional_2_forward_lstm_2_while_cond_360692___redundant_placeholder3
sequential_2_bidirectional_2_forward_lstm_2_while_sequential_2_bidirectional_2_forward_lstm_2_while_cond_360692___redundant_placeholder4>
:sequential_2_bidirectional_2_forward_lstm_2_while_identity
Ь
6sequential_2/bidirectional_2/forward_lstm_2/while/LessLess=sequential_2_bidirectional_2_forward_lstm_2_while_placeholderrsequential_2_bidirectional_2_forward_lstm_2_while_less_sequential_2_bidirectional_2_forward_lstm_2_strided_slice_1*
T0*
_output_shapes
: 28
6sequential_2/bidirectional_2/forward_lstm_2/while/Lessс
:sequential_2/bidirectional_2/forward_lstm_2/while/IdentityIdentity:sequential_2/bidirectional_2/forward_lstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2<
:sequential_2/bidirectional_2/forward_lstm_2/while/Identity"
:sequential_2_bidirectional_2_forward_lstm_2_while_identityCsequential_2/bidirectional_2/forward_lstm_2/while/Identity:output:0*(
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
е
У
while_cond_362476
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_362476___redundant_placeholder04
0while_while_cond_362476___redundant_placeholder14
0while_while_cond_362476___redundant_placeholder24
0while_while_cond_362476___redundant_placeholder3
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
І

Ё
0__inference_bidirectional_2_layer_call_fn_364084

inputs
inputs_1	
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
	unknown_2:	Ш
	unknown_3:	2Ш
	unknown_4:	Ш
identityЂStatefulPartitionedCallК
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
GPU 2J 8 *T
fORM
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_3634122
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
н]

K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_366440
inputs_0=
*lstm_cell_8_matmul_readvariableop_resource:	Ш?
,lstm_cell_8_matmul_1_readvariableop_resource:	2Ш:
+lstm_cell_8_biasadd_readvariableop_resource:	Ш
identityЂ"lstm_cell_8/BiasAdd/ReadVariableOpЂ!lstm_cell_8/MatMul/ReadVariableOpЂ#lstm_cell_8/MatMul_1/ReadVariableOpЂwhileF
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
strided_slice_2В
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOpЊ
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/MatMulИ
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOpІ
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/MatMul_1
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/addБ
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOpЉ
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/BiasAdd|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimя
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_8/split
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Sigmoid
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Sigmoid_1
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Relu
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/mul_1
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/add_1
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Relu_1
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
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
bodyR
while_body_366356*
condR
while_cond_366355*K
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

IdentityХ
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
эa

!backward_lstm_2_while_body_363755<
8backward_lstm_2_while_backward_lstm_2_while_loop_counterB
>backward_lstm_2_while_backward_lstm_2_while_maximum_iterations%
!backward_lstm_2_while_placeholder'
#backward_lstm_2_while_placeholder_1'
#backward_lstm_2_while_placeholder_2'
#backward_lstm_2_while_placeholder_3'
#backward_lstm_2_while_placeholder_4;
7backward_lstm_2_while_backward_lstm_2_strided_slice_1_0w
sbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_06
2backward_lstm_2_while_less_backward_lstm_2_sub_1_0U
Bbackward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0:	ШW
Dbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0:	2ШR
Cbackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ш"
backward_lstm_2_while_identity$
 backward_lstm_2_while_identity_1$
 backward_lstm_2_while_identity_2$
 backward_lstm_2_while_identity_3$
 backward_lstm_2_while_identity_4$
 backward_lstm_2_while_identity_5$
 backward_lstm_2_while_identity_69
5backward_lstm_2_while_backward_lstm_2_strided_slice_1u
qbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor4
0backward_lstm_2_while_less_backward_lstm_2_sub_1S
@backward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource:	ШU
Bbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource:	2ШP
Abackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource:	ШЂ8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOpЂ7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOpЂ9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOpу
Gbackward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gbackward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
9backward_lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0!backward_lstm_2_while_placeholderPbackward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02;
9backward_lstm_2/while/TensorArrayV2Read/TensorListGetItemХ
backward_lstm_2/while/LessLess2backward_lstm_2_while_less_backward_lstm_2_sub_1_0!backward_lstm_2_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_2/while/Lessі
7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOpBbackward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype029
7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp
(backward_lstm_2/while/lstm_cell_8/MatMulMatMul@backward_lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0?backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_2/while/lstm_cell_8/MatMulќ
9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOpDbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02;
9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp§
*backward_lstm_2/while/lstm_cell_8/MatMul_1MatMul#backward_lstm_2_while_placeholder_3Abackward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2,
*backward_lstm_2/while/lstm_cell_8/MatMul_1є
%backward_lstm_2/while/lstm_cell_8/addAddV22backward_lstm_2/while/lstm_cell_8/MatMul:product:04backward_lstm_2/while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%backward_lstm_2/while/lstm_cell_8/addѕ
8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOpCbackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02:
8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp
)backward_lstm_2/while/lstm_cell_8/BiasAddBiasAdd)backward_lstm_2/while/lstm_cell_8/add:z:0@backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_2/while/lstm_cell_8/BiasAddЈ
1backward_lstm_2/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1backward_lstm_2/while/lstm_cell_8/split/split_dimЧ
'backward_lstm_2/while/lstm_cell_8/splitSplit:backward_lstm_2/while/lstm_cell_8/split/split_dim:output:02backward_lstm_2/while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2)
'backward_lstm_2/while/lstm_cell_8/splitХ
)backward_lstm_2/while/lstm_cell_8/SigmoidSigmoid0backward_lstm_2/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_2/while/lstm_cell_8/SigmoidЩ
+backward_lstm_2/while/lstm_cell_8/Sigmoid_1Sigmoid0backward_lstm_2/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_2/while/lstm_cell_8/Sigmoid_1н
%backward_lstm_2/while/lstm_cell_8/mulMul/backward_lstm_2/while/lstm_cell_8/Sigmoid_1:y:0#backward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_2/while/lstm_cell_8/mulМ
&backward_lstm_2/while/lstm_cell_8/ReluRelu0backward_lstm_2/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_2/while/lstm_cell_8/Relu№
'backward_lstm_2/while/lstm_cell_8/mul_1Mul-backward_lstm_2/while/lstm_cell_8/Sigmoid:y:04backward_lstm_2/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_2/while/lstm_cell_8/mul_1х
'backward_lstm_2/while/lstm_cell_8/add_1AddV2)backward_lstm_2/while/lstm_cell_8/mul:z:0+backward_lstm_2/while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_2/while/lstm_cell_8/add_1Щ
+backward_lstm_2/while/lstm_cell_8/Sigmoid_2Sigmoid0backward_lstm_2/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_2/while/lstm_cell_8/Sigmoid_2Л
(backward_lstm_2/while/lstm_cell_8/Relu_1Relu+backward_lstm_2/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_2/while/lstm_cell_8/Relu_1є
'backward_lstm_2/while/lstm_cell_8/mul_2Mul/backward_lstm_2/while/lstm_cell_8/Sigmoid_2:y:06backward_lstm_2/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_2/while/lstm_cell_8/mul_2ъ
backward_lstm_2/while/SelectSelectbackward_lstm_2/while/Less:z:0+backward_lstm_2/while/lstm_cell_8/mul_2:z:0#backward_lstm_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/while/Selectю
backward_lstm_2/while/Select_1Selectbackward_lstm_2/while/Less:z:0+backward_lstm_2/while/lstm_cell_8/mul_2:z:0#backward_lstm_2_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_2/while/Select_1ю
backward_lstm_2/while/Select_2Selectbackward_lstm_2/while/Less:z:0+backward_lstm_2/while/lstm_cell_8/add_1:z:0#backward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_2/while/Select_2Љ
:backward_lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#backward_lstm_2_while_placeholder_1!backward_lstm_2_while_placeholder%backward_lstm_2/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:backward_lstm_2/while/TensorArrayV2Write/TensorListSetItem|
backward_lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_2/while/add/yЉ
backward_lstm_2/while/addAddV2!backward_lstm_2_while_placeholder$backward_lstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/while/add
backward_lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_2/while/add_1/yЦ
backward_lstm_2/while/add_1AddV28backward_lstm_2_while_backward_lstm_2_while_loop_counter&backward_lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/while/add_1Ћ
backward_lstm_2/while/IdentityIdentitybackward_lstm_2/while/add_1:z:0^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2 
backward_lstm_2/while/IdentityЮ
 backward_lstm_2/while/Identity_1Identity>backward_lstm_2_while_backward_lstm_2_while_maximum_iterations^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_2/while/Identity_1­
 backward_lstm_2/while/Identity_2Identitybackward_lstm_2/while/add:z:0^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_2/while/Identity_2к
 backward_lstm_2/while/Identity_3IdentityJbackward_lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_2/while/Identity_3Ц
 backward_lstm_2/while/Identity_4Identity%backward_lstm_2/while/Select:output:0^backward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/while/Identity_4Ш
 backward_lstm_2/while/Identity_5Identity'backward_lstm_2/while/Select_1:output:0^backward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/while/Identity_5Ш
 backward_lstm_2/while/Identity_6Identity'backward_lstm_2/while/Select_2:output:0^backward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/while/Identity_6Ћ
backward_lstm_2/while/NoOpNoOp9^backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp8^backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp:^backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_2/while/NoOp"p
5backward_lstm_2_while_backward_lstm_2_strided_slice_17backward_lstm_2_while_backward_lstm_2_strided_slice_1_0"I
backward_lstm_2_while_identity'backward_lstm_2/while/Identity:output:0"M
 backward_lstm_2_while_identity_1)backward_lstm_2/while/Identity_1:output:0"M
 backward_lstm_2_while_identity_2)backward_lstm_2/while/Identity_2:output:0"M
 backward_lstm_2_while_identity_3)backward_lstm_2/while/Identity_3:output:0"M
 backward_lstm_2_while_identity_4)backward_lstm_2/while/Identity_4:output:0"M
 backward_lstm_2_while_identity_5)backward_lstm_2/while/Identity_5:output:0"M
 backward_lstm_2_while_identity_6)backward_lstm_2/while/Identity_6:output:0"f
0backward_lstm_2_while_less_backward_lstm_2_sub_12backward_lstm_2_while_less_backward_lstm_2_sub_1_0"
Abackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resourceCbackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0"
Bbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resourceDbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0"
@backward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resourceBbackward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0"ш
qbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensorsbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2t
8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp2r
7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp2v
9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Ў`
т
 forward_lstm_2_while_body_364788:
6forward_lstm_2_while_forward_lstm_2_while_loop_counter@
<forward_lstm_2_while_forward_lstm_2_while_maximum_iterations$
 forward_lstm_2_while_placeholder&
"forward_lstm_2_while_placeholder_1&
"forward_lstm_2_while_placeholder_2&
"forward_lstm_2_while_placeholder_3&
"forward_lstm_2_while_placeholder_49
5forward_lstm_2_while_forward_lstm_2_strided_slice_1_0u
qforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_06
2forward_lstm_2_while_greater_forward_lstm_2_cast_0T
Aforward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0:	ШV
Cforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0:	2ШQ
Bforward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0:	Ш!
forward_lstm_2_while_identity#
forward_lstm_2_while_identity_1#
forward_lstm_2_while_identity_2#
forward_lstm_2_while_identity_3#
forward_lstm_2_while_identity_4#
forward_lstm_2_while_identity_5#
forward_lstm_2_while_identity_67
3forward_lstm_2_while_forward_lstm_2_strided_slice_1s
oforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor4
0forward_lstm_2_while_greater_forward_lstm_2_castR
?forward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource:	ШT
Aforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource:	2ШO
@forward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource:	ШЂ7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOpЂ6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOpЂ8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpс
Fforward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fforward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape­
8forward_lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0 forward_lstm_2_while_placeholderOforward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02:
8forward_lstm_2/while/TensorArrayV2Read/TensorListGetItemЫ
forward_lstm_2/while/GreaterGreater2forward_lstm_2_while_greater_forward_lstm_2_cast_0 forward_lstm_2_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_2/while/Greaterѓ
6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOpAforward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype028
6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp
'forward_lstm_2/while/lstm_cell_7/MatMulMatMul?forward_lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0>forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2)
'forward_lstm_2/while/lstm_cell_7/MatMulљ
8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOpCforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02:
8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOpљ
)forward_lstm_2/while/lstm_cell_7/MatMul_1MatMul"forward_lstm_2_while_placeholder_3@forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)forward_lstm_2/while/lstm_cell_7/MatMul_1№
$forward_lstm_2/while/lstm_cell_7/addAddV21forward_lstm_2/while/lstm_cell_7/MatMul:product:03forward_lstm_2/while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$forward_lstm_2/while/lstm_cell_7/addђ
7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOpBforward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype029
7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp§
(forward_lstm_2/while/lstm_cell_7/BiasAddBiasAdd(forward_lstm_2/while/lstm_cell_7/add:z:0?forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(forward_lstm_2/while/lstm_cell_7/BiasAddІ
0forward_lstm_2/while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0forward_lstm_2/while/lstm_cell_7/split/split_dimУ
&forward_lstm_2/while/lstm_cell_7/splitSplit9forward_lstm_2/while/lstm_cell_7/split/split_dim:output:01forward_lstm_2/while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2(
&forward_lstm_2/while/lstm_cell_7/splitТ
(forward_lstm_2/while/lstm_cell_7/SigmoidSigmoid/forward_lstm_2/while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(forward_lstm_2/while/lstm_cell_7/SigmoidЦ
*forward_lstm_2/while/lstm_cell_7/Sigmoid_1Sigmoid/forward_lstm_2/while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_2/while/lstm_cell_7/Sigmoid_1й
$forward_lstm_2/while/lstm_cell_7/mulMul.forward_lstm_2/while/lstm_cell_7/Sigmoid_1:y:0"forward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_2/while/lstm_cell_7/mulЙ
%forward_lstm_2/while/lstm_cell_7/ReluRelu/forward_lstm_2/while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22'
%forward_lstm_2/while/lstm_cell_7/Reluь
&forward_lstm_2/while/lstm_cell_7/mul_1Mul,forward_lstm_2/while/lstm_cell_7/Sigmoid:y:03forward_lstm_2/while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_2/while/lstm_cell_7/mul_1с
&forward_lstm_2/while/lstm_cell_7/add_1AddV2(forward_lstm_2/while/lstm_cell_7/mul:z:0*forward_lstm_2/while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_2/while/lstm_cell_7/add_1Ц
*forward_lstm_2/while/lstm_cell_7/Sigmoid_2Sigmoid/forward_lstm_2/while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22,
*forward_lstm_2/while/lstm_cell_7/Sigmoid_2И
'forward_lstm_2/while/lstm_cell_7/Relu_1Relu*forward_lstm_2/while/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'forward_lstm_2/while/lstm_cell_7/Relu_1№
&forward_lstm_2/while/lstm_cell_7/mul_2Mul.forward_lstm_2/while/lstm_cell_7/Sigmoid_2:y:05forward_lstm_2/while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&forward_lstm_2/while/lstm_cell_7/mul_2ш
forward_lstm_2/while/SelectSelect forward_lstm_2/while/Greater:z:0*forward_lstm_2/while/lstm_cell_7/mul_2:z:0"forward_lstm_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/while/Selectь
forward_lstm_2/while/Select_1Select forward_lstm_2/while/Greater:z:0*forward_lstm_2/while/lstm_cell_7/mul_2:z:0"forward_lstm_2_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/while/Select_1ь
forward_lstm_2/while/Select_2Select forward_lstm_2/while/Greater:z:0*forward_lstm_2/while/lstm_cell_7/add_1:z:0"forward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/while/Select_2Є
9forward_lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"forward_lstm_2_while_placeholder_1 forward_lstm_2_while_placeholder$forward_lstm_2/while/Select:output:0*
_output_shapes
: *
element_dtype02;
9forward_lstm_2/while/TensorArrayV2Write/TensorListSetItemz
forward_lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_2/while/add/yЅ
forward_lstm_2/while/addAddV2 forward_lstm_2_while_placeholder#forward_lstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/while/add~
forward_lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_2/while/add_1/yС
forward_lstm_2/while/add_1AddV26forward_lstm_2_while_forward_lstm_2_while_loop_counter%forward_lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/while/add_1Ї
forward_lstm_2/while/IdentityIdentityforward_lstm_2/while/add_1:z:0^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2
forward_lstm_2/while/IdentityЩ
forward_lstm_2/while/Identity_1Identity<forward_lstm_2_while_forward_lstm_2_while_maximum_iterations^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_2/while/Identity_1Љ
forward_lstm_2/while/Identity_2Identityforward_lstm_2/while/add:z:0^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_2/while/Identity_2ж
forward_lstm_2/while/Identity_3IdentityIforward_lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2!
forward_lstm_2/while/Identity_3Т
forward_lstm_2/while/Identity_4Identity$forward_lstm_2/while/Select:output:0^forward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/while/Identity_4Ф
forward_lstm_2/while/Identity_5Identity&forward_lstm_2/while/Select_1:output:0^forward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/while/Identity_5Ф
forward_lstm_2/while/Identity_6Identity&forward_lstm_2/while/Select_2:output:0^forward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/while/Identity_6І
forward_lstm_2/while/NoOpNoOp8^forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp7^forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp9^forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_2/while/NoOp"l
3forward_lstm_2_while_forward_lstm_2_strided_slice_15forward_lstm_2_while_forward_lstm_2_strided_slice_1_0"f
0forward_lstm_2_while_greater_forward_lstm_2_cast2forward_lstm_2_while_greater_forward_lstm_2_cast_0"G
forward_lstm_2_while_identity&forward_lstm_2/while/Identity:output:0"K
forward_lstm_2_while_identity_1(forward_lstm_2/while/Identity_1:output:0"K
forward_lstm_2_while_identity_2(forward_lstm_2/while/Identity_2:output:0"K
forward_lstm_2_while_identity_3(forward_lstm_2/while/Identity_3:output:0"K
forward_lstm_2_while_identity_4(forward_lstm_2/while/Identity_4:output:0"K
forward_lstm_2_while_identity_5(forward_lstm_2/while/Identity_5:output:0"K
forward_lstm_2_while_identity_6(forward_lstm_2/while/Identity_6:output:0"
@forward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resourceBforward_lstm_2_while_lstm_cell_7_biasadd_readvariableop_resource_0"
Aforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resourceCforward_lstm_2_while_lstm_cell_7_matmul_1_readvariableop_resource_0"
?forward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resourceAforward_lstm_2_while_lstm_cell_7_matmul_readvariableop_resource_0"ф
oforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensorqforward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2r
7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp7forward_lstm_2/while/lstm_cell_7/BiasAdd/ReadVariableOp2p
6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp6forward_lstm_2/while/lstm_cell_7/MatMul/ReadVariableOp2t
8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp8forward_lstm_2/while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
а
ћ
H__inference_sequential_2_layer_call_and_return_conditional_losses_363979

inputs
inputs_1	)
bidirectional_2_363960:	Ш)
bidirectional_2_363962:	2Ш%
bidirectional_2_363964:	Ш)
bidirectional_2_363966:	Ш)
bidirectional_2_363968:	2Ш%
bidirectional_2_363970:	Ш 
dense_2_363973:d
dense_2_363975:
identityЂ'bidirectional_2/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЊ
'bidirectional_2/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_2_363960bidirectional_2_363962bidirectional_2_363964bidirectional_2_363966bidirectional_2_363968bidirectional_2_363970*
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
GPU 2J 8 *T
fORM
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_3634122)
'bidirectional_2/StatefulPartitionedCallЙ
dense_2/StatefulPartitionedCallStatefulPartitionedCall0bidirectional_2/StatefulPartitionedCall:output:0dense_2_363973dense_2_363975*
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
GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3634372!
dense_2/StatefulPartitionedCall
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp(^bidirectional_2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2R
'bidirectional_2/StatefulPartitionedCall'bidirectional_2/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
њ[

J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_362401

inputs=
*lstm_cell_7_matmul_readvariableop_resource:	Ш?
,lstm_cell_7_matmul_1_readvariableop_resource:	2Ш:
+lstm_cell_7_biasadd_readvariableop_resource:	Ш
identityЂ"lstm_cell_7/BiasAdd/ReadVariableOpЂ!lstm_cell_7/MatMul/ReadVariableOpЂ#lstm_cell_7/MatMul_1/ReadVariableOpЂwhileD
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
strided_slice_2В
!lstm_cell_7/MatMul/ReadVariableOpReadVariableOp*lstm_cell_7_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02#
!lstm_cell_7/MatMul/ReadVariableOpЊ
lstm_cell_7/MatMulMatMulstrided_slice_2:output:0)lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/MatMulИ
#lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02%
#lstm_cell_7/MatMul_1/ReadVariableOpІ
lstm_cell_7/MatMul_1MatMulzeros:output:0+lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/MatMul_1
lstm_cell_7/addAddV2lstm_cell_7/MatMul:product:0lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/addБ
"lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02$
"lstm_cell_7/BiasAdd/ReadVariableOpЉ
lstm_cell_7/BiasAddBiasAddlstm_cell_7/add:z:0*lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/BiasAdd|
lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_7/split/split_dimя
lstm_cell_7/splitSplit$lstm_cell_7/split/split_dim:output:0lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_7/split
lstm_cell_7/SigmoidSigmoidlstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Sigmoid
lstm_cell_7/Sigmoid_1Sigmoidlstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Sigmoid_1
lstm_cell_7/mulMullstm_cell_7/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/mulz
lstm_cell_7/ReluRelulstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Relu
lstm_cell_7/mul_1Mullstm_cell_7/Sigmoid:y:0lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/mul_1
lstm_cell_7/add_1AddV2lstm_cell_7/mul:z:0lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/add_1
lstm_cell_7/Sigmoid_2Sigmoidlstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Sigmoid_2y
lstm_cell_7/Relu_1Relulstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Relu_1
lstm_cell_7/mul_2Mullstm_cell_7/Sigmoid_2:y:0 lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_7_matmul_readvariableop_resource,lstm_cell_7_matmul_1_readvariableop_resource+lstm_cell_7_biasadd_readvariableop_resource*
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
bodyR
while_body_362317*
condR
while_cond_362316*K
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

IdentityХ
NoOpNoOp#^lstm_cell_7/BiasAdd/ReadVariableOp"^lstm_cell_7/MatMul/ReadVariableOp$^lstm_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2H
"lstm_cell_7/BiasAdd/ReadVariableOp"lstm_cell_7/BiasAdd/ReadVariableOp2F
!lstm_cell_7/MatMul/ReadVariableOp!lstm_cell_7/MatMul/ReadVariableOp2J
#lstm_cell_7/MatMul_1/ReadVariableOp#lstm_cell_7/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
^

K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_366593

inputs=
*lstm_cell_8_matmul_readvariableop_resource:	Ш?
,lstm_cell_8_matmul_1_readvariableop_resource:	2Ш:
+lstm_cell_8_biasadd_readvariableop_resource:	Ш
identityЂ"lstm_cell_8/BiasAdd/ReadVariableOpЂ!lstm_cell_8/MatMul/ReadVariableOpЂ#lstm_cell_8/MatMul_1/ReadVariableOpЂwhileD
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
strided_slice_2В
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOpЊ
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/MatMulИ
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOpІ
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/MatMul_1
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/addБ
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOpЉ
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/BiasAdd|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimя
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_8/split
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Sigmoid
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Sigmoid_1
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Relu
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/mul_1
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/add_1
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Relu_1
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
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
bodyR
while_body_366509*
condR
while_cond_366508*K
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

IdentityХ
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Т>
Ч
while_body_365704
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_7_matmul_readvariableop_resource_0:	ШG
4while_lstm_cell_7_matmul_1_readvariableop_resource_0:	2ШB
3while_lstm_cell_7_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_7_matmul_readvariableop_resource:	ШE
2while_lstm_cell_7_matmul_1_readvariableop_resource:	2Ш@
1while_lstm_cell_7_biasadd_readvariableop_resource:	ШЂ(while/lstm_cell_7/BiasAdd/ReadVariableOpЂ'while/lstm_cell_7/MatMul/ReadVariableOpЂ)while/lstm_cell_7/MatMul_1/ReadVariableOpУ
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
)while/TensorArrayV2Read/TensorListGetItemЦ
'while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02)
'while/lstm_cell_7/MatMul/ReadVariableOpд
while/lstm_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/MatMulЬ
)while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02+
)while/lstm_cell_7/MatMul_1/ReadVariableOpН
while/lstm_cell_7/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/MatMul_1Д
while/lstm_cell_7/addAddV2"while/lstm_cell_7/MatMul:product:0$while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/addХ
(while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02*
(while/lstm_cell_7/BiasAdd/ReadVariableOpС
while/lstm_cell_7/BiasAddBiasAddwhile/lstm_cell_7/add:z:00while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/BiasAdd
!while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_7/split/split_dim
while/lstm_cell_7/splitSplit*while/lstm_cell_7/split/split_dim:output:0"while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_7/split
while/lstm_cell_7/SigmoidSigmoid while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Sigmoid
while/lstm_cell_7/Sigmoid_1Sigmoid while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Sigmoid_1
while/lstm_cell_7/mulMulwhile/lstm_cell_7/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/mul
while/lstm_cell_7/ReluRelu while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/ReluА
while/lstm_cell_7/mul_1Mulwhile/lstm_cell_7/Sigmoid:y:0$while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/mul_1Ѕ
while/lstm_cell_7/add_1AddV2while/lstm_cell_7/mul:z:0while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/add_1
while/lstm_cell_7/Sigmoid_2Sigmoid while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Sigmoid_2
while/lstm_cell_7/Relu_1Reluwhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Relu_1Д
while/lstm_cell_7/mul_2Mulwhile/lstm_cell_7/Sigmoid_2:y:0&while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/mul_2п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_7/mul_2:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_7/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_7/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5л

while/NoOpNoOp)^while/lstm_cell_7/BiasAdd/ReadVariableOp(^while/lstm_cell_7/MatMul/ReadVariableOp*^while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_7_biasadd_readvariableop_resource3while_lstm_cell_7_biasadd_readvariableop_resource_0"j
2while_lstm_cell_7_matmul_1_readvariableop_resource4while_lstm_cell_7_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_7_matmul_readvariableop_resource2while_lstm_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2T
(while/lstm_cell_7/BiasAdd/ReadVariableOp(while/lstm_cell_7/BiasAdd/ReadVariableOp2R
'while/lstm_cell_7/MatMul/ReadVariableOp'while/lstm_cell_7/MatMul/ReadVariableOp2V
)while/lstm_cell_7/MatMul_1/ReadVariableOp)while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
ЊБ
З
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_365064

inputs
inputs_1	L
9forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource:	ШN
;forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource:	2ШI
:forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource:	ШM
:backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource:	ШO
<backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource:	2ШJ
;backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource:	Ш
identityЂ2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpЂ1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpЂ3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpЂbackward_lstm_2/whileЂ1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpЂ0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpЂ2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpЂforward_lstm_2/while
#forward_lstm_2/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2%
#forward_lstm_2/RaggedToTensor/zeros
#forward_lstm_2/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2%
#forward_lstm_2/RaggedToTensor/Const
2forward_lstm_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor,forward_lstm_2/RaggedToTensor/Const:output:0inputs,forward_lstm_2/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS24
2forward_lstm_2/RaggedToTensor/RaggedTensorToTensorР
9forward_lstm_2/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9forward_lstm_2/RaggedNestedRowLengths/strided_slice/stackФ
;forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1Ф
;forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;forward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2
3forward_lstm_2/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Bforward_lstm_2/RaggedNestedRowLengths/strided_slice/stack:output:0Dforward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1:output:0Dforward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask25
3forward_lstm_2/RaggedNestedRowLengths/strided_sliceФ
;forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stackб
=forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2?
=forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1Ш
=forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=forward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2Ћ
5forward_lstm_2/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Dforward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack:output:0Fforward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Fforward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask27
5forward_lstm_2/RaggedNestedRowLengths/strided_slice_1
)forward_lstm_2/RaggedNestedRowLengths/subSub<forward_lstm_2/RaggedNestedRowLengths/strided_slice:output:0>forward_lstm_2/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2+
)forward_lstm_2/RaggedNestedRowLengths/sub
forward_lstm_2/CastCast-forward_lstm_2/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
forward_lstm_2/Cast
forward_lstm_2/ShapeShape;forward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_2/Shape
"forward_lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"forward_lstm_2/strided_slice/stack
$forward_lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_2/strided_slice/stack_1
$forward_lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_lstm_2/strided_slice/stack_2М
forward_lstm_2/strided_sliceStridedSliceforward_lstm_2/Shape:output:0+forward_lstm_2/strided_slice/stack:output:0-forward_lstm_2/strided_slice/stack_1:output:0-forward_lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_2/strided_slicez
forward_lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_2/zeros/mul/yЈ
forward_lstm_2/zeros/mulMul%forward_lstm_2/strided_slice:output:0#forward_lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros/mul}
forward_lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_2/zeros/Less/yЃ
forward_lstm_2/zeros/LessLessforward_lstm_2/zeros/mul:z:0$forward_lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros/Less
forward_lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_2/zeros/packed/1П
forward_lstm_2/zeros/packedPack%forward_lstm_2/strided_slice:output:0&forward_lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_2/zeros/packed
forward_lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_2/zeros/ConstБ
forward_lstm_2/zerosFill$forward_lstm_2/zeros/packed:output:0#forward_lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/zeros~
forward_lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_2/zeros_1/mul/yЎ
forward_lstm_2/zeros_1/mulMul%forward_lstm_2/strided_slice:output:0%forward_lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros_1/mul
forward_lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
forward_lstm_2/zeros_1/Less/yЋ
forward_lstm_2/zeros_1/LessLessforward_lstm_2/zeros_1/mul:z:0&forward_lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_2/zeros_1/Less
forward_lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
forward_lstm_2/zeros_1/packed/1Х
forward_lstm_2/zeros_1/packedPack%forward_lstm_2/strided_slice:output:0(forward_lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_2/zeros_1/packed
forward_lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_2/zeros_1/ConstЙ
forward_lstm_2/zeros_1Fill&forward_lstm_2/zeros_1/packed:output:0%forward_lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/zeros_1
forward_lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
forward_lstm_2/transpose/permх
forward_lstm_2/transpose	Transpose;forward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0&forward_lstm_2/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
forward_lstm_2/transpose|
forward_lstm_2/Shape_1Shapeforward_lstm_2/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_2/Shape_1
$forward_lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_2/strided_slice_1/stack
&forward_lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_1/stack_1
&forward_lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_1/stack_2Ш
forward_lstm_2/strided_slice_1StridedSliceforward_lstm_2/Shape_1:output:0-forward_lstm_2/strided_slice_1/stack:output:0/forward_lstm_2/strided_slice_1/stack_1:output:0/forward_lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_lstm_2/strided_slice_1Ѓ
*forward_lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*forward_lstm_2/TensorArrayV2/element_shapeю
forward_lstm_2/TensorArrayV2TensorListReserve3forward_lstm_2/TensorArrayV2/element_shape:output:0'forward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_2/TensorArrayV2н
Dforward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2F
Dforward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeД
6forward_lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_2/transpose:y:0Mforward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6forward_lstm_2/TensorArrayUnstack/TensorListFromTensor
$forward_lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_lstm_2/strided_slice_2/stack
&forward_lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_2/stack_1
&forward_lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_2/stack_2ж
forward_lstm_2/strided_slice_2StridedSliceforward_lstm_2/transpose:y:0-forward_lstm_2/strided_slice_2/stack:output:0/forward_lstm_2/strided_slice_2/stack_1:output:0/forward_lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2 
forward_lstm_2/strided_slice_2п
0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp9forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype022
0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOpц
!forward_lstm_2/lstm_cell_7/MatMulMatMul'forward_lstm_2/strided_slice_2:output:08forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2#
!forward_lstm_2/lstm_cell_7/MatMulх
2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp;forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype024
2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOpт
#forward_lstm_2/lstm_cell_7/MatMul_1MatMulforward_lstm_2/zeros:output:0:forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#forward_lstm_2/lstm_cell_7/MatMul_1и
forward_lstm_2/lstm_cell_7/addAddV2+forward_lstm_2/lstm_cell_7/MatMul:product:0-forward_lstm_2/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2 
forward_lstm_2/lstm_cell_7/addо
1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp:forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype023
1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOpх
"forward_lstm_2/lstm_cell_7/BiasAddBiasAdd"forward_lstm_2/lstm_cell_7/add:z:09forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"forward_lstm_2/lstm_cell_7/BiasAdd
*forward_lstm_2/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*forward_lstm_2/lstm_cell_7/split/split_dimЋ
 forward_lstm_2/lstm_cell_7/splitSplit3forward_lstm_2/lstm_cell_7/split/split_dim:output:0+forward_lstm_2/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2"
 forward_lstm_2/lstm_cell_7/splitА
"forward_lstm_2/lstm_cell_7/SigmoidSigmoid)forward_lstm_2/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"forward_lstm_2/lstm_cell_7/SigmoidД
$forward_lstm_2/lstm_cell_7/Sigmoid_1Sigmoid)forward_lstm_2/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_2/lstm_cell_7/Sigmoid_1Ф
forward_lstm_2/lstm_cell_7/mulMul(forward_lstm_2/lstm_cell_7/Sigmoid_1:y:0forward_lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
forward_lstm_2/lstm_cell_7/mulЇ
forward_lstm_2/lstm_cell_7/ReluRelu)forward_lstm_2/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22!
forward_lstm_2/lstm_cell_7/Reluд
 forward_lstm_2/lstm_cell_7/mul_1Mul&forward_lstm_2/lstm_cell_7/Sigmoid:y:0-forward_lstm_2/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_2/lstm_cell_7/mul_1Щ
 forward_lstm_2/lstm_cell_7/add_1AddV2"forward_lstm_2/lstm_cell_7/mul:z:0$forward_lstm_2/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_2/lstm_cell_7/add_1Д
$forward_lstm_2/lstm_cell_7/Sigmoid_2Sigmoid)forward_lstm_2/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22&
$forward_lstm_2/lstm_cell_7/Sigmoid_2І
!forward_lstm_2/lstm_cell_7/Relu_1Relu$forward_lstm_2/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!forward_lstm_2/lstm_cell_7/Relu_1и
 forward_lstm_2/lstm_cell_7/mul_2Mul(forward_lstm_2/lstm_cell_7/Sigmoid_2:y:0/forward_lstm_2/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 forward_lstm_2/lstm_cell_7/mul_2­
,forward_lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2.
,forward_lstm_2/TensorArrayV2_1/element_shapeє
forward_lstm_2/TensorArrayV2_1TensorListReserve5forward_lstm_2/TensorArrayV2_1/element_shape:output:0'forward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_lstm_2/TensorArrayV2_1l
forward_lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_2/time
forward_lstm_2/zeros_like	ZerosLike$forward_lstm_2/lstm_cell_7/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
forward_lstm_2/zeros_like
'forward_lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'forward_lstm_2/while/maximum_iterations
!forward_lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!forward_lstm_2/while/loop_counterч
forward_lstm_2/whileWhile*forward_lstm_2/while/loop_counter:output:00forward_lstm_2/while/maximum_iterations:output:0forward_lstm_2/time:output:0'forward_lstm_2/TensorArrayV2_1:handle:0forward_lstm_2/zeros_like:y:0forward_lstm_2/zeros:output:0forward_lstm_2/zeros_1:output:0'forward_lstm_2/strided_slice_1:output:0Fforward_lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_2/Cast:y:09forward_lstm_2_lstm_cell_7_matmul_readvariableop_resource;forward_lstm_2_lstm_cell_7_matmul_1_readvariableop_resource:forward_lstm_2_lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *,
body$R"
 forward_lstm_2_while_body_364788*,
cond$R"
 forward_lstm_2_while_cond_364787*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
forward_lstm_2/whileг
?forward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2A
?forward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape­
1forward_lstm_2/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_2/while:output:3Hforward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype023
1forward_lstm_2/TensorArrayV2Stack/TensorListStack
$forward_lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2&
$forward_lstm_2/strided_slice_3/stack
&forward_lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_lstm_2/strided_slice_3/stack_1
&forward_lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_lstm_2/strided_slice_3/stack_2є
forward_lstm_2/strided_slice_3StridedSlice:forward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0-forward_lstm_2/strided_slice_3/stack:output:0/forward_lstm_2/strided_slice_3/stack_1:output:0/forward_lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2 
forward_lstm_2/strided_slice_3
forward_lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_lstm_2/transpose_1/permъ
forward_lstm_2/transpose_1	Transpose:forward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0(forward_lstm_2/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
forward_lstm_2/transpose_1
forward_lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_2/runtime
$backward_lstm_2/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$backward_lstm_2/RaggedToTensor/zeros
$backward_lstm_2/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2&
$backward_lstm_2/RaggedToTensor/Const
3backward_lstm_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-backward_lstm_2/RaggedToTensor/Const:output:0inputs-backward_lstm_2/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3backward_lstm_2/RaggedToTensor/RaggedTensorToTensorТ
:backward_lstm_2/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:backward_lstm_2/RaggedNestedRowLengths/strided_slice/stackЦ
<backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1Ц
<backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<backward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2Є
4backward_lstm_2/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cbackward_lstm_2/RaggedNestedRowLengths/strided_slice/stack:output:0Ebackward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_1:output:0Ebackward_lstm_2/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*
end_mask26
4backward_lstm_2/RaggedNestedRowLengths/strided_sliceЦ
<backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stackг
>backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2@
>backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1Ъ
>backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>backward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2А
6backward_lstm_2/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Ebackward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gbackward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gbackward_lstm_2/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask28
6backward_lstm_2/RaggedNestedRowLengths/strided_slice_1
*backward_lstm_2/RaggedNestedRowLengths/subSub=backward_lstm_2/RaggedNestedRowLengths/strided_slice:output:0?backward_lstm_2/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2,
*backward_lstm_2/RaggedNestedRowLengths/subЁ
backward_lstm_2/CastCast.backward_lstm_2/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_2/Cast
backward_lstm_2/ShapeShape<backward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_2/Shape
#backward_lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#backward_lstm_2/strided_slice/stack
%backward_lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_2/strided_slice/stack_1
%backward_lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_lstm_2/strided_slice/stack_2Т
backward_lstm_2/strided_sliceStridedSlicebackward_lstm_2/Shape:output:0,backward_lstm_2/strided_slice/stack:output:0.backward_lstm_2/strided_slice/stack_1:output:0.backward_lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
backward_lstm_2/strided_slice|
backward_lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_2/zeros/mul/yЌ
backward_lstm_2/zeros/mulMul&backward_lstm_2/strided_slice:output:0$backward_lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros/mul
backward_lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
backward_lstm_2/zeros/Less/yЇ
backward_lstm_2/zeros/LessLessbackward_lstm_2/zeros/mul:z:0%backward_lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros/Less
backward_lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_2/zeros/packed/1У
backward_lstm_2/zeros/packedPack&backward_lstm_2/strided_slice:output:0'backward_lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_2/zeros/packed
backward_lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_2/zeros/ConstЕ
backward_lstm_2/zerosFill%backward_lstm_2/zeros/packed:output:0$backward_lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/zeros
backward_lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_2/zeros_1/mul/yВ
backward_lstm_2/zeros_1/mulMul&backward_lstm_2/strided_slice:output:0&backward_lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros_1/mul
backward_lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
backward_lstm_2/zeros_1/Less/yЏ
backward_lstm_2/zeros_1/LessLessbackward_lstm_2/zeros_1/mul:z:0'backward_lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/zeros_1/Less
 backward_lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 backward_lstm_2/zeros_1/packed/1Щ
backward_lstm_2/zeros_1/packedPack&backward_lstm_2/strided_slice:output:0)backward_lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
backward_lstm_2/zeros_1/packed
backward_lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_2/zeros_1/ConstН
backward_lstm_2/zeros_1Fill'backward_lstm_2/zeros_1/packed:output:0&backward_lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/zeros_1
backward_lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
backward_lstm_2/transpose/permщ
backward_lstm_2/transpose	Transpose<backward_lstm_2/RaggedToTensor/RaggedTensorToTensor:result:0'backward_lstm_2/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_2/transpose
backward_lstm_2/Shape_1Shapebackward_lstm_2/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_2/Shape_1
%backward_lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_2/strided_slice_1/stack
'backward_lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_1/stack_1
'backward_lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_1/stack_2Ю
backward_lstm_2/strided_slice_1StridedSlice backward_lstm_2/Shape_1:output:0.backward_lstm_2/strided_slice_1/stack:output:00backward_lstm_2/strided_slice_1/stack_1:output:00backward_lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_lstm_2/strided_slice_1Ѕ
+backward_lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+backward_lstm_2/TensorArrayV2/element_shapeђ
backward_lstm_2/TensorArrayV2TensorListReserve4backward_lstm_2/TensorArrayV2/element_shape:output:0(backward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
backward_lstm_2/TensorArrayV2
backward_lstm_2/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2 
backward_lstm_2/ReverseV2/axisЪ
backward_lstm_2/ReverseV2	ReverseV2backward_lstm_2/transpose:y:0'backward_lstm_2/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
backward_lstm_2/ReverseV2п
Ebackward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2G
Ebackward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeН
7backward_lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"backward_lstm_2/ReverseV2:output:0Nbackward_lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7backward_lstm_2/TensorArrayUnstack/TensorListFromTensor
%backward_lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_lstm_2/strided_slice_2/stack
'backward_lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_2/stack_1
'backward_lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_2/stack_2м
backward_lstm_2/strided_slice_2StridedSlicebackward_lstm_2/transpose:y:0.backward_lstm_2/strided_slice_2/stack:output:00backward_lstm_2/strided_slice_2/stack_1:output:00backward_lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2!
backward_lstm_2/strided_slice_2т
1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp:backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype023
1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOpъ
"backward_lstm_2/lstm_cell_8/MatMulMatMul(backward_lstm_2/strided_slice_2:output:09backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"backward_lstm_2/lstm_cell_8/MatMulш
3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp<backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype025
3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOpц
$backward_lstm_2/lstm_cell_8/MatMul_1MatMulbackward_lstm_2/zeros:output:0;backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$backward_lstm_2/lstm_cell_8/MatMul_1м
backward_lstm_2/lstm_cell_8/addAddV2,backward_lstm_2/lstm_cell_8/MatMul:product:0.backward_lstm_2/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2!
backward_lstm_2/lstm_cell_8/addс
2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp;backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype024
2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOpщ
#backward_lstm_2/lstm_cell_8/BiasAddBiasAdd#backward_lstm_2/lstm_cell_8/add:z:0:backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#backward_lstm_2/lstm_cell_8/BiasAdd
+backward_lstm_2/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+backward_lstm_2/lstm_cell_8/split/split_dimЏ
!backward_lstm_2/lstm_cell_8/splitSplit4backward_lstm_2/lstm_cell_8/split/split_dim:output:0,backward_lstm_2/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2#
!backward_lstm_2/lstm_cell_8/splitГ
#backward_lstm_2/lstm_cell_8/SigmoidSigmoid*backward_lstm_2/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#backward_lstm_2/lstm_cell_8/SigmoidЗ
%backward_lstm_2/lstm_cell_8/Sigmoid_1Sigmoid*backward_lstm_2/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_2/lstm_cell_8/Sigmoid_1Ш
backward_lstm_2/lstm_cell_8/mulMul)backward_lstm_2/lstm_cell_8/Sigmoid_1:y:0 backward_lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22!
backward_lstm_2/lstm_cell_8/mulЊ
 backward_lstm_2/lstm_cell_8/ReluRelu*backward_lstm_2/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/lstm_cell_8/Reluи
!backward_lstm_2/lstm_cell_8/mul_1Mul'backward_lstm_2/lstm_cell_8/Sigmoid:y:0.backward_lstm_2/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_2/lstm_cell_8/mul_1Э
!backward_lstm_2/lstm_cell_8/add_1AddV2#backward_lstm_2/lstm_cell_8/mul:z:0%backward_lstm_2/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_2/lstm_cell_8/add_1З
%backward_lstm_2/lstm_cell_8/Sigmoid_2Sigmoid*backward_lstm_2/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_2/lstm_cell_8/Sigmoid_2Љ
"backward_lstm_2/lstm_cell_8/Relu_1Relu%backward_lstm_2/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"backward_lstm_2/lstm_cell_8/Relu_1м
!backward_lstm_2/lstm_cell_8/mul_2Mul)backward_lstm_2/lstm_cell_8/Sigmoid_2:y:00backward_lstm_2/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!backward_lstm_2/lstm_cell_8/mul_2Џ
-backward_lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2/
-backward_lstm_2/TensorArrayV2_1/element_shapeј
backward_lstm_2/TensorArrayV2_1TensorListReserve6backward_lstm_2/TensorArrayV2_1/element_shape:output:0(backward_lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_lstm_2/TensorArrayV2_1n
backward_lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_2/time
%backward_lstm_2/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2'
%backward_lstm_2/Max/reduction_indices
backward_lstm_2/MaxMaxbackward_lstm_2/Cast:y:0.backward_lstm_2/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/Maxp
backward_lstm_2/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_2/sub/y
backward_lstm_2/subSubbackward_lstm_2/Max:output:0backward_lstm_2/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/sub
backward_lstm_2/Sub_1Subbackward_lstm_2/sub:z:0backward_lstm_2/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_2/Sub_1
backward_lstm_2/zeros_like	ZerosLike%backward_lstm_2/lstm_cell_8/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/zeros_like
(backward_lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(backward_lstm_2/while/maximum_iterations
"backward_lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"backward_lstm_2/while/loop_counterљ
backward_lstm_2/whileWhile+backward_lstm_2/while/loop_counter:output:01backward_lstm_2/while/maximum_iterations:output:0backward_lstm_2/time:output:0(backward_lstm_2/TensorArrayV2_1:handle:0backward_lstm_2/zeros_like:y:0backward_lstm_2/zeros:output:0 backward_lstm_2/zeros_1:output:0(backward_lstm_2/strided_slice_1:output:0Gbackward_lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_2/Sub_1:z:0:backward_lstm_2_lstm_cell_8_matmul_readvariableop_resource<backward_lstm_2_lstm_cell_8_matmul_1_readvariableop_resource;backward_lstm_2_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *-
body%R#
!backward_lstm_2_while_body_364967*-
cond%R#
!backward_lstm_2_while_cond_364966*m
output_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : *
parallel_iterations 2
backward_lstm_2/whileе
@backward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2B
@backward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeБ
2backward_lstm_2/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_2/while:output:3Ibackward_lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype024
2backward_lstm_2/TensorArrayV2Stack/TensorListStackЁ
%backward_lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2'
%backward_lstm_2/strided_slice_3/stack
'backward_lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_lstm_2/strided_slice_3/stack_1
'backward_lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_lstm_2/strided_slice_3/stack_2њ
backward_lstm_2/strided_slice_3StridedSlice;backward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0.backward_lstm_2/strided_slice_3/stack:output:00backward_lstm_2/strided_slice_3/stack_1:output:00backward_lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2!
backward_lstm_2/strided_slice_3
 backward_lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_lstm_2/transpose_1/permю
backward_lstm_2/transpose_1	Transpose;backward_lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0)backward_lstm_2/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
backward_lstm_2/transpose_1
backward_lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_2/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisР
concatConcatV2'forward_lstm_2/strided_slice_3:output:0(backward_lstm_2/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

IdentityИ
NoOpNoOp3^backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp2^backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp4^backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp^backward_lstm_2/while2^forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp1^forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp3^forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp^forward_lstm_2/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2h
2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp2backward_lstm_2/lstm_cell_8/BiasAdd/ReadVariableOp2f
1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp1backward_lstm_2/lstm_cell_8/MatMul/ReadVariableOp2j
3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp3backward_lstm_2/lstm_cell_8/MatMul_1/ReadVariableOp2.
backward_lstm_2/whilebackward_lstm_2/while2f
1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp1forward_lstm_2/lstm_cell_7/BiasAdd/ReadVariableOp2d
0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp0forward_lstm_2/lstm_cell_7/MatMul/ReadVariableOp2h
2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp2forward_lstm_2/lstm_cell_7/MatMul_1/ReadVariableOp2,
forward_lstm_2/whileforward_lstm_2/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Г
ж
>sequential_2_bidirectional_2_backward_lstm_2_while_body_360872v
rsequential_2_bidirectional_2_backward_lstm_2_while_sequential_2_bidirectional_2_backward_lstm_2_while_loop_counter|
xsequential_2_bidirectional_2_backward_lstm_2_while_sequential_2_bidirectional_2_backward_lstm_2_while_maximum_iterationsB
>sequential_2_bidirectional_2_backward_lstm_2_while_placeholderD
@sequential_2_bidirectional_2_backward_lstm_2_while_placeholder_1D
@sequential_2_bidirectional_2_backward_lstm_2_while_placeholder_2D
@sequential_2_bidirectional_2_backward_lstm_2_while_placeholder_3D
@sequential_2_bidirectional_2_backward_lstm_2_while_placeholder_4u
qsequential_2_bidirectional_2_backward_lstm_2_while_sequential_2_bidirectional_2_backward_lstm_2_strided_slice_1_0В
­sequential_2_bidirectional_2_backward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_2_bidirectional_2_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0p
lsequential_2_bidirectional_2_backward_lstm_2_while_less_sequential_2_bidirectional_2_backward_lstm_2_sub_1_0r
_sequential_2_bidirectional_2_backward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0:	Шt
asequential_2_bidirectional_2_backward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0:	2Шo
`sequential_2_bidirectional_2_backward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ш?
;sequential_2_bidirectional_2_backward_lstm_2_while_identityA
=sequential_2_bidirectional_2_backward_lstm_2_while_identity_1A
=sequential_2_bidirectional_2_backward_lstm_2_while_identity_2A
=sequential_2_bidirectional_2_backward_lstm_2_while_identity_3A
=sequential_2_bidirectional_2_backward_lstm_2_while_identity_4A
=sequential_2_bidirectional_2_backward_lstm_2_while_identity_5A
=sequential_2_bidirectional_2_backward_lstm_2_while_identity_6s
osequential_2_bidirectional_2_backward_lstm_2_while_sequential_2_bidirectional_2_backward_lstm_2_strided_slice_1А
Ћsequential_2_bidirectional_2_backward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_2_bidirectional_2_backward_lstm_2_tensorarrayunstack_tensorlistfromtensorn
jsequential_2_bidirectional_2_backward_lstm_2_while_less_sequential_2_bidirectional_2_backward_lstm_2_sub_1p
]sequential_2_bidirectional_2_backward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource:	Шr
_sequential_2_bidirectional_2_backward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource:	2Шm
^sequential_2_bidirectional_2_backward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource:	ШЂUsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOpЂTsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOpЂVsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp
dsequential_2/bidirectional_2/backward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2f
dsequential_2/bidirectional_2/backward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeт
Vsequential_2/bidirectional_2/backward_lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem­sequential_2_bidirectional_2_backward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_2_bidirectional_2_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0>sequential_2_bidirectional_2_backward_lstm_2_while_placeholdermsequential_2/bidirectional_2/backward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02X
Vsequential_2/bidirectional_2/backward_lstm_2/while/TensorArrayV2Read/TensorListGetItemж
7sequential_2/bidirectional_2/backward_lstm_2/while/LessLesslsequential_2_bidirectional_2_backward_lstm_2_while_less_sequential_2_bidirectional_2_backward_lstm_2_sub_1_0>sequential_2_bidirectional_2_backward_lstm_2_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ29
7sequential_2/bidirectional_2/backward_lstm_2/while/LessЭ
Tsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp_sequential_2_bidirectional_2_backward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02V
Tsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp
Esequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMulMatMul]sequential_2/bidirectional_2/backward_lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0\sequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2G
Esequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMulг
Vsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOpasequential_2_bidirectional_2_backward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02X
Vsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOpё
Gsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMul_1MatMul@sequential_2_bidirectional_2_backward_lstm_2_while_placeholder_3^sequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2I
Gsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMul_1ш
Bsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/addAddV2Osequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMul:product:0Qsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2D
Bsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/addЬ
Usequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp`sequential_2_bidirectional_2_backward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02W
Usequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOpѕ
Fsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/BiasAddBiasAddFsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/add:z:0]sequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2H
Fsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/BiasAddт
Nsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2P
Nsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/split/split_dimЛ
Dsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/splitSplitWsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/split/split_dim:output:0Osequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2F
Dsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/split
Fsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/SigmoidSigmoidMsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22H
Fsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/Sigmoid 
Hsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/Sigmoid_1SigmoidMsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22J
Hsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/Sigmoid_1б
Bsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/mulMulLsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/Sigmoid_1:y:0@sequential_2_bidirectional_2_backward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22D
Bsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/mul
Csequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/ReluReluMsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22E
Csequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/Reluф
Dsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/mul_1MulJsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/Sigmoid:y:0Qsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22F
Dsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/mul_1й
Dsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/add_1AddV2Fsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/mul:z:0Hsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22F
Dsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/add_1 
Hsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/Sigmoid_2SigmoidMsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22J
Hsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/Sigmoid_2
Esequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/Relu_1ReluHsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22G
Esequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/Relu_1ш
Dsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/mul_2MulLsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/Sigmoid_2:y:0Ssequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22F
Dsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/mul_2ћ
9sequential_2/bidirectional_2/backward_lstm_2/while/SelectSelect;sequential_2/bidirectional_2/backward_lstm_2/while/Less:z:0Hsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/mul_2:z:0@sequential_2_bidirectional_2_backward_lstm_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22;
9sequential_2/bidirectional_2/backward_lstm_2/while/Selectџ
;sequential_2/bidirectional_2/backward_lstm_2/while/Select_1Select;sequential_2/bidirectional_2/backward_lstm_2/while/Less:z:0Hsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/mul_2:z:0@sequential_2_bidirectional_2_backward_lstm_2_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22=
;sequential_2/bidirectional_2/backward_lstm_2/while/Select_1џ
;sequential_2/bidirectional_2/backward_lstm_2/while/Select_2Select;sequential_2/bidirectional_2/backward_lstm_2/while/Less:z:0Hsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/add_1:z:0@sequential_2_bidirectional_2_backward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22=
;sequential_2/bidirectional_2/backward_lstm_2/while/Select_2К
Wsequential_2/bidirectional_2/backward_lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem@sequential_2_bidirectional_2_backward_lstm_2_while_placeholder_1>sequential_2_bidirectional_2_backward_lstm_2_while_placeholderBsequential_2/bidirectional_2/backward_lstm_2/while/Select:output:0*
_output_shapes
: *
element_dtype02Y
Wsequential_2/bidirectional_2/backward_lstm_2/while/TensorArrayV2Write/TensorListSetItemЖ
8sequential_2/bidirectional_2/backward_lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_2/bidirectional_2/backward_lstm_2/while/add/y
6sequential_2/bidirectional_2/backward_lstm_2/while/addAddV2>sequential_2_bidirectional_2_backward_lstm_2_while_placeholderAsequential_2/bidirectional_2/backward_lstm_2/while/add/y:output:0*
T0*
_output_shapes
: 28
6sequential_2/bidirectional_2/backward_lstm_2/while/addК
:sequential_2/bidirectional_2/backward_lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_2/bidirectional_2/backward_lstm_2/while/add_1/yз
8sequential_2/bidirectional_2/backward_lstm_2/while/add_1AddV2rsequential_2_bidirectional_2_backward_lstm_2_while_sequential_2_bidirectional_2_backward_lstm_2_while_loop_counterCsequential_2/bidirectional_2/backward_lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2:
8sequential_2/bidirectional_2/backward_lstm_2/while/add_1
;sequential_2/bidirectional_2/backward_lstm_2/while/IdentityIdentity<sequential_2/bidirectional_2/backward_lstm_2/while/add_1:z:08^sequential_2/bidirectional_2/backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2=
;sequential_2/bidirectional_2/backward_lstm_2/while/Identityп
=sequential_2/bidirectional_2/backward_lstm_2/while/Identity_1Identityxsequential_2_bidirectional_2_backward_lstm_2_while_sequential_2_bidirectional_2_backward_lstm_2_while_maximum_iterations8^sequential_2/bidirectional_2/backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2?
=sequential_2/bidirectional_2/backward_lstm_2/while/Identity_1Ё
=sequential_2/bidirectional_2/backward_lstm_2/while/Identity_2Identity:sequential_2/bidirectional_2/backward_lstm_2/while/add:z:08^sequential_2/bidirectional_2/backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2?
=sequential_2/bidirectional_2/backward_lstm_2/while/Identity_2Ю
=sequential_2/bidirectional_2/backward_lstm_2/while/Identity_3Identitygsequential_2/bidirectional_2/backward_lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:08^sequential_2/bidirectional_2/backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2?
=sequential_2/bidirectional_2/backward_lstm_2/while/Identity_3К
=sequential_2/bidirectional_2/backward_lstm_2/while/Identity_4IdentityBsequential_2/bidirectional_2/backward_lstm_2/while/Select:output:08^sequential_2/bidirectional_2/backward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22?
=sequential_2/bidirectional_2/backward_lstm_2/while/Identity_4М
=sequential_2/bidirectional_2/backward_lstm_2/while/Identity_5IdentityDsequential_2/bidirectional_2/backward_lstm_2/while/Select_1:output:08^sequential_2/bidirectional_2/backward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22?
=sequential_2/bidirectional_2/backward_lstm_2/while/Identity_5М
=sequential_2/bidirectional_2/backward_lstm_2/while/Identity_6IdentityDsequential_2/bidirectional_2/backward_lstm_2/while/Select_2:output:08^sequential_2/bidirectional_2/backward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22?
=sequential_2/bidirectional_2/backward_lstm_2/while/Identity_6М
7sequential_2/bidirectional_2/backward_lstm_2/while/NoOpNoOpV^sequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOpU^sequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOpW^sequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 29
7sequential_2/bidirectional_2/backward_lstm_2/while/NoOp"
;sequential_2_bidirectional_2_backward_lstm_2_while_identityDsequential_2/bidirectional_2/backward_lstm_2/while/Identity:output:0"
=sequential_2_bidirectional_2_backward_lstm_2_while_identity_1Fsequential_2/bidirectional_2/backward_lstm_2/while/Identity_1:output:0"
=sequential_2_bidirectional_2_backward_lstm_2_while_identity_2Fsequential_2/bidirectional_2/backward_lstm_2/while/Identity_2:output:0"
=sequential_2_bidirectional_2_backward_lstm_2_while_identity_3Fsequential_2/bidirectional_2/backward_lstm_2/while/Identity_3:output:0"
=sequential_2_bidirectional_2_backward_lstm_2_while_identity_4Fsequential_2/bidirectional_2/backward_lstm_2/while/Identity_4:output:0"
=sequential_2_bidirectional_2_backward_lstm_2_while_identity_5Fsequential_2/bidirectional_2/backward_lstm_2/while/Identity_5:output:0"
=sequential_2_bidirectional_2_backward_lstm_2_while_identity_6Fsequential_2/bidirectional_2/backward_lstm_2/while/Identity_6:output:0"к
jsequential_2_bidirectional_2_backward_lstm_2_while_less_sequential_2_bidirectional_2_backward_lstm_2_sub_1lsequential_2_bidirectional_2_backward_lstm_2_while_less_sequential_2_bidirectional_2_backward_lstm_2_sub_1_0"Т
^sequential_2_bidirectional_2_backward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource`sequential_2_bidirectional_2_backward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0"Ф
_sequential_2_bidirectional_2_backward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resourceasequential_2_bidirectional_2_backward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0"Р
]sequential_2_bidirectional_2_backward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_sequential_2_bidirectional_2_backward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0"ф
osequential_2_bidirectional_2_backward_lstm_2_while_sequential_2_bidirectional_2_backward_lstm_2_strided_slice_1qsequential_2_bidirectional_2_backward_lstm_2_while_sequential_2_bidirectional_2_backward_lstm_2_strided_slice_1_0"о
Ћsequential_2_bidirectional_2_backward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_2_bidirectional_2_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor­sequential_2_bidirectional_2_backward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_2_bidirectional_2_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2Ў
Usequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOpUsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp2Ќ
Tsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOpTsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp2А
Vsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOpVsequential_2/bidirectional_2/backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
я

(__inference_dense_2_layer_call_fn_365431

inputs
unknown:d
	unknown_0:
identityЂStatefulPartitionedCallѓ
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
GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3634372
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
е
У
while_cond_366661
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_366661___redundant_placeholder04
0while_while_cond_366661___redundant_placeholder14
0while_while_cond_366661___redundant_placeholder24
0while_while_cond_366661___redundant_placeholder3
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

є
C__inference_dense_2_layer_call_and_return_conditional_losses_363437

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
ђ

G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_361197

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
К
Д
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_362974

inputs(
forward_lstm_2_362957:	Ш(
forward_lstm_2_362959:	2Ш$
forward_lstm_2_362961:	Ш)
backward_lstm_2_362964:	Ш)
backward_lstm_2_362966:	2Ш%
backward_lstm_2_362968:	Ш
identityЂ'backward_lstm_2/StatefulPartitionedCallЂ&forward_lstm_2/StatefulPartitionedCallЫ
&forward_lstm_2/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_2_362957forward_lstm_2_362959forward_lstm_2_362961*
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
GPU 2J 8 *S
fNRL
J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_3629262(
&forward_lstm_2/StatefulPartitionedCallб
'backward_lstm_2/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_2_362964backward_lstm_2_362966backward_lstm_2_362968*
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
GPU 2J 8 *T
fORM
K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_3627532)
'backward_lstm_2/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisа
concatConcatV2/forward_lstm_2/StatefulPartitionedCall:output:00backward_lstm_2/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџd2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

IdentityЁ
NoOpNoOp(^backward_lstm_2/StatefulPartitionedCall'^forward_lstm_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 2R
'backward_lstm_2/StatefulPartitionedCall'backward_lstm_2/StatefulPartitionedCall2P
&forward_lstm_2/StatefulPartitionedCall&forward_lstm_2/StatefulPartitionedCall:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
^

K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_362561

inputs=
*lstm_cell_8_matmul_readvariableop_resource:	Ш?
,lstm_cell_8_matmul_1_readvariableop_resource:	2Ш:
+lstm_cell_8_biasadd_readvariableop_resource:	Ш
identityЂ"lstm_cell_8/BiasAdd/ReadVariableOpЂ!lstm_cell_8/MatMul/ReadVariableOpЂ#lstm_cell_8/MatMul_1/ReadVariableOpЂwhileD
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
strided_slice_2В
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOpЊ
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/MatMulИ
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOpІ
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/MatMul_1
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/addБ
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOpЉ
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/BiasAdd|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimя
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_8/split
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Sigmoid
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Sigmoid_1
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Relu
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/mul_1
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/add_1
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Relu_1
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
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
bodyR
while_body_362477*
condR
while_cond_362476*K
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

IdentityХ
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

я
 forward_lstm_2_while_cond_364470:
6forward_lstm_2_while_forward_lstm_2_while_loop_counter@
<forward_lstm_2_while_forward_lstm_2_while_maximum_iterations$
 forward_lstm_2_while_placeholder&
"forward_lstm_2_while_placeholder_1&
"forward_lstm_2_while_placeholder_2&
"forward_lstm_2_while_placeholder_3<
8forward_lstm_2_while_less_forward_lstm_2_strided_slice_1R
Nforward_lstm_2_while_forward_lstm_2_while_cond_364470___redundant_placeholder0R
Nforward_lstm_2_while_forward_lstm_2_while_cond_364470___redundant_placeholder1R
Nforward_lstm_2_while_forward_lstm_2_while_cond_364470___redundant_placeholder2R
Nforward_lstm_2_while_forward_lstm_2_while_cond_364470___redundant_placeholder3!
forward_lstm_2_while_identity
Л
forward_lstm_2/while/LessLess forward_lstm_2_while_placeholder8forward_lstm_2_while_less_forward_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_2/while/Less
forward_lstm_2/while/IdentityIdentityforward_lstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2
forward_lstm_2/while/Identity"G
forward_lstm_2_while_identity&forward_lstm_2/while/Identity:output:0*(
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
ДU
Ч
!backward_lstm_2_while_body_364620<
8backward_lstm_2_while_backward_lstm_2_while_loop_counterB
>backward_lstm_2_while_backward_lstm_2_while_maximum_iterations%
!backward_lstm_2_while_placeholder'
#backward_lstm_2_while_placeholder_1'
#backward_lstm_2_while_placeholder_2'
#backward_lstm_2_while_placeholder_3;
7backward_lstm_2_while_backward_lstm_2_strided_slice_1_0w
sbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0U
Bbackward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0:	ШW
Dbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0:	2ШR
Cbackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ш"
backward_lstm_2_while_identity$
 backward_lstm_2_while_identity_1$
 backward_lstm_2_while_identity_2$
 backward_lstm_2_while_identity_3$
 backward_lstm_2_while_identity_4$
 backward_lstm_2_while_identity_59
5backward_lstm_2_while_backward_lstm_2_strided_slice_1u
qbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensorS
@backward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource:	ШU
Bbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource:	2ШP
Abackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource:	ШЂ8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOpЂ7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOpЂ9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOpу
Gbackward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџ2I
Gbackward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeМ
9backward_lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0!backward_lstm_2_while_placeholderPbackward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
element_dtype02;
9backward_lstm_2/while/TensorArrayV2Read/TensorListGetItemі
7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOpBbackward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype029
7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp
(backward_lstm_2/while/lstm_cell_8/MatMulMatMul@backward_lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0?backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_2/while/lstm_cell_8/MatMulќ
9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOpDbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02;
9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp§
*backward_lstm_2/while/lstm_cell_8/MatMul_1MatMul#backward_lstm_2_while_placeholder_2Abackward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2,
*backward_lstm_2/while/lstm_cell_8/MatMul_1є
%backward_lstm_2/while/lstm_cell_8/addAddV22backward_lstm_2/while/lstm_cell_8/MatMul:product:04backward_lstm_2/while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%backward_lstm_2/while/lstm_cell_8/addѕ
8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOpCbackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02:
8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp
)backward_lstm_2/while/lstm_cell_8/BiasAddBiasAdd)backward_lstm_2/while/lstm_cell_8/add:z:0@backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_2/while/lstm_cell_8/BiasAddЈ
1backward_lstm_2/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1backward_lstm_2/while/lstm_cell_8/split/split_dimЧ
'backward_lstm_2/while/lstm_cell_8/splitSplit:backward_lstm_2/while/lstm_cell_8/split/split_dim:output:02backward_lstm_2/while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2)
'backward_lstm_2/while/lstm_cell_8/splitХ
)backward_lstm_2/while/lstm_cell_8/SigmoidSigmoid0backward_lstm_2/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_2/while/lstm_cell_8/SigmoidЩ
+backward_lstm_2/while/lstm_cell_8/Sigmoid_1Sigmoid0backward_lstm_2/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_2/while/lstm_cell_8/Sigmoid_1н
%backward_lstm_2/while/lstm_cell_8/mulMul/backward_lstm_2/while/lstm_cell_8/Sigmoid_1:y:0#backward_lstm_2_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_2/while/lstm_cell_8/mulМ
&backward_lstm_2/while/lstm_cell_8/ReluRelu0backward_lstm_2/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_2/while/lstm_cell_8/Relu№
'backward_lstm_2/while/lstm_cell_8/mul_1Mul-backward_lstm_2/while/lstm_cell_8/Sigmoid:y:04backward_lstm_2/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_2/while/lstm_cell_8/mul_1х
'backward_lstm_2/while/lstm_cell_8/add_1AddV2)backward_lstm_2/while/lstm_cell_8/mul:z:0+backward_lstm_2/while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_2/while/lstm_cell_8/add_1Щ
+backward_lstm_2/while/lstm_cell_8/Sigmoid_2Sigmoid0backward_lstm_2/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_2/while/lstm_cell_8/Sigmoid_2Л
(backward_lstm_2/while/lstm_cell_8/Relu_1Relu+backward_lstm_2/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_2/while/lstm_cell_8/Relu_1є
'backward_lstm_2/while/lstm_cell_8/mul_2Mul/backward_lstm_2/while/lstm_cell_8/Sigmoid_2:y:06backward_lstm_2/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_2/while/lstm_cell_8/mul_2Џ
:backward_lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#backward_lstm_2_while_placeholder_1!backward_lstm_2_while_placeholder+backward_lstm_2/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:backward_lstm_2/while/TensorArrayV2Write/TensorListSetItem|
backward_lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_2/while/add/yЉ
backward_lstm_2/while/addAddV2!backward_lstm_2_while_placeholder$backward_lstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/while/add
backward_lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_2/while/add_1/yЦ
backward_lstm_2/while/add_1AddV28backward_lstm_2_while_backward_lstm_2_while_loop_counter&backward_lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/while/add_1Ћ
backward_lstm_2/while/IdentityIdentitybackward_lstm_2/while/add_1:z:0^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2 
backward_lstm_2/while/IdentityЮ
 backward_lstm_2/while/Identity_1Identity>backward_lstm_2_while_backward_lstm_2_while_maximum_iterations^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_2/while/Identity_1­
 backward_lstm_2/while/Identity_2Identitybackward_lstm_2/while/add:z:0^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_2/while/Identity_2к
 backward_lstm_2/while/Identity_3IdentityJbackward_lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_2/while/Identity_3Ь
 backward_lstm_2/while/Identity_4Identity+backward_lstm_2/while/lstm_cell_8/mul_2:z:0^backward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/while/Identity_4Ь
 backward_lstm_2/while/Identity_5Identity+backward_lstm_2/while/lstm_cell_8/add_1:z:0^backward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/while/Identity_5Ћ
backward_lstm_2/while/NoOpNoOp9^backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp8^backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp:^backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_2/while/NoOp"p
5backward_lstm_2_while_backward_lstm_2_strided_slice_17backward_lstm_2_while_backward_lstm_2_strided_slice_1_0"I
backward_lstm_2_while_identity'backward_lstm_2/while/Identity:output:0"M
 backward_lstm_2_while_identity_1)backward_lstm_2/while/Identity_1:output:0"M
 backward_lstm_2_while_identity_2)backward_lstm_2/while/Identity_2:output:0"M
 backward_lstm_2_while_identity_3)backward_lstm_2/while/Identity_3:output:0"M
 backward_lstm_2_while_identity_4)backward_lstm_2/while/Identity_4:output:0"M
 backward_lstm_2_while_identity_5)backward_lstm_2/while/Identity_5:output:0"
Abackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resourceCbackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0"
Bbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resourceDbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0"
@backward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resourceBbackward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0"ш
qbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensorsbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2t
8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp2r
7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp2v
9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
^

K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_366746

inputs=
*lstm_cell_8_matmul_readvariableop_resource:	Ш?
,lstm_cell_8_matmul_1_readvariableop_resource:	2Ш:
+lstm_cell_8_biasadd_readvariableop_resource:	Ш
identityЂ"lstm_cell_8/BiasAdd/ReadVariableOpЂ!lstm_cell_8/MatMul/ReadVariableOpЂ#lstm_cell_8/MatMul_1/ReadVariableOpЂwhileD
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
strided_slice_2В
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOpЊ
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/MatMulИ
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOpІ
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/MatMul_1
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/addБ
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOpЉ
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_8/BiasAdd|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimя
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_8/split
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Sigmoid
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Sigmoid_1
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Relu
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/mul_1
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/add_1
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/Relu_1
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_8/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
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
bodyR
while_body_366662*
condR
while_cond_366661*K
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

IdentityХ
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
у	

0__inference_bidirectional_2_layer_call_fn_364049
inputs_0
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
	unknown_2:	Ш
	unknown_3:	2Ш
	unknown_4:	Ш
identityЂStatefulPartitionedCallБ
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
GPU 2J 8 *T
fORM
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_3625722
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
о[

J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_365788
inputs_0=
*lstm_cell_7_matmul_readvariableop_resource:	Ш?
,lstm_cell_7_matmul_1_readvariableop_resource:	2Ш:
+lstm_cell_7_biasadd_readvariableop_resource:	Ш
identityЂ"lstm_cell_7/BiasAdd/ReadVariableOpЂ!lstm_cell_7/MatMul/ReadVariableOpЂ#lstm_cell_7/MatMul_1/ReadVariableOpЂwhileF
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
strided_slice_2В
!lstm_cell_7/MatMul/ReadVariableOpReadVariableOp*lstm_cell_7_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02#
!lstm_cell_7/MatMul/ReadVariableOpЊ
lstm_cell_7/MatMulMatMulstrided_slice_2:output:0)lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/MatMulИ
#lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02%
#lstm_cell_7/MatMul_1/ReadVariableOpІ
lstm_cell_7/MatMul_1MatMulzeros:output:0+lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/MatMul_1
lstm_cell_7/addAddV2lstm_cell_7/MatMul:product:0lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/addБ
"lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02$
"lstm_cell_7/BiasAdd/ReadVariableOpЉ
lstm_cell_7/BiasAddBiasAddlstm_cell_7/add:z:0*lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_7/BiasAdd|
lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_7/split/split_dimя
lstm_cell_7/splitSplit$lstm_cell_7/split/split_dim:output:0lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_7/split
lstm_cell_7/SigmoidSigmoidlstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Sigmoid
lstm_cell_7/Sigmoid_1Sigmoidlstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Sigmoid_1
lstm_cell_7/mulMullstm_cell_7/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/mulz
lstm_cell_7/ReluRelulstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Relu
lstm_cell_7/mul_1Mullstm_cell_7/Sigmoid:y:0lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/mul_1
lstm_cell_7/add_1AddV2lstm_cell_7/mul:z:0lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/add_1
lstm_cell_7/Sigmoid_2Sigmoidlstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Sigmoid_2y
lstm_cell_7/Relu_1Relulstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/Relu_1
lstm_cell_7/mul_2Mullstm_cell_7/Sigmoid_2:y:0 lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_7/mul_2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_7_matmul_readvariableop_resource,lstm_cell_7_matmul_1_readvariableop_resource+lstm_cell_7_biasadd_readvariableop_resource*
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
bodyR
while_body_365704*
condR
while_cond_365703*K
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

IdentityХ
NoOpNoOp#^lstm_cell_7/BiasAdd/ReadVariableOp"^lstm_cell_7/MatMul/ReadVariableOp$^lstm_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2H
"lstm_cell_7/BiasAdd/ReadVariableOp"lstm_cell_7/BiasAdd/ReadVariableOp2F
!lstm_cell_7/MatMul/ReadVariableOp!lstm_cell_7/MatMul/ReadVariableOp2J
#lstm_cell_7/MatMul_1/ReadVariableOp#lstm_cell_7/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Ы>
Ч
while_body_362842
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_7_matmul_readvariableop_resource_0:	ШG
4while_lstm_cell_7_matmul_1_readvariableop_resource_0:	2ШB
3while_lstm_cell_7_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_7_matmul_readvariableop_resource:	ШE
2while_lstm_cell_7_matmul_1_readvariableop_resource:	2Ш@
1while_lstm_cell_7_biasadd_readvariableop_resource:	ШЂ(while/lstm_cell_7/BiasAdd/ReadVariableOpЂ'while/lstm_cell_7/MatMul/ReadVariableOpЂ)while/lstm_cell_7/MatMul_1/ReadVariableOpУ
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
)while/TensorArrayV2Read/TensorListGetItemЦ
'while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02)
'while/lstm_cell_7/MatMul/ReadVariableOpд
while/lstm_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/MatMulЬ
)while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02+
)while/lstm_cell_7/MatMul_1/ReadVariableOpН
while/lstm_cell_7/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/MatMul_1Д
while/lstm_cell_7/addAddV2"while/lstm_cell_7/MatMul:product:0$while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/addХ
(while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02*
(while/lstm_cell_7/BiasAdd/ReadVariableOpС
while/lstm_cell_7/BiasAddBiasAddwhile/lstm_cell_7/add:z:00while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/BiasAdd
!while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_7/split/split_dim
while/lstm_cell_7/splitSplit*while/lstm_cell_7/split/split_dim:output:0"while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_7/split
while/lstm_cell_7/SigmoidSigmoid while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Sigmoid
while/lstm_cell_7/Sigmoid_1Sigmoid while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Sigmoid_1
while/lstm_cell_7/mulMulwhile/lstm_cell_7/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/mul
while/lstm_cell_7/ReluRelu while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/ReluА
while/lstm_cell_7/mul_1Mulwhile/lstm_cell_7/Sigmoid:y:0$while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/mul_1Ѕ
while/lstm_cell_7/add_1AddV2while/lstm_cell_7/mul:z:0while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/add_1
while/lstm_cell_7/Sigmoid_2Sigmoid while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Sigmoid_2
while/lstm_cell_7/Relu_1Reluwhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Relu_1Д
while/lstm_cell_7/mul_2Mulwhile/lstm_cell_7/Sigmoid_2:y:0&while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/mul_2п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_7/mul_2:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_7/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_7/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5л

while/NoOpNoOp)^while/lstm_cell_7/BiasAdd/ReadVariableOp(^while/lstm_cell_7/MatMul/ReadVariableOp*^while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_7_biasadd_readvariableop_resource3while_lstm_cell_7_biasadd_readvariableop_resource_0"j
2while_lstm_cell_7_matmul_1_readvariableop_resource4while_lstm_cell_7_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_7_matmul_readvariableop_resource2while_lstm_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2T
(while/lstm_cell_7/BiasAdd/ReadVariableOp(while/lstm_cell_7/BiasAdd/ReadVariableOp2R
'while/lstm_cell_7/MatMul/ReadVariableOp'while/lstm_cell_7/MatMul/ReadVariableOp2V
)while/lstm_cell_7/MatMul_1/ReadVariableOp)while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
и
М
/__inference_forward_lstm_2_layer_call_fn_365486

inputs
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
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
GPU 2J 8 *S
fNRL
J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_3629262
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
е
У
while_cond_366005
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_366005___redundant_placeholder04
0while_while_cond_366005___redundant_placeholder14
0while_while_cond_366005___redundant_placeholder24
0while_while_cond_366005___redundant_placeholder3
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
а
ћ
H__inference_sequential_2_layer_call_and_return_conditional_losses_363444

inputs
inputs_1	)
bidirectional_2_363413:	Ш)
bidirectional_2_363415:	2Ш%
bidirectional_2_363417:	Ш)
bidirectional_2_363419:	Ш)
bidirectional_2_363421:	2Ш%
bidirectional_2_363423:	Ш 
dense_2_363438:d
dense_2_363440:
identityЂ'bidirectional_2/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЊ
'bidirectional_2/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_2_363413bidirectional_2_363415bidirectional_2_363417bidirectional_2_363419bidirectional_2_363421bidirectional_2_363423*
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
GPU 2J 8 *T
fORM
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_3634122)
'bidirectional_2/StatefulPartitionedCallЙ
dense_2/StatefulPartitionedCallStatefulPartitionedCall0bidirectional_2/StatefulPartitionedCall:output:0dense_2_363438dense_2_363440*
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
GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3634372!
dense_2/StatefulPartitionedCall
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp(^bidirectional_2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2R
'bidirectional_2/StatefulPartitionedCall'bidirectional_2/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
е
У
while_cond_366355
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_366355___redundant_placeholder04
0while_while_cond_366355___redundant_placeholder14
0while_while_cond_366355___redundant_placeholder24
0while_while_cond_366355___redundant_placeholder3
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
е
У
while_cond_362841
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_362841___redundant_placeholder04
0while_while_cond_362841___redundant_placeholder14
0while_while_cond_362841___redundant_placeholder24
0while_while_cond_362841___redundant_placeholder3
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
Ь
О
/__inference_forward_lstm_2_layer_call_fn_365453
inputs_0
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identityЂStatefulPartitionedCall
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
GPU 2J 8 *S
fNRL
J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_3611342
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
П%
м
while_body_361909
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_8_361933_0:	Ш-
while_lstm_cell_8_361935_0:	2Ш)
while_lstm_cell_8_361937_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_8_361933:	Ш+
while_lstm_cell_8_361935:	2Ш'
while_lstm_cell_8_361937:	ШЂ)while/lstm_cell_8/StatefulPartitionedCallУ
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
)while/TensorArrayV2Read/TensorListGetItemл
)while/lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_8_361933_0while_lstm_cell_8_361935_0while_lstm_cell_8_361937_0*
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
GPU 2J 8 *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_3618292+
)while/lstm_cell_8/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_8/StatefulPartitionedCall:output:0*
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
while/Identity_3Ѓ
while/Identity_4Identity2while/lstm_cell_8/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ѓ
while/Identity_5Identity2while/lstm_cell_8/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5

while/NoOpNoOp*^while/lstm_cell_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_8_361933while_lstm_cell_8_361933_0"6
while_lstm_cell_8_361935while_lstm_cell_8_361935_0"6
while_lstm_cell_8_361937while_lstm_cell_8_361937_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2V
)while/lstm_cell_8/StatefulPartitionedCall)while/lstm_cell_8/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
е
У
while_cond_362668
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_362668___redundant_placeholder04
0while_while_cond_362668___redundant_placeholder14
0while_while_cond_362668___redundant_placeholder24
0while_while_cond_362668___redundant_placeholder3
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
эa

!backward_lstm_2_while_body_363315<
8backward_lstm_2_while_backward_lstm_2_while_loop_counterB
>backward_lstm_2_while_backward_lstm_2_while_maximum_iterations%
!backward_lstm_2_while_placeholder'
#backward_lstm_2_while_placeholder_1'
#backward_lstm_2_while_placeholder_2'
#backward_lstm_2_while_placeholder_3'
#backward_lstm_2_while_placeholder_4;
7backward_lstm_2_while_backward_lstm_2_strided_slice_1_0w
sbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_06
2backward_lstm_2_while_less_backward_lstm_2_sub_1_0U
Bbackward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0:	ШW
Dbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0:	2ШR
Cbackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ш"
backward_lstm_2_while_identity$
 backward_lstm_2_while_identity_1$
 backward_lstm_2_while_identity_2$
 backward_lstm_2_while_identity_3$
 backward_lstm_2_while_identity_4$
 backward_lstm_2_while_identity_5$
 backward_lstm_2_while_identity_69
5backward_lstm_2_while_backward_lstm_2_strided_slice_1u
qbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor4
0backward_lstm_2_while_less_backward_lstm_2_sub_1S
@backward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource:	ШU
Bbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource:	2ШP
Abackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource:	ШЂ8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOpЂ7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOpЂ9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOpу
Gbackward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gbackward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
9backward_lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0!backward_lstm_2_while_placeholderPbackward_lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02;
9backward_lstm_2/while/TensorArrayV2Read/TensorListGetItemХ
backward_lstm_2/while/LessLess2backward_lstm_2_while_less_backward_lstm_2_sub_1_0!backward_lstm_2_while_placeholder*
T0*#
_output_shapes
:џџџџџџџџџ2
backward_lstm_2/while/Lessі
7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOpBbackward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype029
7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp
(backward_lstm_2/while/lstm_cell_8/MatMulMatMul@backward_lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0?backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2*
(backward_lstm_2/while/lstm_cell_8/MatMulќ
9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOpDbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02;
9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp§
*backward_lstm_2/while/lstm_cell_8/MatMul_1MatMul#backward_lstm_2_while_placeholder_3Abackward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2,
*backward_lstm_2/while/lstm_cell_8/MatMul_1є
%backward_lstm_2/while/lstm_cell_8/addAddV22backward_lstm_2/while/lstm_cell_8/MatMul:product:04backward_lstm_2/while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%backward_lstm_2/while/lstm_cell_8/addѕ
8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOpCbackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02:
8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp
)backward_lstm_2/while/lstm_cell_8/BiasAddBiasAdd)backward_lstm_2/while/lstm_cell_8/add:z:0@backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)backward_lstm_2/while/lstm_cell_8/BiasAddЈ
1backward_lstm_2/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1backward_lstm_2/while/lstm_cell_8/split/split_dimЧ
'backward_lstm_2/while/lstm_cell_8/splitSplit:backward_lstm_2/while/lstm_cell_8/split/split_dim:output:02backward_lstm_2/while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2)
'backward_lstm_2/while/lstm_cell_8/splitХ
)backward_lstm_2/while/lstm_cell_8/SigmoidSigmoid0backward_lstm_2/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)backward_lstm_2/while/lstm_cell_8/SigmoidЩ
+backward_lstm_2/while/lstm_cell_8/Sigmoid_1Sigmoid0backward_lstm_2/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_2/while/lstm_cell_8/Sigmoid_1н
%backward_lstm_2/while/lstm_cell_8/mulMul/backward_lstm_2/while/lstm_cell_8/Sigmoid_1:y:0#backward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22'
%backward_lstm_2/while/lstm_cell_8/mulМ
&backward_lstm_2/while/lstm_cell_8/ReluRelu0backward_lstm_2/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22(
&backward_lstm_2/while/lstm_cell_8/Relu№
'backward_lstm_2/while/lstm_cell_8/mul_1Mul-backward_lstm_2/while/lstm_cell_8/Sigmoid:y:04backward_lstm_2/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_2/while/lstm_cell_8/mul_1х
'backward_lstm_2/while/lstm_cell_8/add_1AddV2)backward_lstm_2/while/lstm_cell_8/mul:z:0+backward_lstm_2/while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_2/while/lstm_cell_8/add_1Щ
+backward_lstm_2/while/lstm_cell_8/Sigmoid_2Sigmoid0backward_lstm_2/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22-
+backward_lstm_2/while/lstm_cell_8/Sigmoid_2Л
(backward_lstm_2/while/lstm_cell_8/Relu_1Relu+backward_lstm_2/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(backward_lstm_2/while/lstm_cell_8/Relu_1є
'backward_lstm_2/while/lstm_cell_8/mul_2Mul/backward_lstm_2/while/lstm_cell_8/Sigmoid_2:y:06backward_lstm_2/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22)
'backward_lstm_2/while/lstm_cell_8/mul_2ъ
backward_lstm_2/while/SelectSelectbackward_lstm_2/while/Less:z:0+backward_lstm_2/while/lstm_cell_8/mul_2:z:0#backward_lstm_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
backward_lstm_2/while/Selectю
backward_lstm_2/while/Select_1Selectbackward_lstm_2/while/Less:z:0+backward_lstm_2/while/lstm_cell_8/mul_2:z:0#backward_lstm_2_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_2/while/Select_1ю
backward_lstm_2/while/Select_2Selectbackward_lstm_2/while/Less:z:0+backward_lstm_2/while/lstm_cell_8/add_1:z:0#backward_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџ22 
backward_lstm_2/while/Select_2Љ
:backward_lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#backward_lstm_2_while_placeholder_1!backward_lstm_2_while_placeholder%backward_lstm_2/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:backward_lstm_2/while/TensorArrayV2Write/TensorListSetItem|
backward_lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_2/while/add/yЉ
backward_lstm_2/while/addAddV2!backward_lstm_2_while_placeholder$backward_lstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/while/add
backward_lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_2/while/add_1/yЦ
backward_lstm_2/while/add_1AddV28backward_lstm_2_while_backward_lstm_2_while_loop_counter&backward_lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_2/while/add_1Ћ
backward_lstm_2/while/IdentityIdentitybackward_lstm_2/while/add_1:z:0^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2 
backward_lstm_2/while/IdentityЮ
 backward_lstm_2/while/Identity_1Identity>backward_lstm_2_while_backward_lstm_2_while_maximum_iterations^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_2/while/Identity_1­
 backward_lstm_2/while/Identity_2Identitybackward_lstm_2/while/add:z:0^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_2/while/Identity_2к
 backward_lstm_2/while/Identity_3IdentityJbackward_lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_2/while/NoOp*
T0*
_output_shapes
: 2"
 backward_lstm_2/while/Identity_3Ц
 backward_lstm_2/while/Identity_4Identity%backward_lstm_2/while/Select:output:0^backward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/while/Identity_4Ш
 backward_lstm_2/while/Identity_5Identity'backward_lstm_2/while/Select_1:output:0^backward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/while/Identity_5Ш
 backward_lstm_2/while/Identity_6Identity'backward_lstm_2/while/Select_2:output:0^backward_lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 backward_lstm_2/while/Identity_6Ћ
backward_lstm_2/while/NoOpNoOp9^backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp8^backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp:^backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_2/while/NoOp"p
5backward_lstm_2_while_backward_lstm_2_strided_slice_17backward_lstm_2_while_backward_lstm_2_strided_slice_1_0"I
backward_lstm_2_while_identity'backward_lstm_2/while/Identity:output:0"M
 backward_lstm_2_while_identity_1)backward_lstm_2/while/Identity_1:output:0"M
 backward_lstm_2_while_identity_2)backward_lstm_2/while/Identity_2:output:0"M
 backward_lstm_2_while_identity_3)backward_lstm_2/while/Identity_3:output:0"M
 backward_lstm_2_while_identity_4)backward_lstm_2/while/Identity_4:output:0"M
 backward_lstm_2_while_identity_5)backward_lstm_2/while/Identity_5:output:0"M
 backward_lstm_2_while_identity_6)backward_lstm_2/while/Identity_6:output:0"f
0backward_lstm_2_while_less_backward_lstm_2_sub_12backward_lstm_2_while_less_backward_lstm_2_sub_1_0"
Abackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resourceCbackward_lstm_2_while_lstm_cell_8_biasadd_readvariableop_resource_0"
Bbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resourceDbackward_lstm_2_while_lstm_cell_8_matmul_1_readvariableop_resource_0"
@backward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resourceBbackward_lstm_2_while_lstm_cell_8_matmul_readvariableop_resource_0"ш
qbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensorsbackward_lstm_2_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2: : :џџџџџџџџџ: : : 2t
8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp8backward_lstm_2/while/lstm_cell_8/BiasAdd/ReadVariableOp2r
7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp7backward_lstm_2/while/lstm_cell_8/MatMul/ReadVariableOp2v
9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp9backward_lstm_2/while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Ы>
Ч
while_body_362317
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_7_matmul_readvariableop_resource_0:	ШG
4while_lstm_cell_7_matmul_1_readvariableop_resource_0:	2ШB
3while_lstm_cell_7_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_7_matmul_readvariableop_resource:	ШE
2while_lstm_cell_7_matmul_1_readvariableop_resource:	2Ш@
1while_lstm_cell_7_biasadd_readvariableop_resource:	ШЂ(while/lstm_cell_7/BiasAdd/ReadVariableOpЂ'while/lstm_cell_7/MatMul/ReadVariableOpЂ)while/lstm_cell_7/MatMul_1/ReadVariableOpУ
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
)while/TensorArrayV2Read/TensorListGetItemЦ
'while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	Ш*
dtype02)
'while/lstm_cell_7/MatMul/ReadVariableOpд
while/lstm_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/MatMulЬ
)while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02+
)while/lstm_cell_7/MatMul_1/ReadVariableOpН
while/lstm_cell_7/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/MatMul_1Д
while/lstm_cell_7/addAddV2"while/lstm_cell_7/MatMul:product:0$while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/addХ
(while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02*
(while/lstm_cell_7/BiasAdd/ReadVariableOpС
while/lstm_cell_7/BiasAddBiasAddwhile/lstm_cell_7/add:z:00while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_7/BiasAdd
!while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_7/split/split_dim
while/lstm_cell_7/splitSplit*while/lstm_cell_7/split/split_dim:output:0"while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_7/split
while/lstm_cell_7/SigmoidSigmoid while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Sigmoid
while/lstm_cell_7/Sigmoid_1Sigmoid while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Sigmoid_1
while/lstm_cell_7/mulMulwhile/lstm_cell_7/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/mul
while/lstm_cell_7/ReluRelu while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/ReluА
while/lstm_cell_7/mul_1Mulwhile/lstm_cell_7/Sigmoid:y:0$while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/mul_1Ѕ
while/lstm_cell_7/add_1AddV2while/lstm_cell_7/mul:z:0while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/add_1
while/lstm_cell_7/Sigmoid_2Sigmoid while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Sigmoid_2
while/lstm_cell_7/Relu_1Reluwhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/Relu_1Д
while/lstm_cell_7/mul_2Mulwhile/lstm_cell_7/Sigmoid_2:y:0&while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_7/mul_2п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_7/mul_2:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_7/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_7/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5л

while/NoOpNoOp)^while/lstm_cell_7/BiasAdd/ReadVariableOp(^while/lstm_cell_7/MatMul/ReadVariableOp*^while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_7_biasadd_readvariableop_resource3while_lstm_cell_7_biasadd_readvariableop_resource_0"j
2while_lstm_cell_7_matmul_1_readvariableop_resource4while_lstm_cell_7_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_7_matmul_readvariableop_resource2while_lstm_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : 2T
(while/lstm_cell_7/BiasAdd/ReadVariableOp(while/lstm_cell_7/BiasAdd/ReadVariableOp2R
'while/lstm_cell_7/MatMul/ReadVariableOp'while/lstm_cell_7/MatMul/ReadVariableOp2V
)while/lstm_cell_7/MatMul_1/ReadVariableOp)while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Д
ѕ
,__inference_lstm_cell_8_layer_call_fn_366861

inputs
states_0
states_1
unknown:	Ш
	unknown_0:	2Ш
	unknown_1:	Ш
identity

identity_1

identity_2ЂStatefulPartitionedCallТ
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
GPU 2J 8 *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_3616832
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
а
ћ
H__inference_sequential_2_layer_call_and_return_conditional_losses_363915

inputs
inputs_1	)
bidirectional_2_363896:	Ш)
bidirectional_2_363898:	2Ш%
bidirectional_2_363900:	Ш)
bidirectional_2_363902:	Ш)
bidirectional_2_363904:	2Ш%
bidirectional_2_363906:	Ш 
dense_2_363909:d
dense_2_363911:
identityЂ'bidirectional_2/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЊ
'bidirectional_2/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_2_363896bidirectional_2_363898bidirectional_2_363900bidirectional_2_363902bidirectional_2_363904bidirectional_2_363906*
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
GPU 2J 8 *T
fORM
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_3638522)
'bidirectional_2/StatefulPartitionedCallЙ
dense_2/StatefulPartitionedCallStatefulPartitionedCall0bidirectional_2/StatefulPartitionedCall:output:0dense_2_363909dense_2_363911*
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
GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3634372!
dense_2/StatefulPartitionedCall
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp(^bidirectional_2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2R
'bidirectional_2/StatefulPartitionedCall'bidirectional_2/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs"ЈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*у
serving_defaultЯ
9
args_0/
serving_default_args_0:0џџџџџџџџџ
9
args_0_1-
serving_default_args_0_1:0	џџџџџџџџџ;
dense_20
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:щЗ
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
 :d2dense_2/kernel
:2dense_2/bias
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
D:B	Ш21bidirectional_2/forward_lstm_2/lstm_cell_7/kernel
N:L	2Ш2;bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel
>:<Ш2/bidirectional_2/forward_lstm_2/lstm_cell_7/bias
E:C	Ш22bidirectional_2/backward_lstm_2/lstm_cell_8/kernel
O:M	2Ш2<bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel
?:=Ш20bidirectional_2/backward_lstm_2/lstm_cell_8/bias
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
%:#d2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
I:G	Ш28Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/m
S:Q	2Ш2BAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/m
C:AШ26Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/m
J:H	Ш29Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/m
T:R	2Ш2CAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/m
D:BШ27Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/m
%:#d2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
I:G	Ш28Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/v
S:Q	2Ш2BAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/v
C:AШ26Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/v
J:H	Ш29Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/v
T:R	2Ш2CAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/v
D:BШ27Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/v
(:&d2Adam/dense_2/kernel/vhat
": 2Adam/dense_2/bias/vhat
L:J	Ш2;Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/kernel/vhat
V:T	2Ш2EAdam/bidirectional_2/forward_lstm_2/lstm_cell_7/recurrent_kernel/vhat
F:DШ29Adam/bidirectional_2/forward_lstm_2/lstm_cell_7/bias/vhat
M:K	Ш2<Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/kernel/vhat
W:U	2Ш2FAdam/bidirectional_2/backward_lstm_2/lstm_cell_8/recurrent_kernel/vhat
G:EШ2:Adam/bidirectional_2/backward_lstm_2/lstm_cell_8/bias/vhat
Є2Ё
-__inference_sequential_2_layer_call_fn_363463
-__inference_sequential_2_layer_call_fn_363956Р
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
еBв
!__inference__wrapped_model_360976args_0args_0_1"
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
к2з
H__inference_sequential_2_layer_call_and_return_conditional_losses_363979
H__inference_sequential_2_layer_call_and_return_conditional_losses_364002Р
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
Д2Б
0__inference_bidirectional_2_layer_call_fn_364049
0__inference_bidirectional_2_layer_call_fn_364066
0__inference_bidirectional_2_layer_call_fn_364084
0__inference_bidirectional_2_layer_call_fn_364102ц
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
 2
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_364404
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_364706
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_365064
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_365422ц
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
в2Я
(__inference_dense_2_layer_call_fn_365431Ђ
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
э2ъ
C__inference_dense_2_layer_call_and_return_conditional_losses_365442Ђ
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
вBЯ
$__inference_signature_wrapper_364032args_0args_0_1"
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
2
/__inference_forward_lstm_2_layer_call_fn_365453
/__inference_forward_lstm_2_layer_call_fn_365464
/__inference_forward_lstm_2_layer_call_fn_365475
/__inference_forward_lstm_2_layer_call_fn_365486е
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
2
J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_365637
J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_365788
J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_365939
J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_366090е
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
Ѓ2 
0__inference_backward_lstm_2_layer_call_fn_366101
0__inference_backward_lstm_2_layer_call_fn_366112
0__inference_backward_lstm_2_layer_call_fn_366123
0__inference_backward_lstm_2_layer_call_fn_366134е
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
2
K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_366287
K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_366440
K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_366593
K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_366746е
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
 2
,__inference_lstm_cell_7_layer_call_fn_366763
,__inference_lstm_cell_7_layer_call_fn_366780О
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
ж2г
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_366812
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_366844О
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
 2
,__inference_lstm_cell_8_layer_call_fn_366861
,__inference_lstm_cell_8_layer_call_fn_366878О
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
ж2г
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_366910
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_366942О
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
 С
!__inference__wrapped_model_360976\ЂY
RЂO
MJ4Ђ1
!њџџџџџџџџџџџџџџџџџџ

`
	RaggedTensorSpec
Њ "1Њ.
,
dense_2!
dense_2џџџџџџџџџЬ
K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_366287}OЂL
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
 Ь
K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_366440}OЂL
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
 Ю
K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_366593QЂN
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
 Ю
K__inference_backward_lstm_2_layer_call_and_return_conditional_losses_366746QЂN
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
 Є
0__inference_backward_lstm_2_layer_call_fn_366101pOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2Є
0__inference_backward_lstm_2_layer_call_fn_366112pOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2І
0__inference_backward_lstm_2_layer_call_fn_366123rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2І
0__inference_backward_lstm_2_layer_call_fn_366134rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2н
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_364404\ЂY
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
 н
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_364706\ЂY
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
 э
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_365064lЂi
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
 э
K__inference_bidirectional_2_layer_call_and_return_conditional_losses_365422lЂi
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
 Е
0__inference_bidirectional_2_layer_call_fn_364049\ЂY
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
Њ "џџџџџџџџџdЕ
0__inference_bidirectional_2_layer_call_fn_364066\ЂY
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
Њ "џџџџџџџџџdХ
0__inference_bidirectional_2_layer_call_fn_364084lЂi
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
Њ "џџџџџџџџџdХ
0__inference_bidirectional_2_layer_call_fn_364102lЂi
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
Њ "џџџџџџџџџdЃ
C__inference_dense_2_layer_call_and_return_conditional_losses_365442\/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџ
 {
(__inference_dense_2_layer_call_fn_365431O/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџЫ
J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_365637}OЂL
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
 Ы
J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_365788}OЂL
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
 Э
J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_365939QЂN
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
 Э
J__inference_forward_lstm_2_layer_call_and_return_conditional_losses_366090QЂN
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
 Ѓ
/__inference_forward_lstm_2_layer_call_fn_365453pOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2Ѓ
/__inference_forward_lstm_2_layer_call_fn_365464pOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2Ѕ
/__inference_forward_lstm_2_layer_call_fn_365475rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2Ѕ
/__inference_forward_lstm_2_layer_call_fn_365486rQЂN
GЂD
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2Щ
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_366812§Ђ}
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
 Щ
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_366844§Ђ}
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
 
,__inference_lstm_cell_7_layer_call_fn_366763эЂ}
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
1/1џџџџџџџџџ2
,__inference_lstm_cell_7_layer_call_fn_366780эЂ}
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
1/1џџџџџџџџџ2Щ
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_366910§Ђ}
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
 Щ
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_366942§Ђ}
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
 
,__inference_lstm_cell_8_layer_call_fn_366861эЂ}
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
1/1џџџџџџџџџ2
,__inference_lstm_cell_8_layer_call_fn_366878эЂ}
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
1/1џџџџџџџџџ2ф
H__inference_sequential_2_layer_call_and_return_conditional_losses_363979dЂa
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
 ф
H__inference_sequential_2_layer_call_and_return_conditional_losses_364002dЂa
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
 М
-__inference_sequential_2_layer_call_fn_363463dЂa
ZЂW
MJ4Ђ1
!њџџџџџџџџџџџџџџџџџџ

`
	RaggedTensorSpec
p 

 
Њ "џџџџџџџџџМ
-__inference_sequential_2_layer_call_fn_363956dЂa
ZЂW
MJ4Ђ1
!њџџџџџџџџџџџџџџџџџџ

`
	RaggedTensorSpec
p

 
Њ "џџџџџџџџџЭ
$__inference_signature_wrapper_364032ЄeЂb
Ђ 
[ЊX
*
args_0 
args_0џџџџџџџџџ
*
args_0_1
args_0_1џџџџџџџџџ	"1Њ.
,
dense_2!
dense_2џџџџџџџџџ